import asyncio
import json
import logging
import os
import sqlite3
import gc

import faiss
import numpy as np
import trafilatura
from playwright.async_api import async_playwright

from ai_core.config import IKG_DB_PATH, IKG_MODEL_FILE, IKG_MODEL_PATH
from ai_core.core.embedder import BGEEmbedder
from ai_core.core.indexer import VectorIndexer
from ai_core.core.db_initializer import initialize_database_schema
from ai_core.core.content_validator import validate_content_integrity



logger = logging.getLogger("ai_core.bulk_importer")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def extract_bookmarks(node):
    """브라우저 백업 JSON 트리를 재귀 순회하여 단순 URL 리스트로 파싱"""
    bookmarks = []
    if "typeCode" in node:
        if node["typeCode"] == 1 and "uri" in node:
            bookmarks.append({
                "title": node.get("title", "Untitled"),
                "uri": node["uri"]
            })
        elif node["typeCode"] == 2 and "children" in node:
            for child in node["children"]:
                bookmarks.extend(extract_bookmarks(child))
    return bookmarks


async def fetch_dynamic_content_with_context(context, url: str) -> tuple[str | None, str | None]:
    """주입받은 현재 배치 전용 격리 컨텍스트 탭에서 동적 스크래핑 고속 집행"""
    page = None
    try:
        page = await context.new_page()
        await page.goto(url, timeout=12000, wait_until="networkidle")
        content = await page.content()
        title = await page.title()
        return content, title
    except Exception:
        return None, None
    finally:
        if page and not page.is_closed():
            await page.close()


async def scrape_and_save_metadata(
    context, bm: dict, semaphore: asyncio.Semaphore, current_idx: int, total_idx: int
) -> None:
    """I/O BOUND: 100건 단위 청크 내에서 병렬 구동되며 검증 후 SQLite 영속화 집행 (반환값 없음)"""
    async with semaphore:
        url = bm["uri"]
        title = bm["title"]
        
        # [Idempotency Guard]: 비상 정지 후 재가동 시 중복 수집을 원천 차단하는 체크포인트 복구력
        conn_check = sqlite3.connect(IKG_DB_PATH, timeout=10.0)
        cursor_check = conn_check.cursor()
        cursor_check.execute("SELECT id FROM bookmarks WHERE url = ?", (url,))
        exists = cursor_check.fetchone()
        conn_check.close()
        
        if exists:
            logger.info(f"  ⏩ [CHECKPOINT SKIPPED] [{current_idx}/{total_idx}] 이미 DB에 자산 상주 중 -> 스킵")
            return

        try:
            method_used = "Trafilatura(정적)"
            downloaded = trafilatura.fetch_url(url)
            content = trafilatura.extract(downloaded) if downloaded else None
            
            if not content or len(content) < 300 or "JavaScript is disabled" in content:
                method_used = "Playwright(배치 컨텍스트)"
                dynamic_content, dynamic_title = await fetch_dynamic_content_with_context(context, url)
                if dynamic_content:
                    content = trafilatura.extract(dynamic_content)
                    title = dynamic_title or title

            is_valid, result_payload = validate_content_integrity(title, content)

            conn = sqlite3.connect(IKG_DB_PATH, timeout=60.0)
            cursor = conn.cursor()
            try:
                if is_valid:
                    cursor.execute(
                        "INSERT INTO bookmarks (url, title, content, created_at, is_deleted) VALUES (?, ?, ?, datetime('now', 'localtime'), 0)",
                        (url, title, result_payload)
                    )
                    conn.commit()
                    logger.info(f"  ✅ [{current_idx}/{total_idx}] 무결성 통과 ({method_used}) ──► SQLite 적재 완료")
                else:
                    cursor.execute(
                        "INSERT INTO bookmarks (url, title, content, created_at, is_deleted) VALUES (?, ?, ?, datetime('now', 'localtime'), 2)",
                        (url, title, f"[PENDING_VERIFY_REASON] {result_payload}")
                    )
                    conn.commit()
                    logger.warning(f"  ⏳ [{current_idx}/{total_idx}] 검증 보류 격리 판정 ──► SQLite 적재 완료 (사유: {result_payload})")
            except sqlite3.IntegrityError:
                pass
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"  💥 [{current_idx}/{total_idx}] 런타임 수집 장애 가드 스킵: {str(e)}")


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


async def run_bulk_importer():
    json_file_path = input("북마크 백업 파일명 입력>")
    BATCH_SIZE = 100  # 양대 스테이지가 공유 참조할 마스터 배치 크기 임계치
    
    if not os.path.exists(json_file_path):
        logger.error(f"백업 파싱 파일 지정 경로 부재: {json_file_path}")
        return

    # WAL 모드 및 최적화 인덱스가 장착된 SQLite 스키마 빌드 가동
    initialize_database_schema(IKG_DB_PATH)
    
    with open(json_file_path, encoding="utf-8") as f:
        bookmark_data = json.load(f)
    all_bookmarks = extract_bookmarks(bookmark_data)
    total_bookmarks_count = len(all_bookmarks)

    # =========================================================================
    # [STAGE 1]: 100건 단위 분할 배칭 크롤링 + SQLite 영속화 + 메모리 플러싱 (I/O BOUND)
    # =========================================================================
    logger.info("==========================================================================")
    logger.info(f"[STAGE 1 START] 총 {total_bookmarks_count}건 분할 배치 수집 및 SQLite 선적재 개시")
    logger.info("==========================================================================")
    
    semaphore = asyncio.Semaphore(3)
    
    async with async_playwright() as p:
        # 소켓 TIME_WAIT 고갈을 막기 위해 브라우저 데몬 프로세스는 최외곽에서 1회만 시동
        main_browser = await p.chromium.launch(headless=True)
        
        for chunk_offset, chunk in enumerate(chunk_list(all_bookmarks, BATCH_SIZE)):
            scrape_batch_num = chunk_offset + 1
            start_global_idx = (chunk_offset * BATCH_SIZE) + 1
            
            logger.info(f" -> [SCRAPE BATCH #{scrape_batch_num}] {start_global_idx}번 대역 인메모리 세션 웜업")
            
            # [메모리 가드 1]: 현재 100건 청크만을 격리 수용할 독립 브라우저 컨텍스트 생성
            batch_context = await main_browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            
            tasks = [
                scrape_and_save_metadata(batch_context, bm, semaphore, start_global_idx + local_idx, total_bookmarks_count) 
                for local_idx, bm in enumerate(chunk)
            ]
            # 100건에 대한 병렬 크롤링 및 SQLite 커밋 완전 수렴 대기
            await asyncio.gather(*tasks)
            
            # [메모리 가드 2]: 100건 루프 종료 즉시 해당 청크 크로미움 가상 메모리 및 파이썬 힙 파괴 (플러싱)
            await batch_context.close()
            del tasks
            gc.collect()
            await asyncio.sleep(0.5) # 커널 소켓 정돈을 위한 미세 완충 지대
            
            logger.info(f" -> [SCRAPE BATCH #{scrape_batch_num} FLUSHED] 인메모리 힙 및 세션 가비지 완전 소거 완료.")
            
        await main_browser.close()

    logger.info("[STAGE 1 COMPLETE] 전량 안전 분할 수집 및 단일 진실 공급원(SQLite) 영속 구축 종결.")

    # =========================================================================
    # [STAGE 2]: DB 소스 원천 기점 ──► 100건 단위 ONNX 배치 추론 + FAISS 색인 + GC (CPU BOUND)
    # =========================================================================
    logger.info("==========================================================================")
    logger.info("[STAGE 2 START] SQLite 동기화 소스 기점 100단위 AI 배치 임베딩 및 FAISS 색인 착수")
    logger.info("==========================================================================")
    
    embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
    indexer = VectorIndexer(dimension=1024)
    faiss_index = faiss.read_index(indexer.index_path)
    
    # Stage 1을 거치며 완벽하게 수렴 정제된 '활성 자산(is_deleted=0)' 레코드만 쿼리하여 원천 소스로 삼음
    conn = sqlite3.connect(IKG_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content FROM bookmarks WHERE is_deleted = 0")
    db_active_rows = cursor.fetchall()
    conn.close()
    
    total_active_rows = len(db_active_rows)
    logger.info(f" -> 색인 분석 대상 데이터셋 확정: 총 {total_active_rows}건의 활성 문맥 확인.")
    
    if total_active_rows == 0:
        logger.info("[BULK IMPORTER END] 수렴해야 할 활성 레코드가 없어 작업을 마감합니다.")
        return

    total_successfully_indexed = 0
    
    # 가장 고부하인 AI 임베딩 단계 역시 사용자 사상에 따라 100건 단위로 쪼개어 GC를 집행
    for i in range(0, total_active_rows, BATCH_SIZE):
        chunk_rows = db_active_rows[i:i + BATCH_SIZE]
        current_chunk_count = len(chunk_rows)
        inference_batch_num = (i // BATCH_SIZE) + 1
        
        logger.info(f" -> [INDEX BATCH #{inference_batch_num}] {i + 1}~{i + current_chunk_count}번 자산 ONNX 배치 매트릭스 계산 개시")
        
        # 100건의 원문과 매핑 PK 식별자를 순수 배열 리스트로 바인딩
        batch_texts = [f"{row['title']} {row['content']}" for row in chunk_rows]
        batch_ids = [row['id'] for row in chunk_rows]
        
        # [True Batch Inference]: 단건 순회를 타파하고 100개 행렬을 ONNX C++ 커널에 단 1회 밀어 넣어 하드웨어 최대 가속
        chunk_vectors_np = embedder.encode_batch(batch_texts)
        chunk_ids_np = np.array(batch_ids, dtype=np.int64)
        
        # FAISS 인메모리 공간 적재 가동
        faiss_index.add_with_ids(chunk_vectors_np, chunk_ids_np)
        total_successfully_indexed += current_chunk_count
        
        # [메모리 가드 3]: 100단위 텐서 수학 연산 버퍼 찌꺼기 즉시 물리 소거 및 RAM 반환
        del chunk_vectors_np, chunk_ids_np, batch_texts, batch_ids
        gc.collect()
        
        logger.info(f" -> [INDEX BATCH #{inference_batch_num} FLUSHED] 다차원 벡터 인메모리 병합 및 고밀도 텐서 가비지 소거 완료.")

    # 양대 스테이지의 모든 분할 배치가 완벽하게 안착한 최종 시점에 단 1회의 원자적 파일 영구 커밋(write) 집행
    if total_successfully_indexed > 0:
        faiss.write_index(faiss_index, indexer.index_path)
        logger.info("==========================================================================")
        logger.info(f"[BULK IMPORTER SUCCESS] 2-Stage 하이브리드 미니배치 가드 파이프라인 완수 (최종 벡터 수: {faiss_index.ntotal}개)")
        logger.info("==========================================================================")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    asyncio.run(run_bulk_importer())