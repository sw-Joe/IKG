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
from ai_core.core.batch_optimizer import calculate_optimal_batch_sizes



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
    if not os.path.exists(json_file_path):
        logger.error(f"백업 파싱 파일 지정 경로 부재: {json_file_path}")
        return

    initialize_database_schema(IKG_DB_PATH)
    
    with open(json_file_path, encoding="utf-8") as f:
        bookmark_data = json.load(f)
    all_bookmarks = extract_bookmarks(bookmark_data)
    total_bookmarks_count = len(all_bookmarks)

    # -------------------------------------------------------------------------
    # [OPTIMIZED MATRIX]: 동적 엔진 모듈을 호출하여 양대 스테이지 배치 크기 완전 수령
    # -------------------------------------------------------------------------
    STAGE1_BATCH_SIZE, STAGE2_BATCH_SIZE = calculate_optimal_batch_sizes(total_bookmarks_count)
    # -------------------------------------------------------------------------

    # =========================================================================
    # [STAGE 1]: 수동 산정된 고속 플러싱 청크 기반 크롤링 & DB 적재
    # =========================================================================
    logger.info("==========================================================================")
    logger.info(f"[STAGE 1 START] 동적 계산된 {STAGE1_BATCH_SIZE}단위 청크 크롤링 및 SQLite 영속 적재")
    logger.info("==========================================================================")
    
    semaphore = asyncio.Semaphore(3)
    async with async_playwright() as p:
        main_browser = await p.chromium.launch(headless=True)
        
        # 최적화된 STAGE1_BATCH_SIZE 규격을 적용하여 루프 분할
        for chunk_offset, chunk in enumerate(chunk_list(all_bookmarks, STAGE1_BATCH_SIZE)):
            scrape_batch_num = chunk_offset + 1
            start_global_idx = (chunk_offset * STAGE1_BATCH_SIZE) + 1
            
            logger.info(f" -> [SCRAPE BATCH #{scrape_batch_num}] {start_global_idx}번 대역 자원 격리 웜업 개시")
            
            batch_context = await main_browser.new_context(user_agent="Mozilla/5.0...")
            tasks = [
                scrape_and_save_metadata(batch_context, bm, semaphore, start_global_idx + local_idx, total_bookmarks_count) 
                for local_idx, bm in enumerate(chunk)
            ]
            await asyncio.gather(*tasks)
            
            # 자원 청소 및 메모리 플러싱
            await batch_context.close()
            del tasks
            gc.collect()
            await asyncio.sleep(0.5)
            
        await main_browser.close()

    # =========================================================================
    # [STAGE 2]: 고속화 매트릭스 기반 ONNX True Batch 추론 및 FAISS 색인
    # =========================================================================
    logger.info("==========================================================================")
    logger.info(f"[STAGE 2 START] 동적 계산된 {STAGE2_BATCH_SIZE}단위 행렬 기반 AI 배치 임베딩 집행")
    logger.info("==========================================================================")
    
    embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
    indexer = VectorIndexer(dimension=1024)
    faiss_index = faiss.read_index(indexer.index_path)
    
    conn = sqlite3.connect(IKG_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content FROM bookmarks WHERE is_deleted = 0")
    db_active_rows = cursor.fetchall()
    conn.close()
    
    total_active_rows = len(db_active_rows)
    if total_active_rows == 0:
        logger.info("[BULK IMPORTER END] 색인 대상 활성 자산이 없어 즉시 종료합니다.")
        return

    total_successfully_indexed = 0
    
    # 최적화된 STAGE2_BATCH_SIZE 규격을 적용하여 단건 오버헤드 소거
    for i in range(0, total_active_rows, STAGE2_BATCH_SIZE):
        chunk_rows = db_active_rows[i:i + STAGE2_BATCH_SIZE]
        current_chunk_count = len(chunk_rows)
        inference_batch_num = (i // STAGE2_BATCH_SIZE) + 1
        
        logger.info(f" -> [INDEX BATCH #{inference_batch_num}] {i + 1}~{i + current_chunk_count}번 텐서 병렬 인퍼런스 가동")
        
        batch_texts = [f"{row['title']} {row['content']}" for row in chunk_rows]
        batch_ids = [row['id'] for row in chunk_rows]
        
        chunk_vectors_np = embedder.encode_batch(batch_texts)
        chunk_ids_np = np.array(batch_ids, dtype=np.int64)
        
        faiss_index.add_with_ids(chunk_vectors_np, chunk_ids_np)
        total_successfully_indexed += current_chunk_count
        
        del chunk_vectors_np, chunk_ids_np, batch_texts, batch_ids
        gc.collect()

    if total_successfully_indexed > 0:
        faiss.write_index(faiss_index, indexer.index_path)
        logger.info(f"[BULK IMPORTER SUCCESS] 파이프라인 무결 완수 (벡터 공간 자산 수: {faiss_index.ntotal}개)")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s")
    asyncio.run(run_bulk_importer())