import asyncio
import json
import logging
import os
import sqlite3
import gc
import faiss
import numpy as np
import trafilatura
from datetime import datetime
from playwright.async_api import async_playwright, Browser

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


async def fetch_dynamic_content_with_pooling(browser: Browser, url: str) -> tuple[str | None, str | None]:
    """[OPTIMIZED] 기 생성된 Browser 상주 풀에서 신규 단일 페이지 탭만 열어 고속 동적 스크래핑 집행"""
    page = None
    try:
        # 새로운 브라우저 인스턴스를 매번 켜지 않고 상주 풀의 컨텍스트를 재사용
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        # 네트워크 침묵 시점 타임아웃을 12초로 최적화 다이어트 진행
        await page.goto(url, timeout=12000, wait_until="networkidle")
        content = await page.content()
        title = await page.title()
        await context.close()
        return content, title
    except Exception:
        return None, None
    finally:
        if page and not page.is_closed():
            await page.close()


async def scrape_and_save_metadata(
    browser: Browser, bm: dict, semaphore: asyncio.Semaphore, current_idx: int, total_idx: int
) -> dict | None:
    """I/O Bound: 네트워크 세마포어 통제 하에서 정적 수집 및 상주 풀 기반 동적 수집 교차 집행 후 DB 분기"""
    async with semaphore:
        url = bm["uri"]
        title = bm["title"]
        
        logger.info(f"[{current_idx}/{total_idx}] 스크래핑 파이프라인 가동 ──► URL: {url[:50]}")
        
        try:
            method_used = "Trafilatura(정적)"
            downloaded = trafilatura.fetch_url(url)
            content = trafilatura.extract(downloaded) if downloaded else None
            
            if not content or len(content) < 300 or "JavaScript is disabled" in content:
                method_used = "Playwright(재사용 풀)"
                # 최적화: 공통 상주 브라우저 인스턴스 핸들러 포인터를 전달하여 리소스 누수 해제
                dynamic_content, dynamic_title = await fetch_dynamic_content_with_pooling(browser, url)
                if dynamic_content:
                    content = trafilatura.extract(dynamic_content)
                    title = dynamic_title or title

            is_valid, result_payload = validate_content_integrity(title, content)

            conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            try:
                if is_valid:
                    cursor.execute(
                        """
                        INSERT INTO bookmarks (url, title, content, created_at, is_deleted) 
                        VALUES (?, ?, ?, datetime('now', 'localtime'), 0)
                        """,
                        (url, title, result_payload)
                    )
                    inserted_id = cursor.lastrowid
                    conn.commit()
                    logger.info(f"  ✅ [{current_idx}/{total_idx}] 무결성 검증 완벽 통과 ({method_used}) ──► 할당 ID: #{inserted_id}")
                    return {"id": inserted_id, "content": f"{title} {result_payload}"}
                else:
                    cursor.execute(
                        """
                        INSERT INTO bookmarks (url, title, content, created_at, is_deleted) 
                        VALUES (?, ?, ?, datetime('now', 'localtime'), 2)
                        """,
                        (url, title, f"[PENDING_VERIFY_REASON] {result_payload}")
                    )
                    conn.commit()
                    logger.warning(f"  ⏳ [{current_idx}/{total_idx}] 검증 보류 격리 판정 ──► 사유: {result_payload}")
                    return None
                    
            except sqlite3.IntegrityError:
                logger.info(f"  ⏩ [{current_idx}/{total_idx}] 스킵 (이미 영속 DB에 상주 중인 고유 URL)")
                return None
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"  💥 [{current_idx}/{total_idx}] 런타임 수집 인프라 예외 가드: {str(e)}")
            return None


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


async def run_bulk_importer():
    json_file_path = input("북마크 백업 파일명 입력>")
    BATCH_SIZE = 100
    
    if not os.path.exists(json_file_path):
        logger.error(f"백업 파싱 파일이 지정 경로에 존재하지 않습니다: {json_file_path}")
        return

    # 1. 데이터베이스 영속성 테이블 구조 자동 생성 검증 가드 작동
    initialize_database_schema(IKG_DB_PATH)

    logger.info("[BULK IMPORTER] 대용량 지식 자산 일괄 고속 마이그레이션 엔진을 가동합니다.")
    
    # 2. AI 인프라 코어 컴포넌트 로드
    embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
    indexer = VectorIndexer(dimension=1024)
    
    with open(json_file_path, encoding="utf-8") as f:
        bookmark_data = json.load(f)
    
    all_bookmarks = extract_bookmarks(bookmark_data)
    total_bookmarks_count = len(all_bookmarks)
    logger.info(f" -> 총 {total_bookmarks_count}개의 백업 메타데이터 스캔 완료.")

    semaphore = asyncio.Semaphore(3)
    
    # [OPTIMIZED I/O HOISTING]: 루프 내부에서 매번 읽던 작업을 루프 최외곽 상단으로 호이스팅 전격 탈출
    faiss_index = faiss.read_index(indexer.index_path)
    total_successfully_indexed = 0

    # 3. [OPTIMIZED] Playwright 전역 싱글톤 Browser 인스턴스 가동 (컨텍스트 상시 유지)
    async with async_playwright() as p:
        logger.info("[POOLING ACTOR] Playwright 크로미움 하드웨어 가속 풀링 코어 상주 인입 완결.")
        shared_browser = await p.chromium.launch(headless=True)
        
        # 4. 미니 배치 청크 단위 마이그레이션 파이프라인 전개
        for chunk_offset, chunk in enumerate(chunk_list(all_bookmarks, BATCH_SIZE)):
            processed_chunk_count = chunk_offset + 1
            start_global_idx = (chunk_offset * BATCH_SIZE) + 1
            
            logger.info(f"======================================================================================")
            logger.info(f"[CHUNK START] 미니 배치 파이프라인 가동 ({processed_chunk_count}회차 / 전체 대역: {start_global_idx}번 ~ {min(start_global_idx + BATCH_SIZE - 1, total_bookmarks_count)}번)")
            logger.info(f"======================================================================================")
            
            # 4-1. 현재 청크 병렬 크롤링 집행 (공유 브라우저 인스턴스 전격 주입)
            tasks = [
                scrape_and_save_metadata(shared_browser, bm, semaphore, start_global_idx + local_idx, total_bookmarks_count) 
                for local_idx, bm in enumerate(chunk)
            ]
            scraping_results = await asyncio.gather(*tasks)
            
            valid_documents = [res for res in scraping_results if res is not None]
            chunk_scraped_count = len(valid_documents)
            
            if chunk_scraped_count == 0:
                logger.info(" -> 현재 청크 범위 내에 무결성을 통과한 신규 자산이 없어 하부 연산을 스킵합니다.")
                continue

            # 4-2. [OPTIMIZED TRUE BATCH INFERENCE]: 단건 순회 추론을 제거하고 전체를 하나의 행렬로 통째로 인퍼런스
            logger.info(f" -> [{processed_chunk_count}회차 청크 수집완료] 살아남은 신규 자산 {chunk_scraped_count}건 전체 BGE-M3 ONNX 배치 행렬 일괄 추론 개시...")
            
            # 텍스트 원문 배열과 영속 PK 배열을 인덱스 동기화 매핑 분리 추출
            batch_texts = [doc["content"] for doc in valid_documents]
            batch_ids = [doc["id"] for doc in valid_documents]
            
            # [CRITICAL HIGHLIGHT]: 중량급 ONNX C++ 인퍼런스를 단 1회의 매트릭스 명령어로 통째로 연산 가속
            # 개별 오버헤드가 완전히 증발하며 하드웨어 가속이 폭발적으로 발휘되는 구간입니다.
            chunk_vectors_np = embedder.encode_batch(batch_texts)
            chunk_ids_np = np.array(batch_ids, dtype=np.int64)
            
            # 4-3. FAISS 인메모리 포인터 맵에 결합 적재
            faiss_index.add_with_ids(chunk_vectors_np, chunk_ids_np)
            
            total_successfully_indexed += chunk_scraped_count
            logger.info(f" -> [{processed_chunk_count}회차 청크 색인완료] 디스크 가상 메모리 적재 완결 (누적 수렴 자산: {total_successfully_indexed}건)")
            
            # 4-4. 가비지 컬렉션을 통한 미니 배치 메모리 엄격 회수
            import gc
            del tasks, scraping_results, valid_documents, batch_texts, batch_ids, chunk_vectors_np, chunk_ids_np
            gc.collect()

        # 브라우저 안전 인프라 폐쇄
        await shared_browser.close()

    # 5. [OPTIMIZED HOISTING]: 모든 청크 루프가 무결하게 완수된 직후 최하단에서 딱 1회의 원자적 디스크 쓰기 집행
    if total_successfully_indexed > 0:
        faiss.write_index(faiss_index, indexer.index_path)
        logger.info(f"[BULK IMPORTER SUCCESS] 전량 고속 마이그레이션이 무결하게 종료되었습니다. (FAISS 전역 활성 벡터 수: {faiss_index.ntotal})")
    else:
        logger.info("[BULK IMPORTER END] 수렴해야 할 신규 활성 자산이 없어 디스크 쓰기 생략 후 마감합니다.")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    asyncio.run(run_bulk_importer())