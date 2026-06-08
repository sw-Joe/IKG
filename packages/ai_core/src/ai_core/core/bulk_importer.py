import asyncio
import gc
import json
import logging
import os
import sqlite3

import faiss
import numpy as np
import trafilatura
from playwright.async_api import async_playwright

from ai_core.config import IKG_DB_PATH, IKG_MODEL_FILE, IKG_MODEL_PATH
from ai_core.core.content_validator import validate_content_integrity

# 외부 분리 핵심 인프라 도메인 모듈 전격 임포트
from ai_core.core.db_initializer import initialize_database_schema
from ai_core.core.embedder import BGEEmbedder
from ai_core.core.indexer import VectorIndexer

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


async def fetch_dynamic_content(url: str) -> tuple[str | None, str | None]:
    """trafilatura 실패 시 Playwright Headless 브라우저를 띄워 동적 DOM 렌더링 텍스트 추출"""
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()
            await page.goto(url, timeout=15000, wait_until="networkidle")
            content = await page.content()
            title = await page.title()
            return content, title
        except Exception:
            return None, None
        finally:
            if 'browser' in locals():
                await browser.close()


async def scrape_and_save_metadata(bm: dict, semaphore: asyncio.Semaphore, current_idx: int, total_idx: int) -> dict | None:
    """I/O Bound: 개별 북마크 스크래핑 후 정합성 판정에 의거 분기 적재"""
    async with semaphore:
        url = bm["uri"]
        title = bm["title"]
        
        logger.info(f"[{current_idx}/{total_idx}] 스크래핑 파이프라인 가동 ──► URL: {url[:50]}")
        
        try:
            method_used = "Trafilatura(정적)"
            downloaded = trafilatura.fetch_url(url)
            content = trafilatura.extract(downloaded) if downloaded else None
            
            if not content or len(content) < 300 or "JavaScript is disabled" in content:
                method_used = "Playwright(동적)"
                dynamic_content, dynamic_title = await fetch_dynamic_content(url)
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
                    logger.info(f"  ✅ [{current_idx}/{total_idx}] 수집 무결성 통과 ({method_used}) ──► 할당 ID: #{inserted_id}")
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
            logger.error(f"  💥 [{current_idx}/{total_idx}] 런타임 인프라 예외 가드: {str(e)}")
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

    # [CRITICAL CONNECTED] 공통 도메인 초기화 레이어 호출을 통해 테이블 유실 오류 원천 차단
    initialize_database_schema(IKG_DB_PATH)

    logger.info("[BULK IMPORTER] 대용량 지식 자산 일괄 마이그레이션 엔진을 가동합니다.")
    
    embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
    indexer = VectorIndexer(dimension=1024)
    
    with open(json_file_path, encoding="utf-8") as f:
        bookmark_data = json.load(f)
    
    all_bookmarks = extract_bookmarks(bookmark_data)
    total_bookmarks_count = len(all_bookmarks)
    logger.info(f" -> 총 {total_bookmarks_count}개의 백업 메타데이터 스캔 완료.")

    semaphore = asyncio.Semaphore(3)
    faiss_index = faiss.read_index(indexer.index_path)
    
    processed_chunk_count = 0
    total_successfully_indexed = 0

    for chunk_offset, chunk in enumerate(chunk_list(all_bookmarks, BATCH_SIZE)):
        processed_chunk_count += 1
        start_global_idx = (chunk_offset * BATCH_SIZE) + 1
        
        logger.info(f"[CHUNK START] 미니 배치 파이프라인 가동 ({processed_chunk_count}회차)")
        
        tasks = [
            scrape_and_save_metadata(bm, semaphore, start_global_idx + local_idx, total_bookmarks_count) 
            for local_idx, bm in enumerate(chunk)
        ]
        scraping_results = await asyncio.gather(*tasks)
        
        valid_documents = [res for res in scraping_results if res is not None]
        chunk_scraped_count = len(valid_documents)
        
        if chunk_scraped_count == 0:
            continue

        batch_vectors = []
        batch_ids = []
        
        for doc in valid_documents:
            query_vec = embedder.encode(doc["content"])
            batch_vectors.append(query_vec[0].astype("float32"))
            batch_ids.append(doc["id"])
            
        faiss_index.add_with_ids(np.array(batch_vectors).astype("float32"), np.array(batch_ids, dtype=np.int64))
        total_successfully_indexed += chunk_scraped_count
        
        del tasks, scraping_results, valid_documents, batch_vectors, batch_ids
        gc.collect()

    if total_successfully_indexed > 0:
        faiss.write_index(faiss_index, indexer.index_path)
        logger.info(f"[BULK IMPORTER SUCCESS] 마이그레이션 종료. (FAISS 활성 벡터 공간 계층 수: {faiss_index.ntotal})")
    else:
        logger.info("[BULK IMPORTER END] 추가 적재할 신규 활성 자산이 없어 파일 커밋 없이 마감합니다.")


if __name__ == "__main__":
    # 1. be_api와 커널을 공유하는 전역 하이 레졸루션 로깅 인프라 강제 가동
    # 터미널 실시간 출력을 위해 기본 셋업 수준을 INFO로 바인딩
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 2. asyncio 컨텍스트 커널을 통해 메인 벌크 파이프라인 엔진 강제 가동
    try:
        asyncio.run(run_bulk_importer())
    except KeyboardInterrupt:
        logger.warning("[BULK IMPORTER] 사용자에 의해 마이그레이션 연산이 강제 중단되었습니다.")
    except Exception as e:
        logger.critical(f"[BULK IMPORTER CRASH] 런타임 비상 정지: {e}", exc_info=True)