import asyncio
import json
import logging
import os
import sqlite3
import gc

import faiss
import numpy as np
from playwright.async_api import async_playwright

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH, IKG_MODEL_FILE, IKG_MODEL_PATH
from ai_core.core.embedder import BGEEmbedder
from ai_core.core.indexer import VectorIndexer
from ai_core.core.db_initializer import initialize_database_schema

# 💡 [REFACTORED]: bookmark_scraper 모듈로 이관된 수집 및 검증 기능 레이어 집중 수입
from ai_core.core.bookmark_scraper import (
    extract_bookmarks,
    fetch_dynamic_content_with_context,
    validate_scraped_bookmark
)

logger = logging.getLogger("ai_core.bulk_importer")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


async def run_bulk_import(json_file_path: str):
    """
    [CQRS BATCH]: 대량의 브라우저 백업 JSON 데이터를 로드하여, 웹 스크래핑 및 유효성 검증 후
    SQLite 및 FAISS 벡터 차원 공간에 일괄 적재(Bulk Import)하는 전담 파이프라인
    """
    logger.info(f"[BULK IMPORT] 일괄 적재 시퀀스 가동 -> 대상 파일: {json_file_path}")
    
    # 1. 인프라 정합성 사전 검증 가드
    if not os.path.exists(json_file_path):
        logger.error(f"[BULK IMPORT ERROR] 대상 백업 JSON 파일이 지정된 경로에 부재합니다: {json_file_path}")
        return

    # 데이터베이스 초기 뼈대 무결성 확인
    initialize_database_schema(IKG_DB_PATH)

    with open(json_file_path, "r", encoding="utf-8") as f:
        root_node = json.load(f)

    # 2. bookmark_scraper 전담 모듈로부터 위임 수입한 재귀 트리 파서 가동
    bookmarks = extract_bookmarks(root_node)
    total_bookmarks = len(bookmarks)
    logger.info(f" -> 브라우저 백업 트리 변환 완료: 총 {total_bookmarks}개의 고유 자산 노드 식별됨.")

    if total_bookmarks == 0:
        logger.warning("[BULK IMPORT CANCELLED] 파싱된 유효 자산 수량이 0건입니다.")
        return

    # 3. STAGE 1: Playwright 브라우저 배치 격리 세션 커널 생성 (동적 스크래핑 극대화 가속)
    logger.info("[BULK SCRAPING] 1단계: 동적 커널 기반 데이터 수집 및 정보 가치 필터링 세션 개시...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        conn = sqlite3.connect(IKG_DB_PATH, timeout=60.0)
        cursor = conn.cursor()
        
        try:
            for idx, b in enumerate(bookmarks, 1):
                url = b["uri"]
                logger.info(f"[{idx}/{total_bookmarks}] 대용량 배치 크롤링 타깃 스캔 -> {url}")
                
                # 수입된 동적 컨텍스트 크롤러 및 유효성 가드라인 작동
                title, content = await fetch_dynamic_content_with_context(context, url)
                is_valid, reason_or_content = validate_scraped_bookmark(title, content)
                
                final_title = title or b["title"]
                
                if is_valid:
                    # 정보 무결성을 통과한 정상 자산군 타깃 분기
                    if len(reason_or_content) < 100:
                        cursor.execute(
                            """
                            INSERT INTO bookmarks_isolated (url, title, content, created_at, isolation_reason)
                            VALUES (?, ?, ?, datetime('now', 'localtime'), ?)
                            """,
                            (url, final_title, reason_or_content, "TEXT_LENGTH_INSUFFICIENT_UNDER_100")
                        )
                    else:
                        cursor.execute(
                            """
                            INSERT INTO bookmarks (url, title, content, created_at, is_deleted, index_written)
                            VALUES (?, ?, ?, datetime('now', 'localtime'), 0, 0)
                            """,
                            (url, final_title, reason_or_content)
                        )
                else:
                    # 웹 방화벽 차단, 404, 혹은 본문 훼손 자산 -> 격리 샌드박스 테이블 이관 (FAISS 인큐잉 원천 격리 차단)
                    cursor.execute(
                        """
                        INSERT INTO bookmarks_isolated (url, title, content, created_at, isolation_reason)
                        VALUES (?, ?, ?, datetime('now', 'localtime'), ?)
                        """,
                        (url, final_title, content or "", reason_or_content)
                    )
                
                # 50건 단위 부분 물리 플러시 트랜잭션 최적화로 동시성 DB 잠금 경합 최소화
                if idx % 50 == 0:
                    conn.commit()
                    
            conn.commit()
            logger.info(" -> [STAGE 1 COMPLETE] 전역 지식 원문 텍스트 SQLite 정형 적재 세션 종료.")
        except Exception as e:
            conn.rollback()
            logger.error(f"[BULK IMPORT CRITICAL ERROR] 웹 데이터 일괄 수집 단계 트랜잭션 붕괴: {e}", exc_info=True)
            return
        finally:
            conn.close()
            await browser.close()

    # 4. STAGE 2: 고성능 차원 밀집 행렬 분할 청크 인덱싱 빌드 스테이지 진입
    logger.info("[BULK INDEXING] 2단계: FAISS 고밀도 기하학 벡터 공간 빌딩 프로세스 가동...")
    
    embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
    indexer = VectorIndexer(dimension=1024)
    faiss_index = faiss.read_index(IKG_INDEX_PATH)

    conn = sqlite3.connect(IKG_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, title, content FROM bookmarks WHERE is_deleted = 0 AND index_written = 0")
    unindexed_rows = cursor.fetchall()
    conn.close()

    total_unindexed = len(unindexed_rows)
    logger.info(f" -> 벡터 변환 대상 미인덱싱 가용 자산: 총 {total_unindexed}건 탐색 완료.")

    if total_unindexed == 0:
        logger.info("[BULK INDEXING] 차원 인입이 필요한 신규 자산 스냅샷이 존재하지 않습니다. 동기화를 수렴 종료합니다.")
        return

    batch_size = 100
    total_successfully_indexed = 0

    for i in range(0, total_unindexed, batch_size):
        chunk_rows = unindexed_rows[i : i + batch_size]
        current_chunk_count = len(chunk_rows)
        inference_batch_num = (i // batch_size) + 1

        batch_texts = [f"{row['title']} {row['content']}" for row in chunk_rows]
        batch_ids = [row['id'] for row in chunk_rows]

        # [True Batch Inference]: 단건 순회를 타파하고 100개 행렬을 ONNX C++ 커널에 단 1회 밀어 넣어 하드웨어 최대 가속
        chunk_vectors_np = embedder.encode_batch(batch_texts)
        chunk_ids_np = np.array(batch_ids, dtype=np.int64)

        # FAISS 인메모리 차원 일련 가상 번호 공간 맵 적재
        faiss_index.add_with_ids(chunk_vectors_np, chunk_ids_np)
        total_successfully_indexed += current_chunk_count

        # SQLite 동기화 체크포인트 마킹 일괄 집행
        conn = sqlite3.connect(IKG_DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.executemany(
                "UPDATE bookmarks SET index_written = 1 WHERE id = ?",
                [(b_id,) for b_id in batch_ids]
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"[BULK INDEXING CHECKPOINT ERROR] DB 물리 인덱싱 마킹 실패 변수 제어 가드 작동: {e}")
        finally:
            conn.close()

        # [메모리 가드]: 100단위 텐서 수학 연산 버퍼 찌꺼기 즉시 물리 소거 및 RAM 반환
        del chunk_vectors_np, chunk_ids_np, batch_texts, batch_ids
        gc.collect()

        logger.info(f" -> [INDEX BATCH #{inference_batch_num} FLUSHED] 다차원 벡터 인메모리 병합 및 고밀도 텐서 가비지 소거 완료.")

    # 양대 스테이지의 모든 분할 배치가 완벽하게 안착한 최종 시점에 단 1회의 원자적 파일 영구 커밋(write) 집행
    if total_successfully_indexed > 0:
        faiss.write_index(faiss_index, IKG_INDEX_PATH)
        logger.info(f"[BULK IMPORT SUCCESS] 총 {total_successfully_indexed}건의 지식 가치 자산이 FAISS 공간 및 SQLite 동기화 정착 완료되었습니다.")


if __name__ == "__main__":
    import sys
    target_json = sys.argv[1] if len(sys.argv) > 1 else "bookmarks.json"
    asyncio.run(run_bulk_import(target_json))