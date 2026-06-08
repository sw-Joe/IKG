import asyncio
import gc
import json
import logging
import os
import sqlite3
from datetime import datetime

import faiss
import numpy as np
import trafilatura
from playwright.async_api import async_playwright

from ai_core.config import IKG_DB_PATH, IKG_MODEL_FILE, IKG_MODEL_PATH
from ai_core.core.embedder import BGEEmbedder
from ai_core.core.indexer import VectorIndexer

# 전역 로거 구성 및 외산 라이브러리 업스트림 소음 필터링
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
            # 로컬 하드웨어 자원 보호를 위해 15초 하드 타임아웃 제약 유지
            await page.goto(url, timeout=15000, wait_until="networkidle")
            content = await page.content()
            title = await page.title()
            return content, title
        except Exception as e:
            # 개별 도메인별 세부 타임아웃/접근 거부는 디버그 로그로 격리
            logger.debug(f"[DYNAMIC FETCH TIMEOUT/FAIL] {url} : {e}")
            return None, None
        finally:
            if 'browser' in locals():
                await browser.close()


async def scrape_and_save_metadata(bm: dict, semaphore: asyncio.Semaphore, current_idx: int, total_idx: int) -> dict | None:
    """I/O Bound: 네트워크 세마포어 가드 하에서 웹 페이지 정적/동적 스크래핑 후 SQLite 선적재 및 실시간 경량 로깅"""
    async with semaphore:
        url = bm["uri"]
        title = bm["title"]
        
        # 1. 단일 북마크 처리 개시 로그 (현재 진행 상황 직관적 전달)
        logger.info(f"[{current_idx}/{total_idx}] 스크래핑 시도 중... ──► Title: {title[:20]}... | URL: {url[:50]}")
        
        try:
            method_used = "Trafilatura(정적)"
            # 1. 정적 스크래핑 1차 시도
            downloaded = trafilatura.fetch_url(url)
            content = trafilatura.extract(downloaded) if downloaded else None
            
            # 본문이 부실하거나 JS 차단 문구가 걸릴 경우 동적 렌더링 브라우저 세션 전환
            if not content or len(content) < 300 or "JavaScript is disabled" in content:
                method_used = "Playwright(동적)"
                dynamic_content, dynamic_title = await fetch_dynamic_content(url)
                if dynamic_content:
                    content = trafilatura.extract(dynamic_content)
                    title = dynamic_title or title

            if not content or len(content).strip() < 5:
                logger.warning(f"  ❌ [{current_idx}/{total_idx}] 수집 실패 (본문 텍스트 없음 또는 규격 미달) ──► URL: {url[:50]}")
                return None

            # 2. SQLite DB 선적재 오퍼레이션 집행 (WAL 모드 격리 커밋)
            conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO bookmarks (url, title, content, created_at, is_deleted) 
                    VALUES (?, ?, ?, ?, 0)
                    """,
                    (url, title, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                inserted_id = cursor.lastrowid
                conn.commit()
                
                # 단일 북마크 적재 성공에 대한 명확한 한 줄 피드백
                logger.info(f"  ✅ [{current_idx}/{total_idx}] 수집 성공 ({method_used}) ──► 할당 ID: #{inserted_id} | 원문: {len(content)}자")
                return {"id": inserted_id, "content": f"{title} {content}"}
                
            except sqlite3.IntegrityError:
                # 고유 제약 조건 위배(이미 존재하는 URL 자산) 시 중복 인덱싱 스킵
                logger.info(f"  ⏩ [{current_idx}/{total_idx}] 스킵 (이미 영속 DB에 존재하는 URL 자산입니다.)")
                return None
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"  💥 [{current_idx}/{total_idx}] 크리티컬 예외 발생 ──► URL: {url[:50]} | 에러: {str(e)}")
            return None


def chunk_list(lst, n):
    """대용량 데이터를 시스템이 소화 가능한 미니 배치 청크 단위로 분할 유도"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


async def run_bulk_importer():
    json_file_path = "bookmarks-2026-01-18.json"
    BATCH_SIZE = 100  # 4000건 이상을 안전하게 수렴시키기 위한 최적의 미니 배치 크기
    
    if not os.path.exists(json_file_path):
        logger.error(f"백업 파일이 지정된 경로에 실존하지 않습니다: {json_file_path}")
        return

    logger.info("[BULK IMPORTER] 대용량 지식 자산 일괄 마이그레이션 엔진을 구동합니다.")
    
    # 1. AI Core 인프라 컴포넌트 싱글톤 레이어 로드
    embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
    indexer = VectorIndexer(dimension=1024)
    
    with open(json_file_path, encoding="utf-8") as f:
        bookmark_data = json.load(f)
    
    all_bookmarks = extract_bookmarks(bookmark_data)
    total_bookmarks_count = len(all_bookmarks)
    logger.info(f" -> 총 {total_bookmarks_count}개의 백업 메타데이터 스캔 완결.")

    # 2. 고병렬 I/O 제어 레이어 동시성 3 제약 바인딩
    semaphore = asyncio.Semaphore(3)
    
    # 디스크로부터 영속 FAISS 인덱스 핸들러 단 1회 로드
    faiss_index = faiss.read_index(indexer.index_path)
    
    processed_chunk_count = 0
    total_successfully_imported = 0

    # 3. 미니 배치 청크 단위 루프 전개
    for chunk_offset, chunk in enumerate(chunk_list(all_bookmarks, BATCH_SIZE)):
        processed_chunk_count += 1
        start_global_idx = (chunk_offset * BATCH_SIZE) + 1
        
        logger.info("======================================================================================")
        logger.info(f"[CHUNK START] 미니 배치 파이프라인 가동 ({processed_chunk_count}회차 / 전체 대역: {start_global_idx}번 ~ {min(start_global_idx + BATCH_SIZE - 1, total_bookmarks_count)}번)")
        logger.info("======================================================================================")
        
        # 3-1. 현재 청크 대역 비동기 병렬 스크래핑 및 SQLite 선적재 동시 집행 (글로벌 인덱스 번호 동적 주입)
        tasks = [
            scrape_and_save_metadata(bm, semaphore, start_global_idx + local_idx, total_bookmarks_count) 
            for local_idx, bm in enumerate(chunk)
        ]
        scraping_results = await asyncio.gather(*tasks)
        
        valid_documents = [res for res in scraping_results if res is not None]
        chunk_scraped_count = len(valid_documents)
        
        if chunk_scraped_count == 0:
            logger.info(" -> 현재 청크 범위 내에 신규 적재할 자산이 없습니다. 다음 청크로 이동합니다.")
            continue

        # 3-2. 현재 청크 기점 CPU-Bound ONNX 고속 행렬 임베딩 추론 집행
        logger.info(f" -> [{processed_chunk_count}회차 청크 수집완료] 신규 자산 {chunk_scraped_count}건에 대한 BGE-M3 ONNX 임베딩 연산 착수...")
        batch_vectors = []
        batch_ids = []
        
        for doc in valid_documents:
            bookmark_id = doc["id"]
            text_content = doc["content"]
            
            query_vec = embedder.encode(text_content)
            vector_np = query_vec[0].astype("float32")
            
            batch_vectors.append(vector_np)
            batch_ids.append(bookmark_id)
            
        # 데이터 타입 고속 행렬 매핑 정렬
        final_vectors_np = np.array(batch_vectors).astype("float32")
        final_ids_np = np.array(batch_ids, dtype=np.int64)
        
        # 3-3. FAISS 인메모리 공간에 임시 증분 결합
        faiss_index.add_with_ids(final_vectors_np, final_ids_np)
        
        total_successfully_imported += chunk_scraped_count
        logger.info(f" -> [{processed_chunk_count}회차 청크 색인완료] 누적 수렴 자산: {total_successfully_imported}건 / 전체: {total_bookmarks_count}건")
        
        # 3-4. 메모리 누수 방지를 위한 청크 말단 가비지 컬렉터 강제 호출
        del tasks, scraping_results, valid_documents, batch_vectors, batch_ids
        del final_vectors_np, final_ids_np
        gc.collect()

    # 4. 모든 청크 순회가 끝난 후, 단 1회의 원자적 디스크 영구 플러시 집행 (I/O 비용 최소화)
    if total_successfully_imported > 0:
        faiss.write_index(faiss_index, indexer.index_path)
        logger.info(f"[BULK IMPORTER SUCCESS] 전량 마이그레이션이 무결하게 종료되었습니다. (FAISS 전역 벡터 계층 수: {faiss_index.ntotal})")
    else:
        logger.info("[BULK IMPORTER END] 추가 적재할 신규 고유 자산이 존재하지 않아 파일 쓰기 없이 종료합니다.")


if __name__ == "__main__":
    asyncio.run(run_bulk_importer())