import asyncio
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

# 로거 구성 및 트랜스포머 경고 필터링
logger = logging.getLogger("ai_core.bulk_indexing")
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
        except Exception as e:
            logger.debug(f"[DYNAMIC FETCH FAIL] {url} : {e}")
            return None, None
        finally:
            if 'browser' in locals():
                await browser.close()


async def scrape_and_save_metadata(bm: dict, semaphore: asyncio.Semaphore) -> dict | None:
    """I/O Bound: 네트워크 세마포어 가드 하에서 웹 페이지 정적/동적 스크래핑 후 SQLite 선적재"""
    async with semaphore:
        url = bm["uri"]
        title = bm["title"]
        
        try:
            downloaded = trafilatura.fetch_url(url)
            content = trafilatura.extract(downloaded) if downloaded else None
            
            if not content or len(content) < 300 or "JavaScript is disabled" in content:
                dynamic_content, dynamic_title = await fetch_dynamic_content(url)
                if dynamic_content:
                    content = trafilatura.extract(dynamic_content)
                    title = dynamic_title or title

            if not content or len(content).strip() < 5:
                return None

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
                return {"id": inserted_id, "content": f"{title} {content}"}
            except sqlite3.IntegrityError:
                return None
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"[SCRAPE CRITICAL ERROR] {url} 처리 실패: {e}")
            return None


async def run_indexing():
    json_file_path = input("북마크 백업파일(json) 파일명 입력>")
    
    if not os.path.exists(json_file_path):
        print(f"[ERROR] 백업 파일이 존재하지 않습니다: {json_file_path}")
        return

    print("[BULK INDEXING] 중앙 설정 인프라 기반 고성능 엔진 초기화를 시작합니다.")
    
    # 1. AI Core 컴포넌트 싱글톤 레이어 로드 (ONNX 세션 최적화 구동)
    embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
    
    # 정정: 미사용 변수 경고를 소멸시키고, 하부 루프의 파일 영속 트랙을 관리할 독립 인스턴스로 바인딩
    indexer = VectorIndexer(dimension=1024)
    
    with open(json_file_path, encoding="utf-8") as f:
        bookmark_data = json.load(f)
    
    all_bookmarks = extract_bookmarks(bookmark_data)
    print(f" -> 총 {len(all_bookmarks)}개의 백업 북마크 메타데이터를 스캔했습니다.")

    # 2. 고병렬 I/O 스크래핑 레이어 가동 (동시성 3 제한)
    semaphore = asyncio.Semaphore(3)
    tasks = [scrape_and_save_metadata(bm, semaphore) for bm in all_bookmarks]
    
    print(" -> 1단계: 비동기 웹 문서 콘텍스트 수집 및 관계형 DB 선적재를 시작합니다...")
    scraping_results = await asyncio.gather(*tasks)
    
    valid_documents = [res for res in scraping_results if res is not None]
    total_scraped = len(valid_documents)
    print(f" -> 수집 성공 및 신규 데이터셋 확보: {total_scraped}건 / {len(all_bookmarks)}건")

    if total_scraped == 0:
        print("[BULK INDEXING END] 추가할 신규 자산이 없어 파이썬 프로세스를 종료합니다.")
        return

    # 3. CPU/Disk Bound: 단일 세션 일괄 인덱싱 구조 가동
    print(" -> 2단계: 고성능 BGE-M3 Dense 임베딩 추론 및 FAISS IDMap 단일 세션 일괄 적재를 시작합니다...")
    
    # 정정: 외부 전역 상수를 중복 호출하는 대신, 인덱서 인스턴스가 초기화 검증을 완결한 index_path 속성을 추적 사용
    index = faiss.read_index(indexer.index_path)
    
    batch_vectors = []
    batch_ids = []
    
    for i, doc in enumerate(valid_documents):
        bookmark_id = doc["id"]
        text_content = doc["content"]
        
        query_vec = embedder.encode(text_content)
        vector_np = query_vec[0].astype("float32")
        
        batch_vectors.append(vector_np)
        batch_ids.append(bookmark_id)
        
        if (i + 1) % 10 == 0 or (i + 1) == total_scraped:
            print(f"    [PROGRESS] 임베딩 연산 진행 중... ({i + 1}/{total_scraped})")

    # 고밀도 행렬 및 고유 ID 데이터 타입 정렬
    final_vectors_np = np.array(batch_vectors).astype("float32")
    final_ids_np = np.array(batch_ids, dtype=np.int64)

    # FAISS 메모리 인덱스 공간에 일괄 융합 (IndexIDMap add_with_ids 규격 완수)
    index.add_with_ids(final_vectors_np, final_ids_np)
    
    # 정정: 영속화 플러시 타깃 경로를 인덱서 내부 설정 경로와 1:1 일치화
    faiss.write_index(index, indexer.index_path)
    print(f"[BULK INDEXING SUCCESS] 전체 벌크 색인이 안전하게 완료되었습니다. (FAISS 전역 벡터 수: {index.ntotal})")


if __name__ == "__main__":
    asyncio.run(run_indexing())