import asyncio
import json
import logging
from ai_core.config import IKG_MODEL_PATH, IKG_MODEL_FILE
from ai_core.core.embedder import BGEEmbedder
from ai_core.core.indexer import Indexer

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def extract_bookmarks(node):
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

async def run_indexing():
    json_file_path = "bookmarks-2026-01-18.json"
    
    print("[BULK INDEXING] 중앙 설정 파일 기반 엔진 초기화를 시작합니다.")
    embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
    indexer = Indexer()
    
    with open(json_file_path, encoding="utf-8") as f:
        bookmark_data = json.load(f)
    
    all_bookmarks = extract_bookmarks(bookmark_data)
    print(f" -> 총 {len(all_bookmarks)}개의 북마크 데이터 스캔 성공.")

    semaphore = asyncio.Semaphore(3)

    async def sem_task(bm):
        async with semaphore:
            try:
                await indexer.index_url(bm["uri"], embedder)
            except Exception as e:
                print(f" 예외 스킵 처리 ({bm['uri']}): {e}")

    tasks = [sem_task(bm) for bm in all_bookmarks]
    await asyncio.gather(*tasks)
    
    indexer.save_index()
    print("[BULK INDEXING FINISHED] 전체 벌크 변환이 성공적으로 완결되었습니다.")