import asyncio
import json
import os

from embedder import BGEEmbedder
from IKGindexer import IKGIndexer



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
    # 설정
    json_file_path = "bookmarks-2026-01-18.json"
    model_path = "./bge-m3-onnx-int8"
    file_name = "model_quantized.onnx" # 양자화 모델 파일명 확인
    
    print("엔진 초기화 중...")
    embedder = BGEEmbedder(model_path=model_path, file_name=file_name)
    indexer = IKGIndexer()
    
    with open(json_file_path, "r", encoding="utf-8") as f:
        bookmark_data = json.load(f)
    
    all_bookmarks = extract_bookmarks(bookmark_data)
    print(f"총 {len(all_bookmarks)}개의 북마크를 발견했습니다.")

    # 세마포어를 이용해 동시 실행 브라우저 수 제한 (리소스 방어)
    # 한 번에 3개씩만 동시 처리
    semaphore = asyncio.Semaphore(3)

    async def sem_task(bm):
        async with semaphore:
            await indexer.add_document(bm['uri'], embedder)

    # 전체 혹은 일부 슬라이싱 처리
    tasks = [sem_task(bm) for bm in all_bookmarks[:20]] 
    await asyncio.gather(*tasks)

    print("\n인덱싱 작업이 완료되었습니다.")


if __name__ == "__main__":
    asyncio.run(run_indexing())