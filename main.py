import json
import os
from datetime import datetime

from embedder import BGEEmbedder
from IKGindexer import IKGIndexer



def extract_bookmarks(node):
    """
    북마크 트리 구조를 재귀적으로 탐색하여 (title, uri) 리스트를 반환합니다.
    """
    bookmarks = []
    
    # typeCode 1은 개별 북마크(uri 존재), 2는 폴더(children 존재)를 의미합니다.
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

def main():
    # 1. 초기 설정
    json_file_path = "bookmarks-2026-01-18.json"
    model_path = "./bge-m3-onnx-int8"
    file_name = "model.onnx"
    
    # 2. 임베더 및 인덱서 초기화
    print("엔진 초기화 중...")
    embedder = BGEEmbedder(model_path=model_path, file_name=file_name)
    indexer = IKGIndexer()
    
    # 3. JSON 파일 로드 및 데이터 추출
    if not os.path.exists(json_file_path):
        print(f"오류: {json_file_path} 파일을 찾을 수 없습니다.")
        return

    with open(json_file_path, "r", encoding="utf-8") as f:
        bookmark_data = json.load(f)
    
    all_bookmarks = extract_bookmarks(bookmark_data)
    print(f"총 {len(all_bookmarks)}개의 북마크를 발견했습니다.")

    # 4. 루프를 돌며 인덱싱 수행
    # 테스트를 위해 상위 5개만 먼저 진행해보는 것을 권장합니다.
    for i, bm in enumerate(all_bookmarks[:10]): 
        print(f"[{i+1}/{len(all_bookmarks)}] 처리 중: {bm['title']}")
        try:
            # IKGIndexer의 add_document는 내부적으로 trafilatura를 사용하여 
            # 실제 웹 페이지의 최신 본문을 긁어와 벡터화합니다.
            indexer.add_document(bm['uri'], embedder)
        except Exception as e:
            print(f"실패: {bm['uri']} - 사유: {e}")

    print("\n인덱싱 작업이 완료되었습니다.")


if __name__ == "__main__":
    main()