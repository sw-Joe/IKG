from core.embedder import BGEEmbedder
from core.indexer import Indexer


def test_vector_search():
    # 1. 초기화
    model_path = "./model/bge-m3-onnx-int8"
    file_name = "model_quantized.onnx"
    
    print("--- 엔진 및 인덱스 로드 중 ---")
    embedder = BGEEmbedder(model_path=model_path, file_name=file_name)
    indexer = Indexer()
    
    # 2. 테스트 쿼리 입력
    # 검색어를 바꾸어가며 테스트해보세요.
    query = input("\n검색어를 입력하세요: ")
    
    # 3. 쿼리 임베딩 생성
    print(f"'{query}' 의미 분석 중...")
    query_vector = embedder.encode(query)
    
    # 4. 벡터 검색 수행 (FAISS 전용)
    # IKGIndexer 내부에 정의된 search 메서드 활용
    top_k = 5
    distances, indices = indexer.index.search(query_vector.astype('float32'), top_k)
    
    print(f"\n--- '{query}'에 대한 벡터 검색 결과 ---")
    
    found = False
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
            
        # SQLite에서 메타데이터 조회 (id는 1부터 시작하므로 idx + 1)
        cursor = indexer.conn.execute(
            "SELECT title, url FROM bookmarks WHERE id = ?", (int(idx) + 1,)
        )
        row = cursor.fetchone()
        
        if row:
            found = True
            title, url = row
            score = distances[0][i]
            # BGE-M3 정규화 벡터의 경우 내적(IP) 점수는 코사인 유사도와 같음
            print(f"[{i+1}] 점수: {score:.4f}")
            print(f"    제목: {title}")
            print(f"    링크: {url}\n")
            
    if not found:
        print("관련된 북마크를 찾지 못했습니다.")


if __name__ == "__main__":
    test_vector_search()