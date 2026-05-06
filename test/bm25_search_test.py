import sqlite3

import numpy as np
from rank_bm25 import BM25Okapi


def test_bm25_search():
    # 1. 데이터 로드
    db_path = "ikg_metadata.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("--- SQLite에서 데이터 로드 중 ---")
    cursor.execute("SELECT title, url, content FROM bookmarks")
    rows = cursor.fetchall()
    
    if not rows:
        print("데이터베이스에 인덱싱된 문서가 없습니다. 먼저 인덱싱을 진행해 주세요.")
        return

    documents = [{"title": r[0], "url": r[1], "content": r[2]} for r in rows]
    
    # 2. BM25 역색인 구축
    # 한국어 조사를 완벽히 분리하려면 형태소 분석기가 필요하지만, 
    # 프로토타입 단계에서는 '공백 기반 토큰화'로 베이스라인을 잡습니다.
    print("BM25 역색인 생성 중...")
    corpus = [doc['content'].split() for doc in documents]
    bm25 = BM25Okapi(corpus)

    # 3. 테스트 쿼리 입력
    query = input("\n검색할 키워드를 입력하세요: ")
    tokenized_query = query.split()

    # 4. 키워드 검색 수행
    print(f"'{query}' 키워드 매칭 중...")
    scores = bm25.get_scores(tokenized_query)
    
    # 점수 기준 내림차순 정렬
    top_n = 5
    top_indices = np.argsort(scores)[::-1][:top_n]

    print(f"\n--- '{query}'에 대한 BM25 검색 결과 ---")
    
    found = False
    for i, idx in enumerate(top_indices):
        if scores[idx] > 0:  # 일치하는 단어가 하나라도 있는 경우
            found = True
            doc = documents[idx]
            print(f"[{i+1}] 점수: {scores[idx]:.4f}")
            print(f"    제목: {doc['title']}")
            print(f"    링크: {doc['url']}\n")
            
    if not found:
        print("일치하는 단어를 포함한 문서를 찾지 못했습니다.")


if __name__ == "__main__":
    test_bm25_search()