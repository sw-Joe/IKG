import sqlite3

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ai_core.embedder import BGEEmbedder


class HybridSearcher:
    def __init__(self, db_path="db/ikg_metadata.db", index_path="db/ikg_vector.index", model_path="./model/bge-m3-onnx-int8"):
        # 1. 자원 로드
        self.conn = sqlite3.connect(db_path)
        self.index = faiss.read_index(index_path)
        self.embedder = BGEEmbedder(model_path=model_path, file_name="model_quantized.onnx")
        
        # 2. BM25 역색인 구축을 위한 데이터 로드
        self.documents = self._load_all_documents()
        # 공백 기반 토큰화 (전처리 로직)
        tokenized_corpus = [doc['content'].split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)


    def _load_all_documents(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, url, title, content FROM bookmarks ORDER BY id ASC")
        rows = cursor.fetchall()
        return [{"id": r[0], "url": r[1], "title": r[2], "content": r[3]} for r in rows]


    def _rrf(self, vector_results, bm25_results, k=60):
        """
        Reciprocal Rank Fusion 알고리즘
        두 검색 엔진의 '순위'를 합산하여 최종 점수를 계산합니다.
        """
        scores = {}
        
        # Vector 검색 순위 반영 (FAISS 인덱스 번호 기준)
        for rank, idx in enumerate(vector_results):
            scores[idx] = scores.get(idx, 0) + (1.0 / (k + rank + 1))
            
        # BM25 검색 순위 반영
        for rank, idx in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + (1.0 / (k + rank + 1))
            
        # 점수 기준 정렬된 인덱스 리스트 반환
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


    def search(self, query: str, top_n=5):
        # 1. Vector Search (Semantic)
        query_vec = self.embedder.encode(query)
        _, v_indices = self.index.search(query_vec.astype('float32'), top_n * 2)
        v_results = v_indices[0].tolist()

        # 2. BM25 Search (Keyword)
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        b_results = np.argsort(bm25_scores)[::-1][:top_n * 2].tolist()

        # 3. RRF 통합
        final_indices = self._rrf(v_results, b_results)
        
        # 4. 결과 매핑 및 반환
        search_results = []
        for idx in final_indices[:top_n]:
            if idx < len(self.documents):
                search_results.append(self.documents[idx])
        
        return search_results