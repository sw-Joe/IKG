import re
import sqlite3
from datetime import datetime

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from embedder import BGEEmbedder


class HybridSearcher:
    def __init__(
        self,
        db_path="db/ikg_metadata.db",
        index_path="db/ikg_vector.index",
        model_path="./model/bge-m3-onnx-int8",
        alpha=0.4,  # Lexical 가중치
        beta=0.6,  # Semantic 가중치
        decay_lambda=0.001,
    ):  # 최신성 감쇄 상수

        # 1. 자원 로드
        self.conn = sqlite3.connect(db_path)
        self.index = faiss.read_index(index_path)
        self.embedder = BGEEmbedder(
            model_path=model_path, file_name="model_quantized.onnx"
        )

        # 2. 하이퍼파라미터 설정
        self.alpha = alpha
        self.beta = beta
        self.decay_lambda = decay_lambda

        # 3. 데이터 로드 및 무결성 체크
        self.documents = self._load_all_documents()
        self._check_integrity()  # [V3 개선] 시스템 정합성 가드 도입 [1, 2]

        # 4. 시맨틱 보존형 전처리 및 BM25 구축
        tokenized_corpus = [
            self._preprocess_tech_text(doc["content"]) for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)


    def _check_integrity(self):
        """SQLite와 FAISS의 데이터 개수가 일치하는지 확인 [2]"""
        db_count = len(self.documents)
        faiss_count = self.index.ntotal
        if db_count != faiss_count:
            print(f"[Warning] 무결성 불일치: DB({db_count}) != FAISS({faiss_count})")
        else:
            print(f"[Info] 시스템 무결성 확인 완료 (Count: {db_count})")


    def _preprocess_tech_text(self, text):
        """기술 도메인 기호(C++, C#, .js 등)를 보존하는 전처리 [1, 3]"""
        if not text:
            return []
        # 기술 기호 보존용 정규표현식 적용
        tokens = re.findall(r"[a-zA-Z0-9+#\.]+", text.lower())
        return tokens


    def _load_all_documents(self):
        """최신성 계산을 위해 created_at 필드 추가 로드 [1, 4]"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, url, title, content, created_at FROM bookmarks ORDER BY id ASC"
        )
        rows = cursor.fetchall()
        return [
            {"id": r[0], "url": r[1], "title": r[2], "content": r[3], "created_at": r[4]}
            for r in rows
        ]


    def _get_time_decay(self, created_at_str):
        """지수 감쇄 함수를 이용한 최신성 점수 계산 [1, 4]"""
        try:
            created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
            days_old = (datetime.now() - created_at).days
            return np.exp(-self.decay_lambda * days_old)
        except:
            return 0.5  # 날짜 데이터 오류 시 중간값 부여


    def search(self, query: str, top_n=5):
        # 1. 시맨틱 보존형 쿼리 전처리
        tokenized_query = self._preprocess_tech_text(query)

        # 2. Vector Search (Semantic)
        query_vec = self.embedder.encode(query)
        v_scores, v_indices = self.index.search(
            query_vec.astype("float32"), self.index.ntotal
        )

        # 3. BM25 Search (Lexical)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # BM25 점수 Min-Max 정규화 (0~1 범위) [1, 9]
        if np.max(bm25_scores) > 0:
            bm25_scores_norm = (bm25_scores - np.min(bm25_scores)) / (
                np.max(bm25_scores) - np.min(bm25_scores) + 1e-9
            )
        else:
            bm25_scores_norm = bm25_scores

        # 4. 최종 통합 스코어링 (V3 공식 적용) [1]
        # S_final = (alpha * Norm(S_lex) + beta * Norm(S_sem)) * f_time * P_gate
        combined_results = []

        for i in range(len(self.documents)):
            s_lex = bm25_scores_norm[i]
            # FAISS 인덱스 순서와 self.documents 순서가 일치한다고 가정
            # (정규화된 유사도 점수 추출, BGE-M3는 이미 0~1 범위 선호)
            s_sem = v_scores[np.where(v_indices == i)] if i in v_indices else 0

            # [개선안 1] 최신성 감쇄 적용 [4]
            f_time = self._get_time_decay(self.documents[i]["created_at"])

            # [개선안 2] 환각 방지 게이트 (Lexical Gate) [1, 10]
            # 키워드 일치가 전혀 없으면(s_lex == 0) 패널티 0.5 부여
            p_gate = 1.0 if s_lex > 0 else 0.5

            final_score = (self.alpha * s_lex + self.beta * s_sem) * f_time * p_gate

            combined_results.append({"doc": self.documents[i], "score": final_score})

        # 5. 최종 점수 기준 정렬 및 반환
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return [res["doc"] for res in combined_results[:top_n]]
