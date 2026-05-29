import re
import sqlite3
from datetime import datetime

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ai_core.embedder import BGEEmbedder


class HybridSearcherV3RankScaled:
    def __init__(
        self,
        db_path="db/ikg_metadata.db",
        index_path="db/ikg_vector.index",
        model_path="./model/bge-m3-onnx-int8",
        alpha=0.4,          # Score Fusion 내 Lexical 가중치
        beta=0.6,           # Score Fusion 내 Semantic 가중치
        decay_lambda=0.001, # 최신성 감쇄 상수
    ):
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

        # 3. 데이터 로드 및 정합성 검증
        self.documents = self._load_all_documents()
        self._check_integrity()

        # 4. 렉시컬 인덱스(BM25) 구축
        tokenized_corpus = [
            self._preprocess_tech_text(doc["content"]) for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)


    def _load_all_documents(self):
        """SQLite Row Factory를 활용하여 인덱스 에러 방지 및 가독성 확보"""
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, url, title, content, created_at FROM bookmarks ORDER BY id ASC"
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


    def _check_integrity(self):
        if len(self.documents) != self.index.ntotal:
            raise ValueError(
                f"정합성 오류: DB 문서 수({len(self.documents)})와 FAISS 인덱스 벡터 수({self.index.ntotal})가 일치하지 않습니다."
            )


    def _preprocess_tech_text(self, text):
        text_lower = text.lower()
        tokens = re.findall(r'[a-zA-Z0-9_\-\.]+|[가-힣]+', text_lower)
        return tokens


    def _get_time_decay(self, created_at_str):
        try:
            dt = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            days_diff = (now - dt).days
            if days_diff < 0:
                days_diff = 0
            return np.exp(-self.decay_lambda * days_diff)
        except Exception:
            return 1.0


    def search(self, query: str, top_n=5):
        tokenized_query = self._preprocess_tech_text(query)
        if not tokenized_query:
            return []

        # 1. Dense Semantic Search (FAISS 전역 탐색)
        query_vec = self.embedder.encode(query)
        v_scores, v_indices = self.index.search(
            query_vec.astype("float32"), self.index.ntotal
        )
        
        # FAISS 전역 인덱스 순위 맵 구성 (O(N) 매핑 대응 복잡도 완화 목적)
        faiss_ranks = {int(idx): rank for rank, idx in enumerate(v_indices[0])}

        # 2. Sparse Lexical Search (BM25 전역 역색인 계산)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # BM25 순위 계산을 위한 정렬 인덱스 맵 생성 (0-based Rank)
        sorted_bm25_indices = np.argsort(bm25_scores)[::-1]
        bm25_ranks = {int(idx): rank for rank, idx in enumerate(sorted_bm25_indices)}

        # BM25 점수 Min-Max 정규화 
        max_bm25 = np.max(bm25_scores)
        min_bm25 = np.min(bm25_scores)
        bm25_denom = max_bm25 - min_bm25 + 1e-9
        
        if max_bm25 > 0:
            bm25_scores_norm = (bm25_scores - min_bm25) / bm25_denom
        else:
            bm25_scores_norm = bm25_scores

        # 3. 통합 순위 변조형 스코어링 (Rank-Scaled Fusion)
        combined_results = []
        total_docs = len(self.documents)

        for i in range(total_docs):
            s_lex_norm = bm25_scores_norm[i]
            
            # FAISS 스코어 파싱
            s_sem = float(v_scores[0][np.where(v_indices[0] == i)[0][0]]) if i in faiss_ranks else 0.0

            # 개별 엔진 전역 순위 확보 (순위가 없을 시 최하위 순위 보정)
            rank_lex = bm25_ranks.get(i, total_docs)
            rank_sem = faiss_ranks.get(i, total_docs)

            # 최신성 감쇄 인자 계산
            f_time = self._get_time_decay(self.documents[i]["created_at"])

            # 렉시컬 게이트 제약 조건 적용
            p_gate = 1.0 if bm25_scores[i] > 0 else 0.5

            # Base Score Fusion 연산
            base_fusion_score = (self.alpha * s_lex_norm + self.beta * s_sem)

            # [핵심] Rank-Scaled Damping Filter 적용
            # 자연로그 분모 구성을 통해 순위가 밀릴수록 점수가 부드럽고 강력하게 하락하도록 제어
            # 분모가 0이 되는 것을 방지하기 위해 각 Rank에 +1 보정 진행
            rank_penalty = 1.0 / (np.log(rank_lex + 1) + np.log(rank_sem + 1) + 1e-9)
            
            final_score = base_fusion_score * rank_penalty * f_time * p_gate

            # 대조 실험 데이터 추출을 위한 로그 적재
            res_entry = self.documents[i].copy()
            res_entry.update({
                "score_final": round(float(final_score), 4),
                "score_lex": round(float(s_lex_norm), 4),
                "score_sem": round(float(s_sem), 4),
                "factor_time": round(float(f_time), 4),
                "factor_gate": p_gate,
                "rank_lex": rank_lex,
                "rank_sem": rank_sem,
                "factor_rank_penalty": round(float(rank_penalty), 4)
            })
            combined_results.append(res_entry)

        # 최종 점수 기준 내림차순 정렬 및 반환
        combined_results.sort(key=lambda x: x["score_final"], reverse=True)
        return combined_results[:top_n]