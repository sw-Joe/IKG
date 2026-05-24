import re
import sqlite3
from datetime import datetime
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from core.embedder import BGEEmbedder



class HybridSearcher:
    def __init__(
        self,
        db_path="db/ikg_metadata.db",
        index_path="db/ikg_vector.index",
        model_path="./model/bge-m3-onnx-int8",
        alpha=0.4,  # Lexical 가중치
        beta=0.6,   # Semantic 가중치
        decay_lambda=0.001,  # 최신성 감쇄 상수
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

        # 3. 데이터 로드 및 무결성 체크
        self.documents = self._load_all_documents()
        self._check_integrity()

        # 4. 시맨틱 보존형 전처리 및 BM25 구축
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
        tokens = re.findall(r"[a-zA-Z0-9_\-\.]+|[가-힣]+", text_lower)
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

        # 1. Vector Search (Semantic) - FAISS 전역 탐색
        query_vec = self.embedder.encode(query)
        v_scores, v_indices = self.index.search(
            query_vec.astype("float32"), self.index.ntotal
        )
        
        # [버그 수정]: 2차원 넘파이 배열인 v_indices에서 정확한 전역 순위 및 인덱스 추출을 위한 딕셔너리 매핑
        faiss_ranks = {int(idx): rank for rank, idx in enumerate(v_indices[0])}
        faiss_scores_dict = {int(idx): float(score) for idx, score in zip(v_indices[0], v_scores[0])}

        # 2. BM25 Search (Lexical)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # BM25 순위 매핑 딕셔너리 생성
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

        # 3. 최종 통합 스코어링 및 대조 실험용 로그 적재
        combined_results = []
        total_docs = len(self.documents)

        for i in range(total_docs):
            s_lex_norm = bm25_scores_norm[i]
            
            # [버그 수정]: 딕셔너리 조회를 통해 정확한 매칭 점수 매핑 (순위권 외 문서 가드 처리)
            s_sem = faiss_scores_dict.get(i, 0.0)

            # 최신성 감쇄 적용
            f_time = self._get_time_decay(self.documents[i]["created_at"])

            # 환각 방지용 렉시컬 게이트 적용
            p_gate = 1.0 if bm25_scores[i] > 0 else 0.5

            # v2 스코어 결합 공식 적용
            final_score = (self.alpha * s_lex_norm + self.beta * s_sem) * f_time * p_gate

            # v3 규격 대조 실험 전산 순위 확보
            rank_lex = bm25_ranks.get(i, total_docs)
            rank_sem = faiss_ranks.get(i, total_docs)

            # [핵심 변경 사항]: 요구 문서의 대조 실험용 디버깅 데이터 로그 적재 
            res_entry = self.documents[i].copy()
            res_entry.update({
                "score_final": round(float(final_score), 4),
                "score_lex": round(float(s_lex_norm), 4),
                "score_sem": round(float(s_sem), 4),
                "factor_time": round(float(f_time), 4),
                "factor_gate": p_gate,
                "rank_lex": rank_lex,
                "rank_sem": rank_sem,
                "factor_rank_penalty": 1.0  # Baseline v2는 순위 감쇄가 없으므로 항등원(1.0) 고정 처리
            })
            combined_results.append(res_entry)

        # 최종 가중치 결합 점수 기준 내림차순 정렬 및 반환
        combined_results.sort(key=lambda x: x["score_final"], reverse=True)
        return combined_results[:top_n]