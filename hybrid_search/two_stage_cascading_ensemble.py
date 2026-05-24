import re
import sqlite3
from datetime import datetime

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from core.embedder import BGEEmbedder


class HybridSearcherV3Stage1:
    def __init__(
        self,
        db_path="db/ikg_metadata.db",
        index_path="db/ikg_vector.index",
        model_path="./model/bge-m3-onnx-int8",
        alpha=0.4,          # 2단계 Score Fusion 내 Lexical 가중치
        beta=0.6,           # 2단계 Score Fusion 내 Semantic 가중치
        decay_lambda=0.001, # 최신성 감쇄 상수
        stage1_k=30,        # 1단계에서 각 엔진별로 확보할 1차 후보군 수
        rrf_k=60            # RRF 순위 산출용 상수
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
        self.stage1_k = stage1_k
        self.rrf_k = rrf_k

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

        # =====================================================================
        # STAGE 1: High Recall (각 검색 엔진별 상위 후보군 추출 및 RRF 융합)
        # =====================================================================
        
        # 1-1. Dense Semantic Search (FAISS)
        query_vec = self.embedder.encode(query)
        # 1단계 후보군 Pool을 여유 있게 확보하기 위해 stage1_k만큼 검색
        v_scores, v_indices = self.index.search(
            query_vec.astype("float32"), min(self.stage1_k, self.index.ntotal)
        )
        faiss_rank_list = v_indices[0].tolist()

        # 1-2. Sparse Lexical Search (BM25)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        # 렉시컬 점수 상위 인덱스 정렬 및 추출
        bm25_rank_list = np.argsort(bm25_scores)[::-1][:self.stage1_k].tolist()

        # 1-3. RRF (Reciprocal Rank Fusion)를 통한 후보군 통합 및 상위 압축
        rrf_map = {}
        for rank, idx in enumerate(faiss_rank_list):
            rrf_map[idx] = rrf_map.get(idx, 0.0) + (1.0 / (self.rrf_k + rank + 1))
        for rank, idx in enumerate(bm25_rank_list):
            rrf_map[idx] = rrf_map.get(idx, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # 두 검색 결합 기준 상위 집합 생성 (최종 상위 top_n의 3배수 가량 확보하여 2단계 정밀 여과 진행)
        candidate_pool_size = min(top_n * 3, len(rrf_map))
        sorted_candidates = sorted(rrf_map.keys(), key=lambda x: rrf_map[x], reverse=True)
        candidate_indices = sorted_candidates[:candidate_pool_size]

        # =====================================================================
        # STAGE 2: High Precision (압축된 후보군에 대한 정밀 재점수화 및 정렬)
        # =====================================================================
        
        # 2-1. 후보군 내에서 스코어 정규화를 수행하기 위한 BM25 바운더리 산출
        cand_bm25_scores = bm25_scores[candidate_indices]
        max_bm25 = np.max(cand_bm25_scores)
        min_bm25 = np.min(cand_bm25_scores)
        bm25_denom = max_bm25 - min_bm25 + 1e-9

        combined_results = []
        for idx in candidate_indices:
            # Lexical 점수 정규화
            s_lex_raw = bm25_scores[idx]
            s_lex_norm = (s_lex_raw - min_bm25) / bm25_denom if max_bm25 > 0 else 0.0

            # Semantic 점수 매핑 (BGE-M3 특성 반영 코사인 유사도 원본 점수 매핑)
            if idx in faiss_rank_list:
                match_pos = faiss_rank_list.index(idx)
                s_sem = float(v_scores[0][match_pos])
            else:
                # 1단계 FAISS 반환 순위에 없던 문서가 BM25에 의해 후보군에 진입한 경우 최소 임계 보정
                s_sem = 0.0

            # 최신성 감쇄 인자 계산
            f_time = self._get_time_decay(self.documents[idx]["created_at"])

            # 렉시컬 게이트 제약 조건 적용
            p_gate = 1.0 if s_lex_raw > 0 else 0.5

            # 가중치 결합 공식 적용
            final_score = (self.alpha * s_lex_norm + self.beta * s_sem) * f_time * p_gate

            # 디버깅 정보 포함 로그 적재
            res_entry = self.documents[idx].copy()
            res_entry.update({
                "score_final": round(float(final_score), 4),
                "score_lex": round(float(s_lex_norm), 4),
                "score_sem": round(float(s_sem), 4),
                "factor_time": round(float(f_time), 4),
                "factor_gate": p_gate,
                "rrf_score": round(rrf_map[idx], 4)
            })
            combined_results.append(res_entry)

        # 최종 가중치 스코어 기준 내림차순 정렬
        combined_results.sort(key=lambda x: x["score_final"], reverse=True)
        return combined_results[:top_n]