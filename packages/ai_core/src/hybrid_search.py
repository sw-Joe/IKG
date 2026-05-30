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
        db_path=None,
        index_path=None,
        model_path=None,
        decay_lambda=0.001,      # 최신성 감쇄 상수
        temperature=1.5,          # 어텐션 소프트맥스 민감도 조절 인자
        stage1_k=40,             # 1단계 후보군 추출 크기
        fast_track_threshold=0.92, # 1단계 강제 구출 코사인 유사도 기준선
        zero_hits_threshold=0.10, # 3단계 최종 Zero-Hits 판정 임계값
        embedder=None
    ):
        import os
        # 1. 인프라 자원 로드
        db_path = db_path or os.getenv("IKG_DB_PATH", "db/ikg_metadata.db")
        index_path = index_path or os.getenv("IKG_INDEX_PATH", "db/ikg_vector.index")
        model_path = model_path or os.getenv("IKG_MODEL_PATH", "./model/bge-m3-onnx-int8")
        model_file = os.getenv("IKG_MODEL_FILE", "model_quantized.onnx")

        self.conn = sqlite3.connect(db_path)
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatIP(1024)

        if embedder:
            self.embedder = embedder
        else:
            self.embedder = BGEEmbedder(
                model_path=model_path, file_name=model_file
            )

        # 2. 레이어별 분리형 하이퍼파라미터 정의
        self.decay_lambda = decay_lambda
        self.temperature = temperature
        self.stage1_k = stage1_k
        self.fast_track_threshold = fast_track_threshold
        self.zero_hits_threshold = zero_hits_threshold

        # 3. 데이터셋 로드 및 무결성 정합성 체크
        self.documents = self._load_all_documents()
        
        # 빈 데이터베이스/인덱스인 경우 예외를 발생시키지 않고 빈 전처리
        if len(self.documents) == 0 or self.index.ntotal == 0:
            self.documents = []
            self.bm25 = None
            self.context_mean_vector = np.zeros(1024)
        else:
            if len(self.documents) != self.index.ntotal:
                raise ValueError(f"정합성 오류: DB 문서 수({len(self.documents)})와 FAISS 인덱스 벡터 수({self.index.ntotal})가 불일치합니다.")

            # 4. 전처리 및 렉시컬 인덱스(BM25) 빌드
            tokenized_corpus = [self._preprocess_tech_text(doc["content"]) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)

            # 5. [Core Ranking 전제조건] 전역 컨텍스트 베이스라인 벡터 캐싱 (O(1) 검색 보장)
            self.context_mean_vector = self._compute_context_mean()


    def _load_all_documents(self):
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, url, title, content, created_at FROM bookmarks ORDER BY id ASC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


    def _preprocess_tech_text(self, text):
        text_lower = text.lower()
        return re.findall(r'[a-zA-Z0-9_\-\.]+|[가-힣]+', text_lower)


    def _get_time_decay(self, created_at_str):
        try:
            dt = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
            days_diff = max(0, (datetime.now() - dt).days)
            return np.exp(-self.decay_lambda * days_diff)
        except Exception:
            return 1.0


    def _compute_context_mean(self):
        vectors = [self.index.reconstruct(i) for i in range(self.index.ntotal)]
        return np.mean(np.array(vectors), axis=0)


    def _compute_dynamic_attention_weights(self, query_vec):
        """2단계 코어: 쿼리와 전역 지식 밀도 간의 내적 연산 기반 알파/베타 분배"""
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        c_norm = self.context_mean_vector / (np.linalg.norm(self.context_mean_vector) + 1e-9)
        
        attn_energy = float(np.dot(q_norm, c_norm))
        
        exp_lex = np.exp(attn_energy * self.temperature)
        exp_sem = np.exp((1.0 - attn_energy) * self.temperature)
        
        alpha = exp_lex / (exp_lex + exp_sem)
        beta = 1.0 - alpha
        return alpha, beta, attn_energy


    def _calculate_rank_penalty(self, r_lex, r_sem):
        """3단계 외곽 검문: 수식 교란 방지를 위해 오직 1위 문서 검증용으로만 격리 호출"""
        return 1.0 / (np.log(r_lex + 1) + np.log(r_sem + 1) + 1e-9)


    def search(self, query: str, top_n=5):
        if not self.documents or self.bm25 is None or self.index.ntotal == 0:
            return []
        tokenized_query = self._preprocess_tech_text(query)
        if not tokenized_query:
            return []

        # ==========================================
        # LAYER 1: 인프라 및 후보군 압축 (Stage 1)
        # ==========================================
        query_vec = self.embedder.encode(query)
        
        # 1-1. 각 엔진별 상위 K개 고속 고recall 검색
        v_scores, v_indices = self.index.search(query_vec.astype("float32"), self.stage1_k)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # BM25 상위 인덱스 파싱
        lex_top_indices = np.argsort(bm25_scores)[::-1][:self.stage1_k]

        # 1-2. RRF 순위 기반 풀 통합 변환
        rrf_map = {}
        for rank, idx in enumerate(v_indices[0]):
            rrf_map[idx] = rrf_map.get(idx, 0.0) + (1.0 / (60 + rank + 1))
        for rank, idx in enumerate(lex_top_indices):
            rrf_map[idx] = rrf_map.get(idx, 0.0) + (1.0 / (60 + rank + 1))

        # 1-3. Fast-Track 가드레일: 코사인 유사도가 임계치 이상인 고품질 핵심 문서 강제 구출
        fast_track_count = 0
        for score, idx in zip(v_scores[0], v_indices[0], strict=False):
            if score >= self.fast_track_threshold and idx not in rrf_map:
                rrf_map[idx] = 1.0  # RRF 가중치 최상위 오버라이딩 유도
                fast_track_count += 1

        # RRF 기준 정렬 후 2단계 연산 대상 압축 후보군 리스트업
        compressed_candidates = sorted(rrf_map.keys(), key=lambda x: rrf_map[x], reverse=True)[:self.stage1_k]

        if not compressed_candidates:
            return []

        # ==========================================
        # LAYER 2: 코어 동적 정렬 (Core Ranking)
        # ==========================================
        # BM25 전역 Min-Max 스케일러 범위 확보
        max_bm25 = np.max(bm25_scores)
        min_bm25 = np.min(bm25_scores)
        bm25_denom = (max_bm25 - min_bm25) + 1e-9

        # 실시간 쿼리 맥락 분석 어텐션 가중치 산출
        alpha, beta, attn_energy = self._compute_dynamic_attention_weights(query_vec[0])

        ranked_pool = []
        for idx in compressed_candidates:
            idx = int(idx)
            # 압축 후보군 대상 렉시컬 정규화 스코어링
            s_lex = (bm25_scores[idx] - min_bm25) / bm25_denom if max_bm25 > 0 else 0.0
            
            # 압축 후보군 대상 시맨틱 스코어 매핑
            s_sem = float(v_scores[0][np.where(v_indices == idx)[0][0]]) if idx in v_indices else 0.0

            # 인자 필터 계산
            f_time = self._get_time_decay(self.documents[idx]["created_at"])
            p_gate = 1.0 if s_lex > 0 else 0.5

            # 어텐션 동적 파라미터가 보존된 순수 선형 랭킹 연산 (로그 패널티 결합 전면 배제)
            final_score = (alpha * s_lex + beta * s_sem) * f_time * p_gate

            res_entry = self.documents[idx].copy()
            res_entry.update({
                "score_final": round(float(final_score), 4),
                "score_lex": round(float(s_lex), 4),
                "score_sem": round(float(s_sem), 4),
                "factor_time": round(float(f_time), 4),
                "factor_gate": p_gate,
                "dynamic_alpha": round(alpha, 4),
                "dynamic_beta": round(beta, 4),
                "attn_energy": round(attn_energy, 4)
            })
            ranked_pool.append(res_entry)

        # 2단계 최종 스코어 기반 1차 전체 정렬
        ranked_pool.sort(key=lambda x: x["score_final"], reverse=True)

        # ==========================================
        # LAYER 3: 말단 외곽 검문 및 Zero-Hits 제어
        # ==========================================
        # 전체 정렬이 완료된 상태에서 오직 최종 '1위 문서'의 신뢰성만 엄격하게 검증
        top_1_doc = ranked_pool[0]
        top_1_global_idx = top_1_doc["id"] - 1 # SQLite ID 직렬 인덱스 역보정

        # 전체 풀 기준 독립 엔진 순위(등수) 측정 (1-based index)
        global_lex_rank = int(np.where(np.argsort(bm25_scores)[::-1] == top_1_global_idx)[0][0]) + 1
        global_sem_rank = int(np.where(v_indices[0] == top_1_global_idx)[0][0]) + 1 if top_1_global_idx in v_indices else self.index.ntotal

        # 랭킹 정렬 계산식과 완전 격리된 별도의 검문 인자 계산
        rank_penalty = self._calculate_rank_penalty(global_lex_rank, global_sem_rank)
        verified_cut_score = top_1_doc["score_final"] * rank_penalty

        # [Zero-Hits 최종 판단 검문소]
        # 양쪽 엔진에서 순위 소외가 심각해 verified_cut_score가 벼랑 끝 임계치 미만으로 수렴 시 풀 전체 파괴
        if verified_cut_score < self.zero_hits_threshold:
            return []

        # 최종 검문을 통과한 풀에 랭크 패널티 메트릭 기록 후 최종 반환
        for doc in ranked_pool:
            doc["factor_rank_penalty"] = round(float(rank_penalty), 4)

        return ranked_pool[:top_n]
