import re
import sqlite3
from datetime import datetime

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from core.embedder import BGEEmbedder


class AttentionHybridSearcher:
    def __init__(
        self,
        db_path="db/ikg_metadata.db",
        index_path="db/ikg_vector.index",
        model_path="./model/bge-m3-onnx-int8",
        decay_lambda=0.001,  # 최신성 감쇄 상수
        temperature=1.5      # 어텐션 소프트맥스 민감도 조절 인자
    ):
        # 1. 자원 로드
        self.conn = sqlite3.connect(db_path)
        self.index = faiss.read_index(index_path)
        self.embedder = BGEEmbedder(
            model_path=model_path, file_name="model_quantized.onnx"
        )

        # 2. 하이퍼파라미터 설정
        self.decay_lambda = decay_lambda
        self.temperature = temperature

        # 3. 데이터 로드 및 무결성 체크 (SQLite Row Factory 적용으로 인덱스 오류 방지)
        self.documents = self._load_all_documents()
        self._check_integrity()

        # 4. 시맨틱 보존형 전처리 및 BM25 구축
        tokenized_corpus = [
            self._preprocess_tech_text(doc["content"]) for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 5. [Attention 특화] 전역 컨텍스트 베이스라인 벡터 산출
        # 초기화 시점에 전역 벡터들의 평균 중심점을 계산해두어 검색 시 연산 오버헤드 최소화
        self.context_mean_vector = self._compute_context_mean()


    def _load_all_documents(self):
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


    def _compute_context_mean(self):
        """FAISS 인덱스로부터 전역 문서 벡터들을 추출하여 평균 중심점(Context)을 반환"""
        vectors = []
        for i in range(self.index.ntotal):
            # 전역 벡터 복원
            vec = self.index.reconstruct(i)
            vectors.append(vec)
        
        matrix = np.array(vectors)
        # 각 차원별 평균 벡터 반환 [Dimension]
        return np.mean(matrix, axis=0)


    def _compute_dynamic_attention_weights(self, query_vec):
        """
        쿼리 벡터와 전역 컨텍스트 간의 매칭 에너지를 계산하여
        실시간 동적 알파(Lexical) 및 베타(Semantic) 가중치를 소프트맥스로 산출
        """
        # 정규화 벡터 연산 (코사인 유사도 스케일 확보)
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        c_norm = self.context_mean_vector / (np.linalg.norm(self.context_mean_vector) + 1e-9)
        
        # 쿼리와 내 지식베이스 편향 중심축 간의 내적 연산 (Attention Energy)
        attn_energy = float(np.dot(q_norm, c_norm))
        
        # [핵심 트리거] attn_energy가 고득점(1에 가까움)일수록 현재 데이터가 밀집된 
        # 편향 영역에 진입한 위험 쿼리이므로 오탐 방지를 위해 Lexical 가중치(alpha)를 부드럽게 밀어 올림
        # 반대로 외부 낯선 영역일 경우 문맥을 살리기 위해 Semantic 가중치(beta)를 상향
        exp_lex = np.exp(attn_energy * self.temperature)
        exp_sem = np.exp((1.0 - attn_energy) * self.temperature)
        
        alpha = exp_lex / (exp_lex + exp_sem)
        beta = 1.0 - alpha
        
        return alpha, beta, attn_energy


    def search(self, query: str, top_n=5):
        tokenized_query = self._preprocess_tech_text(query)
        if not tokenized_query:
            return []

        # 1. Dense Semantic Search (FAISS 전역 랭킹 스코어 획득)
        query_vec = self.embedder.encode(query)
        v_scores, v_indices = self.index.search(
            query_vec.astype("float32"), self.index.ntotal
        )

        # 2. Sparse Lexical Search (BM25 전역 랭킹 스코어 획득)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # BM25 점수 Min-Max 정규화
        if np.max(bm25_scores) > 0:
            bm25_scores_norm = (bm25_scores - np.min(bm25_scores)) / (
                np.max(bm25_scores) - np.min(bm25_scores) + 1e-9
            )
        else:
            bm25_scores_norm = bm25_scores

        # 3. [Attention 핵심] 런타임 동적 가중치 산출
        alpha, beta, attn_energy = self._compute_dynamic_attention_weights(query_vec[0])

        # 4. 동적 가중치가 주입된 통합 스코어링 루프
        combined_results = []
        for i in range(len(self.documents)):
            s_lex = bm25_scores_norm[i]
            
            # FAISS 스코어 매핑
            s_sem = float(v_scores[0][np.where(v_indices == i)[0][0]]) if i in v_indices else 0.0

            # 최신성 감쇄 인자 계산
            f_time = self._get_time_decay(self.documents[i]["created_at"])

            # 환각 방지용 하드 렉시컬 게이트 제약 조건 유지
            p_gate = 1.0 if s_lex > 0 else 0.5

            # 어텐션 메커니즘이 실시간 튜닝해준 alpha, beta를 통해 최종 점수 선형 결합
            final_score = (alpha * s_lex + beta * s_sem) * f_time * p_gate

            # 디버깅 정보 포함 로그 적재
            res_entry = self.documents[i].copy()
            res_entry.update({
                "score_final": round(float(final_score), 4),
                "score_lex": round(float(s_lex), 4),
                "score_sem": round(float(s_sem), 4),
                "factor_time": round(float(f_time), 4),
                "factor_gate": p_gate,
                "dynamic_alpha": round(alpha, 4), # 로그 대조용 동적 변동 가중치 기록
                "dynamic_beta": round(beta, 4),
                "attn_energy": round(attn_energy, 4)
            })
            combined_results.append(res_entry)

        # 최종 점수 기준 정렬 및 반환
        combined_results.sort(key=lambda x: x["score_final"], reverse=True)
        return combined_results[:top_n]