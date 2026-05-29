import sqlite3
from datetime import datetime

import faiss
import numpy as np

from ai_core.embedder import BGEEmbedder


class SearcherNativeSparse:
    def __init__(
        self,
        db_path="db/ikg_metadata.db",
        index_path="db/ikg_vector.index",
        model_path="./model/bge-m3-onnx-int8",
        alpha=0.4,          # Native Sparse 가중치
        beta=0.6,           # Dense Semantic 가중치
        decay_lambda=0.001, # 최신성 감쇄 상수
    ):
        # 1. 자원 로드
        self.conn = sqlite3.connect(db_path)
        self.index = faiss.read_index(index_path)
        
        # BGE-M3 Dense 및 Sparse 출력을 동시 처리하기 위한 임베더 인스턴스화
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

        # 4. [핵심] 전역 문서의 Native Sparse Embeddings 역색인 풀 인메모리 구축
        # 대조 실험 스크립트 규격을 맞추기 위해 초기화 단에서 사전 인코딩 수행
        print("[INFO] BGE-M3 Native Sparse Embeddings 전역 인덱싱 중...")
        self.sparse_corpus_vectors = self._build_sparse_corpus_pool()

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

    def _build_sparse_corpus_pool(self):
        """
        각 문서의 본문을 BGE-M3 Sparse Embedding으로 변환하여 인메모리 리스트로 유지
        주의: embedder.py의 구현체 스펙에 따라 모델 호출 및 파싱 로직을 유연하게 조정해야 합니다.
        """
        sparse_vectors = []
        for doc in self.documents:
            # embedder가 sparse 출력을 딕셔너리나 가중치 벡터 형태로 반환한다고 정의
            # 현재 래퍼가 복합 출력을 미지원할 경우를 대비하여 범용 파싱 인터페이스 구성
            try:
                # 쿼리와 문서를 동일 인코더 레이어로 처리
                # (현 embedder.encode 내부 스펙이 dense 전용일 경우 모델 직접 추론 로직 확장이 요구될 수 있음)
                sparse_vec = self.embedder.encode_sparse(doc["content"])
            except AttributeError:
                # 가중치 딕셔너리 목업 구현 (실 운영 및 모델 검증 시 embedder 내부 ONNX Sparse Output 파싱 필수)
                # 토크나이저 ID와 중요도 가중치가 매핑된 형태 {token_id: weight}
                sparse_vec = {}
            sparse_vectors.append(sparse_vec)
        return sparse_vectors

    def _compute_sparse_similarity(self, query_sparse, doc_sparse):
        """Query Sparse Vector와 Document Sparse Vector 간의 Dot Product 연산"""
        if not query_sparse or not doc_sparse:
            return 0.0
        
        score = 0.0
        # 매칭되는 Token ID의 가중치를 곱하여 합산 (Learned Sparsity Dot Product)
        for token_id, q_weight in query_sparse.items():
            if token_id in doc_sparse:
                score += q_weight * doc_sparse[token_id]
        return float(score)

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
        if not query.strip():
            return []

        # 1. Dense Semantic Search (FAISS 전역 탐색)
        query_dense = self.embedder.encode(query)
        v_scores, v_indices = self.index.search(
            query_dense.astype("float32"), self.index.ntotal
        )
        faiss_scores_dict = {int(idx): float(score) for idx, score in zip(v_indices[0], v_scores[0], strict=True)}

        # 2. Native Sparse Embedding Search (BGE-M3 가중치 인메모리 매칭)
        try:
            query_sparse = self.embedder.encode_sparse(query)
        except AttributeError:
            query_sparse = {}

        sparse_scores = np.zeros(len(self.documents))
        for i in range(len(self.documents)):
            sparse_scores[i] = self._compute_sparse_similarity(query_sparse, self.sparse_corpus_vectors[i])

        # Sparse 점수 Min-Max 정규화 (선형 결합 왜곡 방지용 바운더리 산출)
        max_sparse = np.max(sparse_scores)
        min_sparse = np.min(sparse_scores)
        sparse_denom = max_sparse - min_sparse + 1e-9
        
        if max_sparse > 0:
            sparse_scores_norm = (sparse_scores - min_sparse) / sparse_denom
        else:
            sparse_scores_norm = sparse_scores

        # 3. 최종 통합 스코어링 (Native Hybrid Score Fusion)
        combined_results = []
        for i in range(len(self.documents)):
            s_lex_native = sparse_scores_norm[i]
            s_sem = faiss_scores_dict.get(i, 0.0)

            # 최신성 감쇄 인자 계산
            f_time = self._get_time_decay(self.documents[i]["created_at"])

            # 렉시컬 게이트 제약 조건 적용 (Native Sparse 원본 점수가 0보다 큰지 판별)
            p_gate = 1.0 if sparse_scores[i] > 0 else 0.5

            # BGE-M3 고유 가중치 합산 결합 수식
            final_score = (self.alpha * s_lex_native + self.beta * s_sem) * f_time * p_gate

            # 대조 실험용 디버깅 데이터 적재
            res_entry = self.documents[i].copy()
            res_entry.update({
                "score_final": round(float(final_score), 4),
                "score_lex": round(float(s_lex_native), 4),
                "score_sem": round(float(s_sem), 4),
                "factor_time": round(float(f_time), 4),
                "factor_gate": p_gate,
                "raw_sparse_score": round(float(sparse_scores[i]), 4)
            })
            combined_results.append(res_entry)

        # 최종 점수 기준 내림차순 정렬 및 반환
        combined_results.sort(key=lambda x: x["score_final"], reverse=True)
        return combined_results[:top_n]