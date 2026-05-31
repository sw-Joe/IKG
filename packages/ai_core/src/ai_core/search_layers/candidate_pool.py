import logging
import numpy as np



logger = logging.getLogger("ai_core.search_layers.candidate_pool")


class CandidatePoolExtractor:
    """LAYER 1: 렉시컬/시맨틱 도메인 분리 후보군 압축 및 고유사도 자산 강제 구출"""
    def __init__(self, stage1_k: int = 40, fast_track_threshold: float = 0.92):
        self.stage1_k = stage1_k
        self.fast_track_threshold = fast_track_threshold

    def extract(self, bm25, index, query_vector, documents) -> tuple[set[int], np.ndarray]:
        query_vector_32 = query_vector.astype("float32")
        total_vectors = index.ntotal
        v_scores = np.zeros(total_vectors)
        
        # L2 단위 벡터 정규화 내적 연산 (수치 안정성 보정 안전핀)
        norm_q = np.linalg.norm(query_vector_32)
        for i in range(total_vectors):
            try:
                vec = index.reconstruct(i)
                norm_v = np.linalg.norm(vec)
                if norm_q > 0 and norm_v > 0:
                    v_scores[i] = float(np.dot(query_vector_32, vec) / (norm_q * norm_v))
                    v_scores[i] = np.clip(v_scores[i], -1.0, 1.0)
                else:
                    v_scores[i] = 0.0
            except Exception:
                v_scores[i] = 0.0

        v_indices = np.argsort(v_scores)[::-1]
        candidate_set = set()
        
        # 1. BM25 형태소 스코어 상위 압축
        lex_top_k = np.argsort(bm25)[::-1][:self.stage1_k]
        for idx in lex_top_k:
            if bm25[idx] > 0:
                candidate_set.add(idx)

        # 2. FAISS 시맨틱 임베딩 상위 압축
        sem_top_k = v_indices[:self.stage1_k]
        for idx in sem_top_k:
            candidate_set.add(idx)

        # 3. Fast-Track 초우수 자산 강제 구출 분기
        for idx, score in enumerate(v_scores):
            if score >= self.fast_track_threshold and idx not in candidate_set:
                candidate_set.add(idx)
                logger.debug(f"[FAST-TRACK] 문서 ID #{documents[idx]['id']} 구출 성공 (유사도: {score:.4f})")

        logger.info(f"[LAYER 1] 후보 풀 압축 완결: 전역 {len(documents)}건 -> 격리 풀 {len(candidate_set)}건")
        return candidate_set, v_scores