import logging

import numpy as np

logger = logging.getLogger("ai_core.search_layers.candidate_pool")


class CandidatePoolExtractor:
    """LAYER 1: 렉시컬/시맨틱 도메인 분리 후보군 압축 및 고유사도 자산 강제 구출"""
    def __init__(self, stage1_k: int = 40, fast_track_threshold: float = 0.92):
        self.stage1_k = stage1_k
        self.fast_track_threshold = fast_track_threshold

    def extract(
        self, 
        query: str, 
        query_vector: np.ndarray, 
        bm25_instance,          # rank_bm25 인스턴스 정상 수용
        faiss_index,            # [FIXED]: 명세 일치화 및 외부 주입 핸들러 정상 바인딩
        documents_list: list, 
        doc_id_to_idx_map: dict, 
        stage1_k: int
    ) -> tuple[list, dict, dict]:
        query_vector_32 = query_vector.astype("float32").flatten()
        
        # [FIXED]: 네임스페이스 오염 교정 완료
        total_vectors = len(documents_list)
        v_scores = np.zeros(total_vectors)
        
        # 렉시컬 점수 연산 어레이 도출
        tokenized_query = query.split()
        bm25_scores_array = bm25_instance.get_scores(tokenized_query)
        
        v_scores_dict = {}
        bm25_scores_dict = {}

        norm_q = np.linalg.norm(query_vector_32) + 1e-9
        for i, doc in enumerate(documents_list):
            doc_id = int(doc["id"])
            try:
                vec = faiss_index.reconstruct(doc_id)
                norm_v = np.linalg.norm(vec) + 1e-9
                
                score = float(np.dot(query_vector_32, vec) / (norm_q * norm_v))
                v_scores[i] = np.clip(score, -1.0, 1.0)
            except Exception as e:
                logger.error(f"Error in reconstruct or dot: {e}", exc_info=True)
                v_scores[i] = 0.0
                
            v_scores_dict[i] = v_scores[i]
            bm25_scores_dict[i] = float(bm25_scores_array[i])

        v_indices = np.argsort(v_scores)[::-1]
        candidate_set = set()
        
        target_k = stage1_k or self.stage1_k
        
        # 1. BM25 형태소 렉시컬 스코어 상위 압축
        lex_top_k = np.argsort(bm25_scores_array)[::-1][:target_k]
        for idx in lex_top_k:
            if bm25_scores_array[idx] > 0:
                doc_id = int(documents_list[idx]["id"])
                candidate_set.add(doc_id)

        # 2. FAISS 시맨틱 임베딩 공간 상위 압축
        sem_top_k = v_indices[:target_k]
        for idx in sem_top_k:
            if v_scores[idx] > 0:
                doc_id = int(documents_list[idx]["id"])
                candidate_set.add(doc_id)

        # 3. Fast-Track 최우수 자산 강제 구출 분기
        for i, doc in enumerate(documents_list):
            if v_scores[i] >= self.fast_track_threshold:
                candidate_set.add(int(doc["id"]))

        candidate_ids = list(candidate_set)
        return candidate_ids, bm25_scores_dict, v_scores_dict