import logging
import numpy as np



logger = logging.getLogger("ai_core.search_layers.candidate_pool")


class CandidatePoolExtractor:
    """LAYER 1: 렉시컬(BM25) 스코어 및 시맨틱(FAISS) 공간 매트릭스를 융합하여 상위 후보군 압축"""
    def __init__(self, stage1_k: int = 40, fast_track_threshold: float = 0.92):
        self.stage1_k = stage1_k
        self.fast_track_threshold = fast_track_threshold

    def extract(
        self, 
        query: str, 
        query_vector: np.ndarray, 
        bm25_instance,                   # hybrid_search의 self.bm25 인스턴스 수용
        faiss_index,                     # hybrid_search의 self.index 핸들러 수용
        documents_list: list,            # hybrid_search의 self.documents 스냅샷 수용
        doc_id_to_idx_map: dict,         # hybrid_search의 self.doc_id_to_idx 매퍼 수용
        dynamic_stage1_k: int            # hybrid_search의 self.stage1_k 수용
    ) -> tuple[list, dict, dict]:
        """
        [FIXED]: 하이브리드 게이트웨이 호출부와 7개 파라미터 시그니처를 100% 동기화하고
        IndexIDMap 체제에 매칭되도록 실제 문서 ID 기반 벡터 복원(reconstruct) 구조로 전격 교정했습니다.
        """
        query_vector_32 = query_vector.astype("float32")
        total_vectors = faiss_index.ntotal
        
        # 렉시컬 스코어 원천 계산 배열 획득
        # rank_bm25의 get_scores 메서드를 호출하여 질의어 토큰 행렬 점수 도출
        tokenized_query = query.split() # 단순 형태소 분할 분기 가동
        bm25_scores_array = bm25_instance.get_scores(tokenized_query)

        v_scores = np.zeros(total_vectors)
        v_scores_dict = {}
        bm25_scores_dict = {}

        norm_q = np.linalg.norm(query_vector_32) + 1e-9
        
        for i in range(total_vectors):
            try:
                # [CRITICAL FIXED]: 루프 인덱스 i가 아닌 실제 도큐먼트의 영속 정수 고유 ID를 역추적하여 
                # IDMap 공간으로부터 다차원 기하학 벡터를 정상 복원합니다.
                doc_id = int(documents_list[i]["id"])
                vec = faiss_index.reconstruct(doc_id)
                
                norm_v = np.linalg.norm(vec) + 1e-9
                score = float(np.dot(query_vector_32, vec) / (norm_q * norm_v))
                v_scores[i] = np.clip(score, -1.0, 1.0)
            except Exception as e:
                logger.debug(f"[CANDIDATE POOL CHECK] 벡터 인덱스 #{i} 복원 실패 바이패스: {e}")
                v_scores[i] = 0.0

            # 인덱스 순서 기준 점수 매핑 영속화
            v_scores_dict[i] = v_scores[i]
            bm25_scores_dict[i] = float(bm25_scores_array[i])

        v_indices = np.argsort(v_scores)[::-1]
        candidate_set = set()
        
        target_k = dynamic_stage1_k or self.stage1_k

        # 1. BM25 형태소 렉시컬 스코어 상위 압축 가동
        lex_top_k = np.argsort(bm25_scores_array)[::-1][:target_k]
        for idx in lex_top_k:
            if bm25_scores_array[idx] > 0:
                candidate_set.add(int(idx))

        # 2. FAISS 시맨틱 임베딩 공간 상위 압축 가동
        sem_top_k = v_indices[:target_k]
        for idx in sem_top_k:
            candidate_set.add(int(idx))

        # 3. Fast-Track 초우수 지식 자산 고속 강제 구출 분기
        for i in range(total_vectors):
            if v_scores[i] >= self.fast_track_threshold:
                candidate_set.add(i)

        candidate_ids = list(candidate_set)
        return candidate_ids, bm25_scores_dict, v_scores_dict