import logging

import numpy as np

logger = logging.getLogger("ai_core.search_layers.context_attention")


class ContextAttentionRouter:
    """LAYER 2: 글로벌 문맥 대수 공간의 중심축 연산 및 시맨틱/렉시컬 동적 가중치 분배"""
    def __init__(self, temperature: float = 1.5):
        self.temperature = temperature

    def calculate_weights(self, query_vector, index, documents_list: list = None) -> tuple[float, float]:
        """
        [FIXED]: 글로벌 컨텍스트 센터 계산 시 IndexIDMap 정합성을 사수하기 위해 
        실제 영속 ID 기반 reconstruct 연산 체인으로 전격 고도화했습니다.
        """
        query_vector_32 = query_vector.astype("float32").flatten()
        total_vectors = len(documents_list)
        
        if total_vectors == 0 or not documents_list:
            return 0.5, 0.5

        # IDMap2 규격에 부합하도록 전체 실존 벡터 매트릭스를 안전 복원
        valid_vectors = []
        for i in range(total_vectors):
            try:
                doc_id = int(documents_list[i]["id"])
                valid_vectors.append(index.reconstruct(doc_id))
            except Exception:
                continue

        if not valid_vectors:
            return 0.5, 0.5

        all_reconstructed_vecs = np.array(valid_vectors)
        global_context_center = np.mean(all_reconstructed_vecs, axis=0)
        
        norm_q = np.linalg.norm(query_vector_32) + 1e-9
        norm_c = np.linalg.norm(global_context_center) + 1e-9
        
        attn_energy = float(np.dot(query_vector_32, global_context_center) / (norm_q * norm_c))
        attn_energy = np.clip(attn_energy, -1.0, 1.0)
            
        alpha = 1.0 / (1.0 + np.exp(-self.temperature * attn_energy))
        beta = 1.0 - alpha
        
        logger.debug(f"[LAYER 2] 글로벌 문맥 어텐션 에너지 정렬 완료 -> Alpha(시맨틱 가중치): {alpha:.4f}")
        return alpha, beta