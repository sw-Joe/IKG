import logging
import numpy as np



logger = logging.getLogger("ai_core.search_layers.context_attention")


class ContextAttentionRouter:
    """LAYER 2: 글로벌 문맥 대수 공간의 중심축 연산 및 시맨틱/렉시컬 동적 가중치 분배"""
    def __init__(self, temperature: float = 1.5):
        self.temperature = temperature

    def calculate_weights(self, query_vector, index) -> tuple[float, float]:
        query_vector_32 = query_vector.astype("float32")
        total_vectors = index.ntotal
        
        all_reconstructed_vecs = np.array([index.reconstruct(i) for i in range(total_vectors)])
        global_context_center = np.mean(all_reconstructed_vecs, axis=0)
        
        norm_q = np.linalg.norm(query_vector_32)
        norm_c = np.linalg.norm(global_context_center)
        
        if norm_q > 0 and norm_c > 0:
            attn_energy = float(np.dot(query_vector_32, global_context_center) / (norm_q * norm_c))
            attn_energy = np.clip(attn_energy, -1.0, 1.0)
        else:
            attn_energy = 0.0
            
        alpha = 1.0 / (1.0 + np.exp(-self.temperature * attn_energy))
        beta = 1.0 - alpha
        
        logger.debug(f"[LAYER 2] 글로벌 어텐션 밀도 분석: 에너지={attn_energy:.4f} | Alpha(시맨틱)={alpha:.2f}, Beta(렉시컬)={beta:.2f}")
        return alpha, beta