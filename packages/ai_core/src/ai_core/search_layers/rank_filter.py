import logging
import numpy as np



logger = logging.getLogger("ai_core.search_layers.rank_filter")


class RankPenaltyFilter:
    """LAYER 3: 등수 분산 패널티 연산 및 상호 데이터 정합성 오염 검문 제어"""
    def __init__(self, zero_hits_threshold: float = 0.02):
        self.zero_hits_threshold = zero_hits_threshold

    def _calculate_rank_penalty(self, lex_rank: int, sem_rank: int) -> float:
        return 1.0 / (1.0 + np.log1p((lex_rank - 1) * (sem_rank - 1)))

    def verify_and_filter(self, ranked_pool, bm25_scores, v_scores, doc_count, index_total) -> bool:
        if not ranked_pool:
            return False
            
        top_1_doc = ranked_pool[0]
        top_1_global_idx = top_1_doc["id"] - 1

        v_indices = np.argsort(v_scores)[::-1]

        try:
            global_lex_rank = int(np.where(np.argsort(bm25_scores)[::-1] == top_1_global_idx)[0][0]) + 1
        except IndexError:
            global_lex_rank = doc_count

        try:
            global_sem_rank = int(np.where(v_indices == top_1_global_idx)[0][0]) + 1
        except IndexError:
            global_sem_rank = doc_count

        rank_penalty = self._calculate_rank_penalty(global_lex_rank, global_sem_rank)
        verified_cutoff_score = top_1_doc['score_final'] * rank_penalty

        logger.debug(
            f"[LAYER 3] 최외각 검문 메트릭 -> ID #{top_1_doc['id']} | "
            f"LexRank={global_lex_rank}등, SemRank={global_sem_rank}등 | "
            f"패널티 인자={rank_penalty:.4f} | 보정전 스코어={top_1_doc['score_final']:.4f} -> 최종 보정본={verified_cutoff_score:.4f}"
        )

        is_corrupted_environment = (doc_count != index_total)

        if verified_cutoff_score < self.zero_hits_threshold:
            if is_corrupted_environment:
                logger.warning(f"데이터 복원 정합성 불일치(DB:{doc_count}!=INDEX:{index_total}) 우회 안전 가드 발동 통과")
                return True
            else:
                logger.warning(f"[ZERO-HITS FILTRATION ACTIVATE] 임계치 미달실격 처리 (최종점수: {verified_cutoff_score:.4f} < {self.zero_hits_threshold})")
                return False
                
        return True