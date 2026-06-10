import logging

logger = logging.getLogger("ai_core.search_layers.rank_filter")

class AdvancedRankFilter:
    """LAYER 3: 하이브리드 수렴 완료 풀 대상 최종 Top-K 절단 및 동적 임계치 컷오프 가드"""
    def __init__(self, min_absolute_score: float = -100.0):
        self.min_absolute_score = min_absolute_score

    def filter_top_k(self, ranked_pool: list, top_n: int) -> list:
        if not ranked_pool:
            return []
            
        # [FIXED]: 컷오프 및 정렬 타깃 속성 키 명세를 'score'로 일치화 완료
        valid_pool = [item for item in ranked_pool if item.get("score", -1.0) >= self.min_absolute_score]
        
        sorted_pool = sorted(valid_pool, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_pool[:top_n]