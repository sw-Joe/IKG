import os
import sqlite3
import logging
from datetime import datetime
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH, IKG_MODEL_PATH, IKG_MODEL_FILE
from ai_core.core import BGEEmbedder
# 리팩토링 분할 모듈 세션 수입
from ai_core.search_layers import CandidatePoolExtractor, ContextAttentionRouter, RankPenaltyFilter

logger = logging.getLogger("ai_core.hybrid_search")

class HybridSearcher:
    def __init__(
        self,
        db_path=None,
        index_path=None,
        model_path=None,
        decay_lambda=0.001,      
        temperature=1.5,          
        stage1_k=40,             
        fast_track_threshold=0.92, 
        zero_hits_threshold=0.02,  
        embedder=None
    ):
        self.db_path = db_path or IKG_DB_PATH
        self.index_path = index_path or IKG_INDEX_PATH
        self.model_path = model_path or IKG_MODEL_PATH
        self.decay_lambda = decay_lambda

        logger.info("하이브리드 코어 v3 브레인 인프라 컴포넌트 조립 완료")

        self.embedder = embedder or BGEEmbedder(model_path=self.model_path, file_name=IKG_MODEL_FILE)
        
        # 분할된 세부 연산 계층 전략 객체 주입 (Dependency Injection)
        self.pool_extractor = CandidatePoolExtractor(stage1_k, fast_track_threshold)
        self.attention_router = ContextAttentionRouter(temperature)
        self.rank_filter = RankPenaltyFilter(zero_hits_threshold)

        self.documents = []
        self.index = None
        self.bm25 = None

        self.reload_indices()

    def reload_indices(self):
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT id, url, title, content, created_at FROM bookmarks ORDER BY id ASC")
                rows = cursor.fetchall()
                self.documents = [dict(row) for row in rows]

            doc_count = len(self.documents)
            logger.info(f"[ENGINE SYNC] 영속 데이터베이스 레코드 파싱 완료: 총 {doc_count}건 식별")
            
            if doc_count == 0:
                return

            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"[ENGINE SYNC] FAISS 벡터 스냅샷 로드 완결 (총 물리 벡터 수: {self.index.ntotal}개)")
            else:
                self.index = faiss.IndexFlatIP(1024)

            corpus = [f"{doc['title']} {doc['content']}".lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(corpus)

        except Exception:
            logger.exception("런타임 메모리 정합성 수렴 실패 (크리티컬)")

    def _calculate_decay(self, created_at_str):
        try:
            doc_date = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
            days_diff = (datetime.now() - doc_date).days
            return np.exp(-self.decay_lambda * days_diff)
        except Exception:
            return 1.0

    def search(self, query: str, top_n: int = 5) -> list[dict]:
        logger.info(f"실시간 융합 추론 파이프라인 가동 개시 -> 질의어: '{query}'")

        if not self.documents or self.index is None or self.bm25 is None:
            logger.error("메모리 상주 인덱스 매트릭스 결손 상태로 추론 즉시 중단")
            return []

        # 1. 렉시컬 토큰 점수 수집
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # 2. 온엑스 임베딩 인코딩 수행
        query_vector = self.embedder.encode(query)[0]

        # =========================================================================
        # [REFACTOR COUPLING] 분할 계층 레이어 순차 파이프라이닝 전개
        # =========================================================================
        # LAYER 1: 후보군 추출 분리
        candidate_set, v_scores = self.pool_extractor.extract(bm25_scores, self.index, query_vector, self.documents)
        if not candidate_set:
            return []

        # LAYER 2: 글로벌 어텐션 가중치 매트릭스 동적 분배
        alpha, beta = self.attention_router.calculate_weights(query_vector, self.index)

        # 복합 선형 결합 정렬 가동
        ranked_pool = []
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        for idx in candidate_set:
            doc = self.documents[idx]
            score_lex = bm25_scores[idx] / max_bm25
            score_sem = v_scores[idx]
            time_decay = self._calculate_decay(doc["created_at"])
            score_final = (alpha * score_sem + beta * score_lex) * time_decay
            
            ranked_pool.append({
                "id": doc["id"], "url": doc["url"], "title": doc["title"], "content": doc["content"],
                "score_final": score_final, "score_lex_raw": bm25_scores[idx], "score_sem_raw": v_scores[idx],
                "time_decay_factor": time_decay
            })

        ranked_pool.sort(key=lambda x: x["score_final"], reverse=True)

        # LAYER 3: 외곽 패널티 검문 및 Zero-Hits 판정
        is_passed = self.rank_filter.verify_and_filter(
            ranked_pool, bm25_scores, v_scores, len(self.documents), self.index.ntotal
        )
        
        if not is_passed:
            return []

        logger.info(f"하이브리드 파이프라인 추론 성공 마감. 결과 상위 {min(top_n, len(ranked_pool))}건 반환")
        
        final_out = []
        for item in ranked_pool[:top_n]:
            item["score"] = item["score_final"]
            final_out.append(item)
        return final_out