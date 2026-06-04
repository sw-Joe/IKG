import logging
import sqlite3
from datetime import datetime

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH, IKG_MODEL_FILE, IKG_MODEL_PATH
from ai_core.core.embedder import BGEEmbedder

# from ai_core.search_layers.candidate_pool import CandidatePoolExtractor
# from ai_core.search_layers.context_attention import ContextAttentionRouter
# from ai_core.search_layers.rank_filter import RankPenaltyFilter
from ai_core.search_layers import (
    CandidatePoolExtractor,
    ContextAttentionRouter,
    RankPenaltyFilter,
)



# 모듈 단위 네임스페이스 로거 바인딩
logger = logging.getLogger("ai_core.hybrid_search")


class HybridSearcher:
    """CQRS Read 전용: 잠금 경합 프리 구조를 가지며 고해상도 3단계 하이브리드 검색 정렬을 전담하는 매니저"""
    def __init__(
        self,
        db_path=None,
        index_path=None,
        model_path=None,
        decay_lambda=0.001,      
        stage1_k=40
    ):
        self.db_path = db_path or IKG_DB_PATH
        self.index_path = index_path or IKG_INDEX_PATH
        self.model_path = model_path or IKG_MODEL_PATH
        self.decay_lambda = decay_lambda
        self.stage1_k = stage1_k

        # 질의어 인코딩 전용 고속 임베더 싱글톤 결합
        self.embedder = BGEEmbedder(model_path=self.model_path, file_name=IKG_MODEL_FILE)
        
        # 외부 결합 분할 계층 인스턴스 주입
        self.pool_extractor = CandidatePoolExtractor()
        self.attention_router = ContextAttentionRouter()
        self.rank_filter = RankPenaltyFilter()
        
        # 인메모리 매핑 스냅샷 컨텍스트 동기화 빌드
        self.refresh_context()
        logger.info("조회 전담 하이브리드 브레인 v3 인프라 동기화 가동 완결")


    def refresh_context(self):
        """SQLite 상의 활성 자산(is_deleted=0) 상태 데이터 스냅샷 최신화 및 BM25 렉시컬 어휘집 인메모리 동기화 재빌드"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # [핵심 방어 가드] Soft-Delete 처리된 가비지 행은 메모리 인입 단계에서 원천 차단
            cursor.execute("SELECT id, url, title, content, created_at FROM bookmarks WHERE is_deleted = 0")
            rows = cursor.fetchall()
            
            self.documents = []
            corpus = []
            self.doc_id_to_idx = {} # SQLite ID -> 인메모리 배열 오프셋 인덱스 매핑 역추적 장치
            
            for idx, row in enumerate(rows):
                doc_dict = {
                    "id": row["id"],
                    "url": row["url"],
                    "title": row["title"],
                    "content": row["content"],
                    "created_at": row["created_at"]
                }
                self.documents.append(doc_dict)
                corpus.append(f"{row['title']} {row['content']}")
                self.doc_id_to_idx[row["id"]] = idx
                
            # 인메모리 단어 분할 렉시컬 인덱스 동적 동기화
            tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
            self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
            
            # 읽기 전용으로 FAISS IDMap 파일 로드
            self.index = faiss.read_index(self.index_path)
        finally:
            conn.close()


    def search(self, query: str, top_n: int = 5) -> list:
        """Read-Only 파이프라인: 읽기 전용 가상환경에서 3단계 하이브리드 연산 관통"""
        if not self.documents or not self.bm25:
            return []

        # 0. 입력 질의어 수 밀리초 내 고속 Dense 임베딩 수행
        query_vector = self.embedder.encode(query)[0].astype("float32")

        # LAYER 1: 인프라 후보군 추출 및 RRF 압축
        # 주의: IndexIDMap 검색 결과 반환되는 ID 리스트는 이제 행 번호가 아닌 실제 SQLite의 'bookmark_id' 고유값입니다.
        candidate_ids, bm25_scores, v_scores = self.pool_extractor.extract(
            query, query_vector, self.bm25, self.index, self.documents, self.doc_id_to_idx, self.stage1_k
        )

        if not candidate_ids:
            return []

        # LAYER 2: 글로벌 어텐션 가중치 매트릭스 동적 분배 (Alpha / Beta 실시간 분배)
        alpha, beta = self.attention_router.calculate_weights(query_vector, self.index)

        ranked_pool = []
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        for doc_id in candidate_ids:
            # 역추적 매퍼를 거쳐 실제 인메모리 메타데이터 로드
            idx = self.doc_id_to_idx[doc_id]
            doc = self.documents[idx]
            
            score_lex = bm25_scores[idx] / max_bm25
            score_sem = v_scores[idx]
            
            # 시계열 감쇠 수식 연산 결합
            time_decay = self._calculate_decay(doc["created_at"])
            score_final = (alpha * score_sem + beta * score_lex) * time_decay
            
            ranked_pool.append({
                "id": doc["id"], "url": doc["url"], "title": doc["title"], "content": doc["content"],
                "score_final": score_final, "score_lex_raw": bm25_scores[idx], "score_sem_raw": v_scores[idx]
            })

        # 최종 가중 선형 점수 기준 소팅
        ranked_pool.sort(key=lambda x: x["score_final"], reverse=True)

        # LAYER 3: 최외각 패널티 검문 및 Zero-Hits 판정 무결성 검증
        is_passed = self.rank_filter.verify_and_filter(
            ranked_pool, bm25_scores, v_scores, len(self.documents), self.index.ntotal
        )

        return ranked_pool[:top_n] if is_passed else []


    def _calculate_decay(self, created_at_str: str) -> float:
        try:
            dt = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
            delta_days = (datetime.now() - dt).days
            return float(np.exp(-self.decay_lambda * max(0, delta_days)))
        except:
            return 1.0
