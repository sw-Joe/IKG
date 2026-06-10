import logging
import sqlite3

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH, IKG_MODEL_PATH, IKG_MODEL_FILE
from ai_core.core.embedder import BGEEmbedder
from ai_core.search_layers.candidate_pool import CandidatePoolExtractor
from ai_core.search_layers.context_attention import ContextAttentionRouter
from ai_core.search_layers.rank_filter import AdvancedRankFilter



logger = logging.getLogger("ai_core.hybrid_search")


class HybridSearcher:
    """인메모리 CQRS 고속 하이브리드 검색 오케스트레이션 코어 엔진"""
    def __init__(self):
        logger.info("[HYBRID CORE] 하이브리드 지식 검색 컨텍스트 웜업 가동...")
        self.embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
        
        self.layer1_pool = CandidatePoolExtractor()
        self.layer2_attention = ContextAttentionRouter()
        self.layer3_filter = AdvancedRankFilter()
        
        self.refresh_context()

    def refresh_context(self):
        """SQLite 실존 정상 레코드 스냅샷 인메모리 동기화 미러링"""
        conn = sqlite3.connect(IKG_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id, title, content, url FROM bookmarks WHERE is_deleted = 0")
            rows = cursor.fetchall()
            
            self.documents = []
            self.doc_id_to_idx = {}
            
            for idx, row in enumerate(rows):
                doc_dict = {
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"],
                    "url": row["url"]
                }
                self.documents.append(doc_dict)
                self.doc_id_to_idx[row["id"]] = idx
                
            total_docs = len(self.documents)
            logger.info(f" -> [DB RE-INDEX] 인메모리 유효 지식 자산 미러링 완수: {total_docs}건")
            
            if total_docs > 0:
                tokenized_corpus = [doc["content"].split() for doc in self.documents]
                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                self.bm25 = None
        finally:
            conn.close()

    def search(self, query: str, top_n: int = 5, alpha: float = 0.3, stage1_k: int = 40) -> list:
        if not self.documents or self.bm25 is None:
            return []

        # 1. BGE-M3 ONNX 고속 단건 추론 실행
        query_vector = self.embedder.encode_single(query)
        faiss_index = faiss.read_index(IKG_INDEX_PATH)

        # 2. LAYER 1: 후보군 고속 추출 프로토콜 가동 (real_doc_id 체인 반환)
        candidate_ids, bm25_scores, v_scores = self.layer1_pool.extract(
            query=query,
            query_vector=query_vector,
            bm25_instance=self.bm25,
            faiss_index=faiss_index,
            documents_list=self.documents,
            doc_id_to_idx_map=self.doc_id_to_idx,
            stage1_k=stage1_k
        )

        if not candidate_ids:
            return []

        ranked_pool = []
        for doc_id in candidate_ids:
            if doc_id not in self.doc_id_to_idx:
                continue
            idx = self.doc_id_to_idx[doc_id]
            doc = self.documents[idx]

            # 가중 선형 결합 수식 계산 ($Score = \alpha \cdot Lexical + (1-\alpha) \cdot Semantic$)
            score_final = alpha * bm25_scores[idx] + (1.0 - alpha) * v_scores[idx]

            # [FIXED]: FE UI 바인딩 명세와 1:1 정밀 일치화 수렴
            ranked_pool.append({
                "id": doc["id"],
                "url": doc["url"],
                "title": doc["title"],
                "content": doc["content"],
                "score": score_final,
                "score_lex_raw": bm25_scores[idx],
                "score_sem_raw": v_scores[idx]
            })

        # [FIXED]: 정렬 기준 키를 'score_final'에서 'score'로 싱크 정정
        ranked_pool.sort(key=lambda x: x["score"], reverse=True)

        # 3. LAYER 2 & LAYER 3: 후속 컨텍스트 정제 필터 레이어 패스
        final_pool = self.layer3_filter.filter_top_k(ranked_pool, top_n)

        return final_pool