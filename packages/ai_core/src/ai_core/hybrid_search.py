import os
import re
import sqlite3
import logging
from datetime import datetime

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH, IKG_MODEL_PATH, IKG_MODEL_FILE
from ai_core.core import BGEEmbedder

# 모듈 단위 네임스페이스 로거 바인딩
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

        logger.info("하이브리드 코어 v3 브레인 인스턴스 웜업 개시")
        logger.debug(f"데이터 설정 가드 - DB: {self.db_path} | INDEX: {self.index_path}")

        self.embedder = embedder or BGEEmbedder(model_path=self.model_path, file_name=IKG_MODEL_FILE)
        
        self.decay_lambda = decay_lambda
        self.temperature = temperature
        self.stage1_k = stage1_k
        self.fast_track_threshold = fast_track_threshold
        self.zero_hits_threshold = zero_hits_threshold

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
                logger.warning("동기화 대상 DB 레코드가 전무하여 인덱스 컴파일을 대기 상태로 전환합니다.")
                return

            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"[ENGINE SYNC] FAISS 벡터 스냅샷 로드 완결 (총 물리 벡터 수: {self.index.ntotal}개)")
            else:
                logger.warning(f"인덱스 파일({self.index_path})이 존재하지 않아 빈 FlatIP 공간을 할당합니다.")
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

    def _calculate_rank_penalty(self, lex_rank, sem_rank):
        return 1.0 / (1.0 + np.log1p((lex_rank - 1) * (sem_rank - 1)))

    def search(self, query, top_n=5):
        logger.info(f"하이브리드 랭킹 CORE 인퍼런스 구동 -> 질의어: '{query}' | Target Top: {top_n}")

        if not self.documents or self.index is None or self.bm25 is None:
            logger.error("메모리 상주 인덱스 매트릭스 결손 상태로 추론 유실 공백 반환")
            return []

        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        query_vector = self.embedder.encode(query)[0].astype("float32")
        total_vectors = self.index.ntotal
        v_scores = np.zeros(total_vectors)
        
        norm_q = np.linalg.norm(query_vector)
        for i in range(total_vectors):
            try:
                vec = self.index.reconstruct(i)
                norm_v = np.linalg.norm(vec)
                if norm_q > 0 and norm_v > 0:
                    v_scores[i] = float(np.dot(query_vector, vec) / (norm_q * norm_v))
                    v_scores[i] = np.clip(v_scores[i], -1.0, 1.0)
                else:
                    v_scores[i] = 0.0
            except Exception:
                v_scores[i] = 0.0

        v_indices = np.argsort(v_scores)[::-1]
        candidate_set = set()
        
        lex_top_k = np.argsort(bm25_scores)[::-1][:self.stage1_k]
        for idx in lex_top_k:
            if bm25_scores[idx] > 0:
                candidate_set.add(idx)

        sem_top_k = v_indices[:self.stage1_k]
        for idx in sem_top_k:
            candidate_set.add(idx)

        for idx, score in enumerate(v_scores):
            if score >= self.fast_track_threshold and idx not in candidate_set:
                candidate_set.add(idx)
                logger.debug(f"[FAST-TRACK] 문서 ID #{self.documents[idx]['id']} 강제 구출 (유사도: {score:.4f})")

        logger.info(f"[LAYER 1] 후보 풀 압축: 전역 {len(self.documents)}건 -> 격리 풀 {len(candidate_set)}건")

        if not candidate_set:
            logger.warning("1단계 결합 풀 형성 단계 매칭 정보 전무")
            return []

        all_reconstructed_vecs = np.array([self.index.reconstruct(i) for i in range(total_vectors)])
        global_context_center = np.mean(all_reconstructed_vecs, axis=0)
        norm_c = np.linalg.norm(global_context_center)
        
        if norm_q > 0 and norm_c > 0:
            attn_energy = float(np.dot(query_vector, global_context_center) / (norm_q * norm_c))
            attn_energy = np.clip(attn_energy, -1.0, 1.0)
        else:
            attn_energy = 0.0
        
        alpha = 1.0 / (1.0 + np.exp(-self.temperature * attn_energy))
        beta = 1.0 - alpha
        logger.debug(f"[LAYER 2] 어텐션 분석 스케일: 에너지={attn_energy:.4f} | Alpha(시맨틱)={alpha:.2f}, Beta(렉시컬)={beta:.2f}")

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

        top_1_doc = ranked_pool[0]
        top_1_global_idx = top_1_doc["id"] - 1

        try:
            global_lex_rank = int(np.where(np.argsort(bm25_scores)[::-1] == top_1_global_idx)[0][0]) + 1
        except IndexError:
            global_lex_rank = len(self.documents)

        try:
            global_sem_rank = int(np.where(v_indices == top_1_global_idx)[0][0]) + 1
        except IndexError:
            global_sem_rank = len(self.documents)

        rank_penalty = self._calculate_rank_penalty(global_lex_rank, global_sem_rank)
        verified_cutoff_score = top_1_doc['score_final'] * rank_penalty

        logger.debug(f"[LAYER 3] 검문소 메트릭 -> ID #{top_1_doc['id']} | Penalty={rank_penalty:.4f} | 보정전={top_1_doc['score_final']:.4f} -> 보정후={verified_cutoff_score:.4f}")

        is_corrupted_environment = (len(self.documents) != self.index.ntotal)

        if verified_cutoff_score < self.zero_hits_threshold:
            if is_corrupted_environment:
                logger.warning(f"데이터 복원 불일치(DB:{len(self.documents)}!=INDEX:{self.index.ntotal}) 우회 안전 가드 가동 통과")
            else:
                logger.warning(f"[ZERO-HITS FILTRATION ACTIVATE] 최종 점수({verified_cutoff_score:.4f})가 하한 임계치 미달로 후보 풀 전체 파괴")
                return []

        logger.info(f"하이브리드 검문 수식 무결성 통과. 상위 {min(top_n, len(ranked_pool))}건 스트리밍 매핑")
        
        final_out = []
        for item in ranked_pool[:top_n]:
            item["score"] = item["score_final"]
            final_out.append(item)
        return final_out