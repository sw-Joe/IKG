import os
import re
import sqlite3
from datetime import datetime

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH, IKG_MODEL_PATH, IKG_MODEL_FILE
from ai_core.core import BGEEmbedder


class HybridSearcher:
    def __init__(
        self,
        db_path=None,
        index_path=None,
        model_path=None,
        decay_lambda=0.001,      # 최신성 감쇄 상수
        temperature=1.5,          # 어텐션 소프트맥스 민감도 조절 인자
        stage1_k=40,             # 1단계 후보군 추출 크기
        fast_track_threshold=0.92, # 1단계 강제 구출 코사인 유사도 기준선
        zero_hits_threshold=0.02, # 3단계 최종 Zero-Hits 판정 임계값 (0.10 -> 0.02로 현실화 보정)
        embedder=None
    ):
        # 중앙 격리 가드레일 절대 경로 결합 (하드코딩 원천 제거)
        self.db_path = db_path or IKG_DB_PATH
        self.index_path = index_path or IKG_INDEX_PATH
        self.model_path = model_path or IKG_MODEL_PATH

        print(f"\n[AI_CORE] 하이브리드 어텐션 브레인 v3 인스턴스를 웜업합니다.")
        print(f" - 스토리지 바인딩: DB={self.db_path} | INDEX={self.index_path}")

        # 싱글톤 ONNX 추론 임베더 빌드
        self.embedder = embedder or BGEEmbedder(model_path=self.model_path, file_name=IKG_MODEL_FILE)
        
        self.decay_lambda = decay_lambda
        self.temperature = temperature
        self.stage1_k = stage1_k
        self.fast_track_threshold = fast_track_threshold
        self.zero_hits_threshold = zero_hits_threshold

        # 런타임 캐시 데이터 버스
        self.documents = []
        self.index = None
        self.bm25 = None

        # 가동 즉시 실시간 동기화로 데이터 미러링
        self.reload_indices()

    def reload_indices(self):
        """디스크의 최신 바이트 스냅샷을 파싱하여 가상 메모리에 실시간 수렴 (멀티스레드 무결성 보장)"""
        try:
            # [수정 핵심 1]: self.conn 상주 공유를 전면 배제하고 컨텍스트 매니저 기반 스레드 세이프 격리
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT id, url, title, content, created_at FROM bookmarks ORDER BY id ASC")
                rows = cursor.fetchall()
                self.documents = [dict(row) for row in rows]

            doc_count = len(self.documents)

            if doc_count == 0:
                return

            # FAISS 밀집 벡터 디스크 미러링 리로드
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            else:
                self.index = faiss.IndexFlatIP(1024)

            # BM25 렉시컬 토크나이저 역색인 매트릭스 실시간 동적 빌드
            corpus = [f"{doc['title']} {doc['content']}".lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(corpus)

        except Exception as e:
            print(f"❌ [AI_CORE CRITICAL ERROR] 런타임 메모리 정합성 수렴 실패: {e}")

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
        print(f"\n==================== [하이브리드 랭킹 CORE v3 내부 수식 기하 연산] ====================")
        print(f" 인입 검색어 원문: '{query}' | 슬롯 제한 한계: Top {top_n}")

        if not self.documents or self.index is None or self.bm25 is None:
            print(" [WARN] 메모리에 빌드된 정합성 인덱스 매트릭스가 전무하므로 공백 배열 반환.")
            return []

        # ==========================================
        # LAYER 1: 렉시컬/시맨틱 분리 및 후보군 수집
        # ==========================================
        # ① BM25 기반 형태소 점수 추출
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # ② FAISS Dense 임베딩 공간 코사인 내적 행렬 추출
        query_vector = self.embedder.encode(query)[0].astype("float32")
        total_vectors = self.index.ntotal
        v_scores = np.zeros(total_vectors)
        
        for i in range(total_vectors):
            try:
                vec = self.index.reconstruct(i)
                # IP 인덱스 특성 상 정규화 내적 연산으로 완벽한 코사인 유사도 산출
                v_scores[i] = float(np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec) + 1e-9))
            except Exception:
                v_scores[i] = 0.0

        v_indices = np.argsort(v_scores)[::-1]
        candidate_set = set()
        
        # 렉시컬 분리 후보 압축
        lex_top_k = np.argsort(bm25_scores)[::-1][:self.stage1_k]
        for idx in lex_top_k:
            if bm25_scores[idx] > 0:
                candidate_set.add(idx)

        # 시맨틱 분리 후보 압축
        sem_top_k = v_indices[:self.stage1_k]
        for idx in sem_top_k:
            candidate_set.add(idx)

        # Fast-Track 가드레일: 코사인 유사도가 초우수 스케일(0.92) 충족 시 강제 풀 구출
        for idx, score in enumerate(v_scores):
            if score >= self.fast_track_threshold and idx not in candidate_set:
                candidate_set.add(idx)
                print(f"  [FAST-TRACK] 문서 ID #{self.documents[idx]['id']}번 무조건 강제 구출 (유사도: {score:.4f})")

        print(f" [LAYER 1] 후보군 격리 압축 완결: 총 {len(self.documents)}건 -> 1차 필터링 풀 {len(candidate_set)}건")

        if not candidate_set:
            print("  ❌ [ZERO-HITS DETECTION] 1단계 결합 풀 형성 단계에서 매칭 정보가 전무합니다.")
            return []

        # ==========================================
        # LAYER 2: 코어 문맥 중심 동적 정렬 (Core Attention Ranking)
        # ==========================================
        # 전역 공간 행렬의 기하학적 중심축 산출
        all_reconstructed_vecs = np.array([self.index.reconstruct(i) for i in range(total_vectors)])
        global_context_center = np.mean(all_reconstructed_vecs, axis=0)
        
        # 질의 밀도(Attention Energy) 연산 파싱
        attn_energy = float(np.dot(query_vector, global_context_center) / (np.linalg.norm(query_vector) * np.linalg.norm(global_context_center) + 1e-9))
        
        # 알파 및 베타 다이나믹 분배 매트릭스 활성화
        alpha = 1.0 / (1.0 + np.exp(-self.temperature * attn_energy))
        beta = 1.0 - alpha
        print(f" [LAYER 2] 글로벌 어텐션 분석: 밀도 에너지={attn_energy:.4f} | 동적 가중치 Alpha(시맨틱)={alpha:.2f}, Beta(렉시컬)={beta:.2f}")

        ranked_pool = []
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        for idx in candidate_set:
            doc = self.documents[idx]
            
            # 독립 엔진별 스코어 단일 스케일 융합 정규화
            score_lex = bm25_scores[idx] / max_bm25
            score_sem = v_scores[idx]
            
            # 신성 최신성 감쇄 계수 가중
            time_decay = self._calculate_decay(doc["created_at"])
            
            # 선형 지능형 합성 수식
            score_final = (alpha * score_sem + beta * score_lex) * time_decay
            
            ranked_pool.append({
                "id": doc["id"],
                "url": doc["url"],
                "title": doc["title"],
                "content": doc["content"],
                "score_final": score_final,
                "score_lex_raw": bm25_scores[idx],
                "score_sem_raw": v_scores[idx],
                "time_decay_factor": time_decay
            })

        # 선형 스코어 기준 상위 최적 정렬 개시
        ranked_pool.sort(key=lambda x: x["score_final"], reverse=True)

        # ==========================================
        # LAYER 3: 말단 외곽 검문 및 Zero-Hits 제어
        # ==========================================
        top_1_doc = ranked_pool[0]
        # SQLite 순차 레코드 인덱스 오프셋 동기화 복원
        top_1_global_idx = top_1_doc["id"] - 1

        # [수정 핵심 2]: 3단계 랭크 패널티 동점자 분포 및 소규모 자산 수식 예외 완벽 방어 방어벽
        try:
            global_lex_rank = int(np.where(np.argsort(bm25_scores)[::-1] == top_1_global_idx)[0][0]) + 1
        except IndexError:
            global_lex_rank = len(self.documents)

        try:
            global_sem_rank = int(np.where(v_indices == top_1_global_idx)[0][0]) + 1
        except IndexError:
            global_sem_rank = len(self.documents)

        rank_penalty = self._calculate_rank_penalty(global_lex_rank, global_sem_rank)
        verified_cutoff_score = top_1_doc["score_final"] * rank_penalty

        print(f" [LAYER 3] 최외각 검문소 통계 메트릭:")
        print(f"  - 1위 예측 매칭 자산: ID #{top_1_doc['id']} (제목: {top_1_doc['title'][:15]}...)")
        print(f"  - 각 엔진별 독점 전역 순위: BM25={global_lex_rank}등 | FAISS={global_sem_rank}등")
        print(f"  - 로그 등수 분산 패널티 인자 스케일: {rank_penalty:.4f}")
        print(f"  - 보정 전 합성 점수: {top_1_doc['score_final']:.4f} -> 외곽 검문 최종 스코어: {verified_cutoff_score:.4f}")

        # 수선된 임계치 미달 여부 엄격 검증 분기
        if verified_cutoff_score < self.zero_hits_threshold:
            print(f"  ❌ [ZERO-HITS FILTRATION ACTIVATE]")
            print(f"      최종 보정 점수({verified_cutoff_score:.4f})가 하한 마진선({self.zero_hits_threshold}) 미만으로 실격되었습니다.")
            print(f"      질의어 무관 오탐으로 확정 판정하여 후보군 전체를 파괴 배출합니다.")
            return []

        print(f"  [PASS] 하이브리드 검문 통과 완결. 상위 {min(top_n, len(ranked_pool))}건의 자산을 스티칭하여 출력 레이어로 바인딩합니다.")
        print(f"=========================================================================\n")
        
        # 전면부 프론트엔드가 엄격 매칭 대조하는 score 키 규격 매핑
        final_out = []
        for item in ranked_pool[:top_n]:
            item["score"] = item["score_final"]
            final_out.append(item)
        return final_out