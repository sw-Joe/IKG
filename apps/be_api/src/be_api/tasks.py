import os
import sqlite3
import logging

import faiss
import numpy as np
from celery import Celery, Task

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH, IKG_MODEL_PATH, IKG_MODEL_FILE
from ai_core.core.embedder import BGEEmbedder
from ai_core.core.indexer import VectorIndexer



logger = logging.getLogger("be_api.tasks")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("ikg_tasks", broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
    worker_max_tasks_per_child=500,  # ONNX 인퍼런스 누수 가드
    worker_prefetch_multiplier=1     # 순차 처리 태스크 바인딩 최적화
)


class EmbeddingInferenceTask(Task):
    """Celery 분산 워커 환경 가동 시 모델을 싱글톤으로 적재하는 웜업 가드 클래스"""
    _embedder = None
    _db_path = IKG_DB_PATH
    _index_path = IKG_INDEX_PATH

    @property
    def embedder(self):
        if self._embedder is None:
            logger.info("[CELERY SYSTEM] 고성능 ONNX 임베딩 모델 컨텍스트 웜업 가동")
            self._embedder = BGEEmbedder(
                model_path=IKG_MODEL_PATH,
                file_name=IKG_MODEL_FILE
            )
        return self._embedder


class EmbeddedInferenceWorker:
    """단일 스레드 컨커런시(concurrency=1) 환경 하에서 안전하게 증분/수정/물리소거를 전담하는 액터"""
    def __init__(self, db_path=None, index_path=None):
        self.db_path = db_path or IKG_DB_PATH
        self.index_path = index_path or IKG_INDEX_PATH
        
        logger.info("[WORKER INIT] EMBEDDED 모드 전용 BGE-M3 ONNX 인퍼런스 세션을 기동합니다.")
        self.embedder = BGEEmbedder(
            model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE
        )
        self.indexer_engine = VectorIndexer(
            db_path=self.db_path, index_path=self.index_path, dimension=1024
        )

    def execute_upsert_pipeline(self, bookmark_id: int):
        """Create 및 Update 공통 격리 처리 파이프라인 (FAISS 중복 벡터 누적 결함 원천 방어)"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # 1. 활성 상태(is_deleted=0)의 원문 콘텍스트 확보
            cursor.execute("SELECT title, content FROM bookmarks WHERE id = ? AND is_deleted = 0", (bookmark_id,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"[WORKER WARN] #{bookmark_id} 자산이 무효하거나 Soft-Deleted 상태입니다. 색인을 무효화합니다.")
                return {"status": "SKIPPED"}

            combined_text = f"{row['title']} {row['content']}"
            
            # 2. 고부하 AI 추론 연산 격리 실행
            query_vec = self.embedder.encode(combined_text)
            vector_np = query_vec[0].astype("float32")
            
            # 3. FAISS 독점 파일 Lock 획득 후 로드
            index = faiss.read_index(self.index_path)
            
            # [CRITICAL GUARD] 중복 인입에 따른 ID 벡터 누적을 원천 배제하기 위해 기존 ID 물리적 선제 소거
            purge_id_np = np.array([bookmark_id], dtype=np.int64)
            index.remove_ids(purge_id_np)
            
            # 4. 정제 완료된 인덱스 공간 말단에 신규 벡터 결합
            vectors_np = np.expand_dims(vector_np, axis=0)
            index.add_with_ids(vectors_np, purge_id_np)
            
            # 디스크 원자적 저장 플러시
            faiss.write_index(index, self.index_path)
            logger.info(f"[WORKER SUCCESS] 북마크 #{bookmark_id} 벡터 동기화 완결. (전체 풀: {index.ntotal}건)")
            
            # 5. 지연된 물리 소거(Deferred Purge) 감시 스케줄러 전담 트리거
            self.indexer_engine.check_and_purge_garbage(index, purge_threshold=20)
            
            return {"status": "SUCCESS", "bookmark_id": bookmark_id}
        except Exception as e:
            logger.error(f"[WORKER CRITICAL ERROR] 내장 큐 파이프라인 작동 실패: {str(e)}", exc_info=True)
            return {"status": "FAILED", "error": str(e)}
        finally:
            conn.close()


@app.task(base=EmbeddingInferenceTask, bind=True, name="be_tasks.process_new_bookmark")
def process_new_bookmark(self, bookmark_id):
    """대안 A(CELERY) 분산 환경 가동 시 소비 주체 인터페이스 (IDMap 규격 패치 반영)"""
    conn = sqlite3.connect(self._db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT title, content FROM bookmarks WHERE id = ? AND is_deleted = 0", (bookmark_id,))
        row = cursor.fetchone()
        if not row:
            return {"status": "SKIPPED", "reason": "Row missing or inactive"}
            
        combined_text = f"{row['title']} {row['content']}"
        query_vec = self.embedder.encode(combined_text)
        vector_np = query_vec[0].astype("float32")
        
        index = faiss.read_index(self._index_path)
        
        # Celery 환경에서도 IDMap 중복 누적 현상 방어선 결합
        purge_id_np = np.array([bookmark_id], dtype=np.int64)
        index.remove_ids(purge_id_np)
        
        index.add_with_ids(np.expand_dims(vector_np, axis=0), purge_id_np)
        faiss.write_index(index, self._index_path)
        
        logger.info(f"[CELERY WORKER SUCCESS] 문서 ID #{bookmark_id} 인덱싱 플러시 성공.")
        return {"status": "SUCCESS", "bookmark_id": bookmark_id}
    except Exception as e:
        logger.error(f"[CELERY WORKER CRASH] {str(e)}", exc_info=True)
        return {"status": "FAILED", "error": str(e)}
    finally:
        conn.close()