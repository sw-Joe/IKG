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
    worker_max_tasks_per_child=500,
    worker_prefetch_multiplier=1
)


class EmbeddingInferenceTask(Task):
    """Celery 분산 환경 선택 시 추론 모델 싱글톤 적재 가드 클래스"""
    _embedder = None
    _db_path = IKG_DB_PATH
    _index_path = IKG_INDEX_PATH

    @property
    def embedder(self):
        if self._embedder is None:
            logger.info("[CELERY SYSTEM] 고성능 ONNX 임베딩 모델 컨텍스트 웜업 가동")
            self._embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
        return self._embedder


class EmbeddedInferenceWorker:
    """단일 스레드 비동기 컨텍스트(Concurrency=1) 하에서 C/U/D 물리 색인을 전담하는 독점 액터"""
    def __init__(self, db_path=None, index_path=None):
        self.db_path = db_path or IKG_DB_PATH
        self.index_path = index_path or IKG_INDEX_PATH
        
        logger.info("[WORKER INIT] EMBEDDED 모드 전용 BGE-M3 ONNX 인퍼런스 세션을 기동합니다.")
        self.embedder = BGEEmbedder(model_path=IKG_MODEL_PATH, file_name=IKG_MODEL_FILE)
        self.indexer_engine = VectorIndexer(db_path=self.db_path, index_path=self.index_path, dimension=1024)

    def execute_upsert_pipeline(self, bookmark_id: int) -> dict:
        """Create 및 Update 공통 처리 파이프라인 (FAISS 중복 벡터 누적 결함 원천 방어)"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # 1. 활성 상태(is_deleted=0)의 수정된 원문 컨텐츠 로드
            cursor.execute("SELECT title, content FROM bookmarks WHERE id = ? AND is_deleted = 0", (bookmark_id,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"[WORKER WARN] #{bookmark_id} 자산이 무효하거나 Soft-Deleted 상태입니다. 색인을 스킵합니다.")
                return {"status": "SKIPPED"}

            combined_text = f"{row['title']} {row['content']}"
            
            # 2. 고부하 ONNX 임베딩 추론 실행
            query_vec = self.embedder.encode(combined_text)
            vector_np = query_vec[0].astype("float32")
            
            # 3. FAISS 바이너리 독점 로드
            index = faiss.read_index(self.index_path)
            
            # [CRITICAL GUARD] UPDATE/CREATE 시 발생할 수 있는 동일 ID에 대한 벡터 중복 축적을 막기 위해
            # 인덱스 내부 공간에서 기존 매핑 식별자를 선제적으로 완전 소거 후 재배치 집행
            purge_id_np = np.array([bookmark_id], dtype=np.int64)
            index.remove_ids(purge_id_np)
            
            # 4. 소거 완료된 공간 말단에 최신 추론 벡터 및 고유 PK 결합
            vectors_np = np.expand_dims(vector_np, axis=0)
            index.add_with_ids(vectors_np, purge_id_np)
            
            # 디스크 원자적 저장 플러시
            faiss.write_index(index, self.index_path)
            logger.info(f"[WORKER SUCCESS] 북마크 #{bookmark_id} 벡터 오버라이트 완료 (전체 벡터 수: {index.ntotal})")
            
            # 5. 지연된 물리 소거(Deferred Purge) 배치 스케줄러 자동 집행
            self.indexer_engine.check_and_purge_garbage(index, purge_threshold=20)
            
            return {"status": "SUCCESS", "bookmark_id": bookmark_id}
        except Exception as e:
            logger.error(f"[WORKER CRITICAL ERROR] 내장 큐 파이프라인 작동 실패: {str(e)}", exc_info=True)
            return {"status": "FAILED", "error": str(e)}
        finally:
            conn.close()


@app.task(base=EmbeddingInferenceTask, bind=True, name="be_tasks.process_new_bookmark")
def process_new_bookmark(self, bookmark_id):
    """대안 A(CELERY) 분산 환경 가동 시 소비 주체 인터페이스"""
    conn = sqlite3.connect(self._db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT title, content FROM bookmarks WHERE id = ? AND is_deleted = 0", (bookmark_id,))
        row = cursor.fetchone()
        if not row:
            return {"status": "SKIPPED"}
            
        combined_text = f"{row['title']} {row['content']}"
        query_vec = self.embedder.encode(combined_text)
        vector_np = query_vec[0].astype("float32")
        
        index = faiss.read_index(self._index_path)
        
        purge_id_np = np.array([bookmark_id], dtype=np.int64)
        index.remove_ids(purge_id_np)
        
        index.add_with_ids(np.expand_dims(vector_np, axis=0), purge_id_np)
        faiss.write_index(index, self._index_path)
        
        logger.info(f"[CELERY WORKER SUCCESS] 문서 ID #{bookmark_id} 인덱스 리빌드 완료.")
        return {"status": "SUCCESS", "bookmark_id": bookmark_id}
    except Exception as e:
        logger.error(f"[CELERY WORKER CRASH] {str(e)}", exc_info=True)
        return {"status": "FAILED", "error": str(e)}
    finally:
        conn.close()