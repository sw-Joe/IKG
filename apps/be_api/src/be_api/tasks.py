import os
import sqlite3
import faiss
import numpy as np
from celery import Celery, Task

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH, IKG_MODEL_PATH, IKG_MODEL_FILE
from ai_core.core.embedder import BGEEmbedder
from ai_core.core.indexer import VectorIndexer

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
            print("[CELERY SYSTEM] 고성능 ONNX 임베딩 모델 컨텍스트 웜업 가동")
            self._embedder = BGEEmbedder(
                model_path=IKG_MODEL_PATH,
                file_name=IKG_MODEL_FILE
            )
        return self._embedder


class EmbeddedInferenceWorker:
    """단일 프로세스 환경 하에서 CQRS 쓰기 독점을 수행하며 FAISS IDMap 영속을 제어하는 백그라운드 액터"""
    def __init__(self):
        self._db_path = "db/ikg_metadata.db"
        self._index_path = "db/ikg_vector.index"
        
        # 중량급 모델 싱글톤 초기화 웜업
        print("[EMBEDDED WORKER INIT] ONNX 임베딩 모델 로드 중...")
        self.embedder = BGEEmbedder(
            model_path="./model/bge-m3-onnx-int8", file_name="model_quantized.onnx"
        )
        # [신규 바인딩] 쓰기 전담 커맨드 모듈 초기화
        self.indexer_engine = VectorIndexer(
            db_path=self._db_path, index_path=self._index_path, dimension=1024
        )

    def execute_inference_pipeline(self, bookmark_id: int):
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # 활성화 상태인 데이터 타겟 문자열 가공 수집
            cursor.execute("SELECT title, content FROM bookmarks WHERE id = ? AND is_deleted = 0", (bookmark_id,))
            row = cursor.fetchone()
            if not row:
                return {"status": "SKIPPED", "reason": "Active metadata matching row missing"}
            
            combined_text = f"{row['title']} {row['content']}"
            
            # [책임 격리] 분리 완료된 독립 커맨드 엔진에 추론 파이프라인 및 가비지 소거 임무 완전 대리 위임
            self.indexer_engine.add_document_vector(
                bookmark_id=bookmark_id, text_content=combined_text, embedder=self.embedder
            )
            
            return {"status": "SUCCESS", "bookmark_id": bookmark_id}
        except Exception as e:
            print(f"[WORKER PIPELINE CRITICAL ERROR] 백그라운드 색인 실패: {e}")
            return {"status": "FAILED", "error": str(e)}
        finally:
            conn.close()


@app.task(base=EmbeddingInferenceTask, bind=True, name="be_tasks.process_new_bookmark")
def process_new_bookmark(self, bookmark_id):
    """대안 A(CELERY) 분산 환경 선택 가동 시 소비 주체 인터페이스"""
    conn = sqlite3.connect(self._db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT title, content FROM bookmarks WHERE id = ?", (bookmark_id,))
        row = cursor.fetchone()
        if not row:
            return {"status": "FAILED", "error": "Row missing"}
            
        combined_text = f"{row['title']} {row['content']}"
        query_vec = self.embedder.encode(combined_text)
        vector_np = query_vec[0].astype("float32")
        
        index = faiss.read_index(self._index_path)
        index.add(np.expand_dims(vector_np, axis=0))
        faiss.write_index(index, self._index_path)
        
        print(f"[CELERY WORKER SUCCESS] 문서 ID #{bookmark_id} 인덱싱 플러시 성공.")
        return {"status": "SUCCESS", "bookmark_id": bookmark_id}
    except Exception as e:
        print(f"❌ [CELERY WORKER CRASH] {e}")
        return {"status": "FAILED", "error": str(e)}
    finally:
        conn.close()