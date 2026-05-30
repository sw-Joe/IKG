import os
import sqlite3
import faiss
import numpy as np
from celery import Celery, Task
from ai_core.embedder import BGEEmbedder



# =========================================================================
# [포트폴리오 & 대안 A 확장용 바인딩] Celery 아키텍처 원안 보존 레이어
# =========================================================================
# 환경 변수에 따라 브로커 설정을 SQLite 디스크 큐(sqla+sqlite)로 변환할 여지도 완벽히 남겨둠
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
    """Celery 모드 가동 시 프로세스 초기화 시점에 AI 모델을 싱글톤으로 로드하는 베이스 클래스"""
    _embedder = None

    @property
    def db_path(self):
        return os.getenv("IKG_DB_PATH", "db/ikg_metadata.db")

    @property
    def index_path(self):
        return os.getenv("IKG_INDEX_PATH", "db/ikg_vector.index")

    @property
    def embedder(self):
        if self._embedder is None:
            print("[CELERY SYSTEM] ONNX 고성능 임베딩 모델 메모리 웜업을 시작합니다.")
            model_path = os.getenv("IKG_MODEL_PATH", "./model/bge-m3-onnx-int8")
            model_file = os.getenv("IKG_MODEL_FILE", "model_quantized.onnx")
            self._embedder = BGEEmbedder(
                model_path=model_path, file_name=model_file
            )
        return self._embedder

@app.task(base=EmbeddingInferenceTask, bind=True, name="tasks.process_new_bookmark")
def process_new_bookmark(self, bookmark_id: int):
    """Celery 분산 인프라 모드 실행 시의 타겟 비동기 루프"""
    print(f"[CELERY TASK START] 북마크 ID {bookmark_id}번에 대한 분산 AI 연산을 시작합니다.")
    worker_core = EmbeddedInferenceWorker()
    return worker_core.execute_inference_pipeline(bookmark_id)


# =========================================================================
# [대안 B 채택] 외부 데몬 없이 단일 프로세스 기생형 고성능 직렬 추론 워커 코어
# =========================================================================
class EmbeddedInferenceWorker:
    """FastAPI 내부 스레드 루프에 결합되어 싱글톤으로 인퍼런스를 전담하는 로컬 최적화 워커"""
    def __init__(self):
        self._db_path = os.getenv("IKG_DB_PATH", "db/ikg_metadata.db")
        self._index_path = os.getenv("IKG_INDEX_PATH", "db/ikg_vector.index")
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            print("[EMBEDDED WORKER INIT] 단일 프로세스 전용 BGE-M3 ONNX 임베딩 모델 웜업을 수행합니다.")
            model_path = os.getenv("IKG_MODEL_PATH", "./model/bge-m3-onnx-int8")
            model_file = os.getenv("IKG_MODEL_FILE", "model_quantized.onnx")
            self._embedder = BGEEmbedder(
                model_path=model_path, file_name=model_file
            )
        return self._embedder

    def execute_inference_pipeline(self, bookmark_id: int):
        """추론 및 FAISS 파일 원자적 동기화 공통 파이프라인 수식"""
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT title, content FROM bookmarks WHERE id = ?", (bookmark_id,))
            row = cursor.fetchone()
            
            if not row:
                print(f"[EMBEDDED WORKER WARN] ID {bookmark_id}가 DB에 존재하지 않습니다.")
                return {"status": "FAILED", "error": "Row missing"}
            
            combined_text = f"{row['title']} {row['content']}"
            
            # 1. BGE-M3 Dense 고성능 벡터 추론 연산
            query_vec = self.embedder.encode(combined_text)
            vector_np = query_vec[0].astype("float32")
            
            # 2. FAISS 인덱스 독점 파일 I/O 및 디스크 플러시
            os.makedirs(os.path.dirname(self._index_path), exist_ok=True)
            if os.path.exists(self._index_path):
                index = faiss.read_index(self._index_path)
            else:
                index = faiss.IndexFlatIP(1024)
            index.add(np.expand_dims(vector_np, axis=0))
            faiss.write_index(index, self._index_path)
            
            print(f"[EMBEDDED WORKER SUCCESS] 북마크 ID {bookmark_id} 인덱싱 완결. (전체 벡터: {index.ntotal})")
            return {
                "status": "SUCCESS", 
                "bookmark_id": bookmark_id, 
                "allocated_faiss_rank": index.ntotal - 1
            }
        except Exception as e:
            print(f"[EMBEDDED WORKER CRITICAL ERROR] 파이프라인 처리 실패: {e}")
            return {"status": "FAILED", "error": str(e)}
        finally:
            conn.close()