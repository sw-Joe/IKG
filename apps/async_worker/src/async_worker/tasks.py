import os
import sqlite3
import faiss
import numpy as np
from celery import Celery, Task

from ai_core.core.embedder import BGEEmbedder

# 1. Celery 앱 인스턴스 정의 및 브로커 바인딩 (대안 A 모드용)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("ikg_tasks", broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
    worker_max_tasks_per_child=500,  # ONNX 인퍼런스 런타임 메모리 비대화 방지 가드
    worker_prefetch_multiplier=1     # AI 연산 최적화를 위한 1건 직렬 가져오기 강제
)

class EmbeddingInferenceTask(Task):
    """Celery 워커 프로세스 초기화 시점에 AI 모델을 싱글톤으로 로드하는 웜업 베이스 클래스"""
    _embedder = None
    _db_path = "db/ikg_metadata.db"
    _index_path = "db/ikg_vector.index"

    @property
    def embedder(self):
        if self._embedder is None:
            print("[CELERY SYSTEM] ONNX 고성능 임베딩 모델 메모리 웜업을 시작합니다.")
            self._embedder = BGEEmbedder(
                model_path="./model/bge-m3-onnx-int8", 
                file_name="model_quantized.onnx"
            )
        return self._embedder

@app.task(base=EmbeddingInferenceTask, bind=True, name="tasks.process_new_bookmark")
def process_new_bookmark(self, bookmark_id: int):
    """Celery 분산 인프라 모드 구동 시의 타깃 비동기 태스크 라우팅 진입점"""
    print(f"[CELERY TASK START] 북마크 ID {bookmark_id}번에 대한 분산 AI 연산을 시작합니다.")
    # 단일 프로세서의 실행 파이프라인 수식을 공유 호출
    worker_core = EmbeddedInferenceWorker()
    return worker_core.execute_inference_pipeline(bookmark_id)


# =========================================================================
# [대안 B 최적화] 외부 메시지 브로커 없이 프로세스 내부에 기생하여 인퍼런스를 독점하는 워커
# =========================================================================
class EmbeddedInferenceWorker:
    """FastAPI 내부 스레드 루프에 결합되어 싱글톤으로 인퍼런스 및 파일 I/O를 전담하는 워커 코어"""
    def __init__(self):
        self._db_path = "db/ikg_metadata.db"
        self._index_path = "db/ikg_vector.index"
        print("[EMBEDDED WORKER INIT] 단일 프로세스 전용 BGE-M3 ONNX 임베딩 모델 웜업을 수행합니다.")
        self.embedder = BGEEmbedder(
            model_path="./model/bge-m3-onnx-int8", 
            file_name="model_quantized.onnx"
        )

    def execute_inference_pipeline(self, bookmark_id: int):
        """추론 및 FAISS 파일 원자적 동기화 공통 파이프라인 코어 수식"""
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # 1. 메타데이터 파싱
            cursor.execute("SELECT title, content FROM bookmarks WHERE id = ?", (bookmark_id,))
            row = cursor.fetchone()
            
            if not row:
                print(f"[EMBEDDED WORKER WARN] ID {bookmark_id}가 DB에 존재하지 않습니다. 인덱싱을 취소합니다.")
                return {"status": "FAILED", "error": "Row missing"}
            
            # 타이틀과 본문을 결합하여 복합 시맨틱 컨텍스트 보존 (청킹 배제)
            combined_text = f"{row['title']} {row['content']}"
            
            # 2. BGE-M3 Dense 고차원 임베딩 추론 연산 실행 (자원 집중 구간)
            query_vec = self.embedder.encode(combined_text)
            vector_np = query_vec[0].astype("float32")
            
            # 3. FAISS 인덱스 독점 파일 I/O 및 디스크 플러시 (직렬 쓰기 정합성 가드)
            index = faiss.read_index(self._index_path)
            index.add(np.expand_dims(vector_np, axis=0))
            faiss.write_index(index, self._index_path)
            
            print(f"[EMBEDDED WORKER SUCCESS] 북마크 ID {bookmark_id} 인덱싱 완결. (전체 벡터: {index.ntotal})")
            return {
                "status": "SUCCESS", 
                "bookmark_id": bookmark_id, 
                "allocated_faiss_rank": index.ntotal - 1
            }
        except Exception as e:
            print(f"[EMBEDDED WORKER CRITICAL ERROR] 추론 파이프라인 처리 실패: {e}")
            return {"status": "FAILED", "error": str(e)}
        finally:
            conn.close()