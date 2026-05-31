import os
import sqlite3
import faiss
import numpy as np
from celery import Celery, Task

from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH, IKG_MODEL_PATH, IKG_MODEL_FILE
from ai_core.core.embedder import BGEEmbedder

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
    """대안 B(EMBEDDED) 내장 단일 스레드 전용 무오버헤드 인퍼런스 워커 자산"""
    def __init__(self):
        self._db_path = IKG_DB_PATH
        self._index_path = IKG_INDEX_PATH
        print("[EMBEDDED WORKER] 중앙 설정 상수 기반 자원 동기화 마감 완료.")
        self.embedder = BGEEmbedder(
            model_path=IKG_MODEL_PATH,
            file_name=IKG_MODEL_FILE
        )

    def execute_inference_pipeline(self, bookmark_id):
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT title, content FROM bookmarks WHERE id = ?", (bookmark_id,))
            row = cursor.fetchone()
            
            if not row:
                print(f"[EMBEDDED WORKER WARN] ID #{bookmark_id} 문서가 영속 데이터베이스에 실존하지 않습니다.")
                return {"status": "FAILED", "error": "Row missing"}
            
            combined_text = f"{row['title']} {row['content']}"
            
            # 고성능 Dense ONNX Quantized 임베딩 추출 추론 가동
            query_vec = self.embedder.encode(combined_text)
            vector_np = query_vec[0].astype("float32")
            
            # FAISS 바이너리 파일 안전 결합 및 즉각 플러시
            index = faiss.read_index(self._index_path)
            index.add(np.expand_dims(vector_np, axis=0))
            faiss.write_index(index, self._index_path)
            
            print(f"[EMBEDDED WORKER SUCCESS] 문서 ID #{bookmark_id} 인덱싱 최종 완결 (누적 벡터: {index.ntotal}개)")
            return {"status": "SUCCESS", "bookmark_id": bookmark_id, "current_total": index.ntotal}
            
        except Exception as e:
            print(f"❌ [EMBEDDED WORKER CRITICAL CRASH] 추론 파이프라인 예외 파괴: {e}")
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