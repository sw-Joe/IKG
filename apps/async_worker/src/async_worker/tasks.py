import os
import sqlite3

import faiss
import numpy as np
from celery import Celery, Task

from ai_core.embedder import BGEEmbedder

# 1. Celery 앱 인스턴스 정의 (Redis 브로커 바인딩)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("ikg_tasks", broker=REDIS_URL, backend=REDIS_URL)

# Celery 글로벌 콘피그레이션 최적화
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
    worker_max_tasks_per_child=1000, # 메모리 누수 방지를 위한 주기적 자식 프로세스 재시작
    worker_prefetch_multiplier=1     # 중량급 AI 연산 특성에 맞춘 예측 가저오기 제한 (무조건 1개씩 직렬 처리)
)

class EmbeddingInferenceTask(Task):
    """Celery Worker 프로세스 초기화 시점에 AI 모델을 싱글톤으로 로드하는 베이스 클래스"""
    _embedder = None
    _db_path = "db/ikg_metadata.db"
    _index_path = "db/ikg_vector.index"

    @property
    def embedder(self):
        if self._embedder is None:
            print("[WORKER SYSTEM] ONNX 고성능 임베딩 모델 메모리 웜업을 시작합니다.")
            self._embedder = BGEEmbedder(
                model_path="./model/bge-m3-onnx-int8", 
                file_name="model_quantized.onnx"
            )
        return self._embedder

@app.task(base=EmbeddingInferenceTask, bind=True, name="tasks.process_new_bookmark")
def process_new_bookmark(self, bookmark_id: int):
    """백그라운드 비동기 파이프라인 코어 워커 태스크"""
    print(f"[TASK START] 북마크 ID {bookmark_id}번에 대한 비동기 AI 파이프라인 연산을 시작합니다.")
    
    # 1. SQLite 데이터베이스에서 가공 타겟 텍스트 추출
    conn = sqlite3.connect(self._db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT title, content FROM bookmarks WHERE id = ?", (bookmark_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return {"status": "FAILED", "error": f"Bookmark ID {bookmark_id} not found."}
    
    # 2. 콘텐트 가공 및 유효성 결합
    combined_text = f"{row['title']} {row['content']}"
    
    try:
        # 3. ONNX 임베딩 모델을 통한 Dense 고성능 벡터 추론 수행
        # self.embedder property를 통해 싱글톤 인스턴스에 안전하게 접근
        query_vec = self.embedder.encode(combined_text) # [1, Dimension] 크기의 고차원 벡터 반환
        vector_np = query_vec[0].astype("float32")
        
        # 4. FAISS 인덱스 동기화 및 파일 지속성 저장 (Thread-safe 영역 확보 필요)
        index = faiss.read_index(self._index_path)
        
        # 정합성 검증: 현재 FAISS 내의 인덱스 벡터 수와 SQLite 내의 행 번호 간 매핑 확인
        # 새 문서가 항상 ID 순서대로 적재된다고 가정할 때, 원본 벡터를 말단에 추가
        index.add(np.expand_dims(vector_np, axis=0))
        faiss.write_index(index, self._index_path)
        
        print(f"[TASK SUCCESS] 북마크 ID {bookmark_id} 임베딩 연산 및 FAISS 색인이 완료되었습니다. (FAISS Total: {index.ntotal})")
        return {"status": "SUCCESS", "bookmark_id": bookmark_id, "faiss_index": index.ntotal - 1}
        
    except Exception as e:
        print(f"[TASK ERROR] 추론 파이프라인 구동 중 실패: {e}")
        return {"status": "FAILED", "error": str(e)}
    finally:
        conn.close()