import os
import sqlite3
import uuid
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from async_worker.schemas import BookmarkCreateRequest, TaskReceiptResponse

# 큐 모드 감지 (기본값은 대안 B인 'EMBEDDED'로 제어하되, 'CELERY' 모드의 여지를 남겨둠)
QUEUE_MODE = os.getenv("QUEUE_MODE", "EMBEDDED")

app = FastAPI(title="IKG Intelligent Backend API Gateway")
DB_PATH = os.getenv("IKG_DB_PATH", "db/ikg_metadata.db")

@app.on_event("startup")
def startup_event():
    """Ensure database directory and tables are created on startup."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bookmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                created_at TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

# =========================================================================
# [대안 B 최적화] Celery 인프라가 없을 때 프로세스 내부에서 줄을 세우기 위한 임베디드 코어
# =========================================================================
if QUEUE_MODE == "EMBEDDED":
    import queue
    import threading
    from async_worker.tasks import EmbeddedInferenceWorker
    
    # 1. 단일 프로세스 내에서 태스크를 순차적으로 격리할 Thread-safe 단일 큐 선언
    embedded_task_queue = queue.Queue()
    
    # 2. 로컬 전용 단일 싱글톤 워커 인스턴스 초기화 (ONNX 모델 웜업 포함)
    worker_instance = EmbeddedInferenceWorker()
    
    def _queue_worker_loop():
        """Celery Worker --concurrency=1의 동작을 완벽히 모사하는 무한 직렬 루프 스레드"""
        print("[EMBEDDED QUEUE] 로컬 독립형 스레드 큐 루프가 가동되었습니다. (concurrency=1)")
        while True:
            try:
                # 큐에 작업이 들어올 때까지 블로킹 대기 (CPU 자원 소모 0)
                bookmark_id = embedded_task_queue.get()
                if bookmark_id is None:
                    break
                
                # 직렬 방식으로 AI 추론 및 FAISS 파일 I/O 독점 수행
                worker_instance.execute_inference_pipeline(bookmark_id)
            except Exception as e:
                print(f"[EMBEDDED QUEUE ERROR] 백그라운드 스레드 파이프라인 내부 예외: {e}")
            finally:
                embedded_task_queue.task_done()

    # 데몬 스레드로 가동하여 메인 웹 서버 프로세스 종료 시 자동 소멸 유도
    worker_thread = threading.Thread(target=_queue_worker_loop, daemon=True)
    worker_thread.start()
else:
    # 대안 A 혹은 오리지널 분산 구조 모드일 경우 기존 임포트 유지
    from async_worker.tasks import process_new_bookmark


@app.post(
    "/api/bookmarks", 
    response_model=TaskReceiptResponse, 
    status_code=status.HTTP_202_ACCEPTED,
    summary="신규 기술 자산 접수 및 선택형 비동기 파이프라인 분기"
)
def create_bookmark(request: BookmarkCreateRequest, background_tasks: BackgroundTasks):
    """
    데이터 검증 완료 후 SQLite에 메타데이터를 선적재하고 지정된 큐 레이어로 이관합니다.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    
    try:
        # 1. 동기 레이어: 중복 URL 체크 및 메타데이터 고속 선적재
        cursor.execute("SELECT id FROM bookmarks WHERE url = ?", (str(request.url),))
        existing = cursor.fetchone()
        if existing:
            inserted_id = existing[0]
            return TaskReceiptResponse(
                message="이미 등록된 북마크입니다. 기존 인덱스 데이터를 사용합니다.",
                bookmark_id=inserted_id,
                task_id="existing-task-none",
                status="duplicate"
            )

        cursor.execute(
            """
            INSERT INTO bookmarks (url, title, content, created_at) 
            VALUES (?, ?, ?, datetime('now', 'localtime'))
            """,
            (str(request.url), request.title, request.content)
        )
        inserted_id = cursor.lastrowid
        conn.commit()
        
        # 2. 비동기 파이프라인 분기 레이어 (여지를 남겨두는 핵심 아키텍처)
        if QUEUE_MODE == "EMBEDDED":
            # [대안 B] 외부 메시지 브로커를 거치지 않고 내부 직렬 큐 스레드로 직접 바이패스
            generated_task_id = f"local-task-{uuid.uuid4()}"
            
            # BackgroundTasks를 활용하여 HTTP 응답 스트림 배출 직후 큐에 ID 적재
            background_tasks.add_task(embedded_task_queue.put, inserted_id)
            
            msg = "북마크 메타데이터 접수 완료. [내장 스레드 큐(Embedded)]를 통해 로컬 직렬 인덱싱이 시작됩니다."
        else:
            # [오리지널 / 대안 A 확장 모드] Redis 또는 SQLite Celery 브로커로 태스크 토스
            task_receipt = process_new_bookmark.delay(inserted_id)
            generated_task_id = task_receipt.id
            msg = "북마크 메타데이터 접수 완료. [분산 인프라 큐(Celery)]를 통해 백그라운드 인덱싱이 시작됩니다."
            
        return TaskReceiptResponse(
            message=msg,
            bookmark_id=inserted_id,
            task_id=generated_task_id
        )
    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"[인프라 에러] 메타데이터 적재 또는 큐 이관 실패: {str(e)}"
        )
    finally:
        conn.close()