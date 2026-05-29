import sqlite3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from async_worker.tasks import process_new_bookmark

app = FastAPI(title="IKG Intelligent Backend")

class BookmarkCreateRequest(BaseModel):
    url: str
    title: str
    content: str

@app.post("/api/bookmarks", status_code=202)
def create_bookmark(request: BookmarkCreateRequest):
    """새로운 기술 자산 북마크 인입 엔드포인트 (202 Accepted 반환)"""
    conn = sqlite3.connect("db/ikg_metadata.db")
    cursor = conn.cursor()
    
    try:
        # 1. 메타데이터 구조화 저장
        cursor.execute(
            "INSERT INTO bookmarks (url, title, content, created_at) VALUES (?, ?, ?, datetime('now', 'localtime'))",
            (request.url, request.title, request.content)
        )
        inserted_id = cursor.lastrowid
        conn.commit()
        
        # 2. [비동기 코어 트리거] Celery 분산 큐 레이어로 태스크 오프로딩
        # 이 시점에 메인 스레드는 블로킹 없이 무부하 상태로 탈출
        task_receipt = process_new_bookmark.delay(inserted_id)
        
        return {
            "message": "북마크 메타데이터가 정상 접수되었습니다. 백그라운드 AI 인덱싱이 시작됩니다.",
            "bookmark_id": inserted_id,
            "task_id": task_receipt.id
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        conn.close()
