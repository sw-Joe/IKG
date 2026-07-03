import asyncio
from contextlib import asynccontextmanager
import logging
import queue
import sqlite3
import threading
import time

import numpy as np
import uvicorn
import faiss
from fastapi import FastAPI, HTTPException, status, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from ai_core.core.bookmark_scraper import warmup_browser_infrastructure
from ai_core import HybridSearcher
from ai_core.core.bookmark_scraper import scrape_url_standalone, validate_scraped_bookmark
from ai_core.core.resource_guard import DynamicResourceGuard  # 분리된 자원 통제 모듈 수입
from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH
from be_api.logger_config import setup_logging
from be_api.schemas import BookmarkCreateRequest, BookmarkIngestRequest, TaskReceiptResponse
from be_api.tasks import EmbeddedInferenceWorker



# 1. 시스템 전역 로깅 인프라 초기화 가동
setup_logging()
logger = logging.getLogger("be_api.app")

# [LIFESPAN CORES]: 스타트업 인프라 안전장치
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 외부 폭격 요청 진입 전 브라우저 프로세스 선제 완벽 가동
    await warmup_browser_infrastructure()
    yield
    # 2. 셧다운 시 자원 해제 가드 (필요 시 구현)
    pass

app = FastAPI(title="IKG Hybrid Search Gateway", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 💡 호스트 가용 RAM 자원을 스캔하여 백로그 한계를 설정하는 자원 가드 인스턴스 초기화
resource_guard = DynamicResourceGuard(memory_allocation_ratio=0.3, estimated_tab_cost_mb=150)

# SQLite 동시성 쓰기 컨텐션 방어용 비동기 뮤텍스 락
_db_write_lock = asyncio.Lock()


# =========================================================================
# [INFRASTRUCTURE]: 전역 엔드포인트 입출력 추적 고해상도 로깅 미들웨어
# =========================================================================
class UbiquitousLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        path = request.url.path
        method = request.method
        
        # 1. 인바운드 인풋 페이로드 안전 가로채기
        body_str = ""
        if method in ["POST", "PUT", "PATCH"] and "application/json" in request.headers.get("content-type", ""):
            body_bytes = await request.body()
            body_str = body_bytes.decode("utf-8", errors="ignore")
            # 스트림 소비 후 소멸 캐시 복구 주입
            async def receive():
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            request._receive = receive

        logger.info(f"▶▶▶ [HTTP REQ Ingress] {method} {path} | Payload: {body_str if body_str else 'None'}")

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(f"❌❌❌ [INTERNAL CRASH] {method} {path} 연산 실패 사유: {str(exc)}", exc_info=True)
            raise exc

        process_time = (time.time() - start_time) * 1000
        
        # 2. 아웃바운드 아웃풋 페이로드 복제 계측 가드
        response_body = b""
        if response.status_code != 500 and "application/json" in response.headers.get("content-type", ""):
            async for chunk in response.body_iterator:
                response_body += chunk
            response.body_iterator = AsyncBytesIterator([response_body])

        resp_str = response_body.decode("utf-8", errors="ignore")
        if len(resp_str) > 1000:
            resp_str = resp_str[:1000] + f" ... [Truncated, Total Length: {len(resp_str)} Bytes]"

        logger.info(
            f"◀◀◀ [HTTP RESP Egress] {method} {path} | Status: {response.status_code} "
            f"| Latency: {process_time:.2f}ms | Out-Data: {resp_str if resp_str else 'Empty'}"
        )
        return response


class AsyncBytesIterator:
    """FastAPI 스트림 응답 원형 복원을 위한 비동기 이터레이터 래퍼 객체"""
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


app.add_middleware(UbiquitousLoggingMiddleware)

# 전역 코어 검색 엔진 및 인퍼런스 타스크 컴포넌트 로드
searcher_engine = HybridSearcher()
embedded_task_queue = queue.Queue()
worker_actor = EmbeddedInferenceWorker(db_path=IKG_DB_PATH, index_path=IKG_INDEX_PATH)

indexing_state_lock = threading.Lock()
indexing_active_tasks = 0


def _embedded_queue_consumer_loop():
    global indexing_active_tasks
    logger.info("[EMBEDDED BUS] 단일 스레드 비동기 직렬화 컨텍스트 소비 루프 가동 완료.")
    while True:
        task_item = embedded_task_queue.get()
        task_started = False
        try:
            if task_item is None:
                break
            with indexing_state_lock:
                indexing_active_tasks += 1
                task_started = True

            action = task_item.get("action", "ADD")
            bookmark_id = task_item.get("id")
            result = {"status": "SKIPPED"}

            if action in {"ADD", "REINDEX"}:
                result = worker_actor.execute_sequential_inference_pipeline(bookmark_id)
            elif action == "DELETE":
                result = worker_actor.execute_sequential_removal_pipeline(bookmark_id)

            if result.get("status") == "SUCCESS":
                searcher_engine.refresh_context()
        except Exception as e:
            logger.error(f"[EMBEDDED BUS CRITICAL ERROR] 내장 큐 파이프라인 작동 실패: {str(e)}", exc_info=True)
        finally:
            if task_started:
                with indexing_state_lock:
                    indexing_active_tasks -= 1
            embedded_task_queue.task_done()


consumer_thread = threading.Thread(target=_embedded_queue_consumer_loop, daemon=True)
consumer_thread.start()


# =========================================================================
# [CRUD: CREATE] 실시간 단건 인입 라우터 (격리 분리된 모듈 링킹 완료)
# =========================================================================
@app.post("/api/bookmarks", response_model=TaskReceiptResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_bookmark_endpoint(payload: BookmarkIngestRequest):
    target_url = str(payload.url)
    
    # 💡 1. 분리된 가드 모듈을 이용한 메모리 백로그 스로틀링 검문 (초과 시 즉각 429 거절)
    await resource_guard.acquire_ingress_permits()

    try:
        # 전역 세마포어 한계선 내부에서 싱글톤 브라우저 컨텍스트 패칭 가동
        scraped_title, scraped_content = await scrape_url_standalone(target_url)
        
        is_valid_content, validation_result = validate_scraped_bookmark(scraped_title, scraped_content)
        if is_valid_content:
            scraped_content = validation_result
            isolation_reason = ""
        else:
            isolation_reason = validation_result
        
        if is_valid_content and len(scraped_content or "") < 100:
            is_valid_content = False
            isolation_reason = "TEXT_LENGTH_INSUFFICIENT_UNDER_100"

        final_title = scraped_title or payload.title or target_url
        final_content = scraped_content or payload.content or ""

        # SQLite 비동기 쓰기 락 뮤텍스 컨텍스트 진입
        async with _db_write_lock:
            conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            try:
                if is_valid_content:
                    cursor.execute(
                        """
                        INSERT INTO bookmarks (url, title, content, created_at, is_deleted, index_written)
                        VALUES (?, ?, ?, datetime('now', 'localtime'), 0, 0)
                        """,
                        (target_url, final_title, final_content)
                    )
                    assigned_id = cursor.lastrowid
                    cursor.execute("DELETE FROM bookmarks_isolated WHERE url = ?", (target_url,))
                    conn.commit()

                    embedded_task_queue.put({"action": "ADD", "id": assigned_id})

                    return TaskReceiptResponse(
                        message="실시간 웹 스크래핑 및 정합성 검증 통과 완료. 백그라운드 벡터 차원 공간 적재 프로세스를 기동합니다.",
                        bookmark_id=assigned_id,
                        task_id=f"TASK-ADD-{assigned_id}"
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO bookmarks_isolated (url, title, content, created_at, isolation_reason)
                        VALUES (?, ?, ?, datetime('now', 'localtime'), ?)
                        """,
                        (target_url, final_title, final_content, isolation_reason)
                    )
                    assigned_id = cursor.lastrowid
                    conn.commit()

                    return TaskReceiptResponse(
                        message=f"[VALIDATION_HOLD] 실시간 수집 결과 정보 가치 미달로 인한 격리 공간 이관 완수. 사유: {isolation_reason}",
                        bookmark_id=assigned_id,
                        task_id=f"TASK-ISOLATE-{assigned_id}",
                        status="isolated"
                    )

            except sqlite3.IntegrityError:
                raise HTTPException(status_code=400, detail="이미 시스템 내부 인프라에 상주 중인 유일 고유 URL 명세입니다.")
            finally:
                conn.close()
                
    finally:
        # 💡 2. 작업 완료 또는 예외 이탈 시 비동기 카운터 리소스 강제 반환 보장
        await resource_guard.release_ingress_permits()


# =========================================================================
# [CRUD: UPDATE] 자산 정정 수정 라우터
# =========================================================================
@app.put("/api/bookmarks/{bookmark_id}", status_code=status.HTTP_200_OK)
async def update_bookmark_endpoint(bookmark_id: int, payload: BookmarkCreateRequest):
    async with _db_write_lock:
        conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM bookmarks WHERE id = ?", (bookmark_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="수정할 대상 자산이 존재하지 않습니다.")

            cursor.execute(
                "UPDATE bookmarks SET title = ?, content = ?, url = ? WHERE id = ?",
                (payload.title, payload.content, str(payload.url), bookmark_id)
            )
            conn.commit()
            embedded_task_queue.put({"action": "REINDEX", "id": bookmark_id})

            return {"status": "SUCCESS", "updated_id": bookmark_id}
        finally:
            conn.close()


# =========================================================================
# [CRUD: UPDATE - RECOVER] 보류 자산 정정 승격 트랜잭션 (FE 수환 구조 일치)
# =========================================================================
@app.put("/api/bookmarks/recover/{isolated_id}", status_code=status.HTTP_202_ACCEPTED)
async def recover_isolated_bookmark_endpoint(isolated_id: int, payload: BookmarkCreateRequest):
    async with _db_write_lock:
        conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT url, created_at FROM bookmarks_isolated WHERE id = ?", (isolated_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="지정된 격리 자산 식별자가 부재합니다.")
            original_url, original_created_at = row

            cursor.execute("BEGIN TRANSACTION;")
            try:
                cursor.execute(
                    """
                    INSERT INTO bookmarks (url, title, content, created_at, is_deleted, index_written)
                    VALUES (?, ?, ?, ?, 0, 0)
                    """,
                    (original_url, payload.title, payload.content, original_created_at)
                )
                new_main_id = cursor.lastrowid

                cursor.execute("DELETE FROM bookmarks_isolated WHERE id = ?", (isolated_id,))
                conn.commit()
            except sqlite3.IntegrityError:
                conn.rollback()
                raise HTTPException(status_code=400, detail="정정 복구하려는 URL 자산이 이미 메인 자산 테이블에 존재합니다.")
            except Exception as e:
                conn.rollback()
                raise HTTPException(status_code=500, detail=f"데이터베이스 통합 붕괴: {str(e)}")

            embedded_task_queue.put({"action": "ADD", "id": new_main_id})

            updated_node_snapshot = [{
                "id": new_main_id,
                "title": payload.title,
                "url": original_url,
                "score": 1.0
            }]
            return updated_node_snapshot
            
        finally:
            conn.close()


# =========================================================================
# [CRUD: DELETE] 하이브리드 완전 동기화 소거 라우터
# =========================================================================
@app.delete("/api/bookmarks/{bookmark_id}", status_code=status.HTTP_200_OK)
async def delete_bookmark_endpoint(bookmark_id: int):
    async with _db_write_lock:
        conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM bookmarks WHERE id = ?", (bookmark_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="삭제 대상이 부재합니다.")

            cursor.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))
            conn.commit()
            embedded_task_queue.put({"action": "DELETE", "id": bookmark_id})
            return {"status": "SUCCESS", "purged_id": bookmark_id}
        finally:
            conn.close()


# =========================================================================
# [CQRS READ] 하이브리드 시맨틱 검색 게이트웨이 엔드포인트
# =========================================================================
@app.get("/api/search")
async def search_bookmarks_endpoint(q: str | None = None, query: str | None = None, limit: int = 5):
    effective_query = q or query
    if not effective_query or not effective_query.strip():
        return []

    try:
        raw_results = searcher_engine.search(query=effective_query, top_n=limit)
        formatted_list = []
        
        conn = sqlite3.connect(IKG_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            for doc in raw_results:
                target_id = doc.get("id") or doc.get("doc_id")
                if target_id is None:
                    continue
                    
                target_id = int(target_id)
                score_val = round(float(doc.get("score", 0.0)), 4)
                
                cursor.execute("SELECT title, url FROM bookmarks WHERE id = ?", (target_id,))
                db_row = cursor.fetchone()
                
                title_val = db_row["title"] if db_row else doc.get("title", f"지식 자산 #{target_id}")
                url_val = db_row["url"] if db_row else doc.get("url", "")
                
                formatted_list.append({
                    "id": target_id,
                    "title": title_val,
                    "content": doc.get("content", ""),
                    "url": url_val,
                    "score": score_val,
                    "score_lex_raw": round(float(doc.get("score_lex_raw", 0.0)), 4),
                    "score_sem_raw": round(float(doc.get("score_sem_raw", 0.0)), 4)
                })
        finally:
            conn.close()
            
        return formatted_list
        
    except Exception as e:
        logger.error(f"[API SEARCH ERROR] 하이브리드 검색 컨텍스트 연산 장애: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="시맨틱 하이브리드 공간 검색 정렬 붕괴")


# =========================================================================
# [CQRS READ] 3D 글로벌 공간 시각화 토폴로지 추출 레이어
# =========================================================================
@app.get("/api/graph")
async def get_knowledge_graph_matrix_endpoint(threshold: float = 0.85):
    conn = sqlite3.connect(IKG_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, title, url FROM bookmarks WHERE is_deleted = 0")
        rows = cursor.fetchall()
        documents = [{"id": r["id"], "title": r["title"], "url": r["url"]} for r in rows]

        nodes = [
            {
                "id": str(doc["id"]),
                "title": doc["title"],
                "label": doc["title"],
                "url": doc["url"],
                "group": "bookmark",
            }
            for doc in documents
        ]
        edges = []

        faiss_index = faiss.read_index(IKG_INDEX_PATH)

        valid_docs = []
        valid_vectors = []
        for doc in documents:
            try:
                vec = faiss_index.reconstruct(doc["id"])
                valid_docs.append(doc)
                valid_vectors.append(vec)
            except Exception as e:
                logger.warning(
                    f"[API GRAPH WARNING] FAISS 벡터 재구성 실패 -> bookmark_id={doc['id']} | {e}"
                )
                continue

        if valid_vectors:
            vec_matrix = np.array(valid_vectors)
            norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True) + 1e-9
            normalized_matrix = vec_matrix / norms
            sim_matrix = np.dot(normalized_matrix, normalized_matrix.T)

            for i in range(len(valid_docs)):
                for j in range(i + 1, len(valid_docs)):
                    sim_score = float(sim_matrix[i, j])
                    if sim_score >= threshold:
                        edges.append({
                            "source": str(valid_docs[i]["id"]),
                            "target": str(valid_docs[j]["id"]),
                            "value": round(sim_score, 4)
                        })

        return {
            "nodes": nodes,
            "edges": edges,
            "links": edges,
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "vector_count": len(valid_vectors),
                "threshold": threshold,
            },
        }
    except Exception as e:
        logger.error(f"[API GRAPH CRASH] 토폴로지 분석 장애: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="토폴로지 매트릭스 연산 실패")
    finally:
        conn.close()


@app.get("/api/system/indexing/status", status_code=status.HTTP_200_OK)
async def get_indexing_status():
    with indexing_state_lock:
        active_tasks = indexing_active_tasks
    queued_tasks = embedded_task_queue.qsize()
    return {
        "active_tasks": active_tasks,
        "queued_tasks": queued_tasks,
        "idle": active_tasks == 0 and queued_tasks == 0,
    }


@app.post("/api/system/sync", status_code=status.HTTP_200_OK)
async def trigger_infrastructure_integrity_sync():
    try:
        sync_summary = worker_actor.indexer_engine.sync_index_with_database(
            embedder=worker_actor.embedder
        )
        if sync_summary["status"] in ["SYNCHRONIZED", "NO_CHANGE"]:
            logger.info("[API SYSTEM] 디스크 자산 정합성이 확인되었습니다. 메모리 컨텍스트를 동기화합니다.")
            searcher_engine.refresh_context()
        return {"status": "SUCCESS", "metadata": sync_summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("be_api.app:app", host="0.0.0.0", port=8000, reload=False)