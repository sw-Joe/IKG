import asyncio
import logging
import queue
import sqlite3
import threading
import time
from contextlib import asynccontextmanager

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from ai_core import HybridSearcher
from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH
from ai_core.core.bookmark_scraper import (
    scrape_url_standalone,
    validate_scraped_bookmark,
)
from be_api.logger_config import setup_logging
from be_api.schemas import (
    BookmarkCreateRequest,
    BookmarkIngestRequest,
    TaskReceiptResponse,
)
from be_api.tasks import EmbeddedInferenceWorker

# 1. 인프라 전역 로깅 인프라 초기화
setup_logging()
logger = logging.getLogger("be_api.app")

# ---------------------------------------------------------------------
# [ON-DEVICE RESOURCE GUARD]: 인프로세스 스크래핑 비동기 큐 명세
# ---------------------------------------------------------------------
# 버스트 트래픽 유입 시 메모리 코루틴 포화를 막기 위한 완충 버퍼 큐
scraping_task_queue = asyncio.Queue()

# 랩탑 물리 자원 한계를 고려하여 동시 가동할 백그라운드 크롬 워커 수 엄격 제약
MAX_SCRAPING_WORKERS = 2


async def _scraping_queue_consumer_worker(worker_id: int):
    """
    [SAFE CONSUMER] 백그라운드에서 오직 비동기 큐만 바라보며,
    호스트 자원이 소화 가능한 속도로 스크래핑 연산을 철저히 순차 제어합니다.
    """
    logger.info(f"[SCRAPER WORKER #{worker_id}] 부팅 완료. 대기열 관측을 시작합니다.")
    while True:
        try:
            # 큐에 작업이 들어올 때까지 대기 (CPU를 소모하지 않는 완전 비동기 서스펜드)
            target_url, payload_title, payload_content = await scraping_task_queue.get()
            logger.info(
                f"[SCRAPER WORKER #{worker_id}] 작업 자산 인계 수령 ──► URL: {target_url}"
            )

            # STAGE 1: 플레이라이트 싱글톤 브라우저 격리 탭 패칭 집행
            scraped_title, scraped_content = await scrape_url_standalone(target_url)

            final_title = scraped_title or payload_title or target_url
            final_content = scraped_content or payload_content or ""

            # STAGE 2: 정합성 및 정보 가치 검증 필터링
            is_valid_content, validation_result = validate_scraped_bookmark(
                final_title, final_content
            )
            if is_valid_content:
                scraped_content = validation_result
                isolation_reason = ""
            else:
                isolation_reason = validation_result

            if is_valid_content and len(scraped_content or "") < 100:
                is_valid_content = False
                isolation_reason = "TEXT_LENGTH_INSUFFICIENT_UNDER_100"

            # STAGE 3: SQLite 데이터베이스 트랜잭션 원자적 직렬화 격리 진입
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
                            (target_url, final_title, final_content),
                        )
                        assigned_id = cursor.lastrowid
                        cursor.execute(
                            "DELETE FROM bookmarks_isolated WHERE url = ?",
                            (target_url,),
                        )
                        conn.commit()

                        # 인덱싱 전용 백그라운드 스레드 큐로 바톤 터치
                        embedded_task_queue.put({"action": "ADD", "id": assigned_id})
                        logger.info(
                            f"[SCRAPER WORKER #{worker_id}] 메인 자산 적재 및 인덱싱 큐 인큐 완수 -> ID: #{assigned_id}"
                        )
                    else:
                        cursor.execute(
                            """
                            INSERT INTO bookmarks_isolated (url, title, content, created_at, isolation_reason)
                            VALUES (?, ?, ?, datetime('now', 'localtime'), ?)
                            """,
                            (target_url, final_title, final_content, isolation_reason),
                        )
                        conn.commit()
                        logger.info(
                            f"[SCRAPER WORKER #{worker_id}] 가치 미달 자산 격리 처리 완수 -> 사유: {isolation_reason}"
                        )
                except sqlite3.IntegrityError:
                    logger.warning(
                        f"[DB INTEGRITY GUARD] 이미 처리 중이거나 상주 중인 고유 URL 자산 건너뜀: {target_url}"
                    )
                finally:
                    conn.close()

        except asyncio.CancelledError:
            logger.info(
                f"[SCRAPER WORKER #{worker_id}] 스케줄러 취소 시그널 수신 -> 루프 소멸"
            )
            break
        except Exception as e:
            logger.error(
                f"[SCRAPER WORKER #{worker_id} CRITICAL] 태스크 처리 중 예외 발생: {str(e)}",
                exc_info=True,
            )
        finally:
            scraping_task_queue.task_done()


# =========================================================================
# [LIFESPAN]: FastAPI 최신 명세 표준 자원 생명주기 관리자
# =========================================================================
@asynccontextmanager
async def app_lifespan_handler(app: FastAPI):
    # [STARTUP SEQUENCE]
    worker_tasks = []
    for i in range(MAX_SCRAPING_WORKERS):
        task = asyncio.create_task(_scraping_queue_consumer_worker(worker_id=i + 1))
        worker_tasks.append(task)

    logger.info(
        f"[LIFESPAN INIT] 총 {MAX_SCRAPING_WORKERS}개의 온디바이스 인프로세스 스크래핑 가드 워커 스케줄링 등록 완수."
    )

    yield

    # [SHUTDOWN SEQUENCE]
    logger.info(
        "[LIFESPAN SHUTDOWN] 게이트웨이 셧다운 감지 -> 좀비 크롬 방어 시퀀스 가동."
    )
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    logger.info(
        "[LIFESPAN SHUTDOWN COMPLETE] 모든 백그라운드 태스크 안전 커밋 및 자원 반환 종료."
    )


# FastAPI 인스턴스 초기화 바인딩
app = FastAPI(
    title="IKG Hybrid Search Gateway", version="0.9.0", lifespan=app_lifespan_handler
)


# =========================================================================
# [INFRASTRUCTURE]: 전역 엔드포인트 입출력 추적 고해상도 로깅 미들웨어
# =========================================================================
class UbiquitousLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        path = request.url.path
        method = request.method

        body_str = ""
        if method in [
            "POST",
            "PUT",
            "PATCH",
        ] and "application/json" in request.headers.get("content-type", ""):
            body_bytes = await request.body()
            body_str = body_bytes.decode("utf-8", errors="ignore")

            async def receive():
                return {"type": "http.request", "body": body_bytes, "more_body": False}

            request._receive = receive

        logger.info(
            f"▶▶▶ [HTTP REQ Ingress] {method} {path} | Payload: {body_str if body_str else 'None'}"
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                f"❌❌❌ [INTERNAL CRASH] {method} {path} 연산 실패 사유: {str(exc)}",
                exc_info=True,
            )
            raise exc

        process_time = (time.time() - start_time) * 1000

        response_body = b""
        if response.status_code != 500 and "application/json" in response.headers.get(
            "content-type", ""
        ):
            async for chunk in response.body_iterator:
                response_body += chunk
            response.body_iterator = AsyncBytesIterator([response_body])

        resp_str = response_body.decode("utf-8", errors="ignore")
        if len(resp_str) > 1000:
            resp_str = (
                resp_str[:1000]
                + f" ... [Truncated, Total Length: {len(resp_str)} Bytes]"
            )

        logger.info(
            f"◀◀◀ [HTTP RESP Egress] {method} {path} | Status: {response.status_code} "
            f"| Latency: {process_time:.2f}ms | Out-Data: {resp_str if resp_str else 'Empty'}"
        )
        return response


class AsyncBytesIterator:
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

# =========================================================================
# [CORE COMPONENTS RUNTIME COUPLING]
# =========================================================================
searcher_engine = HybridSearcher()
embedded_task_queue = queue.Queue()
worker_actor = EmbeddedInferenceWorker(db_path=IKG_DB_PATH, index_path=IKG_INDEX_PATH)

_db_write_lock = asyncio.Lock()
indexing_state_lock = threading.Lock()
indexing_active_tasks = 0


def _embedded_queue_consumer_loop():
    global indexing_active_tasks
    logger.info(
        "[EMBEDDED BUS] 단일 스레드 비동기 직렬화 컨텍스트 소비 루프 가동 완료."
    )
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
            logger.error(
                f"[EMBEDDED BUS CRITICAL ERROR] 내장 큐 인덱싱 실패: {str(e)}",
                exc_info=True,
            )
        finally:
            if task_started:
                with indexing_state_lock:
                    indexing_active_tasks -= 1
            embedded_task_queue.task_done()


threading.Thread(target=_embedded_queue_consumer_loop, daemon=True).start()


# =========================================================================
# [ROUTER - CREATE]: 초고속 즉각 수렴형 인프로세스 큐 접수 엔드포인트
# =========================================================================
@app.post(
    "/api/bookmarks",
    response_model=TaskReceiptResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_bookmark_endpoint(payload: BookmarkIngestRequest):
    target_url = str(payload.url)

    # 무거운 크롤링 연산을 배후로 밀어내고 메모리 버퍼 큐에 바이트 링크 형태로 적재 (나노초 처리 완료)
    await scraping_task_queue.put((target_url, payload.title, payload.content))
    current_queued_size = scraping_task_queue.qsize()

    logger.info(
        f"[BUFFERING INGEST] 온디바이스 안전 대기열 적재 완수 -> URL: {target_url} (대기열 잔여: {current_queued_size}개)"
    )

    return TaskReceiptResponse(
        message="요청이 호스트 리소스 보호 대기열에 안전하게 접수되었습니다. 백그라운드에서 순차 처리됩니다.",
        bookmark_id=-1,
        task_id=f"QUEUED-TASK-{current_queued_size}",
    )


# =========================================================================
# [ROUTER - UPDATE]: 지식 데이터 자산 정정 수정 레이어
# =========================================================================
@app.put("/api/bookmarks/{bookmark_id}", status_code=status.HTTP_200_OK)
async def update_bookmark_endpoint(bookmark_id: int, payload: BookmarkCreateRequest):
    async with _db_write_lock:
        conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM bookmarks WHERE id = ?", (bookmark_id,))
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=404, detail="수정할 대상 자산이 존재하지 않습니다."
                )

            cursor.execute(
                "UPDATE bookmarks SET title = ?, content = ?, url = ? WHERE id = ?",
                (payload.title, payload.content, str(payload.url), bookmark_id),
            )
            conn.commit()
            embedded_task_queue.put({"action": "REINDEX", "id": bookmark_id})

            return {"status": "SUCCESS", "updated_id": bookmark_id}
        finally:
            conn.close()


# =========================================================================
# [ROUTER - RECOVER]: 격리 보류 자산 수동 정정 승격 트랜잭션
# =========================================================================
@app.put("/api/bookmarks/recover/{isolated_id}", status_code=status.HTTP_202_ACCEPTED)
async def recover_isolated_bookmark_endpoint(
    isolated_id: int, payload: BookmarkCreateRequest
):
    async with _db_write_lock:
        conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT url, created_at FROM bookmarks_isolated WHERE id = ?",
                (isolated_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise HTTPException(
                    status_code=404, detail="지정된 격리 자산 식별자가 부재합니다."
                )
            original_url, original_created_at = row

            cursor.execute("BEGIN TRANSACTION;")
            try:
                cursor.execute(
                    """
                    INSERT INTO bookmarks (url, title, content, created_at, is_deleted, index_written)
                    VALUES (?, ?, ?, ?, 0, 0)
                    """,
                    (original_url, payload.title, payload.content, original_created_at),
                )
                new_main_id = cursor.lastrowid
                cursor.execute(
                    "DELETE FROM bookmarks_isolated WHERE id = ?", (isolated_id,)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                conn.rollback()
                raise HTTPException(
                    status_code=400,
                    detail="정정 복구하려는 URL 자산이 이미 메인 테이블에 상주 중입니다.",
                )
            except Exception as e:
                conn.rollback()
                raise HTTPException(
                    status_code=500, detail=f"데이터베이스 원자성 붕괴: {str(e)}"
                )

            embedded_task_queue.put({"action": "ADD", "id": new_main_id})

            return [
                {
                    "id": new_main_id,
                    "title": payload.title,
                    "url": original_url,
                    "score": 1.0,
                }
            ]
        finally:
            conn.close()


# =========================================================================
# [ROUTER - DELETE]: 완전 삭제 및 동기화 소거 레이어
# =========================================================================
@app.delete("/api/bookmarks/{bookmark_id}", status_code=status.HTTP_200_OK)
async def delete_bookmark_endpoint(bookmark_id: int):
    async with _db_write_lock:
        conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM bookmarks WHERE id = ?", (bookmark_id,))
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=404, detail="삭제 대상 지식 자산이 부재합니다."
                )

            cursor.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))
            conn.commit()
            embedded_task_queue.put({"action": "DELETE", "id": bookmark_id})
            return {"status": "SUCCESS", "purged_id": bookmark_id}
        finally:
            conn.close()


# =========================================================================
# [ROUTER - READ SEARCH]: 하이브리드 시맨틱 자산 정렬 랭킹 엔드포인트
# =========================================================================
@app.get("/api/search")
async def search_bookmarks_endpoint(
    q: str | None = None, query: str | None = None, limit: int = 5
):
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

                cursor.execute(
                    "SELECT title, url FROM bookmarks WHERE id = ?", (target_id,)
                )
                db_row = cursor.fetchone()

                formatted_list.append(
                    {
                        "id": target_id,
                        "title": db_row["title"]
                        if db_row
                        else doc.get("title", f"지식 자산 #{target_id}"),
                        "url": db_row["url"] if db_row else doc.get("url", ""),
                        "score": score_val,
                        "content": doc.get("content", ""),
                        "score_lex_raw": round(float(doc.get("score_lex_raw", 0.0)), 4),
                        "score_sem_raw": round(float(doc.get("score_sem_raw", 0.0)), 4),
                    }
                )
        finally:
            conn.close()
        return formatted_list
    except Exception as e:
        logger.error(f"[API SEARCH ERROR] 검색 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="시맨틱 하이브리드 공간 정렬 붕괴")


# =========================================================================
# [ROUTER - READ GRAPH]: 3D 시각화 공간 토폴로지 데이터 연산 엔드포인트
# =========================================================================
@app.get("/api/graph")
async def get_knowledge_graph_matrix_endpoint(threshold: float = 0.85):
    conn = sqlite3.connect(IKG_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, title, url FROM bookmarks WHERE is_deleted = 0")
        rows = cursor.fetchall()
        documents = [
            {"id": r["id"], "title": r["title"], "url": r["url"]} for r in rows
        ]

        nodes = [
            {
                "id": str(d["id"]),
                "title": d["title"],
                "label": d["title"],
                "url": d["url"],
                "group": "bookmark",
            }
            for d in documents
        ]
        edges = []

        faiss_index = faiss.read_index(IKG_INDEX_PATH)
        valid_docs, valid_vectors = [], []

        for doc in documents:
            try:
                vec = faiss_index.reconstruct(doc["id"])
                valid_docs.append(doc)
                valid_vectors.append(vec)
            except Exception:
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
                        edge_obj = {
                            "source": str(valid_docs[i]["id"]),
                            "target": str(valid_docs[j]["id"]),
                            "value": round(sim_score, 4),
                        }
                        edges.append(edge_obj)

        return {"nodes": nodes, "edges": edges, "links": edges}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"토폴로지 행렬 연산 실패: {str(e)}"
        )
    finally:
        conn.close()


@app.get("/api/system/indexing/status")
async def get_indexing_status():
    with indexing_state_lock:
        active_tasks = indexing_active_tasks
    return {
        "active_tasks": active_tasks,
        "queued_tasks": embedded_task_queue.qsize(),
        "scraping_backlog_size": scraping_task_queue.qsize(),
    }


if __name__ == "__main__":
    uvicorn.run("be_api.app:app", host="0.0.0.0", port=8000, reload=False)
