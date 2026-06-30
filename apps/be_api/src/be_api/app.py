import asyncio
import os
import psutil  # 호스트 자원 정밀 계측 의존성
import logging
import queue
import sqlite3
import threading
import time
import gc  # 💡 물리 메모리 강제 정리를 위해 명시적으로 포함

from contextlib import asynccontextmanager
import numpy as np
import uvicorn
import faiss
from fastapi import FastAPI, HTTPException, status, Request

from ai_core import HybridSearcher
from ai_core.core.bookmark_scraper import scrape_url_standalone, validate_scraped_bookmark
from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH
from be_api.logger_config import setup_logging
from be_api.schemas import BookmarkCreateRequest, BookmarkIngestRequest, TaskReceiptResponse
from be_api.tasks import EmbeddedInferenceWorker

# 전역 로깅 인프라 엔진 초기화
setup_logging()
logger = logging.getLogger("be_api.app")


# =========================================================================
# [HYBRID RESOURCE BINDING]: 호스트 맞춤형 동적 자원 설정 통제 가드
# =========================================================================
def _configure_adaptive_hardware_bounds():
    try:
        cpu_cores = os.cpu_count() or 2
        vm = psutil.virtual_memory()
        available_gb = vm.available / (1024 ** 3)
        
        # 1. 동시 렌더링 코어 한계선 결정 (저사양 랩탑 사수용)
        calculated_workers = max(1, cpu_cores - 2)
        final_workers = min(calculated_workers, 2)  # 랩탑 임계점 2개 강제 바인딩
        
        # 2. 비동기 큐 가용 수용력(Backlog Capacity) 동적 연산
        # 가용 RAM의 30%를 크롬 탭 버퍼로 간주 (탭 1개당 보수적으로 150MB 산정)
        allocated_buffer = vm.available * 0.3
        tab_cost_threshold = 150 * 1024 * 1024
        calculated_backlog = int(allocated_buffer // tab_cost_threshold)
        
        # 저사양 랩탑 생존을 위한 최소 5개 ~ 시스템 자원 낭비 방지를 위한 최대 30개 제약
        final_backlog_limit = max(5, min(calculated_backlog, 30))
        
        logger.info("====================================================================")
        logger.info(f"[RESOURCE MONITOR] 실행 호스트 물리 사양 계측 완수")
        logger.info(f" -> 탐색된 CPU 코어: {cpu_cores} Cores | 가용 RAM: {available_gb:.2f} GB")
        logger.info(f" -> 동적 바인딩 스크래핑 워커 수: {final_workers}개 스케일 책정")
        logger.info(f" -> 안전 메모리 대기열(Backlog) 한계: {final_backlog_limit}개 장벽 설정")
        logger.info("====================================================================")
        
        return final_workers, final_backlog_limit
    except Exception as e:
        logger.error(f"[RESOURCE MONITOR ERROR] 호스트 사양 감전 실패, 최하 생존 가드 강제 발동: {e}")
        return 2, 5

# 런타임 통제 임계치 런타임 확정
MAX_SCRAPING_WORKERS, MAX_ALLOWED_BACKLOG = _configure_adaptive_hardware_bounds()

# 메모리 폭발을 막기 위한 한계 지동 장치 큐 생성
scraping_task_queue = asyncio.Queue(maxsize=MAX_ALLOWED_BACKLOG)


async def _scraping_queue_consumer_worker(worker_id: int):
    """
    [SAFE CONSUMER WITH COOL-DOWN] 백그라운드에서 오직 비동기 큐만 바라보며,
    호스트 자원이 소화 가능한 속도로 스크래핑 연산을 완벽히 제어합니다.
    """
    logger.info(f"[SCRAPER WORKER #{worker_id}] 부팅 완료. 대기열 관측 개시.")
    while True:
        try:
            target_url, payload_title, payload_content = await scraping_task_queue.get()
            logger.info(f"[SCRAPER WORKER #{worker_id}] 작업 자산 인계 수령 ──► URL: {target_url}")
            
            scraped_title, scraped_content = await scrape_url_standalone(target_url)
            
            final_title = scraped_title or payload_title or target_url
            final_content = scraped_content or payload_content or ""

            is_valid_content, validation_result = validate_scraped_bookmark(final_title, final_content)
            if is_valid_content:
                scraped_content = validation_result
                isolation_reason = ""
            else:
                isolation_reason = validation_result
            
            if is_valid_content and len(scraped_content or "") < 100:
                is_valid_content = False
                isolation_reason = "TEXT_LENGTH_INSUFFICIENT_UNDER_100"

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
                    else:
                        cursor.execute(
                            """
                            INSERT INTO bookmarks_isolated (url, title, content, created_at, isolation_reason)
                            VALUES (?, ?, ?, datetime('now', 'localtime'), ?)
                            """,
                            (target_url, final_title, final_content, isolation_reason)
                        )
                        conn.commit()
                except sqlite3.IntegrityError:
                    logger.warning(f"[DB INTEGRITY] 이미 상주 중인 고유 URL 자산 건너뜀: {target_url}")
                finally:
                    conn.close()
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[SCRAPER WORKER #{worker_id} ERROR] 태스크 처리 중 실패: {str(e)}", exc_info=True)
        finally:
            scraping_task_queue.task_done()
            # Chromium 가상 메모리가 OS 커널에 반환되도록 최소한의 시차 보장
            await asyncio.sleep(0.5)


@asynccontextmanager
async def IKG_lifespan_handler(app: FastAPI):
    worker_tasks = []
    for i in range(MAX_SCRAPING_WORKERS):
        task = asyncio.create_task(_scraping_queue_consumer_worker(worker_id=i+1))
        worker_tasks.append(task)
    yield
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)

app = FastAPI(title="IKG Hybrid Search Gateway", version="1.0.0", lifespan=IKG_lifespan_handler)

# =========================================================================
# [MIDDLEWARE LOGGING INFRASTRUCTURE]
# =========================================================================
from starlette.middleware.base import BaseHTTPMiddleware

class UbiquitousLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        path = request.url.path
        method = request.method
        body_str = ""
        if method in ["POST", "PUT", "PATCH"] and "application/json" in request.headers.get("content-type", ""):
            body_bytes = await request.body()
            body_str = body_bytes.decode("utf-8", errors="ignore")
            async def receive(): return {"type": "http.request", "body": body_bytes, "more_body": False}
            request._receive = receive

        logger.info(f"▶▶▶ [HTTP REQ] {method} {path} | Payload: {body_str if body_str else 'None'}")
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(f"❌❌❌ [CRASH] {method} {path} 실패 사유: {str(exc)}", exc_info=True)
            raise exc

        process_time = (time.time() - start_time) * 1000
        response_body = b""
        if response.status_code != 500 and "application/json" in response.headers.get("content-type", ""):
            async for chunk in response.body_iterator: response_body += chunk
            response.body_iterator = AsyncBytesIterator([response_body])

        resp_str = response_body.decode("utf-8", errors="ignore")
        if len(resp_str) > 1000: resp_str = resp_str[:1000] + f" ... [Truncated, {len(resp_str)} Bytes]"
        logger.info(f"◀◀◀ [HTTP RESP] {method} {path} | Status: {response.status_code} | Latency: {process_time:.2f}ms")
        return response

class AsyncBytesIterator:
    def __init__(self, chunks: list[bytes]): self._chunks = chunks; self._idx = 0
    def __aiter__(self): return self
    async def __anext__(self):
        if self._idx >= len(self._chunks): raise StopAsyncIteration
        chunk = self._chunks[self._idx]; self._idx += 1; return chunk

app.add_middleware(UbiquitousLoggingMiddleware)

# 코어 엔진 자원 바인딩
searcher_engine = HybridSearcher()
embedded_task_queue = queue.Queue()
worker_actor = EmbeddedInferenceWorker(db_path=IKG_DB_PATH, index_path=IKG_INDEX_PATH)

_db_write_lock = asyncio.Lock()
indexing_state_lock = threading.Lock()
indexing_active_tasks = 0

# =========================================================================
# 💡 [CORE REFACTORING]: 인덱싱 연속 추론 연산 제동 장치 탑재 루프
# =========================================================================
def _embedded_queue_consumer_loop():
    global indexing_active_tasks
    logger.info("[EMBEDDED BUS] 단일 스레드 비동기 직렬화 컨텍스트 소비 루프 가동 완료.")
    while True:
        task_item = embedded_task_queue.get()
        task_started = False
        try:
            if task_item is None: break
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
            logger.error(f"[INDEX BUS CRITICAL ERROR] 내장 큐 인덱싱 실패: {str(e)}", exc_info=True)
        finally:
            if task_started:
                with indexing_state_lock: indexing_active_tasks -= 1
            embedded_task_queue.task_done()
            
            # -----------------------------------------------------------------
            # 💡 [CRITICAL FIX]: ONNX C++ 힙 메모리 파편 소거 및 스레드 무조건적 1초 대기
            # - 딥러닝 인퍼런스 텐서 연산 찌꺼기를 RAM에서 즉각 강제 소거하고,
            #   OS 커널이 메모리 페이지를 완전히 비울 수 있는 하드웨어적 숨통을 열어줍니다.
            # -----------------------------------------------------------------
            gc.collect()
            time.sleep(1.0)  # 데몬 스레드 루프이므로 time.sleep으로 제어권 완전 이관

threading.Thread(target=_embedded_queue_consumer_loop, daemon=True).start()


# =========================================================================
# [ROUTER - CREATE]: 하드웨어 사양 연동형 장벽 제어 인제스션 레이어
# =========================================================================
@app.post("/api/bookmarks", response_model=TaskReceiptResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_bookmark_endpoint(payload: BookmarkIngestRequest):
    target_url = str(payload.url)
    
    if scraping_task_queue.full():
        logger.warning(f"[ADAPTIVE REJECTION] 호스트 메모리 보호를 위한 인입 차단 조치 실행 -> 임계치: {MAX_ALLOWED_BACKLOG}개")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="서버 자원 수호를 위해 실시간 수집 대기열 인입이 제한되었습니다. 잠시 후 재시도하십시오."
        )
    
    await scraping_task_queue.put((target_url, payload.title, payload.content))
    current_queued_size = scraping_task_queue.qsize()

    return TaskReceiptResponse(
        message="요청이 호스트 리소스 보호 대기열에 안전하게 접수되었습니다. 백그라운드에서 순차 처리됩니다.",
        bookmark_id=-1,
        task_id=f"QUEUED-TASK-{current_queued_size}"
    )


# =========================================================================
# [OTHER ROUTERS - READ/UPDATE/DELETE]
# =========================================================================
@app.put("/api/bookmarks/{bookmark_id}", status_code=status.HTTP_200_OK)
async def update_bookmark_endpoint(bookmark_id: int, payload: BookmarkCreateRequest):
    async with _db_write_lock:
        conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM bookmarks WHERE id = ?", (bookmark_id,))
            if not cursor.fetchone(): raise HTTPException(status_code=404, detail="지재 대상이 실존하지 않습니다.")
            cursor.execute("UPDATE bookmarks SET title = ?, content = ?, url = ? WHERE id = ?", (payload.title, payload.content, str(payload.url), bookmark_id))
            conn.commit()
            embedded_task_queue.put({"action": "REINDEX", "id": bookmark_id})
            return {"status": "SUCCESS", "updated_id": bookmark_id}
        finally: conn.close()

@app.delete("/api/bookmarks/{bookmark_id}", status_code=status.HTTP_200_OK)
async def delete_bookmark_endpoint(bookmark_id: int):
    async with _db_write_lock:
        conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM bookmarks WHERE id = ?", (bookmark_id,))
            if not cursor.fetchone(): raise HTTPException(status_code=404, detail="삭제 대상이 부재합니다.")
            cursor.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))
            conn.commit()
            embedded_task_queue.put({"action": "DELETE", "id": bookmark_id})
            return {"status": "SUCCESS", "purged_id": bookmark_id}
        finally: conn.close()

@app.get("/api/search")
async def search_bookmarks_endpoint(q: str | None = None, query: str | None = None, limit: int = 5):
    effective_query = q or query
    if not effective_query or not effective_query.strip(): return []
    try:
        raw_results = searcher_engine.search(query=effective_query, top_n=limit)
        formatted_list = []
        conn = sqlite3.connect(IKG_DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
        try:
            for doc in raw_results:
                target_id = doc.get("id") or doc.get("doc_id")
                if target_id is None: continue
                cursor.execute("SELECT title, url FROM bookmarks WHERE id = ?", (int(target_id),))
                db_row = cursor.fetchone()
                formatted_list.append({
                    "id": int(target_id), "title": db_row["title"] if db_row else doc.get("title", ""),
                    "url": db_row["url"] if db_row else doc.get("url", ""), "score": round(float(doc.get("score", 0.0)), 4),
                    "content": doc.get("content", ""), "score_lex_raw": round(float(doc.get("score_lex_raw", 0.0)), 4), "score_sem_raw": round(float(doc.get("score_sem_raw", 0.0)), 4)
                })
        finally: conn.close()
        return formatted_list
    except Exception as e: raise HTTPException(status_code=500, detail="검색 실패")

@app.get("/api/graph")
async def get_knowledge_graph_matrix_endpoint(threshold: float = 0.85):
    conn = sqlite3.connect(IKG_DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, title, url FROM bookmarks WHERE is_deleted = 0")
        rows = cursor.fetchall(); documents = [{"id": r["id"], "title": r["title"], "url": r["url"]} for r in rows]
        nodes = [{"id": str(d["id"]), "title": d["title"], "label": d["title"], "url": d["url"], "group": "bookmark"} for d in documents]
        edges = []
        faiss_index = faiss.read_index(IKG_INDEX_PATH); valid_docs, valid_vectors = [], []
        for doc in documents:
            try: vec = faiss_index.reconstruct(doc["id"]); valid_docs.append(doc); valid_vectors.append(vec)
            except Exception: continue
        if valid_vectors:
            vec_matrix = np.array(valid_vectors); norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True) + 1e-9
            normalized_matrix = vec_matrix / norms; sim_matrix = np.dot(normalized_matrix, normalized_matrix.T)
            for i in range(len(valid_docs)):
                for j in range(i + 1, len(valid_docs)):
                    sim_score = float(sim_matrix[i, j])
                    if sim_score >= threshold: edges.append({"source": str(valid_docs[i]["id"]), "target": str(valid_docs[j]["id"]), "value": round(sim_score, 4)})
        return {"nodes": nodes, "edges": edges, "links": edges}
    except Exception as e: raise HTTPException(status_code=500, detail="매트릭스 연산 실패")
    finally: conn.close()

@app.get("/api/system/indexing/status")
async def get_indexing_status():
    with indexing_state_lock: active_tasks = indexing_active_tasks
    return {
        "active_tasks": active_tasks,
        "queued_tasks": embedded_task_queue.qsize(),
        "scraping_backlog_size": scraping_task_queue.qsize()
    }

if __name__ == "__main__":
    uvicorn.run("be_api.app:app", host="0.0.0.0", port=8000, reload=False)