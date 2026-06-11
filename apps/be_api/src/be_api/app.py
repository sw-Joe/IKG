import asyncio
import logging
import queue
import sqlite3
import threading

import numpy as np
import uvicorn
import faiss
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ai_core import HybridSearcher
from ai_core.core.bookmark_scraper import scrape_url_standalone, validate_scraped_bookmark
from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH
from be_api.logger_config import setup_logging
from be_api.schemas import BookmarkCreateRequest, TaskReceiptResponse
from be_api.tasks import EmbeddedInferenceWorker



setup_logging()
logger = logging.getLogger("be_api.app")

app = FastAPI(title="IKG Hybrid Search Gateway", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

searcher_engine = HybridSearcher()
embedded_task_queue = queue.Queue()
worker_actor = EmbeddedInferenceWorker(db_path=IKG_DB_PATH, index_path=IKG_INDEX_PATH)


def _embedded_queue_consumer_loop():
    logger.info("[EMBEDDED BUS] 단일 스레드 비동기 직렬화 컨텍스트 소비 루프 가동 완료.")
    while True:
        try:
            task_item = embedded_task_queue.get()
            if task_item is None:
                break
            action = task_item.get("action", "ADD")
            bookmark_id = task_item.get("id")

            if action == "ADD":
                worker_actor.execute_sequential_inference_pipeline(bookmark_id)
            elif action == "DELETE":
                worker_actor.execute_sequential_removal_pipeline(bookmark_id)
            embedded_task_queue.task_done()
        except Exception as e:
            logger.error(f"[EMBEDDED BUS CRITICAL ERROR] 내장 큐 파이프라인 작동 실패: {str(e)}", exc_info=True)

consumer_thread = threading.Thread(target=_embedded_queue_consumer_loop, daemon=True)
consumer_thread.start()


@app.get("/api/graph")
def get_knowledge_graph_matrix_endpoint(threshold: float = 0.85):
    conn = sqlite3.connect(IKG_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, title, url FROM bookmarks WHERE is_deleted = 0")
        rows = cursor.fetchall()
        documents = [{"id": r["id"], "title": r["title"], "url": r["url"]} for r in rows]

        total_docs = len(documents)
        nodes = [{"id": str(doc["id"]), "label": doc["title"], "url": doc["url"]} for doc in documents]
        edges = []

        faiss_index = faiss.read_index(IKG_INDEX_PATH)

        valid_docs = []
        valid_vectors = []
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
                        edges.append({
                            "source": str(valid_docs[i]["id"]),
                            "target": str(valid_docs[j]["id"]),
                            "value": round(sim_score, 4)
                        })

        return {"nodes": nodes, "links": edges}
    except Exception as e:
        logger.error(f"[API GRAPH CRASH] 토폴로지 분석 장애: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="토폴로지 매트릭스 연산 실패")
    finally:
        conn.close()


@app.post("/api/system/sync", status_code=status.HTTP_200_OK)
def trigger_infrastructure_integrity_sync():
    """SQLite 파일 - FAISS 인덱스 파일 동기화"""
    try:
        sync_summary = worker_actor.indexer_engine.sync_index_with_database(
            embedder=worker_actor.embedder
        )
        if sync_summary["status"] in ["SYNCHRONIZED", "NO_CHANGE"]:
            logger.info("[API SYSTEM] 디스크 자산 정합성이 확인되었습니다. 메모리 컨텍스트를 동기화합니다.")
            searcher_engine.refresh_context()
        return {"status": "SUCCESS", "metadata": sync_summary}
    except Exception as e:
        logger.critical(f"시스템 동기화 붕괴: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bookmarks", response_model=TaskReceiptResponse, status_code=status.HTTP_201_CREATED)
def create_bookmark_endpoint(payload: BookmarkCreateRequest, background_tasks: BackgroundTasks):
    """북마크 인입(추가) 요청"""
    target_url = str(payload.url)
    logger.info(f"[API POST] 단건 북마크 실시간 인입 수신 -> 수집 프로세스 기동: {target_url}")
    
    # 1. AnyIO 스레드 컨텍스트 내부에서 Playwright 비동기 크롬 커널 안전 인보크
    scraped_title, scraped_content = asyncio.run(scrape_url_standalone(target_url))
    
    # 2. 실시간 크롤링 결과에 대한 고밀도 정보 유효성 가드라인 교차 검증 집행
    is_valid_content, isolation_reason = validate_scraped_bookmark(scraped_title, scraped_content)
    
    # 길이가 100자 미만인 하한선 하드웨어 제약도 보완 검증 처리
    if is_valid_content and len(scraped_content) < 100:
        is_valid_content = False
        isolation_reason = "TEXT_LENGTH_INSUFFICIENT_UNDER_100"

    # 최종 저장 뼈대 확정 (실시간 크롤링 완료본이 우선하며, 누락 시 페이로드 백업 승계)
    final_title = scraped_title or payload.title
    final_content = scraped_content or payload.content

    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    try:
        if is_valid_content:
            # Case A: 실시간 스크래핑 및 지식 가치 검증 완벽 통과 -> 메인 고밀도 테이블 적재
            cursor.execute(
                """
                INSERT INTO bookmarks (url, title, content, created_at, is_deleted, index_written)
                VALUES (?, ?, ?, datetime('now', 'localtime'), 0, 0)
                """,
                (target_url, final_title, final_content)
            )
            assigned_id = cursor.lastrowid
            conn.commit()

            # 실시간 비동기 임베딩 인덕션 버스 큐 진입 명령 트리거
            embedded_task_queue.put({"action": "ADD", "id": assigned_id})

            return TaskReceiptResponse(
                message="실시간 웹 스크래핑 및 정합성 검증 통과 완료. 백그라운드 벡터 차원 공간 적재 프로세스를 기동합니다.",
                bookmark_id=assigned_id,
                task_id=f"TASK-ADD-{assigned_id}"
            )
        else:
            # Case B: 웹 방화벽 차단, 404, 혹은 본문 훼손 자산 -> 보류 격리 샌드박스 테이블 적재 (FAISS 인큐잉 영구 격리 차단)
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


# [CRUD: UPDATE] 자산 수정 라우터 추가
@app.put("/api/bookmarks/{bookmark_id}", status_code=status.HTTP_200_OK)
def update_bookmark_endpoint(bookmark_id: int, payload: BookmarkCreateRequest):
    """북마크 수정 요청"""
    logger.info(f"[API PUT] 자산 수정 요청 수신 -> ID: #{bookmark_id}")

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

        # 벡터 인덱스 갱신을 위해 삭제 후 재등록 프로세스 트리거 (비동기 처리)
        embedded_task_queue.put({"action": "DELETE", "id": bookmark_id})
        embedded_task_queue.put({"action": "ADD", "id": bookmark_id})

        return {"status": "SUCCESS", "updated_id": bookmark_id}
    finally:
        conn.close()


@app.put("/api/bookmarks/recover/{isolated_id}", status_code=status.HTTP_202_ACCEPTED)
def recover_isolated_bookmark_endpoint(isolated_id: int, payload: BookmarkCreateRequest):
    """격리 테이블로 격리된 비정상 북마크의 테이블 정정 복구"""
    logger.info(f"[API PUT] 정정 복구 트랜잭션 개시 -> 대상 isolated_id: #{isolated_id}")
    
    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT url, created_at FROM bookmarks_isolated WHERE id = ?", (isolated_id,))
        row = cursor.fetchone()
        if not row:
            logger.warning(f"[API PUT WARNING] 격리 자산 식별자 탐색 실패 -> ID: #{isolated_id}")
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
            logger.info(f" -> [DB MIGRATION SUCCESS] 격리ID #{isolated_id} ──► 메인 최하단 ID #{new_main_id} 이관 수렴")
        except sqlite3.IntegrityError as ie:
            conn.rollback()
            logger.error(f"[API PUT INTEGRITY ERROR] URL 중복 제약 충돌 발생: {ie}")
            raise HTTPException(status_code=400, detail="정정 복구하려는 URL 자산이 이미 메인 자산 테이블에 존재합니다.")
        except Exception as e:
            conn.rollback()
            logger.error(f"[API PUT CRITICAL] 원자적 트랜잭션 내부 붕괴: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"데이터베이스 원자적 트랜잭션 통합 붕괴: {str(e)}")

        # 비동기 직렬화 인퍼런스 버스 인큐잉
        embedded_task_queue.put({"action": "ADD", "id": new_main_id})

        # ---------------------------------------------------------------------
        # 💡 [DEBUG LOGGING INJECT]: 최종 출력 직전 고해상도 가시성 로깅 가드 구축
        # ---------------------------------------------------------------------
        updated_node_snapshot = {
            "id": new_main_id,
            "title": payload.title,
            "url": original_url,
            "score": 1.0
        }
        
        # 1차원 리스트 형태를 상정한 포맷 빌드
        final_payload_array = [updated_node_snapshot]
        
        # 로그 출력 세션 개시
        logger.info("====================================================================")
        logger.info("[DEBUG-MONITOR] PUT 엔드포인트 반환 직전 원시 데이터 정밀 계측")
        logger.info(f" -> 최종 페이로드 타입(Type): {type(final_payload_array)}")
        logger.info(f" -> 최종 페이로드 데이터(Raw): {final_payload_array}")
        
        # 만약 특수 확장 구조를 사용 중이라면 딕셔너리 속성(Dict Keys)까지 투명하게 출력
        if hasattr(final_payload_array, "__dict__"):
            logger.info(f" -> 숨겨진 객체 프로퍼티 속성 축(Keys): {final_payload_array.__dict__.keys()}")
        logger.info("====================================================================")

        return final_payload_array
        
    finally:
        conn.close()


@app.delete("/api/bookmarks/{bookmark_id}", status_code=status.HTTP_200_OK)
def delete_bookmark_endpoint(bookmark_id: int):
    """북마크 삭제"""
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


@app.get("/api/search")
def search_bookmarks_endpoint(q: str | None = None, query: str | None = None, limit: int = 5):
    """북마크 검색"""
    effective_query = q or query
    logger.info(f"[API GET SEARCH] 하이브리드 지식 검색 세션 진입 -> 질의어: '{effective_query}'")
    
    if not effective_query or not effective_query.strip():
        return []

    try:
        # 하부 하이브리드 정렬 코어에서 순수 랭킹 자산 배열 획득
        raw_results = searcher_engine.search(query=effective_query, top_n=limit)
        
        # [MINIMALIST REFACTORING]: 오버엔지니어링 래퍼를 전면 소거하고
        # FE 자바스크립트 가상 돔 루프(.map()) 가드가 즉시 인지 가능한 정형 1차원 리스트로 반환
        formatted_list = []
        for doc in raw_results:
            formatted_list.append({
                "id": int(doc["id"]),
                "title": doc["title"],
                "content": doc.get("content", ""),
                "url": doc["url"],
                "score": round(float(doc["score"]), 4),
                "score_lex_raw": round(float(doc.get("score_lex_raw", 0.0)), 4),
                "score_sem_raw": round(float(doc.get("score_sem_raw", 0.0)), 4)
            })
            
        return formatted_list
        
    except Exception as e:
        logger.error(f"[API SEARCH ERROR] 하이브리드 검색 컨텍스트 연산 장애: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="시맨틱 하이브리드 공간 검색 정렬 붕괴")


if __name__ == "__main__":
    uvicorn.run("be_api.app:app", host="0.0.0.0", port=8000, reload=False)