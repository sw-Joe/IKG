import logging
import queue
import sqlite3
import threading
from datetime import datetime

import numpy as np
import uvicorn
import faiss
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ai_core import HybridSearcher
from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH
from be_api.logger_config import setup_logging
from be_api.schemas import BookmarkCreateRequest, TaskReceiptResponse
from be_api.tasks import EmbeddedInferenceWorker

# 1. 모듈형 중앙 집중 로깅 인프라 가동
setup_logging()
logger = logging.getLogger("be_api.app")

app = FastAPI(title="IKG Intelligent Hybrid Search Gateway", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. CQRS Read 전용 검색 코어 엔진 싱글톤 로드
searcher_engine = HybridSearcher()

# 3. Redis/Celery가 배제된 순수 내장형 인메모리 FIFO 큐 및 단일 워커 액터 스레드 수립
embedded_task_queue = queue.Queue()
worker_actor = EmbeddedInferenceWorker(db_path=IKG_DB_PATH, index_path=IKG_INDEX_PATH)


def _embedded_queue_consumer_loop():
    logger.info("[EMBEDDED BUS] 단일 스레드 비동기 직렬화 컨텍스트 소비 루프 가동 완료. (Concurrency=1)")
    while True:
        try:
            task_item = embedded_task_queue.get()
            if task_item is None:
                break

            action = task_item.get("action", "ADD")
            bookmark_id = task_item.get("id")

            if action == "ADD":
                # 순차 임베딩 추론 및 index_written=1 체크포인트 마킹 위임
                worker_actor.execute_sequential_inference_pipeline(bookmark_id)
            elif action == "DELETE":
                # 단건 삭제 트리거 시 FAISS 벡터 공간 소거 위임
                worker_actor.execute_sequential_removal_pipeline(bookmark_id)

            embedded_task_queue.task_done()
        except Exception as e:
            logger.error(f"[EMBEDDED BUS CRITICAL ERROR] 내장 큐 파이프라인 작동 실패: {str(e)}", exc_info=True)


consumer_thread = threading.Thread(target=_embedded_queue_consumer_loop, daemon=True)
consumer_thread.start()


# =========================================================================
# [CRUD: CREATE] 실시간 단건 인입 라우터 (무결성 검증 결과 기반 스키마 분기)
# =========================================================================
@app.post("/api/bookmarks", response_model=TaskReceiptResponse, status_code=status.HTTP_201_CREATED)
def create_bookmark_endpoint(payload: BookmarkCreateRequest, background_tasks: BackgroundTasks):
    logger.info(f"[API POST] 실시간 단건 자산 인입 수신 -> Target URL: {payload.url}")
    
    is_valid_content = True
    isolation_reason = None

    if len(payload.content) < 100:
        is_valid_content = False
        isolation_reason = "TEXT_LENGTH_INSUFFICIENT_UNDER_100"

    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()

    try:
        if is_valid_content:
            # Case A: 무결성 통과 -> 고밀도 메인 테이블 적재 (index_written=0 기본값 마킹)
            cursor.execute(
                """
                INSERT INTO bookmarks (url, title, content, created_at, is_deleted, index_written)
                VALUES (?, ?, ?, datetime('now', 'localtime'), 0, 0)
                """,
                (str(payload.url), payload.title, payload.content)
            )
            assigned_id = cursor.lastrowid
            conn.commit()

            # 비동기 인덱싱 버스 큐 진입 지시
            embedded_task_queue.put({"action": "ADD", "id": assigned_id})

            return TaskReceiptResponse(
                message="정상 자산 수집 및 SQLite 적재 완수. 백그라운드 실시간 텐서 인덱싱 프로세스를 기동합니다.",
                bookmark_id=assigned_id,
                task_id=f"TASK-ADD-{assigned_id}"
            )
        else:
            # Case B: 검증 보류 -> 격리 샌드박스 테이블로 물리 분리 적재 (FAISS 인큐잉 원천 차단)
            cursor.execute(
                """
                INSERT INTO bookmarks_isolated (url, title, content, created_at, isolation_reason)
                VALUES (?, ?, ?, datetime('now', 'localtime'), ?)
                """,
                (str(payload.url), payload.title, payload.content, isolation_reason)
            )
            assigned_id = cursor.lastrowid
            conn.commit()

            return TaskReceiptResponse(
                message=f"[VALIDATION_HOLD] 자산 원문 오염으로 인한 격리 샌드박스 테이블 적재 완수. 사유: {isolation_reason}",
                bookmark_id=assigned_id,
                task_id=f"TASK-ISOLATE-{assigned_id}",
                status="isolated"
            )

    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="이미 인프라 내에 실존 상주 중인 유일 고유 URL 명세입니다.")
    finally:
        conn.close()


# =========================================================================
# [CRUD: UPDATE - RECOVER] 보류 자산 정정 승격 트랜잭션 (최하단 신규 ID 순차 발급)
# =========================================================================
@app.put("/api/bookmarks/recover/{isolated_id}", status_code=status.HTTP_202_ACCEPTED)
def recover_isolated_bookmark_endpoint(isolated_id: int, payload: BookmarkCreateRequest):
    logger.info(f"[API RECOVER] 격리 샌드박스 자산 정정 승격 요청 수신 -> Target 격리 ID: #{isolated_id}")

    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()

    try:
        # 1. 격리 샌드박스 데이터 원천 실존 여부 확인
        cursor.execute("SELECT url, created_at FROM bookmarks_isolated WHERE id = ?", (isolated_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="지정된 격리 자산 식별자가 보류 테이블 내에 부재합니다.")
        original_url, original_created_at = row

        # 2. 원자적 크로스 테이블 트랜잭션 가동
        cursor.execute("BEGIN TRANSACTION;")
        try:
            # [CRITICAL]: 과거 식별자를 파기(NULL)하여 메인 밀집 테이블의 최하단 sequential AUTOINCREMENT ID 새로 발급
            cursor.execute(
                """
                INSERT INTO bookmarks (url, title, content, created_at, is_deleted, index_written)
                VALUES (?, ?, ?, ?, 0, 0)
                """,
                (original_url, payload.title, payload.content, original_created_at)
            )
            new_main_id = cursor.lastrowid

            # 정상 승격 완료되었으므로 격리 테이블 행 영구 소거
            cursor.execute("DELETE FROM bookmarks_isolated WHERE id = ?", (isolated_id,))
            conn.commit()
            logger.info(f" -> [DB MIGRATION COMPLETE] 격리ID #{isolated_id} ──► 메인 최하단 ID #{new_main_id} 이관 수렴")
        except sqlite3.IntegrityError:
            conn.rollback()
            raise HTTPException(status_code=400, detail=f"정정 복구하려는 URL 자산이 이미 메인 자산 테이블에 존재합니다.")
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"데이터베이스 원자적 트랜잭션 통합 붕괴: {str(e)}")

        # 3. 비동기 실시간 텐서 공간 증분 인큐잉 지시
        embedded_task_queue.put({"action": "ADD", "id": new_main_id})

        return {
            "status": "SUCCESS",
            "source_isolated_id": isolated_id,
            "assigned_main_id": new_main_id,
            "message": "메인 테이블 조밀화 이관 성공. 비동기 임베딩 인덱싱 버스 큐에 접수 완료되었습니다."
        }
    finally:
        conn.close()


# =========================================================================
# [CRUD: DELETE] 하이브리드 완전 동기화 소거 라우터
# =========================================================================
@app.delete("/api/bookmarks/{bookmark_id}", status_code=status.HTTP_200_OK)
def delete_bookmark_endpoint(bookmark_id: int):
    logger.info(f"[API DELETE] 메인 지식 자산 완전 삭제 명령 수신 -> ID: #{bookmark_id}")

    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM bookmarks WHERE id = ?", (bookmark_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="삭제하려는 대상 자산이 메인 테이블에 실존하지 않습니다.")

        # SQLite 행 레코드 물리 완전 삭제
        cursor.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))
        conn.commit()

        # FAISS 가상 차원 레이어 동시 물리 소거를 위해 직렬 큐로 명령 위임 패스
        embedded_task_queue.put({"action": "DELETE", "id": bookmark_id})

        return {"status": "SUCCESS", "purged_id": bookmark_id, "message": "물리 데이터 소거 및 벡터 인덱스 동기 삭제 처리 완료."}
    finally:
        conn.close()


# =========================================================================
# [CQRS READ] 하이브리드 시맨틱 검색 게이트웨이 엔드포인트
# =========================================================================
@app.get("/api/search")
def search_bookmarks_endpoint(q: str, limit: int = 5):
    effective_query = q.strip()
    if not effective_query:
        raise HTTPException(status_code=400, detail="공백 질의어에 대한 하이브리드 검색 연산은 집행될 수 없습니다.")

    try:
        results = searcher_engine.search(query=effective_query, top_n=limit)
        return results
    except Exception as e:
        logger.error(f"[API SEARCH ERROR] 검색 연산 중 내부 장애: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"하이브리드 세션 공간 검색 정렬 붕괴: {str(e)}")


# =========================================================================
# [CQRS READ] 3D 글로벌 공간 시각화 토폴로지 추출 레이어
# =========================================================================
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

        for i in range(total_docs):
            for j in range(i + 1, total_docs):
                try:
                    vec_i = faiss_index.reconstruct(i)
                    vec_j = faiss_index.reconstruct(j)

                    norm_i = np.linalg.norm(vec_i) + 1e-9
                    norm_j = np.linalg.norm(vec_j) + 1e-9
                    sim_score = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))

                    if sim_score >= threshold:
                        edges.append({
                            "source": str(documents[i]["id"]),
                            "target": str(documents[j]["id"]),
                            "value": round(sim_score, 4)
                        })
                except Exception:
                    continue

        return {"nodes": nodes, "links": edges}
    except Exception as e:
        logger.error(f"[API GRAPH CRASH] 토폴로지 분석 행렬 붕괴 원인: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="기하학적 공간 토폴로지 매트릭스 분석 연산 실패")
    finally:
        conn.close()


# =========================================================================
# [SYSTEM] 캐시 무효화 및 강제 전역 동기화
# =========================================================================
@app.post("/api/system/sync", status_code=status.HTTP_200_OK)
def trigger_infrastructure_integrity_sync():
    logger.info("[API SYSTEM] 시스템 전역 인프라 스냅샷 강제 동기화 오퍼레이션 명령 수신")
    try:
        sync_summary = worker_actor.indexer_engine.sync_index_with_database(
            embedder=worker_actor.embedder
        )
        if sync_summary["status"] == "SYNCHRONIZED":
            logger.info("[API SYSTEM] 물리 변동성 확인에 따른 조회 엔진 인메모리 캐시 리프레시 트리거")
            searcher_engine.refresh_context()

        return {"status": "SUCCESS", "metadata": sync_summary}
    except Exception as e:
        logger.critical(f"시스템 동기화 복구 붕괴: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("be_api.app:app", host="0.0.0.0", port=8000, reload=True)