import logging
import queue
import sqlite3
import threading

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ai_core import HybridSearcher
from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH
from be_api.logger_config import setup_logging
from be_api.schemas import BookmarkCreateRequest, TaskReceiptResponse
from be_api.tasks import EmbeddedInferenceWorker

# 1. 고해상도 공통 로깅 아키텍처 파이프라인 수립 가동
setup_logging()
logger = logging.getLogger("be_api.app")

app = FastAPI(title="IKG Intelligent Hybrid Search Gateway", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. CQRS Read 전용 검색 코어 엔진 싱글톤 로드
searcher_engine = HybridSearcher()

# 3. 내장형 독립 스레드 큐 버퍼 및 순차 직렬화 워커 액터 구동 (Backpressure 완전 제어)
embedded_task_queue = queue.Queue()
worker_actor = EmbeddedInferenceWorker(db_path=IKG_DB_PATH, index_path=IKG_INDEX_PATH)

def _embedded_queue_consumer_loop():
    logger.info("[EMBEDDED BUS] 단일 스레드 비동기 직렬화 컨텍스트 소비 루프 가동 완료. (Concurrency=1)")
    while True:
        bookmark_id = embedded_task_queue.get()
        if bookmark_id is None:
            break
        try:
            worker_actor.execute_upsert_pipeline(bookmark_id)
        except Exception as e:
            logger.critical(f"[EMBEDDED BUS CRITICAL] 백그라운드 태스크 무효화 크래시 가드: {e}", exc_info=True)
        finally:
            embedded_task_queue.task_done()

# 데몬 스레드로 백그라운드 상시 가동
threading.Thread(target=_embedded_queue_consumer_loop, daemon=True).start()


# =========================================================================
# [CREATE] 신규 기술 자산 비동기 격리 인입
# =========================================================================
@app.post("/api/bookmarks", status_code=status.HTTP_202_ACCEPTED, response_model=TaskReceiptResponse)
def create_bookmark_endpoint(payload: BookmarkCreateRequest, background_tasks: BackgroundTasks):
    logger.info(f"[API POST] 신규 북마크 입고 요청 수신 -> 타깃 URL: {payload.url}")
    
    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    try:
        # 이중 색인 방지를 위해 1차 데이터베이스 제약 검증 및 선적재 (기본 활성 상태 0 부여)
        cursor.execute(
            "INSERT INTO bookmarks (url, title, content, created_at, is_deleted) VALUES (?, ?, ?, datetime('now', 'localtime'), 0)",
            (str(payload.url), payload.title, payload.content)
        )
        inserted_id = cursor.lastrowid
        conn.commit()
        
        # 영수증 발행 프로세스 직후 무거운 임베딩 태스크를 백그라운드 내장 큐로 즉시 이관
        background_tasks.add_task(embedded_task_queue.put, inserted_id)
        logger.info(f"[API POST SUCCESS] 메타데이터 DB 선적재 종결 -> 발급 식별자: #{inserted_id}")
        
        return TaskReceiptResponse(
            message="북마크 메타데이터 선적재가 완료되었습니다. 비동기 인덱싱이 백그라운드에서 실행됩니다.",
            bookmark_id=inserted_id,
            task_id=f"task-cr-050-{inserted_id}"
        )
    except sqlite3.IntegrityError:
        # [FIXED]: 고유 제약 조건 위배(중복 URL) 시 정제된 에러 피드백 반환 가드 주입
        logger.warning(f"[API POST CONFLICT] 고유 제약 조건 위배 무효화 조치 -> 중복 URL 인입: {payload.url}")
        raise HTTPException(status_code=400, detail="이미 시스템에 등록 영속화된 고유 URL 자산입니다.")
    finally:
        conn.close()


# =========================================================================
# [UPDATE] 기술 자산 정보 수정 및 비동기 벡터 영속 재지정
# =========================================================================
@app.put("/api/bookmarks/{bookmark_id}", status_code=status.HTTP_202_ACCEPTED, response_model=TaskReceiptResponse)
def update_bookmark_endpoint(bookmark_id: int, payload: BookmarkCreateRequest, background_tasks: BackgroundTasks):
    """[FIXED]: 구조적으로 누락되었던 실시간 단건 수정 파이프라인 인터페이스 전격 체결"""
    logger.info(f"[API PUT] 데이터 정정 트랜잭션 수신 -> 대상 ID: #{bookmark_id}")
    
    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM bookmarks WHERE id = ? AND is_deleted = 0", (bookmark_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="수정하려는 대상 기술 자산이 존재하지 않거나 활성 상태가 아닙니다.")
            
        # 1단계: SQLite 원문 메타데이터 정정 집행
        cursor.execute(
            "UPDATE bookmarks SET url = ?, title = ?, content = ? WHERE id = ? AND is_deleted = 0",
            (str(payload.url), payload.title, payload.content, bookmark_id)
        )
        conn.commit()
        
        # 2단계: 중복 누적 방지(remove_ids) 및 재임베딩 유도를 위해 가상 내장 워커 큐에 식별자 위임
        background_tasks.add_task(embedded_task_queue.put, bookmark_id)
        logger.info(f"[API PUT SUCCESS] 메타데이터 수정 커밋 완료 -> 워커 큐 이관 완료 ID: #{bookmark_id}")
        
        return TaskReceiptResponse(
            message="북마크 메타데이터 수정이 완료되었습니다. 벡터 공간 갱신이 백그라운드에서 가동됩니다.",
            bookmark_id=bookmark_id,
            task_id=f"task-up-050-{bookmark_id}"
        )
    finally:
        conn.close()


# =========================================================================
# [DELETE] 논리 소거 오퍼레이션 (Soft-Delete)
# =========================================================================
@app.delete("/api/bookmarks/{bookmark_id}")
def delete_bookmark_endpoint(bookmark_id: int):
    logger.info(f"[API DELETE] 자산 제거 인터페이스 시그널 감지 -> 대상 ID: #{bookmark_id}")
    
    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM bookmarks WHERE id = ?", (bookmark_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="삭제 소거하려는 자산이 실존하지 않습니다.")
            
        # Soft-Delete 플래그 갱신 (is_deleted=1)
        cursor.execute("UPDATE bookmarks SET is_deleted = 1 WHERE id = ?", (bookmark_id,))
        conn.commit()
        
        # 사용자가 체감하는 즉시 정합성(Immediate Consistency) 사수를 위해 인메모리 조회 풀 강제 리프레시
        searcher_engine.refresh_context()
        logger.info(f"[API DELETE SUCCESS] #{bookmark_id} 논리 소거 완료 및 실시간 캐시 동기화 종결.")
        
        return {"message": "기술 자산이 성공적으로 소거 처리되었습니다. 스토리지 자원 청소는 비동기 배치로 위임됩니다.", "bookmark_id": bookmark_id}
    finally:
        conn.close()


# =========================================================================
# [READ] 실시간 하이브리드 다차원 검색 및 토폴로지 동적 조율
# =========================================================================
@app.get("/api/search")
def search_bookmarks_endpoint(q: str | None = None, query: str | None = None, limit: int = 5):
    effective_query = q or query
    logger.info(f"[API GET SEARCH] 하이브리드 검색 처리량 인입 -> 확정질의어: '{effective_query}' | Limit: {limit}")
    
    if not effective_query or not effective_query.strip():
        return []
        
    try:
        results = searcher_engine.search(query=effective_query, top_n=limit)
        return results
    except Exception as e:
        logger.error(f"[API SEARCH ERROR] 검색 연산 중 내부 장애: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="하이브리드 랭킹 스케일 계산 연산 장애")


@app.get("/api/graph")
def get_similarity_graph_topology(threshold: float = 0.85):
    logger.info(f"[API GET GRAPH] 공간 시각화 임계 행렬 매트릭스 추출 요청 -> Threshold: {threshold}")
    
    if threshold < 0.50:
        threshold = 0.50

    try:
        documents = searcher_engine.documents
        faiss_index = searcher_engine.index
        total_count = len(documents)
        
        if total_count == 0:
            return {"nodes": [], "edges": []}
            
        nodes = [{"id": str(doc["id"]), "title": doc["title"], "url": doc["url"]} for doc in documents]
        edges = []
        
        # [CRITICAL BUG FIX]: IndexIDMap 체제에 맞춰 순차 루프 인덱스 i가 아닌 실제 도큐먼트의 고유 정수 PK를 넘겨 공간 복원하도록 정정
        vectors = np.array([faiss_index.reconstruct(int(doc["id"])) for doc in documents]).astype("float32")
        
        # 정규화 및 상호 코사인 유사도 연산 매트릭스 도출
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        normalized_vectors = vectors / norms
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        for i in range(total_count):
            for j in range(i + 1, total_count):
                sim_score = float(similarity_matrix[i][j])
                if sim_score >= threshold:
                    edges.append({
                        "source": str(documents[i]["id"]),
                        "target": str(documents[j]["id"]),
                        "value": round(sim_score, 4)
                    })
                    
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        logger.error(f"[API GRAPH CRASH] 토폴로지 분석 행렬 붕괴: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="기하학적 공간 토폴로지 매트릭스 분석 연산 실패")


# =========================================================================
# [SYSTEM] 캐시 무효화 및 강제 전역 동기화 (Stale Cache 시차 결함 해결)
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
        logger.critical(f"시스템 동기화 복원 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"인프라 동기화 복구 파이프라인 연산 실패: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("be_api.app:app", host="0.0.0.0", port=8000, reload=True)