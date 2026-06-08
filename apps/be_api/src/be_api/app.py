import os
import queue
import threading
import sqlite3
import logging
from typing import Optional

import uvicorn
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from be_api.logger_config import setup_logging
from be_api.schemas import BookmarkCreateRequest, TaskReceiptResponse
from be_api.tasks import EmbeddedInferenceWorker
from ai_core import HybridSearcher
from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH


# =========================================================================
# [BOOTSTRAP] 인프라 자동 프로비저닝 레이어 (최상단 이동 - 앱 최초 구동 결함 방어)
# =========================================================================
def bootstrap_infrastructure():
    """외부 명령 의존성 없이, 최초 가동 시 저장소 디렉토리와 필수 파일 스키마를 원자적 자동 생성"""
    # 1. 파일 상위 디렉토리 존재성 강제 보장
    os.makedirs(os.path.dirname(IKG_DB_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(IKG_INDEX_PATH), exist_ok=True)
    
    # 2. SQLite 메타데이터 스토리지 뼈대 및 최신 Soft-Delete 복합 인덱스 빌드
    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            content TEXT,
            created_at TIMESTAMP,
            is_deleted INTEGER DEFAULT 0
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bookmarks_active ON bookmarks (id, is_deleted);")
        conn.commit()
    finally:
        conn.close()

    # 3. FAISS IndexIDMap 규격 물리 파일 부재 시 즉시 초기화 플러시 (Read 계층 크래시 완전 방어)
    if not os.path.exists(IKG_INDEX_PATH):
        # BGE-M3 Dense 임베딩 차원(1024) 규격에 맞춘 빈 IndexIDMap2 바이너리 선제 빌드
        sub_index = faiss.IndexFlatIP(1024)
        id_map_index = faiss.IndexIDMap2(sub_index)
        faiss.write_index(id_map_index, IKG_INDEX_PATH)

# 싱글톤 컴포넌트 평가 전 인프라 부트스트래핑 선제 강제 집행 (최우선 순위 격상)
bootstrap_infrastructure()


# =========================================================================
# [INITIALIZATION] 전역 서비스 및 로깅 파이프라인 수립
# =========================================================================
setup_logging()
logger = logging.getLogger("be_api.app")
logger.info("[BOOTSTRAP] 로컬 스토리지 인프라 무결성 검증 및 컴파일 완결.")

app = FastAPI(title="IKG Intelligent Unified Backend Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. CQRS Read 전용 하이브리드 검색 싱글톤 엔진 로드 (인프라가 완비된 후 안전하게 바인딩)
searcher_engine = HybridSearcher()

# 5. 내장형 비동기 직렬화 태스크 큐 버퍼 및 워커 런타임 결합
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
            searcher_engine.refresh_context()
            logger.info(f"[EMBEDDED BUS Sync] #{bookmark_id} 자산 색인 완결에 따른 가상 메모리 리프레시 동기화 완료.")
        except Exception as e:
            logger.critical(f"[EMBEDDED BUS CRITICAL] 백그라운드 워커 크래시 가드 작동: {e}", exc_info=True)
        finally:
            embedded_task_queue.task_done()

threading.Thread(target=_embedded_queue_consumer_loop, daemon=True).start()


# =========================================================================
# [CREATE] 신규 기술 자산 비동기 격리 인입
# =========================================================================
@app.post("/api/bookmarks", status_code=status.HTTP_202_ACCEPTED, response_model=TaskReceiptResponse)
def create_bookmark_endpoint(payload: BookmarkCreateRequest, background_tasks: BackgroundTasks):
    logger.info(f"[API POST] 신규 북마크 입고 요청 -> URL: {payload.url}")
    
    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO bookmarks (url, title, content, created_at, is_deleted) VALUES (?, ?, ?, datetime('now', 'localtime'), 0)",
            (str(payload.url), payload.title, payload.content)
        )
        inserted_id = cursor.lastrowid
        conn.commit()
        
        background_tasks.add_task(embedded_task_queue.put, inserted_id)
        logger.info(f"[API POST SUCCESS] DB 선적재 완료 -> 발급 ID: #{inserted_id}")
        
        return TaskReceiptResponse(
            message="북마크 메타데이터 선적재가 완료되었습니다. 비동기 인덱싱이 백그라운드에서 실행됩니다.",
            bookmark_id=inserted_id,
            task_id=f"task-c-{inserted_id}"
        )
    except sqlite3.IntegrityError:
        logger.warning(f"[API POST CONFLICT] 중복 URL 유입 유효성 차단: {payload.url}")
        raise HTTPException(status_code=400, detail="이미 시스템에 등록 영속화된 고유 URL 자산입니다.")
    finally:
        conn.close()


# =========================================================================
# [READ] 실시간 하이브리드 다차원 검색 및 토폴로지 동적 조율
# =========================================================================
@app.get("/api/search")
def search_bookmarks_endpoint(q: Optional[str] = None, query: Optional[str] = None, limit: int = 5):
    effective_query = q or query
    logger.info(f"[API GET SEARCH] 검색 쿼리 수신: '{effective_query}' | Limit: {limit}")
    
    if not effective_query or not effective_query.strip():
        return []
        
    try:
        return searcher_engine.search(query=effective_query, top_n=limit)
    except Exception as e:
        logger.error(f"[API SEARCH ERROR] 하이브리드 랭킹 스케일 연산 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="하이브리드 유사도 정렬 계산 연산 장애")


@app.get("/api/graph")
def get_similarity_graph_topology(threshold: float = 0.85):
    logger.info(f"[API GET GRAPH] 공간 토폴로지 추출 요청 -> Threshold: {threshold}")
    
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
        
        vectors = np.array([faiss_index.reconstruct(int(doc["id"])) for doc in documents]).astype("float32")
        
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
        logger.error(f"[API GRAPH CRASH] 기하학적 공간 행렬 분석 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="벡터 다차원 공간 그래프 컴파일 연산 실패")


# =========================================================================
# [UPDATE] 기술 자산 정보 수정 및 비동기 벡터 영속 재지정
# =========================================================================
@app.put("/api/bookmarks/{bookmark_id}", status_code=status.HTTP_202_ACCEPTED, response_model=TaskReceiptResponse)
def update_bookmark_endpoint(bookmark_id: int, payload: BookmarkCreateRequest, background_tasks: BackgroundTasks):
    logger.info(f"[API PUT] 단건 기술 자산 정보 수정 및 재색인 오퍼레이션 인입 -> ID: #{bookmark_id}")
    
    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM bookmarks WHERE id = ? AND is_deleted = 0", (bookmark_id,))
        if not cursor.fetchone():
            logger.warning(f"[API PUT FAIL] 존재하지 않거나 소거된 대상에 대한 수정 시도 차단 -> ID: #{bookmark_id}")
            raise HTTPException(status_code=404, detail="수정하려는 대상 기술 자산이 존재하지 않거나 이미 삭제되었습니다.")
            
        cursor.execute(
            "UPDATE bookmarks SET url = ?, title = ?, content = ? WHERE id = ?",
            (str(payload.url), payload.title, payload.content, bookmark_id)
        )
        conn.commit()
        
        background_tasks.add_task(embedded_task_queue.put, bookmark_id)
        logger.info(f"[API PUT SUCCESS] 메타데이터 수정 트랜잭션 성공 완료. 비동기 벡터 동기화를 시작합니다 -> ID: #{bookmark_id}")
        
        return TaskReceiptResponse(
            message="북마크 메타데이터 수정이 성공적으로 완료되었습니다. 백그라운드에서 벡터 재색인이 집행됩니다.",
            bookmark_id=bookmark_id,
            task_id=f"task-u-{bookmark_id}"
        )
    finally:
        conn.close()


# =========================================================================
# [DELETE] 논리 소거 오퍼레이션 (Soft-Delete)
# =========================================================================
@app.delete("/api/bookmarks/{bookmark_id}")
def delete_bookmark_endpoint(bookmark_id: int):
    logger.info(f"[API DELETE] 자산 소거 요청 수신 -> 대상 ID: #{bookmark_id}")
    
    conn = sqlite3.connect(IKG_DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM bookmarks WHERE id = ?", (bookmark_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="삭제하려는 기술 자산의 원천 데이터가 실존하지 않습니다.")
            
        cursor.execute("UPDATE bookmarks SET is_deleted = 1 WHERE id = ?", (bookmark_id,))
        conn.commit()
        
        searcher_engine.refresh_context()
        logger.info(f"[API DELETE SUCCESS] #{bookmark_id} 자산 논리 삭제 완료 및 실시간 조회 풀 무효화 종결.")
        
        return {"message": "기술 자산 소거 처리가 종결되었습니다. 인프라 디스크 청소는 백그라운드 스케줄러로 대리 위임됩니다.", "bookmark_id": bookmark_id}
    finally:
        conn.close()


# =========================================================================
# [SYSTEM] 강제 전역 동기화 커널 (차집합 기반 누락/고립 자산 수렴)
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
        logger.critical(f"[API SYSTEM CRITICAL] 전역 동기화 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"인프라 동기화 복구 파이프라인 연산 실패: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("be_api.app:app", host="0.0.0.0", port=8000, reload=True)