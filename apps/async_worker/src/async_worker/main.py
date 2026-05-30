import os
import sqlite3
import uuid
import queue
import threading
import numpy as np
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from async_worker.schemas import BookmarkCreateRequest, TaskReceiptResponse

# ai_core 작업공간 패키지 내부 절대경로 임포트 바인딩
from hybrid_search import HybridSearcher

# 런타임 큐 모드 감지 (기본값: EMBEDDED / 확장 옵션: CELERY)
QUEUE_MODE = os.getenv("QUEUE_MODE", "EMBEDDED")

app = FastAPI(title="IKG Intelligent Backend API Gateway")

# 1. 프론트엔드 크로스 도메인 보안 통신을 위한 CORS 미들웨어 통합 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "chrome-extension://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "db/ikg_metadata.db"
INDEX_PATH = "db/ikg_vector.index"
MODEL_PATH = "./model/bge-m3-onnx-int8"

# 2. 실시간 동기식 검색 엔진 싱글톤 초기화 (서버 기동 시점 웜업 및 베이스라인 캐싱)
searcher_engine = HybridSearcher(
    db_path=DB_PATH,
    index_path=INDEX_PATH,
    model_path=MODEL_PATH
)

# 3. 비동기 작업 계층 (Pluggable Task Queue Layer) 초기화 및 분기
if QUEUE_MODE == "EMBEDDED":
    from async_worker.tasks import EmbeddedInferenceWorker
    
    # 단일 프로세스 내 백압 통제용 무한 루프 스레드 및 메모리 큐 선언
    embedded_task_queue = queue.Queue()
    worker_instance = EmbeddedInferenceWorker()
    
    def _queue_worker_loop():
        print("[EMBEDDED QUEUE] 로컬 독립형 스레드 큐 루프가 가동되었습니다. (concurrency=1)")
        while True:
            try:
                bookmark_id = embedded_task_queue.get()
                if bookmark_id is None:
                    break
                # 자원 독점형 AI 추론 및 FAISS 직렬 쓰기 수행
                worker_instance.execute_inference_pipeline(bookmark_id)
            except Exception as e:
                print(f"[EMBEDDED QUEUE ERROR] 백그라운드 스레드 파이프라인 장애: {e}")
            finally:
                embedded_task_queue.task_done()

    # 메인 웹서버 수명 주기에 종속되는 데몬 스레드 구동
    threading.Thread(target=_queue_worker_loop, daemon=True).start()
else:
    # CELERY 모드 활성화 시 분산 큐 태스크 엔트리포인트 바인딩
    from async_worker.tasks import process_new_bookmark


@app.post(
    "/api/bookmarks", 
    response_model=TaskReceiptResponse, 
    status_code=status.HTTP_202_ACCEPTED,
    summary="신규 북마크 접수 및 비동기 파이프라인 라우팅"
)
def create_bookmark(request: BookmarkCreateRequest, background_tasks: BackgroundTasks):
    # SQLite WAL 모드 락 분쟁 방지를 위한 timeout=30.0 초 마진 확보
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    
    try:
        # 동기 레이어: 가벼운 메타데이터 선적재 (PENDING 상태 확보)
        cursor.execute(
            """
            INSERT INTO bookmarks (url, title, content, created_at) 
            VALUES (?, ?, ?, datetime('now', 'localtime'))
            """,
            (str(request.url), request.title, request.content)
        )
        inserted_id = cursor.lastrowid
        conn.commit()
        
        # 큐 아키텍처 옵션에 따른 가상 큐 라우팅
        if QUEUE_MODE == "EMBEDDED":
            generated_task_id = f"local-task-{uuid.uuid4()}"
            # FastAPI BackgroundTasks 훅을 통해 응답 스트림 배출 즉시 큐에 태스크 인입
            background_tasks.add_task(embedded_task_queue.put, inserted_id)
            msg = "북마크 메타데이터 접수 완료. [내장 스레드 큐(Embedded)]를 통해 로컬 직렬 인덱싱이 시작됩니다."
        else:
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


@app.get("/api/search", summary="v3 최종안 코어 기반 실시간 하이브리드 검색")
def search_bookmarks_endpoint(q: str, limit: int = 5):
    """[Out-of-Scope 영역] 초고속 실시간 UX 수호를 위해 실시간 동기식 랭킹 연산 후 즉각 배출"""
    if not q.strip():
        raise HTTPException(status_code=400, detail="검색 질의어가 비어있습니다.")
    try:
        results = searcher_engine.search(query=q, top_n=limit)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"하이브리드 검색 랭킹 연산 중 장애 발생: {str(e)}"
        )


@app.get("/api/graph", summary="FAISS 벡터 공간 유사도 기반 동적 노드/엣지 토폴로지 추출")
def get_similarity_graph_topology(threshold: float = 0.65):
    """
    오버엔지니어링(가짜 관계 데이터 테이블)을 배제하고, 물리적 벡터 인덱스를 실시간으로 
    역파싱하여 코사인 유사도가 임계치 이상인 자산들만 노드와 엣지로 동적 결합합니다.
    """
    try:
        documents = searcher_engine.documents
        faiss_index = searcher_engine.index
        total_count = len(documents)
        
        if total_count == 0:
            return {"nodes": [], "edges": []}
            
        nodes = [{"id": str(doc["id"]), "title": doc["title"], "url": doc["url"]} for doc in documents]
        edges = []
        
        # FAISS 인덱스로부터 밀집 벡터 복원 및 다차원 유사도 행렬 동기 연산
        vectors = np.array([faiss_index.reconstruct(i) for i in range(total_count)]).astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        normalized_vectors = vectors / norms
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        # 상삼각 행렬 순회를 통한 고유 결합 쌍 추출
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"그래프 시맨틱 토폴로지 분석 실패: {str(e)}"
        )