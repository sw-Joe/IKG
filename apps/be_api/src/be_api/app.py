import os
import sqlite3
import uuid
import queue
import threading
import numpy as np
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# 수선된 ai_core 중앙 허브 구성 상수 및 코어 엔진 임포트
from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH, IKG_MODEL_PATH
from ai_core.hybrid_search import HybridSearcher
from be_api.schemas import BookmarkCreateRequest, TaskReceiptResponse

QUEUE_MODE = os.getenv("QUEUE_MODE", "EMBEDDED")

app = FastAPI(title="IKG Intelligent Backend API Gateway")

# 크로스 오리진 보안 통신 해제를 위한 CORS 미들웨어 통합 레이어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "chrome-extension://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 중앙 격리 설정 상수 바인딩
DB_PATH = IKG_DB_PATH
INDEX_PATH = IKG_INDEX_PATH
MODEL_PATH = IKG_MODEL_PATH

print(f"\n[IKG API STARTUP] 백엔드 게이트웨이 서비스 가동 (중앙 설정 동기화 완결)")

# 실시간 동기식 지능형 검색 엔진 인스턴스 싱글톤 웜업
searcher_engine = HybridSearcher(
    db_path=DB_PATH,
    index_path=INDEX_PATH,
    model_path=MODEL_PATH
)

# 비동기 인프라 스레드 소비자 큐 초기화 분기
if QUEUE_MODE == "EMBEDDED":
    from be_api.tasks import EmbeddedInferenceWorker
    
    embedded_task_queue = queue.Queue()
    worker_instance = EmbeddedInferenceWorker()
    
    def _queue_worker_loop():
        print("[EMBEDDED QUEUE] 단일 프로세스 독립형 스레드 직렬화 큐가 시작되었습니다. (concurrency=1)")
        while True:
            try:
                bookmark_id = embedded_task_queue.get()
                if bookmark_id is None:
                    break
                # 가중 연산 파이프라인 가동
                worker_instance.execute_inference_pipeline(bookmark_id)
            except Exception as e:
                print(f" ❌ [QUEUE ERROR EVENT] 백그라운드 태스크 워커 런타임 장애: {e}")
            finally:
                embedded_task_queue.task_done()

    threading.Thread(target=_queue_worker_loop, daemon=True).start()
else:
    from be_api.tasks import process_new_bookmark


@app.post("/api/bookmarks", response_model=TaskReceiptResponse, status_code=status.HTTP_202_ACCEPTED)
def create_bookmark(request: BookmarkCreateRequest, background_tasks: BackgroundTasks):
    print(f"\n[GATEWAY TRAFFIC] 신규 기술 문서 인입 등록 요청 수신")
    print(f" - 대상 URL: {request.url}")
    print(f" - 메타 제목: {request.title}")
    
    # 스레드 로컬 안전 단발성 커넥션 처리
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO bookmarks (url, title, content, created_at) VALUES (?, ?, ?, datetime('now', 'localtime'))",
            (str(request.url), request.title, request.content)
        )
        inserted_id = cursor.lastrowid
        conn.commit()
        
        if QUEUE_MODE == "EMBEDDED":
            generated_task_id = f"local-task-{uuid.uuid4()}"
            background_tasks.add_task(embedded_task_queue.put, inserted_id)
            msg = "[Embedded] 파이썬 내장 스레드 세이프 메모리 큐로 인덱싱 태스크가 직렬 인입 위임되었습니다."
        else:
            task_receipt = process_new_bookmark.delay(inserted_id)
            generated_task_id = task_receipt.id
            msg = "[Celery] 메시지 브로커 분산 인프라 레이어로 비동기 위임이 접수되었습니다."
            
        print(f"  -> SQLite 동기 적재 완료. 할당 행 고유식별 번호: #{inserted_id}")
        
        return TaskReceiptResponse(
            message=msg,
            bookmark_id=inserted_id,
            task_id=generated_task_id
        )
    except Exception as e:
        conn.rollback()
        print(f"  ❌ [TRANSACTION CRITICAL ROLLBACK] 메타데이터 적재 예외 실패 복구: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/api/search", summary="v3 최종안 동적 어텐션 랭커 기반 실시간 하이브리드 검색")
def search_bookmarks_endpoint(q: str, limit: int = 5):
    print(f"\n[GATEWAY TRAFFIC] 실시간 지식 베이스 하이브리드 검색 질의 수신")
    print(f" - 검색어: '{q}' | 추출 슬롯 제한 한계: {limit}건")
    
    if not q.strip():
        raise HTTPException(status_code=400, detail="검색 질의어가 비어있습니다.")
    try:
        # [핵심 수선 사항: 실시간 캐시 동기화 가드]
        # 검색 API가 호출되는 즉시, 워커 스레드가 디스크에 쓴 최신 바이트 스냅샷을 
        # 메인 웹서버 스레드 메모리 상으로 실시간 리로드하여 완벽한 실시간 정합성 확정
        searcher_engine.reload_indices()
            
        # v3 최종 융합 검색 연산 파이프라인 기동
        results = searcher_engine.search(query=q, top_n=limit)
        
        print(f"  -> 하이브리드 검색 결과 반환 완료 (수량: {len(results)}건)")
        return results
    except Exception as e:
        print(f"  ❌ [INFERENCE CRASH] 랭킹 추론 연산 중 치명적 장애: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"하이브리드 검색 랭킹 연산 중 장애 발생: {str(e)}"
        )


@app.get("/api/graph", summary="FAISS 벡터 공간 유사도 기반 동적 노드/엣지 토폴로지 추출")
def get_similarity_graph_topology(threshold: float = 0.85):
    print(f"\n[GATEWAY TRAFFIC] 네트워크 시각화 전용 실시간 벡터 엣지 토폴로지 추출 요청")
    try:
        searcher_engine.reload_indices()
        documents = searcher_engine.documents
        faiss_index = searcher_engine.index
        total_count = len(documents)
        
        print(f" - 분석 대상 활성 노드 수: {total_count}개")
        
        if total_count == 0:
            return {"nodes": [], "edges": []}
            
        nodes = [{"id": str(doc["id"]), "title": doc["title"], "url": doc["url"]} for doc in documents]
        edges = []
        
        # FAISS 벡터 대수 행렬 변환 및 코사인 내적 고속 정렬
        vectors = np.array([faiss_index.reconstruct(i) for i in range(total_count)]).astype("float32")
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
                    
        print(f"  -> 벡터 공간 그래프 수렴 완결 (노드 {len(nodes)}개, 연결선 {len(edges)}개)")
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        print(f"  ❌ [GRAPH COMPILER CRASH] 행렬 내적 연산 예외 장애: {e}")
        raise HTTPException(status_code=500, detail=f"그래프 토폴로지 분석 실패: {str(e)}")