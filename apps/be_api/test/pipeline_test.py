import json
import urllib.request
import urllib.error
import time
import random



GATEWAY_URL = "http://localhost:8000/api/bookmarks"

# 1. 테스트용 고해상도 테크니컬 더미 데이터셋 정의
# 정상 케이스(유효 데이터) 및 에러 유발 케이스(Pydantic 차단 타겟) 혼합 구성
DUMMY_DATASET = [
    # [정상 케이스 1: 일반 기술 문서]
    {
        "type": "VALID",
        "payload": {
            "url": "https://fastapi.tiangolo.com/advanced/async-backend/",
            "title": "FastAPI Async Concurrency Architecture",
            "content": "Deep dive into ASGI implementation, starlette routing loops, and non-blocking background task threading pools."
        }
    },
    # [정상 케이스 2: 대량 텍스트 본문]
    {
        "type": "VALID",
        "payload": {
            "url": "https://celery-project.org/docs/runtime-configuration/",
            "title": "Celery Distributed Task Queue Tuning",
            "content": "Optimizing prefetched multiplier, worker max tasks per child to mitigate heavy inference memory leaks, and serialization protocols."
        }
    },
    # [정상 케이스 3: 고유명사 및 코드 맥락]
    {
        "type": "VALID",
        "payload": {
            "url": "https://faiss.ai/vector-index-io-locking/",
            "title": "FAISS Index Thread-safe Write Operations",
            "content": "Analyzing binary file persistence, index.add serialization under single concurrency workers, and atomic disk flush mechanisms."
        }
    },
    # [예외 케이스 1: Pydantic 스키마 위반 - 잘못된 URL 포맷]
    {
        "type": "INVALID_URL",
        "payload": {
            "url": "not-a-valid-http-url-string",
            "title": "Malicious Malformed URL Payload",
            "content": "This request should be blocked immediately at the FastAPI gateway layer before reaching the Redis message broker."
        }
    },
    # [예외 케이스 2: Pydantic 스키마 위반 - 제목 공백 누락]
    {
        "type": "INVALID_TITLE",
        "payload": {
            "url": "https://github.com/sw-Joe/MachineLearningPrac",
            "title": "",
            "content": "Testing minimal length validation constraints inside custom validation model schemas."
        }
    },
    # [예외 케이스 3: Pydantic 스키마 위반 - 본문 5자 미만 미달]
    {
        "type": "INVALID_CONTENT",
        "payload": {
            "url": "https://huggingface.co/models",
            "title": "Short Content Bug",
            "content": "Fail"  # 5자 미만으로 세팅하여 벨리데이션 차단 유도
        }
    }
]


def send_post_request(data: dict):
    """FastAPI 게이트웨이 스트림으로 HTTP POST 전송"""
    req_body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        GATEWAY_URL, 
        data=req_body, 
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=5.0) as response:
            res_body = response.read().decode("utf-8")
            return response.status, json.loads(res_body)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(error_body)
        except:
            return e.code, error_body
    except Exception as e:
        return 999, str(e)


def execute_pipeline_stress_test(bulk_count=20):
    print("=" * 60)
    print(" [IKG-Pipeline] 비동기 추론 인프라 스트레스 & 벨리데이션 테스트")
    print("=" * 60)
    print(f"[SYSTEM] 대량 트래픽 백압 시뮬레이션을 시작합니다. (발행 태스크 수: {bulk_count}건)")
    
    success_cnt = 0
    blocked_cnt = 0
    
    for i in range(bulk_count):
        # 더미 데이터셋에서 무작위 추출하여 인입 교란 시뮬레이션
        case = random.choice(DUMMY_DATASET)
        case_type = case["type"]
        payload = case["payload"]
        
        # 순차적인 ID 추적 및 락 테스트를 위해 타이틀 명세에 인덱스 인입 변조
        if case_type == "VALID":
            payload_copy = payload.copy()
            payload_copy["title"] = f"{payload_copy['title']} #{i+1}"
        else:
            payload_copy = payload
            
        status_code, response = send_post_request(payload_copy)
        
        if status_code == 202:
            success_cnt += 1
            print(f"[{i+1:02d}] [✓ ACCEPTED] Type: {case_type} | Bookmark ID: {response.get('bookmark_id')} | Task ID: {response.get('task_id')[:8]}...")
        elif status_code == 422:
            blocked_cnt += 1
            print(f"[{i+1:02d}] [X VALIDATION BLOCKED] Type: {case_type} | Status: 422 Unprocessable Entity")
        else:
            print(f"[{i+1:02d}] [! SYSTEM CRITICAL] Status: {status_code} | Response: {response}")
            
        # 대용량 트래픽이 동시다발적으로 밀려오는 스케일 징후를 재현하기 위해 간격을 매우 좁게 세팅
        time.sleep(0.05)

    print("=" * 60)
    print(" [TEST PIPELINE INFRASTRUCTURE SUMMARY]")
    print("=" * 60)
    print(f" * 게이트웨이 통과 및 큐 이관 완료 (202 Accepted): {success_cnt} 건")
    print(f" * Pydantic 사전 검전기 차단 완료 (422 Blocked) : {blocked_cnt} 건")
    print("=" * 60)
    print("[INFO] Celery 워커 터미널 콘솔을 모니터링하여 AI 추론 및 FAISS 직렬 쓰기가 정상 완결되는지 검증하십시오.")


if __name__ == "__main__":
    # 인프라 컴포넌트(FastAPI, Redis)가 정상 구동 중인 환경에서 실행 유도
    execute_pipeline_stress_test(bulk_count=30)