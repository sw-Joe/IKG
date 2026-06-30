#!/bin/bash

# ====================================================================
# IKG URL-only custom ingestion stress test (Adaptive Resilience Edition)
# - 클라이언트는 URL만 전달한다.
# - BE가 직접 인프로세스 비동기 큐를 통해 스크래핑/검증/색인을 분리 수행한다.
# ====================================================================

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
BOOKMARK_API_URL="${API_BASE_URL}/api/bookmarks"
INDEXING_STATUS_URL="${API_BASE_URL}/api/system/indexing/status"
SYNC_URL="${API_BASE_URL}/api/system/sync"
INDEX_SETTLE_TIMEOUT_SECONDS="${INDEX_SETTLE_TIMEOUT_SECONDS:-300}"
RESPONSE_DIR="${RESPONSE_DIR:-/tmp/ikg_custom_url_responses}"

# [ON-DEVICE TUNING]: 단일 온디바이스 랩탑의 코어와 메모리 사양을 감안한 제어 스펙
MAX_PARALLEL_REQUESTS="${MAX_PARALLEL_REQUESTS:-5}"
# [CRITICAL FIX]: 요청 분기 간 최소한의 소켓/메모리 GC 숨고르기 시차 주입 (단위: 초)
INJECTION_INTERVAL_DELAY="0.2"

# AI 및 개발지식과 관련된 북마크 다수, 영화, 학교사이트 등 노이즈 
URLS=(
  'https://docs.python.org/3/library/dataclasses.html'
  'https://namu.wiki/w/Redox'
  'https://devocean.sk.com/blog/techBoardDetail.do?ID=164657&boardType=techBlog'
  'https://wikidocs.net/204321'
  'https://namu.wiki/w/%EC%8A%A4%ED%84%B8%EB%A7%81%20%EA%B8%B0%EA%B4%80'
  'https://namu.wiki/w/%EA%B3%B5%EB%8C%80%EA%B0%9C%EA%B7%B8'
  'https://namu.wiki/w/ROM'
  'https://www.kjebi.com/search/smart/bidder/detail/8398801249'
  'https://fastapi.tiangolo.com/advanced/custom-request-and-route/'
  'https://fastapi.tiangolo.com/tutorial/bigger-applications/'
  'https://fastapi.tiangolo.com/tutorial/background-tasks/'
  'https://fastapi.tiangolo.com/advanced/async-tests/'
  'https://github.com/vllm-project/vllm'
  'https://github.com/vllm-project/vllm/issues/1000'
  'https://velog.io/@ssong_m/Docker-Compose%EB%A1%AC-Elasticsearch-%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0-%EA%B5%AC%EC%B6%95%ED%95%98%EA%B8%B0'
  'https://velog.io/@w99096/Redis-Cluster-%EA%B5%AC%EC%B6%95-%EB%B0%8F-%ED%85%85%ED%8A%B8'
  'https://bcho.tistory.com/1309'
  'https://bcho.tistory.com/1314'
  'https://bcho.tistory.com/1317'
  'http://deeplearning.net/software/theano/'
  'https://github.com/TG-WinG/Algorithm'
  'https://learn.microsoft.com/ko-kr/windows/apps/dev-tools/winapp-cli/guides/electron-setup'
  'https://huggingface.co/docs/transformers/en/index'
  'https://developer.furiosa.ai/furiosa-models/latest/'
  'https://arcprize.org/arc-agi/1'
  'https://docs.ollama.com/capabilities/tool-calling'
  'https://velog.io/@jkseo50/DVC-%EA%B0%9C%EB%85%90-%EB%B0%8F-%ED%99%9C%EC%9A%A9-%EB%B0%A9%EB%B2%95'
)

ingest_url() {
  local req_id=$1
  local url=$2
  local out_file="${RESPONSE_DIR}/resp_${req_id}.json"

  local http_code
  http_code=$(curl -s -o "$out_file" -w "%{http_code}" \
    -X POST "$BOOKMARK_API_URL" \
    -H "Content-Type: application/json" \
    -d "{\"url\":\"${url}\"}")

  echo "요청 ${req_id}: HTTP ${http_code} -> ${url}"
}

wait_for_indexing_idle() {
  local elapsed=0
  echo "[IKG TEST] 모든 클라이언트 API 요청이 접수되었습니다. 백그라운드 인덱싱 완료를 대기합니다."
  
  while [ "$elapsed" -lt "$INDEX_SETTLE_TIMEOUT_SECONDS" ]; do
    local status
    status=$(curl -s "$INDEXING_STATUS_URL")
    
    if [ -z "$status" ]; then
      echo "[IKG TEST] 백엔드로부터 응답이 끊겼습니다. 서버의 자원 고갈(OOM) 여부를 즉시 감지하십시오."
      return 1
    fi

    # active_tasks와 queued_tasks 및 scraping_backlog_size가 모두 0인지 동적 파싱 검증
    local active
    local queued
    local backlog
    active=$(echo "$status" | grep -o '"active_tasks":[0-9]*' | cut -d: -f2)
    queued=$(echo "$status" | grep -o '"queued_tasks":[0-9]*' | cut -d: -f2)
    backlog=$(echo "$status" | grep -o '"scraping_backlog_size":[0-9]*' | cut -d: -f2)

    if [ "$active" -eq 0 ] && [ "$queued" -eq 0 ] && [ "$backlog" -eq 0 ]; then
      echo "[IKG TEST] 인덱싱 큐 idle 확인 완료. 시스템의 최종 자산 정합성을 수렴 동기화합니다."
      curl -s -X POST "$SYNC_URL" > /dev/null
      echo "[IKG TEST] 전역 벡터 가상 공간 빌드 정착 성공."
      return 0
    fi

    echo "[IKG TEST] 인덱싱 대기 중... ${status}"
    sleep 3
    elapsed=$((elapsed + 3))
  done

  echo "[IKG TEST] 인덱싱 idle 대기 시간 초과 (${INDEX_SETTLE_TIMEOUT_SECONDS}s). 서버 하드웨어 병목을 스캔하십시오."
  return 1
}

mkdir -p "$RESPONSE_DIR"

clear
echo "===================================================================="
echo "    IKG Custom URL Scraping Ingestion Test (Enhanced Edition)"
echo "===================================================================="
echo "[IKG TEST] ${#URLS[@]}개의 URL-only 인입 요청을 백그라운드 동시 트리거합니다."
echo "[IKG TEST] 슬라이딩 윈도우 최대 동시 요청 수: ${MAX_PARALLEL_REQUESTS}"
echo "[IKG TEST] 응답 저장 위치: ${RESPONSE_DIR}"
echo "--------------------------------------------------------------------"

request_no=1
running_jobs=0
for target_url in "${URLS[@]}"; do
  ingest_url "$(printf "%02d" "$request_no")" "$target_url" &
  running_jobs=$((running_jobs + 1))
  request_no=$((request_no + 1))

  # [CRITICAL FIX]: 슬라이딩 포크 직후 극소량의 시차를 벌어주어,
  # 백엔드가 인바운드 처리를 완수하고 비동기 큐에 안전 바인딩할 하드웨어 숨통을 열어줍니다.
  sleep "$INJECTION_INTERVAL_DELAY"

  if [ "$running_jobs" -ge "$MAX_PARALLEL_REQUESTS" ]; then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
done

wait
echo "--------------------------------------------------------------------"
wait_for_indexing_idle