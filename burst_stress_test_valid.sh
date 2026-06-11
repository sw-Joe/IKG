#!/bin/bash

# ====================================================================
# IKG MVP 유효 자산 고밀도 비동기 인제스션(Add Burst) 스트레스 테스트
# 검증 가드라인: 클라이언트는 URL만 전달하고, BE가 직접 스크래핑한 본문으로 검증/색인한다.
# ====================================================================

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
BOOKMARK_API_URL="${API_BASE_URL}/api/bookmarks"
INDEXING_STATUS_URL="${API_BASE_URL}/api/system/indexing/status"
SYNC_URL="${API_BASE_URL}/api/system/sync"
INDEX_SETTLE_TIMEOUT_SECONDS="${INDEX_SETTLE_TIMEOUT_SECONDS:-180}"

URLS=(
  "https://docs.python.org/3/tutorial/"
  "https://fastapi.tiangolo.com/tutorial/"
  "https://docs.pytorch.org/docs/stable/nn.html"
  "https://www.sqlite.org/atomiccommit.html"
  "https://faiss.ai/cpp_api/index.html"
  "https://onnxruntime.ai/docs/performance/"
  "https://huggingface.co/docs/transformers/index"
  "https://react.dev/reference/react"
  "https://vite.dev/guide/features"
  "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API"
)

ingest_url() {
  local request_no="$1"
  local target_url="$2"

  curl -sS -o /tmp/ikg_burst_response_"${request_no}".json -w "요청 ${request_no}: HTTP %{http_code} -> ${target_url}\n" \
    -X POST "$BOOKMARK_API_URL" \
    -H "Content-Type: application/json" \
    -d "{\"url\":\"${target_url}\"}"
}

wait_for_indexing_idle() {
  local elapsed=0

  while [ "$elapsed" -lt "$INDEX_SETTLE_TIMEOUT_SECONDS" ]; do
    local status
    status="$(curl -sS "$INDEXING_STATUS_URL" || true)"

    if echo "$status" | grep -q '"idle":true'; then
      echo "[IKG TEST] 인덱싱 큐 idle 확인 완료."
      return 0
    fi

    echo "[IKG TEST] 인덱싱 대기 중... ${status}"
    sleep 2
    elapsed=$((elapsed + 2))
  done

  echo "[IKG TEST] 인덱싱 idle 대기 시간 초과 (${INDEX_SETTLE_TIMEOUT_SECONDS}s). 서버 로그를 확인하십시오."
  return 1
}

clear
echo "===================================================================="
echo "    IKG Core Engine: 실시간 스크래핑 기반 동시성 스트레스 테스트"
echo "===================================================================="
echo "[IKG TEST] ${#URLS[@]}개의 URL-only 인입 요청을 백그라운드 동시 트리거합니다."
echo "--------------------------------------------------------------------"

request_no=1
for target_url in "${URLS[@]}"; do
  ingest_url "$(printf "%02d" "$request_no")" "$target_url" &
  request_no=$((request_no + 1))
done

wait

echo "--------------------------------------------------------------------"
echo "[IKG TEST] 모든 클라이언트 API 요청이 접수되었습니다. 백그라운드 인덱싱 완료를 대기합니다."

if wait_for_indexing_idle; then
  echo "[IKG TEST] 최종 DB/FAISS/검색 메모리 컨텍스트 동기화를 요청합니다."
  curl -sS -X POST "$SYNC_URL"
  echo
fi

echo "--------------------------------------------------------------------"
echo "[IKG TEST] 테스트 종료. /tmp/ikg_burst_response_*.json 에서 개별 인입 결과를 확인할 수 있습니다."
echo "===================================================================="
