#!/bin/bash

# ====================================================================
# IKG URL-only custom ingestion stress test
# - 클라이언트는 URL만 전달한다.
# - BE가 직접 스크래핑/검증/색인을 수행한다.
# ====================================================================

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
BOOKMARK_API_URL="${API_BASE_URL}/api/bookmarks"
INDEXING_STATUS_URL="${API_BASE_URL}/api/system/indexing/status"
SYNC_URL="${API_BASE_URL}/api/system/sync"
INDEX_SETTLE_TIMEOUT_SECONDS="${INDEX_SETTLE_TIMEOUT_SECONDS:-300}"
RESPONSE_DIR="${RESPONSE_DIR:-/tmp/ikg_custom_url_responses}"
MAX_PARALLEL_REQUESTS="${MAX_PARALLEL_REQUESTS:-5}"

URLS=(
  'https://docs.python.org/3/library/dataclasses.html'
  'https://namu.wiki/w/Redox'
  'https://devocean.sk.com/blog/techBoardDetail.do?ID=164657&boardType=techBlog'
  'https://wikidocs.net/204321'
  'https://namu.wiki/w/%EC%8A%A4%ED%84%B8%EB%A7%81%20%EA%B8%B0%EA%B4%80'
  'https://namu.wiki/w/%EA%B3%B5%EB%8C%80%EA%B0%9C%EA%B7%B8'
  'https://namu.wiki/w/ROM'
  'https://www.kjebi.com/search/smart/bidder/detail/8398801249'
  'https://tls.kku.ac.kr/mod/ubboard/article.php?id=743231&bwid=739510'
  'https://www.google.com/search?q=Do+diamond+also+have+Cleavage&client=firefox-b-d&hs=aPnp&sca_esv=f27198b4d2287e35&sxsrf=ANbL-n775ewMUH5iH7kbyxTZhGUxMNNW3A%3A1777445370414&ei=-qnxaf7_GNWJvr0P5uzu6AE&ved=0ahUKEwj-5KT_u5KUAxXVhK8BHWa2Gx0Q4dUDCBE&uact=5&oq=Do+diamond+also+have+Cleavage&gs_lp=Egxnd3Mtd2l6LXNlcnAiHURvIGRpYW1vbmQgYWxzbyBoYXZlIENsZWF2YWdlMgYQABgWGB4yBhAAGBYYHjIGEAAYFhgeMggQABiABBiiBEikgwFQ3ARY-HtwBngAkAEAmAG6AaABiimqAQQxLjMyuAEDyAEA-AEB-AECmAIkoALlJqgCBsICChAAGEcY1gQYsAPCAhcQABiABBiKBRiRAhjnBhjqAhi0AtgBAcICHRAuGIAEGIoFGJECGOcGGMcBGNEDGOoCGLQC2AEBwgIFEC4YgATCAgUQABiABMICFBAuGIAEGJcFGNwEGN4EGOAE2AECwgIFECEYoAHCAgQQIRgVwgIFECEYnwXCAgUQABjvBcICCBAAGIkFGKIEmAMG8QWjHNkxoyNIAYgGAZAGAroGBAgBGAe6BgYIAhABGBSSBwQ2LjMwoAfepgGyBwQwLjMwuAfLJsIHCDAuOS4yNi4xyAd-gAgB&sclient=gws-wiz-serp'
  'https://namu.wiki/w/%EC%82%AC%EC%9B%90%EC%88%98'
  'https://wikidocs.net/21707'
  'https://namu.wiki/w/%EB%8B%AC%EC%BD%A4%ED%95%9C%20%EC%9D%B8%EC%83%9D(%EC%98%81%ED%99%94)'
  'https://www.reddit.com/r/linuxquestions/comments/184m9ay/hello_is_there_a_way_to_learn_linux_for_free/?tl=ko'
  'https://yozm.wishket.com/magazine/detail/2488/'
  'https://www.google.com/search?client=firefox-b-d&q=%EB%91%90%EA%B4%84%EC%8B%9D'
  'https://antigravity.google/docs/cli-overview'
  'https://namu.wiki/w/CRUD'
  'https://en.wikipedia.org/wiki/Remote_Control_Productions_(American_company)'
  'https://ffighting.net/deep-learning-basic/%eb%94%a5%eb%9f%ac%eb%8b%9d-%ed%95%b5%ec%8b%ac-%ea%b0%9c%eb%85%90/rnn/'
  'https://github.com/TG-WinG/Algorithm'
  'https://learn.microsoft.com/ko-kr/windows/apps/dev-tools/winapp-cli/guides/electron-setup'
  'https://huggingface.co/docs/transformers/en/index'
  'https://developer.furiosa.ai/furiosa-models/latest/'
  'https://arcprize.org/arc-agi/1'
  'https://docs.ollama.com/capabilities/tool-calling'
  'https://velog.io/@jkseo50/DVC-%EA%B0%9C%EB%85%90-%EB%B0%8F-%ED%99%9C%EC%9A%A9-%EB%B0%A9%EB%B2%95'
)

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

ingest_url() {
  local request_no="$1"
  local target_url="$2"
  local escaped_url

  escaped_url="$(json_escape "$target_url")"
  curl -sS -o "${RESPONSE_DIR}/response_${request_no}.json" -w "요청 ${request_no}: HTTP %{http_code} -> ${target_url}\n" \
    -X POST "$BOOKMARK_API_URL" \
    -H "Content-Type: application/json" \
    -d "{\"url\":\"${escaped_url}\"}"
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

mkdir -p "$RESPONSE_DIR"

clear
echo "===================================================================="
echo "    IKG Custom URL Scraping Ingestion Test"
echo "===================================================================="
echo "[IKG TEST] ${#URLS[@]}개의 URL-only 인입 요청을 백그라운드 동시 트리거합니다."
echo "[IKG TEST] 최대 동시 요청 수: ${MAX_PARALLEL_REQUESTS}"
echo "[IKG TEST] 응답 저장 위치: ${RESPONSE_DIR}"
echo "--------------------------------------------------------------------"

request_no=1
running_jobs=0
for target_url in "${URLS[@]}"; do
  ingest_url "$(printf "%02d" "$request_no")" "$target_url" &
  running_jobs=$((running_jobs + 1))
  request_no=$((request_no + 1))

  if [ "$running_jobs" -ge "$MAX_PARALLEL_REQUESTS" ]; then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
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
echo "[IKG TEST] 테스트 종료."
echo "===================================================================="
