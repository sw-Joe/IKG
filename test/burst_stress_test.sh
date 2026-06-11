#!/bin/bash
API_URL="http://localhost:8000/api/bookmarks"

echo "[IKG TEST] 10개 동시 비동기 인제스션 요청 가동..."
for i in {1..10}
do
  curl -s -o /dev/null -w "요청 $i: HTTP %{http_code}\n" -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "{\"url\": \"https://test-repo.com/$i\", \"title\": \"Stress Test Payload $i\", \"content\": \"This is an automated structural test sequence for testing backend queue capacity mapping number $i.\"}" &
done
wait
echo "[IKG TEST] 모든 클라이언트 요청 전송 완결."
