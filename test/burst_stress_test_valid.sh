#!/bin/bash

# ====================================================================
# IKG MVP 유효 자산 고밀도 비동기 인제스션(Add Burst) 스트레스 테스트
# 검증 가드라인: 본문 100자 이상 데이터 진입 시 메인 버스 가동 검수
# ====================================================================

API_URL="http://localhost:8000/api/bookmarks"

clear
echo "===================================================================="
echo "    IKG Core Engine: 100자 이상 유효 자산 동시성 스트레스 테스트"
echo "===================================================================="
echo "[IKG TEST] 10개의 고밀도 시맨틱 페이로드 백그라운드 동시 트리거 개시..."
echo "--------------------------------------------------------------------"

# 10개의 독립 요청 유입 (각 content 필드는 최소 200자 이상으로 격리 테이블 우회 확정)
curl -s -o /dev/null -w "요청 01: HTTP %{http_code}\n" -X POST "$API_URL" -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/sw-Joe/MachineLearningPrac", "title": "ML Prac Repository", "content": "This is a comprehensive structural validation test for checking the end-to-end performance bounds of the bge-m3 embedding inference worker and FAISS vector infrastructure tracking framework for node index entry number 1."}' &

curl -s -o /dev/null -w "요청 02: HTTP %{http_code}\n" -X POST "$API_URL" -H "Content-Type: application/json" \
  -d '{"url": "https://pytorch.org/docs/stable/nn.html", "title": "PyTorch Neural Networks", "content": "Deep learning model reproduction focuses on implementing and optimizing CNN architectures such as ResNet, EfficientNet, and Vision Transformers using PyTorch framework within a Linux-based development environment number 2."}' &

curl -s -o /dev/null -w "요청 03: HTTP %{http_code}\n" -X POST "$API_URL" -H "Content-Type: application/json" \
  -d '{"url": "https://fastapi.tiangolo.com/tutorial", "title": "FastAPI ASGI Gateway", "content": "The intelligent knowledge graphing platform automatically extracts semantic relationships from fragmented web bookmarks to construct a multidimensional knowledge visualization network engine for number 3."}' &

curl -s -o /dev/null -w "요청 04: HTTP %{http_code}\n" -X POST "$API_URL" -H "Content-Type: application/json" \
  -d '{"url": "https://faiss.ai/cpp_api/index.html", "title": "FAISS Indexing Engine", "content": "On-device local first AI inference system guarantees total user data privacy by removing heavy external infrastructure dependencies like Celery or Redis brokers and shifting to embedded serialization bus 4."}' &

curl -s -o /dev/null -w "요청 05: HTTP %{http_code}\n" -X POST "$API_URL" -H "Content-Type: application/json" \
  -d '{"url": "https://sqlite.org/atomiccommit.html", "title": "SQLite Atomic Transaction", "content": "Ensuring data alignment integrity requires mapping the relation database physical primary key to the FAISS dense vector identity tag tightly, blocking index shift or mismatch during real-time CRUD operations 5."}' &

curl -s -o /dev/null -w "요청 06: HTTP %{http_code}\n" -X POST "$API_URL" -H "Content-Type: application/json" \
  -d '{"url": "https://onnxruntime.ai/docs/performance", "title": "ONNX Quantization Spec", "content": "Sanity sync framework triggers automatic infrastructure scanning at server startup to cross-check SQLite record counts and FAISS index dimensions, executing fallback rebuild routines if discrepancies are found 6."}' &

curl -s -o /dev/null -w "요청 07: HTTP %{http_code}\n" -X POST "$API_URL" -H "Content-Type: application/json" \
  -d '{"url": "https://nvidia.com/cuda/architecture", "title": "NVIDIA CUDA Optimization", "content": "High performance hardware utilization includes remote access optimization for NVIDIA RTX 4090 architecture and local machine learning workflow handling on expanded memory allocations for distributed training 7."}' &

curl -s -o /dev/null -w "요청 08: HTTP %{http_code}\n" -X POST "$API_URL" -H "Content-Type: application/json" \
  -d '{"url": "https://huggingface.co/models", "title": "HuggingFace Tokenizer Hub", "content": "The hybrid search pipeline combines dense vector similarity scoring with sparse BM25 lexical constraints, mathematically normalizing and reranking candidate pools through an attention filter layer 8."}' &

curl -s -o /dev/null -w "요청 09: HTTP %{http_code}\n" -X POST "$API_URL" -H "Content-Type: application/json" \
  -d '{"url": "https://react.dev/reference/react", "title": "React Virtual DOM Optimization", "content": "Minimalist backend refactoring eliminates unnecessary abstraction wrappers to emit a flat one-dimensional JSON list format directly, removing data parsing overhead on the React frontend virtual DOM loops 9."}' &

curl -s -o /dev/null -w "요청 10: HTTP %{http_code}\n" -X POST "$API_URL" -H "Content-Type: application/json" \
  -d '{"url": "https://vitejs.dev/guide/features", "title": "Vite Assets Packaging", "content": "A highly optimized viewport rendering layer introduces lazy loading mechanisms for large-scale graph layouts to isolate single-thread JavaScript browser engines from canvas visualization rendering freezes 10."}' &

# 모든 백그라운드 헌터 태스크 프로세스가 완전히 수신 종료될 때까지 블로킹 대기
wait

echo "--------------------------------------------------------------------"
echo "[IKG TEST] 모든 클라이언트 API 요청 레이어 처리가 완결되었습니다."
echo "[IKG TEST] 즉각 uvicorn 서버 로그 창의 [EMBEDDED BUS] 소비 파이프라인을 검수하십시오."
echo "===================================================================="