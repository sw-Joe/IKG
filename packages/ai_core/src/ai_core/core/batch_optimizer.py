import os
import logging
import psutil



logger = logging.getLogger("ai_core.core.batch_optimizer")


def calculate_optimal_batch_sizes(total_bookmarks_count: int) -> tuple[int, int]:
    """
    온디바이스 시스템의 물리 자원을 실시간 스캔하여 Stage 1과 Stage 2에 최적화된
    미니 배치 크기를 수리적으로 역산하여 반환합니다.
    
    Returns:
        tuple[int, int]: (Stage 1 수집 배치 크기, Stage 2 인덱싱 배치 크기)
    """
    logger.info(f"[BATCH OPTIMIZER] 자원 가드 엔진 가동 -> 총 처리 대상 자산: {total_bookmarks_count}건")
    
    # -------------------------------------------------------------------------
    # 1. 시스템 물리 자원 메트릭 수집
    # -------------------------------------------------------------------------
    # 가용 메모리 계측 (Bytes -> Megabytes 변환)
    virtual_mem = psutil.virtual_memory()
    available_ram_mb = virtual_mem.available / (1024 * 1024)
    
    # 논리 CPU 코어 개수 확보
    cpu_cores = os.cpu_count() or 4
    
    logger.info(f" -> 호스트 환경 탐색 완료: 가용 RAM {available_ram_mb:.1f} MB | 논리 CPU {cpu_cores} Cores")

    # -------------------------------------------------------------------------
    # 2. STAGE 1 (스크래핑 & DB 적재) 최적 배치 크기 역산
    #    - 제약 자원: Playwright 크로미움 탭 인스턴스 누수 오버헤드 (건당 약 80~150MB 할당)
    # -------------------------------------------------------------------------
    # 안전 임계 메모리 버퍼 계드 (가용 RAM의 60%만 크롤러 세션에 순수 배정)
    safety_allocated_ram = available_ram_mb * 0.60
    estimated_ram_per_tab = 120.0  # 평균 렌더링 소모 텐서 메모리 상하한선
    
    calculated_scrape_batch = int(safety_allocated_ram / estimated_ram_per_tab)
    
    # 물리 안전 가드레일 제약 (아무리 RAM이 커도 소켓/파일 디스크립터 고갈을 막기 위해 50~150 사이 통제)
    if calculated_scrape_batch > 150:
        stage1_batch = 150
    elif calculated_scrape_batch < 30:
        stage1_batch = 30
    else:
        stage1_batch = calculated_scrape_batch
        
    # 만약 총 북마크 수가 계산된 배치 크기보다 작다면 전체 개수로 수렴
    if total_bookmarks_count < stage1_batch:
        stage1_batch = total_bookmarks_count

    # -------------------------------------------------------------------------
    # 3. STAGE 2 (ONNX 배치 임베딩 & FAISS) 최적 배치 크기 역산
    #    - 제약 자원: CPU L3 캐시 대역폭 및 [B x Length x 1024] 패딩 토큰 매트릭스 확장 오버헤드
    # -------------------------------------------------------------------------
    # BGE-M3 Dense 임베딩의 SIMD 멀티스레딩 최적 Saturation 임계값 수립
    # 코어 수가 많을수록 대형 행렬 곱(GEMM) 연산 가속력이 상향됨을 반영
    if cpu_cores <= 4:
        stage2_batch = 32
    elif cpu_cores <= 8:
        stage2_batch = 64
    else:
        stage2_batch = 100  # 패딩 토큰 낭비 오버헤드를 막기 위한 프로덕션 상하한선 고수
        
    if total_bookmarks_count < stage2_batch:
        stage2_batch = total_bookmarks_count

    # -------------------------------------------------------------------------
    # 4. 데이터셋 규모에 따른 최종 조율 (소량 데이터셋 오버헤드 완화)
    # -------------------------------------------------------------------------
    if total_bookmarks_count <= 100:
        stage1_batch = min(30, total_bookmarks_count)
        stage2_batch = min(30, total_bookmarks_count)

    logger.info(f"[ALLOCATION COMPLETION] 최적 파이프라인 배치 크기 확정")
    logger.info(f" -> Stage 1 (Scraping Task Chunk Size)  : {stage1_batch} 건 주축 구동")
    logger.info(f" -> Stage 2 (Embedding Inference Batch) : {stage2_batch} 건 매트릭스 병렬화")
    
    return stage1_batch, stage2_batch