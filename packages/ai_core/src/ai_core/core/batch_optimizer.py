import os
import logging
import psutil



logger = logging.getLogger("ai_core.core.batch_optimizer")


def calculate_optimal_batch_sizes(total_bookmarks_count: int) -> tuple[int, int]:
    """
    온디바이스 시스템의 물리 자원을 실시간 스캔하여 Stage 1과 Stage 2에 최적화된
    미니 배치 크기를 수리적으로 역산하여 반환합니다.
    
    계측 예외 또는 실패 발생 시, 시스템 안정성을 위해 기본값(각각 100)을 강제 할당합니다.
    
    Returns:
        tuple[int, int]: (Stage 1 수집 배치 크기, Stage 2 인덱싱 배치 크기)
    """
    # -------------------------------------------------------------------------
    # [FALLBACK DEFENSE]: 최적 계산 실패 시 적용할 무결한 기본 배치 사이즈 정의
    # -------------------------------------------------------------------------
    DEFAULT_STAGE1_BATCH = min(100, total_bookmarks_count)
    DEFAULT_STAGE2_BATCH = min(100, total_bookmarks_count)
    
    logger.info(f"[BATCH OPTIMIZER] 자원 가드 엔진 가동 -> 총 처리 대상 자산: {total_bookmarks_count}건")
    
    try:
        # 1. 시스템 물리 자원 메트릭 수집
        virtual_mem = psutil.virtual_memory()
        available_ram_mb = virtual_mem.available / (1024 * 1024)
        cpu_cores = os.cpu_count() or 4
        
        logger.info(f" -> 호스트 환경 탐색 완료: 가용 RAM {available_ram_mb:.1f} MB | 논리 CPU {cpu_cores} Cores")

        # 2. STAGE 1 (스크래핑 & DB 적재) 배치 크기 역산
        safety_allocated_ram = available_ram_mb * 0.60
        estimated_ram_per_tab = 120.0  # 건당 평균 크로미움 점유 메모리 임계치
        
        calculated_scrape_batch = int(safety_allocated_ram / estimated_ram_per_tab)
        
        # 물리 안전 상하한선 제약 (소켓/파일 디스크립터 오염 차단용 50~150 제약)
        if calculated_scrape_batch > 150:
            stage1_batch = 150
        elif calculated_scrape_batch < 30:
            stage1_batch = 30
        else:
            stage1_batch = calculated_scrape_batch
            
        if total_bookmarks_count < stage1_batch:
            stage1_batch = total_bookmarks_count

        # 3. STAGE 2 (ONNX 배치 임베딩 & FAISS) 배치 크기 역산
        if cpu_cores <= 4:
            stage2_batch = 32
        elif cpu_cores <= 8:
            stage2_batch = 64
        else:
            stage2_batch = 100  # 패딩 토큰 행렬 마진 낭비를 차단하기 위한 프로덕션 상한선
            
        if total_bookmarks_count < stage2_batch:
            stage2_batch = total_bookmarks_count

        # 4. 소량 데이터셋 규모 예외 조율
        if total_bookmarks_count <= 100:
            stage1_batch = min(30, total_bookmarks_count)
            stage2_batch = min(30, total_bookmarks_count)

        logger.info(f"[ALLOCATION SUCCESS] 자원 맞춤형 파이프라인 배치 할당")
        logger.info(f" -> Stage 1 (Scraping Batch)  : {stage1_batch} 건")
        logger.info(f" -> Stage 2 (Embedding Batch) : {stage2_batch} 건")
        
        return stage1_batch, stage2_batch

    except Exception as e:
        # [사이드 이펙트 보완]: psutil 부재, 권한 차단 등의 런타임 예외 감지 시 기본 가드레일값 반환
        logger.warning(
            f"[OPTIMIZER FALLBACK EXECUTED] 자원 실시간 스캔 및 연산 실패 ({str(e)}). "
            f"안정적인 기본 배치 크기 명세(Stage1: {DEFAULT_STAGE1_BATCH}, Stage2: {DEFAULT_STAGE2_BATCH})를 강제 적용합니다."
        )
        return DEFAULT_STAGE1_BATCH, DEFAULT_STAGE2_BATCH