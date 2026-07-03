import os
import psutil
import logging
import asyncio
from fastapi import HTTPException, status



logger = logging.getLogger("ai_core.core.resource_guard")


class DynamicResourceGuard:
    """
    [INFRASTRUCTURE RESOURCE GUARD]: 호스트 물리 자원을 실시간 모니터링하여
    비동기 Task 버퍼와 크롬 탭 인스턴스의 폭격으로 인한 OOM(Out of Memory)을 선제 가드합니다.
    """
    def __init__(self, memory_allocation_ratio: float = 0.3, estimated_tab_cost_mb: int = 150):
        self.memory_allocation_ratio = memory_allocation_ratio
        self.estimated_tab_cost = estimated_tab_cost_mb * 1024 * 1024
        
        # 물리 자원 맞춤형 대기열 한계값 자동 책정
        self.max_allowed_backlog = self._calculate_dynamic_backlog_limit()
        
        # 런타임 상태 추적 카운터 및 비동기 락 뮤텍스
        self.current_backlog_count = 0
        self.counter_lock = asyncio.Lock()

    def _calculate_dynamic_backlog_limit(self) -> int:
        try:
            vm = psutil.virtual_memory()
            available_gb = vm.available / (1024 ** 3)
            
            # 가용 RAM의 일정 비율을 크롬 탭 가상 대기 버퍼 공간으로 할당
            allocated_memory_buffer = vm.available * self.memory_allocation_ratio
            calculated_limit = int(allocated_memory_buffer // self.estimated_tab_cost)
            
            # 저사양 랩탑(최소 3개) 및 고사양 서버(최대 30개) 극단 기하 제약 가드
            final_limit = max(3, min(calculated_limit, 30))
            
            logger.info(
                f"[RESOURCE PROFILE] 호스트 가용 RAM: {available_gb:.2f} GB "
                f"──► 동적 계산된 백로그 대기열 상한선: {final_limit}개 확정"
            )
            return final_limit
        except Exception as e:
            logger.error(f"[SPEC DETECT FAILURE] 호스트 자원 계측 실패, 보수적 기본값(5) 강제 적용: {e}")
            return 5

    async def acquire_ingress_permits(self):
        """
        [THROTTLING GUARD]: 인바운드 트래픽 진입 시 백로그 총량을 검문하여
        임계치 초과 시 즉각 HTTP 429 에러를 발생시켜 프로세스를 보호합니다.
        """
        if self.current_backlog_count >= self.max_allowed_backlog:
            logger.warning(
                f"[OVERLOAD BLOCK] 호스트 자원 보호 작동 -> "
                f"현재 백로그: {self.current_backlog_count}개 / 임계치: {self.max_allowed_backlog}개"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="실행 환경의 메모리 보호를 위해 스크래핑 대기열 인입이 일시 제한되었습니다. 잠시 후 재시도하십시오."
            )
        
        async with self.counter_lock:
            self.current_backlog_count += 1
            logger.debug(f"[QUEUE INGRESS] 태스크 인입 완수 -> 가용 백로그 현황: {self.current_backlog_count}/{self.max_allowed_backlog}")

    async def release_ingress_permits(self):
        """태스크가 완전 종결되거나 비동기 예외 누출 시 안전하게 백로그 자원 반환"""
        async with self.counter_lock:
            self.current_backlog_count = max(0, self.current_backlog_count - 1)
            logger.debug(f"[QUEUE EGRESS] 태스크 자원 반환 완료 -> 가용 백로그 현황: {self.current_backlog_count}/{self.max_allowed_backlog}")