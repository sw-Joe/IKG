import logging  
import asyncio
import html
import re
import os
import urllib.request

import trafilatura
from playwright.async_api import async_playwright

from ai_core.core.content_validator import validate_content_integrity



logger = logging.getLogger("ai_core.core.bookmark_scraper")

# ---------------------------------------------------------------------
# [CONCURRENCY GUARD]: 하드웨어 맞춤형 보수적 사양 가드 로직
# ---------------------------------------------------------------------
def _calculate_conservative_concurrency() -> int:
    try:
        cpu_cores = os.cpu_count() or 2
        # 저사양 랩탑 환경을 위한 임계 가드: 가용 코어에서 2개를 제하되 최소 1개 ~ 최대 2개로 제약
        concurrency_limit = max(1, cpu_cores - 2)
        return min(concurrency_limit, 2)
    except Exception:
        return 1

CONCURRENCY_LIMIT = _calculate_conservative_concurrency()

# 전역 비동기 자원 통제 싱글톤 변수
_scraper_semaphore = None
_playwright_instance = None
_global_browser = None

logger.info(f"[RESOURCE GUARD] 시스템 맞춤 가드 작동 -> 크롬 동시 렌더링 한계: {CONCURRENCY_LIMIT}개")


def _extract_title_from_html(raw_html: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return html.unescape(title) if title else None


def fetch_static_content(url: str, timeout_seconds: int = 15) -> tuple[str | None, str | None]:
    """브라우저 런치 실패 또는 정적 문서 수집용 HTTP fallback 스크래퍼."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_bytes = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            raw_html = raw_bytes.decode(charset, errors="ignore")

        title = _extract_title_from_html(raw_html)
        content = trafilatura.extract(raw_html, include_comments=False, include_tables=True)
        return title, content
    except Exception as e:
        logger.warning(f"[STATIC SCRAPER WARNING] URL 직접 수집 실패 -> {url} | Reason: {e}", exc_info=True)
        return None, None


def extract_bookmarks(node) -> list:
    """브라우저 백업 JSON 트리를 재귀 순회하여 단순 URL 리스트로 파싱 (기능 이관)"""
    bookmarks = []
    if "typeCode" in node:
        if node["typeCode"] == 1 and "uri" in node:
            bookmarks.append({
                "title": node.get("title", "Untitled"),
                "uri": node["uri"]
            })
        elif node["typeCode"] == 2 and "children" in node:
            for child in node["children"]:
                bookmarks.extend(extract_bookmarks(child))
    return bookmarks


async def fetch_dynamic_content_with_context(context, url: str) -> tuple[str | None, str | None]:
    """주입받은 외부 배치 격리 브라우저 탭 세션에서 고속 동적 스크래핑 수행 (기능 이관)"""
    page = None
    try:
        page = await context.new_page()
        page.set_default_timeout(20000)
        
        response = await page.goto(url, wait_until="domcontentloaded")
        if response and response.status >= 400:
            return None, f"ERR_HTTP_{response.status}"
            
        title = await page.title()
        html = await page.content()
        
        cleaned_text = trafilatura.extract(html, include_comments=False, include_tables=True)
        return title, cleaned_text
    except Exception as e:
        logger.debug(f"[BATCH SCRAPE WARNING] URL 수집 실패 -> {url} | Reason: {e}")
        return None, str(e)
    finally:
        if page:
            await page.close()


async def get_clean_browser_singleton():
    """OS 자원 고갈을 막기 위한 단일 크롬 브라우저 커널 인스턴스 생명주기 관리"""
    global _playwright_instance, _global_browser
    if _global_browser is None:
        _playwright_instance = await async_playwright().start()
        _global_browser = await _playwright_instance.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]
        )
    return _global_browser


async def scrape_url_standalone(url: str, timeout_ms: int = 15000) -> tuple[str | None, str | None]:
    """
    [SAFE ASYNC ENGINE]: 세마포어 가드선 내부에서 전역 싱글톤 브라우저 탭 컨텍스트를
    안전하게 공유 발급받아 동적 CSR 페이지 본문을 리소스 크래시 없이 원자적으로 포획합니다.
    """
    global _scraper_semaphore
    if _scraper_semaphore is None:
        _scraper_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    title, content = None, None
    
    # 💡 세마포어 풀에 임시 대기 진입하여 동시성 레이턴시 스로틀링 집행
    async with _scraper_semaphore:
        try:
            browser = await get_clean_browser_singleton()
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            response = await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            if response and response.status >= 400:
                logger.warning(f"[STANDALONE SCRAPER] HTTP 경고 상태 코드 반환: {response.status}")
                
            title = (await page.title()).strip()
            raw_html = await page.content()
            
            if raw_html:
                content = trafilatura.extract(raw_html, include_comments=False, include_tables=True)
                
            await page.close()
            await context.close()
        except Exception as e:
            logger.error(f"[SCRAPER CRITICAL ERROR] 단건 실시간 웹 스크래핑 장애 발생 -> URL: {url} | Reason: {str(e)}", exc_info=True)

    if title and content:
        return title, content

    logger.info(f"[SCRAPER FALLBACK] 브라우저 추출 결과가 비어 정적 HTML 수집으로 전환합니다. -> URL: {url}")
    return await asyncio.to_thread(fetch_static_content, url, max(5, timeout_ms // 1000))


def validate_scraped_bookmark(title: str | None, content: str | None) -> tuple[bool, str]:
    """스크래핑 완료된 원문의 정보 가치 및 원문 오염도 정밀 가드라인 교차 검증"""
    if not title or not content or not content.strip():
        return False, "ERR_EMPTY_OR_FETCH_FAILED"
        
    is_valid, reason_or_cleaned = validate_content_integrity(title, content)
    return is_valid, reason_or_cleaned


async def warmup_browser_infrastructure():
    """
    [INFRASTRUCTURE WARM-UP]: 백엔드 게이트웨이 스타트업 시점에 
    Playwright 커널 및 싱글톤 브라우저 프로세스를 미리 백그라운드에 상주시켜 
    런타임 동시 폭격 시의 드라이버 소켓 단절(자살 시그널)을 원천 차단합니다.
    """
    global _playwright_instance, _global_browser
    if _global_browser is None:
        logger.info("[INFRASTRUCTURE] Playwright 전역 싱글톤 브라우저 커널 초기화 웜업 개시...")
        try:
            _playwright_instance = await async_playwright().start()
            _global_browser = await _playwright_instance.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]
            )
            logger.info("[INFRASTRUCTURE] Playwright 싱글톤 브라우저 인프라가 메모리에 완벽히 안착되었습니다.")
        except Exception as e:
            logger.critical(f"[INFRASTRUCTURE CRITICAL] 브라우저 선제 웜업 실패: {e}", exc_info=True)