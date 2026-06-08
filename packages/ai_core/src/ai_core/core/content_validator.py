import logging
import re



logger = logging.getLogger("ai_core.content_validator")


def validate_content_integrity(title: str, content: str) -> tuple[bool, str]:
    """
    수집된 본문 및 제목 콘텍스트의 지식 정보 가치를 정규식 매칭 및 통계 필터로 정밀 스캔합니다.
    
    Returns:
        tuple[bool, str]: (유효성 여부, 정제된 본문 텍스트 또는 탈락 원인 코드)
    """
    if not content or not content.strip():
        return False, "ERR_EMPTY_CONTENT"
        
    cleaned_content = content.strip()
    combined_lower = f"{title} {cleaned_content}".lower()

    # 1. 404 및 웹 서비스 템플릿 에러 정밀 탐지 (통계 리포트 4건 대응)
    if re.search(r"(404\s?(not\s?found|페이지)|존재하지\s?않는\s?페이지|ã€€)", combined_lower):
        return False, "ERR_404_NOT_FOUND"

    # 2. 웹 방화벽 및 Captcha 차단 정밀 탐지 (통계 리포트 543건 대응 - 최우선 가드)
    WAF_PATTERNS = [
        "access denied", "forbidden", "captcha", "보안 확인", "robot",
        "checking your browser", "cloudflare", "sucuri", "ddos protection"
    ]
    if any(pattern in combined_lower for pattern in WAF_PATTERNS):
        return False, "ERR_ACCESS_DENIED_WAF"

    # 3. 자바스크립트 미구동 CSR 블로킹 탐지 (통계 리포트 58건 대응)
    JS_PATTERNS = [
        "javascript is required", "자바스크립트", "enable javascript", 
        "please turn on javascript"
    ]
    if any(pattern in combined_lower for pattern in JS_PATTERNS):
        return False, "ERR_JAVASCRIPT_REQUIRED"

    # 4. 정보 가치 부족 저용량 필터 (통계 리포트 33건 대응)
    # 로그인 세션 튕김, 빈 랜딩 페이지 등을 완벽 차단하기 위해 공백 제외 최소 150자 임계치 유지
    if len(cleaned_content) < 150:
        return False, f"ERR_TOO_SHORT_CONTENT ({len(cleaned_content)} chars)"

    # 5. 텍스트 노이즈 비율 (엔트로피 가드)
    # 특수문자나 기호의 비율이 전체의 40%를 초과하는 소스 코드 덤프 및 인코딩 깨짐 페이지 필터링
    special_chars = len(re.findall(r'[^a-zA-Z0-9가-힣\s]', cleaned_content))
    if special_chars / len(cleaned_content) > 0.40:
        return False, "ERR_HIGH_NOISE_RATIO"

    return True, cleaned_content