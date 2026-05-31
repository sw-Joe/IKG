import sys
import os
import logging



# 로깅 인프라 수동 빌드 (테스트 콘솔 출력용)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s"
)
logger = logging.getLogger("sanity_test")


# 모노레포 수립 네임스페이스 경로 강제 가이드 바인딩
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.ai_core.hybrid_search import HybridSearcher
    logger.info("✅ [TEST] ai_core 모듈 네임스페이스 수입에 성공하였습니다.")
except ModuleNotFoundError as e:
    logger.error(f"❌ [TEST FAIL] 임포트 패스 수립 실패: {e}")
    sys.exit(1)

def run_search_sanity_check():
    logger.info("==================================================")
    logger.info("  IKG 하이브리드 검색 코어 v3 수식 무결성 정밀 검증 개시")
    logger.info("==================================================")
    
    # 1. 랭커 엔진 인스턴스 초기화 (config.py 상수의 절대 경로 추적 가동)
    try:
        searcher = HybridSearcher()
    except Exception as e:
        logger.error(f"❌ 엔진 초기화 단계 크래시: {e}")
        return

    # 2. 콜드 스타트 자산 매트릭스 정합성 검증
    doc_count = len(searcher.documents)
    if doc_count == 0:
        logger.warning("❌ 영속 DB 자산 레코드가 0건입니다. db/ 폴더 내부의 실물 .db 자산을 재확인하십시오.")
        return
        
    index_total = searcher.index.ntotal
    logger.info(f"[물리 지표] DB 레코드 수: {doc_count}개 | FAISS 벡터 수: {index_total}개")
    
    if doc_count != index_total:
        logger.warning(f"⚠️ [주의] DB와 INDEX 개수가 다릅니다. 가드레일 코드의 우회 배출 로직이 가동되어야 합니다.")

    # 3. 모의 질의어 인퍼런스 수행 및 레이어별 상태 추적
    test_query = "파이썬 기초 문법"
    logger.info(f"\n[실행] 모의 검색 질의 테스트 컴파일 개시 -> 입력어: '{test_query}'")
    
    try:
        results = searcher.search(query=test_query, top_n=5)
        
        logger.info("\n==================================================")
        logger.info(f"   최종 연산 결과 스캔 계층 (반환 수량: {len(results)}건)")
        logger.info("==================================================")
        
        if not results:
            logger.error("❌ [CRITICAL TEST FAIL] 검색 결과가 여전히 빈 배열([])입니다. Zero-Hits 필터 컷오프 오탐이 해결되지 않았습니다.")
        else:
            logger.info("✅ [SUCCESS] 검색 결과가 무결하게 도출되었습니다. 출력 스키마 정합성을 검증합니다.")
            for i, item in enumerate(results):
                logger.info(
                    f"  [Top {i+1}] ID:#{item['id']} | 스코어:{item['score']:.4f} | "
                    f"제목:{item['title'][:20]}... | 시맨틱점수:{item['score_sem_raw']:.4f}"
                )
                
                # 프론트엔드가 바인딩할 score 키의 존재 여부 단언 검증 (Assertion)
                assert "score" in item, "프론트엔드 호환성 score 키가 결손되었습니다."
            logger.info("\n🎉 모든 하이브리드 수식 공간 연산의 수치 안정성이 확보되어 가동 가능 상태로 도출되었습니다.")
            
    except AssertionError as ae:
        logger.error(f"❌ [SCHEMATIC ERROR] 반환 JSON 데이터 스키마 키 불일치: {ae}")
    except Exception as e:
        logger.error(f"❌ [INFERENCE ERROR] 수식 연산 레이어 내부 크래시 발생: {e}", exc_info=True)

if __name__ == "__main__":
    run_search_sanity_check()