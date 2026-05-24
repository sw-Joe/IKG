import sys
# 1. 각 버전을 명확히 구분할 수 있도록 에일리어스(Alias) 지정 임포트
from hybrid_search.hybrid_search_v2 import HybridSearcher as HybridSearcherV2
from hybrid_search.two_stage_cascading_ensemble import HybridSearcherV3Stage1
from hybrid_search.rank_scaled_score_fusion import HybridSearcherV3RankScaled
from hybrid_search.native_sparse_embedding_search import SearcherNativeSparse

def test_hybrid_benchmark():
    print("=" * 50)
    print(" [IKG-Search] 하이브리드 검색 알고리즘 대조 실험 벤치마크")
    print("=" * 50)
    print("1: Baseline v2 (정적 가중치 합성 + 최신성 감쇄 + 렉시컬 게이트)")
    print("2: v3 후보 1 (2-Stage Cascading 앙상블 파이프라인)")
    print("3: v3 후보 2 (순위 변조형 가중치 합산)")
    print("4: v3 후보 3 (BGE-M3 Native Sparse Embedding 활용)")
    print("=" * 50)
    
    try:
        choice = input("테스트할 모듈 번호를 선택하세요 (1~4): ").strip()
        if choice == '1':
            print("[SYSTEM] Baseline v2 모듈을 초기화합니다.")
            searcher = HybridSearcherV2()
            version_tag = "V2_BASELINE"
        elif choice == '2':
            print("[SYSTEM] v3 후보 1 (2-Stage Cascading) 모듈을 초기화합니다.")
            searcher = HybridSearcherV3Stage1()
            version_tag = "V3_STAGE1"
        elif choice == '3':
            print("[SYSTEM] v3 후보 2 (Rank-Scaled) 모듈을 초기화합니다.")
            searcher = HybridSearcherV3RankScaled()
            version_tag = "V3_RANK_SCALED"
        elif choice == '4':
            print("[SYSTEM] v3 후보 3 (Native Sparse) 모듈을 초기화합니다.")
            searcher = SearcherNativeSparse()
            version_tag = "V3_NATIVE_SPARSE"
        else:
            print("[ERROR] 잘못된 선택입니다. 프로그램을 종료합니다.")
            return
    except Exception as e:
        print(f"[CRITICAL] 모듈 초기화 실패: {e}")
        print("힌트: 데이터베이스 경로 및 모델 가중치 파일 위치를 확인하세요.")
        sys.exit(1)

    while True:
        query = input(f"\n[{version_tag}] 검색어 입력 (종료: q): ").strip()
        if query.lower() == 'q':
            break
        if not query:
            continue
            
        results = searcher.search(query, top_n=5)
        
        print(f"\n================ [{query}] 통합 검색 결과 ================")
        if not results:
            print("▶ 검색 결과가 존재하지 않습니다. (Zero Hits)")
            continue
            
        for i, res in enumerate(results):
            print(f"[{i+1}] {res['title']}")
            print(f"    URL: {res['url']}")
            
            # v3 공통 및 고유 디버깅 스코어 추출 처리 (KeyError 방지 가드)
            f_score = res.get('score_final', 0.0)
            l_score = res.get('score_lex', 0.0)
            s_score = res.get('score_sem', 0.0)
            t_factor = res.get('factor_time', 1.0)
            g_factor = res.get('factor_gate', 1.0)
            
            print(f"    [Scores] Final: {f_score} | Lexical: {l_score} | Semantic: {s_score}")
            print(f"    [Factors] Time Decay: {t_factor} | Lexical Gate: {g_factor}")
            
            # 후보군별 특화 지표 추가 출력
            if version_tag == "V3_STAGE1" and 'rrf_score' in res:
                print(f"    [Stage1 특화] RRF Score: {res['rrf_score']}")
            elif version_tag == "V3_RANK_SCALED" and 'factor_rank_penalty' in res:
                print(f"    [RankScaled 특화] Rank Penalty 인자: {res['factor_rank_penalty']} (Lex순위: {res.get('rank_lex')}, Sem순위: {res.get('rank_sem')})")
            elif version_tag == "V3_NATIVE_SPARSE" and 'raw_sparse_score' in res:
                print(f"    [NativeSparse 특화] Raw Sparse 내적 점수: {res['raw_sparse_score']}")
                
            print(f"    Snippet: {res['content'][:90].replace('\\n', ' ')}...")
            print("-" * 65)

if __name__ == "__main__":
    test_hybrid_benchmark()