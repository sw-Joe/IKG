import sys

from hybrid_search.attention import AttentionHybridSearcher
from hybrid_search.lagacy.hybrid_search_v2 import HybridSearcherV2
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
    print("5: v4 확장안 (Attention 기반 동적 가중치 융합)  # [추가]")
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
        elif choice == '5':
            print("[SYSTEM] v4 확장안 (Attention-driven Dynamic) 모듈을 초기화합니다.")
            searcher = AttentionHybridSearcher()
            version_tag = "V4_ATTENTION"
        else:
            print("[ERROR] 올바른 번호를 선택하세요 (1~5).")
            return
    except Exception as e:
        print(f"[INIT ERROR] 인덱스 및 모델 로드 중 실패: {e}")
        sys.exit(1)
        return

    while True:
        try:
            query = input("\n검색 질의어 입력 >> ").strip()
            if not query or query.lower() == 'q':
                print("[SYSTEM] 벤치마크 실험을 종료합니다.")
                break
            
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
                elif version_tag == "V4_ATTENTION":
                    # [추가] 어텐션 가중치 변동 추적용 전용 로그 메트릭 출력
                    d_alpha = res.get('dynamic_alpha', 0.0)
                    d_beta = res.get('dynamic_beta', 0.0)
                    energy = res.get('attn_energy', 0.0)
                    print(f"    [Attention 특화] Dynamic Alpha: {d_alpha} | Dynamic Beta: {d_beta} | Attn Energy: {energy}")
                
                    # 컨텐트 스니펫 일부 출력 (구독성 확보)
                    snippet = res.get('content', '')[:120].replace('\n', ' ')
                    print(f"    Snippet: {snippet}...")
                    print("-" * 50)
                
        except KeyboardInterrupt:
            print("\n[SYSTEM] 인터럽트가 감지되어 세션을 종료합니다.")
            break
        except Exception as e:
            print(f"[SEARCH ERROR] 검색 처리 중 예외 발생: {e}")


if __name__ == "__main__":        
    test_hybrid_benchmark()