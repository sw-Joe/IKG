from hybrid_search_lagacy.attention import AttentionHybridSearcher
from hybrid_search import HybridSearcher
from hybrid_search_lagacy.hybrid_search_v2 import HybridSearcherV2
from hybrid_search_lagacy.native_sparse_embedding_search import SearcherNativeSparse
from hybrid_search_lagacy.rank_scaled_score_fusion import HybridSearcherV3RankScaled
from hybrid_search_lagacy.two_stage_cascading_ensemble import HybridSearcherV3Stage1


def test_hybrid_benchmark():
    print("=" * 60)
    print(" [IKG-Search] 하이브리드 검색 알고리즘 대조 실험 벤치마크 v3")
    print("=" * 60)
    print("1: Baseline v2 (정적 가중치 합성 + 최신성 감쇄 + 렉시컬 게이트)")
    print("2: v3 후보 1 (2-Stage Cascading 앙상블 파이프라인)")
    print("3: v3 후보 2 (순위 변조형 가중치 합산)")
    print("4: v3 후보 3 (BGE-M3 Native Sparse Embedding 활용)")
    print("5: v4 확장안 (Attention 기반 동적 가중치 융합 - 전역 루프형)")
    print("6: v3 최종안 (레이어 분리형 차세대 하이브리드 파이프라인)  # [추가]")
    print("=" * 60)
    
    try:
        choice = input("테스트할 모듈 번호를 선택하세요 (1~6): ").strip()
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
            print("[SYSTEM] v4 확장안 (전역 Attention) 모듈을 초기화합니다.")
            searcher = AttentionHybridSearcher()
            version_tag = "V4_ATTENTION"
        elif choice == '6':
            print("[SYSTEM] v3 최종안 (레이어 분리형 통합) 모듈을 초기화합니다.")
            searcher = HybridSearcher()
            version_tag = "V3_FINAL"
        else:
            print("[ERROR] 올바른 번호를 선택하세요 (1~6).")
            return
    except Exception as e:
        print(f"[INIT ERROR] 인덱스 및 모델 로드 중 실패: {e}")
        return

    print(f"\n[{version_tag}] 검색어 입력 (종료: 'q' 또는 엔터)")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n검색 질의어 입력 >> ").strip()
            if not query or query.lower() == 'q':
                print("[SYSTEM] 벤치마크 실험을 종료합니다.")
                break
                
            # 규격화된 top_n 검색 인터페이스 호출
            results = searcher.search(query, top_n=5)
            
            if not results:
                print("[INFO] 질의와 일치하는 검색 결과가 존재하지 않거나, 3단계 외곽 검문소(Zero-Hits Filter)에서 차단되었습니다.")
                continue
                
            for i, res in enumerate(results):
                print(f"[{i+1}] {res['title']}")
                print(f"    URL: {res['url']}")
                
                # 공통 정형 지표 안전 추출 레이어
                f_score = res.get('score_final', 0.0)
                l_score = res.get('score_lex', 0.0)
                s_score = res.get('score_sem', 0.0)
                t_factor = res.get('factor_time', 1.0)
                g_factor = res.get('factor_gate', 1.0)
                
                print(f"    [Scores] Final: {f_score} | Lexical: {l_score} | Semantic: {s_score}")
                print(f"    [Factors] Time Decay: {t_factor} | Lexical Gate: {g_factor}")
                
                # 아키텍처별 전용 지표 분석 가시화 구간
                if version_tag == "V3_STAGE1" and 'rrf_score' in res:
                    print(f"    [Stage1 특화] RRF Score: {res['rrf_score']}")
                    
                elif version_tag == "V3_RANK_SCALED" and 'factor_rank_penalty' in res:
                    print(f"    [RankScaled 특화] Rank Penalty 인자: {res['factor_rank_penalty']} (Lex순위: {res.get('rank_lex')}, Sem순위: {res.get('rank_sem')})")
                    
                elif version_tag == "V4_ATTENTION":
                    print(f"    [Attention 특화] Dynamic Alpha: {res.get('dynamic_alpha')} | Dynamic Beta: {res.get('dynamic_beta')} | Attn Energy: {res.get('attn_energy')}")
                    
                elif version_tag == "V3_FINAL":
                    # [최종안 전용 디버깅 지표 출력]
                    d_alpha = res.get('dynamic_alpha', 0.0)
                    d_beta = res.get('dynamic_beta', 0.0)
                    energy = res.get('attn_energy', 0.0)
                    penalty = res.get('factor_rank_penalty', 1.0)
                    print(f"    [V3_FINAL 코어 어텐션] Dynamic Alpha: {d_alpha} | Dynamic Beta: {d_beta} | Attn Energy: {energy}")
                    print(f"    [V3_FINAL 외곽 검문소] 1위 보증 Rank Penalty: {penalty}")
                
                # 가독성 확보용 본문 스니펫 가공 출력
                snippet = res.get('content', '')[:120].replace('\n', ' ')
                print(f"    Snippet: {snippet}...")
                print("-" * 60)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[SEARCH ERROR] 검색 처리 중 예외 발생: {e}")


if __name__ == "__main__":
    test_hybrid_benchmark()