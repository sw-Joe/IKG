from hybrid_search import HybridSearcher


def test_hybrid():
    searcher = HybridSearcher()
    
    while True:
        query = input("\n하이브리드 검색어 입력 (종료: q): ")
        if query.lower() == 'q':
            break
        
        results = searcher.search(query, top_n=5)
        
        print(f"\n--- '{query}' 통합 검색 결과 ---")
        for i, res in enumerate(results):
            print(f"[{i+1}] {res['title']}")
            print(f"    URL: {res['url']}")
            # 본문 앞부분만 살짝 출력 (디버깅용)
            print(f"    Snippet: {res['content'][:80]}...\n")


if __name__ == "__main__":
    test_hybrid()