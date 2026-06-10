import urllib.request
import urllib.parse
import json

QUERIES = [
    "웹호스팅",
    "한국영화가 아닌 재미있는 영화",
    "오늘 파이썬 코딩 공부를 하려는데 최근 본 파이썬 관련지식 사이트",
    "ResNet에서 경사 소실 문제를 해결하기 위해 도입한 지름길 구조의 수학적 원리",
    "우분투 터미널 환경에서 여러 작업을 동시에 모니터링하기 위해 화면을 분할하고 세션을 유지하는 CLI 도구 활용법"
]

def run_tests():
    report = "# 하이브리드 검색 엔진(v3) 성능 평가 보고서\n\n"
    
    for q in QUERIES:
        url = f"http://127.0.0.1:8000/api/search?q={urllib.parse.quote(q)}&limit=3"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                
                report += f"## 질의어: `{q}`\n"
                if not data:
                    report += "검색 결과 없음\n\n"
                    continue
                
                for idx, item in enumerate(data):
                    report += f"**{idx+1}위**: {item['title']}\n"
                    report += f"- URL: {item['url']}\n"
                    report += f"- 최종 스코어: {item.get('score', 0):.4f} (Lexical: {item.get('score_lex_raw', 0):.4f}, Semantic: {item.get('score_sem_raw', 0):.4f})\n\n"
                
                report += "---\n\n"
        except Exception as e:
            report += f"## 질의어: `{q}`\n오류 발생: {str(e)}\n\n---\n\n"
            
    print(report)
        
if __name__ == "__main__":
    run_tests()
    print("Report generated.")
