import asyncio
from datetime import datetime
import os

import sqlite3
import faiss
import numpy as np
import trafilatura
from playwright.async_api import async_playwright



class IKGIndexer:
    def __init__(self, db_path="ikg_metadata.db", index_path="ikg_vector.index", dim=1024):
        self.db_path = db_path
        self.index_path = index_path
        self.dim = dim
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()
        
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(self.dim)


    def _create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            content TEXT,
            created_at TIMESTAMP
        )
        """
        self.conn.execute(query)
        self.conn.commit()


    async def _get_dynamic_content(self, url):
        """Playwright를 이용한 동적 콘텐츠 수집 (Fallback)"""
        async with async_playwright() as p:
            # 리소스 절약을 위해 브라우저 옵션 최적화
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent="Mozilla/5.0 ...")
            page = await context.new_page()
            
            # 이미지/폰트/스타일시트 로드 차단 (속도 및 리소스 최적화)
            await page.route("**/*", lambda route: 
                route.abort() if route.request.resource_type in ["image", "media", "font"] 
                else route.continue_())

            try:
                # 수정: networkidle 대신 domcontentloaded 사용 및 타임아웃 단축
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                # 본문 렌더링을 위한 최소한의 추가 대기 (선택 사항)
                await page.wait_for_timeout(2000)

                # 렌더링된 HTML을 trafilatura로 다시 정밀 추출
                raw_html = await page.content()
                content = trafilatura.extract(raw_html)
                title = await page.title()

                return content, title
            
            except Exception as e:
                print(f"Dynamic scraping failed for {url}: {e}")

                return None, None
            
            finally:
                await browser.close()


    async def add_document(self, url, embedder):
        # 1단계: Trafilatura (정적 분석)
        downloaded = trafilatura.fetch_url(url)
        content = trafilatura.extract(downloaded)
        title = trafilatura.extract_metadata(downloaded).title if downloaded else "Unknown"

        # 2단계: 실패 시 Playwright (동적 분석)로 전환
        # (콘텐츠가 너무 짧거나 특정 에러 문구가 포함된 경우)
        if not content or len(content) < 300 or "JavaScript is disabled" in content:
            print(f"Low quality content for {url}. Switching to Playwright...")
            dynamic_content, dynamic_title = await self._get_dynamic_content(url)
            if dynamic_content:
                content, title = dynamic_content, dynamic_title

        if not content:
            print(f"Failed to extract content from {url} (All tiers failed)")
            return

        # 3단계: 임베딩 및 저장
        vector = embedder.encode(content)
        
        try:
            # DB 저장 로직 유지
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO bookmarks (url, title, content, created_at) VALUES (?, ?, ?, ?)",
                (url, title, content, datetime.now())
            )
            self.conn.commit()
            
            self.index.add(vector.astype('float32'))
            # 수정: 여기서 faiss.write_index를 매번 호출하지 않고 삭제합니다 (I/O 병목 제거)
            print(f"Successfully indexed in memory: {title}")

        except sqlite3.IntegrityError:
            # Skipping duplicate URL
            pass


    def save_index(self):
        """인덱싱 완료 후 한 번에 파일로 저장하는 메서드 추가"""
        faiss.write_index(self.index, self.index_path)
        print(f"Index successfully saved to {self.index_path}")