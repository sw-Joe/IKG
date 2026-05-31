import os
import sqlite3
from datetime import datetime
import faiss
import trafilatura
from playwright.async_api import async_playwright
from ai_core.config import IKG_DB_PATH, IKG_INDEX_PATH

class Indexer:
    def __init__(self, db_path=None, index_path=None, dim=1024):
        # 격리 설정 상수를 기본 경로로 명시적 매핑
        self.db_path = db_path or IKG_DB_PATH
        self.index_path = index_path or IKG_INDEX_PATH
        self.dim = dim
        
        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
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
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent="Mozilla/5.0 ...")
            page = await context.new_page()
            try:
                await page.goto(url, timeout=15000, wait_until="networkidle")
                content = await page.content()
                title = await page.title()
                return content, title
            except Exception:
                return None, None
            finally:
                await browser.close()

    async def index_url(self, url, embedder):
        downloaded = trafilatura.fetch_url(url)
        content = trafilatura.extract(downloaded)
        title = trafilatura.extract_metadata(downloaded).title if downloaded else "Unknown"

        if not content or len(content) < 300 or "JavaScript is disabled" in content:
            dynamic_content, dynamic_title = await self._get_dynamic_content(url)
            if dynamic_content:
                content, title = dynamic_content, dynamic_title

        if not content:
            return

        vector = embedder.encode(content)
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO bookmarks (url, title, content, created_at) VALUES (?, ?, ?, ?)",
                (url, title, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            self.conn.commit()
            self.index.add(vector.astype('float32'))
            print(f"[INDEXER SUCCESS] 메모리 세션 적재 완료: {title}")
        except sqlite3.IntegrityError:
            pass

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        print(f"[INDEXER FLUSH] 디스크 인덱스 바이너리 영속화 완료 -> {self.index_path}")