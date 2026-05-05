import os
from datetime import datetime

import sqlite3
import faiss
import numpy as np
import trafilatura



class IKGIndexer:
    def __init__(self, db_path="ikg_metadata.db", index_path="ikg_vector.index", dim=1024):
        self.db_path = db_path
        self.index_path = index_path
        self.dim = dim
        
        # 1. SQLite 초기화
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()
        
        # 2. FAISS Index 초기화 (Inner Product 사용 - 정규화 시 코사인 유사도와 동일)
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


    def add_document(self, url, embedder):
        # 1. Web Scraping
        downloaded = trafilatura.fetch_url(url)
        content = trafilatura.extract(downloaded)
        title = trafilatura.extract_metadata(downloaded).title if downloaded else "Unknown Title"
        
        if not content:
            print(f"Failed to extract content from {url}")
            return

        # 2. Embedding Generation (Step 1에서 만든 클래스 활용)
        vector = embedder.encode(content) # (1, 1024)
        
        # 3. Save Metadata to SQLite
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO bookmarks (url, title, content, created_at) VALUES (?, ?, ?, ?)",
            (url, title, content, datetime.now())
        )
        self.conn.commit()
        
        # 4. Save Vector to FAISS
        self.index.add(vector.astype('float32'))
        
        # 5. Persistence
        faiss.write_index(self.index, self.index_path)
        print(f"Successfully indexed: {title}")


    def search(self, query_vector, top_k=5):
        distances, indices = self.index.search(query_vector.astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx == -1: continue
            # rowid는 1부터 시작하므로 FAISS index_id(0부터 시작)에 +1
            cursor = self.conn.execute("SELECT url, title FROM bookmarks WHERE id = ?", (int(idx) + 1,))
            results.append(cursor.fetchone())
        return results