from pydantic import BaseModel, HttpUrl, Field
from typing import Optional
from datetime import datetime

class BookmarkCreateRequest(BaseModel):
    """신규 북마크 데이터 검증 및 유효성 바인딩용 Pydantic 모델 (In-Scope 가드레일)"""
    url: HttpUrl = Field(..., description="올바른 형식의 웹페이지 URL (HTTP/HTTPS 필수)")
    title: str = Field(..., min_length=1, max_length=255, description="북마크 제목 (공백 허용 안 됨)")
    content: str = Field(..., min_length=5, description="본문 텍스트 컨텐츠 (최소 5자 이상 필수)")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://pytorch.org/docs/stable/nn.html",
                "title": "PyTorch NN Module Documentation",
                "content": "Detailed structural overview of Neural Network layers and loss functions in PyTorch runtime."
            }
        }

class TaskReceiptResponse(BaseModel):
    """유저에게 반환하는 즉각적인 비동기 태스크 접수 영수증"""
    message: str
    bookmark_id: int
    task_id: str
    status: str = "accepted"
    timestamp: datetime = Field(default_factory=datetime.now)