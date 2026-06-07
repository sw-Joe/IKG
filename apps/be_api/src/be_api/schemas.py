from pydantic import BaseModel, HttpUrl, Field
from typing import Optional
from datetime import datetime



class BookmarkCreateRequest(BaseModel):
    """신규 등록 및 수정을 위한 데이터 검증 및 유효성 바인딩 Pydantic 모델"""
    url: HttpUrl = Field(..., description="올바른 형식의 웹페이지 URL (HTTP/HTTPS 필수)")
    title: str = Field(..., min_length=1, max_length=255, description="북마크 제목 (공백 불가)")
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
    """비동기 인덱싱 태스크 이관 접수 완료 영수증 응답 모델"""
    message: str
    bookmark_id: int
    task_id: str
    status: str = "accepted"
    timestamp: datetime = Field(default_factory=datetime.now)