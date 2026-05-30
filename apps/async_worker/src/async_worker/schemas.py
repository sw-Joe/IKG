from datetime import datetime

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional



class BookmarkCreateRequest(BaseModel):
    """신규 북마크 데이터 검증 및 유효성 바인딩용 Pydantic 모델"""
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
    """유저에게 반환하는 즉각적인 비동기 태스크 접수 영수증 (대안 B 호환)"""
    message: str
    bookmark_id: int
    task_id: str
    status: str = "accepted"
    timestamp: datetime = Field(default_factory=datetime.now)


class SearchResultEntry(BaseModel):
    """검색 결과 단일 항목 스키마"""
    id: int
    url: str
    title: str
    content: str
    created_at: str
    score_final: float
    score_lex: float
    score_sem: float
    factor_time: float
    factor_gate: float
    dynamic_alpha: float
    dynamic_beta: float
    attn_energy: float
    factor_rank_penalty: Optional[float] = None


class SearchResponse(BaseModel):
    """전체 검색 결과 목록 반환 스키마"""
    query: str
    results: list[SearchResultEntry]