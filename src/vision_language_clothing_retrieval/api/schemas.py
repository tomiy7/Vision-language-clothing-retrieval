from enum import Enum
from typing import Literal

from pydantic import BaseModel


class QueryType(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class SearchResultItem(BaseModel):
    sample_id: str
    score: float

    result_type: Literal["image", "text"]

    # URL slike (npr. "/images/sample_123.jpg") ili tekstualni sadrzaj
    content: str


class SearchResponse(BaseModel):
    query_type: QueryType
    results: list[SearchResultItem]