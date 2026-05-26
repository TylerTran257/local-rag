from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=3, gt=0, le=10)


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=3, gt=0, le=10)
