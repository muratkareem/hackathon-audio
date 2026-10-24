from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Segment(StrictModel):
    id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1, max_length=6000)
    speaker: str | None = None
    language: str | None = None

    @model_validator(mode="after")
    def timestamps(self):
        if self.end < self.start:
            raise ValueError("end precedes start")
        return self


class Evidence(StrictModel):
    text: str = Field(min_length=1, max_length=1500)
    references: list[int] = Field(min_length=1, max_length=20)


class Action(Evidence):
    owner: str | None = Field(default=None, max_length=100)
    deadline: str | None = Field(default=None, max_length=100)


class Report(StrictModel):
    overview: str = Field(max_length=2000)
    topics: list[str] = Field(max_length=12)
    discussion: list[Evidence] = Field(max_length=20)
    decisions: list[Evidence] = Field(max_length=20)
    actions: list[Action] = Field(max_length=30)
    open_questions: list[Evidence] = Field(max_length=20)
    takeaways: list[str] = Field(max_length=12)


class Answer(StrictModel):
    answer: str = Field(max_length=4000)
    references: list[int] = Field(max_length=20)
    insufficient_evidence: bool


class ChatRequest(StrictModel):
    question: str = Field(min_length=1, max_length=1000)
    language: Literal["en", "ru"] = "en"


class RenameRequest(StrictModel):
    names: dict[str, str]

    @model_validator(mode="after")
    def bounded(self):
        if len(self.names) > 100 or any(
            not v.strip() or len(v) > 80 for v in self.names.values()
        ):
            raise ValueError("Invalid speaker names")
        return self
