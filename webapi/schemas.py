"""Validated API request bodies."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from webapi.config import CASES, model_family


Scope = Literal["all", "custom", "single"]
InitializationId = Annotated[str, Field(min_length=1, max_length=128)]


class SelectionRequest(BaseModel):
    case_id: str
    models: list[str] = Field(min_length=1, max_length=16)
    complete_only: bool = True

    @model_validator(mode="after")
    def validate_selection(self):
        if len(set(self.models)) != len(self.models):
            raise ValueError("models must be unique")
        if self.case_id not in CASES:
            raise ValueError("unknown case")
        allowed = set(CASES[self.case_id]["models"])
        unknown = set(self.models).difference(allowed)
        if unknown:
            raise ValueError(f"models are unavailable for this case: {sorted(unknown)}")
        model_family(self.models)
        return self


class PromptRequest(SelectionRequest):
    scope: Scope


class CommonTimeRequest(SelectionRequest):
    initialization_ids: list[InitializationId] = Field(min_length=2, max_length=8)

    @field_validator("initialization_ids")
    @classmethod
    def unique_ids(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("initialization IDs must be unique")
        return value


class TaskRequest(PromptRequest):
    bbox: list[float] = Field(default_factory=lambda: [29.0, 33.5, 108.0, 116.5])
    initialization_ids: list[InitializationId] = Field(default_factory=list, max_length=8)
    comparison_valid_time: str | None = Field(default=None, max_length=64)
    prompt: str = Field(default="", max_length=10000)
    use_default_plan: bool = True

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value):
        if len(value) != 4:
            raise ValueError("bbox must contain lat_min, lat_max, lon_min, lon_max")
        lat_min, lat_max, lon_min, lon_max = map(float, value)
        if not (-90 <= lat_min < lat_max <= 90 and -180 <= lon_min < lon_max <= 180):
            raise ValueError("invalid geographic bounds")
        return [lat_min, lat_max, lon_min, lon_max]

    @model_validator(mode="after")
    def validate_scope_ids(self):
        if len(set(self.initialization_ids)) != len(self.initialization_ids):
            raise ValueError("initialization IDs must be unique")
        if self.scope == "single" and len(self.initialization_ids) != 1:
            raise ValueError("single scope requires exactly one initialization")
        if self.scope == "custom" and not 2 <= len(self.initialization_ids) <= 8:
            raise ValueError("custom scope requires 2 to 8 initializations")
        return self
