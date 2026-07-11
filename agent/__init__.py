"""Constrained natural-language orchestration for forecast verification."""

from agent.errors import AmbiguousCaseError, DataContractError, NoDataError
from agent.insight import build_insight_facts, generate_insight, render_deterministic_summary
from agent.nl_parser import parse_request
from agent.orchestrator import (
    discover_fenglei_cases,
    run_fenglei_case_verification,
    run_verification,
)
from agent.request_resolution import resolve_request
from agent.request_schema import RequestValidationError, default_request, validate_request

__all__ = [
    "run_verification",
    "run_fenglei_case_verification",
    "discover_fenglei_cases",
    "parse_request",
    "validate_request",
    "default_request",
    "resolve_request",
    "build_insight_facts",
    "render_deterministic_summary",
    "generate_insight",
    "NoDataError",
    "AmbiguousCaseError",
    "DataContractError",
    "RequestValidationError",
]
