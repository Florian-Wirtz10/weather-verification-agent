"""Constrained natural-language orchestration for forecast verification."""

from agent.errors import AmbiguousCaseError, DataContractError, NoDataError
from agent.comparison_insight import (
    build_comparison_insight_facts,
    generate_comparison_insight,
    render_comparison_deterministic_summary,
)
from agent.daily_insight import (
    build_daily_insight_facts,
    generate_daily_insight,
    render_daily_deterministic_summary,
)
from agent.daily_request import default_daily_request, parse_daily_request, validate_daily_request
from agent.insight import build_insight_facts, generate_insight, render_deterministic_summary
from agent.nl_parser import parse_request
from agent.multimodel_request import (
    default_multimodel_request,
    parse_multimodel_request,
    validate_multimodel_request,
)
from agent.orchestrator import (
    discover_fenglei_cases,
    run_fenglei_case_verification,
    run_verification,
)
from agent.presentation import resolve_media_layout
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
    "resolve_media_layout",
    "NoDataError",
    "AmbiguousCaseError",
    "DataContractError",
    "RequestValidationError",
    "default_daily_request",
    "parse_daily_request",
    "validate_daily_request",
    "build_daily_insight_facts",
    "render_daily_deterministic_summary",
    "generate_daily_insight",
    "build_comparison_insight_facts",
    "render_comparison_deterministic_summary",
    "generate_comparison_insight",
    "default_multimodel_request",
    "parse_multimodel_request",
    "validate_multimodel_request",
]
