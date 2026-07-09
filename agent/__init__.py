"""Natural-language orchestration layer.

Parses user requests (region, thresholds, lead times, ...) and drives the
data_io -> verification -> visualization -> reports pipeline. No metric
math or plotting lives here — this package only coordinates the others.
"""

from agent.nl_parser import PLOT_MENU, parse_request, select_figures
from agent.orchestrator import run_fenglei_case_verification, run_verification

__all__ = [
    "run_verification",
    "run_fenglei_case_verification",
    "parse_request",
    "select_figures",
    "PLOT_MENU",
]
