"""Report generation.

Assembles verification results and figures produced by the verification
and visualization packages into the artifacts written to outputs/reports/.
"""

from reports.report import save_report
from reports.downloads import (
    build_case_download_artifacts,
    describe_view,
    figure_to_png,
    figure_to_svg,
)

__all__ = [
    "save_report",
    "build_case_download_artifacts",
    "describe_view",
    "figure_to_png",
    "figure_to_svg",
]
