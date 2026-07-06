"""Report generation.

Assembles verification results and figures produced by the verification
and visualization packages into the artifacts written to outputs/reports/.
"""

from reports.report import save_report

__all__ = ["save_report"]
