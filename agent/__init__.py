"""Natural-language orchestration layer.

Parses user requests (region, thresholds, lead times, ...) and drives the
data_io -> verification -> visualization -> reports pipeline. No metric
math or plotting lives here — this package only coordinates the others.
"""
