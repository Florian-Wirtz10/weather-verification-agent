"""Matplotlib figure builders for forecast/observation/error fields.

Real-data plots are deterministic renderers selected through the Phase 3
request schema; the language model never writes plotting code.
"""

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.ticker import MultipleLocator
from PIL import Image

from visualization.geography import add_admin_boundaries


def plot_field(array, title, cmap="Blues"):
    """Render a single 2D field as a heatmap figure."""
    fig, ax = plt.subplots()
    im = ax.imshow(array, cmap=cmap, origin="lower")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="mm")
    return fig


def plot_error(forecast, obs, title="Forecast - Obs"):
    """Render the forecast-minus-obs error field with a diverging, zero-centered colormap."""
    error = forecast - obs
    limit = float(np.max(np.abs(error))) or 1.0

    fig, ax = plt.subplots()
    im = ax.imshow(error, cmap="RdBu_r", origin="lower", vmin=-limit, vmax=limit)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="mm")
    return fig


# ---------------------------------------------------------------------------
# Real-data (geo-referenced) plots. Fixed categorical hue order —
# never cycled/reassigned per-series — per the dataviz skill's color formula.
# ---------------------------------------------------------------------------

CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
_MUTED = "#898781"

# Common threshold reflectivity colour table supplied by the course reference:
# 1, 5, 10, 15 ... 75 dBZ. Values below 1 dBZ (including the display-floor
# 0 dBZ no-echo background) are deliberately white, matching the reference.
_REFLECTIVITY_BOUNDS = np.array(
    [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75], dtype=float
)
_REFLECTIVITY_COLORS = [
    "#94f75f", "#62d53f", "#3f8e27", "#ffff54", "#ffe400", "#e1c140",
    "#fbaa41", "#f98921", "#fd6841", "#ea3323", "#c4291c", "#b02418",
    "#ea33e8", "#891aae", "#a891ea",
]
_REFLECTIVITY_DISPLAY_FLOOR_DBZ = 0.0


def _reflectivity_cmap_norm(minimum_display_dbz=None):
    start = 0
    if minimum_display_dbz is not None:
        start = int(np.searchsorted(_REFLECTIVITY_BOUNDS, float(minimum_display_dbz)))
        start = min(max(start, 0), len(_REFLECTIVITY_COLORS) - 1)
    bounds = _REFLECTIVITY_BOUNDS[start:]
    colors = _REFLECTIVITY_COLORS[start:]
    cmap = ListedColormap(
        colors,
        under="#ffffff",
        over=colors[-1],
        bad="#ffffff",
    )
    norm = BoundaryNorm(bounds, len(colors), clip=False)
    return cmap, norm


def _reflectivity_display_data(data, minimum_display_dbz=None):
    """Return a visual-only reflectivity representation on a common no-echo scale.

    The 风雷 REFMOSAIC product encodes its no-echo background as a negative
    sentinel (currently -18 dBZ), while CREF uses 0 dBZ.  Floor negative
    values to 0 solely for spatial display and display-error rendering so the
    two panels use the same no-echo convention.  Verification data and scores
    are deliberately left unchanged.
    """
    values = np.asarray(np.ma.filled(data, np.nan), dtype=float)
    values = np.maximum(values, _REFLECTIVITY_DISPLAY_FLOOR_DBZ)
    if minimum_display_dbz is not None:
        values = values.copy()
        values[values < float(minimum_display_dbz)] = np.nan
    return values


def _diverging_cmap():
    return LinearSegmentedColormap.from_list("diverging_blue_red", ["#2a78d6", "#f0efec", "#e34948"])


def _style_axes(ax):
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_AXIS)
    ax.spines["bottom"].set_color(_AXIS)
    ax.tick_params(colors=_MUTED)


def plot_reflectivity_field(field, title=None):
    """Render a GriddedField of dBZ reflectivity on its lat/lon extent, using the
    standard 18-level radar color table."""
    cmap, norm = _reflectivity_cmap_norm()
    extent = (
        float(field.lon.min()),
        float(field.lon.max()),
        float(field.lat.min()),
        float(field.lat.max()),
    )

    fig, ax = plt.subplots()
    im = ax.imshow(
        _reflectivity_display_data(field.data),
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    ax.set_title(title or f"{field.product} ({field.valid_time:%Y-%m-%d %H:%M} UTC)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    add_admin_boundaries(ax, extent)
    plt.colorbar(
        im,
        ax=ax,
        label="dBZ (display floor: 0)",
        ticks=_REFLECTIVITY_BOUNDS,
    )
    return fig


def plot_reflectivity_comparison(forecast_field, obs_field, title=None):
    """Render forecast and observation side by side as one auditable figure."""
    cmap, norm = _reflectivity_cmap_norm()
    extent = (
        float(obs_field.lon.min()),
        float(obs_field.lon.max()),
        float(obs_field.lat.min()),
        float(obs_field.lat.max()),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    im = None
    for ax, field, panel_title in (
        (axes[0], forecast_field, "Forecast"),
        (axes[1], obs_field, "Observation"),
    ):
        im = ax.imshow(
            _reflectivity_display_data(field.data),
            cmap=cmap,
            norm=norm,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        ax.set_title(panel_title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        add_admin_boundaries(ax, extent)
    if im is not None:
        plt.colorbar(
            im,
            ax=list(axes),
            label="dBZ (display floor: 0)",
            ticks=_REFLECTIVITY_BOUNDS,
        )
    fig.suptitle(
        title
        or f"Forecast vs observation ({forecast_field.valid_time:%Y-%m-%d %H:%M} UTC)"
    )
    return fig


def plot_reflectivity_error(forecast_field, obs_field, title="Forecast - Obs (dBZ)"):
    """Render the forecast-minus-obs reflectivity error, cropped/regridded onto the
    same grid (see data_io.grid.align_pair), with a zero-centered diverging colormap."""
    diff = _reflectivity_display_data(forecast_field.data) - _reflectivity_display_data(
        obs_field.data
    )
    finite_diff = diff[np.isfinite(diff)]
    limit = float(np.max(np.abs(finite_diff))) if finite_diff.size else 1.0
    limit = limit or 1.0
    extent = (
        float(obs_field.lon.min()),
        float(obs_field.lon.max()),
        float(obs_field.lat.min()),
        float(obs_field.lat.max()),
    )

    fig, ax = plt.subplots()
    im = ax.imshow(
        diff, cmap=_diverging_cmap(), origin="lower", extent=extent,
        vmin=-limit, vmax=limit, aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    add_admin_boundaries(ax, extent)
    plt.colorbar(im, ax=ax, label="dBZ error (display floor: 0)")
    return fig


def render_reflectivity_animation(frames, *, fps=2):
    """Render deterministic forecast/observation comparison frames to an in-memory GIF.

    ``frames`` is an ordered sequence of mappings with ``lead_minutes``,
    ``forecast`` and ``obs`` entries.  The caller decides which already-matched
    frames are allowed; this renderer never reads data or chooses times.
    """
    if not frames:
        raise ValueError("reflectivity animation requires at least one frame")
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 4:
        raise ValueError("fps must be an integer from 1 to 4")

    figures = (
        plot_reflectivity_comparison(
            frame["forecast"],
            frame["obs"],
            title=f"Forecast vs observation — lead {frame['lead_minutes']} min",
        )
        for frame in frames
    )
    return _figures_to_gif(figures, fps=fps)


def render_linked_score_animation(
    frames,
    *,
    lead_minutes,
    scores_by_threshold,
    score_name,
    fps=2,
):
    """Render one GIF linking spatial evolution to a fixed score curve.

    Every frame contains forecast/observation panels plus the complete selected
    score time series.  A vertical cursor and highlighted points identify the
    exact lead represented by the spatial fields, so the relationship is
    explicit without relying on a separate neighbouring chart.
    """
    if not frames:
        raise ValueError("linked score animation requires at least one frame")
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 4:
        raise ValueError("fps must be an integer from 1 to 4")
    expected_leads = [frame["lead_minutes"] for frame in frames]
    if any(lead not in lead_minutes for lead in expected_leads):
        raise ValueError("linked animation frames must occur on the score time axis")

    figures = (
        _plot_linked_score_frame(
                frame,
                lead_minutes=lead_minutes,
                scores_by_threshold=scores_by_threshold,
                score_name=score_name,
            )
        for frame in frames
    )
    return _figures_to_gif(figures, fps=fps)


def _figures_to_gif(figures, *, fps):
    """Encode and close a sequence of equally sized Matplotlib figures."""
    images = []
    for figure in figures:
        buffer = BytesIO()
        try:
            figure.savefig(buffer, format="png", dpi=100, facecolor="white")
            with Image.open(BytesIO(buffer.getvalue())) as image:
                images.append(image.convert("P", palette=Image.Palette.ADAPTIVE))
        finally:
            plt.close(figure)

    output = BytesIO()
    images[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
    )
    return output.getvalue()


def _plot_linked_score_frame(frame, *, lead_minutes, scores_by_threshold, score_name):
    """Build one deterministic spatial-plus-score storyboard frame."""
    forecast, obs = frame["forecast"], frame["obs"]
    extent = (
        float(obs.lon.min()), float(obs.lon.max()), float(obs.lat.min()), float(obs.lat.max())
    )
    cmap, norm = _reflectivity_cmap_norm()
    # A two-row storyboard gives each geographic panel useful map area while
    # keeping the entire score curve readable.  The colourbar has an explicit
    # GridSpec slot: never create it first and then reposition parent axes.
    # Keep the encoded GIF below Streamlit's 1200 px resize threshold. The
    # browser can then stretch the original multi-frame GIF without flattening.
    fig = plt.figure(figsize=(10, 7), layout="constrained")
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1, 1, 0.055), height_ratios=(1.15, 0.85)
    )
    map_axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    colorbar_axis = fig.add_subplot(grid[0, 2])
    score_axis = fig.add_subplot(grid[1, :])
    image = None
    for axis, field, title in (
        (map_axes[0], forecast, "Forecast"),
        (map_axes[1], obs, "Observation"),
    ):
        image = axis.imshow(
            _reflectivity_display_data(field.data), cmap=cmap, norm=norm,
            origin="lower", extent=extent, aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        add_admin_boundaries(axis, extent)
    if image is not None:
        fig.colorbar(
            image, cax=colorbar_axis, label="dBZ (display floor: 0)",
            ticks=_REFLECTIVITY_BOUNDS,
        )

    current_lead = frame["lead_minutes"]
    for index, threshold in enumerate(sorted(scores_by_threshold)):
        values = np.asarray(scores_by_threshold[threshold], dtype=float)
        color = CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)]
        score_axis.plot(
            lead_minutes, values, marker="o", markersize=4, linewidth=2,
            color=color, label=f"{threshold:g} dBZ",
        )
        current_index = list(lead_minutes).index(current_lead)
        if np.isfinite(values[current_index]):
            score_axis.scatter(
                [current_lead], [values[current_index]], s=70, color=color,
                edgecolors="black", linewidths=0.8, zorder=3,
            )
    score_axis.axvline(current_lead, color="#424242", linestyle="--", linewidth=1.4)
    score_axis.set_title(f"{score_name.upper()} vs lead time")
    score_axis.set_xlabel("Forecast lead time (min)")
    score_axis.set_ylabel(score_name.upper())
    score_axis.xaxis.set_major_locator(MultipleLocator(30))
    _style_axes(score_axis)
    score_axis.legend(title="Threshold", frameon=False, fontsize=8)
    fig.suptitle(
        f"Forecast / observation and {score_name.upper()} — lead {current_lead} min",
        fontsize=13,
    )
    return fig


def plot_score_time_series(lead_minutes, scores_by_threshold, score_name, title=None):
    """Line chart of one categorical score vs forecast lead time, one line per
    threshold in a fixed hue order (never reassigned/cycled per call).

    `scores_by_threshold`: {threshold: [score value per lead_minutes entry]}.
    """
    fig, ax = plt.subplots()
    for i, threshold in enumerate(sorted(scores_by_threshold)):
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        x_values, y_values = _break_time_gaps(
            lead_minutes, scores_by_threshold[threshold]
        )
        ax.plot(
            x_values,
            y_values,
            marker="o", markersize=5, linewidth=2, color=color, label=str(threshold),
        )
    ax.set_xlabel("Forecast lead time (min)")
    ax.set_ylabel(score_name.upper())
    ax.set_title(title or f"{score_name.upper()} vs lead time")
    ax.xaxis.set_major_locator(MultipleLocator(30))
    _style_axes(ax)
    ax.legend(title="Threshold", frameon=False)
    return fig


def plot_forecast_observed_rate_trajectory(
    lead_minutes, contingency_tables, threshold_dbz, title=None
):
    """Plot one initialization as a joint rate trajectory with marginal densities.

    Each threshold is rendered by a separate call/figure.  This preserves the
    approved joint-plot structure without combining event definitions that
    naturally occupy different rate ranges.
    """
    leads = np.asarray(lead_minutes, dtype=float)
    if leads.ndim != 1 or leads.size == 0:
        raise ValueError("lead_minutes must be a non-empty one-dimensional sequence")
    observed, forecast = _event_rates(contingency_tables, leads)
    finite = np.isfinite(observed) & np.isfinite(forecast)
    if not np.any(finite):
        raise ValueError("rate trajectory has no finite matched samples")

    fig = plt.figure(figsize=(9.5, 8.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(4.4, 1.15),
        height_ratios=(4.4, 1.15),
        left=0.10,
        right=0.94,
        bottom=0.11,
        top=0.87,
        wspace=0.12,
        hspace=0.20,
    )
    joint = fig.add_subplot(grid[0, 0])
    forecast_density = fig.add_subplot(grid[0, 1], sharey=joint)
    observed_density = fig.add_subplot(grid[1, 0], sharex=joint)
    colorbar_axis = fig.add_subplot(grid[1, 1])

    lead_cmap, lead_norm, lead_ticks = _continuous_lead_scale(leads[finite])

    joint.plot(
        observed[finite],
        forecast[finite],
        color="#6f7780",
        linewidth=1.0,
        alpha=0.7,
        zorder=1,
    )
    scatter = joint.scatter(
        observed[finite],
        forecast[finite],
        c=leads[finite],
        cmap=lead_cmap,
        norm=lead_norm,
        s=38,
        edgecolors="#30343b",
        linewidths=0.45,
        zorder=2,
    )
    joint.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1.1)
    joint.set_xlim(0, 1)
    joint.set_ylim(0, 1)
    joint.set_xlabel("Observed rate")
    joint.set_ylabel("Forecast rate")
    joint.set_title(
        f"(a) {float(threshold_dbz):g} dBZ | {int(finite.sum())} matched leads",
        loc="left",
    )
    joint.grid(False)
    joint.tick_params(direction="in", top=True, right=True)

    x, x_density = _smooth_rate_density(observed[finite])
    y, y_density = _smooth_rate_density(forecast[finite])
    observed_density.fill_between(x, 0, x_density, color="#2a78d6", alpha=0.38)
    observed_density.plot(x, x_density, color="#2a78d6", linewidth=1.25)
    forecast_density.fill_betweenx(y, 0, y_density, color="#2a78d6", alpha=0.38)
    forecast_density.plot(y_density, y, color="#2a78d6", linewidth=1.25)
    observed_density.set_xlabel("Observed rate")
    observed_density.set_ylabel("Density")
    observed_density.grid(False)
    observed_density.tick_params(direction="in")
    forecast_density.set_xlabel("Density")
    forecast_density.grid(False)
    forecast_density.tick_params(
        axis="y", labelleft=False, labelright=True, direction="in"
    )
    forecast_density.tick_params(axis="x", direction="in")

    colorbar = fig.colorbar(scatter, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks(lead_ticks)
    colorbar.set_label("Forecast lead time (min)")
    colorbar_axis.set_box_aspect(0.22)
    fig.suptitle(
        title or "Forecast rate vs observed rate with marginal distributions",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.10,
        0.03,
        "Each point is one matched lead; the thin line follows lead-time order.",
        fontsize=9.2,
        color="#565d66",
    )
    return fig


def plot_forecast_observed_rate_threshold_comparison(
    lead_minutes, contingency_tables_by_threshold, title=None
):
    """Compare several event thresholds in one joint rate plot.

    Colour represents threshold rather than lead time.  Each threshold keeps
    its own ordered trajectory and marginal densities, which makes the change
    in event definition visible without pretending that all points are one
    homogeneous sample.
    """
    leads = np.asarray(lead_minutes, dtype=float)
    if leads.ndim != 1 or leads.size == 0:
        raise ValueError("lead_minutes must be a non-empty one-dimensional sequence")
    if not contingency_tables_by_threshold:
        raise ValueError("at least one threshold is required")

    fig = plt.figure(figsize=(9.5, 8.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(4.4, 1.15),
        height_ratios=(4.4, 1.15),
        left=0.10,
        right=0.94,
        bottom=0.11,
        top=0.87,
        wspace=0.12,
        hspace=0.20,
    )
    joint = fig.add_subplot(grid[0, 0])
    forecast_density = fig.add_subplot(grid[0, 1], sharey=joint)
    observed_density = fig.add_subplot(grid[1, 0], sharex=joint)
    legend_axis = fig.add_subplot(grid[1, 1])
    plotted = 0

    for index, threshold in enumerate(sorted(contingency_tables_by_threshold)):
        observed, forecast = _event_rates(contingency_tables_by_threshold[threshold], leads)
        finite = np.isfinite(observed) & np.isfinite(forecast)
        if not np.any(finite):
            continue
        color = CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)]
        label = f"{float(threshold):g} dBZ"
        joint.plot(
            observed[finite], forecast[finite], color=color, linewidth=1.0, alpha=0.72
        )
        joint.scatter(
            observed[finite],
            forecast[finite],
            s=34,
            color=color,
            label=label,
            edgecolors="#30343b",
            linewidths=0.4,
            zorder=2,
        )
        x, x_density = _smooth_rate_density(observed[finite])
        y, y_density = _smooth_rate_density(forecast[finite])
        observed_density.fill_between(x, 0, x_density, color=color, alpha=0.28)
        observed_density.plot(x, x_density, color=color, linewidth=1.2)
        forecast_density.fill_betweenx(y, 0, y_density, color=color, alpha=0.28)
        forecast_density.plot(y_density, y, color=color, linewidth=1.2)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError("threshold comparison has no finite matched samples")

    joint.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1.1)
    joint.set_xlim(0, 1)
    joint.set_ylim(0, 1)
    joint.set_xlabel("Observed rate")
    joint.set_ylabel("Forecast rate")
    joint.set_title(
        f"Threshold comparison | {int(leads.size)} matched leads", loc="left"
    )
    joint.grid(False)
    joint.tick_params(direction="in", top=True, right=True)

    observed_density.set_xlabel("Observed rate")
    observed_density.set_ylabel("Density")
    observed_density.grid(False)
    observed_density.tick_params(direction="in")
    forecast_density.set_xlabel("Density")
    forecast_density.grid(False)
    forecast_density.tick_params(
        axis="y", labelleft=False, labelright=True, direction="in"
    )
    forecast_density.tick_params(axis="x", direction="in")
    legend_axis.axis("off")
    handles, labels = joint.get_legend_handles_labels()
    legend_axis.legend(handles, labels, title="Threshold", loc="center", frameon=False)
    fig.suptitle(
        title or "Forecast rate vs observed rate by threshold",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.10,
        0.03,
        "Each point is one matched lead; each thin line follows lead-time order.",
        fontsize=9.2,
        color="#565d66",
    )
    return fig


def _event_rates(contingency_tables, leads):
    tables = list(contingency_tables)
    if len(tables) != leads.size:
        raise ValueError("contingency_tables must contain one table per lead")
    observed_rates = []
    forecast_rates = []
    for table in tables:
        valid_count = int(table.get("valid_count", 0))
        if valid_count <= 0:
            observed_rates.append(np.nan)
            forecast_rates.append(np.nan)
            continue
        observed_rates.append((int(table["hits"]) + int(table["misses"])) / valid_count)
        forecast_rates.append(
            (int(table["hits"]) + int(table["false_alarms"])) / valid_count
        )
    return np.asarray(observed_rates, dtype=float), np.asarray(forecast_rates, dtype=float)


def _continuous_lead_scale(lead_minutes):
    """Return a restrained blue-to-red scale for ordered forecast leads."""
    leads = np.unique(np.asarray(lead_minutes, dtype=float))
    leads = leads[np.isfinite(leads)]
    if leads.size == 0:
        raise ValueError("at least one finite forecast lead is required")
    cmap = LinearSegmentedColormap.from_list(
        "forecast_lead_blue_red", ("#2a78d6", "#e34948")
    )
    lower = float(leads[0])
    upper = float(leads[-1]) if leads.size > 1 else lower + 1.0
    norm = Normalize(vmin=lower, vmax=upper)
    preferred_ticks = (6, 60, 120, 180)
    ticks = [value for value in preferred_ticks if np.any(np.isclose(leads, value))]
    for endpoint in (leads[0], leads[-1]):
        if not any(np.isclose(endpoint, tick) for tick in ticks):
            ticks.append(float(endpoint))
    return cmap, norm, sorted(ticks)


def _smooth_rate_density(values, *, bins=80, sigma_bins=2.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    histogram, edges = np.histogram(values, bins=bins, range=(0, 1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radius = max(2, int(round(4 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return centers, np.convolve(histogram, kernel, mode="same")


def _break_time_gaps(lead_minutes, values):
    """Insert NaNs so missing native forecast times are visible as line breaks."""
    leads = np.asarray(lead_minutes, dtype=float)
    scores = np.asarray(values, dtype=float)
    if leads.size != scores.size:
        raise ValueError("lead_minutes and score values must have the same length")
    if leads.size < 3:
        return leads, scores
    positive_steps = np.diff(leads)
    positive_steps = positive_steps[positive_steps > 0]
    if positive_steps.size == 0:
        return leads, scores
    native_step = float(np.min(positive_steps))
    x_values = [leads[0]]
    y_values = [scores[0]]
    for previous, lead, score in zip(leads, leads[1:], scores[1:]):
        if lead - previous > native_step * 1.5:
            x_values.append(np.nan)
            y_values.append(np.nan)
        x_values.append(lead)
        y_values.append(score)
    return np.asarray(x_values), np.asarray(y_values)
