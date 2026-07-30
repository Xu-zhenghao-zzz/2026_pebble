"""Minimal plotting functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .analysis import RadiusResult, _exponential


def plot_distance_response(result: RadiusResult, *, ax=None):
    """Plot the NTC-adjusted empirical response and diagnostic exponential fit."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.6))
    table = result.effect_table
    distance = table["distance_midpoint_um"].to_numpy(dtype=float)
    effect = table["effect"].to_numpy(dtype=float)
    error = table["effect_se"].to_numpy(dtype=float)
    finite_error = np.where(np.isfinite(error), error, 0.0)
    ax.errorbar(
        distance,
        effect,
        yerr=finite_error,
        marker="o",
        linewidth=1.8,
        capsize=3,
        label="empirical NTC-adjusted effect",
    )
    if result.model_fit.success:
        grid = np.linspace(float(distance.min()), float(distance.max()), 200)
        ax.plot(
            grid,
            _exponential(
                grid,
                result.model_fit.amplitude,
                result.model_fit.length_scale_um,
            ),
            linestyle="--",
            linewidth=1.5,
            label="diagnostic exponential fit",
        )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("Distance from source boundary (µm)")
    ax.set_ylabel("Far-centred response")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    status = "radius reportable" if result.radius_decision.reportable else "descriptive only"
    ax.set_title(f"PerturbRadius — {status}")
    return ax


def plot_section_map(
    data: pd.DataFrame,
    *,
    source: pd.DataFrame | None = None,
    color: str | None = None,
    ax=None,
    point_size: float = 1.0,
    title: str | None = None,
    coordinate_label: str = "µm",
):
    """Plot an entire tissue section and optional guide-positive source bins."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    if color is None:
        ax.scatter(data["x_um"], data["y_um"], s=point_size, color="#D5D5D5",
                   rasterized=True)
    elif pd.api.types.is_numeric_dtype(data[color]):
        artist = ax.scatter(data["x_um"], data["y_um"], c=data[color], s=point_size,
                            cmap="viridis", rasterized=True)
        plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04, label=color)
    else:
        categories = pd.Categorical(data[color])
        palette = plt.get_cmap("tab10")
        for index, category in enumerate(categories.categories):
            keep = categories == category
            ax.scatter(data.loc[keep, "x_um"], data.loc[keep, "y_um"],
                       s=point_size, color=palette(index % 10), label=str(category),
                       rasterized=True)
        ax.legend(frameon=False, markerscale=4, bbox_to_anchor=(1.02, 1), loc="upper left")
    if source is not None and len(source):
        ax.scatter(source["x_um"], source["y_um"], s=max(5, point_size * 5),
                   color="black", linewidths=0, label="guide-positive source",
                   rasterized=True)
    ax.set_xlabel(f"x ({coordinate_label})")
    ax.set_ylabel(f"y ({coordinate_label})")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def local_response_grid(
    data: pd.DataFrame,
    source: pd.DataFrame,
    *,
    response_col: str,
    resolution: float,
    pad: float = 300.0,
    sigma_bins: float = 3.0,
) -> dict[str, object]:
    """Rasterize a local response and distance from a source component.

    The returned response is a Gaussian-smoothed numerator/availability ratio;
    missing response rows therefore do not become numerical zeroes.
    """
    if resolution <= 0 or sigma_bins <= 0:
        raise ValueError("resolution and sigma_bins must be positive")
    if source.empty:
        raise ValueError("source is empty")
    xmin, xmax = source.x_um.min() - pad, source.x_um.max() + pad
    ymin, ymax = source.y_um.min() - pad, source.y_um.max() + pad
    local = data.loc[
        data.x_um.between(xmin, xmax) & data.y_um.between(ymin, ymax)
    ].copy()
    col0, row0 = int(np.floor(xmin / resolution)), int(np.floor(ymin / resolution))
    ncol = int(np.ceil((xmax - xmin) / resolution)) + 1
    nrow = int(np.ceil((ymax - ymin) / resolution)) + 1
    rr = np.rint(local.y_um.to_numpy() / resolution).astype(int) - row0
    cc = np.rint(local.x_um.to_numpy() / resolution).astype(int) - col0
    keep = (rr >= 0) & (rr < nrow) & (cc >= 0) & (cc < ncol)
    rr, cc, local = rr[keep], cc[keep], local.iloc[np.flatnonzero(keep)]
    values = local[response_col].to_numpy(dtype=float)
    available = np.isfinite(values)
    numerator = np.zeros((nrow, ncol), dtype=float)
    denominator = np.zeros((nrow, ncol), dtype=float)
    numerator[rr[available], cc[available]] = values[available]
    denominator[rr[available], cc[available]] = 1.0
    smooth_num = gaussian_filter(numerator, sigma_bins, mode="constant")
    smooth_den = gaussian_filter(denominator, sigma_bins, mode="constant")
    heat = np.divide(smooth_num, smooth_den, out=np.full_like(smooth_num, np.nan),
                     where=smooth_den > 0.05)
    sr = np.rint(source.y_um.to_numpy() / resolution).astype(int) - row0
    sc = np.rint(source.x_um.to_numpy() / resolution).astype(int) - col0
    valid_source = (sr >= 0) & (sr < nrow) & (sc >= 0) & (sc < ncol)
    source_mask = np.zeros((nrow, ncol), dtype=bool)
    source_mask[sr[valid_source], sc[valid_source]] = True
    distance = distance_transform_edt(~source_mask) * resolution
    extent = (col0 * resolution, (col0 + ncol) * resolution,
              row0 * resolution, (row0 + nrow) * resolution)
    return {"heat": heat, "distance": distance, "extent": extent, "local": local}


def plot_local_response(
    data: pd.DataFrame,
    source: pd.DataFrame,
    *,
    response_col: str,
    resolution: float,
    contour_levels=(40, 80, 120, 200),
    pad: float = 300.0,
    sigma_bins: float = 3.0,
    ax=None,
    title: str | None = None,
    cbar_label: str | None = None,
    coordinate_label: str = "µm",
    cmap: str = "magma",
    value_limits: tuple[float, float] | None = None,
):
    """Plot a smoothed local response, source core and distance contours."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    grid = local_response_grid(
        data, source, response_col=response_col, resolution=resolution,
        pad=pad, sigma_bins=sigma_bins,
    )
    heat, distance, extent = grid["heat"], grid["distance"], grid["extent"]
    finite = heat[np.isfinite(heat)]
    if value_limits is None:
        value_limits = tuple(np.quantile(finite, [0.03, 0.97])) if len(finite) else (0, 1)
    artist = ax.imshow(heat, origin="lower", extent=extent, cmap=cmap,
                       vmin=value_limits[0], vmax=value_limits[1], interpolation="nearest")
    xx = np.linspace(extent[0], extent[1], heat.shape[1])
    yy = np.linspace(extent[2], extent[3], heat.shape[0])
    levels = [float(x) for x in contour_levels if float(x) < np.nanmax(distance)]
    if levels:
        contours = ax.contour(xx, yy, distance, levels=levels,
                              colors=["white", "#56B4E9", "#F0E442", "#009E73"][:len(levels)],
                              linewidths=1.4)
        ax.clabel(contours, fmt=lambda x: f"{x:g}", fontsize=8)
    ax.scatter(source.x_um, source.y_um, s=8, color="black", alpha=0.8,
               label="guide-positive source", rasterized=True)
    ax.set_xlabel(f"x ({coordinate_label})")
    ax.set_ylabel(f"y ({coordinate_label})")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title)
    bar = plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label(cbar_label or response_col)
    ax.legend(frameon=True, fontsize=8, loc="best")
    return ax, grid
