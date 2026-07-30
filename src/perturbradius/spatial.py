"""Source-component calling and boundary-distance rings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class ComponentSet:
    """Spatial source components and their source-row membership."""

    summary: pd.DataFrame
    membership: pd.DataFrame


def call_components(
    data: pd.DataFrame,
    *,
    perturbation: str | None = None,
    eps_um: float = 24.0,
    min_samples: int = 8,
    min_source_bins: int = 15,
) -> ComponentSet:
    """Call DBSCAN source components separately by section, target and guide."""
    if eps_um <= 0:
        raise ValueError("eps_um must be positive")
    if min_samples < 1 or min_source_bins < 1:
        raise ValueError("min_samples and min_source_bins must be positive")
    if "_row_id" not in data.columns:
        raise ValueError("Call validate_data() before call_components()")

    source = data.loc[data["is_source"]].copy()
    if perturbation is not None:
        source = source.loc[
            source["target_gene"].eq(str(perturbation)) | source["is_ntc"]
        ]

    summaries: list[dict[str, object]] = []
    memberships: list[pd.DataFrame] = []
    component_number = 0
    group_columns = ["section", "target_gene", "guide"]

    for (section, target, guide), group in source.groupby(
        group_columns, sort=True, observed=True
    ):
        coordinates = group[["x_um", "y_um"]].to_numpy(dtype=float)
        labels = DBSCAN(eps=eps_um, min_samples=min_samples).fit_predict(coordinates)
        for label in sorted(set(labels) - {-1}):
            keep = labels == label
            if int(keep.sum()) < min_source_bins:
                continue
            component_number += 1
            component_id = f"C{component_number:04d}"
            selected = group.loc[keep]
            memberships.append(
                pd.DataFrame(
                    {
                        "_row_id": selected["_row_id"].to_numpy(dtype=np.int64),
                        "component_id": component_id,
                    }
                )
            )
            summaries.append(
                {
                    "component_id": component_id,
                    "section": str(section),
                    "target_gene": str(target),
                    "guide": str(guide),
                    "is_ntc": bool(selected["is_ntc"].all()),
                    "n_source_bins": int(len(selected)),
                    "centroid_x_um": float(selected["x_um"].mean()),
                    "centroid_y_um": float(selected["y_um"].mean()),
                }
            )

    summary = pd.DataFrame(
        summaries,
        columns=[
            "component_id",
            "section",
            "target_gene",
            "guide",
            "is_ntc",
            "n_source_bins",
            "centroid_x_um",
            "centroid_y_um",
        ],
    )
    membership = pd.concat(memberships, ignore_index=True) if memberships else pd.DataFrame(
        columns=["_row_id", "component_id"]
    )
    return ComponentSet(summary=summary, membership=membership)


def assign_rings(
    distances_um: Sequence[float] | np.ndarray,
    ring_edges: Sequence[float],
) -> np.ndarray:
    """Assign distances to left-closed, right-open rings.

    A distance equal to the final edge is outside the analysed field.
    """
    edges = np.asarray(ring_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("ring_edges must contain at least two values")
    if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0) or edges[0] < 0:
        raise ValueError("ring_edges must be finite, increasing and non-negative")

    distances = np.asarray(distances_um, dtype=float)
    labels = np.searchsorted(edges, distances, side="right") - 1
    valid = (
        np.isfinite(distances)
        & (distances >= edges[0])
        & (distances < edges[-1])
        & (labels >= 0)
        & (labels < len(edges) - 1)
    )
    return np.where(valid, labels, -1).astype(int)


def build_distance_rings(
    data: pd.DataFrame,
    components: ComponentSet,
    *,
    ring_edges: Sequence[float] = (0, 20, 40, 80, 120, 200),
    bin_size_um: float = 0.0,
    exclude_all_sources: bool = True,
) -> pd.DataFrame:
    """Assign each target row to its nearest source-component boundary."""
    if components.summary.empty:
        raise ValueError("No source components are available")
    if bin_size_um < 0:
        raise ValueError("bin_size_um cannot be negative")

    edges = np.asarray(ring_edges, dtype=float)
    assign_rings([0.0], edges)
    source_rows = components.membership.merge(
        data[["_row_id", "x_um", "y_um"]],
        on="_row_id",
        how="left",
        validate="one_to_one",
    )
    source_rows = source_rows.merge(
        components.summary[
            ["component_id", "section", "target_gene", "guide", "is_ntc"]
        ],
        on="component_id",
        how="left",
        validate="many_to_one",
    )

    if exclude_all_sources:
        target_pool = data.loc[~data["is_source"]].copy()
    else:
        target_pool = data.copy()
    assignments: list[pd.DataFrame] = []

    for section, targets in target_pool.groupby("section", sort=True, observed=True):
        section_summary = components.summary.loc[
            components.summary["section"].eq(str(section))
        ].reset_index(drop=True)
        if section_summary.empty:
            continue
        target_coordinates = targets[["x_um", "y_um"]].to_numpy(dtype=float)
        component_distances: list[np.ndarray] = []
        for component_id in section_summary["component_id"]:
            coordinates = source_rows.loc[
                source_rows["component_id"].eq(component_id), ["x_um", "y_um"]
            ].to_numpy(dtype=float)
            distance, _ = cKDTree(coordinates).query(target_coordinates, k=1)
            component_distances.append(np.maximum(distance - bin_size_um / 2.0, 0.0))

        distance_matrix = np.column_stack(component_distances)
        nearest_index = np.argmin(distance_matrix, axis=1)
        nearest_distance = distance_matrix[np.arange(len(targets)), nearest_index]
        if distance_matrix.shape[1] > 1:
            second_distance = np.partition(distance_matrix, 1, axis=1)[:, 1]
        else:
            second_distance = np.full(len(targets), np.inf)

        ring_index = assign_rings(nearest_distance, edges)
        keep = ring_index >= 0
        if not np.any(keep):
            continue
        selected = targets.iloc[np.flatnonzero(keep)].copy()
        selected["component_id"] = section_summary.iloc[nearest_index[keep]][
            "component_id"
        ].to_numpy()
        selected["distance_um"] = nearest_distance[keep]
        selected["ring_index"] = ring_index[keep]
        selected["ring_low_um"] = edges[ring_index[keep]]
        selected["ring_high_um"] = edges[ring_index[keep] + 1]
        selected["ring"] = [
            f"{low:g}-{high:g}" for low, high in zip(
                selected["ring_low_um"], selected["ring_high_um"], strict=True
            )
        ]
        selected["multi_source_exposure"] = second_distance[keep] < edges[-1]
        assignments.append(selected)

    if not assignments:
        raise ValueError("No target rows fell inside the requested distance rings")
    ring_table = pd.concat(assignments, ignore_index=True)
    ring_table = ring_table.merge(
        components.summary[
            ["component_id", "target_gene", "guide", "is_ntc"]
        ].rename(
            columns={
                "target_gene": "target_gene_exposure",
                "guide": "guide_exposure",
                "is_ntc": "is_ntc_exposure",
            }
        ),
        on="component_id",
        how="left",
        validate="many_to_one",
    )
    return ring_table
