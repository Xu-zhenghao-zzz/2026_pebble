"""Readers for the raw AnnData-like H5 files used by the SPAC tutorials.

The functions in this module deliberately read only the axes or marker genes
needed by an analysis.  They never write into the source-data directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


GUIDE_RE = re.compile(r"^sg(?P<target>.+?)(?:_(?P<number>\d+))?$", re.IGNORECASE)


def _h5py():
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Raw H5 input requires h5py; install perturbradius[real]."
        ) from error
    return h5py


def decode(values: np.ndarray) -> np.ndarray:
    """Decode an HDF5 byte-string axis into Python strings."""
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
         for value in values],
        dtype=object,
    )


def parse_guide(guide: str, *, ntc_names: Sequence[str] = ("NTC", "sgNTC")) -> dict:
    """Parse the guide conventions used in the three supplied SPAC cohorts."""
    guide = str(guide)
    compact = guide.lower().replace("-", "").replace("_", "")
    configured_ntc = {str(x).lower().replace("-", "").replace("_", "") for x in ntc_names}
    match = GUIDE_RE.match(guide)
    target = match.group("target") if match else guide
    target_compact = target.lower().replace("-", "").replace("_", "")
    is_ntc = compact in configured_ntc or target_compact in {
        "ntc", "nontargeting", "nontarget"
    }
    return {
        "guide": guide,
        "target_gene": "NTC" if is_ntc else target,
        "guide_number": int(match.group("number")) if match and match.group("number") else None,
        "is_ntc": is_ntc,
    }


def h5_schema(path: str | Path) -> pd.DataFrame:
    """Return dataset names, shapes and dtypes without loading matrix values."""
    h5py = _h5py()
    rows: list[dict] = []
    with h5py.File(path, "r") as handle:
        def visitor(name, obj):
            if hasattr(obj, "shape"):
                rows.append({"dataset": name, "shape": " × ".join(map(str, obj.shape)),
                             "dtype": str(obj.dtype)})
        handle.visititems(visitor)
    return pd.DataFrame(rows)


def read_h5_axes(path: str | Path) -> pd.DataFrame:
    """Read barcodes and spatial axes from a sparse or dense AnnData-like H5."""
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        barcode = decode(handle["obs/_index"][:])
        spatial = np.asarray(handle["obsm/spatial"][:])
        frame = pd.DataFrame({"barcode": barcode})
        if spatial.shape[1] == 2:
            frame["x"] = spatial[:, 0].astype(float)
            frame["y"] = spatial[:, 1].astype(float)
        elif spatial.shape[1] >= 3:
            frame["z"] = spatial[:, 0].astype(float)
            frame["x"] = spatial[:, 1].astype(float)
            frame["y"] = spatial[:, 2].astype(float)
        for name in ("array_row", "array_col", "in_tissue"):
            key = f"obs/{name}"
            if key in handle:
                frame[name] = np.asarray(handle[key][:])
    return frame


def _top_calls(matrix: sparse.csr_matrix, names: np.ndarray) -> pd.DataFrame:
    total = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    top_count = np.zeros(matrix.shape[0], dtype=float)
    second_count = np.zeros(matrix.shape[0], dtype=float)
    top_index = np.full(matrix.shape[0], -1, dtype=np.int32)
    for row in np.flatnonzero(np.diff(matrix.indptr)):
        start, stop = matrix.indptr[row:row + 2]
        values = matrix.data[start:stop]
        columns = matrix.indices[start:stop]
        order = np.argsort(values)
        top_count[row] = values[order[-1]]
        top_index[row] = columns[order[-1]]
        if len(order) > 1:
            second_count[row] = values[order[-2]]
    guide = np.full(matrix.shape[0], "", dtype=object)
    called = top_index >= 0
    guide[called] = names[top_index[called]]
    fraction = np.divide(top_count, total, out=np.zeros_like(total), where=total > 0)
    return pd.DataFrame({
        "guide_total": total,
        "top_guide_count": top_count,
        "second_guide_count": second_count,
        "top_guide": guide,
        "top_fraction": fraction,
    })


def read_sparse_guide_calls(
    path: str | Path,
    *,
    ntc_names: Sequence[str] = ("NTC", "sgNTC"),
) -> pd.DataFrame:
    """Read top-guide calls from a CSR AnnData H5 without densifying it."""
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        barcodes = decode(handle["obs/_index"][:])
        names = decode(handle["var/_index"][:])
        indptr = np.asarray(handle["X/indptr"][:])
        matrix = sparse.csr_matrix(
            (np.asarray(handle["X/data"][:]), np.asarray(handle["X/indices"][:]), indptr),
            shape=(len(barcodes), len(names)),
        )
    calls = _top_calls(matrix, names)
    calls.insert(0, "barcode", barcodes)
    mapping = pd.DataFrame([parse_guide(x, ntc_names=ntc_names) for x in names])
    calls = calls.join(mapping.set_index("guide"), on="top_guide")
    calls["is_ntc"] = calls["is_ntc"].eq(True)
    return calls


def read_dense_guide_calls(
    path: str | Path,
    *,
    ntc_names: Sequence[str] = ("NTC", "sgNTC"),
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Read a dense guide H5 and return calls, guide names and the count matrix."""
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        axes = read_h5_axes(path)
        names = decode(handle["var/_index"][:])
        matrix = np.asarray(handle["X"][:])
    calls = _top_calls(sparse.csr_matrix(matrix), names)
    calls = pd.concat([axes.reset_index(drop=True), calls], axis=1)
    mapping = pd.DataFrame([parse_guide(x, ntc_names=ntc_names) for x in names])
    calls = calls.join(mapping.set_index("guide"), on="top_guide")
    calls["is_ntc"] = calls["is_ntc"].eq(True)
    return calls, names, matrix


def empirical_source_posteriors(
    matrix: np.ndarray,
    eligible: np.ndarray,
    xy: np.ndarray,
    *,
    neighbor_radius: float = 60.0,
    pseudocount: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimate guide-source probability from enrichment over background tissue.

    This is useful when guide counts are sparse and most positive bins contain
    only one or two UMIs.  It preserves singleton, multi-UMI and spatially
    supported evidence tiers rather than declaring every non-zero call a source.
    """
    matrix = np.asarray(matrix)
    eligible = np.asarray(eligible, dtype=bool)
    xy = np.asarray(xy, dtype=float)
    total = matrix.sum(axis=1)
    top_index = matrix.argmax(axis=1)
    top_count = matrix[np.arange(len(matrix)), top_index]
    second_count = np.partition(matrix, -2, axis=1)[:, -2]
    unambiguous = (total > 0) & (top_count > second_count)
    neighbor_count = np.zeros(len(matrix), dtype=np.int32)
    valid = np.flatnonzero(unambiguous)
    if len(valid):
        for guide_index in np.unique(top_index[valid]):
            index = valid[top_index[valid] == guide_index]
            tree = cKDTree(xy[index])
            neighbor_count[index] = tree.query_ball_point(
                xy[index], r=neighbor_radius, return_length=True
            ) - 1
    tier = (top_count >= 2).astype(np.int8) + 2 * (neighbor_count > 0).astype(np.int8)
    posterior = np.zeros(len(matrix), dtype=np.float32)
    rows: list[dict] = []
    n_eligible, n_background = int(eligible.sum()), int((~eligible).sum())
    if not n_eligible or not n_background:
        return posterior, neighbor_count, pd.DataFrame()
    for guide_index in range(matrix.shape[1]):
        for evidence_tier in range(4):
            pattern = unambiguous & (top_index == guide_index) & (tier == evidence_tier)
            eligible_count = int((pattern & eligible).sum())
            background_count = int((pattern & ~eligible).sum())
            eligible_rate = (eligible_count + pseudocount) / (n_eligible + 2 * pseudocount)
            background_rate = (background_count + pseudocount) / (n_background + 2 * pseudocount)
            excess = float(np.clip(1.0 - background_rate / eligible_rate, 0.0, 1.0))
            posterior[pattern & eligible] = excess
            rows.append({
                "guide_index": guide_index,
                "evidence_tier": evidence_tier,
                "eligible_count": eligible_count,
                "background_count": background_count,
                "eligible_rate": eligible_rate,
                "background_rate": background_rate,
                "excess_fraction": excess,
            })
    return posterior, neighbor_count, pd.DataFrame(rows)


def stream_module_scores(
    path: str | Path,
    modules: Mapping[str, Sequence[str]],
    *,
    chunk_rows: int = 10_000,
    scale_factor: float = 10_000.0,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Compute log-normalized marker-module means while streaming CSR rows."""
    h5py = _h5py()
    outputs: list[pd.DataFrame] = []
    with h5py.File(path, "r") as handle:
        barcodes = decode(handle["obs/_index"][:])
        genes = decode(handle["var/_index"][:])
        lookup = {gene: index for index, gene in enumerate(genes)}
        present_names = {
            name: [gene for gene in markers if gene in lookup]
            for name, markers in modules.items()
        }
        selected = sorted({lookup[gene] for values in present_names.values() for gene in values})
        if not selected:
            raise ValueError("None of the requested genes occur in the matrix")
        selected_position = {index: position for position, index in enumerate(selected)}
        positions = {
            name: [selected_position[lookup[gene]] for gene in genes_present]
            for name, genes_present in present_names.items()
        }
        indptr = np.asarray(handle["X/indptr"][:])
        for start_row in range(0, len(barcodes), chunk_rows):
            stop_row = min(start_row + chunk_rows, len(barcodes))
            value_start, value_stop = int(indptr[start_row]), int(indptr[stop_row])
            local_indptr = indptr[start_row:stop_row + 1] - value_start
            matrix = sparse.csr_matrix(
                (handle["X/data"][value_start:value_stop],
                 handle["X/indices"][value_start:value_stop], local_indptr),
                shape=(stop_row - start_row, len(genes)),
            )
            library = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
            counts = matrix[:, selected].toarray().astype(float)
            normalized = np.log1p(np.divide(
                counts * scale_factor,
                library[:, None],
                out=np.zeros_like(counts),
                where=library[:, None] > 0,
            ))
            values: dict[str, np.ndarray] = {
                "barcode": barcodes[start_row:stop_row],
                "rna_total_counts": library,
            }
            for name, position in positions.items():
                values[f"score_{name}"] = (
                    normalized[:, position].mean(axis=1) if position
                    else np.full(stop_row - start_row, np.nan)
                )
                values[f"detected_{name}"] = (
                    (counts[:, position] > 0).sum(axis=1) if position
                    else np.zeros(stop_row - start_row, dtype=int)
                )
            outputs.append(pd.DataFrame(values))
    return pd.concat(outputs, ignore_index=True), present_names


def marker_lineage(
    scores: pd.DataFrame,
    lineages: Sequence[str],
    *,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Convert lineage marker scores into a transparent softmax label."""
    columns = [f"score_{name}" for name in lineages]
    values = scores[columns].to_numpy(dtype=float)
    mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    z = np.divide(values - mean, std, out=np.zeros_like(values), where=std > 0)
    z -= np.nanmax(z, axis=1, keepdims=True)
    probability = np.exp(z)
    probability /= probability.sum(axis=1, keepdims=True)
    best = np.argmax(probability, axis=1)
    confidence = probability[np.arange(len(probability)), best]
    labels = np.asarray(lineages, dtype=object)[best]
    detected = scores[[f"detected_{name}" for name in lineages]].sum(axis=1).to_numpy()
    labels[(confidence < threshold) | (detected == 0)] = "mixed_or_unknown"
    result = scores.copy()
    result["inferred_lineage"] = labels
    result["lineage_confidence"] = confidence
    for index, name in enumerate(lineages):
        result[f"probability_{name}"] = probability[:, index]
    return result
