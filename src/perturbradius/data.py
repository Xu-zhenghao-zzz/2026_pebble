"""Input validation for PerturbRadius."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "section",
    "x_um",
    "y_um",
    "target_gene",
    "guide",
    "is_source",
    "is_ntc",
)


def _coerce_boolean(series: pd.Series, name: str) -> pd.Series:
    """Convert common Boolean encodings without treating arbitrary strings as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "yes": True,
        "no": False,
    }
    converted = series.map(mapping)
    if converted.isna().any():
        values = sorted(series[converted.isna()].astype(str).unique())
        raise ValueError(f"{name!r} contains unsupported Boolean values: {values}")
    return converted.astype(bool)


def validate_data(
    data: pd.DataFrame,
    *,
    response_col: str = "response",
    extra_required: Iterable[str] = (),
) -> pd.DataFrame:
    """Validate and copy a standardized spatial perturbation table.

    Parameters
    ----------
    data
        One row per spatial bin or cell.
    response_col
        Numeric response column to analyse.
    extra_required
        Additional columns required by a caller, for example ``("mouse",)``.

    Returns
    -------
    pandas.DataFrame
        A defensive copy with a deterministic internal ``_row_id`` column.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas.DataFrame")
    required = [*REQUIRED_COLUMNS, response_col, *extra_required]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(data) == 0:
        raise ValueError("data is empty")

    result = data.copy(deep=True).reset_index(drop=True)
    if "_row_id" in result.columns:
        result = result.drop(columns="_row_id")
    result.insert(0, "_row_id", np.arange(len(result), dtype=np.int64))

    for coordinate in ("x_um", "y_um"):
        result[coordinate] = pd.to_numeric(result[coordinate], errors="raise")
        if not np.isfinite(result[coordinate].to_numpy(dtype=float)).all():
            raise ValueError(f"{coordinate!r} must contain finite values")

    result[response_col] = pd.to_numeric(result[response_col], errors="coerce")
    result["is_source"] = _coerce_boolean(result["is_source"], "is_source")
    result["is_ntc"] = _coerce_boolean(result["is_ntc"], "is_ntc")

    for column in ("section", "target_gene", "guide"):
        if result[column].isna().any():
            raise ValueError(f"{column!r} cannot contain missing values")
        result[column] = result[column].astype(str)

    if (result.loc[result["is_source"], "guide"].str.len() == 0).any():
        raise ValueError("Every source row must have a non-empty guide")
    return result
