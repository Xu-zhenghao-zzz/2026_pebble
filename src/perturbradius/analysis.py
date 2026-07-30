"""Distance-response estimation, diagnostic fitting and evidence gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .data import validate_data
from .spatial import ComponentSet, build_distance_rings, call_components


@dataclass(frozen=True)
class ModelFit:
    """Diagnostic signed exponential fit."""

    success: bool
    amplitude: float = np.nan
    length_scale_um: float = np.nan
    r50_um: float = np.nan
    r5_um: float = np.nan
    r_squared: float = np.nan
    monotonic: bool = False
    direction: str = "unknown"
    n_points: int = 0
    error: str | None = None


@dataclass(frozen=True)
class RadiusDecision:
    """Whether a numerical propagation radius is scientifically reportable."""

    reportable: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RadiusResult:
    """All outputs from a minimal PerturbRadius analysis."""

    components: ComponentSet
    ring_table: pd.DataFrame
    effect_table: pd.DataFrame
    model_fit: ModelFit
    radius_decision: RadiusDecision


def _standard_error(values: pd.Series) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return np.nan
    return float(np.std(clean, ddof=1) / np.sqrt(len(clean)))


def estimate_distance_response(
    ring_table: pd.DataFrame,
    *,
    perturbation: str,
    response_col: str = "response",
    exclude_multi_source: bool = True,
) -> pd.DataFrame:
    """Estimate a component-centred perturbation-versus-NTC distance curve."""
    if response_col not in ring_table:
        raise ValueError(f"Response column {response_col!r} is unavailable")
    table = ring_table.copy()
    if exclude_multi_source:
        table = table.loc[~table["multi_source_exposure"]]
    table = table.loc[table[response_col].notna()]
    if table.empty:
        raise ValueError("No eligible response values remain")

    component_means = (
        table.groupby(
            [
                "component_id",
                "target_gene_exposure",
                "is_ntc_exposure",
                "ring_index",
                "ring",
                "ring_low_um",
                "ring_high_um",
            ],
            observed=True,
            as_index=False,
        )[response_col]
        .mean()
        .rename(columns={response_col: "ring_mean"})
    )
    far_index = int(table["ring_index"].max())
    far = component_means.loc[
        component_means["ring_index"].eq(far_index),
        ["component_id", "ring_mean"],
    ].rename(columns={"ring_mean": "far_mean"})
    component_means = component_means.merge(
        far, on="component_id", how="inner", validate="many_to_one"
    )
    component_means["far_centered"] = (
        component_means["ring_mean"] - component_means["far_mean"]
    )

    perturbation_rows = component_means.loc[
        component_means["target_gene_exposure"].eq(str(perturbation))
        & ~component_means["is_ntc_exposure"]
    ]
    ntc_rows = component_means.loc[component_means["is_ntc_exposure"]]
    if perturbation_rows.empty:
        raise ValueError(f"No component-level responses found for {perturbation!r}")

    keys = ["ring_index", "ring", "ring_low_um", "ring_high_um"]
    perturbation_summary = (
        perturbation_rows.groupby(keys, observed=True)["far_centered"]
        .agg(perturbation_effect="mean", n_perturbation_components="count")
        .reset_index()
    )
    perturbation_se = (
        perturbation_rows.groupby(keys, observed=True)["far_centered"]
        .apply(_standard_error)
        .rename("perturbation_se")
        .reset_index()
    )
    effect = perturbation_summary.merge(perturbation_se, on=keys, how="left")

    if ntc_rows.empty:
        effect["ntc_effect"] = np.nan
        effect["ntc_se"] = np.nan
        effect["n_ntc_components"] = 0
        effect["effect"] = effect["perturbation_effect"]
        effect["effect_se"] = effect["perturbation_se"]
    else:
        ntc_summary = (
            ntc_rows.groupby(keys, observed=True)["far_centered"]
            .agg(ntc_effect="mean", n_ntc_components="count")
            .reset_index()
        )
        ntc_se = (
            ntc_rows.groupby(keys, observed=True)["far_centered"]
            .apply(_standard_error)
            .rename("ntc_se")
            .reset_index()
        )
        effect = effect.merge(ntc_summary, on=keys, how="left").merge(
            ntc_se, on=keys, how="left"
        )
        effect["effect"] = effect["perturbation_effect"] - effect["ntc_effect"]
        effect["effect_se"] = np.sqrt(
            effect["perturbation_se"].fillna(0) ** 2
            + effect["ntc_se"].fillna(0) ** 2
        )

    effect["distance_midpoint_um"] = (
        effect["ring_low_um"] + effect["ring_high_um"]
    ) / 2.0
    return effect.sort_values("ring_index").reset_index(drop=True)


def _exponential(distance: np.ndarray, amplitude: float, length_scale: float) -> np.ndarray:
    return amplitude * np.exp(-distance / length_scale)


def fit_exponential(
    effect_table: pd.DataFrame,
    *,
    effect_col: str = "effect",
    monotonic_tolerance: float = 0.05,
) -> ModelFit:
    """Fit a signed exponential curve as a diagnostic, not a causal claim."""
    clean = effect_table[["distance_midpoint_um", effect_col]].dropna()
    if len(clean) < 3:
        return ModelFit(False, n_points=len(clean), error="fewer than three points")
    distance = clean["distance_midpoint_um"].to_numpy(dtype=float)
    response = clean[effect_col].to_numpy(dtype=float)
    if np.allclose(response, 0):
        return ModelFit(False, n_points=len(clean), error="all effects are zero")

    amplitude_start = float(response[np.argmin(distance)])
    if np.isclose(amplitude_start, 0):
        amplitude_start = float(response[np.argmax(np.abs(response))])
    length_start = max(float(np.median(distance)), 1.0)
    try:
        parameters, _ = curve_fit(
            _exponential,
            distance,
            response,
            p0=(amplitude_start, length_start),
            bounds=([-np.inf, 1e-6], [np.inf, 1e6]),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return ModelFit(False, n_points=len(clean), error=str(error))

    amplitude, length_scale = map(float, parameters)
    predicted = _exponential(distance, amplitude, length_scale)
    residual_sum = float(np.sum((response - predicted) ** 2))
    total_sum = float(np.sum((response - np.mean(response)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else np.nan
    sign = 1.0 if amplitude >= 0 else -1.0
    aligned = response * sign
    tolerance = monotonic_tolerance * max(float(np.nanmax(np.abs(aligned))), 1e-12)
    monotonic = bool(
        aligned[0] > 0
        and np.all(np.diff(aligned[np.argsort(distance)]) <= tolerance)
    )
    return ModelFit(
        success=True,
        amplitude=amplitude,
        length_scale_um=length_scale,
        r50_um=length_scale * np.log(2.0),
        r5_um=length_scale * np.log(20.0),
        r_squared=r_squared,
        monotonic=monotonic,
        direction="enrichment" if amplitude >= 0 else "depletion",
        n_points=len(clean),
    )


def assess_radius_support(
    data: pd.DataFrame,
    components: ComponentSet,
    effect_table: pd.DataFrame,
    model_fit: ModelFit,
    *,
    perturbation: str,
    replicate_col: str | None,
    min_guides: int = 2,
    min_replicates: int = 2,
    min_ntc_components: int = 2,
    min_r_squared: float = 0.5,
) -> RadiusDecision:
    """Apply a conservative gate before reporting a numerical radius."""
    reasons: list[str] = []
    target_summary = components.summary.loc[
        components.summary["target_gene"].eq(str(perturbation))
        & ~components.summary["is_ntc"]
    ]
    ntc_summary = components.summary.loc[components.summary["is_ntc"]]

    if target_summary["guide"].nunique() < min_guides:
        reasons.append(f"fewer than {min_guides} independent guides")
    if len(ntc_summary) < min_ntc_components:
        reasons.append(f"fewer than {min_ntc_components} NTC components")

    if replicate_col is None or replicate_col not in data:
        reasons.append("biological replicate column unavailable")
    else:
        target_rows = components.membership.merge(
            data[["_row_id", replicate_col]],
            on="_row_id",
            how="left",
            validate="one_to_one",
        )
        target_ids = set(target_summary["component_id"])
        target_replicates = target_rows.loc[
            target_rows["component_id"].isin(target_ids), replicate_col
        ].nunique()
        if target_replicates < min_replicates:
            reasons.append(f"fewer than {min_replicates} biological replicates")

    if effect_table["n_ntc_components"].max() < min_ntc_components:
        reasons.append("insufficient NTC support in complete distance rings")
    if not model_fit.success:
        reasons.append(f"exponential fit failed: {model_fit.error}")
    else:
        if not model_fit.monotonic:
            reasons.append("distance response is not monotonic")
        if not np.isfinite(model_fit.r_squared) or model_fit.r_squared < min_r_squared:
            reasons.append(f"exponential fit R² is below {min_r_squared:g}")
    return RadiusDecision(reportable=not reasons, reasons=tuple(reasons))


def analyze_radius(
    data: pd.DataFrame,
    *,
    perturbation: str,
    response_col: str = "response",
    ring_edges: Sequence[float] = (0, 20, 40, 80, 120, 200),
    eps_um: float = 24.0,
    min_samples: int = 8,
    min_source_bins: int = 15,
    bin_size_um: float = 0.0,
    replicate_col: str | None = None,
    exclude_multi_source: bool = True,
) -> RadiusResult:
    """Run the minimal component-to-radius workflow."""
    validated = validate_data(
        data,
        response_col=response_col,
        extra_required=() if replicate_col is None else (replicate_col,),
    )
    components = call_components(
        validated,
        perturbation=perturbation,
        eps_um=eps_um,
        min_samples=min_samples,
        min_source_bins=min_source_bins,
    )
    rings = build_distance_rings(
        validated,
        components,
        ring_edges=ring_edges,
        bin_size_um=bin_size_um,
    )
    effects = estimate_distance_response(
        rings,
        perturbation=perturbation,
        response_col=response_col,
        exclude_multi_source=exclude_multi_source,
    )
    model_fit = fit_exponential(effects)
    decision = assess_radius_support(
        validated,
        components,
        effects,
        model_fit,
        perturbation=perturbation,
        replicate_col=replicate_col,
    )
    return RadiusResult(
        components=components,
        ring_table=rings,
        effect_table=effects,
        model_fit=model_fit,
        radius_decision=decision,
    )

