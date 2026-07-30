from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_screen(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    axis = np.arange(0.0, 601.0, 10.0)
    xx, yy = np.meshgrid(axis, axis)
    coordinates = np.column_stack([xx.ravel(), yy.ravel()])
    perturbation_center = np.array([150.0, 300.0])
    ntc_center = np.array([450.0, 300.0])
    source_radius = 32.0

    for section_index, section in enumerate(("S1", "S2"), start=1):
        distance_perturbation = np.linalg.norm(
            coordinates - perturbation_center, axis=1
        )
        distance_ntc = np.linalg.norm(coordinates - ntc_center, axis=1)
        perturbation_source = distance_perturbation <= source_radius
        ntc_source = distance_ntc <= source_radius
        boundary_distance = np.maximum(distance_perturbation - source_radius, 0.0)
        response = 0.8 * np.exp(-boundary_distance / 55.0)
        response += rng.normal(0.0, 0.012, len(response))

        target_gene = np.full(len(response), "unassigned", dtype=object)
        guide = np.full(len(response), "", dtype=object)
        target_gene[perturbation_source] = "GeneA"
        guide[perturbation_source] = f"sgGeneA_{section_index}"
        target_gene[ntc_source] = "NTC"
        guide[ntc_source] = f"NTC_{section_index}"

        frames.append(
            pd.DataFrame(
                {
                    "section": section,
                    "mouse": f"M{section_index}",
                    "x_um": coordinates[:, 0],
                    "y_um": coordinates[:, 1],
                    "target_gene": target_gene,
                    "guide": guide,
                    "is_source": perturbation_source | ntc_source,
                    "is_ntc": ntc_source,
                    "response": response,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def synthetic_screen() -> pd.DataFrame:
    return make_synthetic_screen()

