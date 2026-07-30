import numpy as np

import perturbradius as pr


def test_assign_rings_uses_left_closed_intervals():
    observed = pr.assign_rings(
        np.array([0, 19.999, 20, 39.999, 40, 199.999, 200, np.nan]),
        (0, 20, 40, 80, 120, 200),
    )
    assert observed.tolist() == [0, 0, 1, 1, 2, 4, -1, -1]


def test_components_are_called_separately_by_section(synthetic_screen):
    data = pr.validate_data(synthetic_screen)
    components = pr.call_components(
        data,
        perturbation="GeneA",
        eps_um=15,
        min_samples=4,
        min_source_bins=15,
    )
    assert len(components.summary) == 4
    assert components.summary.groupby("section").size().to_dict() == {"S1": 2, "S2": 2}
    assert components.summary["component_id"].is_unique


def test_validation_does_not_mutate_input(synthetic_screen):
    columns = synthetic_screen.columns.tolist()
    validated = pr.validate_data(synthetic_screen)
    assert synthetic_screen.columns.tolist() == columns
    assert "_row_id" not in synthetic_screen
    assert validated["_row_id"].is_unique

