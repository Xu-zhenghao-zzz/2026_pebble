from pathlib import Path

import numpy as np
import pytest

import perturbradius as pr


h5py = pytest.importorskip("h5py")


def _write_sparse_fixture(path: Path):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("obs/_index", data=np.asarray([b"b0", b"b1", b"b2"]))
        handle.create_dataset("obs/array_row", data=[0, 0, 1])
        handle.create_dataset("obs/array_col", data=[0, 1, 0])
        handle.create_dataset("obs/in_tissue", data=[1, 1, 1])
        handle.create_dataset("obsm/spatial", data=[[0, 0, 0], [0, 8, 0], [0, 0, 8]])
        handle.create_dataset("var/_index", data=np.asarray([b"GeneA", b"GeneB"]))
        handle.create_dataset("X/data", data=np.asarray([2, 1, 3], dtype=np.float32))
        handle.create_dataset("X/indices", data=np.asarray([0, 1, 0], dtype=np.int32))
        handle.create_dataset("X/indptr", data=np.asarray([0, 1, 2, 3], dtype=np.int32))


def test_stream_module_scores_reads_only_requested_markers(tmp_path):
    path = tmp_path / "rna.h5"
    _write_sparse_fixture(path)
    scores, present = pr.stream_module_scores(
        path, {"a": ["GeneA", "Missing"], "b": ["GeneB"]}, chunk_rows=2
    )
    assert present == {"a": ["GeneA"], "b": ["GeneB"]}
    assert list(scores.barcode) == ["b0", "b1", "b2"]
    assert scores.loc[0, "score_a"] > 0
    assert scores.loc[1, "score_a"] == 0
    assert scores.loc[1, "score_b"] > 0


def test_sparse_guide_call_and_parser(tmp_path):
    path = tmp_path / "guide.h5"
    _write_sparse_fixture(path)
    with h5py.File(path, "r+") as handle:
        del handle["var/_index"]
        handle.create_dataset("var/_index", data=np.asarray([b"sgCcn1_2", b"sgnon-targeting_1"]))
    calls = pr.read_sparse_guide_calls(path)
    assert calls.loc[0, "top_guide"] == "sgCcn1_2"
    assert calls.loc[0, "target_gene"] == "Ccn1"
    assert calls.loc[1, "is_ntc"]
    assert calls.loc[2, "top_guide_count"] == 3


def test_local_grid_preserves_missing_as_unavailable():
    import pandas as pd

    data = pd.DataFrame({
        "x_um": [0, 10, 0, 10], "y_um": [0, 0, 10, 10],
        "response": [1.0, np.nan, 0.0, np.nan],
    })
    source = data.iloc[[0]].copy()
    grid = pr.local_response_grid(
        data, source, response_col="response", resolution=10, pad=10, sigma_bins=1
    )
    assert np.isfinite(grid["heat"]).any()
    assert np.nanmax(grid["distance"]) > 0
