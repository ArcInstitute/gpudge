# tests/test_output.py
import numpy as np
import polars as pl
import pytest
from gpudge._output import (
    DEFAULT_OUTPUT_COLUMNS, assemble_dataframe,
)


def _stub_arrays(n_guides=3, n_genes=4):
    g = np.array([f"G{i}" for i in range(n_guides)])
    f = np.array([f"g{i}" for i in range(n_genes)])
    return dict(
        target=g, feature=f,
        target_mean=np.full((n_guides, n_genes), 2.0),
        ref_mean=np.full((n_guides, n_genes), 1.0),
        target_ncells=np.full(n_guides, 100, dtype=np.int64),
        ref_ncells=200,
        log2_fold_change=np.ones((n_guides, n_genes)),
        p_value=np.full((n_guides, n_genes), 0.05),
        test_statistic=np.full((n_guides, n_genes), 0.5),
        p_adj=np.full((n_guides, n_genes), 0.1),
    )


def test_default_columns_present_in_full_schema():
    df = assemble_dataframe(**_stub_arrays(), output_columns=None)
    assert df.columns == list(DEFAULT_OUTPUT_COLUMNS)
    assert df.height == 3 * 4


def test_output_columns_renames_and_selects():
    df = assemble_dataframe(
        **_stub_arrays(),
        output_columns={"target": "guide", "feature": "gene",
                        "log2_fold_change": "log2fc", "p_adj": "fdr"},
    )
    assert df.columns == ["guide", "gene", "log2fc", "fdr"]
    assert df.height == 3 * 4


def test_output_columns_unknown_key_raises():
    with pytest.raises(KeyError, match="bogus"):
        assemble_dataframe(**_stub_arrays(),
                          output_columns={"bogus": "x"})


def test_flat_keep_all_true_matches_unfiltered():
    args = _stub_arrays(n_guides=3, n_genes=4)
    full = assemble_dataframe(**args)
    keep = np.ones(3 * 4, dtype=bool)
    filt = assemble_dataframe(**args, flat_keep=keep)
    assert filt.height == full.height
    for c in full.columns:
        assert filt[c].to_list() == full[c].to_list(), c


def test_flat_keep_partial_drops_correct_rows():
    args = _stub_arrays(n_guides=3, n_genes=4)
    # Per-row distinct values so we can assert row-level identity.
    args["log2_fold_change"] = np.arange(12, dtype=np.float64).reshape(3, 4)
    args["p_value"] = (np.arange(12, dtype=np.float64) / 100).reshape(3, 4)
    keep = np.zeros(12, dtype=bool)
    keep[[0, 5, 11]] = True
    df = assemble_dataframe(**args, flat_keep=keep)
    assert df.height == 3
    # Row 0 → guide 0, gene 0;  row 5 → guide 1, gene 1;  row 11 → guide 2, gene 3.
    assert df["target"].to_list() == ["G0", "G1", "G2"]
    assert df["feature"].to_list() == ["g0", "g1", "g3"]
    assert df["log2_fold_change"].to_list() == [0.0, 5.0, 11.0]


def test_flat_keep_all_false_gives_empty():
    args = _stub_arrays(n_guides=3, n_genes=4)
    df = assemble_dataframe(**args, flat_keep=np.zeros(12, dtype=bool))
    assert df.height == 0
    assert df.columns == list(DEFAULT_OUTPUT_COLUMNS)


def test_flat_keep_with_1d_ref_mean():
    args = _stub_arrays(n_guides=3, n_genes=4)
    # 1D ref_mean (one per gene) — assemble_dataframe should index by gene.
    args["ref_mean"] = np.array([10.0, 20.0, 30.0, 40.0])
    keep = np.array([False, True, False, True,  # guide 0: genes 1, 3
                     True,  False, False, False,  # guide 1: gene 0
                     False, False, True,  False], # guide 2: gene 2
                    dtype=bool)
    df = assemble_dataframe(**args, flat_keep=keep)
    assert df["ref_mean"].to_list() == [20.0, 40.0, 10.0, 30.0]
    assert df["target"].to_list() == ["G0", "G0", "G1", "G2"]
    assert df["feature"].to_list() == ["g1", "g3", "g0", "g2"]


def test_flat_keep_with_scalar_ref_ncells():
    args = _stub_arrays(n_guides=3, n_genes=4)
    args["ref_ncells"] = 200  # scalar
    keep = np.zeros(12, dtype=bool)
    keep[[0, 4, 8]] = True
    df = assemble_dataframe(**args, flat_keep=keep)
    assert df.height == 3
    assert df["ref_ncells"].to_list() == [200, 200, 200]


def test_flat_keep_wrong_size_raises():
    args = _stub_arrays(n_guides=3, n_genes=4)
    with pytest.raises(ValueError, match=r"flat_keep\.size=10"):
        assemble_dataframe(**args, flat_keep=np.zeros(10, dtype=bool))
    with pytest.raises(ValueError, match=r"flat_keep\.size=20"):
        assemble_dataframe(**args, flat_keep=np.zeros(20, dtype=bool))


def test_1d_ref_mean_wrong_length_raises():
    args = _stub_arrays(n_guides=3, n_genes=4)
    args["ref_mean"] = np.array([10.0, 20.0, 30.0])  # length 3 ≠ n_genes=4
    with pytest.raises(ValueError, match=r"ref_mean has shape \(3,\)"):
        assemble_dataframe(**args)
    args["ref_mean"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # length 5
    with pytest.raises(ValueError, match=r"ref_mean has shape \(5,\)"):
        assemble_dataframe(**args)
