# tests/test_output.py
import numpy as np
import polars as pl
import pytest
from gpudge._output import (
    DEFAULT_OUTPUT_COLUMNS, assemble_dataframe, effect_size_from_u,
)


def test_effect_size_from_u():
    np.testing.assert_array_equal(
        effect_size_from_u(
            np.array([[20.0, 0.0, 10.0]]), np.array([4]), 5),
        np.array([[1.0, -1.0, 0.0]]),
    )

    got = effect_size_from_u(
        np.array([[6.0, 3.0], [20.0, 10.0]]),
        np.array([2, 4]),
        5,
    )
    np.testing.assert_allclose(
        got,
        np.array([[0.2, -0.4], [1.0, 0.0]]),
    )

    # per-target ref_ncells ARRAY (the ALL_OTHERS case, n = n_cells - m per group)
    arr = effect_size_from_u(
        np.array([[10.0, 20.0], [0.0, 8.0]]),   # (2 targets, 2 genes) U
        np.array([4, 8]),                        # m per target
        np.array([5, 2]),                        # n per target (array, not scalar)
    )
    # target0 m·n=20 -> [2*10/20-1, 2*20/20-1]; target1 m·n=16 -> [2*0/16-1, 2*8/16-1]
    np.testing.assert_allclose(arr, np.array([[0.0, 1.0], [-1.0, 0.0]]))

    degenerate = effect_size_from_u(
        np.array([[0.0], [0.0]]), np.array([0, 4]), 5)
    assert np.isnan(degenerate[0, 0])
    assert degenerate[1, 0] == -1.0


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
                        "log2_fold_change": "log2fc",
                        "Ueffect": "rank_effect", "p_adj": "fdr"},
    )
    assert df.columns == ["guide", "gene", "log2fc", "rank_effect", "fdr"]
    assert df["rank_effect"].to_list() == [0.5] * 12
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


def test_empty_output_frame_is_typed_and_matches_assemble():
    """empty_output_frame() must be a 0-row frame with the SAME typed schema as
    a real assemble_dataframe output (not Null columns), and must honour
    output_columns the same way. Regression for the ultrareview empty-archive
    schema finding + the Gemini review note: a frame built from empty lists has
    all-Null columns that mismatch the non-empty result downstream."""
    from gpudge._output import empty_output_frame
    real = assemble_dataframe(
        target=np.array(["g0"]), feature=np.array(["a", "b"]),
        target_mean=np.zeros((1, 2)), ref_mean=np.zeros(2),
        target_ncells=np.array([3], dtype=np.int64), ref_ncells=int(4),
        log2_fold_change=np.zeros((1, 2)), p_value=np.ones((1, 2)),
        test_statistic=np.zeros((1, 2)), p_adj=np.zeros((1, 2)),
        flat_keep=None, output_columns=None)
    empty = empty_output_frame()
    assert empty.height == 0
    assert empty.columns == list(DEFAULT_OUTPUT_COLUMNS)
    assert empty.schema == real.schema                  # typed, matches real path
    assert pl.Null not in empty.schema.values()         # no Null columns
    # output_columns select+rename applies the same as the non-empty path
    renamed = empty_output_frame({"target": "guide", "p_value": "p"})
    assert renamed.columns == ["guide", "p"]
    assert renamed.schema["guide"] == pl.String
    assert renamed.schema["p"] == pl.Float64


def test_assemble_dataframe_extra_columns_unfiltered():
    import numpy as np
    from gpudge._output import assemble_dataframe
    n_g, n_f = 3, 4
    base = dict(
        target=np.array(["a", "b", "c"]),
        feature=np.array(["f0", "f1", "f2", "f3"]),
        target_mean=np.ones((n_g, n_f)), ref_mean=np.ones((n_g, n_f)),
        target_ncells=np.array([5, 6, 7]), ref_ncells=9,
        log2_fold_change=np.zeros((n_g, n_f)), p_value=np.zeros((n_g, n_f)),
        test_statistic=np.zeros((n_g, n_f)), p_adj=np.zeros((n_g, n_f)),
    )
    extra = np.arange(n_g * n_f, dtype=np.float64).reshape(n_g, n_f)
    df = assemble_dataframe(**base,
                            extra_columns={"tau=+0.5_p": extra})
    assert "tau=+0.5_p" in df.columns
    assert df["tau=+0.5_p"].to_numpy().tolist() == \
        extra.ravel().tolist()


def test_assemble_dataframe_extra_columns_respect_flat_keep():
    """Directional columns must be filtered by the SAME mask as base rows."""
    import numpy as np
    from gpudge._output import assemble_dataframe
    n_g, n_f = 2, 3
    keep = np.array([[True, False, True], [False, True, False]])
    base = dict(
        target=np.array(["a", "b"]),
        feature=np.array(["f0", "f1", "f2"]),
        target_mean=np.ones((n_g, n_f)), ref_mean=np.ones((n_g, n_f)),
        target_ncells=np.array([5, 6]), ref_ncells=9,
        log2_fold_change=np.zeros((n_g, n_f)), p_value=np.zeros((n_g, n_f)),
        test_statistic=np.zeros((n_g, n_f)), p_adj=np.zeros((n_g, n_f)),
    )
    extra = np.arange(n_g * n_f, dtype=np.float64).reshape(n_g, n_f)
    df = assemble_dataframe(**base, flat_keep=keep.ravel(),
                            extra_columns={"tau=-1_x": extra})
    assert df.height == 3
    assert df["tau=-1_x"].to_numpy().tolist() == [0.0, 2.0, 4.0]


def test_assemble_dataframe_extra_columns_shadowing_a_default_raises():
    import numpy as np
    import pytest
    from gpudge._output import assemble_dataframe
    n_g, n_f = 2, 2
    base = dict(
        target=np.array(["a", "b"]), feature=np.array(["f0", "f1"]),
        target_mean=np.ones((n_g, n_f)), ref_mean=np.ones((n_g, n_f)),
        target_ncells=np.array([5, 6]), ref_ncells=9,
        log2_fold_change=np.zeros((n_g, n_f)), p_value=np.zeros((n_g, n_f)),
        test_statistic=np.zeros((n_g, n_f)), p_adj=np.zeros((n_g, n_f)),
    )
    with pytest.raises(KeyError, match="shadows"):
        assemble_dataframe(**base,
                           extra_columns={"p_value": np.zeros((n_g, n_f))})


def test_empty_output_frame_carries_extra_names():
    from gpudge._output import empty_output_frame
    import polars as pl
    names = ["tau=+0.5_p", "tau=+0.5_padj"]
    df = empty_output_frame(extra_names=names)
    assert df.height == 0
    for n in names:
        assert df.schema[n] == pl.Float64


def test_fully_filtered_result_keeps_the_populated_schema():
    """A 0-row result must be typed like a populated one AND like
    empty_output_frame.

    polars infers dtype from values: a 0-length OBJECT-dtype numpy array gives
    pl.Object where a non-empty one gives pl.String. adata.var_names.to_numpy()
    and the ingest label array are both object dtype, so without the empty-case
    cast in assemble_dataframe a fully-filtered de() result carried
    target/feature as pl.Object -- mismatching on concat, the exact failure
    empty_output_frame exists to prevent. Found by the lfc_threshold GPU gate
    (test_zero_row_result_keeps_full_directional_schema).
    """
    import numpy as np
    from gpudge._output import assemble_dataframe, empty_output_frame
    n_g, n_f = 2, 3
    base = dict(
        target=np.array(["a", "b"], dtype=object),
        feature=np.array(["f0", "f1", "f2"], dtype=object),
        target_mean=np.ones((n_g, n_f)), ref_mean=np.ones((n_g, n_f)),
        target_ncells=np.array([5, 6]), ref_ncells=9,
        log2_fold_change=np.zeros((n_g, n_f)), p_value=np.zeros((n_g, n_f)),
        test_statistic=np.zeros((n_g, n_f)), p_adj=np.zeros((n_g, n_f)),
    )
    extra = {"tau=+0.5_p": np.zeros((n_g, n_f))}
    full = assemble_dataframe(**base, extra_columns=extra)
    none = assemble_dataframe(**base, extra_columns=extra,
                              flat_keep=np.zeros(n_g * n_f, dtype=bool))
    assert none.height == 0
    assert none.schema == full.schema
    assert none.schema == empty_output_frame(extra_names=list(extra)).schema
