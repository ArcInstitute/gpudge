# tests/test_ingest.py
import numpy as np
import pytest
from gpudge._ingest import ALL_OTHERS, ingest


def test_ingest_basic_extracts_labels(synth_small):
    state = ingest(synth_small, groupby="comparison", reference="ntc")
    assert state.n_cells == 200
    assert state.n_genes == 50
    assert state.ref_label == "ntc"
    # Labels are integers; ref_label_idx points at the encoded reference.
    assert state.labels.shape == (200,)
    assert state.labels.dtype.kind in "iu"
    assert state.unique_labels[state.ref_label_idx] == "ntc"
    # Target labels exclude the reference.
    assert "ntc" not in state.target_labels


def test_ingest_missing_groupby_column_raises(synth_small):
    with pytest.raises(ValueError, match="groupby column"):
        ingest(synth_small, groupby="nope", reference="ntc")


def test_ingest_unknown_reference_raises(synth_small):
    with pytest.raises(ValueError, match="reference"):
        ingest(synth_small, groupby="comparison", reference="not-a-group")


@pytest.mark.parametrize("missing", [np.nan, None])
def test_ingest_missing_groupby_label_raises(synth_small, missing):
    """NaN/None in the groupby column must be rejected, not silently bucketed
    into a 'nan'/'None' group (would skew literal-reference and all-others)."""
    col = synth_small.obs["comparison"].astype(object).to_numpy().copy()
    col[0] = missing
    synth_small.obs["comparison"] = col
    with pytest.raises(ValueError, match="missing"):
        ingest(synth_small, groupby="comparison", reference="ntc")


def test_ingest_all_others_special(synth_small):
    state = ingest(synth_small, groupby="comparison", reference=ALL_OTHERS)
    assert state.ref_label == ALL_OTHERS
    assert state.ref_label_idx is None
    # All groups become "targets" — each compared vs rest.
    assert set(state.target_labels) == set(synth_small.obs["comparison"].unique())


def test_ingest_legacy_all_others_string_is_deprecated_but_works(synth_small):
    """Direct ingest() callers also get the DeprecationWarning + remap."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        state = ingest(synth_small, groupby="comparison",
                       reference="all_others")
    assert state.ref_label == ALL_OTHERS
    assert state.ref_label_idx is None


def test_reject_missing_group_labels_policy():
    """One shared policy for the stringified-missing-value labels, used by both
    streaming backends. Discriminating in BOTH directions: the four reachable
    spellings must raise, and the three that .astype(str) cannot produce must be
    ACCEPTED -- rejecting those refused archives the in-memory path accepts.
    (ultrareview 2026-08; set narrowed after the codex review.)
    """
    import pytest
    from gpudge._ingest import (
        MISSING_LABEL_SPELLINGS, reject_missing_group_labels,
    )
    kw = dict(where="X.csad", remedy="Drop them.")
    for bad in ("nan", "None", "<NA>", "NaT"):
        with pytest.raises(ValueError, match="missing"):
            reject_missing_group_labels(["g1", bad], **kw)
    # Cannot arise from the conversion -> must NOT be rejected.
    for ok in ("NaN", "NAN", "", "nan_cluster", "None_of_the_above"):
        reject_missing_group_labels(["g1", ok], **kw)
    assert MISSING_LABEL_SPELLINGS == {"nan", "None", "<NA>", "NaT"}
    # The message must name the offender and say a real group can be renamed.
    with pytest.raises(ValueError, match=r"rename"):
        reject_missing_group_labels(["nan"], **kw)
