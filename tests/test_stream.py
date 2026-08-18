# tests/test_stream.py
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from gpudge._stream import (
    iter_gene_chunks,
    _auto_gene_chunk_size,
    _pinned_buf_width,
    run_gene_chunks_with_recovery,
)
from conftest import needs_cuda


@needs_cuda
def test_dense_chunks_reassemble_full_matrix():
    X = np.arange(60, dtype=np.float32).reshape(5, 12)
    chunks = list(iter_gene_chunks(X, chunk_size=4, device="cuda"))
    got = torch.cat([c for _, c in chunks], dim=1).cpu().numpy()
    np.testing.assert_array_equal(got, X)
    starts = [s for s, _ in chunks]
    assert starts == [0, 4, 8]


@needs_cuda
def test_sparse_chunks_reassemble_full_matrix():
    X_dense = np.arange(60, dtype=np.float32).reshape(5, 12)
    X = sp.csr_matrix(X_dense)
    chunks = list(iter_gene_chunks(X, chunk_size=5, device="cuda"))
    got = torch.cat([c for _, c in chunks], dim=1).cpu().numpy()
    np.testing.assert_array_equal(got, X_dense)


@needs_cuda
def test_chunk_size_exceeds_n_genes_yields_one_chunk():
    X = np.zeros((3, 5), dtype=np.float32)
    chunks = list(iter_gene_chunks(X, chunk_size=100, device="cuda"))
    assert len(chunks) == 1


# --- _auto_gene_chunk_size (pure heuristic; GPU-free) ---

def test_auto_chunk_ref_mode_precise():
    # c50 regime with free=39 GiB passed explicitly: NTC ref 73,230 cells,
    # 4673 groups, arithmetic. 0.20 budget -> 4416 (deterministic formula
    # output; on the bench 4096-8192 throughput plateau).
    chunk = _auto_gene_chunk_size(
        free_bytes=39 * 1024**3, budget_n=73_230, n_groups=4673,
        mean_calc="arithmetic", n_genes=18_533, ref_mode=True)
    assert chunk == 4416
    assert chunk % 64 == 0


def test_auto_chunk_all_others_uses_020_fraction():
    # all_others (ref_mode=False) is wrapped by OOM recovery (gpudge#27) and
    # uses the 0.20 budget. L6 recalibration: the per-cell coefficient is 64
    # (not 24) -- _rank_with_ties holds ~6 full (n_cells, ch) f64/int64 arrays
    # simultaneously -- and the per-chunk (n_groups, ch) f64 accumulators are
    # now budgeted, so 100k cells / 20 groups -> 1280 (was 3456 at 24 B/cell).
    chunk = _auto_gene_chunk_size(
        free_bytes=39 * 1024**3, budget_n=100_000, n_groups=20,
        mean_calc="arithmetic", n_genes=18_533, ref_mode=False)
    assert chunk == 1280


def test_auto_chunk_scales_inversely_with_reference():
    small_ref = _auto_gene_chunk_size(
        free_bytes=39 * 1024**3, budget_n=73_230, n_groups=4673,
        mean_calc="arithmetic", n_genes=18_533, ref_mode=True)
    big_ref = _auto_gene_chunk_size(
        free_bytes=39 * 1024**3, budget_n=200_000, n_groups=4673,
        mean_calc="arithmetic", n_genes=18_533, ref_mode=True)
    assert big_ref < small_ref            # bigger reference -> smaller chunk


def test_auto_chunk_caps_at_n_genes_and_floors_at_16():
    assert _auto_gene_chunk_size(
        free_bytes=39 * 1024**3, budget_n=10, n_groups=2,
        mean_calc="arithmetic", n_genes=50, ref_mode=True) == 50
    assert _auto_gene_chunk_size(
        free_bytes=1 * 1024**3, budget_n=50_000_000, n_groups=4673,
        mean_calc="arithmetic", n_genes=18_533, ref_mode=True) == 16


def test_auto_gene_chunk_size_n_combos_zero_is_a_no_op():
    """lfc_threshold=None must not move the chunk by a single byte."""
    kw = dict(free_bytes=40 * 1024**3, budget_n=50_000, n_groups=4672,
              mean_calc="arithmetic", n_genes=18_500, ref_mode=True)
    assert _auto_gene_chunk_size(**kw, n_combos=0) == _auto_gene_chunk_size(**kw)


def test_auto_gene_chunk_size_shrinks_monotonically_in_n_combos():
    kw = dict(free_bytes=40 * 1024**3, budget_n=50_000, n_groups=4672,
              mean_calc="arithmetic", n_genes=18_500, ref_mode=True,
              max_group_rows=500)                   # reference-dominated
    sizes = [_auto_gene_chunk_size(**kw, n_combos=k) for k in (0, 2, 6, 10)]
    assert sizes == [6528, 5824, 4864, 4160]          # computed from the formula
    assert sizes == sorted(sizes, reverse=True)


def test_auto_gene_chunk_size_budgets_a_target_dominated_workload():
    """Small reference, huge target groups: the Phase-1 target tile dominates,
    and the sizer budgets it EVEN WITH NO FEATURE ACTIVE.

    Before the 2026-08 ultrareview the target term was gated on
    ``n_combos or n_levels``, so this shape returned 18496 in the base case --
    the whole gene axis, a chunk far too large for Phase 1 -- and paid an OOM
    downshift (or, under oom_recovery=False, an outright OOM). 384 is the point
    of that fix, not a regression.
    """
    kw = dict(free_bytes=40 * 1024**3, budget_n=2_000, n_groups=8,
              mean_calc="arithmetic", n_genes=18_500, ref_mode=True,
              max_group_rows=200_000)
    base = _auto_gene_chunk_size(**kw, n_combos=0)
    with_grid = _auto_gene_chunk_size(**kw, n_combos=6)
    assert base == _auto_gene_chunk_size(**kw)        # defaults are a no-op
    assert base == 384 and with_grid == 192           # from the formula
    assert with_grid < base                          # the grid adds tiles
    # The un-gating is what moved `base`: the same shape with an UNKNOWN tile
    # height (max_group_rows=0, i.e. cell_source_de) is deliberately unchanged
    # and still returns the whole gene axis.
    assert _auto_gene_chunk_size(**{**kw, "max_group_rows": 0}, n_combos=0) == 18496


def test_target_and_inmem_tile_constants_agree():
    """Two sizers, one physical cost. If these drift the literal/streaming
    sizer silently models a different target tile from the in-mem one."""
    from gpudge import _refpool
    from gpudge._stream import _TARGET_TILE_BYTES
    assert _TARGET_TILE_BYTES == _refpool._INMEM_TILE_BYTES


# --- n_levels (tau_star) in both sizers -----------------------------------
# Under-budgeting here has NO test-visible symptom: the allocation OOMs at
# full scale, the recovery loop downshifts the gene chunk to 64, and the run
# merely gets 10-20x slower. Small integration fixtures never reach it, so
# these pure-arithmetic assertions are the only guard.

def test_auto_gene_chunk_size_n_levels_zero_is_a_no_op():
    """tau_star=None must not move the chunk by a single byte."""
    kw = dict(free_bytes=40 * 1024**3, budget_n=50_000, n_groups=4672,
              mean_calc="arithmetic", n_genes=18_500, ref_mode=True)
    assert _auto_gene_chunk_size(**kw, n_levels=0) == _auto_gene_chunk_size(**kw)
    # ...and neither feature alone re-enables the other's term.
    assert (_auto_gene_chunk_size(**kw, n_combos=0, n_levels=0)
            == _auto_gene_chunk_size(**kw))


def test_auto_gene_chunk_size_shrinks_monotonically_in_n_levels():
    kw = dict(free_bytes=40 * 1024**3, budget_n=50_000, n_groups=4672,
              mean_calc="arithmetic", n_genes=18_500, ref_mode=True,
              max_group_rows=500)
    sizes = [_auto_gene_chunk_size(**kw, n_levels=k) for k in (0, 1, 2, 5)]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] > sizes[-1], "n_levels must actually cost something"


def test_sizer_treats_n_levels_as_a_row_count_not_a_level_count():
    """tau_star_se adds three ROWS to the tau* accumulator; the drivers pass
    len(levels) + 3, so the sizer must shrink accordingly."""
    kw = dict(free_bytes=40 * 1024**3, budget_n=20_000, n_groups=500,
              mean_calc="arithmetic", n_genes=20_000, ref_mode=True,
              max_group_rows=4_000)
    assert (_auto_gene_chunk_size(**kw, n_levels=4)
            < _auto_gene_chunk_size(**kw, n_levels=1))


def test_auto_gene_chunk_size_budgets_tau_star_on_a_target_dominated_load():
    """Small reference, huge target groups. Without the tau* target-side term
    the sizer would hand back the tau_star=None chunk and OOM."""
    kw = dict(free_bytes=40 * 1024**3, budget_n=2_000, n_groups=8,
              mean_calc="arithmetic", n_genes=18_500, ref_mode=True,
              max_group_rows=200_000)
    base = _auto_gene_chunk_size(**kw, n_levels=0)
    with_ts = _auto_gene_chunk_size(**kw, n_levels=2)
    assert with_ts < base


def test_auto_gene_chunk_size_counts_both_features_together():
    """Both active must budget MORE than either alone -- the tile term adds
    _LFC_TILE_BYTES and _TAUSTAR_TILE_BYTES independently."""
    kw = dict(free_bytes=40 * 1024**3, budget_n=2_000, n_groups=8,
              mean_calc="arithmetic", n_genes=18_500, ref_mode=True,
              max_group_rows=200_000)
    both = _auto_gene_chunk_size(**kw, n_combos=6, n_levels=2)
    assert both < _auto_gene_chunk_size(**kw, n_combos=6, n_levels=0)
    assert both < _auto_gene_chunk_size(**kw, n_combos=0, n_levels=2)


def test_inmem_sizer_n_levels_zero_is_a_no_op():
    from gpudge._refpool import _auto_gene_chunk_size_inmem
    kw = dict(free_bytes=40 * 1024**3, n_ref=200_000, n_genes=18_500,
              max_group_rows=200_000)
    assert (_auto_gene_chunk_size_inmem(**kw, n_levels=0)
            == _auto_gene_chunk_size_inmem(**kw))
    assert (_auto_gene_chunk_size_inmem(**kw, n_combos=0, n_levels=0)
            == _auto_gene_chunk_size_inmem(**kw))


def test_inmem_sizer_budgets_all_three_n_levels_terms():
    """The in-memory sizer has THREE n_combos terms and tau* needed an
    analogue in each: the chunk-INDEPENDENT resident accumulator, the
    per-target-cell tile working set, and the kernel's own (n_levels, chunk)
    output. Each is exercised in the regime where it alone can move the answer
    -- a single regime does not do it. Verified during review that a test using
    only the target-dominated regime leaves the resident and kernel-output
    terms completely unpinned: deleting either one changed nothing.
    """
    from gpudge._refpool import _auto_gene_chunk_size_inmem as sizer

    # (1) TILE term. Target-dominated, so bytes_per_gene is
    # max_group_rows * tile_bytes and _TAUSTAR_TILE_BYTES is the only thing
    # that can move the answer. Deleting that term restores 1024.
    tile = dict(free_bytes=40 * 1024**3, n_ref=1_000, n_genes=200_000,
                max_group_rows=200_000)
    assert sizer(**tile, n_levels=0) == 1024
    assert sizer(**tile, n_levels=1) == 576

    # (2) + (3) RESIDENT and KERNEL-OUTPUT terms, together, as an EXACT value.
    # An inequality cannot separate these two -- verified during review that
    # `<` still holds with either deleted -- so the assertion is the number.
    # The regime is tuned so both are live and neither the 64 floor nor the
    # n_genes cap binds: max_group_rows = 0 removes the tile term, n_ref = 1
    # puts 8 * n_levels above the n_ref * 40 ref-sort floor, and n_levels = 6
    # is the narrow window where the resident term still leaves a positive
    # budget while the base case stays under the cap.
    #   deleting the resident term  -> 604160
    #   deleting the kernel output  -> 124992
    # so either omission fails this assertion.
    iso = dict(free_bytes=1024**3 + 70_000_000, n_ref=1, n_genes=1_000_000,
               max_group_rows=0)
    assert sizer(**iso, n_levels=0) == 724992
    assert sizer(**iso, n_levels=6) == 104128

    # Both features together cost more than either alone.
    both = sizer(**tile, n_combos=6, n_levels=2)
    assert both < sizer(**tile, n_combos=6, n_levels=0)
    assert both < sizer(**tile, n_combos=0, n_levels=2)


def test_auto_gene_chunk_size_inmem_bump_is_constant_in_n_combos():
    """The per-combo transients are freed each iteration, so the TILE term must
    NOT scale with n_combos -- a x n_combos term would be the deleted
    reference-scaling design (spec 4.5).

    Parameters matter: bytes_per_gene is max(n_ref * REF_SORT, rows * tile + ...),
    so with n_ref=50k / max_group_rows=5k the reference-sort term dominates and
    the chunk does not move at all (14912 / 14912 / 14848 -- this test could not
    fail). n_ref=20k / max_group_rows=40k puts the tile term in charge.
    """
    from gpudge._refpool import _auto_gene_chunk_size_inmem
    kw = dict(free_bytes=60 * 1024**3, n_ref=20_000, n_genes=18_500,
              max_group_rows=40_000)
    base = _auto_gene_chunk_size_inmem(**kw, n_combos=0)
    two = _auto_gene_chunk_size_inmem(**kw, n_combos=2)
    twenty = _auto_gene_chunk_size_inmem(**kw, n_combos=20)
    assert base == _auto_gene_chunk_size_inmem(**kw)      # default is 0
    assert base == 8000 and two == 4352                   # from the formula
    assert two < base
    # 20 combos must not cost ~10x what 2 do: the only n_combos-scaled terms are
    # the tiny accumulator reserve and the kernel's (n_combos, chunk) outputs.
    assert twenty == pytest.approx(two, rel=0.05)


def test_pinned_buf_width_caps_at_n_genes():
    # user pins a chunk far larger than n_genes -> clamp to n_genes
    assert _pinned_buf_width(100_000, 18_500) == 18_500
    # pinned chunk below n_genes -> unchanged
    assert _pinned_buf_width(2_304, 18_500) == 2_304
    # equal -> that common value
    assert _pinned_buf_width(18_500, 18_500) == 18_500


# --- run_gene_chunks_with_recovery (OOM driver; GPU-free) ---

def test_driver_covers_all_genes_no_oom():
    calls = []
    run_gene_chunks_with_recovery(
        50, 20, lambda a, b: calls.append((a, b)), oom_recovery=True)
    assert calls == [(0, 20), (20, 40), (40, 50)]


def test_driver_halves_and_retries_on_oom():
    seen = []
    state = {"failed": False}

    def process(a, b):
        seen.append((a, b))
        if not state["failed"] and (b - a) > 10:   # OOM once at the big chunk
            state["failed"] = True
            raise torch.cuda.OutOfMemoryError("simulated")

    run_gene_chunks_with_recovery(50, 20, process, oom_recovery=True, floor=1)
    assert seen[0] == (0, 20)            # big chunk attempted first
    successful = seen[1:]
    assert successful[0] == (0, 10)      # retried from the same start, halved
    covered = 0
    for a, b in successful:
        assert a == covered and (b - a) <= 10
        covered = b
    assert covered == 50


def test_driver_subfloor_initial_downshifts_before_raising():
    # initial_chunk (40) < default floor (64): recovery should still halve once
    # (to ~20) instead of raising on the first OOM. (Gemini review, PR #25.)
    seen = []
    n = {"calls": 0}

    def process(a, b):
        seen.append((a, b))
        n["calls"] += 1
        if n["calls"] == 1:        # OOM the first (40-wide) attempt
            raise torch.cuda.OutOfMemoryError("simulated")

    run_gene_chunks_with_recovery(40, 40, process, oom_recovery=True)  # floor=64
    assert seen[0] == (0, 40)                 # tried 40 first
    assert (seen[1][1] - seen[1][0]) <= 20    # downshifted, not raised
    assert seen[-1][1] == 40                  # finished covering all genes


def test_driver_below_16_initial_downshifts_once():
    # explicit sub-16 initial chunk should still downshift once (~initial//2)
    # before raising, not raise on the first OOM. (Gemini review, PR #25.)
    seen = []
    n = {"c": 0}

    def process(a, b):
        seen.append((a, b))
        n["c"] += 1
        if n["c"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated")

    run_gene_chunks_with_recovery(8, 8, process, oom_recovery=True)  # floor=64
    assert seen[0] == (0, 8)
    assert (seen[1][1] - seen[1][0]) <= 4     # downshifted to <=4, not raised
    assert seen[-1][1] == 8


def test_driver_oom_recovery_false_raises():
    def process(a, b):
        raise torch.cuda.OutOfMemoryError("simulated")
    with pytest.raises(RuntimeError, match=r"oom_recovery=False"):
        run_gene_chunks_with_recovery(50, 20, process, oom_recovery=False)


def test_driver_floor_exhaustion_raises():
    def process(a, b):
        raise torch.cuda.OutOfMemoryError("simulated")
    with pytest.raises(RuntimeError, match=r"CUDA OOM at gpu_gene_chunk_size=64"):
        run_gene_chunks_with_recovery(1000, 64, process, oom_recovery=True, floor=64)


def test_driver_returns_final_chunk_no_oom():
    # No OOM: returns the initial chunk unchanged. Lets the streaming driver
    # carry a (possibly downshifted) width across groups. (ultrareview #43)
    final = run_gene_chunks_with_recovery(50, 20, lambda a, b: None,
                                          oom_recovery=True)
    assert final == 20


def test_driver_returns_final_downshifted_chunk():
    # OOM once at the full width -> halves to 10 and never OOMs again -> the
    # driver returns 10 so the caller can start the next pass there instead of
    # rediscovering the downshift. (ultrareview #43)
    seen = []
    def process(a, b):
        seen.append((a, b))
        if len(seen) == 1:                     # OOM only on the first (20-wide) attempt
            raise torch.cuda.OutOfMemoryError("simulated")
    final = run_gene_chunks_with_recovery(40, 20, process,
                                          oom_recovery=True, floor=1)
    assert final == 10


# --- device-decode (cupy) OOM recovery (#78) ---

def test_driver_downshifts_on_nontorch_oom(monkeypatch):
    # A cupy-style OOM (MemoryError-based — disjoint from torch's RuntimeError-based
    # OutOfMemoryError) must trigger the SAME chunk-halving recovery, not crash.
    # Simulated without cupy/GPU by injecting a fake OOM type into the tuple.
    from gpudge import _stream

    class _FakeCupyOOM(MemoryError):
        pass

    monkeypatch.setattr(
        _stream, "oom_error_types",
        lambda: (torch.cuda.OutOfMemoryError, _FakeCupyOOM),
    )
    seen = []
    state = {"failed": False}

    def process(a, b):
        seen.append((a, b))
        if not state["failed"] and (b - a) > 10:
            state["failed"] = True
            raise _FakeCupyOOM("device pool exhausted")

    final = run_gene_chunks_with_recovery(50, 20, process,
                                          oom_recovery=True, floor=1)
    assert seen[0] == (0, 20)            # big chunk attempted first
    assert seen[1] == (0, 10)            # retried from same start, halved
    assert seen[-1][1] == 50            # finished covering all genes
    assert final == 10


def test_driver_does_not_catch_oom_type_absent_from_tuple(monkeypatch):
    # We catch SPECIFIC OOM types, not all MemoryError: an OOM type not in the
    # resolved tuple must propagate (regression guard against over-broadening).
    from gpudge import _stream

    class _FakeCupyOOM(MemoryError):
        pass

    monkeypatch.setattr(
        _stream, "oom_error_types",
        lambda: (torch.cuda.OutOfMemoryError,),   # torch-only; fake type absent
    )

    def process(a, b):
        raise _FakeCupyOOM("must propagate")

    with pytest.raises(_FakeCupyOOM):
        run_gene_chunks_with_recovery(50, 20, process,
                                      oom_recovery=True, floor=1)


# --- 2026-08 ultrareview (lows): degenerate-shape handling -------------------

def test_auto_gene_chunk_size_floors_at_one_for_zero_genes():
    """A zero-gene input must not drive the sizer to 0.

    `run_gene_chunks_with_recovery` rejects `initial_chunk <= 0` with a message
    naming an internal parameter the caller never passed, so `de()` on a
    0-var AnnData died there instead of returning the typed empty frame it
    already returns when `gpu_gene_chunk_size` is pinned.
    """
    for ref_mode in (True, False):
        assert _auto_gene_chunk_size(
            free_bytes=39 * 1024**3, budget_n=1000, n_groups=3,
            mean_calc="arithmetic", n_genes=0, ref_mode=ref_mode) == 1


def test_auto_gene_chunk_size_inmem_floors_at_one_for_zero_genes():
    from gpudge._refpool import _auto_gene_chunk_size_inmem
    assert _auto_gene_chunk_size_inmem(
        free_bytes=39 * 1024**3, n_ref=1000, n_genes=0,
        max_group_rows=500) == 1


def test_zero_genes_drives_zero_chunks_without_raising():
    """The floor is only useful if the driver then no-ops. Pins the pairing."""
    seen = []
    run_gene_chunks_with_recovery(0, 1, lambda s, e: seen.append((s, e)))
    assert seen == []


# --- 2026-08 ultrareview (lows): archive reference= validation --------------
#
# Both backends gather the archive's reference pool WHOLE, so `reference=` can
# only NAME it, never subset it. Two inputs went wrong before
# `validate_archive_reference`: a list was stringified into a
# self-contradictory "not among ... labels" message. Naming ONE label of a
# multi-label pool is specified to be legal and still use the whole pool, so
# that is pinned here too rather than "fixed".

def test_validate_archive_reference_none_reports_the_whole_pool():
    from gpudge._stream_backend import validate_archive_reference
    assert validate_archive_reference(None, {"safe", "ntc"}) == "ntc|safe"


def test_validate_archive_reference_accepts_the_sole_label():
    from gpudge._stream_backend import validate_archive_reference
    assert validate_archive_reference("ntc", {"ntc"}) == "ntc"


def test_validate_archive_reference_accepts_a_list_naming_the_whole_pool():
    """The self-contradictory message: `reference=['ntc','safe'] is not among
    the archive's reference labels ['ntc','safe']`."""
    from gpudge._stream_backend import validate_archive_reference
    assert validate_archive_reference(["safe", "ntc"],
                                      {"ntc", "safe"}) == "ntc|safe"


@pytest.mark.parametrize("subset,want", [("ntc", "ntc"), (["ntc"], "ntc"),
                                         ("safe", "safe"), (("safe",), "safe")])
def test_validate_archive_reference_accepts_a_partial_pool(subset, want):
    """SPECIFIED behaviour, not an oversight: "reference=<label> is validated
    for membership in the reference labels and otherwise does not subset -- the
    pool is all reference rows" (2026-07-31 cell-layout design). Pinned so the
    membership-only contract is not "tightened" by a future reader (I tried).
    The EXACT label is asserted -- `in ("ntc", "safe")` would pass if the wrong
    known label came back."""
    from gpudge._stream_backend import validate_archive_reference
    assert validate_archive_reference(subset, {"ntc", "safe"}) == want


@pytest.mark.parametrize("bad", ["nope", ["ntc", "nope"], ["nope"]])
def test_validate_archive_reference_rejects_an_unknown_label(bad):
    from gpudge._stream_backend import validate_archive_reference
    with pytest.raises(ValueError, match="is not among the archive"):
        validate_archive_reference(bad, {"ntc", "safe"})


def test_validate_archive_reference_does_not_iterate_a_string():
    """'ntc' must not decompose into {'n','t','c'} — str is a Sequence."""
    from gpudge._stream_backend import validate_archive_reference
    with pytest.raises(ValueError, match="is not among the archive"):
        validate_archive_reference("ntc", {"n", "t", "c"})


@pytest.mark.parametrize("ref", [
    "ntc",                                       # plain str
    b"ntc",                                      # bytes
    bytearray(b"ntc"),
    np.str_("ntc"),                              # str subclass -- must not leak
    np.array("ntc"),                             # 0-d: Iterable, but un-iterable
    np.array(b"ntc"),                            # 0-d bytes -> "np.bytes_(...)"
])
def test_validate_archive_reference_returns_a_plain_str_label(ref):
    from gpudge._stream_backend import validate_archive_reference
    got = validate_archive_reference(ref, {"ntc"})
    assert got == "ntc"
    assert type(got) is str


def test_validate_archive_reference_consumes_a_generator_once():
    """`reference` is read twice (membership, then the message), so it has to
    be materialized -- a generator would otherwise validate as empty."""
    from gpudge._stream_backend import validate_archive_reference
    gen = (x for x in ["safe", "ntc"])
    assert validate_archive_reference(gen, {"ntc", "safe"}) == "ntc|safe"


def test_validate_archive_reference_rejects_an_empty_sequence():
    """`[]` names NONE of the pool, not "part" of it."""
    from gpudge._stream_backend import validate_archive_reference
    with pytest.raises(ValueError, match="is an empty sequence"):
        validate_archive_reference([], {"ntc", "safe"})
