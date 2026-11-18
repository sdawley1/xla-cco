#!/usr/bin/env python3

"""
Unit tests of the atomic MD checkpoints (``xla_cco.dynamics.checkpoint``).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
import tempfile
from pathlib import Path
from typing import Any, Optional

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from md_states import leaves_bitwise_equal, random_state

# ----- project -----
from xla_cco.dynamics import checkpoint
from xla_cco.dynamics.checkpoint import load_state, read_meta, save_state, tmp_path_of

# ============================================================
# CONSTANTS
# ============================================================

META = {"phase": 3, "cycle": [1, 2], "note": "resume", "t": 0.1 + 0.2, "x": None}

# ============================================================
# HELPERS
# ============================================================


def _zeroed(state: Any) -> Any:
    """Same structure, every leaf zero (a template that differs in values)."""
    return jax.tree_util.tree_map(jnp.zeros_like, state)


def _partial_savez(file: Any, *args: Any, **kwds: Any) -> None:
    """np.savez that writes a few bytes and then fails (a crash mid-write)."""
    if hasattr(file, "write"):
        file.write(b"PK\x03\x04partial")
    else:
        name = str(file) if str(file).endswith(".npz") else str(file) + ".npz"
        Path(name).write_bytes(b"PK\x03\x04partial")
    raise OSError("simulated crash mid-write")


# ============================================================
# ROUND TRIP
# ============================================================


@pytest.mark.critical
@settings(max_examples=40)
@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.sampled_from((1, 3, 8)),
    d=st.sampled_from((1, 2, 3)),
    w=st.sampled_from((0, 1, 5)),
    n_rec=st.sampled_from((0, 1, 4)),
    n_trace=st.sampled_from((0, 2)),
    batch=st.sampled_from((None, 1, 3)),
    int_dtype=st.sampled_from((np.int32, np.int64)),
)
@example(seed=0, n=1, d=1, w=0, n_rec=0, n_trace=0, batch=None, int_dtype=np.int64)
def test_round_trip_is_bitwise(
    seed: int, n: int, d: int, w: int, n_rec: int, n_trace: int,
    batch: Optional[int], int_dtype: Any,
) -> None:  # fmt: skip
    state = random_state(seed, n, d, w, n_rec, n_trace, batch, int_dtype)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.npz"
        save_state(path, state, META)
        back, meta = load_state(path, _zeroed(state))
        assert not tmp_path_of(path).exists()
    assert leaves_bitwise_equal(back, state)
    assert meta == META
    assert np.asarray(back.key).dtype == np.uint32


@pytest.mark.critical
def test_round_trip_restores_key_and_accumulator() -> None:
    state = random_state(11, 4, 2, 3, 2, 1, batch=2)
    state = dataclasses.replace(
        state,
        key=jnp.asarray([[0xFFFFFFFF, 7], [123456789, 0]], dtype=jnp.uint32),
        reset_acc=jnp.asarray([0.99, 1e-300]),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.npz"
        save_state(path, state, {})
        back, meta = load_state(path, _zeroed(state))
    assert meta == {}
    np.testing.assert_array_equal(np.asarray(back.key), np.asarray(state.key))
    assert np.asarray(back.key).dtype == np.uint32
    assert np.asarray(back.reset_acc).tobytes() == np.asarray(state.reset_acc).tobytes()
    # the restored key continues the same random stream
    a = jax.random.normal(back.key[0], (3,))
    b = jax.random.normal(state.key[0], (3,))
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


@pytest.mark.high
def test_meta_round_trip_and_template_of_shapes() -> None:
    state = random_state(5, 2, 2, 2, 1, 1, batch=None)
    like = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state
    )
    meta = {"nested": {"a": [1.5, -2.0]}, "flag": True, "big": 2**53 + 1}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.npz"
        save_state(path, state, meta)
        back, got = load_state(path, like)
    assert got == meta and leaves_bitwise_equal(back, state)


# ============================================================
# ATOMICITY
# ============================================================


@pytest.mark.critical
def test_crash_mid_write_keeps_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = random_state(1, 3, 2, 2, 1, 1, batch=2)
    new = random_state(2, 3, 2, 2, 1, 1, batch=2)
    path = tmp_path / "ckpt.npz"
    save_state(path, old, {"phase": 1})
    with monkeypatch.context() as m:
        m.setattr(checkpoint.np, "savez", _partial_savez)
        with pytest.raises(OSError, match="mid-write"):
            save_state(path, new, {"phase": 2})
    back, meta = load_state(path, _zeroed(old))
    assert meta == {"phase": 1} and leaves_bitwise_equal(back, old)
    assert not tmp_path_of(path).exists()


@pytest.mark.high
def test_stale_tmp_file_is_ignored_and_replaced(tmp_path: Path) -> None:
    old = random_state(3, 2, 1, 1, 1, 0, batch=1)
    new = random_state(4, 2, 1, 1, 1, 0, batch=1)
    path = tmp_path / "ckpt.npz"
    save_state(path, old, {"v": 1})
    tmp_path_of(path).write_bytes(b"garbage from a killed job")
    back, meta = load_state(path, _zeroed(old))
    assert meta == {"v": 1} and leaves_bitwise_equal(back, old)
    save_state(path, new, {"v": 2})
    back, meta = load_state(path, _zeroed(new))
    assert meta == {"v": 2} and leaves_bitwise_equal(back, new)
    assert not tmp_path_of(path).exists()


# ============================================================
# ERROR CONTRACTS
# ============================================================


@pytest.mark.medium
@pytest.mark.parametrize(
    "other",
    [
        dict(n=4, d=2, w=2, n_rec=1, n_trace=1, batch=2),  # n differs
        dict(n=3, d=2, w=3, n_rec=1, n_trace=1, batch=2),  # w differs
        dict(n=3, d=2, w=2, n_rec=1, n_trace=1, batch=3),  # lanes differ
        dict(n=3, d=2, w=2, n_rec=2, n_trace=1, batch=2),  # capacity differs
        dict(n=3, d=2, w=2, n_rec=1, n_trace=1, batch=2, int_dtype=np.int32),
    ],
)
def test_mismatched_template_raises(tmp_path: Path, other: Any) -> None:
    state = random_state(0, n=3, d=2, w=2, n_rec=1, n_trace=1, batch=2)
    path = tmp_path / "ckpt.npz"
    save_state(path, state, {})
    with pytest.raises(ValueError, match="leaf"):
        load_state(path, random_state(0, **other))


@pytest.mark.medium
def test_rejects_non_checkpoint_and_bad_meta(tmp_path: Path) -> None:
    state = random_state(0, 2, 1, 1, 1, 1)
    np.savez(tmp_path / "plain.npz", a=np.zeros(2))
    with pytest.raises(ValueError, match="not an MD checkpoint"):
        load_state(tmp_path / "plain.npz", state)
    with pytest.raises(TypeError):
        save_state(tmp_path / "x.npz", state, {"bad": object()})
    with pytest.raises(TypeError, match="MDState"):
        save_state(tmp_path / "x.npz", state.snap, {})  # type: ignore[arg-type]
    assert not (tmp_path / "x.npz").exists()
    with pytest.raises(OSError):
        load_state(tmp_path / "missing.npz", state)


@pytest.mark.high
@settings(max_examples=60)
@given(cut=st.floats(0.0, 1.0, exclude_max=True))
@example(cut=0.5)
@example(cut=0.0)
def test_regression_D040_truncated_checkpoint_raises_value_error(cut: float) -> None:
    """D-040: a truncated checkpoint raised zipfile.BadZipFile, not ValueError."""
    state = random_state(7, 3, 2, 2, 1, 1, batch=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.npz"
        save_state(path, state, {"a": 1})
        raw = path.read_bytes()
        path.write_bytes(raw[: int(cut * len(raw))])
        with pytest.raises(ValueError, match="truncated or corrupt|not an MD"):
            load_state(path, state)
        with pytest.raises(ValueError, match="truncated or corrupt|not an MD"):
            read_meta(path)


@pytest.mark.medium
@pytest.mark.parametrize(
    "case, match",
    [
        ("garbage", "truncated or corrupt"),
        ("no_meta", "not an MD checkpoint"),
        ("meta_not_json", "truncated or corrupt"),
        ("meta_not_object", "not a JSON object"),
        ("missing_leaf", "truncated or corrupt"),
    ],
)
def test_damaged_checkpoint_raises_value_error(
    tmp_path: Path, case: str, match: str
) -> None:
    state = random_state(8, 2, 1, 1, 1, 1)
    path = tmp_path / "ckpt.npz"
    save_state(path, state, {"a": 1})
    with np.load(path) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    if case == "garbage":
        path.write_bytes(b"\x00\x01 not a zip file" * 10)
    else:
        if case == "no_meta":
            del arrays[checkpoint.META_KEY]
        elif case == "meta_not_json":
            arrays[checkpoint.META_KEY] = np.asarray("{not json")
        elif case == "meta_not_object":
            arrays[checkpoint.META_KEY] = np.asarray("[1, 2]")
        else:
            del arrays[checkpoint.LEAF_FORMAT.format(3)]
        np.savez(path, **arrays)
    with pytest.raises(ValueError, match=match):
        load_state(path, state)
    if case != "missing_leaf":  # the metadata alone is still readable there
        with pytest.raises(ValueError, match=match):
            read_meta(path)


@pytest.mark.high
def test_read_meta_matches_load_state(tmp_path: Path) -> None:
    state = random_state(9, 2, 2, 2, 2, 1, batch=2)
    path = tmp_path / "ckpt.npz"
    save_state(path, state, META)
    assert read_meta(path) == META == load_state(path, state)[1]
