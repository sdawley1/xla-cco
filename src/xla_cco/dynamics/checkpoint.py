#!/usr/bin/env python3

"""
Atomic checkpoints of (batched) MD states.

``save_state`` writes every MDState leaf (including the PRNG key and the
Andersen accumulator) plus a JSON metadata string to ``<path>.tmp`` and
renames it over ``path`` (os.replace), so a crash never leaves a partial
checkpoint. ``load_state`` rebuilds the state with the tree structure of a
template (``like``) and returns the metadata; ``read_meta`` reads only the
metadata. Unreadable files (truncated, garbage, missing entries) raise
CheckpointError, a ValueError.

File layout (npz, no pickles): ``leaf_000``, ``leaf_001``, ... in
``jax.tree_util`` flattening order of the MDState, ``__n_leaves__`` (int64)
and ``__meta__`` (0-d unicode, JSON). The template's leaves may be arrays or
``jax.ShapeDtypeStruct`` (only shape and dtype are read).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import contextlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.dynamics.state import MDState

# ============================================================
# CONSTANTS
# ============================================================

META_KEY: str = "__meta__"
COUNT_KEY: str = "__n_leaves__"
LEAF_FORMAT: str = "leaf_{:03d}"
TMP_SUFFIX: str = ".tmp"

# ============================================================
# FUNCTIONS
# ============================================================


class CheckpointError(ValueError):
    """A checkpoint file that is unreadable or does not match its template."""


def tmp_path_of(path: Path) -> Path:
    """Path of the temporary file that ``save_state`` renames over ``path``."""
    return Path(str(path) + TMP_SUFFIX)


def save_state(path: Path, state: MDState, meta: Dict[str, Any]) -> None:
    """Write ``state`` and ``meta`` atomically to ``path`` (.npz).

    The file is written to ``<path>.tmp``, flushed to disk and renamed over
    ``path``; on any error the temporary file is removed and ``path`` keeps
    its previous content.

    Raises
    ------
    TypeError
        If ``state`` is not an MDState or ``meta`` is not a JSON-serializable
        dict.
    """
    if not isinstance(state, MDState):
        raise TypeError(f"state must be an MDState, got {type(state).__name__}")
    if not isinstance(meta, dict):
        raise TypeError(f"meta must be a dict, got {type(meta).__name__}")
    meta_json = json.dumps(meta, sort_keys=True)  # TypeError if not serializable
    leaves = jax.tree_util.tree_leaves(state)
    assert len(leaves) > 0, "an MDState has array leaves"
    arrays: Dict[str, Any] = {
        LEAF_FORMAT.format(i): np.asarray(x) for i, x in enumerate(leaves)
    }
    arrays[META_KEY] = np.asarray(meta_json)
    arrays[COUNT_KEY] = np.int64(len(leaves))
    path = Path(path)
    tmp = tmp_path_of(path)
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _template_specs(like: MDState) -> Tuple[List[Tuple[Tuple[int, ...], Any]], Any]:
    """(shape, dtype) of every template leaf and the template's treedef."""
    assert isinstance(like, MDState), "template must be an MDState"
    leaves, treedef = jax.tree_util.tree_flatten(like)
    specs = [(tuple(int(n) for n in x.shape), np.dtype(x.dtype)) for x in leaves]
    assert len(specs) == len(leaves) > 0, "an MDState has array leaves"
    return specs, treedef


def _read_meta(z: Any, path: Path) -> Dict[str, Any]:
    """Validate the bookkeeping entries of an open checkpoint; return the meta."""
    files = set(z.files)
    if META_KEY not in files or COUNT_KEY not in files:
        raise CheckpointError(f"{path} is not an MD checkpoint (no {META_KEY})")
    meta = json.loads(str(z[META_KEY][()]))
    if not isinstance(meta, dict):
        raise CheckpointError(f"{path}: metadata is not a JSON object")
    assert int(z[COUNT_KEY]) >= 0, "leaf count is non-negative"
    return meta


def _read_leaves(z: Any, path: Path, specs: List[Tuple[Any, Any]]) -> List[Any]:
    """Leaves of an open checkpoint, checked against the template specs."""
    n_file = int(z[COUNT_KEY])
    if n_file != len(specs):
        raise CheckpointError(
            f"{path} holds {n_file} leaves; the template has {len(specs)}"
        )
    out = []
    for i, (shape, dtype) in enumerate(specs):
        a = np.asarray(z[LEAF_FORMAT.format(i)])
        if a.shape != shape or a.dtype != dtype:
            raise CheckpointError(
                f"{path} leaf {i}: stored {a.shape} {a.dtype}, template {shape} {dtype}"
            )
        out.append(jnp.asarray(a, dtype=dtype))
    assert len(out) == len(specs), "one array per template leaf"
    return out


def _corrupt(path: Path, exc: Exception) -> CheckpointError:
    """The ValueError raised for an unreadable (truncated, garbage) file."""
    assert exc is not None, "need the underlying error"
    assert not isinstance(exc, CheckpointError), "already a clear message"
    return CheckpointError(
        f"{path} is not a readable MD checkpoint (truncated or corrupt: "
        f"{type(exc).__name__}: {exc})"
    )


def read_meta(path: Path) -> Dict[str, Any]:
    """Return only the metadata of a checkpoint (no template needed).

    Raises
    ------
    ValueError
        If the file is truncated, corrupt, not an MD checkpoint, or its
        metadata is not a JSON object.
    OSError
        If the file cannot be opened.
    """
    try:
        with np.load(Path(path), allow_pickle=False) as z:
            meta = _read_meta(z, Path(path))
    except CheckpointError:
        raise
    except (zipfile.BadZipFile, EOFError, KeyError, ValueError) as exc:
        raise _corrupt(Path(path), exc) from exc
    assert isinstance(meta, dict), "meta is a JSON object"
    return meta


def load_state(path: Path, like: MDState) -> Tuple[MDState, Dict[str, Any]]:
    """Read a checkpoint into the structure of ``like``; returns (state, meta).

    Raises
    ------
    ValueError
        If the file is truncated or corrupt (zip, npy or JSON errors), lacks
        an entry, is not an MD checkpoint, its leaf count, any leaf shape or
        dtype differs from the template, or the metadata is not a JSON
        object (all as CheckpointError, a ValueError).
    TypeError
        If ``like`` is not an MDState.
    OSError
        If the file cannot be opened.
    """
    if not isinstance(like, MDState):
        raise TypeError(f"like must be an MDState, got {type(like).__name__}")
    specs, treedef = _template_specs(like)
    try:
        with np.load(Path(path), allow_pickle=False) as z:
            meta = _read_meta(z, Path(path))
            out = _read_leaves(z, Path(path), specs)
    except CheckpointError:
        raise
    except (zipfile.BadZipFile, EOFError, KeyError, ValueError) as exc:
        raise _corrupt(Path(path), exc) from exc
    state = jax.tree_util.tree_unflatten(treedef, out)
    assert isinstance(state, MDState), "template structure is an MDState"
    return state, meta
