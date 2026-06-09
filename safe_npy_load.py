"""Safe numpy loader for object-dtype .npy files.

Replaces `np.load(path, allow_pickle=True)` with a version that only allows
a hardcoded allowlist of numpy reconstruction primitives to be resolved
during unpickling. Any other GLOBAL opcode raises UnpicklingError before
the import-and-call sequence runs.

Usage:
    from safe_npy_load import safe_load_npy
    arr = safe_load_npy("path/to/file.npy")
"""
from __future__ import annotations

import ast
import io
import pickle
import struct
from pathlib import Path
from typing import Union
import os #debug

import numpy as np
from numpy._core.multiarray import _reconstruct, scalar

# (module, name) pairs that legitimate numpy object-dtype pickles emit.
# Confirmed via fickling fingerprint scan of this dataset.
# Anything outside this set is blocked.
ALLOWED = {
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy._core.multiarray", "scalar"): scalar,
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        key = (module, name)
        if key not in ALLOWED:
            raise pickle.UnpicklingError(
                f"blocked unpickling of {module}.{name} "
                f"(not in numpy allowlist)"
            )
        return ALLOWED[key]


def _parse_npy_header(fh):
    if fh.read(6) != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major, _minor = fh.read(2)
    if major == 1:
        (hlen,) = struct.unpack("<H", fh.read(2))
    elif major in (2, 3):
        (hlen,) = struct.unpack("<I", fh.read(4))
    else:
        raise ValueError(f"unsupported .npy version {major}")
    return ast.literal_eval(fh.read(hlen).decode("latin-1").strip())


def _descr_uses_pickle(descr):
    if isinstance(descr, str):
        return "O" in descr
    if isinstance(descr, list):
        return any(_descr_uses_pickle(d[1]) for d in descr if len(d) >= 2)
    return False


def safe_load_npy(path: Union[str, Path]) -> np.ndarray:
    """Load a .npy file. Object dtype goes through an allowlist-restricted unpickler."""
    path = str(path)
    
    print("cwd:", os.getcwd()) # debug
    print("path repr:", repr(path)) # debug
    print("abspath:", os.path.abspath(path)) # debug
    print("exists:", os.path.exists(path)) # debug
    print("isfile:", os.path.isfile(path)) # debug

    with open(path, "rb") as fh:
        meta = _parse_npy_header(fh)
        if not _descr_uses_pickle(meta.get("descr")):
            return np.load(path, allow_pickle=False)
        data = fh.read()
    return _SafeUnpickler(io.BytesIO(data)).load()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: safe_npy_load.py <file.npy> [file.npy ...]")
        sys.exit(1)
    rc = 0
    for p in sys.argv[1:]:
        try:
            arr = safe_load_npy(p)
            print(f"OK    {p}  shape={arr.shape} dtype={arr.dtype}")
        except Exception as e:
            print(f"FAIL  {p}  {type(e).__name__}: {e}")
            rc = 1
    sys.exit(rc)
