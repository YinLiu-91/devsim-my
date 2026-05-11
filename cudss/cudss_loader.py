import ctypes
import ctypes.util
import glob
import importlib
import os
import site
from dataclasses import dataclass
from typing import Optional


_CUDSS_LIB_CANDIDATES = (
    "libcudss.so",
    "libcudss.so.0",
)


@dataclass
class CuDSSRuntime:
    """Descriptor for detected cuDSS runtime in current process."""

    python_module: Optional[object]
    library: Optional[ctypes.CDLL]
    source: str

    @property
    def available(self) -> bool:
        return bool(self.python_module or self.library)


def _load_python_binding() -> Optional[object]:
    """
    Prefer a Python binding when present.

    Current ecosystem variants are not guaranteed; keep this best-effort and
    avoid hard dependency so DEVSIM can still run in CPU-only environments.
    """
    module_candidates = ("nvidia.cudss",)
    for name in module_candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    return None


def _load_shared_library() -> Optional[ctypes.CDLL]:
    """
    Load libcudss via explicit path or default loader search path.
    """
    explicit = os.environ.get("DEVSIM_CUDSS_LIB")
    if explicit:
        return ctypes.CDLL(explicit)

    found = ctypes.util.find_library("cudss")
    if found:
        return ctypes.CDLL(found)

    # Pip runtime package usually installs to:
    #   <site-packages>/nvidia/cu12/lib/libcudss.so.0
    # Try those explicit paths before generic loader names.
    path_candidates = []
    for root in site.getsitepackages():
        path_candidates.extend(
            (
                os.path.join(root, "nvidia", "cu12", "lib", "libcudss.so"),
                os.path.join(root, "nvidia", "cu12", "lib", "libcudss.so.0"),
                os.path.join(root, "nvidia", "cudss", "lib", "libcudss.so"),
                os.path.join(root, "nvidia", "cudss", "lib", "libcudss.so.0"),
            )
        )
        path_candidates.extend(
            glob.glob(os.path.join(root, "nvidia", "cu*", "lib", "libcudss.so*"))
        )
    for p in path_candidates:
        if os.path.isfile(p):
            try:
                return ctypes.CDLL(p)
            except OSError:
                pass

    for name in _CUDSS_LIB_CANDIDATES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    return None


def detect_cudss_runtime() -> CuDSSRuntime:
    """
    Detect available cuDSS runtime (python binding or shared library).
    """
    py_mod = _load_python_binding()
    if py_mod is not None:
        return CuDSSRuntime(python_module=py_mod, library=None, source="python-module")

    try:
        lib = _load_shared_library()
    except OSError:
        lib = None
    if lib is not None:
        return CuDSSRuntime(python_module=None, library=lib, source="shared-library")

    return CuDSSRuntime(python_module=None, library=None, source="unavailable")


def get_unavailable_message() -> str:
    return (
        "cuDSS runtime is unavailable. Install NVIDIA cuDSS runtime/bindings or set "
        "DEVSIM_CUDSS_LIB to libcudss path."
    )
