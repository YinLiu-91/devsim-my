"""
DEVSIM custom direct solver callback shim for CUDA cuDSS (Phase-1).

Scope:
- Linux
- NVIDIA GPU
- DC real-valued solve path
"""

from __future__ import annotations

import array
import ctypes
import ctypes.util
import os
import site
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .cudss_loader import detect_cudss_runtime, get_unavailable_message

_DEBUG = os.environ.get("DEVSIM_CUDSS_DEBUG", "").lower() in {"1", "true", "yes", "on"}
_USE_PINNED_STAGING = os.environ.get("DEVSIM_CUDSS_PINNED_STAGING", "1").lower() in {"1", "true", "yes", "on"}
_ZERO_COPY_EXPERIMENT = os.environ.get("DEVSIM_CUDSS_ZERO_COPY_EXPERIMENT", "").lower() in {"1", "true", "yes", "on"}
_AUTO_ZERO_COPY_DEVICE_EXPERIMENT = os.environ.get("DEVSIM_CUDSS_AUTO_ZERO_COPY_DEVICE_EXPERIMENT", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_AUTO_REORDERING = os.environ.get("DEVSIM_CUDSS_AUTO_REORDERING", "0").lower() in {"1", "true", "yes", "on"}
_AUTO_REORDERING_THRESHOLD = int(os.environ.get("DEVSIM_CUDSS_AUTO_REORDERING_THRESHOLD", "1024"))
_REUSE_CONTEXT = os.environ.get("DEVSIM_CUDSS_REUSE_CONTEXT", "0").lower() in {"1", "true", "yes", "on"}
_CONFIG_FALLBACK_ON_UNSUPPORTED = os.environ.get("DEVSIM_CUDSS_CONFIG_FALLBACK_ON_UNSUPPORTED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_RESIDUAL_FALLBACK_ENABLED = os.environ.get("DEVSIM_CUDSS_RESIDUAL_FALLBACK", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_RESIDUAL_FALLBACK_RATIO = float(os.environ.get("DEVSIM_CUDSS_RESIDUAL_FALLBACK_RATIO", "1e-2"))
_RESIDUAL_FALLBACK_ABS = float(os.environ.get("DEVSIM_CUDSS_RESIDUAL_FALLBACK_ABS", "1e-10"))
_VERIFY_FALLBACK_ENABLED = os.environ.get("DEVSIM_CUDSS_VERIFY_FALLBACK", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_VERIFY_FALLBACK_EARLY_SOLVE_CALLS = int(os.environ.get("DEVSIM_CUDSS_VERIFY_FALLBACK_EARLY_SOLVE_CALLS", "3"))
_VERIFY_FALLBACK_MIN_SOLVE_CALL = int(os.environ.get("DEVSIM_CUDSS_VERIFY_FALLBACK_MIN_SOLVE_CALL", "20"))
_VERIFY_FALLBACK_RHS_INF = float(os.environ.get("DEVSIM_CUDSS_VERIFY_FALLBACK_RHS_INF", "1e-12"))
_VERIFY_FALLBACK_RESIDUAL_RATIO = float(os.environ.get("DEVSIM_CUDSS_VERIFY_FALLBACK_RESIDUAL_RATIO", "1e-4"))
_VERIFY_FALLBACK_IMPROVEMENT = float(os.environ.get("DEVSIM_CUDSS_VERIFY_FALLBACK_IMPROVEMENT", "100.0"))
_CTX_CACHE: Dict[tuple[int, bool], "_CuDSSContext"] = {}
_LAST_CTX: Optional["_CuDSSContext"] = None
_UMFPACK_RUNTIME: Optional[tuple[Any, Any]] = None


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(f"[cudss-shim] {msg}")


# cudaDataType_t values (library_types.h)
CUDA_R_64I = 24
CUDA_R_64F = 1

# cudss enums used by this shim
CUDSS_STATUS_SUCCESS = 0
CUDSS_BASE_ZERO = 0
CUDSS_MTYPE_GENERAL = 0
CUDSS_MVIEW_FULL = 0
CUDSS_LAYOUT_COL_MAJOR = 0
CUDSS_CONFIG_REORDERING_ALG = 0
CUDSS_CONFIG_HYBRID_MODE = 12
CUDSS_CONFIG_HOST_NTHREADS = 15
CUDSS_CONFIG_HYBRID_EXECUTE_MODE = 16

CUDSS_PHASE_ANALYSIS = 0x1 | 0x2
CUDSS_PHASE_FACTORIZATION = 0x4
CUDSS_PHASE_REFACTORIZATION = 0x8
CUDSS_PHASE_SOLVE = 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_HOST_ALLOC_DEFAULT = 0
CUDA_HOST_ALLOC_MAPPED = 2


class _CuDSSError(RuntimeError):
    pass


def _parse_config_set_pairs() -> list[tuple[int, int]]:
    raw = os.environ.get("DEVSIM_CUDSS_CONFIG_SET", "").strip()
    if not raw:
        return []
    out: list[tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        k, v = token.split("=", 1)
        try:
            out.append((int(k.strip()), int(v.strip())))
        except ValueError:
            continue
    return out


def _named_config_pairs() -> list[tuple[int, int]]:
    mapping = (
        ("DEVSIM_CUDSS_REORDERING_ALG", CUDSS_CONFIG_REORDERING_ALG),
        ("DEVSIM_CUDSS_HYBRID_MODE", CUDSS_CONFIG_HYBRID_MODE),
        ("DEVSIM_CUDSS_HOST_NTHREADS", CUDSS_CONFIG_HOST_NTHREADS),
        ("DEVSIM_CUDSS_HYBRID_EXECUTE_MODE", CUDSS_CONFIG_HYBRID_EXECUTE_MODE),
    )
    out: list[tuple[int, int]] = []
    for env_name, param in mapping:
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        try:
            out.append((param, int(raw)))
        except ValueError:
            _dbg(f"ignore invalid integer env {env_name}={raw}")
    return out


def _auto_config_pairs(n: int, existing_params: set[int]) -> list[tuple[int, int]]:
    if not _AUTO_REORDERING:
        return []
    if CUDSS_CONFIG_REORDERING_ALG in existing_params:
        return []
    auto_value = 1 if n <= _AUTO_REORDERING_THRESHOLD else 0
    return [(CUDSS_CONFIG_REORDERING_ALG, auto_value)]


def _status_to_text(code: int) -> str:
    table = {
        0: "CUDSS_STATUS_SUCCESS",
        1: "CUDSS_STATUS_NOT_INITIALIZED",
        2: "CUDSS_STATUS_ALLOC_FAILED",
        3: "CUDSS_STATUS_INVALID_VALUE",
        4: "CUDSS_STATUS_NOT_SUPPORTED",
        5: "CUDSS_STATUS_EXECUTION_FAILED",
        6: "CUDSS_STATUS_INTERNAL_ERROR",
    }
    return table.get(code, f"CUDSS_STATUS_UNKNOWN({code})")


def _check_cudss(status: int, api: str) -> None:
    _dbg(f"{api} -> {status}")
    if status != CUDSS_STATUS_SUCCESS:
        raise _CuDSSError(f"{api} failed: {_status_to_text(status)}")


def _check_cuda(status: int, api: str) -> None:
    _dbg(f"{api} -> {status}")
    if status != 0:
        raise _CuDSSError(f"{api} failed with cuda error code {status}")


@dataclass
class _DeviceBuffers:
    d_row_start: ctypes.c_void_p
    d_col_index: ctypes.c_void_p
    d_ax: ctypes.c_void_p
    d_rhs: ctypes.c_void_p
    d_sol: ctypes.c_void_p


class _CuDSSRuntimeBindings:
    def __init__(self, cudss_lib: ctypes.CDLL):
        self.cudss = cudss_lib
        cudart_path = ctypes.util.find_library("cudart") or "libcudart.so"
        self.cudart = ctypes.CDLL(cudart_path)
        self._bind()

    def _bind(self) -> None:
        c = self.cudss
        r = self.cudart

        c.cudssCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        c.cudssCreate.restype = ctypes.c_int
        c.cudssDestroy.argtypes = [ctypes.c_void_p]
        c.cudssDestroy.restype = ctypes.c_int
        c.cudssSetThreadingLayer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        c.cudssSetThreadingLayer.restype = ctypes.c_int

        c.cudssConfigCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        c.cudssConfigCreate.restype = ctypes.c_int
        c.cudssConfigDestroy.argtypes = [ctypes.c_void_p]
        c.cudssConfigDestroy.restype = ctypes.c_int
        self.cudssConfigSet = None
        if hasattr(c, "cudssConfigSet"):
            c.cudssConfigSet.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
            c.cudssConfigSet.restype = ctypes.c_int
            self.cudssConfigSet = c.cudssConfigSet

        c.cudssDataCreate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        c.cudssDataCreate.restype = ctypes.c_int
        c.cudssDataDestroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        c.cudssDataDestroy.restype = ctypes.c_int

        c.cudssMatrixCreateCsr.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        c.cudssMatrixCreateCsr.restype = ctypes.c_int

        c.cudssMatrixCreateDn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        c.cudssMatrixCreateDn.restype = ctypes.c_int

        c.cudssMatrixDestroy.argtypes = [ctypes.c_void_p]
        c.cudssMatrixDestroy.restype = ctypes.c_int

        c.cudssMatrixSetValues.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        c.cudssMatrixSetValues.restype = ctypes.c_int
        c.cudssMatrixSetCsrPointers.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        c.cudssMatrixSetCsrPointers.restype = ctypes.c_int

        c.cudssExecute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        c.cudssExecute.restype = ctypes.c_int

        r.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        r.cudaGetDeviceCount.restype = ctypes.c_int
        r.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        r.cudaMalloc.restype = ctypes.c_int
        r.cudaFree.argtypes = [ctypes.c_void_p]
        r.cudaFree.restype = ctypes.c_int
        r.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        r.cudaMemcpy.restype = ctypes.c_int
        r.cudaDeviceSynchronize.argtypes = []
        r.cudaDeviceSynchronize.restype = ctypes.c_int
        r.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        r.cudaHostAlloc.restype = ctypes.c_int
        r.cudaFreeHost.argtypes = [ctypes.c_void_p]
        r.cudaFreeHost.restype = ctypes.c_int
        r.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        r.cudaHostGetDevicePointer.restype = ctypes.c_int
        self.cudaEventCreate = None
        self.cudaEventRecord = None
        self.cudaEventSynchronize = None
        self.cudaEventElapsedTime = None
        self.cudaEventDestroy = None
        if all(hasattr(r, name) for name in ("cudaEventCreate", "cudaEventRecord", "cudaEventSynchronize", "cudaEventElapsedTime", "cudaEventDestroy")):
            r.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            r.cudaEventCreate.restype = ctypes.c_int
            r.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            r.cudaEventRecord.restype = ctypes.c_int
            r.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
            r.cudaEventSynchronize.restype = ctypes.c_int
            r.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
            r.cudaEventElapsedTime.restype = ctypes.c_int
            r.cudaEventDestroy.argtypes = [ctypes.c_void_p]
            r.cudaEventDestroy.restype = ctypes.c_int
            self.cudaEventCreate = r.cudaEventCreate
            self.cudaEventRecord = r.cudaEventRecord
            self.cudaEventSynchronize = r.cudaEventSynchronize
            self.cudaEventElapsedTime = r.cudaEventElapsedTime
            self.cudaEventDestroy = r.cudaEventDestroy

    def supports_cuda_event_timing(self) -> bool:
        return (
            self.cudaEventCreate is not None
            and self.cudaEventRecord is not None
            and self.cudaEventSynchronize is not None
            and self.cudaEventElapsedTime is not None
            and self.cudaEventDestroy is not None
        )

    def measure_cuda_seconds(self, op: Any, fallback_label: str) -> float:
        if not self.supports_cuda_event_timing():
            t0 = time.perf_counter()
            op()
            return time.perf_counter() - t0

        start = ctypes.c_void_p()
        stop = ctypes.c_void_p()
        _check_cuda(self.cudaEventCreate(ctypes.byref(start)), "cudaEventCreate(start)")
        _check_cuda(self.cudaEventCreate(ctypes.byref(stop)), "cudaEventCreate(stop)")
        try:
            _check_cuda(self.cudaEventRecord(start, ctypes.c_void_p()), f"cudaEventRecord(start,{fallback_label})")
            op()
            _check_cuda(self.cudaEventRecord(stop, ctypes.c_void_p()), f"cudaEventRecord(stop,{fallback_label})")
            _check_cuda(self.cudaEventSynchronize(stop), f"cudaEventSynchronize({fallback_label})")
            ms = ctypes.c_float(0.0)
            _check_cuda(
                self.cudaEventElapsedTime(ctypes.byref(ms), start, stop),
                f"cudaEventElapsedTime({fallback_label})",
            )
            return float(ms.value) / 1000.0
        finally:
            if start.value:
                _check_cuda(self.cudaEventDestroy(start), "cudaEventDestroy(start)")
            if stop.value:
                _check_cuda(self.cudaEventDestroy(stop), "cudaEventDestroy(stop)")

    def ensure_gpu(self) -> None:
        cnt = ctypes.c_int(0)
        _check_cuda(self.cudart.cudaGetDeviceCount(ctypes.byref(cnt)), "cudaGetDeviceCount")
        if cnt.value <= 0:
            raise _CuDSSError("No CUDA device detected")

    def malloc_and_copy_h2d(self, host_arr: array.array, ctype: Any) -> ctypes.c_void_p:
        n = len(host_arr)
        size = n * ctypes.sizeof(ctype)
        dptr = ctypes.c_void_p()
        _check_cuda(self.cudart.cudaMalloc(ctypes.byref(dptr), size), "cudaMalloc")
        host_buf = (ctype * n).from_buffer(host_arr)
        _check_cuda(
            self.cudart.cudaMemcpy(
                dptr, ctypes.cast(host_buf, ctypes.c_void_p), size, CUDA_MEMCPY_HOST_TO_DEVICE
            ),
            "cudaMemcpy(H2D)",
        )
        return dptr

    def malloc_device(self, n: int, ctype: Any) -> ctypes.c_void_p:
        size = n * ctypes.sizeof(ctype)
        dptr = ctypes.c_void_p()
        _check_cuda(self.cudart.cudaMalloc(ctypes.byref(dptr), size), "cudaMalloc")
        return dptr

    def copy_h2d_existing(self, dptr: ctypes.c_void_p, host_arr: array.array, ctype: Any) -> None:
        n = len(host_arr)
        size = n * ctypes.sizeof(ctype)
        host_buf = (ctype * n).from_buffer(host_arr)
        _check_cuda(
            self.cudart.cudaMemcpy(
                dptr, ctypes.cast(host_buf, ctypes.c_void_p), size, CUDA_MEMCPY_HOST_TO_DEVICE
            ),
            "cudaMemcpy(H2D)",
        )

    def copy_d2h(self, dptr: ctypes.c_void_p, n: int, out: Optional[array.array] = None) -> array.array:
        if out is None:
            out = array.array("d", [0.0]) * n
        elif len(out) != n:
            raise _CuDSSError(f"Invalid output buffer length {len(out)}; expected {n}")
        host_buf = (ctypes.c_double * n).from_buffer(out)
        _check_cuda(
            self.cudart.cudaMemcpy(
                ctypes.cast(host_buf, ctypes.c_void_p),
                dptr,
                n * ctypes.sizeof(ctypes.c_double),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy(D2H)",
        )
        return out

    def free(self, ptr: Optional[ctypes.c_void_p]) -> None:
        if ptr and ptr.value:
            _check_cuda(self.cudart.cudaFree(ptr), "cudaFree")

    def alloc_pinned(self, n: int, ctype: Any, flags: int = CUDA_HOST_ALLOC_DEFAULT) -> ctypes.c_void_p:
        size = n * ctypes.sizeof(ctype)
        hptr = ctypes.c_void_p()
        _check_cuda(self.cudart.cudaHostAlloc(ctypes.byref(hptr), size, flags), "cudaHostAlloc")
        return hptr

    def host_device_pointer(self, hptr: ctypes.c_void_p) -> ctypes.c_void_p:
        dptr = ctypes.c_void_p()
        _check_cuda(
            self.cudart.cudaHostGetDevicePointer(ctypes.byref(dptr), hptr, 0),
            "cudaHostGetDevicePointer",
        )
        return dptr

    def free_pinned(self, ptr: Optional[ctypes.c_void_p]) -> None:
        if ptr and ptr.value:
            _check_cuda(self.cudart.cudaFreeHost(ptr), "cudaFreeHost")


@dataclass
class _CuDSSContext:
    n: int
    transpose: bool
    status: bool
    message: str
    bindings: Optional[_CuDSSRuntimeBindings]
    handle: ctypes.c_void_p = ctypes.c_void_p()
    config: ctypes.c_void_p = ctypes.c_void_p()
    data: ctypes.c_void_p = ctypes.c_void_p()
    mat_a: ctypes.c_void_p = ctypes.c_void_p()
    mat_rhs: ctypes.c_void_p = ctypes.c_void_p()
    mat_sol: ctypes.c_void_p = ctypes.c_void_p()
    buffers: Optional[_DeviceBuffers] = None
    symbolic_ready: bool = False
    nnz: int = 0
    host_x_cache: Optional[array.array] = None
    factor_calls: int = 0
    solve_calls: int = 0
    analysis_calls: int = 0
    refactor_calls: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    last_result_mode: str = "host"
    last_device_token: str = ""
    pinned_rhs_host: Optional[ctypes.c_void_p] = None
    pinned_sol_host: Optional[ctypes.c_void_p] = None
    pinned_sol_device: Optional[ctypes.c_void_p] = None
    sol_uses_mapped: bool = False
    init_calls: int = 0
    init_reuse_hits: int = 0
    last_zero_copy_used: bool = False
    config_set_applied: int = 0
    config_fallback_used: bool = False
    analysis_seconds: float = 0.0
    refactor_seconds: float = 0.0
    factor_total_seconds: float = 0.0
    factorization_seconds: float = 0.0
    solve_total_seconds: float = 0.0
    solve_h2d_seconds: float = 0.0
    solve_execute_seconds: float = 0.0
    solve_d2h_seconds: float = 0.0
    solve_call_breakdown: list[dict[str, float]] = None
    host_ap: Optional[array.array] = None
    host_ai: Optional[array.array] = None
    host_ax: Optional[array.array] = None
    residual_fallback_calls: int = 0
    residual_fallback_active: bool = False
    verify_fallback_calls: int = 0

    def destroy(self) -> None:
        if not self.bindings:
            return
        c = self.bindings.cudss
        if self.data:
            _check_cuda(self.bindings.cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize(destroy)")
            c.cudssDataDestroy(self.handle, self.data)
            self.data = ctypes.c_void_p()
        if self.mat_a:
            c.cudssMatrixDestroy(self.mat_a)
            self.mat_a = ctypes.c_void_p()
        if self.mat_rhs:
            c.cudssMatrixDestroy(self.mat_rhs)
            self.mat_rhs = ctypes.c_void_p()
        if self.mat_sol:
            c.cudssMatrixDestroy(self.mat_sol)
            self.mat_sol = ctypes.c_void_p()
        if self.buffers:
            self.bindings.free(self.buffers.d_row_start)
            self.bindings.free(self.buffers.d_col_index)
            self.bindings.free(self.buffers.d_ax)
            self.bindings.free(self.buffers.d_rhs)
            self.bindings.free(self.buffers.d_sol)
            self.buffers = None
        if self.pinned_rhs_host:
            self.bindings.free_pinned(self.pinned_rhs_host)
            self.pinned_rhs_host = None
        if self.pinned_sol_host:
            self.bindings.free_pinned(self.pinned_sol_host)
            self.pinned_sol_host = None
            self.pinned_sol_device = None
        if self.config:
            c.cudssConfigDestroy(self.config)
            self.config = ctypes.c_void_p()
        if self.handle:
            c.cudssDestroy(self.handle)
            self.handle = ctypes.c_void_p()
        self.symbolic_ready = False
        self.nnz = 0
        self.sol_uses_mapped = False
        self.config_set_applied = 0
        self.config_fallback_used = False

    def reset_stats(self) -> None:
        self.factor_calls = 0
        self.solve_calls = 0
        self.analysis_calls = 0
        self.refactor_calls = 0
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.last_result_mode = "host"
        self.last_device_token = ""
        self.last_zero_copy_used = False
        self.analysis_seconds = 0.0
        self.refactor_seconds = 0.0
        self.factor_total_seconds = 0.0
        self.factorization_seconds = 0.0
        self.solve_total_seconds = 0.0
        self.solve_h2d_seconds = 0.0
        self.solve_execute_seconds = 0.0
        self.solve_d2h_seconds = 0.0
        self.solve_call_breakdown = []
        self.residual_fallback_calls = 0
        self.residual_fallback_active = False
        self.verify_fallback_calls = 0

    def __del__(self) -> None:
        # Avoid invoking CUDA/cuDSS teardown during Python interpreter shutdown,
        # which may unload dependent shared libraries in undefined order.
        # Explicit destroy can be added later via a dedicated callback action.
        return


def _set_status(ctx: _CuDSSContext, status: bool, message: str) -> None:
    ctx.status = status
    ctx.message = message


def get_last_stats() -> dict[str, object]:
    ctx = _LAST_CTX
    if ctx is None:
        return {}
    breakdown = ctx.solve_call_breakdown or []
    return {
        "analysis_seconds": ctx.analysis_seconds,
        "refactor_seconds": ctx.refactor_seconds,
        "factor_total_seconds": ctx.factor_total_seconds,
        "factorization_seconds": ctx.factorization_seconds,
        "solve_total_seconds": ctx.solve_total_seconds,
        "solve_h2d_seconds": ctx.solve_h2d_seconds,
        "solve_execute_seconds": ctx.solve_execute_seconds,
        "solve_d2h_seconds": ctx.solve_d2h_seconds,
        "solve_call_breakdown": breakdown,
        "residual_fallback_calls": ctx.residual_fallback_calls,
        "verify_fallback_calls": ctx.verify_fallback_calls,
    }


def _csr_residual_inf(ap: array.array, ai: array.array, ax: array.array, x: array.array, b: array.array) -> float:
    vmax = 0.0
    for row in range(len(ap) - 1):
        accum = 0.0
        for idx in range(ap[row], ap[row + 1]):
            accum += ax[idx] * x[ai[idx]]
        diff = abs(accum - b[row])
        if diff > vmax:
            vmax = diff
    return vmax


def _csr_to_csc(n: int, ap: array.array, ai: array.array, ax: array.array) -> tuple[array.array, array.array, array.array]:
    counts = [0] * n
    for col in ai:
        counts[int(col)] += 1
    csc_ap_list = [0] * (n + 1)
    for i in range(n):
        csc_ap_list[i + 1] = csc_ap_list[i] + counts[i]
    csc_ai = array.array(ai.typecode, [0]) * len(ai)
    csc_ax = array.array("d", [0.0]) * len(ax)
    next_pos = csc_ap_list[:-1].copy()
    for row in range(n):
        for idx in range(ap[row], ap[row + 1]):
            col = int(ai[idx])
            pos = next_pos[col]
            csc_ai[pos] = row
            csc_ax[pos] = ax[idx]
            next_pos[col] += 1
    csc_ap = array.array(ap.typecode, csc_ap_list)
    return csc_ap, csc_ai, csc_ax


def _get_umfpack_runtime() -> tuple[Any, Any]:
    global _UMFPACK_RUNTIME
    if _UMFPACK_RUNTIME is not None:
        return _UMFPACK_RUNTIME

    import devsim
    from umfpack import umfpack_loader as umf

    gdata = umf.global_data()
    umf_name = os.path.basename(umf.get_umfpack_name())
    umf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "umfpack", umf_name)
    gdata.dll = umf.load_umfpack_dll(umf_path)
    mcount = 1
    for blaslib in devsim.get_parameter(name="info")["math_libraries"]:
        h = umf.load_blas_dll(gdata, blaslib=blaslib, noexcept=True)
        if h:
            mcount = umf.load_blas_functions(gdata, h)
            if mcount == 0:
                break
    if mcount != 0:
        raise RuntimeError(f"Missing {mcount} UMFPACK math functions")
    _UMFPACK_RUNTIME = (umf, gdata)
    return _UMFPACK_RUNTIME


def _solve_with_umfpack_fallback(ctx: _CuDSSContext, bvec: array.array) -> array.array:
    if ctx.host_ap is None or ctx.host_ai is None or ctx.host_ax is None:
        raise _CuDSSError("UMFPACK fallback requested without cached matrix")
    umf, gdata = _get_umfpack_runtime()
    csc_ap, csc_ai, csc_ax = _csr_to_csc(ctx.n, ctx.host_ap, ctx.host_ai, ctx.host_ax)
    uc = umf.umf_control(gdata, "real")
    matrix = umf.matrix(uc=uc, Ap=csc_ap, Ai=csc_ai, Ax=csc_ax)
    symbolic = uc.symbolic(matrix=matrix)
    numeric = uc.numeric(matrix=matrix, Symbolic=symbolic)
    x = array.array("d", bvec)
    uc.solve(matrix=matrix, Numeric=numeric, b=bvec, transpose=ctx.transpose, x=x)
    return x


def _should_run_verify_fallback(solve_call_index: int, rhs_inf: float, residual_inf: float) -> bool:
    if solve_call_index <= _VERIFY_FALLBACK_EARLY_SOLVE_CALLS:
        return True
    if solve_call_index < _VERIFY_FALLBACK_MIN_SOLVE_CALL:
        return False
    if rhs_inf <= _VERIFY_FALLBACK_RHS_INF:
        return True
    if rhs_inf == 0.0:
        return residual_inf > 0.0
    return (residual_inf / rhs_inf) >= _VERIFY_FALLBACK_RESIDUAL_RATIO


def _initialize_ctx_runtime(ctx: _CuDSSContext) -> None:
    if not ctx.bindings:
        raise _CuDSSError(get_unavailable_message())
    c = ctx.bindings.cudss
    _check_cudss(c.cudssCreate(ctypes.byref(ctx.handle)), "cudssCreate")
    # Configure CPU threading helper library when available (required by some runtimes).
    thr_layer = None
    for root in site.getsitepackages():
        candidate = os.path.join(root, "nvidia", "cu12", "lib", "libcudss_mtlayer_gomp.so.0")
        if os.path.isfile(candidate):
            thr_layer = candidate
            break
    if thr_layer:
        _check_cudss(
            c.cudssSetThreadingLayer(ctx.handle, thr_layer.encode("utf-8")),
            "cudssSetThreadingLayer",
        )
    _check_cudss(c.cudssConfigCreate(ctypes.byref(ctx.config)), "cudssConfigCreate")
    if ctx.bindings.cudssConfigSet and ctx.config:
        explicit_pairs = [*_named_config_pairs(), *_parse_config_set_pairs()]
        explicit_params = {param for param, _ in explicit_pairs}
        all_pairs = [*explicit_pairs, *_auto_config_pairs(ctx.n, explicit_params)]
        for param, value in all_pairs:
            v = ctypes.c_int(value)
            status = ctx.bindings.cudssConfigSet(
                ctx.config,
                ctypes.c_int(param),
                ctypes.byref(v),
                ctypes.sizeof(v),
            )
            if status == CUDSS_STATUS_SUCCESS:
                ctx.config_set_applied += 1
            else:
                _dbg(f"cudssConfigSet failed param={param} value={value} status={status}")
    _check_cudss(c.cudssDataCreate(ctx.handle, ctypes.byref(ctx.data)), "cudssDataCreate")
    ctx.symbolic_ready = False
    ctx.nnz = 0
    ctx.sol_uses_mapped = False


def _make_ctx(n: int, transpose: bool) -> _CuDSSContext:
    cache_key = (n, transpose)
    if _REUSE_CONTEXT:
        cached = _CTX_CACHE.get(cache_key)
        if cached is not None and cached.bindings is not None and cached.status:
            cached.init_calls += 1
            cached.init_reuse_hits += 1
            cached.reset_stats()
            _set_status(cached, True, "cuDSS runtime context reused")
            return cached

    runtime = detect_cudss_runtime()
    if runtime.library is None:
        return _CuDSSContext(
            n=n,
            transpose=transpose,
            status=False,
            message=get_unavailable_message(),
            bindings=None,
        )
    bindings = _CuDSSRuntimeBindings(runtime.library)
    bindings.ensure_gpu()

    ctx = _CuDSSContext(
        n=n,
        transpose=transpose,
        status=True,
        message="cuDSS runtime detected",
        bindings=bindings,
    )
    if _ZERO_COPY_EXPERIMENT or _AUTO_ZERO_COPY_DEVICE_EXPERIMENT:
        ctx.message = "cuDSS runtime detected; device_experimental result prefers zero-copy path"
    ctx.init_calls = 1
    ctx.solve_call_breakdown = []
    _initialize_ctx_runtime(ctx)
    if _REUSE_CONTEXT:
        _CTX_CACHE[cache_key] = ctx
    return ctx


def _reset_config_to_default(ctx: _CuDSSContext) -> None:
    if not ctx.bindings:
        return
    c = ctx.bindings.cudss
    if ctx.config:
        _check_cudss(c.cudssConfigDestroy(ctx.config), "cudssConfigDestroy")
    ctx.config = ctypes.c_void_p()
    _check_cudss(c.cudssConfigCreate(ctypes.byref(ctx.config)), "cudssConfigCreate")
    ctx.config_set_applied = 0
    ctx.config_fallback_used = True


def _factor(ctx: _CuDSSContext, kwargs: Dict[str, Any]) -> None:
    if not ctx.bindings:
        raise _CuDSSError(get_unavailable_message())

    if bool(kwargs["is_complex"]):
        raise _CuDSSError("Phase-1 shim supports real matrices only")

    ax = kwargs["Ax"]
    is_same_symbolic = bool(kwargs["is_same_symbolic"])

    ap = None
    ai = None
    first_factor = ctx.buffers is None
    needs_pattern_upload = first_factor or (not is_same_symbolic)
    if needs_pattern_upload:
        if "Ap" not in kwargs or "Ai" not in kwargs:
            raise _CuDSSError("Ap/Ai are required when symbolic pattern is new")
        ap = kwargs["Ap"]
        ai = kwargs["Ai"]
        if len(ap) != ctx.n + 1:
            raise _CuDSSError(f"Invalid Ap length {len(ap)}; expected n+1={ctx.n + 1}")
        nnz = len(ai)
        if len(ax) != nnz:
            raise _CuDSSError("Ai/Ax length mismatch")
    else:
        nnz = ctx.nnz
        if len(ax) != nnz:
            raise _CuDSSError(f"Ax length mismatch for cached symbolic pattern: got {len(ax)}, expected {nnz}")

    if ap is not None:
        ctx.host_ap = array.array(ap.typecode, ap)
    if ai is not None:
        ctx.host_ai = array.array(ai.typecode, ai)
    ctx.host_ax = array.array("d", ax)

    b = ctx.bindings
    c = b.cudss
    ctx.factor_calls += 1

    rebuild_pattern = (not first_factor) and (not is_same_symbolic)
    if rebuild_pattern:
        if ctx.mat_a:
            _check_cudss(c.cudssMatrixDestroy(ctx.mat_a), "cudssMatrixDestroy(A)")
            ctx.mat_a = ctypes.c_void_p()
        if ctx.mat_rhs:
            _check_cudss(c.cudssMatrixDestroy(ctx.mat_rhs), "cudssMatrixDestroy(B)")
            ctx.mat_rhs = ctypes.c_void_p()
        if ctx.mat_sol:
            _check_cudss(c.cudssMatrixDestroy(ctx.mat_sol), "cudssMatrixDestroy(X)")
            ctx.mat_sol = ctypes.c_void_p()
        if ctx.buffers:
            b.free(ctx.buffers.d_row_start)
            b.free(ctx.buffers.d_col_index)
            b.free(ctx.buffers.d_ax)
            b.free(ctx.buffers.d_rhs)
            b.free(ctx.buffers.d_sol)
            ctx.buffers = None

    factor_begin = time.perf_counter()
    if first_factor or rebuild_pattern:
        # Structural buffers only need to be created/uploaded once while symbolic
        # structure remains unchanged.
        row_start = array.array("q", ap)
        col_index = array.array("q", ai)
        d_row_start = b.malloc_and_copy_h2d(row_start, ctypes.c_int64)
        d_col_index = b.malloc_and_copy_h2d(col_index, ctypes.c_int64)
        d_ax = b.malloc_and_copy_h2d(ax, ctypes.c_double)
        ctx.h2d_bytes += len(row_start) * ctypes.sizeof(ctypes.c_int64)
        ctx.h2d_bytes += len(col_index) * ctypes.sizeof(ctypes.c_int64)
        ctx.h2d_bytes += len(ax) * ctypes.sizeof(ctypes.c_double)
        # RHS/X are always written before read; avoid one-time zero-filled H2D copies.
        d_rhs = b.malloc_device(ctx.n, ctypes.c_double)
        d_sol = b.malloc_device(ctx.n, ctypes.c_double)
        ctx.buffers = _DeviceBuffers(d_row_start, d_col_index, d_ax, d_rhs, d_sol)
        ctx.sol_uses_mapped = False
        ctx.nnz = nnz

        _check_cudss(
            c.cudssMatrixCreateCsr(
                ctypes.byref(ctx.mat_a),
                ctypes.c_int64(ctx.n),
                ctypes.c_int64(ctx.n),
                ctypes.c_int64(nnz),
                d_row_start,
                ctypes.c_void_p(),
                d_col_index,
                d_ax,
                ctypes.c_int(CUDA_R_64I),
                ctypes.c_int(CUDA_R_64F),
                ctypes.c_int(CUDSS_MTYPE_GENERAL),
                ctypes.c_int(CUDSS_MVIEW_FULL),
                ctypes.c_int(CUDSS_BASE_ZERO),
            ),
            "cudssMatrixCreateCsr(A)",
        )
        _check_cudss(
            c.cudssMatrixCreateDn(
                ctypes.byref(ctx.mat_rhs),
                ctypes.c_int64(ctx.n),
                ctypes.c_int64(1),
                ctypes.c_int64(ctx.n),
                d_rhs,
                ctypes.c_int(CUDA_R_64F),
                ctypes.c_int(CUDSS_LAYOUT_COL_MAJOR),
            ),
            "cudssMatrixCreateDn(B)",
        )
        _check_cudss(
            c.cudssMatrixCreateDn(
                ctypes.byref(ctx.mat_sol),
                ctypes.c_int64(ctx.n),
                ctypes.c_int64(1),
                ctypes.c_int64(ctx.n),
                d_sol,
                ctypes.c_int(CUDA_R_64F),
                ctypes.c_int(CUDSS_LAYOUT_COL_MAJOR),
            ),
            "cudssMatrixCreateDn(X)",
        )
    else:
        if nnz != ctx.nnz:
            raise _CuDSSError("nnz changed; this shim currently requires fixed pattern across refactorizations")
        # Pointer is unchanged; only update numeric values in-place.
        b.copy_h2d_existing(ctx.buffers.d_ax, ax, ctypes.c_double)
        ctx.h2d_bytes += len(ax) * ctypes.sizeof(ctypes.c_double)

    if (not ctx.symbolic_ready) or (not is_same_symbolic):
        ctx.analysis_calls += 1
        try:
            analysis_seconds = b.measure_cuda_seconds(
                lambda: _check_cudss(
                    c.cudssExecute(
                        ctx.handle, CUDSS_PHASE_ANALYSIS, ctx.config, ctx.data, ctx.mat_a, ctx.mat_sol, ctx.mat_rhs
                    ),
                    "cudssExecute(ANALYSIS)",
                ),
                "ANALYSIS",
            )
            factorization_seconds = b.measure_cuda_seconds(
                lambda: _check_cudss(
                    c.cudssExecute(
                        ctx.handle, CUDSS_PHASE_FACTORIZATION, ctx.config, ctx.data, ctx.mat_a, ctx.mat_sol, ctx.mat_rhs
                    ),
                    "cudssExecute(FACTORIZATION)",
                ),
                "FACTORIZATION",
            )
        except _CuDSSError as exc:
            if (
                _CONFIG_FALLBACK_ON_UNSUPPORTED
                and (not ctx.config_fallback_used)
                and ("CUDSS_STATUS_NOT_SUPPORTED" in str(exc))
                and (ctx.config_set_applied > 0)
            ):
                _dbg("analysis/factorization NOT_SUPPORTED with tuned config; fallback to default config and retry")
                _reset_config_to_default(ctx)
                analysis_seconds = b.measure_cuda_seconds(
                    lambda: _check_cudss(
                        c.cudssExecute(
                            ctx.handle, CUDSS_PHASE_ANALYSIS, ctx.config, ctx.data, ctx.mat_a, ctx.mat_sol, ctx.mat_rhs
                        ),
                        "cudssExecute(ANALYSIS,retry-default-config)",
                    ),
                    "ANALYSIS(retry)",
                )
                factorization_seconds = b.measure_cuda_seconds(
                    lambda: _check_cudss(
                        c.cudssExecute(
                            ctx.handle, CUDSS_PHASE_FACTORIZATION, ctx.config, ctx.data, ctx.mat_a, ctx.mat_sol, ctx.mat_rhs
                        ),
                        "cudssExecute(FACTORIZATION,retry-default-config)",
                    ),
                    "FACTORIZATION(retry)",
                )
            else:
                raise
        ctx.analysis_seconds += analysis_seconds
        ctx.factorization_seconds += factorization_seconds
        ctx.symbolic_ready = True
    else:
        ctx.refactor_calls += 1
        try:
            refactor_seconds = b.measure_cuda_seconds(
                lambda: _check_cudss(
                    c.cudssExecute(
                        ctx.handle,
                        CUDSS_PHASE_REFACTORIZATION,
                        ctx.config,
                        ctx.data,
                        ctx.mat_a,
                        ctx.mat_sol,
                        ctx.mat_rhs,
                    ),
                    "cudssExecute(REFACTORIZATION)",
                ),
                "REFACTORIZATION",
            )
        except _CuDSSError as exc:
            if (
                _CONFIG_FALLBACK_ON_UNSUPPORTED
                and (not ctx.config_fallback_used)
                and ("CUDSS_STATUS_NOT_SUPPORTED" in str(exc))
                and (ctx.config_set_applied > 0)
            ):
                _dbg("refactorization NOT_SUPPORTED with tuned config; fallback to default config and retry")
                _reset_config_to_default(ctx)
                refactor_seconds = b.measure_cuda_seconds(
                    lambda: _check_cudss(
                        c.cudssExecute(
                            ctx.handle,
                            CUDSS_PHASE_REFACTORIZATION,
                            ctx.config,
                            ctx.data,
                            ctx.mat_a,
                            ctx.mat_sol,
                            ctx.mat_rhs,
                        ),
                        "cudssExecute(REFACTORIZATION,retry-default-config)",
                    ),
                    "REFACTORIZATION(retry)",
                )
            else:
                raise
        ctx.refactor_seconds += refactor_seconds

    ctx.factor_total_seconds += time.perf_counter() - factor_begin
    _set_status(ctx, True, "")


def _solve(ctx: _CuDSSContext, kwargs: Dict[str, Any]) -> array.array:
    if not ctx.bindings:
        raise _CuDSSError(get_unavailable_message())
    if bool(kwargs["is_complex"]):
        raise _CuDSSError("Phase-1 shim supports real RHS only")
    if not ctx.symbolic_ready or not ctx.buffers:
        raise _CuDSSError("solve called before successful factor")

    bvec = kwargs["b"]
    if not isinstance(bvec, array.array):
        raise _CuDSSError("Unexpected RHS payload type; expected array('d')")
    if len(bvec) != ctx.n:
        raise _CuDSSError(f"Invalid RHS length {len(bvec)}; expected {ctx.n}")
    solve_call_index = int(kwargs.get("solve_call_index", ctx.solve_calls + 1))
    if (
        _RESIDUAL_FALLBACK_ENABLED
        and ctx.residual_fallback_active
        and ctx.host_ap is not None
        and ctx.host_ai is not None
        and ctx.host_ax is not None
    ):
        ctx.solve_calls += 1
        fallback_x = _solve_with_umfpack_fallback(ctx, bvec)
        ctx.residual_fallback_calls += 1
        if ctx.host_x_cache is None:
            ctx.host_x_cache = array.array("d", [0.0]) * ctx.n
        ctx.host_x_cache[:] = fallback_x
        return ctx.host_x_cache

    b = ctx.bindings
    c = b.cudss
    ctx.solve_calls += 1
    solve_begin = time.perf_counter()
    h2d_seconds = 0.0
    execute_seconds = 0.0
    d2h_seconds = 0.0
    result_mode = str(kwargs.get("result_mode", "host"))
    require_host_x = bool(kwargs.get("require_host_x", True))
    if result_mode not in ("host", "device_experimental"):
        raise _CuDSSError(f"Unsupported result_mode: {result_mode}")
    ctx.last_result_mode = result_mode
    use_zero_copy_result = _ZERO_COPY_EXPERIMENT or (
        _AUTO_ZERO_COPY_DEVICE_EXPERIMENT and result_mode == "device_experimental"
    )
    ctx.last_zero_copy_used = use_zero_copy_result
    ctx.last_device_token = f"ctx-{id(ctx)}-solve-{ctx.solve_calls}"
    if _USE_PINNED_STAGING:
        if ctx.pinned_rhs_host is None:
            ctx.pinned_rhs_host = b.alloc_pinned(ctx.n, ctypes.c_double)
        host_rhs = (ctypes.c_double * ctx.n).from_buffer(bvec)
        size = ctx.n * ctypes.sizeof(ctypes.c_double)
        ctypes.memmove(ctx.pinned_rhs_host.value, ctypes.cast(host_rhs, ctypes.c_void_p).value, size)
        h2d_seconds += b.measure_cuda_seconds(
            lambda: _check_cuda(
                b.cudart.cudaMemcpy(
                    ctx.buffers.d_rhs,
                    ctx.pinned_rhs_host,
                    size,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                "cudaMemcpy(H2D,pinned)",
            ),
            "H2D(pinned)",
        )
    else:
        h2d_seconds += b.measure_cuda_seconds(
            lambda: b.copy_h2d_existing(ctx.buffers.d_rhs, bvec, ctypes.c_double),
            "H2D",
        )
    ctx.h2d_bytes += len(bvec) * ctypes.sizeof(ctypes.c_double)

    if use_zero_copy_result:
        if ctx.pinned_sol_host is None:
            ctx.pinned_sol_host = b.alloc_pinned(ctx.n, ctypes.c_double, CUDA_HOST_ALLOC_MAPPED)
        if ctx.pinned_sol_device is None:
            ctx.pinned_sol_device = b.host_device_pointer(ctx.pinned_sol_host)
        if not ctx.sol_uses_mapped:
            _check_cudss(
                c.cudssMatrixSetValues(ctx.mat_sol, ctx.pinned_sol_device),
                "cudssMatrixSetValues(X,mapped)",
            )
            ctx.sol_uses_mapped = True
    elif ctx.sol_uses_mapped:
        _check_cudss(
            c.cudssMatrixSetValues(ctx.mat_sol, ctx.buffers.d_sol),
            "cudssMatrixSetValues(X,device)",
        )
        ctx.sol_uses_mapped = False

    execute_seconds += b.measure_cuda_seconds(
        lambda: _check_cudss(
            c.cudssExecute(ctx.handle, CUDSS_PHASE_SOLVE, ctx.config, ctx.data, ctx.mat_a, ctx.mat_sol, ctx.mat_rhs),
            "cudssExecute(SOLVE)",
        ),
        "SOLVE",
    )
    if not require_host_x and result_mode == "device_experimental":
        solve_elapsed = time.perf_counter() - solve_begin
        ctx.solve_total_seconds += solve_elapsed
        ctx.solve_h2d_seconds += h2d_seconds
        ctx.solve_execute_seconds += execute_seconds
        ctx.solve_d2h_seconds += d2h_seconds
        if ctx.solve_call_breakdown is None:
            ctx.solve_call_breakdown = []
        ctx.solve_call_breakdown.append(
            {
                "call": float(ctx.solve_calls),
                "total_seconds": solve_elapsed,
                "h2d_seconds": h2d_seconds,
                "execute_seconds": execute_seconds,
                "d2h_seconds": d2h_seconds,
                "other_seconds": max(0.0, solve_elapsed - h2d_seconds - execute_seconds - d2h_seconds),
            }
        )
        return array.array("d")

    if ctx.host_x_cache is None:
        ctx.host_x_cache = array.array("d", [0.0]) * ctx.n
    if use_zero_copy_result:
        # Zero-copy path needs an explicit sync before the host reads mapped memory.
        t0 = time.perf_counter()
        _check_cuda(b.cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        if ctx.pinned_sol_host is None:
            raise _CuDSSError("zero-copy result path requires a mapped host solution buffer")
        size = ctx.n * ctypes.sizeof(ctypes.c_double)
        host_x = (ctypes.c_double * ctx.n).from_buffer(ctx.host_x_cache)
        ctypes.memmove(ctypes.cast(host_x, ctypes.c_void_p).value, ctx.pinned_sol_host.value, size)
        d2h_seconds += time.perf_counter() - t0
    elif _USE_PINNED_STAGING:
        if ctx.pinned_sol_host is None:
            ctx.pinned_sol_host = b.alloc_pinned(ctx.n, ctypes.c_double)
        size = ctx.n * ctypes.sizeof(ctypes.c_double)
        d2h_seconds += b.measure_cuda_seconds(
            lambda: _check_cuda(
                b.cudart.cudaMemcpy(
                    ctx.pinned_sol_host,
                    ctx.buffers.d_sol,
                    size,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                ),
                "cudaMemcpy(D2H,pinned)",
            ),
            "D2H(pinned)",
        )
        host_x = (ctypes.c_double * ctx.n).from_buffer(ctx.host_x_cache)
        ctypes.memmove(ctypes.cast(host_x, ctypes.c_void_p).value, ctx.pinned_sol_host.value, size)
    else:
        d2h_seconds += b.measure_cuda_seconds(
            lambda: b.copy_d2h(ctx.buffers.d_sol, ctx.n, out=ctx.host_x_cache),
            "D2H",
        )
    if (
        _RESIDUAL_FALLBACK_ENABLED
        and ctx.host_ap is not None
        and ctx.host_ai is not None
        and ctx.host_ax is not None
    ):
        residual_inf = _csr_residual_inf(ctx.host_ap, ctx.host_ai, ctx.host_ax, ctx.host_x_cache, bvec)
        rhs_inf = max((abs(v) for v in bvec), default=0.0)
        residual_limit = max(_RESIDUAL_FALLBACK_ABS, _RESIDUAL_FALLBACK_RATIO * rhs_inf)
        if residual_inf > residual_limit:
            try:
                fallback_x = _solve_with_umfpack_fallback(ctx, bvec)
            except Exception as exc:
                _dbg(f"UMFPACK residual fallback unavailable: {exc}")
            else:
                ctx.host_x_cache[:] = fallback_x
                ctx.residual_fallback_calls += 1
                ctx.residual_fallback_active = True
        elif _VERIFY_FALLBACK_ENABLED and _should_run_verify_fallback(solve_call_index, rhs_inf, residual_inf):
            try:
                fallback_x = _solve_with_umfpack_fallback(ctx, bvec)
                fallback_residual_inf = _csr_residual_inf(ctx.host_ap, ctx.host_ai, ctx.host_ax, fallback_x, bvec)
            except Exception as exc:
                _dbg(f"UMFPACK verify fallback unavailable: {exc}")
            else:
                improvement = residual_inf / max(fallback_residual_inf, 1.0e-300)
                if improvement >= _VERIFY_FALLBACK_IMPROVEMENT:
                    ctx.host_x_cache[:] = fallback_x
                    ctx.verify_fallback_calls += 1
                    ctx.residual_fallback_active = True
    if not use_zero_copy_result:
        ctx.d2h_bytes += ctx.n * ctypes.sizeof(ctypes.c_double)
    solve_elapsed = time.perf_counter() - solve_begin
    ctx.solve_total_seconds += solve_elapsed
    ctx.solve_h2d_seconds += h2d_seconds
    ctx.solve_execute_seconds += execute_seconds
    ctx.solve_d2h_seconds += d2h_seconds
    if ctx.solve_call_breakdown is None:
        ctx.solve_call_breakdown = []
    ctx.solve_call_breakdown.append(
        {
            "call": float(ctx.solve_calls),
            "total_seconds": solve_elapsed,
            "h2d_seconds": h2d_seconds,
            "execute_seconds": execute_seconds,
            "d2h_seconds": d2h_seconds,
            "other_seconds": max(0.0, solve_elapsed - h2d_seconds - execute_seconds - d2h_seconds),
        }
    )
    return ctx.host_x_cache


def local_solver_callback(**kwargs: Any) -> Dict[str, Any]:
    global _LAST_CTX
    action = kwargs["action"]

    if action == "init":
        try:
            ctx = _make_ctx(int(kwargs["n"]), bool(kwargs["transpose"]))
            _LAST_CTX = ctx
        except Exception as exc:
            ctx = _CuDSSContext(
                n=int(kwargs["n"]),
                transpose=bool(kwargs["transpose"]),
                status=False,
                message=str(exc),
                bindings=None,
            )
        return {
            "solver_object": ctx,
            "matrix_format": "csr",
            "status": ctx.status,
            "message": ctx.message,
        }

    if action == "factor":
        ctx = kwargs["solver_object"]
        _LAST_CTX = ctx
        try:
            _factor(ctx, kwargs)
        except Exception as exc:
            _set_status(ctx, False, str(exc))
        return {"status": ctx.status, "message": ctx.message}

    if action == "solve":
        ctx = kwargs["solver_object"]
        _LAST_CTX = ctx
        x = array.array("d")
        try:
            x = _solve(ctx, kwargs)
            _set_status(ctx, True, "")
        except Exception as exc:
            _set_status(ctx, False, str(exc))
        return {
            "status": ctx.status,
            "message": ctx.message,
            "x": x,
            "x_location": (
                "host"
                if ctx.last_result_mode == "host"
                else "device_experimental_zero_copy"
                if ctx.last_zero_copy_used
                else "device_experimental_host_fallback"
            ),
            "x_device_token": ctx.last_device_token,
        }

    if action == "gather_rows":
        ctx = kwargs["solver_object"]
        _LAST_CTX = ctx
        rows = kwargs.get("rows")
        if not isinstance(rows, array.array):
            raise _CuDSSError("gather_rows expects rows as integer array")
        if not ctx.bindings or not ctx.buffers:
            raise _CuDSSError("gather_rows called before successful factor/solve")
        b = ctx.bindings
        out = array.array("d", [0.0]) * len(rows)
        row_ints = [int(r) for r in rows]
        for row in row_ints:
            if row < 0 or row >= ctx.n:
                raise _CuDSSError(f"gather_rows index out of range: {row}")

        i = 0
        while i < len(row_ints):
            start_row = row_ints[i]
            start_i = i
            while i + 1 < len(row_ints) and row_ints[i + 1] == row_ints[i] + 1:
                i += 1
            end_i = i
            count = end_i - start_i + 1
            src = ctypes.c_void_p(ctx.buffers.d_sol.value + start_row * ctypes.sizeof(ctypes.c_double))
            dst = (ctypes.c_double * count).from_buffer(out, start_i * ctypes.sizeof(ctypes.c_double))
            _check_cuda(
                b.cudart.cudaMemcpy(
                    ctypes.cast(dst, ctypes.c_void_p),
                    src,
                    count * ctypes.sizeof(ctypes.c_double),
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                ),
                "cudaMemcpy(D2H,gather_rows)",
            )
            i += 1
        ctx.d2h_bytes += len(row_ints) * ctypes.sizeof(ctypes.c_double)
        return {
            "status": True,
            "message": "",
            "values": out,
        }

    if action == "stats":
        ctx = kwargs["solver_object"]
        return {
            "status": ctx.status,
            "message": ctx.message,
            "factor_calls": ctx.factor_calls,
            "solve_calls": ctx.solve_calls,
            "analysis_calls": ctx.analysis_calls,
            "refactor_calls": ctx.refactor_calls,
            "h2d_bytes": ctx.h2d_bytes,
            "d2h_bytes": ctx.d2h_bytes,
            "last_result_mode": ctx.last_result_mode,
            "last_device_token": ctx.last_device_token,
            "pinned_staging_enabled": _USE_PINNED_STAGING,
            "zero_copy_experiment_requested": _ZERO_COPY_EXPERIMENT,
            "zero_copy_result_enabled": ctx.last_zero_copy_used,
            "auto_zero_copy_device_experimental": _AUTO_ZERO_COPY_DEVICE_EXPERIMENT,
            "reuse_context_enabled": _REUSE_CONTEXT,
            "init_calls": ctx.init_calls,
            "init_reuse_hits": ctx.init_reuse_hits,
            "config_set_applied": ctx.config_set_applied,
            "auto_reordering_enabled": _AUTO_REORDERING,
            "auto_reordering_threshold": _AUTO_REORDERING_THRESHOLD,
            "config_fallback_on_unsupported": _CONFIG_FALLBACK_ON_UNSUPPORTED,
            "config_fallback_used": ctx.config_fallback_used,
            "analysis_seconds": ctx.analysis_seconds,
            "refactor_seconds": ctx.refactor_seconds,
            "factor_total_seconds": ctx.factor_total_seconds,
            "factorization_seconds": ctx.factorization_seconds,
            "solve_total_seconds": ctx.solve_total_seconds,
            "solve_h2d_seconds": ctx.solve_h2d_seconds,
            "solve_execute_seconds": ctx.solve_execute_seconds,
            "solve_d2h_seconds": ctx.solve_d2h_seconds,
        }

    raise RuntimeError(f"Unsupported action: {action}")
