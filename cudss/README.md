# cuDSS shim for DEVSIM custom direct solver

This directory contains a Phase-1 integration scaffold for using NVIDIA cuDSS
through DEVSIM's `direct_solver=custom` callback protocol.

Current scope:

- Linux
- NVIDIA GPU
- DC real-valued matrix path

Usage in a simulation script:

```python
import devsim
from devsim.cudss import local_solver_callback

devsim.set_parameter(name="direct_solver", value="custom")
devsim.set_parameter(name="solver_callback", value=local_solver_callback)
```

Optional environment variable:

- `DEVSIM_CUDSS_LIB=/path/to/libcudss.so`
- `DEVSIM_CUDSS_DEBUG=1` (prints cuDSS/CUDA API call status for tracing)

Notes:

- The shim now calls cuDSS C API via `ctypes` for real DC path.
- Matrix format returned to DEVSIM is `csr` (required by `cudssMatrixCreateCsr`).
- If cuDSS runtime/GPU is unavailable, the callback returns explicit errors
  and does not silently fall back.
