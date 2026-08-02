"""Thread-env limits — import this FIRST in every entrypoint to keep CPU usage sane.

The SSH server (x640) has 32 cores. Without limits, numpy BLAS / scanpy /
torch CPU ops will spawn 32 threads each, causing the 3100% CPU load the user
saw. We cap to 8 threads (still fast, leaves headroom for other users).

Usage at the very top of any entrypoint:
    from perturbgnn._thread_limits import limit_threads
    limit_threads()

For tight per-call control (the only fully reliable way to tame scanpy/sklearn):
    from perturbgnn._thread_limits import threadpool_ctx
    with threadpool_ctx(4):
        sc.tl.pca(...)  # cannot escape the cap
"""
from __future__ import annotations
import os
from contextlib import contextmanager


def limit_threads(max_threads: int = 8):
    """Cap BLAS / OpenMP / torch CPU threads via env vars. Call BEFORE importing numpy/torch."""
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    os.environ["PYTORCH_CPU_ALLOC_CONF"] = "max_split_size_mb:256"


@contextmanager
def threadpool_ctx(max_threads: int = 4):
    """Hard-cap BLAS threads inside a `with` block via threadpoolctl.

    This catches OpenMP / MKL / OpenBLAS even when the env-var approach fails
    (e.g. scanpy.pp.scale internally re-spawns workers). Falls back to no-op
    if threadpoolctl isn't installed.
    """
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=max_threads):
            yield
    except ImportError:
        yield

