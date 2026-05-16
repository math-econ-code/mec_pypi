"""ctypes bindings for libdw (the C RROA solver).

The shared library is built by `make lib` in this directory. Loading is
lazy so importing mec.dw never fails on a machine without the C build.
"""

from __future__ import annotations

import ctypes
import os
import sys
import numpy as np
from numpy.ctypeslib import ndpointer


_LIB = None


def _shlib_name() -> str:
    if sys.platform == "darwin":   return "libdw.dylib"
    if sys.platform.startswith("linux"): return "libdw.so"
    if sys.platform == "win32":    return "dw.dll"
    raise RuntimeError(f"Unsupported platform: {sys.platform}")


def load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, _shlib_name())
    if not os.path.exists(path):
        raise RuntimeError(
            f"libdw not found at {path}. Build it first:\n    (cd {here} && make lib)"
        )

    lib = ctypes.CDLL(path)
    dbl_p = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    int_p = ndpointer(dtype=np.int32,   flags="C_CONTIGUOUS")

    lib.tu_matching_create.restype  = ctypes.c_void_p
    lib.tu_matching_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        dbl_p, dbl_p, dbl_p, dbl_p, dbl_p, int_p, int_p,
    ]
    lib.tu_matching_free.restype  = None
    lib.tu_matching_free.argtypes = [ctypes.c_void_p]
    lib.tu_matching_rroa_solve.restype  = ctypes.c_int
    lib.tu_matching_rroa_solve.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int, dbl_p,
    ]

    lib.tu_estimation_create.restype  = ctypes.c_void_p
    lib.tu_estimation_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        int_p, int_p, int_p, dbl_p, dbl_p, dbl_p, dbl_p, dbl_p,
    ]
    lib.tu_estimation_free.restype  = None
    lib.tu_estimation_free.argtypes = [ctypes.c_void_p]
    lib.tu_estimation_rroa_solve.restype  = ctypes.c_int
    lib.tu_estimation_rroa_solve.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int,
        dbl_p, dbl_p,
    ]

    _LIB = lib
    return _LIB


def _f64(arr): return np.ascontiguousarray(arr, dtype=np.float64)
def _i32(arr): return np.ascontiguousarray(arr, dtype=np.int32)


class _CTUMatching:
    """RAII handle wrapping tu_matching_create / tu_matching_free."""

    def __init__(self, Phi_x_y, eps_i_y, eta_x_j, eps_i_0, eta_0_j, x_i, y_j):
        lib = load_lib()
        self._buf = (Phi := _f64(Phi_x_y), eiy := _f64(eps_i_y), exj := _f64(eta_x_j),
                     ei0 := _f64(eps_i_0), e0j := _f64(eta_0_j),
                     xi := _i32(x_i), yj := _i32(y_j))
        I, Y = eiy.shape; X, J = exj.shape
        self.I, self.J, self.X, self.Y = I, J, X, Y
        self._handle = lib.tu_matching_create(I, J, X, Y, Phi, eiy, exj, ei0, e0j, xi, yj)
        if not self._handle:
            raise RuntimeError("tu_matching_create returned NULL")

    def __del__(self):
        try:
            if getattr(self, "_handle", None):
                load_lib().tu_matching_free(self._handle)
                self._handle = None
        except Exception:
            pass

    def rroa_solve(self, max_iter=200, rc_tol=1e-6, verbose=0):
        obj = np.zeros(1, dtype=np.float64)
        rc = load_lib().tu_matching_rroa_solve(
            self._handle, int(max_iter), float(rc_tol), int(verbose), obj)
        if rc:
            raise RuntimeError(f"tu_matching_rroa_solve rc={rc}")
        return float(obj[0])


class _CTUMatchingEstimation:
    """RAII handle for tu_estimation_create / tu_estimation_free."""

    def __init__(self, mu_x_y, mu_x0, mu_0y, phi_x_y_k,
                 eps_i_0, eta_0_j, eps_i_y, eta_x_j):
        lib = load_lib()
        self._buf = (mu := _i32(mu_x_y), mx0 := _i32(mu_x0), m0y := _i32(mu_0y),
                     phi := _f64(phi_x_y_k),
                     ei0 := _f64(eps_i_0), e0j := _f64(eta_0_j),
                     eiy := _f64(eps_i_y), exj := _f64(eta_x_j))
        X, Y, K = phi.shape
        self.X, self.Y, self.K = X, Y, K
        self.I = eiy.shape[0]; self.J = exj.shape[1]
        self._handle = lib.tu_estimation_create(X, Y, K, mu, mx0, m0y, phi,
                                                ei0, e0j, eiy, exj)
        if not self._handle:
            raise RuntimeError("tu_estimation_create returned NULL")

    def __del__(self):
        try:
            if getattr(self, "_handle", None):
                load_lib().tu_estimation_free(self._handle)
                self._handle = None
        except Exception:
            pass

    def rroa_solve(self, max_iter=200, rc_tol=1e-6, verbose=0):
        lam = np.zeros(self.K, dtype=np.float64)
        obj = np.zeros(1,      dtype=np.float64)
        rc = load_lib().tu_estimation_rroa_solve(
            self._handle, int(max_iter), float(rc_tol), int(verbose), lam, obj)
        if rc:
            raise RuntimeError(f"tu_estimation_rroa_solve rc={rc}")
        return lam, float(obj[0])
