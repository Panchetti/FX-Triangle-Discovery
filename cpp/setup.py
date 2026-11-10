"""Build the optional fxpd_fast C++ extension (pybind11) -- a BENCHMARK, not a dependency.

    ./.venv/bin/python -m pip install ./cpp

Core fxpd never imports this: fxpd.econo.hy runs numba, which beat this port on a real
day (12 ms vs 19 ms). It is kept for the notebook-04 benchmark and a parity test.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "fxpd_fast",
        ["src/fxpd_fast.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]

setup(
    name="fxpd_fast",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
