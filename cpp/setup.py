"""Build the optional fxpd_fast C++ extension (pybind11).

    ./.venv/bin/python -m pip install ./cpp

Core fxpd never imports this directly; fxpd.econo.hy falls back to numba/numpy
when it is not built.
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
