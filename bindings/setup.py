# setup.py
import os
from setuptools import setup, Extension
import pybind11

SRC_DIR = "linalg/src"
PYTHON_DIR = "linalg/python"

ext_modules = [
    Extension(
        "linalg_core",  # Название расширения
        sources=[
            os.path.join(PYTHON_DIR, "bindings.cpp"),
            os.path.join(SRC_DIR, "LinearAlgebra.cpp"),
        ],
        include_dirs=[
            pybind11.get_include(),
            SRC_DIR,
        ],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3", "-march=native"],
        extra_link_args=["-lopenblas"],
    ),
]

setup(
    name="linalg",
    version="0.1",
    description="Linear Algebra library with C++ and pybind11",
    ext_modules=ext_modules,
    packages=["linalg"],
    package_dir={"linalg": "linalg/python"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
    zip_safe=False,
)
