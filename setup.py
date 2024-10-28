from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "linalg_core",
        sources=[
            "bindings.cpp",  # C++ файл для pybind11
            "LinearAlgebra.cpp",  # Основной файл с реализацией функции transpose
        ],
        include_dirs=[
            pybind11.get_include(),  # Включаем pybind11 заголовки
            ".",  # Указываем текущую директорию для поиска LinearAlgebra.h
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
    packages=[],
    zip_safe=False,
)
