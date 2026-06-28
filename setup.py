from setuptools import setup, find_packages
import pathlib

BASE_DIR = pathlib.Path(__file__).parent

setup(
    name="estructura_topologica",
    version="0.1.0",
    description="Proyecto de análisis topológico y optimización SIMP",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
