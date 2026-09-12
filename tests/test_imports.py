"""Smoke tests — verifica que los módulos principales importan sin error."""

import importlib
import pkgutil

import pytest

streamlit = pytest.importorskip("streamlit", reason="Streamlit no instalado")


@pytest.fixture(scope="session")
def package_name() -> str:
    return "tda"


def test_package_importable(package_name: str) -> None:
    """El paquete principal se importa sin errores."""
    mod = importlib.import_module(package_name)
    assert mod is not None


# Módulos que requieren streamlit — se skippean si no está disponible
STREAMLIT_MODULES = {"tda.app.theme", "tda.app.plataforma_tda_simp", "tda.app.download_utils"}
STREAMLIT_PAGE_PREFIX = "tda.app.pages."


def test_all_submodules_importable(package_name: str) -> None:
    """Todos los submódulos descubiertos se importan sin errores."""
    parent = importlib.import_module(package_name)
    errors: list[str] = []

    for info in pkgutil.walk_packages(parent.__path__, prefix=f"{package_name}."):
        if info.name.startswith(STREAMLIT_PAGE_PREFIX):
            continue
        if info.name in STREAMLIT_MODULES:
            continue
        try:
            importlib.import_module(info.name)
        except Exception as exc:
            errors.append(f"{info.name}: {exc}")

    assert not errors, f"Módulos que fallaron al importar:\n" + "\n".join(errors)


def test_numpy_available() -> None:
    """Dependencia crítica: numpy."""
    import numpy as np  # noqa: F401


def test_streamlit_available() -> None:
    """Dependencia crítica: streamlit."""
    import streamlit as st  # noqa: F401


def test_matplotlib_available() -> None:
    """Dependencia crítica: matplotlib."""
    import matplotlib  # noqa: F401
