"""Tests del módulo app — verifica que las páginas Streamlit importan sin error.

No se puede testear Streamlit sin headless browser, pero al menos verificamos
que los imports resuelven correctamente y el código de módulo no falla.
"""

import importlib
import os

import pytest

pytest.importorskip("streamlit", reason="Streamlit no instalado")

# Lista de módulos de páginas a verificar (nombres con dígito requieren importlib)
PAGE_FILENAMES = [
    "1_H.E.1_Robustez_TDA_vs_Euclidianos.py",
    "2_H.E.2_Optimizacion_SIMP_Metrica_Compuesta.py",
    "3_H.G._Comparacion_Integrada_TDA-SIMP.py",
    "4_Playground.py"
]


class TestPagesImport:
    """Cada página Streamlit importa sin error fuera de contexto de servidor."""

    @pytest.mark.parametrize("filename", PAGE_FILENAMES)
    def test_page_imports_without_error(self, filename, pages_path):
        """La página {filename} se puede importar sin errores de sintaxis."""
        full_path = os.path.join(pages_path, filename)
        spec = importlib.util.spec_from_file_location(filename[:-3], full_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod is not None

    def test_all_pages_in_pages_path(self, pages_path):
        """Todas las páginas listadas existen en el directorio de páginas."""
        for filename in PAGE_FILENAMES:
            full_path = os.path.join(pages_path, filename)
            assert os.path.isfile(full_path), (
                f"Página no encontrada: {full_path}"
            )

    def test_pages_path_is_directory(self, pages_path):
        """El path de páginas debe ser un directorio existente."""
        assert os.path.isdir(pages_path), f"No es directorio: {pages_path}"
