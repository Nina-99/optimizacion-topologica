"""Tests del módulo app — verifica que las páginas Streamlit importan sin error.

No se puede testear Streamlit sin headless browser, pero al menos verificamos
que los imports resuelven correctamente y el código de módulo no falla.
"""

import importlib

import pytest

# Lista de módulos de páginas a verificar (nombres con dígito requieren importlib)
PAGE_MODULES = [
    "tda.app.pages.1_TDA_vs_KMedias",
    "tda.app.pages.2_Optimizacion_Topologica",
    "tda.app.pages.3_Optimizacion_SIMP",
]


class TestPagesImport:
    """Cada página Streamlit importa sin error fuera de contexto de servidor."""

    @pytest.mark.parametrize("mod_name", PAGE_MODULES)
    def test_page_imports_without_error(self, mod_name):
        """La página {mod_name} importa sin lanzar excepción."""
        mod = importlib.import_module(mod_name)
        assert mod is not None

    def test_all_pages_in_pages_path(self, pages_path):
        """Todas las páginas listadas existen en el directorio de páginas."""
        import os
        for mod_name in PAGE_MODULES:
            # Convertir nombre de módulo a ruta de archivo
            relative = mod_name.replace(".", "/") + ".py"
            full_path = os.path.join(pages_path, os.path.basename(relative))
            assert os.path.isfile(full_path), (
                f"Página no encontrada: {full_path}"
            )

    def test_pages_path_is_directory(self, pages_path):
        """El path de páginas debe ser un directorio existente."""
        import os
        assert os.path.isdir(pages_path), f"No es directorio: {pages_path}"
