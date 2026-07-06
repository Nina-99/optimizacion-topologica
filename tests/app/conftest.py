"""Fixtures compartidas para tests del módulo app — path de páginas Streamlit.

Proporciona la ruta al directorio de páginas de la aplicación Streamlit
para verificar que los archivos existen e importan sin error.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def pages_path():
    """Ruta absoluta al directorio de páginas Streamlit.

    Retorna un Path al directorio src/tda/app/pages/ que contiene
    las páginas de la aplicación (1_TDA_vs_KMedias.py, etc.).
    """
    return str(Path(__file__).resolve().parent.parent.parent / "src" / "tda" / "app" / "pages")
