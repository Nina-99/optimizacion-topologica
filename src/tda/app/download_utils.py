"""
download_utils.py — Descargas multi-entorno (browser / .exe compilado).

Problema:
  En el .exe generado con streamlit-desktop-app + PyInstaller, los
  st.download_button() fallan porque pywebview/WebView2 no maneja
  correctamente los Blob URLs que genera Streamlit en el frontend.

Solución:
  - En navegador (desarrollo): usa st.download_button() normalmente,
    pasando todos los kwargs adicionales.
  - En .exe compilado (sys.frozen): guarda el archivo a "Documentos"
    del usuario y muestra la ruta, evitando el Blob URL del webview.
"""

import os
import sys
from pathlib import Path

import streamlit as st


def _get_export_dir() -> Path:
    """Determina dónde guardar los archivos exportados en modo .exe.

    Orden de precedencia:
      1. Variable de entorno EXPORT_TDA_DIR
      2. ~/Documents/TDA-SIMP-Export
      3. ~/Downloads/TDA-SIMP-Export (fallback)
    """
    env_dir = os.environ.get("EXPORT_TDA_DIR")
    if env_dir:
        return Path(env_dir)

    docs = Path.home() / "Documents" / "TDA-SIMP-Export"
    if docs.is_dir() or docs.parent.is_dir():
        return docs

    return Path.home() / "Downloads" / "TDA-SIMP-Export"


def download_button(
    label: str,
    data: bytes | str,
    file_name: str,
    mime: str = "application/octet-stream",
    **kwargs,
) -> None:
    """Botón de descarga que funciona en navegador y en .exe compilado.

    Args:
        label: Texto del botón.
        data: Contenido del archivo (bytes para binario, str para texto).
        file_name: Nombre del archivo (ej. "reporte.pdf").
        mime: Tipo MIME (ej. "application/pdf", "text/csv", "image/png").
        **kwargs: Argumentos adicionales para st.download_button
                  (use_container_width, key, etc.) — solo en modo navegador.
    """
    if getattr(sys, "frozen", False):
        _save_file_frozen(data, file_name)
    else:
        _save_file_browser(label, data, file_name, mime, kwargs)


def _save_file_frozen(data: bytes | str, file_name: str) -> None:
    """Modo .exe: guarda a disco y muestra la ruta."""
    export_dir = _get_export_dir()
    export_dir.mkdir(parents=True, exist_ok=True)
    filepath = export_dir / file_name

    if isinstance(data, str):
        filepath.write_text(data, encoding="utf-8")
    else:
        filepath.write_bytes(data)

    parent = filepath.parent
    st.success(
        f"✅ Archivo guardado:\n\n"
        f"📁 **{parent}**\n\n"
        f"📄 **{file_name}** "
        f"({_human_size(filepath.stat().st_size)})"
    )


def _save_file_browser(
    label: str,
    data: bytes | str,
    file_name: str,
    mime: str,
    kwargs: dict,
) -> None:
    """Modo navegador: usa st.download_button."""
    if isinstance(data, str):
        data = data.encode("utf-8")
        if mime == "application/octet-stream":
            mime = "text/plain; charset=utf-8"
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        **kwargs,
    )


def _human_size(size_bytes: int) -> str:
    """Formato legible de tamaño (KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024**2:.1f} MB"
