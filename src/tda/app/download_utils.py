"""
download_utils.py — Descargas multi-entorno (browser / .exe compilado).

Problema original:
  En el .exe generado con streamlit-desktop-app + PyInstaller, los
  st.download_button() fallan porque pywebview/WebView2 no maneja
  correctamente los Blob URLs que genera Streamlit en el frontend.

Solución:
  - En navegador (desarrollo): usa st.download_button() normalmente.
  - En .exe compilado (sys.frozen): guarda el archivo a disco y muestra
    la ruta, evitando el Blob URL del webview.

Flujo en modo .exe:
  1. Si el usuario ya configuró EXPORT_TDA_DIR vía UI o env var → guarda ahí.
  2. Si no → muestra un prompt pidiendo la ubicación (text_input).
  3. El usuario puede cambiar la ruta en cualquier momento desde la UI.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

import streamlit as st

# ─────────────────────────────────────────────────────────
#  Sesión: clave única para el directorio de exportación
# ─────────────────────────────────────────────────────────
_SESSION_KEY = "_export_dir"
_BOOL_KEY = "_export_configured"


def _is_frozen() -> bool:
    """¿Estamos ejecutando como .exe compilado?"""
    return getattr(sys, "frozen", False)


def _default_export_dir() -> Path:
    """Directorio por defecto para exportaciones en modo .exe."""
    env_dir = os.environ.get("EXPORT_TDA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / "Documents" / "TDA-SIMP-Export"


# ─────────────────────────────────────────────────────────
#  UI de configuración
# ─────────────────────────────────────────────────────────


def export_settings_ui():
    """Widget compacto para configurar el directorio de exportación.

    Usar en sidebar o expander — se ve solo en modo .exe.
    En navegador no tiene efecto (se usa st.download_button).
    """
    if not _is_frozen():
        return

    _init_session_state()

    with st.expander("📁 Exportación (configurar)", expanded=False):
        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            nuevo_path = st.text_input(
                "Directorio de exportación",
                value=st.session_state[_SESSION_KEY],
                key="_export_dir_input",
                help="Los archivos exportados se guardarán en esta carpeta.",
                placeholder="C:\\Users\\...\\TDA-SIMP-Export",
            )

        with col2:
            st.write("")  # spacer
            st.write("")  # spacer
            if st.button("📂 Abrir", key="_export_btn_open", use_container_width=True):
                _abrir_carpeta(st.session_state[_SESSION_KEY])

        with col3:
            st.write("")  # spacer
            st.write("")  # spacer
            if st.button(
                "↺ Default", key="_export_btn_reset", use_container_width=True
            ):
                nuevo_path = str(_default_export_dir())

        # Sincronizar si cambió
        if nuevo_path != st.session_state[_SESSION_KEY]:
            st.session_state[_SESSION_KEY] = nuevo_path
            st.session_state[_BOOL_KEY] = True
            st.rerun()


def prompt_export_dir() -> str:
    """Pregunta la ubicación si no está configurada.

    Returns:
        Ruta absoluta del directorio de exportación.
    """
    _init_session_state()

    if st.session_state.get(_BOOL_KEY, False):
        return st.session_state[_SESSION_KEY]

    st.info(
        "💡 Configurá el directorio donde se guardarán los archivos exportados. "
        "Podés cambiarlo después en el menú de exportación."
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        ruta = st.text_input(
            "📁 Directorio de exportación",
            value=st.session_state[_SESSION_KEY],
            key="_export_prompt_input",
        )
    with col2:
        st.write("")  # spacer vertical
        st.write("")  # spacer vertical
        if st.button("✅ Usar esta ruta", key="_export_prompt_btn", type="primary"):
            ruta_val = ruta.strip() or str(_default_export_dir())
            st.session_state[_SESSION_KEY] = ruta_val
            st.session_state[_BOOL_KEY] = True
            st.rerun()
            return ruta_val

    # Si ya pulsó "Usar esta ruta" en una ejecución anterior
    if st.session_state.get(_BOOL_KEY, False):
        return st.session_state[_SESSION_KEY]

    st.stop()  # No seguir hasta que configure
    return ""  # unreachable


def _init_session_state():
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = str(_default_export_dir())
    if _BOOL_KEY not in st.session_state:
        # Marcar como configurado si la env var está seteada
        st.session_state[_BOOL_KEY] = bool(os.environ.get("EXPORT_TDA_DIR"))


# ─────────────────────────────────────────────────────────
#  Download button principal
# ─────────────────────────────────────────────────────────


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
    if _is_frozen():
        _save_file_frozen(data, file_name)
    else:
        _save_file_browser(label, data, file_name, mime, kwargs)


def _save_file_frozen(data: bytes | str, file_name: str) -> None:
    """Modo .exe: pregunta ubicación (si no está configurada) y guarda."""
    export_dir = Path(prompt_export_dir())
    export_dir.mkdir(parents=True, exist_ok=True)
    filepath = export_dir / file_name

    if isinstance(data, str):
        filepath.write_text(data, encoding="utf-8")
    else:
        filepath.write_bytes(data)

    st.success(
        f"✅ Archivo guardado:\n\n"
        f"📁 **{filepath.parent}**\n\n"
        f"📄 **{file_name}** "
        f"({_human_size(filepath.stat().st_size)})"
    )

    # Botón para abrir la carpeta
    if st.button("📂 Abrir carpeta", key=f"_open_folder_{file_name}"):
        _abrir_carpeta(str(export_dir))


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


# ─────────────────────────────────────────────────────────
#  Utilitarios
# ─────────────────────────────────────────────────────────


def _abrir_carpeta(path: str) -> None:
    """Abre la carpeta en el explorador de archivos del SO."""
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        st.warning(f"No se pudo abrir la carpeta: {e}")


def _human_size(size_bytes: int) -> str:
    """Formato legible de tamaño (KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024**2:.1f} MB"
