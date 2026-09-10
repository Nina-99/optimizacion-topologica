@echo off
setlocal enabledelayedexpansion

REM ============================================
REM install_package.bat — Estructura Topológica
REM Versión 3.0 — Optimizada con uv
REM ============================================

set VENV_DIR=.venv
set REQUIREMENTS=requirements.txt
set APP_PATH=src\tda\app\app_master.py

echo.
echo === Verificando Gestor de Paquetes (uv) ===

REM Buscar uv en PATH
where uv 2>nul
if %errorlevel% equ 0 (
    echo [INFO] uv detectado.
) else (
    echo [INFO] uv no detectado. Instalando uv via PowerShell...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    REM Actualizar PATH para la sesion actual
    set "PATH=%PATH%;%USERPROFILE%\.cargo\bin"
    if %errorlevel% neq 0 (
        echo [ERROR] No se pudo instalar uv automaticamente.
        echo   Instalalo manualmente desde: https://astral.sh/uv
        pause
        exit /b 1
    )
)
echo.

REM ─────────── Verificar requirements.txt ───────────
echo === Verificando dependencias ===
if not exist "%REQUIREMENTS%" (
    echo [ERROR] No se encuentra %REQUIREMENTS%
    pause
    exit /b 1
)
echo OK - %REQUIREMENTS% encontrado
echo.

REM ─────────── Entorno virtual con uv ───────────
echo === Entorno virtual (ultra-rapido) ===

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [WARN] Ya existe un entorno virtual en '%VENV_DIR%\'
    set /p RECREAR="^> Recrearlo? (s/N): "
    if /i "!RECREAR!"=="s" (
        echo Eliminando entorno existente...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo Usando entorno existente.
    )
)

echo Creando entorno virtual con uv...
uv venv %VENV_DIR%

set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe
set VENV_STREAMLIT=%VENV_DIR%\Scripts\streamlit.exe

REM ─────────── Instalar dependencias con uv ───────────
echo.
echo === Instalando dependencias ===
echo Sincronizando paquetes...
echo.

REM Instalacion ultra-rapida de dependencias base
uv pip install -r %REQUIREMENTS%
if %errorlevel% neq 0 (
    echo [ERROR] Fallo la instalacion de dependencias base con uv.
    pause
    exit /b 1
)

REM Instalar el paquete local en modo editable
echo [INFO] Instalando paquete local en modo editable...
uv pip install -e .
if !errorlevel! neq 0 (
    echo [ERROR] No se pudo instalar el paquete local.
    pause
    exit /b 1
)

REM Instalacion de ripser sin dependencias (para evitar persim en Windows/ARM)
echo [INFO] Instalando ripser (sin dependencias)...
uv pip install ripser --no-deps
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo instalar ripser.
    pause
    exit /b 1
)

REM ─────────── Resumen ───────────
echo.
echo ======================================
echo Instalacion completada exitosamente con uv.
echo ======================================
echo.

REM ─────────── Ejecutar app (opcional) ───────────
if exist "%APP_PATH%" (
    set /p EJECUTAR="^> Ejecutar la app ahora? (s/N): "
    if /i "!EJECUTAR!"=="s" (
        echo.
        echo === Iniciando Streamlit ===
        echo.
        %VENV_STREAMLIT% run "%APP_PATH%"
    ) else (
        echo.
        echo   Para ejecutar la app despues:
        echo     %VENV_STREAMLIT% run %APP_PATH%
        echo.
    )
) else (
    echo [WARN] No se encuentra la app en %APP_PATH%
)

pause
