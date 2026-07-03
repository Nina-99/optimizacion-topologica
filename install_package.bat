@echo off
setlocal enabledelayedexpansion

REM ============================================
REM install_package.bat — Estructura Topológica
REM Versión 2.0 — multiplataforma
REM ============================================

set VENV_DIR=venv
set REQUIREMENTS=requirements.txt
set APP_PATH=src\tda\app\app_master.py

echo.
echo === Verificando Python ===

REM Buscar Python en PATH
set PYTHON_CMD=
where python 2>nul >nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
) else (
    where python3 2>nul >nul
    if !errorlevel! equ 0 (
        set PYTHON_CMD=python3
    )
)

if not defined PYTHON_CMD (
    echo [ERROR] Python no esta instalado.
    echo.
    echo   Descargalo desde: https://www.python.org/downloads/
    echo   IMPORTANTE: Marca "Add Python to PATH" durante la instalacion.
    echo.
    pause
    exit /b 1
)

%PYTHON_CMD% --version
echo.

REM Verificar que venv funciona
%PYTHON_CMD% -c "import venv" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] El modulo 'venv' no esta disponible.
    echo   Reinstala Python y asegurate de incluir 'venv'.
    pause
    exit /b 1
)

REM ─────────── Verificar requirements.txt ───────────
echo === Verificando dependencias ===
if not exist "%REQUIREMENTS%" (
    echo [ERROR] No se encuentra %REQUIREMENTS%
    pause
    exit /b 1
)
echo OK - %REQUIREMENTS% encontrado
echo.

REM ─────────── Entorno virtual ───────────
echo === Entorno virtual ===

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [WARN] Ya existe un entorno virtual en '%VENV_DIR%\'
    set /p RECREAR="^> Recrearlo? (s/N): "
    if /i "!RECREAR!"=="s" (
        echo Eliminando entorno existente...
        rmdir /s /q "%VENV_DIR%"
        echo Creando nuevo entorno virtual...
        %PYTHON_CMD% -m venv "%VENV_DIR%"
    ) else (
        echo Usando entorno existente.
    )
) else (
    echo Creando entorno virtual...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
)

if not exist "%VENV_DIR%\Scripts\pip.exe" (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)

set VENV_PIP=%VENV_DIR%\Scripts\pip.exe
set VENV_STREAMLIT=%VENV_DIR%\Scripts\streamlit.exe

REM ─────────── Instalar dependencias ───────────
echo.
echo === Instalando dependencias ===
echo Esto puede tomar varios minutos...
echo.

REM NOTA: ripser depende de persim, y persim NO tiene wheels para Windows.
REM Estrategia:
REM   1. Instalar requirements.txt (numpy, scipy, scikit-learn, etc.)
REM   2. Instalar ripser con --no-deps (evita persim, pero ya tiene numpy/scipy)

echo [1/2] Instalando dependencias base...
%VENV_PIP% install -r %REQUIREMENTS%
if %errorlevel% neq 0 (
    echo [ERROR] Fallo la instalacion de dependencias base.
    echo   Revisa el mensaje de error de arriba.
    pause
    exit /b 1
)

echo [2/2] Instalando ripser (sin persim)...
%VENV_PIP% install ripser --no-deps
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo instalar ripser.
    pause
    exit /b 1
)

REM ─────────── Resumen ───────────
echo.
echo ======================================
echo Instalacion completada exitosamente.
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
    echo   Revisa la ruta o ejecutala manualmente con:
    echo     %VENV_STREAMLIT% run ^<tu-app^>.py
    echo.
)

pause
