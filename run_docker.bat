@echo off
REM ============================================================
REM  TraderPro - Single launcher for Docker (Windows)
REM  Usage:
REM    Double-click -> DEV mode (hot-reload)
REM    run_docker.bat prod -> PRODUCTION mode (detached)
REM ============================================================

setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0"

REM --- Check Docker is available ---
docker --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker is not installed or not in PATH.
  echo Install Docker Desktop and try again.
  pause
  exit /b 1
)

REM --- Ensure .env exists (copy from example if present) ---
if not exist ".env" (
  if exist ".env.example" (
    echo [.env] not found. Creating from .env.example ...
    copy /Y ".env.example" ".env" >nul
  ) else (
    echo [WARN] .env not found and .env.example missing.
    echo Creating a minimal .env with placeholders...
    (
      echo TWELVEDATA_API_KEY=
      echo FINNHUB_API_KEY=
      echo POLYGON_API_KEY=
      echo FMP_API_KEY=
      echo ALPHAVANTAGE_API_KEY=
      echo IBKR_HOST=host.docker.internal
      echo IBKR_PORT=7497
      echo IBKR_CLIENT_ID=7
    ) > ".env"
  )
)

set COMPOSE_CONVERT_WINDOWS_PATHS=1

REM --- DEV (default) vs PROD (arg) ---
set MODE=dev
if /I "%~1"=="prod" set MODE=prod

if /I "!MODE!"=="prod" (
  if not exist "docker-compose.prod.yml" (
    echo [ERROR] docker-compose.prod.yml not found in %cd%
    pause
    exit /b 1
  )
  echo.
  echo === Starting PRODUCTION (detached) ===
  docker compose -f docker-compose.prod.yml up --build -d
  if errorlevel 1 (
    echo [ERROR] Failed to start production stack.
    pause
    exit /b 1
  )
  echo.
  echo Frontend: http://localhost:8080
  echo Backend : http://localhost:8000
  echo To stop : run_docker.bat stop
  echo.
  exit /b 0
) else (
  if not exist "docker-compose.dev.yml" (
    echo [ERROR] docker-compose.dev.yml not found in %cd%
    pause
    exit /b 1
  )
  echo.
  echo === Starting DEV (attached) ===
  echo (Close with Ctrl+C or run run_docker.bat stop in another window)
  echo.
  docker compose -f docker-compose.dev.yml up --build
  exit /b %errorlevel%
)

:stop



:stop
setlocal
set COMPOSE_CONVERT_WINDOWS_PATHS=1
if exist "docker-compose.dev.yml" docker compose -f docker-compose.dev.yml down
if exist "docker-compose.prod.yml" docker compose -f docker-compose.prod.yml down
echo Stopped (if running).
