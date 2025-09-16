@echo off
echo Starting Trading Platform...

REM Start backend (try different python commands)
if exist backend_stock_scanner.py (
    start cmd /k python backend_stock_scanner.py
    timeout /t 2 >nul
)

if exist watchlist_backend.py (
    start cmd /k python watchlist_backend.py  
    timeout /t 2 >nul
)

REM Wait a bit for backends to start
timeout /t 3 >nul

REM Start React frontend
if exist package.json (
    start cmd /k npm start
) else (
    echo ERROR: package.json not found - make sure you're in the React directory
    pause
)

echo Done! Check the opened windows for any errors.
pause