@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Virtual environment not found. Run: python -m venv .venv
    echo Then run: .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "ocr_locator_app.py"
