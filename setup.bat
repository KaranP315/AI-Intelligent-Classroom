@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo AI-EdTech Project Setup
echo ============================================================

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Bootstrapping SSCon models if needed...
python setup_models.py

echo.
echo Setup complete.
echo.
echo Run a full classroom session:
echo   venv\Scripts\activate
echo   python session_runner.py
echo.
echo Run text-only SSCon inference:
echo   python inference.py
echo.
echo Generate report from existing logs:
echo   python score_attentiveness.py
echo.

endlocal
