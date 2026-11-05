@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul || (echo Python not found. Install Python 3.10+ from python.org and tick "Add to PATH". & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv || (pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"
fc /b requirements.txt ".venv\installed_requirements.txt" >nul 2>nul
if errorlevel 1 (
  echo Installing requirements - may take several minutes...
  python -m pip install --upgrade pip
  pip install -r requirements.txt || (echo Install failed. & pause & exit /b 1)
  copy /y requirements.txt ".venv\installed_requirements.txt" >nul
)
echo Starting app at http://localhost:8501 ...
streamlit run app.py
pause
