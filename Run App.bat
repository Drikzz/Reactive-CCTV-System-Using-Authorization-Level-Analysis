@echo off
echo Starting Reactive CCTV System...
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m streamlit run scripts\streamlit_app.py
pause