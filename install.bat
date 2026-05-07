@echo off
echo Installing Reactive CCTV System...
echo.

echo Step 1: Creating virtual environment...
python -m venv "%~dp0.venv"
echo.

echo Step 2: Installing dependencies...
"%~dp0.venv\Scripts\pip.exe" install -r "%~dp0requirements.txt"
echo.

echo Installation complete!
echo You can now run the app using the desktop shortcut.
pause