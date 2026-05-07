import subprocess
import os
import sys

# This correctly gets the exe's actual location
if getattr(sys, 'frozen', False):
    base = os.path.dirname(sys.executable)
else:
    base = os.path.dirname(os.path.abspath(__file__))

python = os.path.join(base, ".venv", "Scripts", "python.exe")
install_bat = os.path.join(base, "install.bat")

# Check if venv exists
if not os.path.exists(python):
    print("Virtual environment not found!")
    print("Running install.bat to set up the app...")
    print()
    subprocess.run([install_bat], shell=True, cwd=base)

# Now launch the app
subprocess.run([python, "-m", "streamlit", "run",
    os.path.join(base, "scripts", "streamlit_app.py")], cwd=base)