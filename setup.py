"""
setup.py
--------
One-time setup: installs dependencies and creates the data/ folder.

Run:
    python setup.py
"""

import os
import sys
import subprocess


def create_dirs():
    os.makedirs("data", exist_ok=True)
    print("[OK] data/ folder ready.")


def install_deps():
    print("\n[INFO] Installing dependencies (this may take a few minutes)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    )
    if result.returncode != 0:
        print("[ERROR] Installation failed. See errors above.")
        sys.exit(1)
    print("[OK] Dependencies installed.")


def print_next_steps():
    print()


if __name__ == "__main__":
    print("=== DeepFace Facial Recognition — Setup ===\n")
    create_dirs()
    install_deps()
    print_next_steps()