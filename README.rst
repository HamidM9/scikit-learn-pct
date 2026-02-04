This is a brief walkthrough of how to test our modifications. The assumption is you are using MacOS and Pycharm.

First, we have to download and install this modified version of scikit-learn:
This command sequence recreates a Python 3.11 venv for the project in your computer and installs this local modified scikit-learn in editable mode.

git clone https://github.com/HamidM9/scikit-learn-pct.git
cd scikit-learn-pct


# 1) Remove any existing venv (safe even if it doesn't exist)
deactivate 2>/dev/null || true
rm -rf .venv


# 2) Create + activate a Python 3.11 venv
python3.11 -m venv .venv
source .venv/bin/activate


# 3) Upgrade packaging tools
python -m pip install -U pip setuptools wheel


# 4) Install build prerequisites (includes meson-python / mesonpy backend)
python -m pip install -U "numpy>=2.0" "scipy>=1.11" cython pybind11 meson ninja meson-python


# 5) Install this repository in editable mode
python -m pip install -e . --no-build-isolation


# 6) Quick verification: should print a path from this local checkout
python -c "import sklearn; print(sklearn.__file__)"



Second,


