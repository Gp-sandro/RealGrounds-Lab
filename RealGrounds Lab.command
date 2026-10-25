#!/bin/zsh
# Double-click to start RealGrounds Lab (opens in your browser). Close this window to stop it.
cd "$(dirname "$0")"
source ../.venv/bin/activate
python app.py
