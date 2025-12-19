#!/bin/bash
cd "$(dirname "$0")/src/backend"
python -m uvicorn main_d3:app --reload --port 8000
