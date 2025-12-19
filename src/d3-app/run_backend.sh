#!/bin/bash
cd "$(dirname "$0")/src/backend"
uvicorn main_d3:app --reload --port 8000
