@echo off
cd /d "%~dp0src\backend"
uvicorn main_d3:app --reload --port 8000
