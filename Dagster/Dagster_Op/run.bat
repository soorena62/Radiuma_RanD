@echo off
REM Activate virtual environment
call venv\Scripts\activate
cd /d C:\Users\Omen16\Documents\Radiuma_Mini

REM Run the Radiuma Mini application
python -m app.main

pause
