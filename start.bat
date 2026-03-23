@echo off
cd /d "%~dp0"
REM 若未全局安装依赖，请先在本窗口执行： path\to\venv\Scripts\activate.bat
start "RAG-API" cmd /k "cd /d \"%~dp0\" && uvicorn backend_api:app --host 127.0.0.1 --port 8000"
timeout /t 2 /nobreak >nul
streamlit run streamlit_app.py
pause
