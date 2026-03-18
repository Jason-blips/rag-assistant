@echo off
REM 1. 激活虚拟环境
call D:\ai_env\venv\Scripts\activate.bat

REM 2. 切到项目目录
cd /d D:\ai_env\venv\rag_study

REM 4. 启动 Web 页面（Gradio）
python web_app.py

REM 3. 先启动浏览器（不会阻塞）
start "" "http://127.0.0.1:7860"

REM 5. 结束后按任意键关闭窗口（方便看日志）
pause