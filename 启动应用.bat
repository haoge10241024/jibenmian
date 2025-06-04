@echo off
echo 正在启动期货基本面综合分析系统...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到Python，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

REM 检查是否已安装依赖
echo 检查依赖包...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误：依赖包安装失败
        pause
        exit /b 1
    )
)

echo 依赖检查完成，正在启动应用...
echo.
echo 应用启动后将自动打开浏览器
echo 如果浏览器未自动打开，请手动访问：http://localhost:8501
echo.
echo 按 Ctrl+C 可停止应用
echo.

REM 启动Streamlit应用
streamlit run 综合期货分析系统.py

pause 