#!/bin/bash

echo "正在启动期货基本面综合分析系统..."
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误：未检测到Python3，请先安装Python 3.8或更高版本"
    exit 1
fi

# 检查是否已安装依赖
echo "检查依赖包..."
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "正在安装依赖包..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "错误：依赖包安装失败"
        exit 1
    fi
fi

echo "依赖检查完成，正在启动应用..."
echo
echo "应用启动后将自动打开浏览器"
echo "如果浏览器未自动打开，请手动访问：http://localhost:8501"
echo
echo "按 Ctrl+C 可停止应用"
echo

# 启动Streamlit应用
streamlit run 综合期货分析系统.py 