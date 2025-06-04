#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货基本面综合分析系统 - 部署检查脚本
用于验证所有文件是否准备就绪，可以部署到Streamlit Cloud
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}: {filepath} ({size} bytes)")
        return True
    else:
        print(f"❌ {description}: {filepath} - 文件不存在")
        return False

def check_file_content(filepath, required_content, description):
    """检查文件内容"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if required_content in content:
                print(f"✅ {description}: 内容检查通过")
                return True
            else:
                print(f"❌ {description}: 缺少必需内容")
                return False
    except Exception as e:
        print(f"❌ {description}: 读取文件失败 - {e}")
        return False

def main():
    print("🔍 期货基本面综合分析系统 - 部署检查")
    print("=" * 50)
    
    # 检查必需文件
    print("\n📋 检查必需文件:")
    required_files = [
        ("综合期货分析系统.py", "主程序文件"),
        ("requirements.txt", "依赖包列表"),
        (".streamlit/config.toml", "Streamlit配置"),
        ("README.md", "项目说明"),
        (".gitignore", "Git忽略文件")
    ]
    
    all_required_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_required_exist = False
    
    # 检查主程序文件内容
    print("\n🔍 检查主程序文件内容:")
    main_file_checks = [
        ("综合期货分析系统.py", "def main():", "包含main函数"),
        ("综合期货分析系统.py", "st.set_page_config", "包含Streamlit配置"),
        ("综合期货分析系统.py", "import streamlit as st", "导入Streamlit"),
        ("综合期货分析系统.py", "import akshare as ak", "导入AKShare")
    ]
    
    main_content_ok = True
    for filepath, content, description in main_file_checks:
        if not check_file_content(filepath, content, description):
            main_content_ok = False
    
    # 检查requirements.txt内容
    print("\n📦 检查依赖包:")
    required_packages = [
        "streamlit", "akshare", "pandas", "numpy", 
        "matplotlib", "plotly", "scipy", "openpyxl"
    ]
    
    packages_ok = True
    try:
        with open("requirements.txt", 'r') as f:
            requirements_content = f.read()
            for package in required_packages:
                if package in requirements_content:
                    print(f"✅ 依赖包: {package}")
                else:
                    print(f"❌ 依赖包: {package} - 未找到")
                    packages_ok = False
    except Exception as e:
        print(f"❌ 读取requirements.txt失败: {e}")
        packages_ok = False
    
    # 检查可选文件
    print("\n📁 检查可选文件:")
    optional_files = [
        ("futures_basis_strategy.py", "基差策略模块"),
        ("basis_strategy_example.py", "基差策略示例"),
        ("README_基差策略.md", "基差策略说明"),
        ("基差策略原理详解.md", "原理详解"),
        ("快速使用指南.md", "使用指南"),
        ("DEPLOYMENT.md", "部署指南"),
        ("部署文件清单.md", "文件清单")
    ]
    
    for filepath, description in optional_files:
        check_file_exists(filepath, description)
    
    # 检查不应存在的文件
    print("\n🚫 检查不应存在的文件:")
    should_not_exist = [
        "cache/",
        "__pycache__/",
        ".env",
        "*.pyc"
    ]
    
    clean_ok = True
    for pattern in should_not_exist:
        if pattern.endswith('/'):
            # 检查目录
            if os.path.exists(pattern):
                print(f"⚠️ 警告: 目录 {pattern} 存在，应该删除")
                clean_ok = False
            else:
                print(f"✅ 目录 {pattern} 不存在")
        else:
            # 检查文件模式
            import glob
            files = glob.glob(pattern)
            if files:
                print(f"⚠️ 警告: 找到文件 {files}，应该删除")
                clean_ok = False
            else:
                print(f"✅ 文件模式 {pattern} 无匹配")
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 检查总结:")
    
    if all_required_exist:
        print("✅ 所有必需文件存在")
    else:
        print("❌ 缺少必需文件")
    
    if main_content_ok:
        print("✅ 主程序文件内容正确")
    else:
        print("❌ 主程序文件内容有问题")
    
    if packages_ok:
        print("✅ 依赖包配置正确")
    else:
        print("❌ 依赖包配置有问题")
    
    if clean_ok:
        print("✅ 没有不应存在的文件")
    else:
        print("⚠️ 存在应该删除的文件")
    
    # 最终结论
    if all_required_exist and main_content_ok and packages_ok:
        print("\n🎉 部署检查通过！可以上传到GitHub并部署到Streamlit Cloud")
        print("\n📝 下一步操作:")
        print("1. 创建GitHub仓库")
        print("2. 上传所有文件")
        print("3. 在Streamlit Cloud中选择仓库")
        print("4. 设置主文件: 综合期货分析系统.py")
        print("5. 点击Deploy!")
        return True
    else:
        print("\n❌ 部署检查失败！请修复上述问题后重新检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 