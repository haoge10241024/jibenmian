# 📊 期货基本面综合分析系统

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

> 专业的期货基本面分析工具，通过库存分析和基差分析两个维度，为投资者提供科学的投资决策支持。

## 🎯 系统概述

本系统是一个基于Streamlit的Web应用，集成了期货库存分析、基差分析和综合信号分析功能，帮助投资者进行期货基本面分析。

### 核心功能

- **📈 库存分析**: 基于库存变化趋势判断价格方向
- **💰 基差分析**: 基于现货期货价差进行统计套利
- **🔍 综合分析**: 多维度信号共振提高投资可靠性

## 🚀 在线体验

点击上方的Streamlit徽章即可在线体验系统功能。

## 📋 功能特点

### 库存分析模块
- ✅ 支持70+期货品种库存数据分析
- ✅ 智能识别累库、去库、稳定三种趋势
- ✅ 提供信号强度和置信度评估
- ✅ 支持自定义时间范围和参数

### 基差分析模块
- ✅ 基于Z-Score标准化的统计分析
- ✅ 多技术指标综合评估（RSI、布林带等）
- ✅ 智能识别买基差、卖基差机会
- ✅ 详细的风险评估和持仓建议

### 综合分析模块
- ✅ 信号共振分析，提高投资可靠性
- ✅ 冲突信号识别和处理建议
- ✅ 投资优先级排序
- ✅ 一键导出分析报告

## 🛠️ 技术栈

- **前端框架**: Streamlit
- **数据源**: AKShare
- **数据处理**: Pandas, NumPy
- **图表可视化**: Plotly, Matplotlib
- **统计分析**: SciPy
- **报告导出**: OpenPyXL

## 📦 本地部署

### 环境要求
- Python 3.8+
- 网络连接（用于获取实时数据）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-username/futures-analysis.git
cd futures-analysis
```

2. **创建虚拟环境**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **运行应用**
```bash
streamlit run 综合期货分析系统.py
```

5. **访问应用**
打开浏览器访问 `http://localhost:8501`

## 🌐 Streamlit Cloud 部署

### 一键部署
1. Fork 本仓库到您的GitHub账号
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 使用GitHub账号登录
4. 选择您的仓库
5. 设置主文件路径：`综合期货分析系统.py`
6. 点击 "Deploy!"

### 部署配置
- **主文件**: `综合期货分析系统.py`
- **Python版本**: 3.8+
- **依赖文件**: `requirements.txt`
- **配置文件**: `.streamlit/config.toml`

## 📖 使用指南

### 快速开始
1. **库存分析**: 选择品种 → 设置参数 → 开始分析 → 查看结果
2. **基差分析**: 设置时间范围 → 调整置信度 → 运行分析 → 筛选机会
3. **综合分析**: 完成前两步 → 查看信号共振 → 获取投资建议

### 高级功能
- **自定义筛选**: 支持多维度筛选和排序
- **图表分析**: 交互式图表，支持缩放和详细查看
- **报告导出**: 支持Excel、CSV格式导出
- **缓存优化**: 智能缓存，提高分析速度

## 📊 系统截图

### 主界面
![主界面](screenshots/main.png)

### 库存分析
![库存分析](screenshots/inventory.png)

### 基差分析
![基差分析](screenshots/basis.png)

### 综合分析
![综合分析](screenshots/comprehensive.png)

## 📁 项目结构

```
期货基本面分析/
├── 综合期货分析系统.py          # 主程序文件
├── requirements.txt             # 依赖包列表
├── .streamlit/
│   └── config.toml             # Streamlit配置
├── .gitignore                  # Git忽略文件
├── README.md                   # 项目说明
├── DEPLOYMENT.md               # 部署指南
├── futures_basis_strategy.py   # 基差策略模块
├── basis_strategy_example.py   # 基差策略示例
├── README_基差策略.md          # 基差策略说明
├── 基差策略原理详解.md         # 原理详解
├── 快速使用指南.md             # 使用指南
├── 启动应用.sh                 # Linux/macOS启动脚本
└── 启动应用.bat                # Windows启动脚本
```

## ⚠️ 免责声明

- 本系统仅供学习和研究使用，不构成投资建议
- 投资有风险，决策需谨慎
- 建议结合其他分析方法综合判断
- 注意控制仓位和设置止损

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进系统功能。

### 开发环境设置
1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📞 技术支持

- **作者**: 7haoge
- **邮箱**: 953534947@qq.com
- **创建时间**: 2025.06

如遇到问题，请通过以下方式联系：
1. 提交GitHub Issue
2. 发送邮件到 953534947@qq.com

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🌟 Star History

如果这个项目对您有帮助，请给个Star支持一下！

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/futures-analysis&type=Date)](https://star-history.com/#your-username/futures-analysis&Date)

---

**最后更新**: 2025年6月  
**版本**: v1.0  
**状态**: ✅ 生产就绪 