# 期货基本面综合分析系统 - 部署指南

## 📋 部署前检查清单

### 必需文件
- [x] `综合期货分析系统.py` - 主程序文件
- [x] `requirements.txt` - Python依赖包
- [x] `.streamlit/config.toml` - Streamlit配置
- [x] `README.md` - 项目说明
- [x] `.gitignore` - Git忽略文件

### 可选文件（用于本地开发）
- [ ] `futures_basis_strategy.py` - 基差策略模块
- [ ] `basis_strategy_example.py` - 基差策略示例
- [ ] `README_基差策略.md` - 基差策略说明
- [ ] `基差策略原理详解.md` - 原理详解
- [ ] `快速使用指南.md` - 使用指南
- [ ] `启动应用.sh` / `启动应用.bat` - 本地启动脚本

## 🚀 Streamlit Cloud 部署步骤

### 1. 准备GitHub仓库
```bash
# 1. 创建新的GitHub仓库
# 2. 克隆到本地
git clone https://github.com/your-username/futures-analysis.git

# 3. 复制文件到仓库
cp -r "期货基本面分析"/* futures-analysis/

# 4. 提交代码
cd futures-analysis
git add .
git commit -m "Initial commit: 期货基本面综合分析系统"
git push origin main
```

### 2. 部署到Streamlit Cloud
1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 使用GitHub账号登录
3. 点击 "New app"
4. 选择您的仓库和分支
5. 设置主文件路径：`综合期货分析系统.py`
6. 点击 "Deploy!"

### 3. 环境变量配置（如需要）
在Streamlit Cloud的App设置中，可以添加环境变量：
```
# 示例（如果需要API密钥）
AKSHARE_TOKEN=your_token_here
```

## 🔧 本地部署步骤

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行应用
```bash
# 方法1：直接运行
streamlit run 综合期货分析系统.py

# 方法2：使用启动脚本
# Windows:
启动应用.bat
# macOS/Linux:
chmod +x 启动应用.sh
./启动应用.sh
```

## 📁 文件结构说明

```
期货基本面分析/
├── 综合期货分析系统.py          # 主程序文件（必需）
├── requirements.txt             # 依赖包列表（必需）
├── .streamlit/
│   └── config.toml             # Streamlit配置（必需）
├── .gitignore                  # Git忽略文件（推荐）
├── README.md                   # 项目说明（推荐）
├── DEPLOYMENT.md               # 部署指南（本文件）
├── cache/                      # 缓存文件夹（自动生成，不上传）
├── futures_basis_strategy.py   # 基差策略模块（可选）
├── basis_strategy_example.py   # 基差策略示例（可选）
├── README_基差策略.md          # 基差策略说明（可选）
├── 基差策略原理详解.md         # 原理详解（可选）
├── 快速使用指南.md             # 使用指南（可选）
├── 启动应用.sh                 # Linux/macOS启动脚本（可选）
└── 启动应用.bat                # Windows启动脚本（可选）
```

## ⚠️ 注意事项

### 部署前必须检查
1. **主程序文件名**：确保是 `综合期货分析系统.py`
2. **依赖包版本**：确保 `requirements.txt` 中的版本兼容
3. **缓存文件夹**：`cache/` 文件夹不要上传到GitHub
4. **配置文件**：`.streamlit/config.toml` 配置正确

### 常见问题解决
1. **模块导入错误**：检查 `requirements.txt` 是否包含所有依赖
2. **编码问题**：确保所有文件使用UTF-8编码
3. **内存不足**：Streamlit Cloud有内存限制，避免加载过大数据
4. **网络超时**：akshare数据获取可能超时，已内置重试机制

### 性能优化建议
1. **缓存机制**：系统已内置智能缓存，重复分析更快
2. **数据筛选**：建议分批分析，避免一次性分析过多品种
3. **图表显示**：按需显示图表，避免同时显示过多

## 📞 技术支持

- **作者**：7haoge
- **邮箱**：953534947@qq.com
- **创建时间**：2025.06

如遇到部署问题，请联系作者获取技术支持。 