# 期货基差投资策略分析系统

## 📖 项目简介

期货基差投资策略分析系统是一个专业的量化分析工具，通过分析期货与现货价格差异（基差），识别投资机会并提供决策支持。

### 🎯 核心功能

- **基差数据获取**: 自动获取期货和现货价格数据
- **技术指标计算**: RSI、布林带、移动平均等技术分析
- **投资机会识别**: 基于Z-Score和技术指标的综合评分
- **风险评估**: 多维度风险等级评估
- **可视化分析**: 专业的图表分析和报告生成
- **结果导出**: 支持CSV、Excel等格式导出

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 网络连接（用于获取实时数据）

### 安装依赖

```bash
pip install -r requirements_basis.txt
```

### 基本使用

```python
from futures_basis_strategy import FuturesBasisStrategy

# 创建策略分析器
strategy = FuturesBasisStrategy()

# 运行分析
opportunities = strategy.run_analysis(
    end_day="20241219",    # 结束日期
    days_back=30,          # 分析天数
    min_confidence=50.0    # 最低置信度阈值
)

# 显示结果
strategy.display_opportunities()

# 导出结果
strategy.export_results("基差分析结果.csv")
```

## 📊 核心概念

### 基差 (Basis)

基差 = 现货价格 - 期货价格

- **正基差**: 现货价格 > 期货价格
- **负基差**: 现货价格 < 期货价格

### 投资策略

#### 买基差策略
- **操作**: 买入现货 + 卖出期货
- **适用**: 基差异常偏小时（期货被高估）
- **预期**: 基差回归正常，获得价差收益

#### 卖基差策略
- **操作**: 卖出现货 + 买入期货
- **适用**: 基差异常偏大时（现货被高估）
- **预期**: 基差回归正常，获得价差收益

### Z-Score 信号分类

| Z-Score 范围 | 信号类型 | 投资建议 |
|-------------|----------|----------|
| < -1.5 | 极端买基差 | 强烈看空期货 |
| -1.5 ~ -1.0 | 中等买基差 | 看空期货 |
| -1.0 ~ -0.8 | 弱买基差 | 轻微看空 |
| -0.8 ~ 0.8 | 正常范围 | 无明显机会 |
| 0.8 ~ 1.0 | 弱卖基差 | 轻微看多 |
| 1.0 ~ 1.5 | 中等卖基差 | 看多期货 |
| > 1.5 | 极端卖基差 | 强烈看多期货 |

## 🔧 参数配置

### 分析参数

- **end_day**: 分析结束日期 (格式: YYYYMMDD)
- **days_back**: 历史数据天数 (建议: 20-60天)
- **min_confidence**: 最低置信度阈值 (建议: 30-70%)

### 置信度阈值建议

| 投资风格 | 推荐阈值 | 特点 |
|----------|----------|------|
| 保守型 | 60-70% | 机会少但质量高 |
| 平衡型 | 40-50% | 数量质量兼顾 |
| 激进型 | 30-40% | 机会多需自行筛选 |

## 📈 技术指标

### RSI (相对强弱指数)
- **超卖**: RSI < 35
- **超买**: RSI > 65
- **用途**: 确认基差反转信号

### 布林带
- **下轨突破**: 基差位置 < 25%
- **上轨突破**: 基差位置 > 75%
- **用途**: 识别基差极值区域

### 趋势反转
- **看涨反转**: 基差偏小且开始回升
- **看跌反转**: 基差偏大且开始回落

## ⚠️ 风险评估

### 风险等级

| 风险等级 | 评分范围 | 特征 |
|----------|----------|------|
| 低风险 | 0-2分 | 波动率低，数据质量好 |
| 中风险 | 3-4分 | 中等波动率或轻微异常 |
| 高风险 | 5分以上 | 高波动率，极端异常 |

### 风险因子

1. **波动率风险**: 基差历史波动程度
2. **极端程度**: Z-Score绝对值大小
3. **数据质量**: 历史数据完整性
4. **趋势一致性**: 现货期货价格趋势匹配度

## 📋 使用示例

### 示例1: 快速分析

```python
# 使用默认参数进行快速分析
strategy = FuturesBasisStrategy()
opportunities = strategy.run_analysis("20241219")
strategy.display_opportunities(top_n=5)
```

### 示例2: 自定义参数

```python
# 自定义分析参数
strategy = FuturesBasisStrategy()
opportunities = strategy.run_analysis(
    end_day="20241219",
    days_back=45,
    min_confidence=40.0
)
```

### 示例3: 图表分析

```python
# 生成特定品种的详细图表
if opportunities:
    variety = opportunities[0].variety
    strategy.plot_opportunity_analysis(variety, f"{variety}_分析图表.png")
```

### 示例4: 批量测试

```python
# 测试不同置信度阈值的效果
confidence_levels = [30, 40, 50, 60, 70]
for confidence in confidence_levels:
    opportunities = strategy.run_analysis(
        end_day="20241219",
        min_confidence=confidence
    )
    print(f"置信度{confidence}%: 发现{len(opportunities)}个机会")
```

## 📊 结果解读

### 投资机会表格

| 字段 | 说明 |
|------|------|
| 品种名称 | 期货品种名称 |
| 机会类型 | 买基差/卖基差机会 |
| 置信度 | 投资成功概率 (0-100%) |
| 预期收益 | 预期收益率 (%) |
| 风险等级 | 低/中/高风险 |
| 建议持仓 | 推荐持仓天数 |
| Z-Score | 标准化基差值 |
| 当前基差 | 当前基差数值 |

### 图表分析

1. **价格走势图**: 现货vs期货价格对比
2. **基差分析图**: 基差走势、布林带、历史均值
3. **基差分布图**: 基差历史分布和当前位置
4. **技术指标图**: RSI指标和基差走势

## 🛠️ 高级功能

### 策略解释

```python
# 查看策略逻辑解释
strategy.explain_simple_logic()
strategy.explain_confidence_threshold()
strategy.explain_analysis_criteria()
```

### 分析摘要

```python
# 获取分析结果摘要
summary = strategy.get_analysis_summary()
print(f"总机会数: {summary['total_opportunities']}")
print(f"平均置信度: {summary['avg_confidence']:.1f}%")
```

### 结果导出

```python
# 导出CSV格式
strategy.export_results("基差分析结果.csv")

# 导出Excel格式（需要在Streamlit中使用）
# 包含详细的分析报告和图表
```

## 🔍 常见问题

### Q1: 为什么没有发现投资机会？

**A**: 可能的原因：
- 置信度阈值设置过高，尝试降低到30-40%
- 市场处于正常状态，基差没有异常
- 数据获取失败，检查网络连接

### Q2: 如何选择合适的置信度阈值？

**A**: 建议策略：
- 新手：从50%开始，观察结果数量
- 机会太少：降低到40%或30%
- 机会太多：提高到60%或70%

### Q3: 基差交易的实际操作？

**A**: 基差交易需要：
- 同时操作现货和期货市场
- 具备现货交易渠道和资质
- 考虑交易成本和资金占用
- 建议先进行模拟交易

### Q4: 如何理解Z-Score？

**A**: Z-Score表示当前基差偏离历史均值的程度：
- Z-Score = (当前基差 - 历史均值) / 历史标准差
- 绝对值越大，偏离程度越大，投资机会越明显
- 正值表示基差偏大，负值表示基差偏小

## 📞 技术支持

如有问题或建议，请通过以下方式联系：

- 查看代码注释和文档
- 运行示例程序了解用法
- 使用`explain_*`方法查看策略解释

## 📄 免责声明

本系统仅供学习和研究使用，不构成投资建议。期货交易存在风险，投资者应：

- 充分了解期货交易风险
- 根据自身情况制定投资策略
- 谨慎使用分析结果
- 建议先进行模拟交易

投资有风险，入市需谨慎！ 