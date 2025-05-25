import akshare as ak
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class FuturesInventoryAnalyzer:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.seasonal_periods = {
            '农产品': 12,  # 月度季节性
            '工业品': 4,   # 季度季节性
            '能源化工': 4  # 季度季节性
        }
        
    def calculate_seasonal_factor(self, df: pd.DataFrame, category: str) -> pd.Series:
        """计算季节性因子"""
        period = self.seasonal_periods.get(category, 12)
        seasonal = df['库存'].rolling(window=period).mean()
        return seasonal
    
    def calculate_inventory_consumption_ratio(self, df: pd.DataFrame, consumption_data: Optional[pd.DataFrame] = None) -> float:
        """计算库存消费比"""
        if consumption_data is None:
            # 如果没有消费数据，使用历史平均库存作为参考
            return df['库存'].mean() / df['库存'].std()
        return df['库存'].iloc[-1] / consumption_data['消费量'].iloc[-1]
    
    def calculate_inventory_velocity(self, df: pd.DataFrame, days: int = 30) -> float:
        """计算库存周转率"""
        recent_data = df.tail(days)
        return recent_data['增减'].abs().sum() / recent_data['库存'].mean()
    
    def calculate_trend_strength(self, df: pd.DataFrame, window: int = 30) -> float:
        """计算趋势强度"""
        try:
            # 确保数据量足够
            if len(df) < window:
                return 0.0
            
            # 计算价格变化
            price_change = df['库存'].diff().dropna()
            if len(price_change) < window:
                return 0.0
            
            # 计算正向和负向移动
            positive_moves = price_change[price_change > 0].rolling(window=window).sum()
            negative_moves = price_change[price_change < 0].rolling(window=window).sum()
            
            # 确保有足够的数据
            if positive_moves.empty or negative_moves.empty:
                return 0.0
            
            # 获取最后一个有效值
            last_positive = positive_moves.iloc[-1] if not pd.isna(positive_moves.iloc[-1]) else 0
            last_negative = negative_moves.iloc[-1] if not pd.isna(negative_moves.iloc[-1]) else 0
            
            # 计算趋势强度
            total_moves = last_positive + abs(last_negative)
            if total_moves == 0:
                return 0.0
            
            return abs(last_positive - abs(last_negative)) / total_moves
        
        except Exception as e:
            print(f"计算趋势强度时出错: {str(e)}")
            return 0.0
    
    def calculate_dynamic_threshold(self, df: pd.DataFrame, window: int = 60) -> float:
        """计算动态阈值"""
        volatility = df['增减'].rolling(window=window).std().iloc[-1]
        return volatility * stats.norm.ppf(self.confidence_level)
    
    def analyze_inventory_trend(self, df: pd.DataFrame, category: str) -> Dict:
        """综合分析库存趋势"""
        try:
            # 基础指标
            recent_data = df.tail(30)
            total_change = recent_data['增减'].sum()
            avg_change = total_change / len(recent_data)
            
            # 优化变化率计算
            start_inventory = recent_data['库存'].iloc[0]
            end_inventory = recent_data['库存'].iloc[-1]
            if start_inventory > 0:  # 确保起始库存大于0
                change_rate = (end_inventory - start_inventory) / start_inventory * 100
            else:
                change_rate = 0
            
            # 限制变化率范围
            change_rate = min(max(change_rate, -100), 100)  # 限制在-100%到100%之间
            
            # 高级指标
            seasonal_factor = self.calculate_seasonal_factor(df, category)
            inventory_velocity = self.calculate_inventory_velocity(df)
            trend_strength = self.calculate_trend_strength(df)
            dynamic_threshold = self.calculate_dynamic_threshold(df)
            
            # 趋势判断 - 使用更严格的判断标准
            trend = '稳定'
            if change_rate > 10 and avg_change > 0 and trend_strength > 0.3:  # 提高阈值到10%，增加趋势强度要求
                trend = '累库'
            elif change_rate < -10 and avg_change < 0 and trend_strength > 0.3:  # 提高阈值到10%，增加趋势强度要求
                trend = '去库'
            
            # 信号强度计算
            signal_strength = min(abs(change_rate) / max(dynamic_threshold, 1), 1.0)
            
            return {
                '趋势': trend,
                '变化率': change_rate,
                '平均日变化': avg_change,
                '趋势强度': trend_strength,
                '信号强度': signal_strength,
                '库存周转率': inventory_velocity,
                '季节性因子': seasonal_factor.iloc[-1] if not seasonal_factor.empty else 0,
                '动态阈值': dynamic_threshold
            }
        except Exception as e:
            print(f"分析库存趋势时出错: {str(e)}")
            return {
                '趋势': '稳定',
                '变化率': 0,
                '平均日变化': 0,
                '趋势强度': 0,
                '信号强度': 0,
                '库存周转率': 0,
                '季节性因子': 0,
                '动态阈值': 0
            }
    
    def plot_advanced_analysis(self, df: pd.DataFrame, symbol: str, category: str, save_path: str = "inventory_analysis"):
        """绘制高级分析图表"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            # 库存趋势图
            axes[0].plot(df['日期'], df['库存'], label='库存')
            seasonal_factor = self.calculate_seasonal_factor(df, category)
            if not seasonal_factor.empty:
                axes[0].plot(df['日期'], seasonal_factor, 
                            label='季节性趋势', linestyle='--')
            axes[0].set_title(f'{symbol}库存趋势分析')
            axes[0].legend()
            
            # 库存变化率图
            change_rate = df['库存'].pct_change() * 100
            axes[1].bar(df['日期'], change_rate, label='日变化率')
            axes[1].axhline(y=0, color='r', linestyle='-')
            axes[1].set_title('库存变化率')
            
            # 趋势强度图
            trend_strength = []
            for i in range(len(df)):
                if i >= 30:  # 确保有足够的数据计算趋势强度
                    strength = self.calculate_trend_strength(df.iloc[:i+1])
                    trend_strength.append(strength)
                else:
                    trend_strength.append(0)
            
            axes[2].plot(df['日期'], trend_strength, label='趋势强度')
            axes[2].axhline(y=0.6, color='r', linestyle='--', label='强趋势阈值')
            axes[2].set_title('趋势强度分析')
            axes[2].legend()
            
            plt.tight_layout()
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'{symbol}_高级分析.png'))
            plt.close()
        
        except Exception as e:
            print(f"绘制{symbol}高级分析图表时出错: {str(e)}")
            plt.close()  # 确保关闭图表

def get_futures_category(symbol: str) -> str:
    """获取期货品种分类"""
    categories = {
        '农产品': ['豆一', '豆二', '豆粕', '豆油', '玉米', '玉米淀粉', '菜粕', '菜油', '棕榈', '白糖', '棉花', '苹果'],
        '工业品': ['螺纹钢', '热卷', '铁矿石', '焦煤', '焦炭', '不锈钢', '沪铜', '沪铝', '沪锌', '沪铅', '沪镍', '沪锡'],
        '能源化工': ['原油', '燃油', '沥青', 'PTA', '甲醇', '乙二醇', 'PVC', 'PP', '塑料', '橡胶', '20号胶']
    }
    
    for category, symbols in categories.items():
        if symbol in symbols:
            return category
    return '其他'

def get_futures_inventory_data_memory():
    """
    直接通过接口获取所有期货品种的库存数据，返回字典，不保存文件
    """
    futures_symbols = [
        "沪铜", "镍", "锡", "沪铝", "苯乙烯", "液化石油气", "低硫燃料油", "棉纱", 
        "不锈钢", "短纤", "沪铅", "多晶硅", "丁二烯橡胶", "沪锌", "硅铁", "鸡蛋", 
        "瓶片", "工业硅", "沥青", "20号胶", "原木", "豆一", "玉米", "燃油", 
        "菜籽", "碳酸锂", "纸浆", "玉米淀粉", "沪银", "沪金", "塑料", "聚丙烯", 
        "铁矿石", "豆二", "豆粕", "棕榈", "玻璃", "豆油", "橡胶", "烧碱", 
        "菜粕", "PTA", "纯碱", "对二甲苯", "菜油", "生猪", "尿素", "PVC", 
        "乙二醇", "氧化铝", "焦炭", "郑棉", "甲醇", "白糖", "锰硅", "焦煤", 
        "红枣", "螺纹钢", "花生", "苹果", "热卷"
    ]
    data_dict = {}
    for chinese_name in futures_symbols:
        try:
            print(f"正在获取 {chinese_name} 的库存数据...")
            df = ak.futures_inventory_em(symbol=chinese_name)
            
            # 数据验证
            if df is None:
                print(f"{chinese_name} 数据为空")
                continue
                
            if df.empty:
                print(f"{chinese_name} 数据为空DataFrame")
                continue
                
            if '日期' not in df.columns or '库存' not in df.columns:
                print(f"{chinese_name} 数据格式不正确，缺少必要列")
                continue
                
            # 数据类型转换
            try:
                df['日期'] = pd.to_datetime(df['日期'])
                df['库存'] = pd.to_numeric(df['库存'], errors='coerce')
                df = df.dropna(subset=['日期', '库存'])
                
                if len(df) < 2:
                    print(f"{chinese_name} 有效数据量不足")
                    continue
                    
                df['增减'] = df['库存'].diff()
                df = df.dropna(subset=['增减'])
                
                if len(df) < 2:
                    print(f"{chinese_name} 增减数据量不足")
                    continue
                    
                data_dict[chinese_name] = df
                print(f"{chinese_name} 数据获取成功，共 {len(df)} 条记录")
            except Exception as e:
                print(f"{chinese_name} 数据处理失败: {str(e)}")
                continue
                
        except Exception as e:
            print(f"获取 {chinese_name} 数据失败: {str(e)}")
            continue
            
    return data_dict

def plot_inventory_trends(df, symbol, save_path="inventory_plots"):
    """
    绘制库存趋势图
    """
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # 绘制库存曲线
    ax1.plot(df['日期'], df['库存'], label='库存', color='blue', linewidth=2)
    
    # 计算30日移动平均
    df['30日移动平均'] = df['库存'].rolling(window=30).mean()
    ax1.plot(df['日期'], df['30日移动平均'], label='30日移动平均', color='red', linestyle='--', linewidth=1.5)
    
    # 添加变化率标注
    start_inventory = df['库存'].iloc[0]
    end_inventory = df['库存'].iloc[-1]
    if start_inventory != 0:
        change_rate = (end_inventory - start_inventory) / start_inventory * 100
        ax1.text(df['日期'].iloc[-1], df['库存'].iloc[-1], 
                f'变化率: {change_rate:.2f}%', 
                verticalalignment='bottom', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax1.set_title(f'{symbol}库存趋势分析', fontsize=14, fontweight='bold')
    ax1.set_ylabel('库存量', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 绘制增减柱状图
    colors = ['green' if x > 0 else 'red' for x in df['增减']]
    ax2.bar(df['日期'], df['增减'], label='日增减', color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('增减量', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{symbol}_库存趋势.png'), dpi=300, bbox_inches='tight')
    plt.close()

def validate_and_preprocess_data(df):
    """
    验证和预处理数据
    """
    try:
        # 检查必要列是否存在
        if '日期' not in df.columns or '库存' not in df.columns:
            return None, "缺少必要的数据列"
            
        # 数据清洗
        df = df.dropna(subset=['日期', '库存'])
        if len(df) < 2:
            return None, "数据量不足"
            
        # 数据类型转换
        try:
            df['日期'] = pd.to_datetime(df['日期'])
            df['库存'] = pd.to_numeric(df['库存'], errors='coerce')
            df['增减'] = df['库存'].diff()
            df = df.dropna(subset=['增减'])
        except Exception as e:
            return None, f"数据转换失败: {str(e)}"
            
        return df, None
    except Exception as e:
        return None, f"数据验证失败: {str(e)}"

def calculate_optimal_threshold(df, confidence_level=0.95):
    """
    计算最优阈值，增加错误处理
    """
    try:
        # 移除NaN值
        clean_data = df['增减'].dropna()
        
        # 确保数据量足够
        if len(clean_data) < 2:
            return abs(clean_data.mean()) if not clean_data.empty else 0
            
        # 检查数据标准差
        std = clean_data.std()
        if std == 0 or np.isclose(std, 0):
            return abs(clean_data.mean())
            
        # 移除异常值
        try:
            z_scores = np.abs(stats.zscore(clean_data))
            clean_data = clean_data[z_scores < 3]
        except Exception:
            # 如果Z-score计算失败，使用原始数据
            pass
            
        # 计算基本统计量
        mean = clean_data.mean()
        std = clean_data.std()
        
        # 使用正态分布计算阈值
        threshold = stats.norm.ppf(confidence_level) * std
        
        return threshold
    except Exception as e:
        print(f"警告：阈值计算失败，使用默认阈值: {str(e)}")
        return abs(df['增减'].mean()) if not df['增减'].empty else 0

def analyze_recent_trends(df, days=30):
    """
    分析近期库存趋势
    """
    recent_data = df.tail(days)
    print(f"--- 近期数据预览：{recent_data.head()} ---")
    print(f"--- 近期数据索引：{recent_data.index} ---")
    print(f"--- 近期数据列：{recent_data.columns} ---")
    
    # 计算基本统计量
    total_change = recent_data['增减'].sum()
    avg_change = total_change / len(recent_data)
    std_change = recent_data['增减'].std()
    
    # 计算变化率
    start_inventory = recent_data['库存'].iloc[0]
    end_inventory = recent_data['库存'].iloc[-1]
    if start_inventory != 0:
        change_rate = (end_inventory - start_inventory) / start_inventory * 100
    else:
        change_rate = float('inf')
    
    # 计算连续变化天数
    consecutive_days = 0
    current_direction = np.sign(recent_data['增减'].iloc[-1])
    for change in reversed(recent_data['增减'].values):
        if np.sign(change) == current_direction:
            consecutive_days += 1
        else:
            break
    
    # 计算库存变化趋势
    inventory_trend = '稳定'
    if change_rate > 10 and avg_change > 0:  # 库存增加超过10%且平均变化为正
        inventory_trend = '累库'
    elif change_rate < -10 and avg_change < 0:  # 库存减少超过10%且平均变化为负
        inventory_trend = '去库'
    
    return {
        '总变化量': total_change,
        '平均日变化': avg_change,
        '变化标准差': std_change,
        '变化率': change_rate,
        '连续变化天数': consecutive_days,
        '变化方向': '增加' if current_direction > 0 else '减少',
        '起始库存': start_inventory,
        '当前库存': end_inventory,
        '库存趋势': inventory_trend
    }

def plot_inventory_trends_for_signals(data_dict, inventory_trends):
    """
    绘制累库和去库品种的库存走势图
    """
    for trend, symbols in inventory_trends.items():
        if trend in ['累库品种', '去库品种'] and symbols:
            for symbol in symbols:
                if symbol in data_dict:
                    df = data_dict[symbol]
                    plot_inventory_trends(df, symbol, save_path=f"inventory_plots_{trend}")

def plot_signal_analysis(data_dict, inventory_trends, results_df):
    """
    为累库和去库品种绘制专业的信号分析图
    """
    # 创建累库品种分析图
    if inventory_trends['累库品种']:
        plot_category_analysis(data_dict, inventory_trends['累库品种'], '累库品种', results_df, 'green')
    
    # 创建去库品种分析图
    if inventory_trends['去库品种']:
        plot_category_analysis(data_dict, inventory_trends['去库品种'], '去库品种', results_df, 'red')

def plot_category_analysis(data_dict, symbols, category_name, results_df, color_theme):
    """
    绘制特定类别品种的综合分析图
    """
    if not symbols:
        return
    
    # 限制显示数量，选择信号强度最高的前8个
    category_results = results_df[results_df['品种'].isin(symbols)].head(8)
    top_symbols = category_results['品种'].tolist()
    
    # 计算子图布局
    n_symbols = len(top_symbols)
    if n_symbols <= 4:
        rows, cols = 2, 2
    elif n_symbols <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    if n_symbols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'{category_name}库存走势分析 (按信号强度排序)', fontsize=16, fontweight='bold')
    
    for i, symbol in enumerate(top_symbols):
        if symbol not in data_dict:
            continue
            
        df = data_dict[symbol]
        ax = axes[i]
        
        # 获取该品种的分析结果
        symbol_result = category_results[category_results['品种'] == symbol].iloc[0]
        
        # 绘制库存趋势
        ax.plot(df['日期'], df['库存'], color=color_theme, linewidth=2, alpha=0.8)
        
        # 添加30日移动平均线
        ma30 = df['库存'].rolling(window=30).mean()
        ax.plot(df['日期'], ma30, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='30日均线')
        
        # 标注关键信息
        change_rate = symbol_result['变化率']
        signal_strength = symbol_result['信号强度']
        trend_strength = symbol_result['趋势强度']
        
        # 设置标题和标签
        ax.set_title(f'{symbol}\n变化率: {change_rate:.1f}% | 信号强度: {signal_strength:.2f} | 趋势强度: {trend_strength:.2f}', 
                    fontsize=10, fontweight='bold')
        ax.set_ylabel('库存量', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # 添加趋势箭头
        if change_rate > 0:
            ax.annotate('↗', xy=(0.95, 0.95), xycoords='axes fraction', 
                       fontsize=20, color='green', ha='right', va='top')
        else:
            ax.annotate('↘', xy=(0.95, 0.95), xycoords='axes fraction', 
                       fontsize=20, color='red', ha='right', va='top')
    
    # 隐藏多余的子图
    for i in range(n_symbols, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = f"signal_analysis_{category_name}"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{category_name}_综合分析.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_dashboard(results_df, inventory_trends):
    """
    创建分析结果总览仪表板
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. 趋势分布饼图
    ax1 = fig.add_subplot(gs[0, 0])
    trend_counts = [len(inventory_trends['累库品种']), 
                   len(inventory_trends['去库品种']), 
                   len(inventory_trends['库存稳定品种'])]
    labels = ['累库品种', '去库品种', '稳定品种']
    colors = ['green', 'red', 'gray']
    ax1.pie(trend_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('库存趋势分布', fontweight='bold')
    
    # 2. 信号强度分布直方图
    ax2 = fig.add_subplot(gs[0, 1])
    signal_strengths = results_df['信号强度'].dropna()
    ax2.hist(signal_strengths, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_title('信号强度分布', fontweight='bold')
    ax2.set_xlabel('信号强度')
    ax2.set_ylabel('品种数量')
    
    # 3. 变化率分布
    ax3 = fig.add_subplot(gs[0, 2])
    change_rates = results_df['变化率'].replace([np.inf, -np.inf], np.nan).dropna()
    ax3.hist(change_rates, bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
    ax3.set_title('变化率分布', fontweight='bold')
    ax3.set_xlabel('变化率 (%)')
    ax3.set_ylabel('品种数量')
    
    # 4. 品种分类分布
    ax4 = fig.add_subplot(gs[0, 3])
    category_counts = results_df['分类'].value_counts()
    ax4.bar(category_counts.index, category_counts.values, color=['wheat', 'lightblue', 'lightgreen', 'pink'])
    ax4.set_title('品种分类分布', fontweight='bold')
    ax4.set_ylabel('品种数量')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. 累库品种TOP10
    ax5 = fig.add_subplot(gs[1, :2])
    accumulation_data = results_df[results_df['趋势'] == '累库'].head(10)
    if not accumulation_data.empty:
        bars = ax5.barh(accumulation_data['品种'], accumulation_data['变化率'], color='green', alpha=0.7)
        ax5.set_title('累库品种TOP10 (按变化率)', fontweight='bold')
        ax5.set_xlabel('变化率 (%)')
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                    ha='left', va='center', fontsize=8)
    
    # 6. 去库品种TOP10
    ax6 = fig.add_subplot(gs[1, 2:])
    depletion_data = results_df[results_df['趋势'] == '去库'].head(10)
    if not depletion_data.empty:
        bars = ax6.barh(depletion_data['品种'], depletion_data['变化率'], color='red', alpha=0.7)
        ax6.set_title('去库品种TOP10 (按变化率)', fontweight='bold')
        ax6.set_xlabel('变化率 (%)')
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax6.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                    ha='right', va='center', fontsize=8)
    
    # 7. 关键指标统计表
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    # 创建统计表格
    stats_data = [
        ['指标', '累库品种', '去库品种', '稳定品种', '总计'],
        ['品种数量', len(inventory_trends['累库品种']), len(inventory_trends['去库品种']), 
         len(inventory_trends['库存稳定品种']), len(results_df)],
        ['平均变化率', f"{results_df[results_df['趋势']=='累库']['变化率'].mean():.1f}%",
         f"{results_df[results_df['趋势']=='去库']['变化率'].mean():.1f}%",
         f"{results_df[results_df['趋势']=='稳定']['变化率'].mean():.1f}%",
         f"{results_df['变化率'].replace([np.inf, -np.inf], np.nan).mean():.1f}%"],
        ['平均信号强度', f"{results_df[results_df['趋势']=='累库']['信号强度'].mean():.2f}",
         f"{results_df[results_df['趋势']=='去库']['信号强度'].mean():.2f}",
         f"{results_df[results_df['趋势']=='稳定']['信号强度'].mean():.2f}",
         f"{results_df['信号强度'].mean():.2f}"]
    ]
    
    table = ax7.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('期货库存分析总览仪表板', fontsize=18, fontweight='bold', y=0.98)
    
    # 保存仪表板
    os.makedirs("analysis_dashboard", exist_ok=True)
    plt.savefig("analysis_dashboard/库存分析总览仪表板.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_analysis_report(results_df, inventory_trends, data_dict):
    """
    生成分析报告
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("期货库存分析报告")
    report_lines.append("=" * 80)
    report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"分析品种总数: {len(results_df)}")
    report_lines.append("")
    
    # 趋势统计
    report_lines.append("一、库存趋势统计")
    report_lines.append("-" * 40)
    report_lines.append(f"累库品种: {len(inventory_trends['累库品种'])} 个")
    report_lines.append(f"去库品种: {len(inventory_trends['去库品种'])} 个")
    report_lines.append(f"稳定品种: {len(inventory_trends['库存稳定品种'])} 个")
    report_lines.append("")
    
    # 重点关注品种
    if inventory_trends['累库品种']:
        report_lines.append("二、重点累库品种 (TOP5)")
        report_lines.append("-" * 40)
        top_accumulation = results_df[results_df['趋势'] == '累库'].head(5)
        for _, row in top_accumulation.iterrows():
            report_lines.append(f"{row['品种']}: 变化率 {row['变化率']:.1f}%, 信号强度 {row['信号强度']:.2f}")
        report_lines.append("")
    
    if inventory_trends['去库品种']:
        report_lines.append("三、重点去库品种 (TOP5)")
        report_lines.append("-" * 40)
        top_depletion = results_df[results_df['趋势'] == '去库'].head(5)
        for _, row in top_depletion.iterrows():
            report_lines.append(f"{row['品种']}: 变化率 {row['变化率']:.1f}%, 信号强度 {row['信号强度']:.2f}")
        report_lines.append("")
    
    # 分类统计
    report_lines.append("四、分类统计")
    report_lines.append("-" * 40)
    category_stats = results_df.groupby(['分类', '趋势']).size().unstack(fill_value=0)
    report_lines.append(category_stats.to_string())
    report_lines.append("")
    
    # 保存报告
    os.makedirs("analysis_reports", exist_ok=True)
    with open("analysis_reports/库存分析报告.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print("分析报告已保存到 analysis_reports/库存分析报告.txt")

def main():
    """
    主函数，直接用接口数据分析
    """
    print("开始获取期货库存数据...")
    data_dict = get_futures_inventory_data_memory()
    if not data_dict:
        print("未获取到任何有效数据，无法分析")
        return
    print("\n开始分析库存数据...")
    results = []
    inventory_trends = {
        '累库品种': [],
        '去库品种': [],
        '库存稳定品种': []
    }
    recent_analysis = {}
    analyzer = FuturesInventoryAnalyzer()
    
    for symbol, df in data_dict.items():
        try:
            # 数据预处理
            df = df.dropna(subset=['日期', '库存'])
            if len(df) < 30:  # 确保有足够的数据进行分析
                print(f"{symbol} 数据量不足30天，跳过分析")
                continue
                
            category = get_futures_category(symbol)
            analysis = analyzer.analyze_inventory_trend(df, category)
            
            # 绘制高级分析图表
            analyzer.plot_advanced_analysis(df, symbol, category)
            
            results.append({
                '品种': symbol,
                '分类': category,
                **analysis
            })
            
            # 添加到recent_analysis
            recent_analysis[symbol] = {
                '变化率': analysis['变化率'],
                '平均日变化': analysis['平均日变化'],
                '趋势': analysis['趋势']
            }
            
            trend = analysis['趋势']
            if trend == '累库':
                inventory_trends['累库品种'].append(symbol)
            elif trend == '去库':
                inventory_trends['去库品种'].append(symbol)
            else:
                inventory_trends['库存稳定品种'].append(symbol)
            
            print(f"成功分析 {symbol} 的数据")
        except Exception as e:
            print(f"分析 {symbol} 数据时出错: {str(e)}")
            traceback.print_exc()
            
    # 输出结果
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('信号强度', ascending=False)
        print("\n期货库存分析结果：")
        print(results_df.to_string(index=False))
        results_df.to_csv("advanced_inventory_analysis.csv", index=False, encoding='utf-8-sig')
        print("\n分析结果已保存到 advanced_inventory_analysis.csv")
        
        print("\n库存趋势分析：")
        print("\n累库品种：")
        print(", ".join(inventory_trends['累库品种']))
        print("\n去库品种：")
        print(", ".join(inventory_trends['去库品种']))
        print("\n库存稳定品种：")
        print(", ".join(inventory_trends['库存稳定品种']))
        
        if recent_analysis:
            print("\n近期库存变化分析（最近30天）：")
            recent_df = pd.DataFrame.from_dict(recent_analysis, orient='index')
            recent_df = recent_df.sort_values('变化率', ascending=False)
            print(recent_df.to_string())
        
        # 生成专业图表和报告
        print("\n正在生成专业分析图表...")
        plot_signal_analysis(data_dict, inventory_trends, results_df)
        create_summary_dashboard(results_df, inventory_trends)
        generate_analysis_report(results_df, inventory_trends, data_dict)
        
        print("\n专业分析完成！生成的文件包括：")
        print("1. 累库品种综合分析图: signal_analysis_累库品种/")
        print("2. 去库品种综合分析图: signal_analysis_去库品种/")
        print("3. 分析总览仪表板: analysis_dashboard/库存分析总览仪表板.png")
        print("4. 详细分析报告: analysis_reports/库存分析报告.txt")
        print("5. 高级分析图表: inventory_analysis/")
        
    else:
        print("未发现显著的库存变化")

    # 绘制趋势图
    plot_inventory_trends_for_signals(data_dict, inventory_trends)

if __name__ == "__main__":
    main() 