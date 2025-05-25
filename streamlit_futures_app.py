import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import base64
from typing import Dict, List, Tuple, Optional
import time

warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="期货库存分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class FuturesInventoryAnalyzer:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.seasonal_periods = {
            '农产品': 12,
            '工业品': 4,
            '能源化工': 4
        }
        
    def calculate_seasonal_factor(self, df: pd.DataFrame, category: str) -> pd.Series:
        """计算季节性因子"""
        period = self.seasonal_periods.get(category, 12)
        seasonal = df['库存'].rolling(window=period).mean()
        return seasonal
    
    def calculate_inventory_velocity(self, df: pd.DataFrame, days: int = 30) -> float:
        """计算库存周转率"""
        recent_data = df.tail(days)
        return recent_data['增减'].abs().sum() / recent_data['库存'].mean()
    
    def calculate_trend_strength(self, df: pd.DataFrame, window: int = 30) -> float:
        """计算趋势强度"""
        try:
            if len(df) < window:
                return 0.0
            
            price_change = df['库存'].diff().dropna()
            if len(price_change) < window:
                return 0.0
            
            positive_moves = price_change[price_change > 0].rolling(window=window).sum()
            negative_moves = price_change[price_change < 0].rolling(window=window).sum()
            
            if positive_moves.empty or negative_moves.empty:
                return 0.0
            
            last_positive = positive_moves.iloc[-1] if not pd.isna(positive_moves.iloc[-1]) else 0
            last_negative = negative_moves.iloc[-1] if not pd.isna(negative_moves.iloc[-1]) else 0
            
            total_moves = last_positive + abs(last_negative)
            if total_moves == 0:
                return 0.0
            
            return abs(last_positive - abs(last_negative)) / total_moves
        
        except Exception:
            return 0.0
    
    def calculate_dynamic_threshold(self, df: pd.DataFrame, window: int = 60) -> float:
        """计算动态阈值"""
        try:
            volatility = df['增减'].rolling(window=window).std().iloc[-1]
            return volatility * stats.norm.ppf(self.confidence_level)
        except:
            return 0.0
    
    def analyze_inventory_trend(self, df: pd.DataFrame, category: str) -> Dict:
        """综合分析库存趋势"""
        try:
            recent_data = df.tail(30)
            total_change = recent_data['增减'].sum()
            avg_change = total_change / len(recent_data)
            
            start_inventory = recent_data['库存'].iloc[0]
            end_inventory = recent_data['库存'].iloc[-1]
            if start_inventory > 0:
                change_rate = (end_inventory - start_inventory) / start_inventory * 100
            else:
                change_rate = 0
            
            change_rate = min(max(change_rate, -100), 100)
            
            seasonal_factor = self.calculate_seasonal_factor(df, category)
            inventory_velocity = self.calculate_inventory_velocity(df)
            trend_strength = self.calculate_trend_strength(df)
            dynamic_threshold = self.calculate_dynamic_threshold(df)
            
            trend = '稳定'
            if change_rate > 10 and avg_change > 0 and trend_strength > 0.3:
                trend = '累库'
            elif change_rate < -10 and avg_change < 0 and trend_strength > 0.3:
                trend = '去库'
            
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
        except Exception:
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

@st.cache_data(ttl=3600)  # 缓存1小时
def get_futures_inventory_data(symbols_list):
    """获取期货库存数据"""
    data_dict = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols_list):
        try:
            status_text.text(f"正在获取 {symbol} 的库存数据... ({i+1}/{len(symbols_list)})")
            df = ak.futures_inventory_em(symbol=symbol)
            
            if df is not None and not df.empty and '日期' in df.columns and '库存' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'])
                df['库存'] = pd.to_numeric(df['库存'], errors='coerce')
                df = df.dropna(subset=['日期', '库存'])
                
                if len(df) >= 2:
                    df['增减'] = df['库存'].diff()
                    df = df.dropna(subset=['增减'])
                    
                    if len(df) >= 30:
                        data_dict[symbol] = df
            
            progress_bar.progress((i + 1) / len(symbols_list))
            time.sleep(0.1)  # 避免请求过快
            
        except Exception as e:
            st.warning(f"获取 {symbol} 数据失败: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    return data_dict

def create_plotly_trend_chart(df, symbol, analysis_result):
    """创建Plotly交互式趋势图"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} 库存趋势', '库存增减'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # 库存趋势线
    fig.add_trace(
        go.Scatter(
            x=df['日期'],
            y=df['库存'],
            mode='lines',
            name='库存',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # 30日移动平均
    ma30 = df['库存'].rolling(window=30).mean()
    fig.add_trace(
        go.Scatter(
            x=df['日期'],
            y=ma30,
            mode='lines',
            name='30日均线',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    # 增减柱状图
    colors = ['green' if x > 0 else 'red' for x in df['增减']]
    fig.add_trace(
        go.Bar(
            x=df['日期'],
            y=df['增减'],
            name='日增减',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=600,
        title=f"{symbol} 库存分析 - 趋势: {analysis_result['趋势']} | 变化率: {analysis_result['变化率']:.1f}%",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="库存量", row=1, col=1)
    fig.update_yaxes(title_text="增减量", row=2, col=1)
    
    return fig

def create_summary_charts(results_df, inventory_trends):
    """创建汇总图表"""
    col1, col2 = st.columns(2)
    
    with col1:
        # 趋势分布饼图
        trend_counts = [
            len(inventory_trends['累库品种']),
            len(inventory_trends['去库品种']),
            len(inventory_trends['库存稳定品种'])
        ]
        labels = ['累库品种', '去库品种', '稳定品种']
        colors = ['green', 'red', 'gray']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=trend_counts,
            marker_colors=colors,
            textinfo='label+percent'
        )])
        fig_pie.update_layout(title="库存趋势分布")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 信号强度分布
        fig_hist = px.histogram(
            results_df,
            x='信号强度',
            nbins=20,
            title='信号强度分布',
            color_discrete_sequence=['skyblue']
        )
        st.plotly_chart(fig_hist, use_container_width=True)

def main():
    st.title("📊 期货库存分析系统")
    st.markdown("---")
    
    # 侧边栏配置
    st.sidebar.header("分析配置")
    
    # 期货品种选择
    all_symbols = [
        "沪铜", "镍", "锡", "沪铝", "苯乙烯", "液化石油气", "低硫燃料油", "棉纱",
        "不锈钢", "短纤", "沪铅", "多晶硅", "丁二烯橡胶", "沪锌", "硅铁", "鸡蛋",
        "瓶片", "工业硅", "沥青", "20号胶", "原木", "豆一", "玉米", "燃油",
        "菜籽", "碳酸锂", "纸浆", "玉米淀粉", "沪银", "沪金", "塑料", "聚丙烯",
        "铁矿石", "豆二", "豆粕", "棕榈", "玻璃", "豆油", "橡胶", "烧碱",
        "菜粕", "PTA", "纯碱", "对二甲苯", "菜油", "生猪", "尿素", "PVC",
        "乙二醇", "氧化铝", "焦炭", "郑棉", "甲醇", "白糖", "锰硅", "焦煤",
        "红枣", "螺纹钢", "花生", "苹果", "热卷"
    ]
    
    analysis_mode = st.sidebar.radio(
        "选择分析模式",
        ["全品种分析", "自定义品种分析", "单品种详细分析"]
    )
    
    if analysis_mode == "全品种分析":
        selected_symbols = all_symbols
    elif analysis_mode == "自定义品种分析":
        selected_symbols = st.sidebar.multiselect(
            "选择要分析的品种",
            all_symbols,
            default=all_symbols[:10]
        )
    else:  # 单品种详细分析
        selected_symbols = [st.sidebar.selectbox("选择品种", all_symbols)]
    
    # 分析参数
    st.sidebar.subheader("分析参数")
    confidence_level = st.sidebar.slider("置信水平", 0.90, 0.99, 0.95, 0.01)
    change_threshold = st.sidebar.slider("变化率阈值 (%)", 5, 20, 10, 1)
    trend_threshold = st.sidebar.slider("趋势强度阈值", 0.1, 0.8, 0.3, 0.1)
    
    # 开始分析按钮
    if st.sidebar.button("🚀 开始分析", type="primary"):
        if not selected_symbols:
            st.error("请至少选择一个品种进行分析！")
            return
        
        st.info(f"正在分析 {len(selected_symbols)} 个品种...")
        
        # 获取数据
        data_dict = get_futures_inventory_data(selected_symbols)
        
        if not data_dict:
            st.error("未获取到任何有效数据，请检查网络连接或稍后重试。")
            return
        
        st.success(f"成功获取 {len(data_dict)} 个品种的数据")
        
        # 分析数据
        analyzer = FuturesInventoryAnalyzer(confidence_level)
        results = []
        inventory_trends = {'累库品种': [], '去库品种': [], '库存稳定品种': []}
        
        for symbol, df in data_dict.items():
            try:
                category = get_futures_category(symbol)
                analysis = analyzer.analyze_inventory_trend(df, category)
                
                results.append({
                    '品种': symbol,
                    '分类': category,
                    **analysis
                })
                
                trend = analysis['趋势']
                if trend == '累库':
                    inventory_trends['累库品种'].append(symbol)
                elif trend == '去库':
                    inventory_trends['去库品种'].append(symbol)
                else:
                    inventory_trends['库存稳定品种'].append(symbol)
                    
            except Exception as e:
                st.warning(f"分析 {symbol} 时出错: {str(e)}")
        
        if not results:
            st.error("分析失败，未生成任何结果。")
            return
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('信号强度', ascending=False)
        
        # 显示结果
        st.header("📈 分析结果")
        
        # 汇总统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总品种数", len(results_df))
        with col2:
            st.metric("累库品种", len(inventory_trends['累库品种']), 
                     delta=f"{len(inventory_trends['累库品种'])/len(results_df)*100:.1f}%")
        with col3:
            st.metric("去库品种", len(inventory_trends['去库品种']),
                     delta=f"{len(inventory_trends['去库品种'])/len(results_df)*100:.1f}%")
        with col4:
            st.metric("稳定品种", len(inventory_trends['库存稳定品种']),
                     delta=f"{len(inventory_trends['库存稳定品种'])/len(results_df)*100:.1f}%")
        
        # 汇总图表
        st.subheader("📊 汇总图表")
        create_summary_charts(results_df, inventory_trends)
        
        # 详细结果表格
        st.subheader("📋 详细分析结果")
        
        # 筛选选项
        trend_filter = st.selectbox(
            "筛选趋势类型",
            ["全部", "累库", "去库", "稳定"]
        )
        
        if trend_filter != "全部":
            filtered_df = results_df[results_df['趋势'] == trend_filter]
        else:
            filtered_df = results_df
        
        # 格式化显示
        display_df = filtered_df.copy()
        display_df['变化率'] = display_df['变化率'].apply(lambda x: f"{x:.2f}%")
        display_df['信号强度'] = display_df['信号强度'].apply(lambda x: f"{x:.3f}")
        display_df['趋势强度'] = display_df['趋势强度'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # 重点品种分析
        if analysis_mode == "单品种详细分析" and selected_symbols[0] in data_dict:
            st.subheader(f"🔍 {selected_symbols[0]} 详细分析")
            
            symbol = selected_symbols[0]
            df = data_dict[symbol]
            analysis_result = results_df[results_df['品种'] == symbol].iloc[0].to_dict()
            
            # 创建交互式图表
            fig = create_plotly_trend_chart(df, symbol, analysis_result)
            st.plotly_chart(fig, use_container_width=True)
            
            # 关键指标
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("库存变化率", f"{analysis_result['变化率']:.2f}%")
                st.metric("趋势强度", f"{analysis_result['趋势强度']:.3f}")
            with col2:
                st.metric("信号强度", f"{analysis_result['信号强度']:.3f}")
                st.metric("库存周转率", f"{analysis_result['库存周转率']:.3f}")
            with col3:
                st.metric("平均日变化", f"{analysis_result['平均日变化']:.2f}")
                st.metric("动态阈值", f"{analysis_result['动态阈值']:.2f}")
        
        # 重点关注品种
        if inventory_trends['累库品种'] or inventory_trends['去库品种']:
            st.subheader("⚠️ 重点关注品种")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if inventory_trends['累库品种']:
                    st.markdown("**🟢 累库品种 TOP5**")
                    top_accumulation = results_df[results_df['趋势'] == '累库'].head(5)
                    for _, row in top_accumulation.iterrows():
                        st.write(f"• {row['品种']}: {row['变化率']:.1f}% (信号强度: {row['信号强度']:.2f})")
            
            with col2:
                if inventory_trends['去库品种']:
                    st.markdown("**🔴 去库品种 TOP5**")
                    top_depletion = results_df[results_df['趋势'] == '去库'].head(5)
                    for _, row in top_depletion.iterrows():
                        st.write(f"• {row['品种']}: {row['变化率']:.1f}% (信号强度: {row['信号强度']:.2f})")
        
        # 下载结果
        st.subheader("💾 下载结果")
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="下载分析结果 (CSV)",
            data=csv,
            file_name=f"期货库存分析结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main() 