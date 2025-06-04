#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货基本面综合分析系统
Futures Fundamental Analysis System

作者: 7haoge
邮箱: 953534947@qq.com
创建时间: 2025.06


系统功能:
- 期货库存分析：基于库存变化趋势判断价格方向
- 期货基差分析：基于现货期货价差进行统计套利
- 综合信号分析：多维度信号共振提高投资可靠性

技术栈:
- Streamlit: Web应用框架
- AKShare: 金融数据接口
- Pandas/Numpy: 数据处理
- Plotly: 交互式图表
- Scipy: 统计分析

免责声明:
本系统仅供学习和研究使用，不构成投资建议。
投资有风险，决策需谨慎。
"""

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
import zipfile
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import LineChart, Reference
import tempfile
import os
from dataclasses import dataclass
import concurrent.futures
import pickle
import hashlib
import json
from pathlib import Path

# ==================== 页面配置 - 必须在最开始 ====================

st.set_page_config(
    page_title="期货基本面综合分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 缓存和性能优化系统 ====================

class CacheManager:
    """高级缓存管理器"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.cache_ttl = {
            'inventory_data': 3600,  # 库存数据缓存1小时
            'basis_data': 1800,      # 基差数据缓存30分钟
            'price_data': 1800,      # 价格数据缓存30分钟
            'analysis_results': 7200  # 分析结果缓存2小时
        }
    
    def _get_cache_key(self, data_type: str, **kwargs) -> str:
        """生成缓存键"""
        key_data = f"{data_type}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, data_type: str, **kwargs):
        """获取缓存数据"""
        cache_key = self._get_cache_key(data_type, **kwargs)
        
        # 先检查内存缓存
        if cache_key in self.memory_cache:
            data, timestamp = self.memory_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl.get(data_type, 3600):
                return data
            else:
                del self.memory_cache[cache_key]
        
        # 检查磁盘缓存
        cache_file = self._get_cache_file(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data, timestamp = pickle.load(f)
                
                if time.time() - timestamp < self.cache_ttl.get(data_type, 3600):
                    # 加载到内存缓存
                    self.memory_cache[cache_key] = (data, timestamp)
                    return data
                else:
                    cache_file.unlink()  # 删除过期文件
            except Exception:
                cache_file.unlink()  # 删除损坏文件
        
        return None
    
    def set(self, data_type: str, data, **kwargs):
        """设置缓存数据"""
        cache_key = self._get_cache_key(data_type, **kwargs)
        timestamp = time.time()
        
        # 保存到内存缓存
        self.memory_cache[cache_key] = (data, timestamp)
        
        # 保存到磁盘缓存
        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((data, timestamp), f)
        except Exception as e:
            st.warning(f"缓存保存失败: {e}")
    
    def clear_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        
        # 清理内存缓存
        expired_keys = []
        for key, (data, timestamp) in self.memory_cache.items():
            if current_time - timestamp > max(self.cache_ttl.values()):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # 清理磁盘缓存
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    data, timestamp = pickle.load(f)
                if current_time - timestamp > max(self.cache_ttl.values()):
                    cache_file.unlink()
            except Exception:
                cache_file.unlink()

# 全局缓存管理器
@st.cache_resource
def get_cache_manager():
    return CacheManager()

cache_manager = get_cache_manager()

# ==================== 辅助函数 ====================

def get_analysis_id(analysis_type: str, **kwargs) -> str:
    """生成分析ID用于缓存"""
    key_parts = [analysis_type]
    for key, value in sorted(kwargs.items()):
        if isinstance(value, list):
            key_parts.append(f"{key}_{hash(tuple(value))}")
        else:
            key_parts.append(f"{key}_{value}")
    return "_".join(str(part) for part in key_parts)

def get_cached_analysis_result(analysis_id: str):
    """获取缓存的分析结果"""
    return cache_manager.get(f"analysis_{analysis_id}")

def cache_analysis_result(analysis_id: str, result):
    """缓存分析结果"""
    cache_manager.set(f"analysis_{analysis_id}", result, ttl=3600)  # 1小时过期

# ==================== 缓存装饰器函数 ====================

@st.cache_data(ttl=1800)  # 30分钟缓存
def cached_futures_inventory_em(symbol: str) -> Optional[pd.DataFrame]:
    """缓存的期货库存数据获取"""
    try:
        df = ak.futures_inventory_em(symbol=symbol)
        if df is not None and not df.empty:
            return df
        return None
    except Exception:
        return None

@st.cache_data(ttl=1800)  # 30分钟缓存  
def cached_futures_hist_em(symbol: str) -> Optional[pd.DataFrame]:
    """缓存的期货历史数据获取"""
    try:
        # 使用 futures_main_sina 接口，返回的列名包含 '收盘价'
        df = ak.futures_main_sina(symbol=symbol)
        if df is not None and not df.empty:
            # 处理日期列
            if '日期' in df.columns:
                # 如果已经有日期列，确保格式正确
                df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            else:
                # 如果没有日期列，检查索引是否是日期
                if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
                    df['日期'] = pd.to_datetime(df.index)
                else:
                    # 尝试从其他可能的列名获取日期
                    date_columns = ['date', 'Date', 'DATE', 'time', 'Time']
                    date_col_found = False
                    for col in date_columns:
                        if col in df.columns:
                            df['日期'] = pd.to_datetime(df[col], errors='coerce')
                            date_col_found = True
                            break
                    
                    if not date_col_found:
                        # 如果找不到日期列，使用索引作为日期
                        df['日期'] = pd.to_datetime(df.index, errors='coerce')
            
            # 统一收盘价列名
            if '收盘价' in df.columns:
                df['收盘'] = df['收盘价']
            elif 'close' in df.columns:
                df['收盘'] = df['close']
            elif 'Close' in df.columns:
                df['收盘'] = df['Close']
            elif 'current_price' in df.columns:
                df['收盘'] = df['current_price']
            
            # 过滤掉无效的日期
            df = df.dropna(subset=['日期'])
            
            # 确保日期列是datetime类型且有效
            if len(df) > 0 and df['日期'].dtype == 'datetime64[ns]':
                # 过滤掉异常日期（如1970年的数据）
                df = df[df['日期'] > pd.Timestamp('2000-01-01')]
                
            return df if len(df) > 0 else None
        return None
    except Exception as e:
        print(f"获取价格数据失败: {e}")
        return None

@st.cache_data(ttl=3600)  # 1小时缓存
def cached_futures_spot_price(symbol: str) -> Optional[pd.DataFrame]:
    """缓存的现货价格数据获取"""
    try:
        df = ak.futures_spot_price_em(symbol=symbol)
        if df is not None and not df.empty:
            return df
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600)  # 1小时缓存
def cached_futures_basis_analysis(symbol: str, end_date: str, days_back: int = 30) -> Optional[pd.DataFrame]:
    """缓存的基差分析数据"""
    try:
        # 这里应该调用基差分析的具体实现
        # 由于akshare可能没有直接的基差接口，这里返回None
        # 实际实现中需要通过现货价格和期货价格计算基差
        return None
    except Exception:
        return None

# ==================== 数据获取优化 ====================

@st.cache_data(ttl=3600, max_entries=1000, show_spinner=False)
def cached_futures_spot_price_daily(start_day, end_day, vars_list):
    """优化的基差数据获取"""
    # 先尝试从高级缓存获取
    cache_key = f"{start_day}_{end_day}_{','.join(vars_list)}"
    cached_data = cache_manager.get('basis_data', cache_key=cache_key)
    if cached_data is not None:
        return cached_data
    
    try:
        data = ak.futures_spot_price_daily(
            start_day=start_day,
            end_day=end_day,
            vars_list=vars_list
        )
        if data is not None and not data.empty:
            # 保存到高级缓存
            cache_manager.set('basis_data', data, cache_key=cache_key)
        return data
    except Exception:
        return None

# 并行数据获取
def get_multiple_data_parallel(symbols: List[str], data_type: str = 'inventory', max_workers: int = 8):
    """并行获取多个品种的数据"""
    results = {}
    
    def fetch_data(symbol):
        try:
            if data_type == 'inventory':
                return symbol, cached_futures_inventory_em(symbol)
            elif data_type == 'price':
                return symbol, cached_futures_hist_em(f"{symbol}主连")
            else:
                return symbol, None
        except Exception:
            return symbol, None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(fetch_data, symbol): symbol for symbol in symbols}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol, data = future.result()
            if data is not None and not data.empty:
                results[symbol] = data
    
    return results

# ==================== 会话状态管理 ====================

def init_session_state():
    """初始化会话状态"""
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}
    
    if 'current_analysis_id' not in st.session_state:
        st.session_state.current_analysis_id = None
    
    if 'inventory_results' not in st.session_state:
        st.session_state.inventory_results = None
    
    if 'basis_results' not in st.session_state:
        st.session_state.basis_results = None
    
    if 'show_charts' not in st.session_state:
        st.session_state.show_charts = {}

# ==================== 报告导出系统 ====================

class ReportExporter:
    """报告导出器"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "futures_analysis"
        self.temp_dir.mkdir(exist_ok=True)
    
    def export_inventory_excel(self, results_df: pd.DataFrame, inventory_trends: Dict, data_dict: Dict) -> Path:
        """导出库存分析Excel报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.temp_dir / f"库存分析报告_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 分析结果
            results_df.to_excel(writer, sheet_name='分析结果', index=False)
            
            # 趋势汇总
            trend_summary = pd.DataFrame({
                '趋势类型': ['累库品种', '去库品种', '稳定品种'],
                '品种数量': [len(inventory_trends['累库品种']), 
                           len(inventory_trends['去库品种']), 
                           len(inventory_trends['库存稳定品种'])],
                '品种列表': [', '.join(inventory_trends['累库品种']),
                           ', '.join(inventory_trends['去库品种']),
                           ', '.join(inventory_trends['库存稳定品种'])]
            })
            trend_summary.to_excel(writer, sheet_name='趋势汇总', index=False)
            
            # 原始数据（前5个品种）
            for i, (symbol, df) in enumerate(list(data_dict.items())[:5]):
                df.to_excel(writer, sheet_name=f'{symbol}_数据', index=False)
        
        return filepath
    
    def export_basis_excel(self, opportunities: List, analysis_stats: Dict) -> Path:
        """导出基差分析Excel报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.temp_dir / f"基差分析报告_{timestamp}.xlsx"
        
        # 创建结果数据
        results_data = []
        for opp in opportunities:
            results_data.append({
                '品种': opp.name,
                '代码': opp.variety,
                '机会类型': opp.opportunity_type,
                '置信度(%)': opp.confidence,
                '预期收益(%)': opp.expected_return,
                '风险等级': opp.risk_level,
                '建议持仓(天)': opp.holding_period,
                'Z-Score': opp.z_score,
                '当前基差': opp.current_basis,
                '历史均值': opp.basis_mean,
                '历史标准差': opp.basis_std,
                '百分位数': opp.percentile
            })
        
        results_df = pd.DataFrame(results_data)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 分析结果
            results_df.to_excel(writer, sheet_name='投资机会', index=False)
            
            # 统计信息
            stats_df = pd.DataFrame({
                '统计项目': ['成功获取数据', '获取失败', '检测到信号', '总机会数'],
                '数量': [len(analysis_stats.get('successful_varieties', [])),
                        len(analysis_stats.get('failed_varieties', [])),
                        len(analysis_stats.get('analyzed_varieties', [])),
                        len(opportunities)]
            })
            stats_df.to_excel(writer, sheet_name='分析统计', index=False)
        
        return filepath
    
    def create_comprehensive_report(self, inventory_results: Tuple, basis_results: Tuple) -> Path:
        """创建综合分析报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.temp_dir / f"综合分析报告_{timestamp}.xlsx"
        
        results_df, inventory_trends, data_dict = inventory_results
        opportunities, strategy = basis_results
        
        # 信号共振分析
        buy_basis_symbols = [opp.variety for opp in opportunities if '买基差' in opp.opportunity_type]
        sell_basis_symbols = [opp.variety for opp in opportunities if '卖基差' in opp.opportunity_type]
        
        short_resonance = set(inventory_trends['累库品种']) & set(buy_basis_symbols)
        long_resonance = set(inventory_trends['去库品种']) & set(sell_basis_symbols)
        conflict_1 = set(inventory_trends['累库品种']) & set(sell_basis_symbols)
        conflict_2 = set(inventory_trends['去库品种']) & set(buy_basis_symbols)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 库存分析结果
            results_df.to_excel(writer, sheet_name='库存分析', index=False)
            
            # 基差分析结果
            basis_data = []
            for opp in opportunities:
                basis_data.append({
                    '品种': opp.name,
                    '代码': opp.variety,
                    '机会类型': opp.opportunity_type,
                    '置信度(%)': opp.confidence,
                    '预期收益(%)': opp.expected_return,
                    '风险等级': opp.risk_level
                })
            basis_df = pd.DataFrame(basis_data)
            basis_df.to_excel(writer, sheet_name='基差分析', index=False)
            
            # 信号共振分析
            resonance_data = []
            
            # 做空共振
            for symbol in short_resonance:
                resonance_data.append({
                    '品种': symbol,
                    '信号类型': '做空共振',
                    '库存信号': '累库',
                    '基差信号': '买基差',
                    '投资建议': '看空，考虑做空操作'
                })
            
            # 做多共振
            for symbol in long_resonance:
                resonance_data.append({
                    '品种': symbol,
                    '信号类型': '做多共振',
                    '库存信号': '去库',
                    '基差信号': '卖基差',
                    '投资建议': '看多，考虑做多操作'
                })
            
            # 信号冲突
            for symbol in conflict_1:
                resonance_data.append({
                    '品种': symbol,
                    '信号类型': '信号冲突',
                    '库存信号': '累库',
                    '基差信号': '卖基差',
                    '投资建议': '观望或深入分析'
                })
            
            for symbol in conflict_2:
                resonance_data.append({
                    '品种': symbol,
                    '信号类型': '信号冲突',
                    '库存信号': '去库',
                    '基差信号': '买基差',
                    '投资建议': '观望或深入分析'
                })
            
            if resonance_data:
                resonance_df = pd.DataFrame(resonance_data)
                resonance_df.to_excel(writer, sheet_name='信号共振分析', index=False)
            
            # 综合摘要
            total_analyzed = len(set(inventory_trends['累库品种'] + inventory_trends['去库品种']) | 
                               set(buy_basis_symbols + sell_basis_symbols))
            resonance_rate = (len(short_resonance) + len(long_resonance)) / max(total_analyzed, 1) * 100
            
            summary_data = [{
                '分析时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '库存分析品种数': len(results_df),
                '基差分析机会数': len(opportunities),
                '做空信号共振': len(short_resonance),
                '做多信号共振': len(long_resonance),
                '信号冲突品种': len(conflict_1) + len(conflict_2),
                '共振率(%)': f"{resonance_rate:.1f}%"
            }]
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='综合摘要', index=False)
        
        return filepath

def get_report_exporter() -> ReportExporter:
    """获取报告导出器实例"""
    return ReportExporter()


# ==================== 基差分析相关类和函数 ====================

@dataclass
class BasisOpportunity:
    """基差投资机会数据类"""
    variety: str
    name: str
    current_basis: float
    basis_mean: float
    basis_std: float
    z_score: float
    percentile: float
    opportunity_type: str
    confidence: float
    risk_level: str
    expected_return: float
    holding_period: int

class FuturesBasisStrategy:
    """期货基差投资策略分析系统"""
    
    def __init__(self):
        self.contracts = None
        self.opportunities = []
        self.analysis_results = {}
        
    def get_main_contracts(self) -> pd.DataFrame:
        """获取所有主力合约品种信息"""
        # 先尝试从缓存获取
        cached_contracts = cache_manager.get('contracts_data')
        if cached_contracts is not None:
            self.contracts = cached_contracts
            return cached_contracts
        
        try:
            contract_name = ak.futures_display_main_sina()
            contract_name['symbol'] = contract_name['symbol'].str.replace('0', '')
            contracts = contract_name[['symbol', 'name']]
            
            # 保存到缓存
            cache_manager.set('contracts_data', contracts)
            self.contracts = contracts
            return contracts
        except Exception as e:
            st.error(f"获取合约信息失败: {e}")
            return pd.DataFrame()
    
    def get_basis_data(self, variety: str, start_day: str, end_day: str) -> Optional[pd.DataFrame]:
        """获取并处理基差数据"""
        try:
            df = cached_futures_spot_price_daily(
                start_day=start_day,
                end_day=end_day,
                vars_list=[variety]
            )
            
            if df is None or df.empty:
                return None
            
            # 数据预处理
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['basis'] = df['spot_price'] - df['dominant_contract_price']
            df['basis_rate'] = df['basis'] / df['spot_price'] * 100
            
            # 添加技术指标
            df = self._add_technical_indicators(df)
            
            return df.sort_values('date')
            
        except Exception as e:
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术分析指标"""
        # 基差移动平均
        df['basis_ma5'] = df['basis'].rolling(window=5, min_periods=1).mean()
        df['basis_ma10'] = df['basis'].rolling(window=10, min_periods=1).mean()
        
        # 基差布林带
        rolling_std = df['basis'].rolling(window=10, min_periods=1).std()
        df['basis_upper'] = df['basis_ma10'] + 2 * rolling_std
        df['basis_lower'] = df['basis_ma10'] - 2 * rolling_std
        
        # 基差RSI
        df['basis_rsi'] = self._calculate_rsi(df['basis'])
        
        # 价格动量
        df['price_momentum'] = df['spot_price'].pct_change(5) * 100
        df['futures_momentum'] = df['dominant_contract_price'].pct_change(5) * 100
        
        return df
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def analyze_basis_opportunity(self, df: pd.DataFrame, variety: str, name: str) -> Optional[BasisOpportunity]:
        """分析基差投资机会"""
        if df is None or len(df) < 10:
            return None
        
        # 基础统计指标
        basis_mean = df['basis'].mean()
        basis_std = df['basis'].std()
        current_basis = df['basis'].iloc[-1]
        
        if basis_std == 0:
            return None
        
        # 标准化基差（Z-score）
        z_score = (current_basis - basis_mean) / basis_std
        
        # 基差分位数
        basis_percentile = (df['basis'] <= current_basis).mean() * 100
        
        # 趋势分析
        recent_trend = df['basis'].tail(5).mean() - df['basis'].head(5).mean()
        volatility = df['basis'].std() / abs(df['basis'].mean()) if df['basis'].mean() != 0 else float('inf')
        
        # 技术指标分析
        current_rsi = df['basis_rsi'].iloc[-1] if not pd.isna(df['basis_rsi'].iloc[-1]) else 50
        
        # 布林带位置
        current_upper = df['basis_upper'].iloc[-1]
        current_lower = df['basis_lower'].iloc[-1]
        bb_position = (current_basis - current_lower) / (current_upper - current_lower) if (current_upper - current_lower) != 0 else 0.5
        
        # 投资机会识别
        opportunity_type, confidence, expected_return, holding_period = self._identify_opportunity(
            z_score, basis_percentile, recent_trend, current_rsi, bb_position, volatility
        )
        
        if opportunity_type == "无明显机会":
            return None
        
        # 风险评估
        risk_level = self._assess_risk(volatility, abs(z_score), df)
        
        return BasisOpportunity(
            variety=variety,
            name=name,
            current_basis=current_basis,
            basis_mean=basis_mean,
            basis_std=basis_std,
            z_score=z_score,
            percentile=basis_percentile,
            opportunity_type=opportunity_type,
            confidence=confidence,
            risk_level=risk_level,
            expected_return=expected_return,
            holding_period=holding_period
        )
    
    def _identify_opportunity(self, z_score: float, percentile: float, trend: float, 
                            rsi: float, bb_position: float, volatility: float) -> Tuple[str, float, float, int]:
        """识别投资机会"""
        
        # 基差异常程度评分
        extreme_score = abs(z_score)
        
        # 趋势反转信号
        reversal_signal = 0
        if z_score < -1.2 and trend > 0:
            reversal_signal += 1
        if z_score > 1.2 and trend < 0:
            reversal_signal += 1
        
        # RSI超买超卖信号
        rsi_signal = 0
        if rsi < 35:
            rsi_signal = 1
        elif rsi > 65:
            rsi_signal = -1
        
        # 布林带信号
        bb_signal = 0
        if bb_position < 0.25:
            bb_signal = 1
        elif bb_position > 0.75:
            bb_signal = -1
        
        # 综合评分
        if z_score < -1.5:
            opportunity_type = "买基差机会"
            base_confidence = min(extreme_score * 30, 85)
            expected_return = min(abs(z_score) * 2, 8)
            holding_period = max(10, int(20 - extreme_score * 3))
            
        elif z_score > 1.5:
            opportunity_type = "卖基差机会"
            base_confidence = min(extreme_score * 30, 85)
            expected_return = min(abs(z_score) * 2, 8)
            holding_period = max(10, int(20 - extreme_score * 3))
            
        elif abs(z_score) > 1.0:
            if z_score < 0:
                opportunity_type = "买基差机会"
            else:
                opportunity_type = "卖基差机会"
            base_confidence = min(extreme_score * 25, 70)
            expected_return = min(abs(z_score) * 1.5, 5)
            holding_period = max(15, int(25 - extreme_score * 2))
            
        elif abs(z_score) > 0.8:
            if z_score < 0:
                opportunity_type = "弱买基差机会"
            else:
                opportunity_type = "弱卖基差机会"
            base_confidence = min(extreme_score * 20, 50)
            expected_return = min(abs(z_score) * 1.2, 3)
            holding_period = max(20, int(30 - extreme_score * 2))
            
        else:
            opportunity_type = "无明显机会"
            base_confidence = 0
            expected_return = 0
            holding_period = 0
        
        # 调整置信度
        if opportunity_type != "无明显机会":
            confidence_adjustment = reversal_signal * 8 + rsi_signal * 4 + bb_signal * 4
            final_confidence = max(0, min(95, base_confidence + confidence_adjustment))
            
            # 波动率调整
            if volatility > 0.6:
                final_confidence *= 0.85
            elif volatility < 0.15:
                final_confidence *= 1.05
                
            final_confidence = min(95, final_confidence)
        else:
            final_confidence = 0
        
        return opportunity_type, final_confidence, expected_return, holding_period
    
    def _assess_risk(self, volatility: float, z_score_abs: float, df: pd.DataFrame) -> str:
        """评估投资风险等级"""
        risk_score = 0
        
        # 波动率风险
        if volatility > 0.5:
            risk_score += 3
        elif volatility > 0.3:
            risk_score += 2
        elif volatility > 0.15:
            risk_score += 1
        
        # 极端程度风险
        if z_score_abs > 3:
            risk_score += 2
        elif z_score_abs > 2.5:
            risk_score += 1
        
        # 数据质量风险
        if len(df) < 15:
            risk_score += 1
        
        # 价格趋势一致性
        spot_trend = df['spot_price'].iloc[-1] - df['spot_price'].iloc[0]
        futures_trend = df['dominant_contract_price'].iloc[-1] - df['dominant_contract_price'].iloc[0]
        if spot_trend * futures_trend < 0:
            risk_score += 1
        
        if risk_score >= 5:
            return "高风险"
        elif risk_score >= 3:
            return "中风险"
        else:
            return "低风险"
    
    def run_analysis_streamlit(self, end_day: str, days_back: int = 30, min_confidence: float = 50.0, 
                              progress_callback=None) -> List[BasisOpportunity]:
        """运行完整的基差分析（Streamlit版本）"""
        
        # 生成分析ID
        analysis_id = get_analysis_id('basis', end_day=end_day, days_back=days_back, min_confidence=min_confidence)
        
        # 检查缓存
        cached_result = get_cached_analysis_result(analysis_id)
        if cached_result is not None:
            self.opportunities = cached_result['opportunities']
            self.analysis_results = cached_result['analysis_results']
            if hasattr(cached_result, 'analysis_stats'):
                self.analysis_stats = cached_result['analysis_stats']
            return self.opportunities
        
        # 获取合约信息
        if self.contracts is None:
            self.contracts = self.get_main_contracts()
        
        if self.contracts.empty:
            st.error("无法获取合约信息")
            return []
        
        # 计算日期范围
        end_date = datetime.strptime(end_day, "%Y%m%d")
        start_date = end_date - timedelta(days=days_back)
        start_day = start_date.strftime("%Y%m%d")
        
        opportunities = []
        successful_varieties = []
        failed_varieties = []
        analyzed_varieties = []
        
        total_varieties = len(self.contracts)
        
        # 遍历所有品种
        for idx, row in self.contracts.iterrows():
            symbol = row['symbol']
            name = row['name']
            
            if progress_callback:
                progress_callback(idx + 1, total_varieties, f"正在分析 {name} ({symbol})")
            
            # 获取数据
            df = self.get_basis_data(symbol, start_day, end_day)
            
            if df is not None and len(df) >= 10:
                successful_varieties.append({'symbol': symbol, 'name': name, 'data_points': len(df)})
                
                # 分析投资机会
                opportunity = self.analyze_basis_opportunity(df, symbol, name)
                
                if opportunity:
                    analyzed_varieties.append({
                        'symbol': symbol, 
                        'name': name, 
                        'z_score': opportunity.z_score,
                        'confidence': opportunity.confidence,
                        'opportunity_type': opportunity.opportunity_type,
                        'current_basis': opportunity.current_basis,
                        'basis_mean': opportunity.basis_mean
                    })
                    
                    if opportunity.confidence >= min_confidence:
                        opportunities.append(opportunity)
                        self.analysis_results[symbol] = df
            else:
                failed_varieties.append({'symbol': symbol, 'name': name, 'reason': '数据不足' if df is not None else '获取失败'})
            
            # 避免请求过快
            time.sleep(0.05)
        
        # 按置信度排序
        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        self.opportunities = opportunities
        
        # 存储统计信息
        self.analysis_stats = {
            'successful_varieties': successful_varieties,
            'failed_varieties': failed_varieties,
            'analyzed_varieties': analyzed_varieties
        }
        
        # 缓存结果
        result_to_cache = {
            'opportunities': opportunities,
            'analysis_results': self.analysis_results,
            'analysis_stats': self.analysis_stats
        }
        cache_analysis_result(analysis_id, result_to_cache)
        
        return opportunities

# ==================== 库存分析相关类和函数 ====================

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
        try:
            if len(df) < 5:
                return pd.Series([0])
            
            period = self.seasonal_periods.get(category, 12)
            period = min(period, len(df))  # 确保周期不超过数据长度
            
            if period < 2:
                return pd.Series([0])
            
            seasonal = df['库存'].rolling(window=period, min_periods=1).mean()
            
            # 确保返回的Series不包含nan
            if seasonal.empty or seasonal.isna().all():
                return pd.Series([0])
            
            # 用最后一个有效值填充nan
            seasonal = seasonal.fillna(method='ffill').fillna(0)
            
            return seasonal
            
        except Exception as e:
            print(f"计算季节性因子时出错: {e}")
            return pd.Series([0])
    
    def calculate_inventory_velocity(self, df: pd.DataFrame, days: int = 30) -> float:
        """计算库存周转率"""
        try:
            if len(df) < 5:
                return 0.0
            
            recent_data = df.tail(min(days, len(df)))
            
            if len(recent_data) == 0:
                return 0.0
            
            # 计算平均库存
            avg_inventory = recent_data['库存'].mean()
            if pd.isna(avg_inventory) or avg_inventory <= 0:
                return 0.0
            
            # 计算库存变化的绝对值总和
            inventory_changes = recent_data['增减'].abs().sum()
            if pd.isna(inventory_changes):
                return 0.0
            
            # 库存周转率 = 库存变化总量 / 平均库存
            velocity = inventory_changes / avg_inventory
            
            # 确保结果不是nan
            if pd.isna(velocity):
                return 0.0
            
            return max(0, velocity)  # 确保非负
            
        except Exception as e:
            print(f"计算库存周转率时出错: {e}")
            return 0.0
    
    def calculate_trend_strength(self, df: pd.DataFrame, window: int = 30) -> float:
        """计算趋势强度"""
        try:
            if len(df) < window:
                window = max(5, len(df) // 2)  # 调整窗口大小
            
            if len(df) < 5:
                return 0.0
            
            price_change = df['库存'].diff().dropna()
            if len(price_change) < 5:
                return 0.0
            
            # 使用更稳健的方法计算趋势强度
            positive_changes = price_change[price_change > 0]
            negative_changes = price_change[price_change < 0]
            
            if len(positive_changes) == 0 and len(negative_changes) == 0:
                return 0.0
            
            # 计算正负变化的总和
            positive_sum = positive_changes.sum() if len(positive_changes) > 0 else 0
            negative_sum = abs(negative_changes.sum()) if len(negative_changes) > 0 else 0
            
            total_change = positive_sum + negative_sum
            if total_change == 0:
                return 0.0
            
            # 趋势强度 = |正变化 - 负变化| / 总变化
            trend_strength = abs(positive_sum - negative_sum) / total_change
            
            # 确保结果不是nan
            if pd.isna(trend_strength):
                return 0.0
            
            return min(1.0, trend_strength)  # 限制在0-1之间
        
        except Exception as e:
            print(f"计算趋势强度时出错: {e}")
            return 0.0
    
    def calculate_dynamic_threshold(self, df: pd.DataFrame, window: int = 60) -> float:
        """计算动态阈值"""
        try:
            if len(df) < 5:  # 数据太少时返回默认值
                return 5.0
            
            volatility = df['增减'].rolling(window=min(window, len(df))).std().iloc[-1]
            if pd.isna(volatility) or volatility == 0:
                return 5.0
            
            return volatility * stats.norm.ppf(self.confidence_level)
        except:
            return 5.0
    
    def analyze_inventory_trend(self, df: pd.DataFrame, category: str) -> Dict:
        """综合分析库存趋势"""
        try:
            if len(df) < 5:  # 数据太少
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
            
            recent_data = df.tail(min(30, len(df)))
            total_change = recent_data['增减'].sum()
            avg_change = total_change / len(recent_data)
            
            # 优化变化率计算
            start_inventory = recent_data['库存'].iloc[0]
            end_inventory = recent_data['库存'].iloc[-1]
            
            # 更稳健的变化率计算
            if start_inventory > 0:
                change_rate = (end_inventory - start_inventory) / start_inventory * 100
                # 对于极端值进行合理限制
                if abs(change_rate) > 200:
                    # 使用平均库存作为基准重新计算
                    avg_inventory = recent_data['库存'].mean()
                    if avg_inventory > 0:
                        change_rate = (end_inventory - start_inventory) / avg_inventory * 100
                    else:
                        change_rate = 0
            else:
                # 起始库存为0时，使用不同的计算方法
                if end_inventory > 0:
                    change_rate = 100
                else:
                    change_rate = 0
            
            # 限制变化率范围到合理区间
            change_rate = min(max(change_rate, -150), 150)
            
            # 计算其他指标
            seasonal_factor = self.calculate_seasonal_factor(df, category)
            inventory_velocity = self.calculate_inventory_velocity(df)
            trend_strength = self.calculate_trend_strength(df)
            dynamic_threshold = self.calculate_dynamic_threshold(df)
            
            # 趋势判断 - 使用固定的科学阈值
            trend = '稳定'
            if abs(change_rate) > 10:  # 固定10%阈值
                if change_rate > 10 and avg_change > 0:
                    trend = '累库'
                elif change_rate < -10 and avg_change < 0:
                    trend = '去库'
            elif abs(change_rate) > 5:  # 次级阈值5%
                if change_rate > 5 and avg_change > 0 and trend_strength > 0.2:
                    trend = '累库'
                elif change_rate < -5 and avg_change < 0 and trend_strength > 0.2:
                    trend = '去库'
            
            # 信号强度计算 - 修复nan问题
            try:
                if dynamic_threshold > 0:
                    signal_strength = min(abs(change_rate) / max(dynamic_threshold, 1), 1.0)
                else:
                    signal_strength = min(abs(change_rate) / 10, 1.0)  # 使用固定阈值
                
                # 确保信号强度不是nan
                if pd.isna(signal_strength):
                    signal_strength = abs(change_rate) / 100  # 简单计算
                    
                signal_strength = max(0, min(1, signal_strength))  # 限制在0-1之间
            except:
                signal_strength = abs(change_rate) / 100
                signal_strength = max(0, min(1, signal_strength))
            
            return {
                '趋势': trend,
                '变化率': change_rate,
                '平均日变化': avg_change,
                '趋势强度': trend_strength if not pd.isna(trend_strength) else 0,
                '信号强度': signal_strength,
                '库存周转率': inventory_velocity if not pd.isna(inventory_velocity) else 0,
                '季节性因子': seasonal_factor.iloc[-1] if not seasonal_factor.empty and not pd.isna(seasonal_factor.iloc[-1]) else 0,
                '动态阈值': dynamic_threshold
            }
        except Exception as e:
            print(f"分析趋势时出错: {e}")
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
        '农产品': ['豆一', '豆二', '豆粕', '豆油', '玉米', '玉米淀粉', '菜粕', '菜油', '棕榈', '白糖', '棉花', '苹果', '鸡蛋', '生猪', '红枣', '花生'],
        '工业品': ['螺纹钢', '热卷', '铁矿石', '焦煤', '焦炭', '不锈钢', '沪铜', '沪铝', '沪锌', '沪铅', '沪镍', '沪锡', '沪银', '沪金'],
        '能源化工': ['原油', '燃油', '沥青', 'PTA', '甲醇', '乙二醇', 'PVC', 'PP', '塑料', '橡胶', '20号胶', '苯乙烯', '液化石油气', '低硫燃料油']
    }
    
    for category, symbols in categories.items():
        if symbol in symbols:
            return category
    return '其他'

def get_single_inventory_data_streamlit(symbol: str, end_date=None, days_back=30) -> Optional[pd.DataFrame]:
    """获取单个期货品种的库存数据（Streamlit版本）"""
    try:
        df = cached_futures_inventory_em(symbol)
        
        if df is not None and not df.empty and '日期' in df.columns and '库存' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期')
            
            # 如果指定了日期范围，进行筛选
            if end_date is not None:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date).date()
                elif hasattr(end_date, 'date'):
                    end_date = end_date.date()
                
                # 计算开始日期
                start_date = end_date - timedelta(days=days_back)
                
                # 筛选日期范围
                df = df[
                    (df['日期'].dt.date >= start_date) & 
                    (df['日期'].dt.date <= end_date)
                ]
            
            if len(df) > 0:
                df['增减'] = df['库存'].diff().fillna(0)
                return df
            else:
                return None
        else:
            return None
    except Exception:
        return None

def run_inventory_analysis(selected_symbols: List[str], confidence_level: float = 0.95, 
                          progress_callback=None, end_date=None, days_back=30) -> Tuple[pd.DataFrame, Dict, Dict]:
    """运行库存分析"""
    
    # 生成分析ID（包含日期参数）
    analysis_id = get_analysis_id(
        'inventory', 
        symbols=selected_symbols, 
        confidence_level=confidence_level,
        end_date=str(end_date) if end_date else None,
        days_back=days_back
    )
    
    # 检查缓存
    cached_result = get_cached_analysis_result(analysis_id)
    if cached_result is not None:
        return cached_result['results_df'], cached_result['inventory_trends'], cached_result['data_dict']
    
    # 获取数据
    data_dict = {}
    total_symbols = len(selected_symbols)
    
    for i, symbol in enumerate(selected_symbols):
        if progress_callback:
            progress_callback(i + 1, total_symbols, f"正在获取 {symbol} 的库存数据")
        
        df = get_single_inventory_data_streamlit(symbol, end_date, days_back)
        if df is not None:
            data_dict[symbol] = df
        
        time.sleep(0.05)  # 避免请求过快
    
    if not data_dict:
        return pd.DataFrame(), {}, {}
    
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
        return pd.DataFrame(), {}, {}
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('信号强度', ascending=False)
    
    # 缓存结果
    result_to_cache = {
        'results_df': results_df,
        'inventory_trends': inventory_trends,
        'data_dict': data_dict
    }
    cache_analysis_result(analysis_id, result_to_cache)
    
    return results_df, inventory_trends, data_dict

def get_futures_price_data(symbol: str) -> Optional[pd.DataFrame]:
    """获取期货价格数据 - 改进版本，支持多种合约名称格式"""
    
    # 期货合约名称映射表 - 使用正确的格式
    symbol_mapping = {
        '镍': ['NI0', '镍连续', 'NI连续'],
        '沪铜': ['CU0', '沪铜连续', 'CU连续'],
        '锡': ['SN0', '锡连续', 'SN连续'],
        '沪铝': ['AL0', '沪铝连续', 'AL连续'],
        '苯乙烯': ['EB0', '苯乙烯连续', 'EB连续'],
        '液化石油气': ['PG0', '液化石油气连续', 'PG连续'],
        '低硫燃料油': ['LU0', '低硫燃料油连续', 'LU连续'],
        '多晶硅': ['OQ0', '多晶硅连续', 'OQ连续'],
        '硅铁': ['SF0', '硅铁连续', 'SF连续'],
        '原木': ['WO0', '原木连续', 'WO连续'],
        '铁矿石': ['I0', '铁矿石连续', 'I连续'],
        '螺纹钢': ['RB0', '螺纹钢连续', 'RB连续'],
        '热卷': ['HC0', '热卷连续', 'HC连续'],
        '焦煤': ['JM0', '焦煤连续', 'JM连续'],
        '焦炭': ['J0', '焦炭连续', 'J连续'],
        '豆一': ['A0', '豆一连续', 'A连续'],
        '豆二': ['B0', '豆二连续', 'B连续'],
        '豆粕': ['M0', '豆粕连续', 'M连续'],
        '豆油': ['Y0', '豆油连续', 'Y连续'],
        '玉米': ['C0', '玉米连续', 'C连续'],
        '玉米淀粉': ['CS0', '玉米淀粉连续', 'CS连续'],
        '菜粕': ['RM0', '菜粕连续', 'RM连续'],
        '菜油': ['OI0', '菜油连续', 'OI连续'],
        '棕榈': ['P0', '棕榈连续', 'P连续'],
        '白糖': ['SR0', '白糖连续', 'SR连续'],
        '棉花': ['CF0', '棉花连续', 'CF连续'],
        '郑棉': ['CF0', '郑棉连续', 'CF连续'],
        '苹果': ['AP0', '苹果连续', 'AP连续'],
        '鸡蛋': ['JD0', '鸡蛋连续', 'JD连续'],
        '生猪': ['LH0', '生猪连续', 'LH连续'],
        '红枣': ['CJ0', '红枣连续', 'CJ连续'],
        '花生': ['PK0', '花生连续', 'PK连续'],
        'PTA': ['TA0', 'PTA连续', 'TA连续'],
        '甲醇': ['MA0', '甲醇连续', 'MA连续'],
        '乙二醇': ['EG0', '乙二醇连续', 'EG连续'],
        'PVC': ['V0', 'PVC连续', 'V连续'],
        'PP': ['PP0', 'PP连续'],
        '聚丙烯': ['PP0', '聚丙烯连续', 'PP连续'],
        '塑料': ['L0', '塑料连续', 'L连续'],
        '橡胶': ['RU0', '橡胶连续', 'RU连续'],
        '20号胶': ['NR0', '20号胶连续', 'NR连续'],
        '沥青': ['BU0', '沥青连续', 'BU连续'],
        '燃油': ['FU0', '燃油连续', 'FU连续'],
        '原油': ['SC0', '原油连续', 'SC连续'],
        '纯碱': ['SA0', '纯碱连续', 'SA连续'],
        '玻璃': ['FG0', '玻璃连续', 'FG连续'],
        '尿素': ['UR0', '尿素连续', 'UR连续'],
        '短纤': ['PF0', '短纤连续', 'PF连续'],
        '纸浆': ['SP0', '纸浆连续', 'SP连续'],
        '不锈钢': ['SS0', '不锈钢连续', 'SS连续'],
        '沪锌': ['ZN0', '沪锌连续', 'ZN连续'],
        '沪铅': ['PB0', '沪铅连续', 'PB连续'],
        '沪镍': ['NI0', '沪镍连续', 'NI连续'],
        '沪银': ['AG0', '沪银连续', 'AG连续'],
        '沪金': ['AU0', '沪金连续', 'AU连续'],
        '锰硅': ['SM0', '锰硅连续', 'SM连续'],
        '氧化铝': ['AO0', '氧化铝连续', 'AO连续'],
        '碳酸锂': ['LC0', '碳酸锂连续', 'LC连续'],
        '工业硅': ['SI0', '工业硅连续', 'SI连续'],
        '烧碱': ['SH0', '烧碱连续', 'SH连续'],
        '对二甲苯': ['PX0', '对二甲苯连续', 'PX连续'],
        '瓶片': ['BP0', '瓶片连续', 'BP连续'],
        '丁二烯橡胶': ['BR0', '丁二烯橡胶连续', 'BR连续'],
        '棉纱': ['CY0', '棉纱连续', 'CY连续']
    }
    
    # 获取可能的合约名称列表
    possible_names = symbol_mapping.get(symbol, [f"{symbol}0", f"{symbol}连续"])
    
    # 尝试不同的合约名称
    for contract_name in possible_names:
        try:
            df = cached_futures_hist_em(contract_name)
            if df is not None and not df.empty and '收盘' in df.columns:
                return df
        except Exception:
            continue
    
    # 如果所有尝试都失败，返回None
    return None

def align_inventory_and_price_data(inventory_df: pd.DataFrame, price_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """对齐库存和价格数据的时间范围 - 改进版本"""
    # 确保日期列是datetime类型
    if inventory_df['日期'].dtype != 'datetime64[ns]':
        inventory_df['日期'] = pd.to_datetime(inventory_df['日期'])
    if price_df['日期'].dtype != 'datetime64[ns]':
        price_df['日期'] = pd.to_datetime(price_df['日期'])
    
    # 找到时间范围
    inventory_start = inventory_df['日期'].min()
    inventory_end = inventory_df['日期'].max()
    price_start = price_df['日期'].min()
    price_end = price_df['日期'].max()
    
    # 计算重叠范围
    common_start = max(inventory_start, price_start)
    common_end = min(inventory_end, price_end)
    
    # 检查是否有重叠
    if common_start > common_end:
        # 没有重叠，尝试更宽松的匹配
        # 使用较大的时间范围，允许部分数据缺失
        extended_start = min(inventory_start, price_start)
        extended_end = max(inventory_end, price_end)
        
        # 筛选数据 - 使用更宽松的条件
        aligned_inventory = inventory_df[
            (inventory_df['日期'] >= extended_start) & 
            (inventory_df['日期'] <= extended_end)
        ].copy()
        
        aligned_price = price_df[
            (price_df['日期'] >= extended_start) & 
            (price_df['日期'] <= extended_end)
        ].copy()
        
        # 如果还是没有数据，返回原始数据的最近部分
        if len(aligned_inventory) == 0:
            aligned_inventory = inventory_df.tail(min(100, len(inventory_df))).copy()
        if len(aligned_price) == 0:
            aligned_price = price_df.tail(min(100, len(price_df))).copy()
            
    else:
        # 有重叠，使用交集
        aligned_inventory = inventory_df[
            (inventory_df['日期'] >= common_start) & 
            (inventory_df['日期'] <= common_end)
        ].copy()
        
        aligned_price = price_df[
            (price_df['日期'] >= common_start) & 
            (price_df['日期'] <= common_end)
        ].copy()
    
    return aligned_inventory, aligned_price

def create_plotly_trend_chart(df: pd.DataFrame, symbol: str, analysis_result: Dict):
    """创建库存趋势图"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} 库存走势', '库存变化量'),
        vertical_spacing=0.1
    )
    
    # 库存走势
    fig.add_trace(
        go.Scatter(
            x=df['日期'],
            y=df['库存'],
            mode='lines+markers',
            name='库存',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # 库存变化量
    colors = ['red' if x < 0 else 'green' for x in df['增减']]
    fig.add_trace(
        go.Bar(
            x=df['日期'],
            y=df['增减'],
            name='库存变化',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text=f"{symbol} 库存分析 - 趋势: {analysis_result['趋势']}",
        showlegend=True
    )
    
    return fig

def create_plotly_inventory_price_chart(inventory_df: pd.DataFrame, price_df: pd.DataFrame, 
                                       symbol: str, analysis_result: Dict):
    """创建库存价格对比图 - 双Y轴显示"""
    # 创建双Y轴子图
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]],
        subplot_titles=[f'{symbol} 库存与价格走势对比']
    )
    
    # 添加价格走势（主Y轴）
    fig.add_trace(
        go.Scatter(
            x=price_df['日期'],
            y=price_df['收盘'],
            mode='lines',
            name='期货价格',
            line=dict(color='red', width=2),
            yaxis='y'
        ),
        secondary_y=False
    )
    
    # 添加库存走势（次Y轴）
    fig.add_trace(
        go.Scatter(
            x=inventory_df['日期'],
            y=inventory_df['库存'],
            mode='lines+markers',
            name='库存',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            yaxis='y2'
        ),
        secondary_y=True
    )
    
    # 设置Y轴标签
    fig.update_yaxes(title_text="期货价格", secondary_y=False, side="left")
    fig.update_yaxes(title_text="库存", secondary_y=True, side="right")
    
    # 设置X轴标签
    fig.update_xaxes(title_text="日期")
    
    # 更新布局
    fig.update_layout(
        height=600,
        title_text=f"{symbol} 库存与价格走势对比分析 - 趋势: {analysis_result['趋势']}",
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_summary_charts(results_df: pd.DataFrame, inventory_trends: Dict):
    """创建汇总图表"""
    col1, col2 = st.columns(2)
    
    with col1:
        # 趋势分布饼图
        trend_counts = results_df['趋势'].value_counts()
        fig = px.pie(
            values=trend_counts.values,
            names=trend_counts.index,
            title="库存趋势分布",
            color_discrete_map={
                '累库': '#ff6b6b',
                '去库': '#51cf66',
                '稳定': '#74c0fc'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 信号强度分布
        fig = px.histogram(
            results_df,
            x='信号强度',
            title="信号强度分布",
            nbins=20,
            color_discrete_sequence=['#339af0']
        )
        st.plotly_chart(fig, use_container_width=True)

def show_basis_detailed_chart(df, opportunity):
    """显示基差详细分析图表"""
    st.subheader(f"📊 {opportunity.name} 详细分析")
    
    # 创建四个子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('价格走势', '基差分析', '基差分布', '技术指标'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # 1. 价格走势图
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['spot_price'], name='现货价格', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['dominant_contract_price'], name='期货价格', line=dict(color='red')),
        row=1, col=1
    )
    
    # 2. 基差分析
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['basis'], name='基差', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['basis_ma10'], name='基差MA10', line=dict(color='orange', dash='dash')),
        row=1, col=2
    )
    
    # 3. 基差分布直方图
    fig.add_trace(
        go.Histogram(x=df['basis'], name='基差分布', nbinsx=20),
        row=2, col=1
    )
    
    # 4. RSI指标
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['basis_rsi'], name='基差RSI', line=dict(color='purple')),
        row=2, col=2
    )
    
    # 添加RSI参考线
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=2)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=2)
    
    fig.update_layout(height=800, showlegend=True, title_text=f"{opportunity.name} 基差分析详情")
    st.plotly_chart(fig, use_container_width=True)
    
    # 投资建议
    st.subheader("💡 投资建议")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **机会类型**: {opportunity.opportunity_type}
        **置信度**: {opportunity.confidence:.1f}%
        **预期收益**: {opportunity.expected_return:.1f}%
        **建议持仓(天)**: {opportunity.holding_period}天
        """)
    
    with col2:
        st.warning(f"""
        **风险等级**: {opportunity.risk_level}
        **Z-Score**: {opportunity.z_score:.2f}
        **当前基差**: {opportunity.current_basis:.2f}
        **历史均值**: {opportunity.basis_mean:.2f}
        """)
    
    # 操作说明
    if "买基差" in opportunity.opportunity_type:
        st.success("📈 **买基差操作**: 买入现货 + 卖出期货（类似做空期货）")
    else:
        st.error("📉 **卖基差操作**: 卖出现货 + 买入期货（类似做多期货）")

# ==================== 页面函数 ====================

def inventory_analysis_page():
    """库存分析页面"""
    st.header("📈 期货库存分析")
    
    # 参数设置
    st.subheader("🔧 分析参数设置")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
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
        
        analysis_mode = st.selectbox(
            "选择分析模式",
            ["全品种分析", "自定义品种分析", "单品种详细分析"]
        )
    
    with col2:
        # 日期选择
        end_date = st.date_input(
            "分析截止日期", 
            value=datetime.now().date(),
            help="选择库存数据的截止日期"
        )
        
        days_back = st.slider(
            "分析天数", 
            min_value=15, 
            max_value=90, 
            value=30,
            help="从截止日期往前分析的天数"
        )
    
    with col3:
        # 排序方式选择
        sort_method = st.selectbox(
            "排序方式",
            ["信号强度", "变化率绝对值", "趋势强度"],
            help="选择详细分析结果的排序依据"
        )
        
        # 排序说明 - 移到这里
        with st.expander("📊 排序方式说明"):
            st.markdown(f"""
            **当前排序方式：{sort_method}**
            
            - **信号强度**：综合考虑变化率和统计显著性，反映库存变化的可靠程度（推荐）
            - **变化率绝对值**：纯粹按库存变化幅度排序，数值越大变化越明显  
            - **趋势强度**：反映库存变化的持续性和方向性
            
            💡 **建议**：信号强度最适合投资决策，因为它考虑了统计显著性
            """)
        
        # 显示筛选选项
        show_advanced = st.checkbox("显示高级筛选", value=False)
    
    with col4:
        st.info("""
        **默认参数说明**：
        - 累库/去库阈值：10%
        - 趋势强度阈值：0.2
        - 置信水平：95%
        
        这些是经过优化的科学参数
        """)
    
    # 高级筛选选项（可选）
    if show_advanced:
        st.subheader("🔍 高级筛选选项")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_level = st.slider("置信水平", 0.90, 0.99, 0.95, 0.01, 
                                       help="统计学置信水平，用于计算动态阈值")
        with col2:
            change_threshold = st.slider("变化率阈值 (%)", 5, 20, 10, 1,
                                       help="判断累库/去库的最小变化率")
        with col3:
            trend_threshold = st.slider("趋势强度阈值", 0.1, 0.8, 0.2, 0.1,
                                      help="判断趋势可靠性的阈值")
    else:
        # 使用默认的最佳参数
        confidence_level = 0.95
        change_threshold = 10
        trend_threshold = 0.2
    
    # 品种选择
    if analysis_mode == "全品种分析":
        selected_symbols = all_symbols
        st.info(f"将分析全部 {len(all_symbols)} 个品种")
    elif analysis_mode == "自定义品种分析":
        selected_symbols = st.multiselect(
            "选择要分析的品种",
            all_symbols,
            default=all_symbols[:10]
        )
    else:  # 单品种详细分析
        selected_symbols = [st.selectbox("选择品种", all_symbols)]
    
    # 分析按钮
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        start_analysis = st.button("🚀 开始库存分析", type="primary")
    with col2:
        # 检查是否有已有的分析结果
        has_existing_results = st.session_state.get('inventory_results') is not None
        use_existing = st.button("📊 使用已有数据", disabled=not has_existing_results)
    with col3:
        if st.button("🔄 重新分析"):
            # 清除缓存
            if 'inventory_results' in st.session_state:
                del st.session_state.inventory_results
            cache_manager.clear_expired()
            st.rerun()
    with col4:
        if st.button("🗑️ 清除所有缓存"):
            cache_manager.clear_expired()
            st.session_state.clear()
            st.success("缓存已清除")
            st.rerun()
    
    # 显示已有数据信息
    if has_existing_results:
        results_df, inventory_trends, data_dict = st.session_state.inventory_results
        st.info(f"💾 已有分析数据：{len(results_df)}个品种，包含{len(inventory_trends['累库品种'])}个累库品种，{len(inventory_trends['去库品种'])}个去库品种")
    
    # 执行分析
    if start_analysis or use_existing or st.session_state.get('inventory_results') is not None:
        if not selected_symbols:
            st.error("请至少选择一个品种进行分析！")
            return
        
        # 如果是新的分析请求，执行分析
        if start_analysis:
            # 创建进度显示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current, total, message):
                progress_bar.progress(current / total)
                status_text.text(f"{message} [{current}/{total}]")
            
            with st.spinner(f"正在分析 {len(selected_symbols)} 个品种的库存数据..."):
                try:
                    results_df, inventory_trends, data_dict = run_inventory_analysis(
                        selected_symbols, 
                        confidence_level,
                        progress_callback,
                        end_date=end_date,
                        days_back=days_back
                    )
                    
                    if results_df.empty:
                        st.error("未获取到任何有效数据，请检查网络连接或稍后重试。")
                        return
                    
                    # 保存到session state
                    st.session_state.inventory_results = (results_df, inventory_trends, data_dict)
                    
                    # 清除进度显示
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ 成功分析 {len(results_df)} 个品种的数据")
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"分析过程中出错: {str(e)}")
                    return
        elif use_existing:
            st.success("✅ 使用已有分析数据")
        
        # 从session state获取数据
        if st.session_state.get('inventory_results') is None:
            st.warning("没有可用的分析结果，请重新分析。")
            return
            
        results_df, inventory_trends, data_dict = st.session_state.inventory_results
        
        # 检查数据完整性
        if results_df.empty:
            st.error("分析结果为空，请重新分析。")
            return
        
        # 显示结果
        st.header("📈 库存分析结果")
        
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
        
        # 库存分析逻辑和原理说明
        with st.expander("📖 库存分析逻辑与原理"):
            st.markdown("""
            ### 🔍 分析逻辑说明
            
            #### 1. 趋势判断逻辑
            **累库/去库判断标准**：
            - **主要阈值**：变化率 > 10% 且平均日变化 > 0 → 累库
            - **主要阈值**：变化率 < -10% 且平均日变化 < 0 → 去库  
            - **次级阈值**：变化率 > 5% 且趋势强度 > 0.2 → 累库
            - **次级阈值**：变化率 < -5% 且趋势强度 > 0.2 → 去库
            - **其他情况**：库存稳定
            
            #### 2. 关键指标计算
            **变化率**：(期末库存 - 期初库存) / 期初库存 × 100%
            - 反映库存的总体变化幅度
            - 正值表示累库，负值表示去库
            
            **信号强度**：abs(变化率) / max(动态阈值, 固定阈值)
            - 综合考虑变化幅度和统计显著性
            - 值越大表示信号越可靠
            
            **趋势强度**：|正变化总和 - 负变化总和| / 总变化量
            - 反映库存变化的方向一致性
            - 值越大表示趋势越明确
            
            **库存周转率**：库存变化总量 / 平均库存
            - 反映库存的活跃程度
            - 值越大表示库存流动性越强
            
            #### 3. 投资逻辑
            **累库信号** → **看空信号**：
            - 库存增加通常意味着供应过剩或需求不足
            - 可能导致价格下跌压力
            - 建议：考虑做空操作
            
            **去库信号** → **看多信号**：
            - 库存减少通常意味着需求旺盛或供应紧张
            - 可能推动价格上涨
            - 建议：考虑做多操作
            
            #### 4. 信号可靠性评估
            - **信号强度 > 0.5**：高可靠性信号
            - **信号强度 0.2-0.5**：中等可靠性信号  
            - **信号强度 < 0.2**：弱信号，需谨慎
            
            💡 **注意**：库存分析需结合价格走势、基本面等因素综合判断
            """)
        
        # 筛选选项
        col1, col2 = st.columns(2)
        with col1:
            trend_filter = st.selectbox(
                "筛选趋势类型",
                ["全部", "累库", "去库", "稳定"]
            )
        with col2:
            min_signal_strength = st.slider("最小信号强度", 0.0, 1.0, 0.0, 0.1)
        
        # 应用筛选和排序
        filtered_df = results_df.copy()
        if trend_filter != "全部":
            filtered_df = filtered_df[filtered_df['趋势'] == trend_filter]
        filtered_df = filtered_df[filtered_df['信号强度'] >= min_signal_strength]
        
        # 根据选择的排序方式排序
        if sort_method == "信号强度":
            filtered_df = filtered_df.sort_values('信号强度', ascending=False)
        elif sort_method == "变化率绝对值":
            filtered_df = filtered_df.sort_values('变化率', key=abs, ascending=False)
        elif sort_method == "趋势强度":
            filtered_df = filtered_df.sort_values('趋势强度', ascending=False)
        
        # 检查筛选后的数据
        if filtered_df.empty:
            st.warning("筛选条件过于严格，没有符合条件的数据。请调整筛选条件。")
            return
        
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
        
        # 图表展示选项
        st.subheader("📊 图表分析")
        
        # 选择要查看图表的品种
        chart_symbols = st.multiselect(
            "选择要查看图表的品种（最多5个）",
            options=list(data_dict.keys()),
            default=[],
            max_selections=5
        )
        
        if chart_symbols:
            chart_type = st.radio(
                "选择图表类型",
                ["库存走势图", "库存价格对比图"],
                horizontal=True
            )
            
            # 显示图表的开关
            show_charts_key = f"show_inventory_charts_{hash(tuple(chart_symbols))}"
            if show_charts_key not in st.session_state.show_charts:
                st.session_state.show_charts[show_charts_key] = False
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("📈 显示图表", key=f"show_{show_charts_key}"):
                    st.session_state.show_charts[show_charts_key] = True
            with col2:
                if st.button("🔄 隐藏图表", key=f"hide_{show_charts_key}"):
                    st.session_state.show_charts[show_charts_key] = False
            
            # 显示图表
            if st.session_state.show_charts.get(show_charts_key, False):
                for symbol in chart_symbols:
                    if symbol in data_dict:
                        df = data_dict[symbol]
                        analysis_result = results_df[results_df['品种'] == symbol].iloc[0].to_dict()
                        
                        st.subheader(f"📊 {symbol} 详细分析")
                        
                        if chart_type == "库存走势图":
                            # 创建库存趋势图
                            fig = create_plotly_trend_chart(df, symbol, analysis_result)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # 创建库存价格对比图
                            with st.spinner(f"正在获取{symbol}的价格数据..."):
                                price_df = get_futures_price_data(symbol)
                            
                            if price_df is not None:
                                # 显示原始数据时间范围
                                inventory_start = df['日期'].min().date()
                                inventory_end = df['日期'].max().date()
                                price_start = price_df['日期'].min().date()
                                price_end = price_df['日期'].max().date()
                                
                                st.info(f"📅 数据时间范围 - 库存: {inventory_start} 到 {inventory_end} | 价格: {price_start} 到 {price_end}")
                                
                                # 对齐数据时间范围
                                aligned_inventory, aligned_price = align_inventory_and_price_data(df, price_df)
                                
                                if len(aligned_inventory) > 0 and len(aligned_price) > 0:
                                    # 显示对齐后的时间范围
                                    aligned_start = max(aligned_inventory['日期'].min().date(), aligned_price['日期'].min().date())
                                    aligned_end = min(aligned_inventory['日期'].max().date(), aligned_price['日期'].max().date())
                                    st.success(f"✅ 数据对齐成功 - 分析时间范围: {aligned_start} 到 {aligned_end}")
                                    
                                    # 创建库存价格对比图
                                    fig = create_plotly_inventory_price_chart(aligned_inventory, aligned_price, symbol, analysis_result)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 显示数据统计
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("库存数据点", len(aligned_inventory))
                                    with col2:
                                        st.metric("价格数据点", len(aligned_price))
                                else:
                                    st.warning(f"⚠️ {symbol}的数据对齐后为空，显示库存走势图")
                                    fig = create_plotly_trend_chart(df, symbol, analysis_result)
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                # 显示详细的错误信息
                                st.warning(f"⚠️ 无法获取{symbol}的价格数据")
                                
                                # 期货合约名称映射表（简化版，用于显示尝试的合约名称）
                                symbol_mapping = {
                                    '镍': ['NI0', '镍连续', 'NI连续'],
                                    '沪铜': ['CU0', '沪铜连续', 'CU连续'],
                                    '锡': ['SN0', '锡连续', 'SN连续'],
                                    '沪铝': ['AL0', '沪铝连续', 'AL连续'],
                                    '苯乙烯': ['EB0', '苯乙烯连续', 'EB连续'],
                                    '液化石油气': ['PG0', '液化石油气连续', 'PG连续'],
                                    '低硫燃料油': ['LU0', '低硫燃料油连续', 'LU连续'],
                                    '多晶硅': ['OQ0', '多晶硅连续', 'OQ连续'],
                                    '硅铁': ['SF0', '硅铁连续', 'SF连续'],
                                    '原木': ['WO0', '原木连续', 'WO连续']
                                }
                                
                                tried_names = symbol_mapping.get(symbol, [f"{symbol}0", f"{symbol}连续"])
                                st.info(f"💡 已尝试的合约名称: {', '.join(tried_names[:3])}...")
                                st.info("📝 可能的原因: 1) 该品种暂无价格数据 2) 合约名称不匹配 3) 数据源暂时不可用")
                                
                                # 显示库存走势图作为替代
                                st.info("📊 显示库存走势图作为替代:")
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
        
        # 重点关注品种 - 改进排序逻辑
        if inventory_trends['累库品种'] or inventory_trends['去库品种']:
            st.subheader("⚠️ 重点关注品种")
            
            # 根据选择的排序方式对累库和去库品种进行排序
            def get_sorted_symbols(symbol_list, trend_type):
                """根据排序方式对品种列表进行排序"""
                if not symbol_list:
                    return pd.DataFrame()
                
                # 获取这些品种的数据
                trend_df = results_df[results_df['品种'].isin(symbol_list)].copy()
                
                if trend_df.empty:
                    return pd.DataFrame()
                
                # 根据排序方式排序
                if sort_method == "信号强度":
                    trend_df = trend_df.sort_values('信号强度', ascending=False)
                elif sort_method == "变化率绝对值":
                    trend_df = trend_df.sort_values('变化率', key=abs, ascending=False)
                elif sort_method == "趋势强度":
                    trend_df = trend_df.sort_values('趋势强度', ascending=False)
                
                return trend_df
            
            col1, col2 = st.columns(2)
            
            with col1:
                if inventory_trends['累库品种']:
                    st.markdown(f"**🟢 累库品种 (按{sort_method}排序)**")
                    sorted_cumulative = get_sorted_symbols(inventory_trends['累库品种'], '累库')
                    
                    if not sorted_cumulative.empty:
                        for idx, (_, row) in enumerate(sorted_cumulative.head(5).iterrows()):
                            symbol = row['品种']
                            change_rate = row['变化率']
                            signal_strength = row['信号强度']
                            trend_strength = row['趋势强度']
                            
                            if sort_method == "信号强度":
                                main_value = f"信号强度: {signal_strength:.3f}"
                            elif sort_method == "变化率绝对值":
                                main_value = f"变化率: {change_rate:.2f}%"
                            else:
                                main_value = f"趋势强度: {trend_strength:.3f}"
                            
                            st.write(f"{idx+1}. **{symbol}**: {main_value}")
                            st.caption(f"   变化率: {change_rate:.2f}% | 信号强度: {signal_strength:.3f}")
                    else:
                        st.info("暂无累库品种数据")
            
            with col2:
                if inventory_trends['去库品种']:
                    st.markdown(f"**🔴 去库品种 (按{sort_method}排序)**")
                    sorted_depletion = get_sorted_symbols(inventory_trends['去库品种'], '去库')
                    
                    if not sorted_depletion.empty:
                        for idx, (_, row) in enumerate(sorted_depletion.head(5).iterrows()):
                            symbol = row['品种']
                            change_rate = row['变化率']
                            signal_strength = row['信号强度']
                            trend_strength = row['趋势强度']
                            
                            if sort_method == "信号强度":
                                main_value = f"信号强度: {signal_strength:.3f}"
                            elif sort_method == "变化率绝对值":
                                main_value = f"变化率: {change_rate:.2f}%"
                            else:
                                main_value = f"趋势强度: {trend_strength:.3f}"
                            
                            st.write(f"{idx+1}. **{symbol}**: {main_value}")
                            st.caption(f"   变化率: {change_rate:.2f}% | 信号强度: {signal_strength:.3f}")
                    else:
                        st.info("暂无去库品种数据")
        
        # 导出功能
        st.subheader("📥 导出分析结果")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 导出Excel报告"):
                try:
                    exporter = get_report_exporter()
                    filepath = exporter.export_inventory_excel(results_df, inventory_trends, data_dict)
                    
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label="⬇️ 下载Excel报告",
                            data=f.read(),
                            file_name=filepath.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    st.success("Excel报告生成成功！")
                except Exception as e:
                    st.error(f"导出失败: {str(e)}")
        
        with col2:
            if st.button("📈 导出CSV数据"):
                csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="⬇️ 下载CSV数据",
                    data=csv,
                    file_name=f"库存分析结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📋 复制分析摘要"):
                # 安全地获取排序后的品种列表
                def safe_get_sorted_symbols(symbol_list):
                    if not symbol_list:
                        return []
                    trend_df = results_df[results_df['品种'].isin(symbol_list)].copy()
                    if trend_df.empty:
                        return []
                    
                    if sort_method == "信号强度":
                        trend_df = trend_df.sort_values('信号强度', ascending=False)
                    elif sort_method == "变化率绝对值":
                        trend_df = trend_df.sort_values('变化率', key=abs, ascending=False)
                    elif sort_method == "趋势强度":
                        trend_df = trend_df.sort_values('趋势强度', ascending=False)
                    
                    return trend_df['品种'].head(5).tolist()
                
                cumulative_symbols = safe_get_sorted_symbols(inventory_trends['累库品种'])
                depletion_symbols = safe_get_sorted_symbols(inventory_trends['去库品种'])
                
                summary_text = f"""
库存分析摘要 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
=====================================
分析参数: 截止日期{end_date}, 分析{days_back}天数据
排序方式: {sort_method}
总品种数: {len(results_df)}
累库品种: {len(inventory_trends['累库品种'])} 个
去库品种: {len(inventory_trends['去库品种'])} 个
稳定品种: {len(inventory_trends['库存稳定品种'])} 个

重点累库品种: {', '.join(cumulative_symbols)}
重点去库品种: {', '.join(depletion_symbols)}
"""
                st.code(summary_text)
                st.info("摘要已显示，可手动复制")

def basis_analysis_page():
    """基差分析页面"""
    st.header("💰 期货基差分析")
    
    # 参数设置
    st.subheader("🔧 分析参数设置")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        end_date = st.date_input("结束日期", value=datetime.now().date())
        end_day = end_date.strftime("%Y%m%d")
    
    with col2:
        days_back = st.slider("分析天数", min_value=15, max_value=90, value=30)
    
    with col3:
        min_confidence = st.slider("最低置信度(%)", min_value=20, max_value=80, value=50)
    
    with col4:
        st.write("缓存状态")
        cache_info = f"内存缓存: {len(cache_manager.memory_cache)} 项"
        st.info(cache_info)
    
    # 置信度说明
    with st.expander("📖 置信度说明"):
        st.markdown("""
        **置信度是什么？**
        - 置信度表示基差投资机会成功的预期概率
        - 它是基于多个技术指标综合计算得出的评分（0-100%）
        
        **置信度计算因子：**
        - **Z-Score权重最高**：基差偏离历史均值的标准化程度
        - **趋势反转信号**：基差趋势是否出现反转迹象
        - **RSI指标**：基差是否处于超买/超卖状态
        - **布林带位置**：基差在布林带中的相对位置
        - **波动率调整**：根据品种波动率特征进行调整
        
        **为什么按置信度排序？**
        - 置信度高的机会，基差回归的概率更大
        - 综合考虑了多个维度，比单一指标更可靠
        - 有助于投资者优先关注最有把握的机会
        
        **置信度阈值建议：**
        - **保守型**：60-70%以上（机会较少但质量高）
        - **平衡型**：40-50%以上（机会与质量平衡）
        - **激进型**：30-40%以上（机会较多但需谨慎）
        """)
    
    # 分析按钮
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        start_analysis = st.button("🚀 开始基差分析", type="primary")
    with col2:
        if st.button("🔄 重新分析"):
            if 'basis_results' in st.session_state:
                del st.session_state.basis_results
            cache_manager.clear_expired()
            st.rerun()
    with col3:
        if st.button("🗑️ 清除所有缓存"):
            cache_manager.clear_expired()
            st.session_state.clear()
            st.success("缓存已清除")
            st.rerun()
    
    # 执行分析
    if start_analysis or st.session_state.get('basis_results') is not None:
        
        # 如果是新的分析请求，执行分析
        if start_analysis:
            strategy = FuturesBasisStrategy()
            
            # 创建进度显示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current, total, message):
                progress_bar.progress(current / total)
                status_text.text(f"{message} [{current}/{total}]")
            
            with st.spinner("正在分析基差数据..."):
                try:
                    opportunities = strategy.run_analysis_streamlit(
                        end_day=end_day,
                        days_back=days_back,
                        min_confidence=min_confidence,
                        progress_callback=progress_callback
                    )
                    
                    # 保存到session state
                    st.session_state.basis_results = (opportunities, strategy)
                    
                    # 清除进度显示
                    progress_bar.empty()
                    status_text.empty()
                    
                    if opportunities:
                        st.success(f"🎯 发现 {len(opportunities)} 个投资机会！")
                    else:
                        st.warning("未发现符合条件的投资机会")
                        
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"分析过程中出错: {str(e)}")
                    return
        
        # 从session state获取数据
        if st.session_state.get('basis_results') is None:
            st.warning("没有可用的分析结果，请重新分析。")
            return
            
        opportunities, strategy = st.session_state.basis_results
        
        # 显示结果
        if opportunities:
            st.header("💰 基差分析结果")
            
            # 基差分析逻辑和原理说明
            with st.expander("📖 基差分析逻辑与原理"):
                st.markdown("""
                ### 🔍 基差分析核心原理
                
                #### 1. 基差定义与计算
                **基差 = 现货价格 - 期货价格**
                - 正基差：现货价格 > 期货价格（现货升水）
                - 负基差：现货价格 < 期货价格（期货升水）
                - 基差会围绕历史均值波动，存在回归特性
                
                #### 2. 投资机会识别逻辑
                
                **Z-Score标准化**：
                ```
                Z-Score = (当前基差 - 历史均值) / 历史标准差
                ```
                
                **机会类型判断**：
                - **Z-Score < -1.5** → **买基差机会**（期货被高估）
                - **Z-Score > 1.5** → **卖基差机会**（现货被高估）
                - **-1.5 ≤ Z-Score ≤ 1.5** → 根据强度判断弱机会
                
                #### 3. 置信度计算体系
                
                **基础评分**：
                - 极端机会（|Z-Score| > 1.5）：基础分 = min(|Z-Score| × 30, 85)
                - 中等机会（|Z-Score| > 1.0）：基础分 = min(|Z-Score| × 25, 70)
                - 弱机会（|Z-Score| > 0.8）：基础分 = min(|Z-Score| × 20, 50)
                
                **技术指标调整**：
                - **趋势反转信号**：±8分调整
                - **RSI超买超卖**：±4分调整（RSI<35或RSI>65）
                - **布林带位置**：±4分调整（位置<25%或>75%）
                - **波动率调整**：高波动率×0.85，低波动率×1.05
                
                #### 4. 投资操作策略
                
                **买基差操作**（做空期货信号）：
                - **操作**：买入现货 + 卖出期货
                - **逻辑**：期货价格相对现货被高估，等待基差回归
                - **盈利方式**：期货价格下跌或现货价格上涨
                
                **卖基差操作**（做多期货信号）：
                - **操作**：卖出现货 + 买入期货  
                - **逻辑**：现货价格相对期货被高估，等待基差回归
                - **盈利方式**：期货价格上涨或现货价格下跌
                
                #### 5. 风险评估体系
                
                **风险等级判断**：
                - **波动率风险**：高波动率增加风险分数
                - **极端程度风险**：|Z-Score|过大增加风险
                - **数据质量风险**：数据不足增加风险
                - **趋势一致性**：现货期货趋势背离增加风险
                
                **风险分级**：
                - **低风险**：风险分数 < 3，适合稳健投资者
                - **中风险**：风险分数 3-4，需要一定风险承受能力
                - **高风险**：风险分数 ≥ 5，仅适合激进投资者
                
                #### 6. 投资决策建议
                
                **机会选择优先级**：
                1. **置信度 > 70%** + **低风险** → 优先考虑
                2. **置信度 60-70%** + **中风险** → 谨慎考虑
                3. **置信度 < 60%** 或 **高风险** → 观望或小仓位试探
                
                **持仓期建议**：
                - 基于Z-Score绝对值计算：|Z-Score|越大，建议持仓期越短
                - 一般建议10-30天，等待基差回归
                
                💡 **核心理念**：基差交易本质是统计套利，利用价格关系的异常进行投资
                """)
            
            # 排序说明
            st.info("""
            📊 **结果排序说明**：
            - 结果按**置信度从高到低**排序
            - 置信度越高，表示基差回归的可能性越大
            - 建议优先关注置信度较高的投资机会
            """)
            
            # 创建结果表格
            results_data = []
            for opp in opportunities:
                signal_strength = "🔴极端" if abs(opp.z_score) > 2.0 else "🟡中等" if abs(opp.z_score) > 1.5 else "🟢弱"
                results_data.append({
                    "品种": opp.name,
                    "代码": opp.variety,
                    "信号强度": signal_strength,
                    "机会类型": opp.opportunity_type,
                    "置信度(%)": f"{opp.confidence:.1f}%",
                    "预期收益(%)": f"{opp.expected_return:.1f}%",
                    "风险等级": opp.risk_level,
                    "建议持仓(天)": opp.holding_period,
                    "Z-Score": f"{opp.z_score:.2f}",
                    "当前基差": f"{opp.current_basis:.2f}"
                })
            
            results_df = pd.DataFrame(results_data)
            
            # 筛选选项
            st.subheader("🔍 筛选选项")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                opportunity_filter = st.selectbox(
                    "机会类型筛选",
                    ["全部", "买基差机会", "卖基差机会", "弱买基差机会", "弱卖基差机会"]
                )
            
            with col2:
                risk_filter = st.selectbox(
                    "风险等级筛选",
                    ["全部", "低风险", "中风险", "高风险"]
                )
            
            with col3:
                min_confidence_display = st.slider("显示置信度阈值(%)", 0, 95, min_confidence)
            
            # 应用筛选
            filtered_opportunities = opportunities.copy()
            if opportunity_filter != "全部":
                filtered_opportunities = [opp for opp in filtered_opportunities if opp.opportunity_type == opportunity_filter]
            if risk_filter != "全部":
                filtered_opportunities = [opp for opp in filtered_opportunities if opp.risk_level == risk_filter]
            filtered_opportunities = [opp for opp in filtered_opportunities if opp.confidence >= min_confidence_display]
            
            # 更新显示表格
            if filtered_opportunities:
                filtered_results_data = []
                for opp in filtered_opportunities:
                    signal_strength = "🔴极端" if abs(opp.z_score) > 2.0 else "🟡中等" if abs(opp.z_score) > 1.5 else "🟢弱"
                    filtered_results_data.append({
                        "品种": opp.name,
                        "代码": opp.variety,
                        "信号强度": signal_strength,
                        "机会类型": opp.opportunity_type,
                        "置信度(%)": f"{opp.confidence:.1f}%",
                        "预期收益(%)": f"{opp.expected_return:.1f}%",
                        "风险等级": opp.risk_level,
                        "建议持仓(天)": opp.holding_period,
                        "Z-Score": f"{opp.z_score:.2f}",
                        "当前基差": f"{opp.current_basis:.2f}"
                    })
                
                filtered_results_df = pd.DataFrame(filtered_results_data)
                st.dataframe(filtered_results_df, use_container_width=True)
            else:
                st.warning("没有符合筛选条件的投资机会")
            
            # 统计摘要
            st.subheader("📊 分析摘要")
            col1, col2, col3, col4 = st.columns(4)
            
            buy_basis_count = len([o for o in opportunities if '买基差' in o.opportunity_type])
            sell_basis_count = len([o for o in opportunities if '卖基差' in o.opportunity_type])
            avg_confidence = np.mean([o.confidence for o in opportunities])
            avg_return = np.mean([o.expected_return for o in opportunities])
            
            with col1:
                st.metric("总机会数", len(opportunities))
            with col2:
                st.metric("买基差机会", buy_basis_count)
            with col3:
                st.metric("卖基差机会", sell_basis_count)
            with col4:
                st.metric("平均置信度", f"{avg_confidence:.1f}%")
            
            # 风险分布
            risk_dist = {}
            for opp in opportunities:
                risk_dist[opp.risk_level] = risk_dist.get(opp.risk_level, 0) + 1
            
            st.subheader("⚠️ 风险分布")
            risk_cols = st.columns(3)
            with risk_cols[0]:
                st.metric("低风险", risk_dist.get("低风险", 0))
            with risk_cols[1]:
                st.metric("中风险", risk_dist.get("中风险", 0))
            with risk_cols[2]:
                st.metric("高风险", risk_dist.get("高风险", 0))
            
            # 详细图表分析
            st.subheader("📊 详细图表分析")
            
            # 选择要查看图表的品种
            available_varieties = [opp.variety for opp in opportunities if opp.variety in strategy.analysis_results]
            chart_varieties = st.multiselect(
                "选择要查看详细分析的品种（最多3个）",
                options=available_varieties,
                default=[],
                max_selections=3,
                format_func=lambda x: next(opp.name for opp in opportunities if opp.variety == x)
            )
            
            if chart_varieties:
                # 显示图表的开关
                show_charts_key = f"show_basis_charts_{hash(tuple(chart_varieties))}"
                if show_charts_key not in st.session_state.show_charts:
                    st.session_state.show_charts[show_charts_key] = False
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("📈 显示详细图表", key=f"show_{show_charts_key}"):
                        st.session_state.show_charts[show_charts_key] = True
                with col2:
                    if st.button("🔄 隐藏图表", key=f"hide_{show_charts_key}"):
                        st.session_state.show_charts[show_charts_key] = False
                
                # 显示图表
                if st.session_state.show_charts.get(show_charts_key, False):
                    for variety in chart_varieties:
                        if variety in strategy.analysis_results:
                            opportunity = next(opp for opp in opportunities if opp.variety == variety)
                            show_basis_detailed_chart(strategy.analysis_results[variety], opportunity)
            
            # 投资建议汇总
            st.subheader("💡 投资建议汇总")
            
            # 按机会类型分组
            buy_basis_opps = [opp for opp in opportunities if '买基差' in opp.opportunity_type]
            sell_basis_opps = [opp for opp in opportunities if '卖基差' in opp.opportunity_type]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if buy_basis_opps:
                    st.markdown("**🟢 买基差机会（做空信号）**")
                    for opp in buy_basis_opps[:5]:
                        st.write(f"• {opp.name}: 置信度 {opp.confidence:.1f}%, 预期收益 {opp.expected_return:.1f}%")
                    
                    st.info("💡 买基差操作：买入现货 + 卖出期货（类似做空期货）")
            
            with col2:
                if sell_basis_opps:
                    st.markdown("**🔴 卖基差机会（做多信号）**")
                    for opp in sell_basis_opps[:5]:
                        st.write(f"• {opp.name}: 置信度 {opp.confidence:.1f}%, 预期收益 {opp.expected_return:.1f}%")
                    
                    st.info("💡 卖基差操作：卖出现货 + 买入期货（类似做多期货）")
            
            # 导出功能
            st.subheader("📥 导出分析结果")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 导出Excel报告"):
                    try:
                        exporter = get_report_exporter()
                        filepath = exporter.export_basis_excel(opportunities, strategy.analysis_stats)
                        
                        with open(filepath, 'rb') as f:
                            st.download_button(
                                label="⬇️ 下载Excel报告",
                                data=f.read(),
                                file_name=filepath.name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        st.success("Excel报告生成成功！")
                    except Exception as e:
                        st.error(f"导出失败: {str(e)}")
            
            with col2:
                if st.button("📈 导出CSV数据"):
                    csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="⬇️ 下载CSV数据",
                        data=csv,
                        file_name=f"基差分析结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("📋 复制分析摘要"):
                    summary_text = f"""
基差分析摘要 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
=====================================
分析参数: {days_back}天数据, 最低置信度{min_confidence}%
排序方式: 按置信度从高到低
总机会数: {len(opportunities)}
买基差机会: {buy_basis_count} 个
卖基差机会: {sell_basis_count} 个
平均置信度: {avg_confidence:.1f}%
平均预期收益: {avg_return:.1f}%

重点买基差品种: {', '.join([opp.name for opp in buy_basis_opps[:3]])}
重点卖基差品种: {', '.join([opp.name for opp in sell_basis_opps[:3]])}
"""
                    st.code(summary_text)
                    st.info("摘要已显示，可手动复制")
        
        else:
            st.warning("未发现符合条件的投资机会")
            st.info("💡 建议尝试降低置信度阈值或增加分析天数")
            
            # 显示分析统计信息
            if hasattr(strategy, 'analysis_stats'):
                st.subheader("📊 分析统计")
                stats = strategy.analysis_stats
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("成功获取数据", len(stats['successful_varieties']))
                with col2:
                    st.metric("获取失败", len(stats['failed_varieties']))
                with col3:
                    st.metric("检测到信号", len(stats['analyzed_varieties']))
                
                # 显示接近阈值的机会
                near_threshold = [v for v in stats['analyzed_varieties'] if v['confidence'] >= min_confidence * 0.8]
                if near_threshold:
                    st.subheader("💡 接近阈值的机会")
                    for variety in sorted(near_threshold, key=lambda x: x['confidence'], reverse=True)[:5]:
                        st.write(f"• {variety['name']}: {variety['opportunity_type']} | 置信度: {variety['confidence']:.1f}%")
                    st.info(f"建议: 可考虑降低置信度阈值至 {min_confidence*0.8:.0f}% 或更低")

def comprehensive_analysis_page():
    """综合分析页面"""
    st.header("🔍 综合基本面分析")
    
    # 检查是否有分析结果
    has_inventory = st.session_state.get('inventory_results') is not None
    has_basis = st.session_state.get('basis_results') is not None
    
    if not has_inventory and not has_basis:
        st.warning("请先进行库存分析或基差分析，然后再查看综合分析结果。")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔗 前往库存分析"):
                st.session_state.page = "库存分析"
                st.rerun()
        with col2:
            if st.button("🔗 前往基差分析"):
                st.session_state.page = "基差分析"
                st.rerun()
        return
    
    # 获取分析结果
    inventory_results = st.session_state.get('inventory_results')
    basis_results = st.session_state.get('basis_results')
    
    st.subheader("📊 分析结果概览")
    
    # 显示各模块的分析结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 库存分析结果")
        if has_inventory:
            results_df, inventory_trends, data_dict = inventory_results
            
            # 库存分析摘要
            st.metric("分析品种数", len(results_df))
            
            col1_1, col1_2, col1_3 = st.columns(3)
            with col1_1:
                st.metric("累库品种", len(inventory_trends['累库品种']))
            with col1_2:
                st.metric("去库品种", len(inventory_trends['去库品种']))
            with col1_3:
                st.metric("稳定品种", len(inventory_trends['库存稳定品种']))
            
            # 显示所有累库品种
            if inventory_trends['累库品种']:
                st.markdown("**🟢 所有累库品种（做空信号）:**")
                # 按信号强度排序累库品种
                cumulative_symbols = []
                for symbol in inventory_trends['累库品种']:
                    change_rate = results_df[results_df['品种'] == symbol]['变化率'].iloc[0]
                    signal_strength = results_df[results_df['品种'] == symbol]['信号强度'].iloc[0]
                    cumulative_symbols.append((symbol, change_rate, signal_strength))
                
                # 按信号强度降序排序
                cumulative_symbols.sort(key=lambda x: x[2], reverse=True)
                
                # 显示所有累库品种
                for symbol, change_rate, signal_strength in cumulative_symbols:
                    st.write(f"• {symbol}: {change_rate:.2f}% (信号强度: {signal_strength:.3f})")
            
            # 显示所有去库品种
            if inventory_trends['去库品种']:
                st.markdown("**🔴 所有去库品种（做多信号）:**")
                # 按信号强度排序去库品种
                depletion_symbols = []
                for symbol in inventory_trends['去库品种']:
                    change_rate = results_df[results_df['品种'] == symbol]['变化率'].iloc[0]
                    signal_strength = results_df[results_df['品种'] == symbol]['信号强度'].iloc[0]
                    depletion_symbols.append((symbol, change_rate, signal_strength))
                
                # 按信号强度降序排序
                depletion_symbols.sort(key=lambda x: x[2], reverse=True)
                
                # 显示所有去库品种
                for symbol, change_rate, signal_strength in depletion_symbols:
                    st.write(f"• {symbol}: {change_rate:.2f}% (信号强度: {signal_strength:.3f})")
        else:
            st.info("暂无库存分析结果")
    
    with col2:
        st.markdown("### 💰 基差分析结果")
        if has_basis:
            opportunities, strategy = basis_results
            
            # 基差分析摘要
            st.metric("投资机会数", len(opportunities))
            
            buy_basis_count = len([o for o in opportunities if '买基差' in o.opportunity_type])
            sell_basis_count = len([o for o in opportunities if '卖基差' in o.opportunity_type])
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("买基差机会", buy_basis_count)
            with col2_2:
                st.metric("卖基差机会", sell_basis_count)
            
            # 显示所有买基差机会
            buy_basis_opps = [opp for opp in opportunities if '买基差' in opp.opportunity_type]
            sell_basis_opps = [opp for opp in opportunities if '卖基差' in opp.opportunity_type]
            
            # 按置信度排序
            buy_basis_opps.sort(key=lambda x: x.confidence, reverse=True)
            sell_basis_opps.sort(key=lambda x: x.confidence, reverse=True)
            
            if buy_basis_opps:
                st.markdown("**🟢 所有买基差机会（做空信号）:**")
                for opp in buy_basis_opps:
                    st.write(f"• {opp.name}: 置信度 {opp.confidence:.1f}% (Z-Score: {opp.z_score:.2f})")
            
            if sell_basis_opps:
                st.markdown("**🔴 所有卖基差机会（做多信号）:**")
                for opp in sell_basis_opps:
                    st.write(f"• {opp.name}: 置信度 {opp.confidence:.1f}% (Z-Score: {opp.z_score:.2f})")
        else:
            st.info("暂无基差分析结果")
    
    # 信号共振分析
    if has_inventory and has_basis:
        st.subheader("🎯 信号共振分析")
        
        # 信号共振分析逻辑说明
        with st.expander("📖 信号共振分析逻辑"):
            st.markdown("""
            ### 🔍 信号共振分析原理
            
            #### 1. 共振分析逻辑
            **信号共振**是指库存分析和基差分析得出相同方向的投资信号，这种情况下投资机会的可靠性显著提高。
            
            #### 2. 共振类型识别
            
            **做空信号共振**：
            - **库存信号**：累库（库存增加）
            - **基差信号**：买基差（期货被高估）
            - **投资逻辑**：供应过剩 + 期货高估 → 强烈看空
            - **操作建议**：考虑做空期货或买基差操作
            
            **做多信号共振**：
            - **库存信号**：去库（库存减少）
            - **基差信号**：卖基差（现货被高估）
            - **投资逻辑**：供应紧张 + 现货高估 → 强烈看多
            - **操作建议**：考虑做多期货或卖基差操作
            
            #### 3. 信号冲突处理
            
            **冲突情况**：
            - 累库 + 卖基差：库存看空 vs 基差看多
            - 去库 + 买基差：库存看多 vs 基差看空
            
            **处理策略**：
            - 深入分析冲突原因
            - 考虑时间周期差异
            - 观望或等待信号明确
            
            #### 4. 投资优先级
            
            **优先级排序**：
            1. **双重共振** > **单一强信号** > **单一弱信号** > **信号冲突**
            2. 共振品种的投资成功概率更高
            3. 冲突品种需要更谨慎的分析
            
            #### 5. 风险控制
            - 即使是共振信号也要控制仓位
            - 设置止损点，防范系统性风险
            - 关注宏观经济和政策变化
            
            💡 **核心理念**：多维度信号验证，提高投资决策的可靠性
            """)
        
        st.markdown("---")
        
        results_df, inventory_trends, data_dict = inventory_results
        opportunities, strategy = basis_results
        
        # 获取基差信号品种
        buy_basis_symbols = [opp.variety for opp in opportunities if '买基差' in opp.opportunity_type]
        sell_basis_symbols = [opp.variety for opp in opportunities if '卖基差' in opp.opportunity_type]
        
        # 做空信号共振（累库 + 买基差）
        short_resonance = set(inventory_trends['累库品种']) & set(buy_basis_symbols)
        
        # 做多信号共振（去库 + 卖基差）
        long_resonance = set(inventory_trends['去库品种']) & set(sell_basis_symbols)
        
        # 信号冲突（累库 + 卖基差 或 去库 + 买基差）
        conflict_1 = set(inventory_trends['累库品种']) & set(sell_basis_symbols)  # 累库但卖基差
        conflict_2 = set(inventory_trends['去库品种']) & set(buy_basis_symbols)   # 去库但买基差
        
        # 显示共振分析结果
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("做空信号共振", len(short_resonance))
            st.metric("做多信号共振", len(long_resonance))
        
        with col2:
            st.metric("信号冲突品种", len(conflict_1) + len(conflict_2))
            total_analyzed = len(set(inventory_trends['累库品种'] + inventory_trends['去库品种']) | 
                               set(buy_basis_symbols + sell_basis_symbols))
            resonance_rate = (len(short_resonance) + len(long_resonance)) / max(total_analyzed, 1) * 100
            st.metric("共振率", f"{resonance_rate:.1f}%")
        
        with col3:
            if short_resonance or long_resonance:
                st.success("发现信号共振！")
            else:
                st.warning("未发现明显共振")
        
        # 详细共振分析
        if short_resonance or long_resonance or conflict_1 or conflict_2:
            st.subheader("📋 详细共振分析")
            
            # 做空信号共振
            if short_resonance:
                st.markdown("#### 🔴 做空信号共振品种")
                st.success("库存累积 + 买基差信号 = 强烈做空信号")
                
                resonance_data = []
                for symbol in short_resonance:
                    # 获取库存数据
                    inventory_change = results_df[results_df['品种'] == symbol]['变化率'].iloc[0]
                    inventory_signal = results_df[results_df['品种'] == symbol]['信号强度'].iloc[0]
                    
                    # 获取基差数据
                    basis_opp = next(opp for opp in opportunities if opp.variety == symbol)
                    
                    resonance_data.append({
                        '品种': symbol,
                        '库存变化率': f"{inventory_change:.2f}%",
                        '库存信号强度': f"{inventory_signal:.3f}",
                        '基差机会类型': basis_opp.opportunity_type,
                        '基差置信度': f"{basis_opp.confidence:.1f}%",
                        '综合建议': '强烈看空',
                        '操作建议': '考虑做空操作'
                    })
                
                resonance_df = pd.DataFrame(resonance_data)
                st.dataframe(resonance_df, use_container_width=True)
            
            # 做多信号共振
            if long_resonance:
                st.markdown("#### 🟢 做多信号共振品种")
                st.success("库存去化 + 卖基差信号 = 强烈做多信号")
                
                resonance_data = []
                for symbol in long_resonance:
                    # 获取库存数据
                    inventory_change = results_df[results_df['品种'] == symbol]['变化率'].iloc[0]
                    inventory_signal = results_df[results_df['品种'] == symbol]['信号强度'].iloc[0]
                    
                    # 获取基差数据
                    basis_opp = next(opp for opp in opportunities if opp.variety == symbol)
                    
                    resonance_data.append({
                        '品种': symbol,
                        '库存变化率': f"{inventory_change:.2f}%",
                        '库存信号强度': f"{inventory_signal:.3f}",
                        '基差机会类型': basis_opp.opportunity_type,
                        '基差置信度': f"{basis_opp.confidence:.1f}%",
                        '综合建议': '强烈看多',
                        '操作建议': '考虑做多操作'
                    })
                
                resonance_df = pd.DataFrame(resonance_data)
                st.dataframe(resonance_df, use_container_width=True)
            
            # 信号冲突
            if conflict_1 or conflict_2:
                st.markdown("#### ⚠️ 信号冲突品种")
                st.warning("库存信号与基差信号方向相反，需要谨慎分析")
                
                conflict_data = []
                
                # 累库但卖基差
                for symbol in conflict_1:
                    inventory_change = results_df[results_df['品种'] == symbol]['变化率'].iloc[0]
                    basis_opp = next(opp for opp in opportunities if opp.variety == symbol)
                    
                    conflict_data.append({
                        '品种': symbol,
                        '库存信号': '累库（看空）',
                        '基差信号': f"{basis_opp.opportunity_type}（看多）",
                        '库存变化率': f"{inventory_change:.2f}%",
                        '基差置信度': f"{basis_opp.confidence:.1f}%",
                        '冲突类型': '库存看空 vs 基差看多',
                        '建议': '观望或深入分析'
                    })
                
                # 去库但买基差
                for symbol in conflict_2:
                    inventory_change = results_df[results_df['品种'] == symbol]['变化率'].iloc[0]
                    basis_opp = next(opp for opp in opportunities if opp.variety == symbol)
                    
                    conflict_data.append({
                        '品种': symbol,
                        '库存信号': '去库（看多）',
                        '基差信号': f"{basis_opp.opportunity_type}（看空）",
                        '库存变化率': f"{inventory_change:.2f}%",
                        '基差置信度': f"{basis_opp.confidence:.1f}%",
                        '冲突类型': '库存看多 vs 基差看空',
                        '建议': '观望或深入分析'
                    })
                
                if conflict_data:
                    conflict_df = pd.DataFrame(conflict_data)
                    st.dataframe(conflict_df, use_container_width=True)
        
        # 投资建议总结
        st.subheader("💡 综合投资建议")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 优先关注品种")
            priority_symbols = list(short_resonance) + list(long_resonance)
            if priority_symbols:
                for symbol in priority_symbols:
                    if symbol in short_resonance:
                        st.write(f"🔴 {symbol}: 双重做空信号")
                    else:
                        st.write(f"🟢 {symbol}: 双重做多信号")
            else:
                st.info("暂无明显的信号共振品种")
            
            # 显示所有做空信号品种（包括单一信号）
            st.markdown("#### 📉 所有做空信号品种")
            all_short_signals = set(inventory_trends['累库品种']) | set(buy_basis_symbols)
            if all_short_signals:
                for symbol in sorted(all_short_signals):
                    signals = []
                    if symbol in inventory_trends['累库品种']:
                        signals.append("累库")
                    if symbol in buy_basis_symbols:
                        signals.append("买基差")
                    signal_text = " + ".join(signals)
                    if len(signals) > 1:
                        st.write(f"🔴 {symbol}: {signal_text} (共振)")
                    else:
                        st.write(f"🟡 {symbol}: {signal_text}")
            else:
                st.info("暂无做空信号品种")
        
        with col2:
            st.markdown("#### ⚠️ 谨慎观察品种")
            caution_symbols = list(conflict_1) + list(conflict_2)
            if caution_symbols:
                for symbol in caution_symbols:
                    st.write(f"⚠️ {symbol}: 信号冲突，需深入分析")
            else:
                st.info("暂无明显的信号冲突品种")
            
            # 显示所有做多信号品种（包括单一信号）
            st.markdown("#### 📈 所有做多信号品种")
            all_long_signals = set(inventory_trends['去库品种']) | set(sell_basis_symbols)
            if all_long_signals:
                for symbol in sorted(all_long_signals):
                    signals = []
                    if symbol in inventory_trends['去库品种']:
                        signals.append("去库")
                    if symbol in sell_basis_symbols:
                        signals.append("卖基差")
                    signal_text = " + ".join(signals)
                    if len(signals) > 1:
                        st.write(f"🟢 {symbol}: {signal_text} (共振)")
                    else:
                        st.write(f"🟡 {symbol}: {signal_text}")
            else:
                st.info("暂无做多信号品种")
        
        # 导出综合报告
        st.subheader("📥 导出综合报告")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 导出综合Excel报告"):
                try:
                    exporter = get_report_exporter()
                    filepath = exporter.create_comprehensive_report(inventory_results, basis_results)
                    
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label="⬇️ 下载综合报告",
                            data=f.read(),
                            file_name=filepath.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    st.success("综合报告生成成功！")
                except Exception as e:
                    st.error(f"导出失败: {str(e)}")
        
        with col2:
            if st.button("📈 导出共振分析"):
                # 创建共振分析数据
                resonance_analysis = []
                
                # 做空共振
                for symbol in short_resonance:
                    resonance_analysis.append({
                        '品种': symbol,
                        '信号类型': '做空共振',
                        '库存信号': '累库',
                        '基差信号': '买基差',
                        '投资建议': '看空，考虑做空操作'
                    })
                
                # 做多共振
                for symbol in long_resonance:
                    resonance_analysis.append({
                        '品种': symbol,
                        '信号类型': '做多共振',
                        '库存信号': '去库',
                        '基差信号': '卖基差',
                        '投资建议': '看多，考虑做多操作'
                    })
                
                if resonance_analysis:
                    resonance_df = pd.DataFrame(resonance_analysis)
                    csv = resonance_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="⬇️ 下载共振分析",
                        data=csv,
                        file_name=f"信号共振分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("暂无共振信号可导出")
        
        with col3:
            if st.button("📋 复制综合摘要"):
                summary_text = f"""
综合分析摘要 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
=====================================
库存分析: {len(results_df)} 个品种
基差分析: {len(opportunities)} 个机会

信号共振分析:
- 做空信号共振: {len(short_resonance)} 个品种
- 做多信号共振: {len(long_resonance)} 个品种
- 信号冲突: {len(conflict_1) + len(conflict_2)} 个品种
- 共振率: {resonance_rate:.1f}%

重点关注品种:
做空共振: {', '.join(list(short_resonance)[:3]) if short_resonance else '无'}
做多共振: {', '.join(list(long_resonance)[:3]) if long_resonance else '无'}
"""
                st.code(summary_text)
                st.info("摘要已显示，可手动复制")
    
    else:
        st.info("需要同时完成库存分析和基差分析才能进行信号共振分析。")
        
        missing_analysis = []
        if not has_inventory:
            missing_analysis.append("库存分析")
        if not has_basis:
            missing_analysis.append("基差分析")
        
        st.warning(f"缺少: {', '.join(missing_analysis)}")
        
        # 提供快速导航
        col1, col2 = st.columns(2)
        with col1:
            if not has_inventory and st.button("🔗 前往库存分析"):
                st.session_state.page = "库存分析"
                st.rerun()
        with col2:
            if not has_basis and st.button("🔗 前往基差分析"):
                st.session_state.page = "基差分析"
                st.rerun()

# ==================== 主应用程序 ====================

def main():
    # 初始化会话状态
    init_session_state()
    
    st.title("📊 期货基本面综合分析系统")
    st.markdown("*by 7haoge 953534947@qq.com*")
    st.markdown("---")
    
    # 系统介绍
    with st.expander("📖 系统介绍与使用指南", expanded=False):
        st.markdown("""
        ### 🎯 系统概述
        本系统是一个专业的期货基本面分析工具，通过**库存分析**和**基差分析**两个维度，
        为投资者提供科学的投资决策支持。
        
        ### 📊 核心功能
        
        #### 1. 库存分析模块
        - **功能**：分析期货品种的库存变化趋势
        - **原理**：基于库存供需关系判断价格方向
        - **输出**：累库（看空）、去库（看多）、稳定信号
        - **应用**：适合中长期趋势判断
        
        #### 2. 基差分析模块  
        - **功能**：分析现货与期货的价格差异
        - **原理**：基于统计套利原理，利用基差回归特性
        - **输出**：买基差（做空）、卖基差（做多）机会
        - **应用**：适合短中期套利交易
        
        #### 3. 综合分析模块
        - **功能**：整合库存和基差分析结果
        - **原理**：通过信号共振提高投资可靠性
        - **输出**：共振信号、冲突信号、投资优先级
        - **应用**：提供最终投资决策建议
        
        ### 🔍 分析逻辑
        
        **库存分析逻辑**：
        ```
        库存增加 → 供应过剩 → 价格下跌压力 → 看空信号
        库存减少 → 供应紧张 → 价格上涨动力 → 看多信号
        ```
        
        **基差分析逻辑**：
        ```
        基差异常偏离 → 价格关系失衡 → 回归预期 → 套利机会
        ```
        
        **综合分析逻辑**：
        ```
        信号共振 → 多维度验证 → 高可靠性 → 优先投资
        信号冲突 → 深入分析 → 谨慎观望 → 等待明确
        ```
        
        ### 💡 使用建议
        
        1. **新手用户**：建议从单模块分析开始，理解基本逻辑
        2. **进阶用户**：使用综合分析，关注信号共振机会
        3. **专业用户**：结合高级筛选，自定义分析参数
        
        ### ⚠️ 风险提示
        
        - 本系统仅提供分析工具，不构成投资建议
        - 投资有风险，决策需谨慎
        - 建议结合其他分析方法综合判断
        - 注意控制仓位和设置止损
        """)
    
    # 侧边栏导航
    st.sidebar.title("🔧 分析模块选择")
    
    # 页面选择
    if 'page' not in st.session_state:
        st.session_state.page = "📈 库存分析"
    
    analysis_type = st.sidebar.selectbox(
        "请选择分析类型",
        ["📈 库存分析", "💰 基差分析", "🔍 综合分析"],
        index=["📈 库存分析", "💰 基差分析", "🔍 综合分析"].index(st.session_state.page) if st.session_state.page in ["📈 库存分析", "💰 基差分析", "🔍 综合分析"] else 0
    )
    
    # 更新页面状态
    st.session_state.page = analysis_type
    
    # 侧边栏状态信息
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 分析状态")
    
    # 显示分析状态
    has_inventory = st.session_state.get('inventory_results') is not None
    has_basis = st.session_state.get('basis_results') is not None
    
    if has_inventory:
        results_df, inventory_trends, _ = st.session_state.inventory_results
        st.sidebar.success(f"✅ 库存分析完成 ({len(results_df)}个品种)")
    else:
        st.sidebar.info("⏳ 库存分析未完成")
    
    if has_basis:
        opportunities, _ = st.session_state.basis_results
        st.sidebar.success(f"✅ 基差分析完成 ({len(opportunities)}个机会)")
    else:
        st.sidebar.info("⏳ 基差分析未完成")
    
    # 缓存管理
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗂️ 缓存管理")
    
    cache_info = f"内存缓存: {len(cache_manager.memory_cache)} 项"
    st.sidebar.info(cache_info)
    
    if st.sidebar.button("🗑️ 清除过期缓存"):
        cache_manager.clear_expired()
        st.sidebar.success("过期缓存已清除")
    
    if st.sidebar.button("🔄 清除所有缓存"):
        cache_manager.clear_expired()
        st.session_state.clear()
        st.sidebar.success("所有缓存已清除")
        st.rerun()
    
    # 系统信息
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ 系统信息")
    st.sidebar.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 作者信息
    st.sidebar.markdown("---")
    st.sidebar.subheader("👨‍💻 作者信息")
    st.sidebar.markdown("""
    **作者**: 7haoge  
    **邮箱**: 953534947@qq.com  
    **年份**: 2025.06
    """)
    
    # 使用说明
    with st.sidebar.expander("📖 使用说明"):
        st.markdown("""
        **库存分析**：
        - 分析期货品种库存变化趋势
        - 识别累库、去库、稳定三种状态
        - 提供投资方向建议
        
        **基差分析**：
        - 分析现货与期货价格差异
        - 识别买基差、卖基差机会
        - 提供置信度评估
        
        **综合分析**：
        - 整合库存和基差分析结果
        - 识别信号共振机会
        - 提供综合投资建议
        """)
    
    # 根据选择显示对应页面
    if analysis_type == "📈 库存分析":
        inventory_analysis_page()
    elif analysis_type == "💰 基差分析":
        basis_analysis_page()
    elif analysis_type == "🔍 综合分析":
        comprehensive_analysis_page()
    
    # 页面底部信息
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("💡 **提示**: 建议先完成库存分析和基差分析，再查看综合分析")
    
    with col2:
        st.info("⚡ **性能**: 系统已启用智能缓存，重复分析将更快完成")
    
    with col3:
        st.info("📊 **数据**: 所有数据来源于akshare，请确保网络连接正常")
    
    # 作者信息和版权声明
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px; padding: 20px;'>
        <p><strong>期货基本面综合分析系统 </strong></p>
        <p>👨‍💻 作者: <strong>7haoge</strong> | 📧 邮箱: <strong>953534947@qq.com</strong></p>
        <p>🔧 技术栈: Streamlit + AKShare + Pandas + Plotly + Scipy</p>
        <p>⚠️ 免责声明: 本系统仅供学习研究使用，不构成投资建议。投资有风险，决策需谨慎。</p>
        <p>© 2025 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 