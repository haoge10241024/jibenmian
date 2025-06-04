import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import seaborn as sns
import matplotlib

# 设置中文字体和样式
import matplotlib
# 简化字体设置，避免报错
try:
    # 尝试设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    USE_CHINESE = True
except:
    # 如果中文字体不可用，使用英文
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    USE_CHINESE = False

warnings.filterwarnings('ignore')

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
        try:
            contract_name = ak.futures_display_main_sina()
            contract_name['symbol'] = contract_name['symbol'].str.replace('0', '')
            self.contracts = contract_name[['symbol', 'name']]
            print(f"✓ 成功获取 {len(self.contracts)} 个主力合约品种")
            return self.contracts
        except Exception as e:
            print(f"✗ 获取合约信息失败: {e}")
            return pd.DataFrame()
    
    def get_basis_data(self, variety: str, start_day: str, end_day: str) -> Optional[pd.DataFrame]:
        """获取并处理基差数据"""
        try:
            df = ak.futures_spot_price_daily(
                start_day=start_day,
                end_day=end_day,
                vars_list=[variety]
            )
            
            if df.empty:
                return None
            
            # 数据预处理
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['basis'] = df['spot_price'] - df['dominant_contract_price']
            df['basis_rate'] = df['basis'] / df['spot_price'] * 100
            
            # 添加技术指标
            df = self._add_technical_indicators(df)
            
            return df.sort_values('date')
            
        except Exception as e:
            print(f"品种 {variety} 数据获取失败: {e}")
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
        """识别投资机会 - 调整后的更宽松条件"""
        
        # 基差异常程度评分
        extreme_score = abs(z_score)
        
        # 趋势反转信号
        reversal_signal = 0
        if z_score < -1.2 and trend > 0:  # 降低阈值从-1.5到-1.2
            reversal_signal += 1
        if z_score > 1.2 and trend < 0:   # 降低阈值从1.5到1.2
            reversal_signal += 1
        
        # RSI超买超卖信号
        rsi_signal = 0
        if rsi < 35:  # 放宽从30到35
            rsi_signal = 1
        elif rsi > 65:  # 放宽从70到65
            rsi_signal = -1
        
        # 布林带信号
        bb_signal = 0
        if bb_position < 0.25:  # 放宽从0.2到0.25
            bb_signal = 1
        elif bb_position > 0.75:  # 放宽从0.8到0.75
            bb_signal = -1
        
        # 调整后的综合评分 - 降低阈值
        if z_score < -1.5:  # 保持强信号阈值
            opportunity_type = "买基差机会"
            base_confidence = min(extreme_score * 30, 85)  # 提高基础分
            expected_return = min(abs(z_score) * 2, 8)
            holding_period = max(10, int(20 - extreme_score * 3))
            
        elif z_score > 1.5:  # 保持强信号阈值
            opportunity_type = "卖基差机会"
            base_confidence = min(extreme_score * 30, 85)  # 提高基础分
            expected_return = min(abs(z_score) * 2, 8)
            holding_period = max(10, int(20 - extreme_score * 3))
            
        elif abs(z_score) > 1.0:  # 降低中等信号阈值从1.5到1.0
            if z_score < 0:
                opportunity_type = "买基差机会"
            else:
                opportunity_type = "卖基差机会"
            base_confidence = min(extreme_score * 25, 70)  # 提高基础分
            expected_return = min(abs(z_score) * 1.5, 5)
            holding_period = max(15, int(25 - extreme_score * 2))
            
        elif abs(z_score) > 0.8:  # 新增弱信号类别
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
            confidence_adjustment = reversal_signal * 8 + rsi_signal * 4 + bb_signal * 4  # 降低调整幅度
            final_confidence = max(0, min(95, base_confidence + confidence_adjustment))
            
            # 波动率调整 - 更温和
            if volatility > 0.6:  # 提高阈值
                final_confidence *= 0.85  # 减少惩罚
            elif volatility < 0.15:  # 降低阈值
                final_confidence *= 1.05  # 减少奖励
                
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
        if spot_trend * futures_trend < 0:  # 趋势不一致
            risk_score += 1
        
        if risk_score >= 5:
            return "高风险"
        elif risk_score >= 3:
            return "中风险"
        else:
            return "低风险"
    
    def run_analysis(self, end_day: str, days_back: int = 30, min_confidence: float = 50.0) -> List[BasisOpportunity]:
        """运行完整的基差分析"""
        print("=" * 60)
        print("🚀 期货基差投资策略分析系统")
        print("=" * 60)
        
        # 获取合约信息
        if self.contracts is None:
            self.get_main_contracts()
        
        if self.contracts.empty:
            print("❌ 无法获取合约信息，分析终止")
            return []
        
        # 计算日期范围
        end_date = datetime.strptime(end_day, "%Y%m%d")
        start_date = end_date - timedelta(days=days_back)
        start_day = start_date.strftime("%Y%m%d")
        
        print(f"📅 分析期间: {start_day} 至 {end_day}")
        print(f"🎯 最低置信度阈值: {min_confidence}%")
        print(f"📊 待分析品种数量: {len(self.contracts)}")
        print("-" * 60)
        
        # 统计变量
        opportunities = []
        failed_varieties = []
        successful_varieties = []
        analyzed_varieties = []
        
        # 遍历所有品种
        for idx, row in self.contracts.iterrows():
            symbol = row['symbol']
            name = row['name']
            
            print(f"🔍 [{idx+1:2d}/{len(self.contracts)}] 分析 {name} ({symbol})...", end=" ")
            
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
                        print(f"✅ {opportunity.opportunity_type} (置信度: {opportunity.confidence:.1f}%)")
                    else:
                        print(f"⚪ {opportunity.opportunity_type} (置信度: {opportunity.confidence:.1f}% < {min_confidence}%)")
                else:
                    print("⚪ 无明显机会")
            else:
                failed_varieties.append({'symbol': symbol, 'name': name, 'reason': '数据不足' if df is not None else '获取失败'})
                print("❌ 数据不足" if df is not None else "❌ 获取失败")
            
            # 避免请求过快
            time.sleep(0.5)
        
        # 详细统计报告
        self._print_analysis_report(successful_varieties, failed_varieties, analyzed_varieties, opportunities, min_confidence)
        
        # 按置信度排序
        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        self.opportunities = opportunities
        
        return opportunities
    
    def _print_analysis_report(self, successful_varieties, failed_varieties, analyzed_varieties, opportunities, min_confidence):
        """打印详细的分析报告"""
        print("\n" + "=" * 80)
        print("📊 数据获取与分析报告")
        print("=" * 80)
        
        # 数据获取统计
        total_varieties = len(successful_varieties) + len(failed_varieties)
        success_rate = len(successful_varieties) / total_varieties * 100 if total_varieties > 0 else 0
        
        print(f"📈 数据获取统计:")
        print(f"   总品种数: {total_varieties}")
        print(f"   成功获取: {len(successful_varieties)} 个 ({success_rate:.1f}%)")
        print(f"   获取失败: {len(failed_varieties)} 个 ({100-success_rate:.1f}%)")
        
        # 成功获取数据的品种
        if successful_varieties:
            print(f"\n✅ 成功获取数据的品种 ({len(successful_varieties)}个):")
            for i, variety in enumerate(successful_varieties[:20], 1):  # 显示前20个
                print(f"   {i:2d}. {variety['name']} ({variety['symbol']}) - {variety['data_points']}个数据点")
            if len(successful_varieties) > 20:
                print(f"   ... 还有 {len(successful_varieties)-20} 个品种")
        
        # 获取失败的品种
        if failed_varieties:
            print(f"\n❌ 获取失败的品种 ({len(failed_varieties)}个):")
            for i, variety in enumerate(failed_varieties[:15], 1):  # 显示前15个
                print(f"   {i:2d}. {variety['name']} ({variety['symbol']}) - {variety['reason']}")
            if len(failed_varieties) > 15:
                print(f"   ... 还有 {len(failed_varieties)-15} 个品种")
        
        # 分析结果统计
        if analyzed_varieties:
            print(f"\n🔍 基差分析统计:")
            
            # 按Z-Score分类
            extreme_negative = [v for v in analyzed_varieties if v['z_score'] < -2.0]
            extreme_positive = [v for v in analyzed_varieties if v['z_score'] > 2.0]
            moderate_negative = [v for v in analyzed_varieties if -2.0 <= v['z_score'] < -1.5]
            moderate_positive = [v for v in analyzed_varieties if 1.5 < v['z_score'] <= 2.0]
            normal_range = [v for v in analyzed_varieties if -1.5 <= v['z_score'] <= 1.5]
            
            print(f"   极端买基差信号 (Z-Score < -2.0): {len(extreme_negative)} 个")
            print(f"   中等买基差信号 (-2.0 ≤ Z-Score < -1.5): {len(moderate_negative)} 个")
            print(f"   正常范围 (-1.5 ≤ Z-Score ≤ 1.5): {len(normal_range)} 个")
            print(f"   中等卖基差信号 (1.5 < Z-Score ≤ 2.0): {len(moderate_positive)} 个")
            print(f"   极端卖基差信号 (Z-Score > 2.0): {len(extreme_positive)} 个")
            
            # 显示所有有信号的品种
            signal_varieties = extreme_negative + extreme_positive + moderate_negative + moderate_positive
            if signal_varieties:
                print(f"\n📋 所有检测到信号的品种:")
                for variety in sorted(signal_varieties, key=lambda x: abs(x['z_score']), reverse=True):
                    signal_type = "买基差" if variety['z_score'] < 0 else "卖基差"
                    strength = "极端" if abs(variety['z_score']) > 2.0 else "中等"
                    print(f"   {variety['name']} ({variety['symbol']}): {strength}{signal_type} | Z-Score: {variety['z_score']:.2f} | 置信度: {variety['confidence']:.1f}%")
        
        # 最终机会统计
        print(f"\n🎯 投资机会汇总:")
        print(f"   检测到信号的品种: {len(analyzed_varieties)} 个")
        print(f"   达到置信度阈值 (≥{min_confidence}%): {len(opportunities)} 个")
        
        if len(opportunities) == 0 and len(analyzed_varieties) > 0:
            # 找出最接近阈值的机会
            near_threshold = [v for v in analyzed_varieties if v['confidence'] >= min_confidence * 0.8]
            if near_threshold:
                print(f"\n💡 接近阈值的机会 (置信度 ≥ {min_confidence*0.8:.1f}%):")
                for variety in sorted(near_threshold, key=lambda x: x['confidence'], reverse=True)[:5]:
                    print(f"   {variety['name']} ({variety['symbol']}): {variety['opportunity_type']} | 置信度: {variety['confidence']:.1f}%")
                print(f"   建议: 可考虑降低置信度阈值至 {min_confidence*0.8:.0f}% 或更低")
    
    def display_opportunities(self, top_n: int = 10):
        """显示投资机会汇总"""
        if not self.opportunities:
            print("❌ 暂无投资机会")
            return
        
        print("\n" + "=" * 80)
        print("🎯 投资机会排行榜")
        print("=" * 80)
        
        for i, opp in enumerate(self.opportunities[:top_n], 1):
            print(f"\n🏆 第 {i} 名: {opp.name} ({opp.variety})")
            print(f"   📊 机会类型: {opp.opportunity_type}")
            print(f"   🎯 置信度: {opp.confidence:.1f}%")
            print(f"   💰 预期收益: {opp.expected_return:.1f}%")
            print(f"   ⏱️  建议持仓: {opp.holding_period} 天")
            print(f"   ⚠️  风险等级: {opp.risk_level}")
            print(f"   📈 当前基差: {opp.current_basis:.2f}")
            print(f"   📊 Z-Score: {opp.z_score:.2f}")
            print(f"   📍 分位数: {opp.percentile:.1f}%")
    
    def plot_opportunity_analysis(self, variety: str, save_path: str = None):
        """绘制投资机会分析图"""
        if variety not in self.analysis_results:
            print(f"❌ 未找到品种 {variety} 的分析数据")
            return
        
        df = self.analysis_results[variety]
        opportunity = next((opp for opp in self.opportunities if opp.variety == variety), None)
        
        if opportunity is None:
            print(f"❌ 品种 {variety} 无投资机会数据")
            return
        
        # 标签设置（中英文）
        labels = {
            'title': f'{opportunity.name} ({variety}) 基差投资机会分析' if USE_CHINESE else f'{opportunity.name} ({variety}) Basis Analysis',
            'spot_price': '现货价格' if USE_CHINESE else 'Spot Price',
            'futures_price': '期货价格' if USE_CHINESE else 'Futures Price',
            'price_trend': '现货与期货价格走势' if USE_CHINESE else 'Spot vs Futures Price Trend',
            'price': '价格' if USE_CHINESE else 'Price',
            'basis': '基差' if USE_CHINESE else 'Basis',
            'basis_ma10': '基差MA10' if USE_CHINESE else 'Basis MA10',
            'bollinger': '布林带' if USE_CHINESE else 'Bollinger Bands',
            'historical_mean': '历史均值' if USE_CHINESE else 'Historical Mean',
            'basis_trend': '基差走势分析' if USE_CHINESE else 'Basis Trend Analysis',
            'basis_dist': '基差分布直方图' if USE_CHINESE else 'Basis Distribution',
            'current_basis': '当前基差' if USE_CHINESE else 'Current Basis',
            'frequency': '频次' if USE_CHINESE else 'Frequency',
            'technical': '技术指标分析' if USE_CHINESE else 'Technical Analysis',
            'basis_rsi': '基差RSI' if USE_CHINESE else 'Basis RSI',
            'overbought': '超买线' if USE_CHINESE else 'Overbought',
            'oversold': '超卖线' if USE_CHINESE else 'Oversold'
        }
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(labels['title'], fontsize=16, fontweight='bold')
        
        # 1. 价格走势图
        ax1.plot(df['date'], df['spot_price'], label=labels['spot_price'], color='blue', linewidth=2)
        ax1.plot(df['date'], df['dominant_contract_price'], label=labels['futures_price'], color='red', linewidth=2)
        ax1.set_title(labels['price_trend'])
        ax1.set_ylabel(labels['price'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 基差走势与布林带
        ax2.plot(df['date'], df['basis'], label=labels['basis'], color='green', linewidth=2)
        ax2.plot(df['date'], df['basis_ma10'], label=labels['basis_ma10'], color='orange', linestyle='--')
        ax2.fill_between(df['date'], df['basis_upper'], df['basis_lower'], alpha=0.2, color='gray', label=labels['bollinger'])
        ax2.axhline(y=opportunity.basis_mean, color='red', linestyle=':', label=labels['historical_mean'])
        ax2.set_title(labels['basis_trend'])
        ax2.set_ylabel(labels['basis'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 基差分布直方图
        ax3.hist(df['basis'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=opportunity.current_basis, color='red', linestyle='--', linewidth=2, 
                   label=f'{labels["current_basis"]}: {opportunity.current_basis:.2f}')
        ax3.axvline(x=opportunity.basis_mean, color='green', linestyle='--', linewidth=2, 
                   label=f'{labels["historical_mean"]}: {opportunity.basis_mean:.2f}')
        ax3.set_title(labels['basis_dist'])
        ax3.set_xlabel(labels['basis'])
        ax3.set_ylabel(labels['frequency'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 技术指标
        ax4_twin = ax4.twinx()
        ax4.plot(df['date'], df['basis_rsi'], label=labels['basis_rsi'], color='purple', linewidth=2)
        ax4.axhline(y=70, color='red', linestyle=':', alpha=0.7, label=labels['overbought'])
        ax4.axhline(y=30, color='green', linestyle=':', alpha=0.7, label=labels['oversold'])
        ax4.set_ylabel('RSI', color='purple')
        ax4.set_ylim(0, 100)
        
        ax4_twin.plot(df['date'], df['basis'], color='green', alpha=0.5, linewidth=1)
        ax4_twin.set_ylabel(labels['basis'], color='green')
        
        ax4.set_title(labels['technical'])
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # 添加投资建议文本框（使用英文避免字体问题）
        textstr = f'''Investment Advice:
Type: {opportunity.opportunity_type}
Confidence: {opportunity.confidence:.1f}%
Expected Return: {opportunity.expected_return:.1f}%
Holding Period: {opportunity.holding_period} days
Risk Level: {opportunity.risk_level}
Z-Score: {opportunity.z_score:.2f}'''
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        fig.text(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 图表已保存至: {save_path}")
        
        plt.show()
    
    def export_results(self, filename: str = "basis_opportunities.csv"):
        """导出分析结果"""
        if not self.opportunities:
            print("❌ 无数据可导出")
            return
        
        # 转换为DataFrame
        data = []
        for opp in self.opportunities:
            data.append({
                '品种代码': opp.variety,
                '品种名称': opp.name,
                '机会类型': opp.opportunity_type,
                '置信度(%)': round(opp.confidence, 1),
                '预期收益(%)': round(opp.expected_return, 1),
                '建议持仓(天)': opp.holding_period,
                '风险等级': opp.risk_level,
                '当前基差': round(opp.current_basis, 2),
                '历史均值': round(opp.basis_mean, 2),
                '标准差': round(opp.basis_std, 2),
                'Z-Score': round(opp.z_score, 2),
                '分位数(%)': round(opp.percentile, 1)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ 分析结果已导出至: {filename}")
        
        return df
    
    def explain_analysis_criteria(self):
        """解释分析判断条件和逻辑"""
        print("\n" + "=" * 80)
        print("📚 基差投资策略判断条件详解")
        print("=" * 80)
        
        print("🎯 Z-Score 信号分类:")
        print("   极端买基差信号: Z-Score < -1.5 (基差异常偏小)")
        print("   中等买基差信号: -1.5 ≤ Z-Score < -1.0")
        print("   弱买基差信号:   -1.0 ≤ Z-Score < -0.8")
        print("   正常范围:       -0.8 ≤ Z-Score ≤ 0.8")
        print("   弱卖基差信号:   0.8 < Z-Score ≤ 1.0")
        print("   中等卖基差信号: 1.0 < Z-Score ≤ 1.5")
        print("   极端卖基差信号: Z-Score > 1.5 (基差异常偏大)")
        
        print("\n📊 置信度计算逻辑:")
        print("   基础分数 = |Z-Score| × 系数")
        print("   - 极端信号: 系数 30, 最高85分")
        print("   - 中等信号: 系数 25, 最高70分")
        print("   - 弱信号:   系数 20, 最高50分")
        
        print("\n🔧 调整因子:")
        print("   趋势反转信号: ±8分")
        print("   - 基差偏小且开始回升: +8分")
        print("   - 基差偏大且开始回落: +8分")
        
        print("   RSI 超买超卖: ±4分")
        print("   - RSI < 35 (超卖): +4分")
        print("   - RSI > 65 (超买): -4分")
        
        print("   布林带位置: ±4分")
        print("   - 接近下轨 (<25%): +4分")
        print("   - 接近上轨 (>75%): -4分")
        
        print("   波动率调整:")
        print("   - 高波动率 (>0.6): 置信度 × 0.85")
        print("   - 低波动率 (<0.15): 置信度 × 1.05")
        
        print("\n⚠️ 风险等级评估:")
        print("   评分维度:")
        print("   - 波动率风险: 0-3分")
        print("   - 极端程度风险: 0-2分")
        print("   - 数据质量风险: 0-1分")
        print("   - 趋势一致性风险: 0-1分")
        
        print("   等级划分:")
        print("   - 低风险: 0-2分")
        print("   - 中风险: 3-4分")
        print("   - 高风险: 5分以上")
        
        print("\n💡 投资建议:")
        print("   1. 优先选择置信度 ≥ 70% 的机会")
        print("   2. 低风险品种优先考虑")
        print("   3. 极端信号比中等信号更可靠")
        print("   4. 关注趋势反转信号的确认")
        print("   5. 避免高波动率品种（除非有特殊把握）")
    
    def get_analysis_summary(self) -> Dict:
        """获取分析结果摘要"""
        if not hasattr(self, 'opportunities') or not self.opportunities:
            return {}
        
        summary = {
            'total_opportunities': len(self.opportunities),
            'buy_basis_count': len([o for o in self.opportunities if '买基差' in o.opportunity_type]),
            'sell_basis_count': len([o for o in self.opportunities if '卖基差' in o.opportunity_type]),
            'avg_confidence': np.mean([o.confidence for o in self.opportunities]),
            'avg_expected_return': np.mean([o.expected_return for o in self.opportunities]),
            'risk_distribution': {
                '低风险': len([o for o in self.opportunities if o.risk_level == '低风险']),
                '中风险': len([o for o in self.opportunities if o.risk_level == '中风险']),
                '高风险': len([o for o in self.opportunities if o.risk_level == '高风险'])
            }
        }
        
        return summary
    
    def explain_simple_logic(self):
        """用简单语言解释程序逻辑"""
        print("\n" + "=" * 80)
        print("🎓 基差投资策略 - 简单易懂版")
        print("=" * 80)
        
        print("💡 核心思想：")
        print("   当期货和现货的价格关系出现异常时，它们会回归正常")
        print("   我们就是要抓住这种'回归'的机会赚钱")
        
        print("\n🔍 程序是怎么找机会的？")
        print("   1️⃣ 计算基差：现货价格 - 期货价格")
        print("   2️⃣ 看基差是否异常：比较当前基差和历史平均值")
        print("   3️⃣ 判断异常程度：用Z-Score衡量（就像考试分数的排名）")
        print("   4️⃣ 技术确认：用RSI、布林带等指标确认信号")
        print("   5️⃣ 计算可信度：综合所有因素给出置信度分数")
        
        print("\n📊 Z-Score简单理解：")
        print("   Z-Score = (当前基差 - 历史平均) / 历史波动幅度")
        print("   🔴 Z-Score < -1.5：基差异常偏小 → 买基差机会")
        print("   🟢 Z-Score 在 -0.8 到 0.8：正常范围 → 无机会")  
        print("   🔴 Z-Score > 1.5：基差异常偏大 → 卖基差机会")
        
        print("\n🎯 置信度阈值的作用：")
        print("   就像设置一个'质量标准'")
        print("   📈 设置50%：只看置信度≥50%的机会（平衡选择）")
        print("   📈 设置30%：看更多机会，但质量可能较低")
        print("   📈 设置70%：只看高质量机会，但数量较少")
        
        print("\n💰 买基差 vs 卖基差：")
        print("   🟢 买基差 ≈ 做空期货：")
        print("      - 操作：买现货 + 卖期货")
        print("      - 预期：期货价格相对下跌")
        print("      - 适用：期货被高估时")
        
        print("   🔴 卖基差 ≈ 做多期货：")
        print("      - 操作：卖现货 + 买期货") 
        print("      - 预期：期货价格相对上涨")
        print("      - 适用：现货被高估时")
        
        print("\n⚠️ 重要提醒：")
        print("   基差交易不是赌价格涨跌，而是赌价格关系的修复")
        print("   风险相对较小，但需要同时操作现货和期货两个市场")
        
        print("\n🎯 实用建议：")
        print("   新手建议：置信度阈值设置50-60%")
        print("   有经验：置信度阈值可以设置40-50%")
        print("   保守型：置信度阈值设置60-70%")
        
    def explain_confidence_threshold(self):
        """专门解释置信度阈值"""
        print("\n" + "=" * 60)
        print("🎯 置信度阈值详解")
        print("=" * 60)
        
        print("🤔 什么是置信度？")
        print("   置信度 = 这个投资机会成功的可能性（0-100%）")
        print("   就像天气预报说'降雨概率70%'一样")
        
        print("\n🎚️ 阈值的作用：")
        print("   阈值 = 你设定的最低要求")
        print("   只有达到这个要求的机会才会显示给你")
        
        print("\n📊 不同阈值的效果：")
        print("   🟢 30%阈值：机会很多，但质量参差不齐")
        print("   🟡 50%阈值：机会适中，质量较好（推荐）")
        print("   🔴 70%阈值：机会较少，但质量很高")
        
        print("\n💡 如何选择阈值？")
        print("   保守投资者：60-70%（宁缺毋滥）")
        print("   平衡投资者：40-50%（数量质量兼顾）")
        print("   激进投资者：30-40%（更多机会，自己筛选）")
        
        print("\n🎯 建议策略：")
        print("   1. 先用50%试试，看看有多少机会")
        print("   2. 如果机会太少，降到40%或30%")
        print("   3. 如果机会太多，提高到60%或70%")
        print("   4. 找到适合自己的平衡点")

def main():
    """主程序"""
    # 创建策略分析器
    strategy = FuturesBasisStrategy()
    
    # 用户输入
    print("期货基差投资策略分析系统")
    print("-" * 40)
    
    end_day = input("请输入结束日期（格式：YYYYMMDD，例如20250530）：").strip()
    if not end_day:
        end_day = "20250530"
    
    days_back = input("请输入分析天数（默认30天）：").strip()
    if not days_back:
        days_back = 30
    else:
        days_back = int(days_back)
    
    min_confidence = input("请输入最低置信度阈值（默认50%）：").strip()
    if not min_confidence:
        min_confidence = 50.0
    else:
        min_confidence = float(min_confidence)
    
    # 运行分析
    opportunities = strategy.run_analysis(end_day, days_back, min_confidence)
    
    # 显示结果
    strategy.display_opportunities()
    
    # 导出结果
    if opportunities:
        strategy.export_results()
        
        # 询问是否生成图表
        while True:
            variety = input("\n请输入要查看图表的品种代码（输入'quit'退出）：").strip().upper()
            if variety == 'QUIT':
                break
            
            if variety in strategy.analysis_results:
                strategy.plot_opportunity_analysis(variety)
            else:
                print(f"❌ 未找到品种 {variety} 的数据")
                available = list(strategy.analysis_results.keys())
                if available:
                    print(f"可用品种: {', '.join(available[:10])}")

if __name__ == "__main__":
    main() 