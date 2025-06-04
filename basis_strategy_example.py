#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货基差投资策略 - 使用示例
============================

本文件提供了基差策略的多种使用方式示例，帮助用户快速上手。

作者: AI Assistant
日期: 2024-12-19
版本: 1.0
"""

from futures_basis_strategy import FuturesBasisStrategy
from datetime import datetime, timedelta
import pandas as pd

def quick_analysis_example():
    """快速分析示例 - 使用默认参数"""
    print("=" * 60)
    print("🚀 快速分析示例")
    print("=" * 60)
    
    # 创建策略分析器
    strategy = FuturesBasisStrategy()
    
    # 使用默认参数进行分析
    end_day = datetime.now().strftime("%Y%m%d")
    opportunities = strategy.run_analysis(
        end_day=end_day,
        days_back=30,
        min_confidence=50.0
    )
    
    # 显示结果
    strategy.display_opportunities(top_n=5)
    
    # 导出结果
    if opportunities:
        strategy.export_results("快速分析结果.csv")
        print(f"\n✅ 发现 {len(opportunities)} 个投资机会")
    else:
        print("\n⚠️ 未发现符合条件的投资机会")
    
    return strategy

def custom_analysis_example():
    """自定义分析示例 - 调整参数"""
    print("=" * 60)
    print("🔧 自定义分析示例")
    print("=" * 60)
    
    # 创建策略分析器
    strategy = FuturesBasisStrategy()
    
    # 自定义参数
    end_day = "20241215"  # 指定结束日期
    days_back = 45        # 分析45天数据
    min_confidence = 40.0 # 降低置信度阈值，获得更多机会
    
    print(f"📅 分析参数:")
    print(f"   结束日期: {end_day}")
    print(f"   分析天数: {days_back}")
    print(f"   最低置信度: {min_confidence}%")
    print()
    
    # 运行分析
    opportunities = strategy.run_analysis(
        end_day=end_day,
        days_back=days_back,
        min_confidence=min_confidence
    )
    
    # 显示结果
    strategy.display_opportunities(top_n=8)
    
    # 分类显示机会
    if opportunities:
        buy_basis_opps = [opp for opp in opportunities if '买基差' in opp.opportunity_type]
        sell_basis_opps = [opp for opp in opportunities if '卖基差' in opp.opportunity_type]
        
        print(f"\n📊 机会分类统计:")
        print(f"   买基差机会: {len(buy_basis_opps)} 个")
        print(f"   卖基差机会: {len(sell_basis_opps)} 个")
        
        # 导出分类结果
        strategy.export_results("自定义分析结果.csv")
    
    return strategy

def specific_variety_analysis():
    """特定品种分析示例"""
    print("=" * 60)
    print("🎯 特定品种分析示例")
    print("=" * 60)
    
    # 创建策略分析器
    strategy = FuturesBasisStrategy()
    
    # 先运行完整分析
    end_day = datetime.now().strftime("%Y%m%d")
    opportunities = strategy.run_analysis(
        end_day=end_day,
        days_back=30,
        min_confidence=30.0  # 使用较低阈值确保有结果
    )
    
    if not opportunities:
        print("⚠️ 未发现投资机会，无法进行特定品种分析")
        return
    
    # 选择第一个机会进行详细分析
    target_opportunity = opportunities[0]
    variety = target_opportunity.variety
    
    print(f"\n🔍 详细分析品种: {target_opportunity.name} ({variety})")
    print(f"   机会类型: {target_opportunity.opportunity_type}")
    print(f"   置信度: {target_opportunity.confidence:.1f}%")
    print(f"   Z-Score: {target_opportunity.z_score:.2f}")
    
    # 生成图表分析
    save_path = f"{variety}_基差分析图表.png"
    strategy.plot_opportunity_analysis(variety, save_path)
    
    # 显示详细信息
    print(f"\n📈 详细投资建议:")
    print(f"   预期收益: {target_opportunity.expected_return:.1f}%")
    print(f"   建议持仓: {target_opportunity.holding_period} 天")
    print(f"   风险等级: {target_opportunity.risk_level}")
    print(f"   当前基差: {target_opportunity.current_basis:.2f}")
    print(f"   历史均值: {target_opportunity.basis_mean:.2f}")
    
    return strategy

def detailed_analysis_with_explanation():
    """详细分析示例 - 包含策略解释"""
    print("=" * 60)
    print("📚 详细分析示例（含策略解释）")
    print("=" * 60)
    
    # 创建策略分析器
    strategy = FuturesBasisStrategy()
    
    # 首先解释策略逻辑
    strategy.explain_simple_logic()
    strategy.explain_confidence_threshold()
    
    # 运行分析
    end_day = datetime.now().strftime("%Y%m%d")
    opportunities = strategy.run_analysis(
        end_day=end_day,
        days_back=30,
        min_confidence=50.0
    )
    
    # 显示结果
    strategy.display_opportunities()
    
    # 显示详细的判断条件
    strategy.explain_analysis_criteria()
    
    # 获取分析摘要
    summary = strategy.get_analysis_summary()
    if summary:
        print(f"\n📊 分析摘要:")
        print(f"   总机会数: {summary['total_opportunities']}")
        print(f"   买基差机会: {summary['buy_basis_count']}")
        print(f"   卖基差机会: {summary['sell_basis_count']}")
        print(f"   平均置信度: {summary['avg_confidence']:.1f}%")
        print(f"   平均预期收益: {summary['avg_expected_return']:.1f}%")
        print(f"   风险分布: {summary['risk_distribution']}")
    
    return strategy

def batch_analysis_example():
    """批量分析示例 - 不同置信度阈值对比"""
    print("=" * 60)
    print("📊 批量分析示例 - 置信度阈值对比")
    print("=" * 60)
    
    # 测试不同的置信度阈值
    confidence_levels = [30, 40, 50, 60, 70]
    results = {}
    
    end_day = datetime.now().strftime("%Y%m%d")
    
    for confidence in confidence_levels:
        print(f"\n🔍 测试置信度阈值: {confidence}%")
        print("-" * 40)
        
        strategy = FuturesBasisStrategy()
        opportunities = strategy.run_analysis(
            end_day=end_day,
            days_back=30,
            min_confidence=confidence
        )
        
        results[confidence] = {
            'total_opportunities': len(opportunities),
            'buy_basis_count': len([o for o in opportunities if '买基差' in o.opportunity_type]),
            'sell_basis_count': len([o for o in opportunities if '卖基差' in o.opportunity_type]),
            'avg_confidence': sum([o.confidence for o in opportunities]) / len(opportunities) if opportunities else 0,
            'opportunities': opportunities
        }
        
        print(f"   发现机会数: {len(opportunities)}")
        if opportunities:
            print(f"   平均置信度: {results[confidence]['avg_confidence']:.1f}%")
    
    # 汇总对比结果
    print(f"\n📈 置信度阈值对比结果:")
    print("=" * 60)
    print(f"{'阈值':<8} {'机会数':<8} {'买基差':<8} {'卖基差':<8} {'平均置信度':<12}")
    print("-" * 60)
    
    for confidence in confidence_levels:
        result = results[confidence]
        print(f"{confidence}%{'':<5} {result['total_opportunities']:<8} {result['buy_basis_count']:<8} "
              f"{result['sell_basis_count']:<8} {result['avg_confidence']:<12.1f}")
    
    # 推荐最佳阈值
    best_confidence = None
    best_score = 0
    
    for confidence, result in results.items():
        # 综合评分：机会数量 + 平均置信度
        score = result['total_opportunities'] * 0.7 + result['avg_confidence'] * 0.3
        if score > best_score:
            best_score = score
            best_confidence = confidence
    
    if best_confidence:
        print(f"\n💡 推荐置信度阈值: {best_confidence}%")
        print(f"   该阈值下发现 {results[best_confidence]['total_opportunities']} 个机会")
        print(f"   平均置信度: {results[best_confidence]['avg_confidence']:.1f}%")
    
    return results

def main():
    """主函数 - 演示所有示例"""
    print("🎯 期货基差投资策略 - 使用示例集合")
    print("=" * 80)
    
    examples = {
        "1": ("快速分析示例", quick_analysis_example),
        "2": ("自定义分析示例", custom_analysis_example),
        "3": ("特定品种分析示例", specific_variety_analysis),
        "4": ("详细分析示例（含解释）", detailed_analysis_with_explanation),
        "5": ("批量分析示例", batch_analysis_example),
        "0": ("运行所有示例", None)
    }
    
    print("请选择要运行的示例:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\n请输入选择 (0-5): ").strip()
    
    if choice == "0":
        # 运行所有示例
        for key, (name, func) in examples.items():
            if func is not None:
                print(f"\n{'='*20} {name} {'='*20}")
                try:
                    func()
                except Exception as e:
                    print(f"❌ 示例运行出错: {e}")
                print("\n" + "="*60)
                input("按回车键继续下一个示例...")
    
    elif choice in examples and examples[choice][1] is not None:
        name, func = examples[choice]
        print(f"\n运行示例: {name}")
        try:
            result = func()
            print(f"\n✅ 示例 '{name}' 运行完成")
        except Exception as e:
            print(f"❌ 示例运行出错: {e}")
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main() 