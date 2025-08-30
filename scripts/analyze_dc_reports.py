#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QZSS DCレポートデータ解析スクリプト
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_clean_data(file_path):
    """データの読み込みとクリーニング"""
    print("データを読み込み中...")
    
    # CSVファイルを読み込み（改行を含むメッセージに対応）
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # データを解析
    data = []
    current_record = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 新しいレコードの開始（タイムスタンプで始まる行）
        if line.startswith('2025/'):
            # 前のレコードを保存
            if current_record:
                data.append(current_record)
            
            # 新しいレコードを開始
            parts = line.split(',', 4)  # 最初の4つのカンマで分割
            if len(parts) >= 5:
                current_record = {
                    'timestamp': parts[0],
                    'report_type': parts[1],
                    'satellite': parts[2],
                    'priority': parts[3],
                    'message': parts[4]
                }
        else:
            # メッセージの続き
            if current_record:
                current_record['message'] += '\n' + line
    
    # 最後のレコードを追加
    if current_record:
        data.append(current_record)
    
    # DataFrameに変換
    df = pd.DataFrame(data)
    
    # タイムスタンプをdatetime型に変換
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S JST')
    
    # 日付と時間の列を追加
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    
    print(f"データ読み込み完了: {len(df)} レコード")
    return df

def analyze_basic_statistics(df):
    """基本統計の分析"""
    print("\n=== 基本統計 ===")
    print(f"総レコード数: {len(df):,}")
    print(f"期間: {df['timestamp'].min()} から {df['timestamp'].max()}")
    print(f"データ期間: {(df['timestamp'].max() - df['timestamp'].min()).days} 日間")
    
    print(f"\nレポートタイプ別集計:")
    report_type_counts = df['report_type'].value_counts()
    for report_type, count in report_type_counts.items():
        print(f"  {report_type}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\n衛星別集計:")
    satellite_counts = df['satellite'].value_counts()
    for satellite, count in satellite_counts.items():
        print(f"  {satellite}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\n優先度別集計:")
    priority_counts = df['priority'].value_counts()
    for priority, count in priority_counts.items():
        print(f"  優先度 {priority}: {count:,} ({count/len(df)*100:.1f}%)")

def analyze_disaster_types(df):
    """災害タイプの分析"""
    print("\n=== 災害タイプ分析 ===")
    
    # DC Reportのみを抽出
    dc_reports = df[df['report_type'] == 'DC Report'].copy()
    
    # メッセージから災害タイプを抽出
    disaster_types = []
    for message in dc_reports['message']:
        if '災危通報(気象)' in str(message):
            disaster_types.append('気象')
        elif '災危通報(震源)' in str(message):
            disaster_types.append('地震')
        elif '災危通報(海上)' in str(message):
            disaster_types.append('海上')
        elif '災危通報(洪水)' in str(message):
            disaster_types.append('洪水')
        else:
            disaster_types.append('その他')
    
    dc_reports['disaster_type'] = disaster_types
    
    print("災害タイプ別集計:")
    disaster_counts = Counter(disaster_types)
    for disaster_type, count in disaster_counts.items():
        print(f"  {disaster_type}: {count:,} ({count/len(disaster_types)*100:.1f}%)")
    
    return dc_reports

def analyze_temporal_patterns(df):
    """時間的パターンの分析"""
    print("\n=== 時間的パターン分析 ===")
    
    # 時間帯別集計
    hourly_counts = df['hour'].value_counts().sort_index()
    print("時間帯別レポート数:")
    for hour, count in hourly_counts.items():
        print(f"  {hour:02d}時: {count:,}")
    
    # 曜日別集計
    day_counts = df['day_of_week'].value_counts()
    print("\n曜日別レポート数:")
    for day, count in day_counts.items():
        print(f"  {day}: {count:,}")
    
    return hourly_counts, day_counts

def create_visualizations(df, dc_reports, hourly_counts, day_counts):
    """可視化の作成"""
    print("\n=== 可視化作成中 ===")
    
    # 図のサイズ設定
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('QZSS DCレポート分析結果', fontsize=16, fontweight='bold')
    
    # 1. レポートタイプ別分布
    report_type_counts = df['report_type'].value_counts()
    axes[0, 0].pie(report_type_counts.values, labels=report_type_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('レポートタイプ別分布')
    
    # 2. 衛星別分布
    satellite_counts = df['satellite'].value_counts()
    axes[0, 1].pie(satellite_counts.values, labels=satellite_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('衛星別分布')
    
    # 3. 時間帯別レポート数
    axes[1, 0].bar(hourly_counts.index, hourly_counts.values)
    axes[1, 0].set_title('時間帯別レポート数')
    axes[1, 0].set_xlabel('時間')
    axes[1, 0].set_ylabel('レポート数')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 日別レポート数
    daily_counts = df['date'].value_counts().sort_index()
    axes[1, 1].plot(daily_counts.index, daily_counts.values, marker='o')
    axes[1, 1].set_title('日別レポート数')
    axes[1, 1].set_xlabel('日付')
    axes[1, 1].set_ylabel('レポート数')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dc_reports_analysis.png', dpi=300, bbox_inches='tight')
    print("可視化を 'dc_reports_analysis.png' に保存しました")
    
    # 災害タイプ別の可視化（DC Reportのみ）
    if len(dc_reports) > 0:
        plt.figure(figsize=(10, 6))
        disaster_counts = dc_reports['disaster_type'].value_counts()
        plt.pie(disaster_counts.values, labels=disaster_counts.index, autopct='%1.1f%%')
        plt.title('災害タイプ別分布 (DC Reportのみ)')
        plt.savefig('disaster_types_analysis.png', dpi=300, bbox_inches='tight')
        print("災害タイプ分析を 'disaster_types_analysis.png' に保存しました")

def generate_summary_report(df, dc_reports):
    """サマリーレポートの生成"""
    print("\n=== サマリーレポート ===")
    
    # 基本情報
    total_records = len(df)
    dc_report_count = len(df[df['report_type'] == 'DC Report'])
    dcx_count = len(df[df['report_type'] == 'DCX'])
    
    # 期間情報
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    duration_days = (end_date - start_date).days
    
    # 衛星情報
    satellites = df['satellite'].unique()
    
    # 災害タイプ情報
    if len(dc_reports) > 0:
        disaster_types = dc_reports['disaster_type'].value_counts()
        most_common_disaster = disaster_types.index[0]
    else:
        disaster_types = pd.Series()
        most_common_disaster = "なし"
    
    # レポート生成
    report = f"""
# QZSS DCレポート分析結果

## 概要
- **分析期間**: {start_date.strftime('%Y年%m月%d日')} から {end_date.strftime('%Y年%m月%d日')}
- **データ期間**: {duration_days} 日間
- **総レコード数**: {total_records:,} 件

## レポートタイプ別内訳
- **DC Report**: {dc_report_count:,} 件 ({dc_report_count/total_records*100:.1f}%)
- **DCX**: {dcx_count:,} 件 ({dcx_count/total_records*100:.1f}%)

## 衛星別内訳
"""
    
    for satellite in satellites:
        count = len(df[df['satellite'] == satellite])
        report += f"- **{satellite}**: {count:,} 件 ({count/total_records*100:.1f}%)\n"
    
    if len(disaster_types) > 0:
        report += f"\n## 災害タイプ別内訳 (DC Reportのみ)\n"
        for disaster_type, count in disaster_types.items():
            report += f"- **{disaster_type}**: {count:,} 件 ({count/len(dc_reports)*100:.1f}%)\n"
        
        report += f"\n## 主要な災害タイプ\n"
        report += f"- **最多**: {most_common_disaster}\n"
    
    report += f"\n## 分析結果の要点\n"
    report += f"1. データは {duration_days} 日間にわたるQZSS DCレポートの記録です\n"
    report += f"2. 実際の災害・危機管理通報（DC Report）は {dc_report_count:,} 件です\n"
    report += f"3. テストメッセージ（DCX）は {dcx_count:,} 件です\n"
    
    if len(disaster_types) > 0:
        report += f"4. 最も多い災害タイプは「{most_common_disaster}」です\n"
    
    return report

def main():
    """メイン処理"""
    print("QZSS DCレポートデータ解析を開始します...")
    
    # データ読み込み
    df = load_and_clean_data('dc_reports_boot_00003.csv')
    
    # 基本統計分析
    analyze_basic_statistics(df)
    
    # 災害タイプ分析
    dc_reports = analyze_disaster_types(df)
    
    # 時間的パターン分析
    hourly_counts, day_counts = analyze_temporal_patterns(df)
    
    # 可視化作成
    create_visualizations(df, dc_reports, hourly_counts, day_counts)
    
    # サマリーレポート生成
    report = generate_summary_report(df, dc_reports)
    
    # レポートをファイルに保存
    with open('dc_reports_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n=== 分析完了 ===")
    print("以下のファイルが生成されました:")
    print("- dc_reports_analysis_report.md: 分析レポート")
    print("- dc_reports_analysis.png: 基本分析の可視化")
    print("- disaster_types_analysis.png: 災害タイプ分析の可視化")
    
    # レポート内容を表示
    print("\n" + "="*50)
    print("分析レポート")
    print("="*50)
    print(report)

if __name__ == "__main__":
    main()
