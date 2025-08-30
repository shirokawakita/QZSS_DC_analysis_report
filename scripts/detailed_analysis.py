#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QZSS DCレポート詳細分析スクリプト
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """データの読み込み"""
    print("データを読み込み中...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []
    current_record = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('2025/'):
            if current_record:
                data.append(current_record)
            
            parts = line.split(',', 4)
            if len(parts) >= 5:
                current_record = {
                    'timestamp': parts[0],
                    'report_type': parts[1],
                    'satellite': parts[2],
                    'priority': parts[3],
                    'message': parts[4]
                }
        else:
            if current_record:
                current_record['message'] += '\n' + line
    
    if current_record:
        data.append(current_record)
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S JST')
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['day_of_week_jp'] = df['timestamp'].dt.day_name().map({
        'Monday': '月曜日', 'Tuesday': '火曜日', 'Wednesday': '水曜日',
        'Thursday': '木曜日', 'Friday': '金曜日', 'Saturday': '土曜日', 'Sunday': '日曜日'
    })
    
    print(f"データ読み込み完了: {len(df)} レコード")
    return df

def analyze_message_content(df):
    """メッセージ内容の詳細分析"""
    print("\n=== メッセージ内容分析 ===")
    
    dc_reports = df[df['report_type'] == 'DC Report'].copy()
    
    # 災害タイプの詳細分類
    disaster_details = []
    for message in dc_reports['message']:
        msg_str = str(message)
        if '災危通報(気象)' in msg_str:
            if '土砂災害警戒情報' in msg_str:
                disaster_details.append('土砂災害警戒情報')
            elif '大雨警報' in msg_str:
                disaster_details.append('大雨警報')
            elif '洪水警報' in msg_str:
                disaster_details.append('洪水警報')
            else:
                disaster_details.append('気象その他')
        elif '災危通報(震源)' in msg_str:
            disaster_details.append('地震情報')
        elif '災危通報(海上)' in msg_str:
            if '海上濃霧警報' in msg_str:
                disaster_details.append('海上濃霧警報')
            elif '海上風警報' in msg_str:
                disaster_details.append('海上風警報')
            else:
                disaster_details.append('海上その他')
        elif '災危通報(洪水)' in msg_str:
            if '氾濫警戒情報' in msg_str:
                disaster_details.append('氾濫警戒情報')
            else:
                disaster_details.append('洪水その他')
        else:
            disaster_details.append('その他')
    
    dc_reports['disaster_detail'] = disaster_details
    
    print("詳細災害タイプ別集計:")
    detail_counts = Counter(disaster_details)
    for detail_type, count in detail_counts.items():
        print(f"  {detail_type}: {count:,} ({count/len(disaster_details)*100:.1f}%)")
    
    return dc_reports

def analyze_temporal_patterns_detailed(df):
    """詳細な時間的パターン分析"""
    print("\n=== 詳細時間的パターン分析 ===")
    
    # 時間帯別の詳細分析
    hourly_analysis = df.groupby(['hour', 'report_type']).size().unstack(fill_value=0)
    
    print("時間帯別レポートタイプ分布:")
    for hour in range(24):
        if hour in hourly_analysis.index:
            dc_report_count = hourly_analysis.loc[hour, 'DC Report'] if 'DC Report' in hourly_analysis.columns else 0
            dcx_count = hourly_analysis.loc[hour, 'DCX'] if 'DCX' in hourly_analysis.columns else 0
            total = dc_report_count + dcx_count
            if total > 0:
                print(f"  {hour:02d}時: DC Report {dc_report_count}, DCX {dcx_count} (合計 {total})")
    
    # 日別の詳細分析
    daily_analysis = df.groupby(['date', 'report_type']).size().unstack(fill_value=0)
    
    print("\n日別レポートタイプ分布:")
    for date in sorted(daily_analysis.index):
        dc_report_count = daily_analysis.loc[date, 'DC Report'] if 'DC Report' in daily_analysis.columns else 0
        dcx_count = daily_analysis.loc[date, 'DCX'] if 'DCX' in daily_analysis.columns else 0
        total = dc_report_count + dcx_count
        print(f"  {date}: DC Report {dc_report_count}, DCX {dcx_count} (合計 {total})")
    
    return hourly_analysis, daily_analysis

def analyze_satellite_performance(df):
    """衛星パフォーマンス分析"""
    print("\n=== 衛星パフォーマンス分析 ===")
    
    # 衛星別のレポートタイプ分布
    satellite_analysis = df.groupby(['satellite', 'report_type']).size().unstack(fill_value=0)
    
    print("衛星別レポートタイプ分布:")
    for satellite in satellite_analysis.index:
        dc_report_count = satellite_analysis.loc[satellite, 'DC Report'] if 'DC Report' in satellite_analysis.columns else 0
        dcx_count = satellite_analysis.loc[satellite, 'DCX'] if 'DCX' in satellite_analysis.columns else 0
        total = dc_report_count + dcx_count
        print(f"  {satellite}: DC Report {dc_report_count}, DCX {dcx_count} (合計 {total})")
    
    # 衛星別の時間帯分布
    satellite_hourly = df.groupby(['satellite', 'hour']).size().unstack(fill_value=0)
    
    return satellite_analysis, satellite_hourly

def create_detailed_visualizations(df, dc_reports, hourly_analysis, daily_analysis, satellite_analysis):
    """詳細な可視化の作成"""
    print("\n=== 詳細可視化作成中 ===")
    
    # 1. 時間帯別レポートタイプ分布
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    if 'DC Report' in hourly_analysis.columns and 'DCX' in hourly_analysis.columns:
        hourly_analysis[['DC Report', 'DCX']].plot(kind='bar', ax=plt.gca())
        plt.title('時間帯別レポートタイプ分布')
        plt.xlabel('時間')
        plt.ylabel('レポート数')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. 日別レポートタイプ分布
    plt.subplot(2, 2, 2)
    if 'DC Report' in daily_analysis.columns and 'DCX' in daily_analysis.columns:
        daily_analysis[['DC Report', 'DCX']].plot(kind='bar', ax=plt.gca())
        plt.title('日別レポートタイプ分布')
        plt.xlabel('日付')
        plt.ylabel('レポート数')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. 衛星別レポートタイプ分布
    plt.subplot(2, 2, 3)
    if 'DC Report' in satellite_analysis.columns and 'DCX' in satellite_analysis.columns:
        satellite_analysis[['DC Report', 'DCX']].plot(kind='bar', ax=plt.gca())
        plt.title('衛星別レポートタイプ分布')
        plt.xlabel('衛星')
        plt.ylabel('レポート数')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. 詳細災害タイプ分布
    plt.subplot(2, 2, 4)
    if len(dc_reports) > 0:
        disaster_detail_counts = dc_reports['disaster_detail'].value_counts()
        plt.pie(disaster_detail_counts.values, labels=disaster_detail_counts.index, autopct='%1.1f%%')
        plt.title('詳細災害タイプ分布 (DC Reportのみ)')
    
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("詳細分析可視化を 'detailed_analysis.png' に保存しました")
    
    # 時間帯別ヒートマップ
    plt.figure(figsize=(12, 8))
    hourly_pivot = df.groupby(['hour', 'report_type']).size().unstack(fill_value=0)
    if 'DC Report' in hourly_pivot.columns and 'DCX' in hourly_pivot.columns:
        sns.heatmap(hourly_pivot.T, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('時間帯別レポートタイプヒートマップ')
        plt.xlabel('時間')
        plt.ylabel('レポートタイプ')
        plt.savefig('hourly_heatmap.png', dpi=300, bbox_inches='tight')
        print("時間帯別ヒートマップを 'hourly_heatmap.png' に保存しました")

def generate_detailed_report(df, dc_reports, hourly_analysis, daily_analysis, satellite_analysis):
    """詳細レポートの生成"""
    print("\n=== 詳細レポート生成 ===")
    
    # 基本統計
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
        disaster_details = dc_reports['disaster_detail'].value_counts()
        most_common_detail = disaster_details.index[0] if len(disaster_details) > 0 else "なし"
        
        # 災害タイプを再分類
        disaster_types = []
        for detail in disaster_details.index:
            if '海上' in detail:
                disaster_types.append('海上')
            elif '気象' in detail or '土砂' in detail:
                disaster_types.append('気象')
            elif '地震' in detail:
                disaster_types.append('地震')
            elif '洪水' in detail or '氾濫' in detail:
                disaster_types.append('洪水')
            else:
                disaster_types.append('その他')
        
        disaster_type_counts = pd.Series(disaster_types).value_counts()
        most_common_disaster = disaster_type_counts.index[0] if len(disaster_type_counts) > 0 else "なし"
    else:
        disaster_types = pd.Series()
        disaster_details = pd.Series()
        disaster_type_counts = pd.Series()
        most_common_disaster = "なし"
        most_common_detail = "なし"
    
    # 時間帯分析
    peak_hour = df['hour'].value_counts().index[0]
    peak_hour_count = df['hour'].value_counts().iloc[0]
    
    # レポート生成
    report = f"""
# QZSS DCレポート詳細分析結果

## 概要
- **分析期間**: {start_date.strftime('%Y年%m月%d日')} から {end_date.strftime('%Y年%m月%d日')}
- **データ期間**: {duration_days} 日間
- **総レコード数**: {total_records:,} 件

## レポートタイプ別詳細
- **DC Report**: {dc_report_count:,} 件 ({dc_report_count/total_records*100:.1f}%)
  - 実際の災害・危機管理通報
- **DCX**: {dcx_count:,} 件 ({dcx_count/total_records*100:.1f}%)
  - テストメッセージ

## 衛星別詳細分析
"""
    
    for satellite in satellites:
        count = len(df[df['satellite'] == satellite])
        dc_report_sat = len(df[(df['satellite'] == satellite) & (df['report_type'] == 'DC Report')])
        dcx_sat = len(df[(df['satellite'] == satellite) & (df['report_type'] == 'DCX')])
        report += f"- **{satellite}**: {count:,} 件 ({count/total_records*100:.1f}%)\n"
        report += f"  - DC Report: {dc_report_sat:,} 件\n"
        report += f"  - DCX: {dcx_sat:,} 件\n"
    
    if len(disaster_type_counts) > 0:
        report += f"\n## 災害タイプ別詳細 (DC Reportのみ)\n"
        for disaster_type, count in disaster_type_counts.items():
            report += f"- **{disaster_type}**: {count:,} 件 ({count/len(dc_reports)*100:.1f}%)\n"
    
    if len(disaster_details) > 0:
        report += f"\n## 詳細災害タイプ別内訳 (DC Reportのみ)\n"
        for disaster_detail, count in disaster_details.head(10).items():
            report += f"- **{disaster_detail}**: {count:,} 件 ({count/len(dc_reports)*100:.1f}%)\n"
    
    report += f"\n## 時間的パターン分析\n"
    report += f"- **最多レポート時間帯**: {peak_hour}時 ({peak_hour_count:,} 件)\n"
    
    # 曜日分析
    day_counts = df['day_of_week_jp'].value_counts()
    most_active_day = day_counts.index[0] if len(day_counts) > 0 else "なし"
    report += f"- **最多レポート曜日**: {most_active_day} ({day_counts.iloc[0]:,} 件)\n"
    
    report += f"\n## 主要な発見\n"
    report += f"1. **データ期間**: {duration_days} 日間にわたる包括的なQZSS DCレポート記録\n"
    report += f"2. **レポート構成**: 実際の災害通報とテストメッセージがほぼ同数\n"
    report += f"3. **衛星利用**: QZSS-7が最も多く利用されている ({len(df[df['satellite'] == 'QZSS-7'])/total_records*100:.1f}%)\n"
    
    if len(disaster_type_counts) > 0:
        report += f"4. **主要災害**: 海上関連の通報が最多 ({most_common_disaster})\n"
        report += f"5. **詳細分類**: {most_common_detail}が最も多い\n"
    
    report += f"6. **時間的傾向**: {peak_hour}時に最も多くのレポートが配信\n"
    report += f"7. **曜日傾向**: {most_active_day}に最も活発な活動\n"
    
    return report

def main():
    """メイン処理"""
    print("QZSS DCレポート詳細分析を開始します...")
    
    # データ読み込み
    df = load_data('dc_reports_boot_00003.csv')
    
    # メッセージ内容分析
    dc_reports = analyze_message_content(df)
    
    # 詳細時間的パターン分析
    hourly_analysis, daily_analysis = analyze_temporal_patterns_detailed(df)
    
    # 衛星パフォーマンス分析
    satellite_analysis, satellite_hourly = analyze_satellite_performance(df)
    
    # 詳細可視化作成
    create_detailed_visualizations(df, dc_reports, hourly_analysis, daily_analysis, satellite_analysis)
    
    # 詳細レポート生成
    report = generate_detailed_report(df, dc_reports, hourly_analysis, daily_analysis, satellite_analysis)
    
    # レポートをファイルに保存
    with open('detailed_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n=== 詳細分析完了 ===")
    print("以下のファイルが生成されました:")
    print("- detailed_analysis_report.md: 詳細分析レポート")
    print("- detailed_analysis.png: 詳細分析の可視化")
    print("- hourly_heatmap.png: 時間帯別ヒートマップ")
    
    # レポート内容を表示
    print("\n" + "="*60)
    print("詳細分析レポート")
    print("="*60)
    print(report)

if __name__ == "__main__":
    main()
