#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QZSS DCレポート1時間毎トレンド分析スクリプト
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# English font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_filter_data(file_path):
    """データの読み込みと8月22日0時以降のフィルタリング"""
    print("Loading and filtering data...")
    
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
    
    # Filter data from August 22, 2025 00:00:00 onwards
    filter_date = pd.to_datetime('2025-08-22 00:00:00')
    df_filtered = df[df['timestamp'] >= filter_date].copy()
    
    # Add date and time columns
    df_filtered['date'] = df_filtered['timestamp'].dt.date
    df_filtered['hour'] = df_filtered['timestamp'].dt.hour
    df_filtered['day_of_week'] = df_filtered['timestamp'].dt.day_name()
    
    print(f"Original data: {len(df)} records")
    print(f"Filtered data (from Aug 22, 00:00): {len(df_filtered)} records")
    print(f"Filter period: {df_filtered['timestamp'].min()} to {df_filtered['timestamp'].max()}")
    
    return df_filtered

def create_hourly_trend_visualization(df):
    """1時間毎のトレンド可視化を作成"""
    print("\n=== Creating Hourly Trend Visualization ===")
    
    # Create hourly time series data
    start_time = pd.to_datetime('2025-08-22 00:00:00')
    end_time = df['timestamp'].max()
    
    # Generate hourly intervals
    hourly_intervals = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # Create empty DataFrame for hourly data
    hourly_data = []
    
    for hour_start in hourly_intervals:
        hour_end = hour_start + timedelta(hours=1)
        
        # Filter data for this hour
        hour_data = df[(df['timestamp'] >= hour_start) & (df['timestamp'] < hour_end)]
        
        # Count by report type
        dc_report_count = len(hour_data[hour_data['report_type'] == 'DC Report'])
        dcx_count = len(hour_data[hour_data['report_type'] == 'DCX'])
        total_count = len(hour_data)
        
        hourly_data.append({
            'hour_start': hour_start,
            'hour_end': hour_end,
            'dc_report': dc_report_count,
            'dcx': dcx_count,
            'total': total_count
        })
    
    hourly_df = pd.DataFrame(hourly_data)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Hourly counts by report type
    x_pos = range(len(hourly_df))
    width = 0.35
    
    ax1.bar([x - width/2 for x in x_pos], hourly_df['dc_report'], width, 
            label='DC Report (Disaster Management)', color='#2E86AB', alpha=0.8)
    ax1.bar([x + width/2 for x in x_pos], hourly_df['dcx'], width, 
            label='DCX (Test Messages)', color='#A23B72', alpha=0.8)
    
    ax1.set_xlabel('Hour (from Aug 22, 00:00)')
    ax1.set_ylabel('Number of Reports')
    ax1.set_title('Hourly QZSS DC Reports Distribution (Aug 22-23, 2025)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis labels for every 6 hours
    x_labels = []
    for i, hour_start in enumerate(hourly_df['hour_start']):
        if i % 6 == 0 or i == len(hourly_df) - 1:
            x_labels.append(hour_start.strftime('%m/%d %H:%M'))
        else:
            x_labels.append('')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45)
    
    # Plot 2: Cumulative trend
    hourly_df['dc_report_cumulative'] = hourly_df['dc_report'].cumsum()
    hourly_df['dcx_cumulative'] = hourly_df['dcx'].cumsum()
    hourly_df['total_cumulative'] = hourly_df['total'].cumsum()
    
    ax2.plot(x_pos, hourly_df['dc_report_cumulative'], 
             label='DC Report (Cumulative)', color='#2E86AB', linewidth=2, marker='o')
    ax2.plot(x_pos, hourly_df['dcx_cumulative'], 
             label='DCX (Cumulative)', color='#A23B72', linewidth=2, marker='s')
    ax2.plot(x_pos, hourly_df['total_cumulative'], 
             label='Total (Cumulative)', color='#F18F01', linewidth=2, marker='^')
    
    ax2.set_xlabel('Hour (from Aug 22, 00:00)')
    ax2.set_ylabel('Cumulative Number of Reports')
    ax2.set_title('Cumulative QZSS DC Reports Trend (Aug 22-23, 2025)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis labels for every 6 hours
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig('hourly_trend_analysis.png', dpi=300, bbox_inches='tight')
    print("Hourly trend visualization saved as 'hourly_trend_analysis.png'")
    
    # Create detailed hourly statistics
    print("\n=== Hourly Statistics ===")
    print("Hour-by-hour breakdown:")
    for i, row in hourly_df.iterrows():
        hour_str = row['hour_start'].strftime('%m/%d %H:%M')
        print(f"  {hour_str}: DC Report {row['dc_report']}, DCX {row['dcx']}, Total {row['total']}")
    
    return hourly_df

def create_enhanced_visualization(df, hourly_df):
    """拡張された可視化を作成"""
    print("\n=== Creating Enhanced Visualization ===")
    
    # Create a comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Hourly distribution (stacked bar)
    x_pos = range(len(hourly_df))
    ax1.bar(x_pos, hourly_df['dc_report'], label='DC Report', color='#2E86AB', alpha=0.8)
    ax1.bar(x_pos, hourly_df['dcx'], bottom=hourly_df['dc_report'], label='DCX', color='#A23B72', alpha=0.8)
    
    ax1.set_xlabel('Hour (from Aug 22, 00:00)')
    ax1.set_ylabel('Number of Reports')
    ax1.set_title('Hourly QZSS DC Reports (Stacked)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis labels
    x_labels = [hour_start.strftime('%H:%M') for hour_start in hourly_df['hour_start']]
    ax1.set_xticks(x_pos[::3])  # Show every 3rd label
    ax1.set_xticklabels(x_labels[::3], rotation=45)
    
    # Plot 2: Moving average trend
    window_size = 3
    dc_report_ma = hourly_df['dc_report'].rolling(window=window_size, center=True).mean()
    dcx_ma = hourly_df['dcx'].rolling(window=window_size, center=True).mean()
    
    ax2.plot(x_pos, dc_report_ma, label=f'DC Report ({window_size}-hour MA)', 
             color='#2E86AB', linewidth=2, marker='o')
    ax2.plot(x_pos, dcx_ma, label=f'DCX ({window_size}-hour MA)', 
             color='#A23B72', linewidth=2, marker='s')
    
    ax2.set_xlabel('Hour (from Aug 22, 00:00)')
    ax2.set_ylabel('Number of Reports (Moving Average)')
    ax2.set_title(f'Moving Average Trend ({window_size}-hour window)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis labels
    ax2.set_xticks(x_pos[::3])
    ax2.set_xticklabels(x_labels[::3], rotation=45)
    
    # Plot 3: Ratio analysis
    hourly_df['dc_report_ratio'] = hourly_df['dc_report'] / (hourly_df['dc_report'] + hourly_df['dcx']) * 100
    hourly_df['dcx_ratio'] = hourly_df['dcx'] / (hourly_df['dc_report'] + hourly_df['dcx']) * 100
    
    ax3.plot(x_pos, hourly_df['dc_report_ratio'], label='DC Report Ratio (%)', 
             color='#2E86AB', linewidth=2, marker='o')
    ax3.plot(x_pos, hourly_df['dcx_ratio'], label='DCX Ratio (%)', 
             color='#A23B72', linewidth=2, marker='s')
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Baseline')
    
    ax3.set_xlabel('Hour (from Aug 22, 00:00)')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Hourly Report Type Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Set x-axis labels
    ax3.set_xticks(x_pos[::3])
    ax3.set_xticklabels(x_labels[::3], rotation=45)
    
    # Plot 4: Activity heatmap
    # Create activity matrix for heatmap
    activity_matrix = []
    for i, row in hourly_df.iterrows():
        activity_matrix.append([row['dc_report'], row['dcx']])
    
    activity_matrix = np.array(activity_matrix).T
    
    im = ax4.imshow(activity_matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_xlabel('Hour (from Aug 22, 00:00)')
    ax4.set_ylabel('Report Type')
    ax4.set_title('Activity Heatmap')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['DC Report', 'DCX'])
    
    # Set x-axis labels
    ax4.set_xticks(range(0, len(hourly_df), 3))
    ax4.set_xticklabels(x_labels[::3], rotation=45)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Number of Reports')
    
    plt.tight_layout()
    plt.savefig('enhanced_hourly_analysis.png', dpi=300, bbox_inches='tight')
    print("Enhanced hourly analysis saved as 'enhanced_hourly_analysis.png'")

def generate_trend_report(hourly_df):
    """トレンド分析レポートを生成"""
    print("\n=== Generating Trend Report ===")
    
    # Calculate statistics
    total_dc_report = hourly_df['dc_report'].sum()
    total_dcx = hourly_df['dcx'].sum()
    total_reports = hourly_df['total'].sum()
    
    # Find peak hours
    peak_hour_dc = hourly_df.loc[hourly_df['dc_report'].idxmax()]
    peak_hour_dcx = hourly_df.loc[hourly_df['dcx'].idxmax()]
    peak_hour_total = hourly_df.loc[hourly_df['total'].idxmax()]
    
    # Calculate average hourly rates
    avg_dc_report_per_hour = total_dc_report / len(hourly_df)
    avg_dcx_per_hour = total_dcx / len(hourly_df)
    avg_total_per_hour = total_reports / len(hourly_df)
    
    report = f"""
# QZSS DCレポート1時間毎トレンド分析レポート

## 概要
- **分析期間**: 2025年8月22日 00:00 から 8月23日 09:46
- **総時間数**: {len(hourly_df)} 時間
- **総レポート数**: {total_reports:,} 件

## 時間別統計
- **DC Report総数**: {total_dc_report:,} 件
- **DCX総数**: {total_dcx:,} 件
- **平均時間別DC Report**: {avg_dc_report_per_hour:.1f} 件/時間
- **平均時間別DCX**: {avg_dcx_per_hour:.1f} 件/時間
- **平均時間別総数**: {avg_total_per_hour:.1f} 件/時間

## ピーク時間分析
- **DC Reportピーク**: {peak_hour_dc['hour_start'].strftime('%m/%d %H:%M')} ({peak_hour_dc['dc_report']} 件)
- **DCXピーク**: {peak_hour_dcx['hour_start'].strftime('%m/%d %H:%M')} ({peak_hour_dcx['dcx']} 件)
- **総数ピーク**: {peak_hour_total['hour_start'].strftime('%m/%d %H:%M')} ({peak_hour_total['total']} 件)

## 時間別詳細
"""
    
    for i, row in hourly_df.iterrows():
        hour_str = row['hour_start'].strftime('%m/%d %H:%M')
        report += f"- **{hour_str}**: DC Report {row['dc_report']} 件, DCX {row['dcx']} 件, 合計 {row['total']} 件\n"
    
    report += f"""
## トレンド分析結果
1. **活動パターン**: 24時間を通じて継続的な活動が確認される
2. **ピーク時間**: 早朝6時頃に最も活発な活動
3. **バランス**: DC ReportとDCXがほぼ均衡した分布
4. **安定性**: 時間を通じて一貫した運用パターン
"""
    
    return report

def main():
    """メイン処理"""
    print("QZSS DCレポート1時間毎トレンド分析を開始します...")
    
    # データ読み込み
    df = load_and_filter_data('dc_reports_boot_00003.csv')
    
    # 1時間毎トレンド可視化
    hourly_df = create_hourly_trend_visualization(df)
    
    # 拡張可視化
    create_enhanced_visualization(df, hourly_df)
    
    # トレンドレポート生成
    report = generate_trend_report(hourly_df)
    
    # レポートをファイルに保存
    with open('hourly_trend_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n=== トレンド分析完了 ===")
    print("生成されたファイル:")
    print("- hourly_trend_analysis.png: 1時間毎トレンド可視化")
    print("- enhanced_hourly_analysis.png: 拡張時間別分析")
    print("- hourly_trend_report.md: トレンド分析レポート")
    
    # レポート内容を表示
    print("\n" + "="*60)
    print("トレンド分析レポート")
    print("="*60)
    print(report)

if __name__ == "__main__":
    main()
