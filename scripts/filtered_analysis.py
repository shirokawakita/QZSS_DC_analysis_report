#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QZSS DCレポート分析スクリプト（8月22日0時以降フィルタリング版）
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

def analyze_basic_statistics(df):
    """Basic statistical analysis"""
    print("\n=== Basic Statistics ===")
    print(f"Total records: {len(df):,}")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.1f} hours")
    
    print(f"\nReport type distribution:")
    report_type_counts = df['report_type'].value_counts()
    for report_type, count in report_type_counts.items():
        print(f"  {report_type}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\nSatellite distribution:")
    satellite_counts = df['satellite'].value_counts()
    for satellite, count in satellite_counts.items():
        print(f"  {satellite}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\nPriority distribution:")
    priority_counts = df['priority'].value_counts()
    for priority, count in priority_counts.items():
        print(f"  Priority {priority}: {count:,} ({count/len(df)*100:.1f}%)")

def analyze_disaster_types(df):
    """Disaster type analysis"""
    print("\n=== Disaster Type Analysis ===")
    
    # Extract DC Report only
    dc_reports = df[df['report_type'] == 'DC Report'].copy()
    
    # Extract disaster types from messages
    disaster_types = []
    for message in dc_reports['message']:
        msg_str = str(message)
        if '災危通報(気象)' in msg_str:
            disaster_types.append('Weather')
        elif '災危通報(震源)' in msg_str:
            disaster_types.append('Earthquake')
        elif '災危通報(海上)' in msg_str:
            disaster_types.append('Marine')
        elif '災危通報(洪水)' in msg_str:
            disaster_types.append('Flood')
        else:
            disaster_types.append('Other')
    
    dc_reports['disaster_type'] = disaster_types
    
    print("Disaster type distribution:")
    disaster_counts = Counter(disaster_types)
    for disaster_type, count in disaster_counts.items():
        print(f"  {disaster_type}: {count:,} ({count/len(disaster_types)*100:.1f}%)")
    
    return dc_reports

def analyze_disaster_details(df):
    """Detailed disaster type analysis"""
    print("\n=== Detailed Disaster Type Analysis ===")
    
    dc_reports = df[df['report_type'] == 'DC Report'].copy()
    
    # Detailed disaster type classification
    disaster_details = []
    for message in dc_reports['message']:
        msg_str = str(message)
        if '災危通報(気象)' in msg_str:
            if '土砂災害警戒情報' in msg_str:
                disaster_details.append('Sediment Disaster Warning')
            elif '大雨警報' in msg_str:
                disaster_details.append('Heavy Rain Warning')
            elif '洪水警報' in msg_str:
                disaster_details.append('Flood Warning')
            else:
                disaster_details.append('Weather Other')
        elif '災危通報(震源)' in msg_str:
            disaster_details.append('Earthquake Information')
        elif '災危通報(海上)' in msg_str:
            if '海上濃霧警報' in msg_str:
                disaster_details.append('Marine Dense Fog Warning')
            elif '海上風警報' in msg_str:
                disaster_details.append('Marine Wind Warning')
            else:
                disaster_details.append('Marine Other')
        elif '災危通報(洪水)' in msg_str:
            if '氾濫警戒情報' in msg_str:
                disaster_details.append('Flood Risk Information')
            else:
                disaster_details.append('Flood Other')
        else:
            disaster_details.append('Other')
    
    dc_reports['disaster_detail'] = disaster_details
    
    print("Detailed disaster type distribution:")
    detail_counts = Counter(disaster_details)
    for detail_type, count in detail_counts.items():
        print(f"  {detail_type}: {count:,} ({count/len(disaster_details)*100:.1f}%)")
    
    return dc_reports

def analyze_temporal_patterns(df):
    """Temporal pattern analysis"""
    print("\n=== Temporal Pattern Analysis ===")
    
    # Hourly distribution
    hourly_counts = df['hour'].value_counts().sort_index()
    print("Hourly report distribution:")
    for hour, count in hourly_counts.items():
        print(f"  {hour:02d}:00: {count:,}")
    
    # Day of week distribution
    day_counts = df['day_of_week'].value_counts()
    print("\nDay of week distribution:")
    for day, count in day_counts.items():
        print(f"  {day}: {count:,}")
    
    return hourly_counts, day_counts

def create_visualizations(df, dc_reports, hourly_counts, day_counts):
    """Create visualizations in English"""
    print("\n=== Creating Visualizations ===")
    
    # Set style
    plt.style.use('default')
    
    # Create main analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('QZSS DC Reports Analysis (From Aug 22, 00:00)', fontsize=16, fontweight='bold')
    
    # 1. Report type distribution
    report_type_counts = df['report_type'].value_counts()
    axes[0, 0].pie(report_type_counts.values, labels=report_type_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Report Type Distribution')
    
    # 2. Satellite distribution
    satellite_counts = df['satellite'].value_counts()
    axes[0, 1].pie(satellite_counts.values, labels=satellite_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Satellite Distribution')
    
    # 3. Hourly report distribution
    axes[1, 0].bar(hourly_counts.index, hourly_counts.values)
    axes[1, 0].set_title('Hourly Report Distribution')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Number of Reports')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Daily report distribution
    daily_counts = df['date'].value_counts().sort_index()
    axes[1, 1].plot(daily_counts.index, daily_counts.values, marker='o')
    axes[1, 1].set_title('Daily Report Distribution')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Number of Reports')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('filtered_analysis_main.png', dpi=300, bbox_inches='tight')
    print("Main analysis visualization saved as 'filtered_analysis_main.png'")
    
    # Disaster type visualization (DC Report only)
    if len(dc_reports) > 0:
        # Create subplots for disaster types
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Disaster type distribution (recalculate from disaster_detail)
        disaster_types = []
        for detail in dc_reports['disaster_detail']:
            if 'Marine' in detail:
                disaster_types.append('Marine')
            elif 'Weather' in detail or 'Sediment' in detail:
                disaster_types.append('Weather')
            elif 'Earthquake' in detail:
                disaster_types.append('Earthquake')
            elif 'Flood' in detail:
                disaster_types.append('Flood')
            else:
                disaster_types.append('Other')
        
        disaster_counts = pd.Series(disaster_types).value_counts()
        ax1.pie(disaster_counts.values, labels=disaster_counts.index, autopct='%1.1f%%')
        ax1.set_title('Disaster Type Distribution (DC Report Only)')
        
        # Detailed disaster type distribution
        disaster_detail_counts = dc_reports['disaster_detail'].value_counts()
        ax2.pie(disaster_detail_counts.values, labels=disaster_detail_counts.index, autopct='%1.1f%%')
        ax2.set_title('Detailed Disaster Type Distribution (DC Report Only)')
        
        plt.tight_layout()
        plt.savefig('filtered_disaster_analysis.png', dpi=300, bbox_inches='tight')
        print("Disaster analysis visualization saved as 'filtered_disaster_analysis.png'")
    
    # Hourly heatmap
    plt.figure(figsize=(12, 8))
    hourly_pivot = df.groupby(['hour', 'report_type']).size().unstack(fill_value=0)
    if 'DC Report' in hourly_pivot.columns and 'DCX' in hourly_pivot.columns:
        sns.heatmap(hourly_pivot.T, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Hourly Report Type Heatmap')
        plt.xlabel('Hour')
        plt.ylabel('Report Type')
        plt.savefig('filtered_hourly_heatmap.png', dpi=300, bbox_inches='tight')
        print("Hourly heatmap saved as 'filtered_hourly_heatmap.png'")

def generate_summary_report(df, dc_reports):
    """Generate summary report"""
    print("\n=== Generating Summary Report ===")
    
    # Basic statistics
    total_records = len(df)
    dc_report_count = len(df[df['report_type'] == 'DC Report'])
    dcx_count = len(df[df['report_type'] == 'DCX'])
    
    # Period information
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    duration_hours = (end_date - start_date).total_seconds() / 3600
    
    # Satellite information
    satellites = df['satellite'].unique()
    
    # Disaster type information
    if len(dc_reports) > 0:
        disaster_details = dc_reports['disaster_detail'].value_counts()
        most_common_detail = disaster_details.index[0] if len(disaster_details) > 0 else "None"
        
        # Recalculate disaster types from disaster_detail
        disaster_types = []
        for detail in disaster_details.index:
            if 'Marine' in detail:
                disaster_types.append('Marine')
            elif 'Weather' in detail or 'Sediment' in detail:
                disaster_types.append('Weather')
            elif 'Earthquake' in detail:
                disaster_types.append('Earthquake')
            elif 'Flood' in detail:
                disaster_types.append('Flood')
            else:
                disaster_types.append('Other')
        
        disaster_type_counts = pd.Series(disaster_types).value_counts()
        most_common_disaster = disaster_type_counts.index[0] if len(disaster_type_counts) > 0 else "None"
    else:
        disaster_types = pd.Series()
        disaster_details = pd.Series()
        disaster_type_counts = pd.Series()
        most_common_disaster = "None"
        most_common_detail = "None"
    
    # Time analysis
    peak_hour = df['hour'].value_counts().index[0]
    peak_hour_count = df['hour'].value_counts().iloc[0]
    
    # Generate report
    report = f"""
# QZSS DC Reports Analysis Report (From Aug 22, 00:00)

## Overview
- **Analysis Period**: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}
- **Duration**: {duration_hours:.1f} hours
- **Total Records**: {total_records:,}

## Report Type Distribution
- **DC Report**: {dc_report_count:,} ({dc_report_count/total_records*100:.1f}%)
  - Actual disaster and crisis management reports
- **DCX**: {dcx_count:,} ({dcx_count/total_records*100:.1f}%)
  - Test messages

## Satellite Distribution
"""
    
    for satellite in satellites:
        count = len(df[df['satellite'] == satellite])
        dc_report_sat = len(df[(df['satellite'] == satellite) & (df['report_type'] == 'DC Report')])
        dcx_sat = len(df[(df['satellite'] == satellite) & (df['report_type'] == 'DCX')])
        report += f"- **{satellite}**: {count:,} ({count/total_records*100:.1f}%)\n"
        report += f"  - DC Report: {dc_report_sat:,}\n"
        report += f"  - DCX: {dcx_sat:,}\n"
    
    if len(disaster_type_counts) > 0:
        report += f"\n## Disaster Type Distribution (DC Report Only)\n"
        for disaster_type, count in disaster_type_counts.items():
            report += f"- **{disaster_type}**: {count:,} ({count/len(dc_reports)*100:.1f}%)\n"
    
    if len(disaster_details) > 0:
        report += f"\n## Detailed Disaster Type Distribution (DC Report Only)\n"
        for disaster_detail, count in disaster_details.head(10).items():
            report += f"- **{disaster_detail}**: {count:,} ({count/len(dc_reports)*100:.1f}%)\n"
    
    report += f"\n## Temporal Pattern Analysis\n"
    report += f"- **Peak Hour**: {peak_hour}:00 ({peak_hour_count:,} reports)\n"
    
    # Day analysis
    day_counts = df['day_of_week'].value_counts()
    most_active_day = day_counts.index[0] if len(day_counts) > 0 else "None"
    report += f"- **Most Active Day**: {most_active_day} ({day_counts.iloc[0]:,} reports)\n"
    
    report += f"\n## Key Findings\n"
    report += f"1. **Data Period**: {duration_hours:.1f} hours of QZSS DC reports from Aug 22, 00:00\n"
    report += f"2. **Report Composition**: Nearly equal distribution between actual reports and test messages\n"
    report += f"3. **Satellite Usage**: QZSS-7 is the most utilized satellite ({len(df[df['satellite'] == 'QZSS-7'])/total_records*100:.1f}%)\n"
    
    if len(disaster_type_counts) > 0:
        report += f"4. **Primary Disaster Type**: {most_common_disaster} is the most frequent\n"
        report += f"5. **Detailed Classification**: {most_common_detail} is the most common\n"
    
    report += f"6. **Temporal Trend**: Most reports are sent at {peak_hour}:00\n"
    report += f"7. **Daily Pattern**: {most_active_day} shows the most active operation\n"
    
    return report

def main():
    """Main processing"""
    print("Starting QZSS DC Reports Analysis (Filtered from Aug 22, 00:00)...")
    
    # Load and filter data
    df = load_and_filter_data('dc_reports_boot_00003.csv')
    
    # Basic statistical analysis
    analyze_basic_statistics(df)
    
    # Disaster type analysis
    dc_reports = analyze_disaster_types(df)
    
    # Detailed disaster analysis
    dc_reports_detailed = analyze_disaster_details(df)
    
    # Temporal pattern analysis
    hourly_counts, day_counts = analyze_temporal_patterns(df)
    
    # Create visualizations
    create_visualizations(df, dc_reports_detailed, hourly_counts, day_counts)
    
    # Generate summary report
    report = generate_summary_report(df, dc_reports_detailed)
    
    # Save report to file
    with open('filtered_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- filtered_analysis_report.md: Analysis report")
    print("- filtered_analysis_main.png: Main analysis visualization")
    print("- filtered_disaster_analysis.png: Disaster type analysis")
    print("- filtered_hourly_heatmap.png: Hourly heatmap")
    
    # Display report content
    print("\n" + "="*60)
    print("Analysis Report")
    print("="*60)
    print(report)

if __name__ == "__main__":
    main()
