import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from typing import Dict

class RiskDashboard:
    """
    Creates visualizations and dashboards for risk assessment results.
    """
    
    def __init__(self):
        """
        Initialize the risk dashboard.
        """
        # Set up color schemes
        self.risk_cmap = LinearSegmentedColormap.from_list(
            'risk_colors', ['green', 'yellow', 'orange', 'red']
        )
        
        # Set default style for plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def create_summary_dashboard(self, df: pd.DataFrame, 
                                output_path: str, 
                                dataset_name: str,
                                risk_col: str = 'normalized_risk_score',
                                unit_col: str = 'unit_number',
                                time_col: str = 'time_cycles',
                                threshold: float = 0.7):
        """
        Create a comprehensive dashboard summary for a dataset.
        
        Args:
            df: DataFrame with risk assessment data
            output_path: Path to save dashboard
            dataset_name: Name of the dataset
            risk_col: Column name for risk score
            unit_col: Column name for unit identifier
            time_col: Column name for time
            threshold: High risk threshold
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Risk score distribution
        ax1 = fig.add_subplot(gs[0, 0:2])
        sns.histplot(df[risk_col], kde=True, ax=ax1, color='steelblue')
        ax1.axvline(x=threshold, color='red', linestyle='--', label=f'Alert Threshold ({threshold})')
        ax1.set_title('Risk Score Distribution', fontsize=14)
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Risk level pie chart
        ax2 = fig.add_subplot(gs[0, 2])
        risk_levels = df['risk_level'].value_counts()
        ax2.pie(risk_levels, labels=risk_levels.index, autopct='%1.1f%%', 
                colors=['green', 'yellow', 'orange', 'red'], startangle=90)
        ax2.set_title('Risk Level Distribution', fontsize=14)
        
        # Top 10 highest risk units
        ax3 = fig.add_subplot(gs[1, 0:2])
        top_units = df.groupby(unit_col)[risk_col].mean().sort_values(ascending=False).head(10)
        bars = sns.barplot(x=top_units.index, y=top_units.values, ax=ax3, palette='YlOrRd')
        ax3.set_title('Top 10 Units by Risk Score', fontsize=14)
        ax3.set_xlabel('Unit Number')
        ax3.set_ylabel('Average Risk Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add risk score values above bars
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 0.02,
                f'{top_units.values[i]:.3f}',
                ha='center', fontsize=8
            )
        
        # Risk score over time for top 5 units
        ax4 = fig.add_subplot(gs[1, 2])
        top5_units = top_units.head(5).index
        for unit in top5_units:
            unit_data = df[df[unit_col] == unit].sort_values(time_col)
            ax4.plot(unit_data[time_col], unit_data[risk_col], label=f'Unit {unit}')
        ax4.axhline(y=threshold, color='red', linestyle='--', label=f'Alert Threshold')
        ax4.set_title('Risk Score Trend (Top 5 Units)', fontsize=14)
        ax4.set_xlabel('Time Cycles')
        ax4.set_ylabel('Risk Score')
        ax4.legend()
        
        # Risk score vs time-to-next-stage scatter plot
        ax5 = fig.add_subplot(gs[2, 0])
        sc = ax5.scatter(df['predicted_time_to_next_stage'], df[risk_col], 
                     c=df['predicted_stage'], cmap='viridis', alpha=0.6)
        ax5.set_title('Risk Score vs Predicted Time', fontsize=14)
        ax5.set_xlabel('Predicted Time to Next Stage')
        ax5.set_ylabel('Risk Score')
        cbar = plt.colorbar(sc, ax=ax5)
        cbar.set_label('Current Stage')
        
        # Risk score heatmap by unit and time
        ax6 = fig.add_subplot(gs[2, 1:3])
        # Create pivot table for select units (top 10) and limited time range
        units_to_plot = top_units.index[:10]
        heatmap_data = df[df[unit_col].isin(units_to_plot)]
        if len(heatmap_data) > 0:
            pivot = pd.pivot_table(
                heatmap_data, 
                values=risk_col, 
                index=unit_col, 
                columns=time_col,
                aggfunc='mean',
                fill_value=0
            )
            # Keep only columns where there is data for at least one unit
            pivot = pivot.loc[:, pivot.sum() > 0]
            # Select every nth column to avoid crowding
            column_step = max(1, len(pivot.columns) // 20)
            pivot = pivot.iloc[:, ::column_step]
            
            sns.heatmap(pivot, cmap=self.risk_cmap, linewidths=0.5, ax=ax6)
            ax6.set_title('Risk Score Heatmap (Top 10 Units)', fontsize=14)
            ax6.set_xlabel('Time Cycles')
            ax6.set_ylabel('Unit Number')
        else:
            ax6.text(0.5, 0.5, 'Insufficient data for heatmap', 
                    ha='center', va='center', fontsize=12)
        
        # Dashboard title and metadata
        plt.suptitle(f'Risk Assessment Dashboard - {dataset_name}', fontsize=18, y=0.98)
        fig.text(0.5, 0.01, f'Total Units: {df[unit_col].nunique()} | ' 
                           f'High Risk Units: {df.groupby(unit_col)["risk_level"].apply(lambda x: "HIGH" in x.values).sum()} | '
                           f'Total Observations: {len(df)}', 
                ha='center', fontsize=12)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_path, 'risk_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    def create_unit_dashboard(self, df: pd.DataFrame, 
                            unit_number: int,
                            output_path: str, 
                            dataset_name: str,
                            risk_col: str = 'normalized_risk_score',
                            unit_col: str = 'unit_number',
                            time_col: str = 'time_cycles',
                            threshold: float = 0.7):
        """
        Create a dashboard for a specific unit.
        
        Args:
            df: DataFrame with risk assessment data
            unit_number: Unit number to analyze
            output_path: Path to save dashboard
            dataset_name: Name of the dataset
            risk_col: Column name for risk score
            unit_col: Column name for unit identifier
            time_col: Column name for time
            threshold: High risk threshold
        """
        # Filter data for the specific unit
        unit_df = df[df[unit_col] == unit_number].sort_values(time_col)
        
        if len(unit_df) == 0:
            print(f"No data found for unit {unit_number}")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Risk score over time
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.plot(unit_df[time_col], unit_df[risk_col], color='blue', marker='o')
        ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Alert Threshold ({threshold})')
        ax1.fill_between(unit_df[time_col], unit_df[risk_col], threshold, 
                        where=unit_df[risk_col] >= threshold, color='red', alpha=0.3)
        ax1.set_title(f'Risk Score Over Time - Unit {unit_number}', fontsize=14)
        ax1.set_xlabel('Time Cycles')
        ax1.set_ylabel('Risk Score')
        ax1.legend()
        
        # Current and predicted stage over time
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(unit_df[time_col], unit_df['predicted_stage'], color='purple', marker='o', label='Current Stage')
        ax2.set_title(f'Degradation Stage - Unit {unit_number}', fontsize=14)
        ax2.set_xlabel('Time Cycles')
        ax2.set_ylabel('Stage')
        ax2.set_yticks(range(5))  # Assuming 5 stages (0-4)
        ax2.legend()
        
        # Predicted time to next stage
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(unit_df[time_col], unit_df['predicted_time_to_next_stage'], color='green', marker='o')
        ax3.set_title(f'Predicted Time to Next Stage - Unit {unit_number}', fontsize=14)
        ax3.set_xlabel('Time Cycles')
        ax3.set_ylabel('Time to Next Stage (cycles)')
        
        # Risk level distribution pie chart
        ax4 = fig.add_subplot(gs[1, 1])
        risk_levels = unit_df['risk_level'].value_counts()
        wedges, texts, autotexts = ax4.pie(
            risk_levels, 
            labels=risk_levels.index, 
            autopct='%1.1f%%', 
            colors=['green', 'yellow', 'orange', 'red'],
            startangle=90,
            explode=[0.1 if level == 'HIGH' else 0 for level in risk_levels.index]
        )
        ax4.set_title(f'Risk Level Distribution - Unit {unit_number}', fontsize=14)
        
        # Risk metrics table
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')  # Turn off axis for table
        
        # Calculate risk metrics
        metrics = {
            'Maximum Risk Score': unit_df[risk_col].max(),
            'Average Risk Score': unit_df[risk_col].mean(),
            'Current Risk Score': unit_df.iloc[-1][risk_col],
            'Total Cycles': len(unit_df),
            'High Risk Cycles': sum(unit_df[risk_col] >= threshold),
            'Current Stage': unit_df.iloc[-1]['predicted_stage'],
            'Est. Time to Next Stage': unit_df.iloc[-1]['predicted_time_to_next_stage'],
            'Current Risk Level': unit_df.iloc[-1]['risk_level']
        }
        
        # Format metrics table
        table_data = [[k, f"{v:.4f}" if isinstance(v, float) else v] 
                    for k, v in metrics.items()]
        table = ax5.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            loc='center',
            cellLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        # Color the risk level cell based on its value
        risk_level_idx = [i for i, row in enumerate(table_data) if row[0] == 'Current Risk Level'][0] + 1
        risk_colors = {'HIGH': '#ff9999', 'MEDIUM': '#ffcc99', 'LOW': '#ffffb3', 'NEGLIGIBLE': '#ccffcc'}
        table[(risk_level_idx, 1)].set_facecolor(risk_colors.get(metrics['Current Risk Level'], 'white'))
        
        ax5.set_title(f'Risk Metrics - Unit {unit_number}', fontsize=14)
        
        # Dashboard title
        plt.suptitle(f'Unit {unit_number} Risk Dashboard - {dataset_name}', fontsize=16, y=0.98)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        unit_dir = os.path.join(output_path, 'unit_dashboards')
        os.makedirs(unit_dir, exist_ok=True)
        plt.savefig(os.path.join(unit_dir, f'unit_{unit_number}_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    def create_multi_dataset_comparison(self, 
                                       metrics: Dict[str, Dict],
                                       output_path: str):
        """
        Create visualizations comparing risk metrics across datasets.
        
        Args:
            metrics: Dictionary mapping dataset names to metrics dictionaries
            output_path: Path to save the comparison visualizations
        """
        if not metrics:
            return
            
        # Create a DataFrame from the metrics dictionary
        metrics_df = pd.DataFrame([
            {'dataset': dataset, **metrics_dict} 
            for dataset, metrics_dict in metrics.items()
        ])
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Mean risk score by dataset
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = sns.barplot(x='dataset', y='mean_risk_score', data=metrics_df, ax=ax1, palette='YlOrRd')
        ax1.set_title('Average Risk Score by Dataset', fontsize=14)
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Mean Risk Score')
        # Add values above bars
        for i, bar in enumerate(bars1.patches):
            bars1.text(
                bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 0.01,
                f'{bar.get_height():.3f}',
                ha='center', fontsize=9
            )
        
        # Percentage of high risk units by dataset
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = sns.barplot(x='dataset', y='percentage_high_risk_units', data=metrics_df, ax=ax2, palette='YlOrRd')
        ax2.set_title('Percentage of High Risk Units by Dataset', fontsize=14)
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Percentage of Units (%)')
        # Add values above bars
        for i, bar in enumerate(bars2.patches):
            bars2.text(
                bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 1,
                f'{bar.get_height():.1f}%',
                ha='center', fontsize=9
            )
        
        # Threshold comparison
        ax3 = fig.add_subplot(gs[1, 0])
        thresholds = metrics_df[['dataset', 'high_risk_threshold', 'medium_risk_threshold']]
        thresholds_melted = pd.melt(
            thresholds, 
            id_vars=['dataset'], 
            value_vars=['high_risk_threshold', 'medium_risk_threshold'],
            var_name='Threshold Type', 
            value_name='Threshold Value'
        )
        sns.barplot(x='dataset', y='Threshold Value', hue='Threshold Type', data=thresholds_melted, ax=ax3)
        ax3.set_title('Risk Thresholds by Dataset', fontsize=14)
        ax3.set_xlabel('Dataset')
        ax3.set_ylabel('Threshold Value')
        ax3.legend(title='')
        
        # Total high risk alerts by dataset
        ax4 = fig.add_subplot(gs[1, 1])
        bars4 = sns.barplot(x='dataset', y='total_high_risk_alerts', data=metrics_df, ax=ax4, palette='YlOrRd')
        ax4.set_title('Total High Risk Alerts by Dataset', fontsize=14)
        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Count')
        # Add values above bars
        for i, bar in enumerate(bars4.patches):
            bars4.text(
                bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 0.1 * bars4.get_ylim()[1],
                f'{int(bar.get_height())}',
                ha='center', fontsize=9
            )
        
        # Dashboard title
        plt.suptitle('Multi-Dataset Risk Comparison', fontsize=16, y=0.98)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_path, 'multi_dataset_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)