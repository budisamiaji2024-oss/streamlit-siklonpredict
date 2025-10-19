import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import SelectFromModel
import warnings
import os
import io
import base64
from datetime import datetime

warnings.filterwarnings('ignore')

# Model class tetap sama
class FixedBoostedCyclonePredictor:
    def __init__(self):
        self.models = {}
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.feature_selector = None
        self.feature_columns = []
        self.all_feature_columns = []
        self.selected_feature_columns = []
        self.target_columns = ['next_LAT', 'next_LON', 'next_WindSpeed', 'next_Pressure']
        self.combined_results = None
        self.cyclone_metrics = {}
        self.best_params = {}
        
    def create_smart_features(self, df):
        print("Creating smart physics-informed features...")
        
        df = df.dropna(subset=['LAT', 'LON', 'WindSpeed Knots', 'Press mb', 'Name', 'Time'])
        df = df.sort_values(['Name', 'Time']).reset_index(drop=True)
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Basic features
        basic_features = ['LAT', 'LON', 'WindSpeed Knots', 'Press mb']
        
        # Fitur-fitur penting
        df['pressure_deficit'] = (1013.25 - df['Press mb']).clip(0, 150)
        
        # Movement features
        time_diff = df.groupby('Name')['Time'].diff().dt.total_seconds().fillna(3600) / 3600
        df['movement_lat'] = df.groupby('Name')['LAT'].diff().fillna(0) / time_diff.clip(0.1, 24)
        df['movement_lon'] = df.groupby('Name')['LON'].diff().fillna(0) / time_diff.clip(0.1, 24)
        df['movement_speed'] = np.sqrt(df['movement_lat']**2 + df['movement_lon']**2).clip(0, 50)
        
        # Historical trends
        df['wind_trend_3'] = df.groupby('Name')['WindSpeed Knots'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['pressure_trend_3'] = df.groupby('Name')['Press mb'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Cyclone intensity features
        df['intensity_index'] = df['WindSpeed Knots'] / (df['pressure_deficit'] + 1)
        
        # Geographical features
        df['distance_from_equator'] = np.abs(df['LAT'])
        
        # Time features
        df['hour_sin'] = np.sin(2 * np.pi * df['Time'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Time'].dt.hour / 24)
        
        # Semua fitur
        self.all_feature_columns = basic_features + [
            'pressure_deficit', 'movement_lat', 'movement_lon', 'movement_speed',
            'wind_trend_3', 'pressure_trend_3', 'intensity_index', 
            'distance_from_equator', 'hour_sin', 'hour_cos'
        ]
        
        # Filter existing columns
        self.all_feature_columns = [col for col in self.all_feature_columns if col in df.columns]
        self.feature_columns = self.all_feature_columns.copy()
        
        print(f"Smart features created: {len(self.all_feature_columns)}")
        
        # Create targets
        df['next_LAT'] = df.groupby('Name')['LAT'].shift(-1)
        df['next_LON'] = df.groupby('Name')['LON'].shift(-1)
        df['next_WindSpeed'] = df.groupby('Name')['WindSpeed Knots'].shift(-1)
        df['next_Pressure'] = df.groupby('Name')['Press mb'].shift(-1)
        
        # Clean data
        df = df.dropna(subset=self.target_columns + self.all_feature_columns)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"Final dataset: {df.shape}")
        
        return df
    
    def smart_feature_selection(self, X_train, y_train):
        print("Performing smart feature selection...")
        
        temp_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        temp_model.fit(X_train, y_train[:, 0])
        
        importances = temp_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.all_feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        n_selected = min(10, len(self.all_feature_columns))
        selected_features = feature_importance_df.head(n_selected)['feature'].tolist()
        
        print(f"Selected {len(selected_features)} most important features:")
        for feat in selected_features:
            print(f"  - {feat}")
        
        selected_indices = [i for i, col in enumerate(self.all_feature_columns) if col in selected_features]
        
        return selected_features, selected_indices
    
    def train_with_cross_validation(self, df):
        print("Training with robust cross-validation...")
        
        df_processed = self.create_smart_features(df)
        train_data = df_processed[df_processed['status'] == 'train']
        
        if len(train_data) == 0:
            raise ValueError("No training data available!")
        
        X_train = train_data[self.all_feature_columns].values
        y_train = train_data[self.target_columns].values
        
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        y_train_scaled = self.target_scaler.fit_transform(y_train)
        
        self.selected_feature_columns, selected_indices = self.smart_feature_selection(X_train_scaled, y_train_scaled)
        X_train_selected = X_train_scaled[:, selected_indices]
        self.feature_columns = self.selected_feature_columns
        
        print(f"Final training data shape: {X_train_selected.shape}")
        
        for i, target_name in enumerate(self.target_columns):
            print(f"\nTraining model for {target_name}...")
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=5,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(xgb_model, X_train_selected, y_train_scaled[:, i], 
                                      cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
            
            print(f"  CV RMSE: {-cv_scores.mean():.4f} (+/- {-cv_scores.std() * 2:.4f})")
            
            xgb_model.fit(X_train_selected, y_train_scaled[:, i])
            self.models[target_name] = xgb_model
            
            train_pred = xgb_model.predict(X_train_selected)
            train_rmse = np.sqrt(mean_squared_error(y_train_scaled[:, i], train_pred))
            print(f"  Train RMSE: {train_rmse:.4f}")
        
        print("All models trained with robust regularization!")
        
        self.evaluate_and_combine_results(df_processed)
    
    def evaluate_and_combine_results(self, df_processed):
        print("\n=== EVALUATING AND COMBINING RESULTS ===")
        
        all_results = []
        
        train_data = df_processed[df_processed['status'] == 'train']
        if len(train_data) > 0:
            train_results = self._process_data_split(train_data, 'train')
            all_results.append(train_results)
        
        test_data = df_processed[df_processed['status'] == 'test']
        if len(test_data) > 0:
            test_results = self._process_data_split(test_data, 'test')
            all_results.append(test_results)
        
        predict_data = df_processed[df_processed['status'] == 'predict']
        if len(predict_data) > 0:
            predict_results = self._process_data_split(predict_data, 'predict')
            all_results.append(predict_results)
        
        if all_results:
            self.combined_results = pd.concat(all_results, ignore_index=True)
            
            status_order = {'train': 0, 'test': 1, 'predict': 2}
            self.combined_results['status_order'] = self.combined_results['status'].map(status_order)
            self.combined_results = self.combined_results.sort_values(['Name', 'status_order']).drop('status_order', axis=1)
            
            print(f"Combined results shape: {self.combined_results.shape}")
            
            self.calculate_cyclone_metrics()
            self.print_performance_summary()
    
    def _process_data_split(self, data, status):
        X = data[self.all_feature_columns].values
        X_scaled = self.feature_scaler.transform(X)
        
        selected_indices = [i for i, col in enumerate(self.all_feature_columns) 
                          if col in self.selected_feature_columns]
        X_processed = X_scaled[:, selected_indices]
        
        predictions_scaled = np.column_stack([
            self.models[target].predict(X_processed) 
            for target in self.target_columns
        ])
        
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        
        results = data.copy()
        results['predicted_LAT'] = predictions[:, 0]
        results['predicted_LON'] = predictions[:, 1]
        results['predicted_WindSpeed'] = predictions[:, 2]
        results['predicted_Pressure'] = predictions[:, 3]
        
        if all(col in results.columns for col in self.target_columns):
            results['lat_error'] = np.abs(results['next_LAT'] - results['predicted_LAT'])
            results['lon_error'] = np.abs(results['next_LON'] - results['predicted_LON'])
            results['wind_error'] = np.abs(results['next_WindSpeed'] - results['predicted_WindSpeed'])
            results['pressure_error'] = np.abs(results['next_Pressure'] - results['predicted_Pressure'])
            
            results['lat_squared_error'] = (results['next_LAT'] - results['predicted_LAT']) ** 2
            results['lon_squared_error'] = (results['next_LON'] - results['predicted_LON']) ** 2
            results['wind_squared_error'] = (results['next_WindSpeed'] - results['predicted_WindSpeed']) ** 2
            results['pressure_squared_error'] = (results['next_Pressure'] - results['predicted_Pressure']) ** 2
            
            results['overall_squared_error'] = (
                results['lat_squared_error'] + results['lon_squared_error'] + 
                results['wind_squared_error'] + results['pressure_squared_error']
            ) / 4
            
            results['overall_absolute_error'] = (
                results['lat_error'] + results['lon_error'] + 
                results['wind_error'] + results['pressure_error']
            ) / 4
        
        return results
    
    def calculate_cyclone_metrics(self):
        print("\n=== CALCULATING CYCLONE METRICS ===")
        
        self.cyclone_metrics = {}
        unique_cyclones = self.combined_results['Name'].unique()
        
        for cyclone in unique_cyclones:
            cyclone_data = self.combined_results[self.combined_results['Name'] == cyclone]
            cyclone_metrics = {}
            
            for status in ['train', 'test', 'predict']:
                status_data = cyclone_data[cyclone_data['status'] == status]
                if len(status_data) > 0:
                    if status in ['train', 'test'] and all(col in status_data.columns for col in self.target_columns):
                        y_true = status_data[self.target_columns].values
                        y_pred = status_data[['predicted_LAT', 'predicted_LON', 'predicted_WindSpeed', 'predicted_Pressure']].values
                        
                        mae = mean_absolute_error(y_true, y_pred)
                        mse = mean_squared_error(y_true, y_pred)
                        rmse = np.sqrt(mse)
                    else:
                        if 'overall_absolute_error' in status_data.columns:
                            mae = status_data['overall_absolute_error'].mean()
                            mse = status_data['overall_squared_error'].mean()
                            rmse = np.sqrt(mse)
                        else:
                            mae = mse = rmse = 0
                    
                    cyclone_metrics[status] = {
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': rmse,
                        'Records': len(status_data)
                    }
            
            self.cyclone_metrics[cyclone] = cyclone_metrics
        
        print(f"✓ Metrics calculated for {len(unique_cyclones)} cyclones")
    
    def print_performance_summary(self):
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY - TEST ERROR FOCUS")
        print("="*60)
        
        test_data = self.combined_results[self.combined_results['status'] == 'test']
        train_data = self.combined_results[self.combined_results['status'] == 'train']
        
        if len(test_data) > 0:
            print(f"\n📊 TEST SET PERFORMANCE:")
            if 'overall_absolute_error' in test_data.columns:
                print(f"   Overall MAE:  {test_data['overall_absolute_error'].mean():.4f}")
                print(f"   Overall RMSE: {np.sqrt(test_data['overall_squared_error'].mean()):.4f}")
            
            if all(col in test_data.columns for col in ['lat_error', 'lon_error', 'wind_error', 'pressure_error']):
                print(f"   Latitude Error:  {test_data['lat_error'].mean():.4f}")
                print(f"   Longitude Error: {test_data['lon_error'].mean():.4f}")
                print(f"   Wind Speed Error: {test_data['wind_error'].mean():.4f}")
                print(f"   Pressure Error:  {test_data['pressure_error'].mean():.4f}")
        
        if len(train_data) > 0 and len(test_data) > 0:
            if 'overall_absolute_error' in train_data.columns and 'overall_absolute_error' in test_data.columns:
                train_error = train_data['overall_absolute_error'].mean()
                test_error = test_data['overall_absolute_error'].mean()
                gap = test_error - train_error
                
                print(f"\n🔍 OVERFITTING ANALYSIS:")
                print(f"   Train Error: {train_error:.4f}")
                print(f"   Test Error:  {test_error:.4f}")
                print(f"   Generalization Gap: {gap:.4f}")
                
                if gap < 0.02:
                    print("   ✅ Excellent generalization!")
                elif gap < 0.05:
                    print("   👍 Good generalization")
                else:
                    print("   ⚠️  Consider more regularization")
        
        print(f"\n🌀 TEST PERFORMANCE BY CYCLONE:")
        print("-" * 50)
        
        test_performance = []
        for cyclone, metrics_dict in self.cyclone_metrics.items():
            if 'test' in metrics_dict:
                test_metrics = metrics_dict['test']
                test_performance.append((cyclone, test_metrics['MAE']))
                print(f"  {cyclone:15}: MAE = {test_metrics['MAE']:.4f}")
        
        if test_performance:
            best_cyclone = min(test_performance, key=lambda x: x[1])
            worst_cyclone = max(test_performance, key=lambda x: x[1])
            avg_test_mae = np.mean([x[1] for x in test_performance])
            
            print(f"\n  🏆 Best:  {best_cyclone[0]} (MAE: {best_cyclone[1]:.4f})")
            print(f"  📉 Worst: {worst_cyclone[0]} (MAE: {worst_cyclone[1]:.4f})")
            print(f"  📊 Average Test MAE: {avg_test_mae:.4f}")
    
    def predict(self, df):
        if not self.models:
            raise ValueError("Models not trained!")
        
        print("Making predictions with robust model...")
        
        df_processed = self.create_smart_features(df)
        self.evaluate_and_combine_results(df_processed)
        
        return self.combined_results
    
    def save_results(self, filename='cyclone_predictions_robust.csv'):
        if self.combined_results is not None:
            base_columns = ['Name', 'Time', 'status', 'LAT', 'LON', 'WindSpeed Knots', 'Press mb']
            prediction_columns = ['predicted_LAT', 'predicted_LON', 'predicted_WindSpeed', 'predicted_Pressure']
            error_columns = ['lat_error', 'lon_error', 'wind_error', 'pressure_error']
            
            all_columns = base_columns + prediction_columns + error_columns
            
            existing_columns = [col for col in all_columns if col in self.combined_results.columns]
            
            self.combined_results[existing_columns].to_csv(filename, index=False)
            print(f"✓ Results saved to '{filename}'")
            print(f"  Total records: {len(self.combined_results)}")
            print(f"  Cyclones: {self.combined_results['Name'].nunique()}")
            
            self.save_performance_summary()
        else:
            print("No results to save!")
    
    def save_performance_summary(self):
        if self.cyclone_metrics:
            summary_data = []
            
            for cyclone, metrics_dict in self.cyclone_metrics.items():
                row = {'Cyclone_Name': cyclone}
                
                for status in ['train', 'test', 'predict']:
                    if status in metrics_dict:
                        for metric in ['MAE', 'MSE', 'RMSE']:
                            row[f'{status}_{metric}'] = metrics_dict[status][metric]
                        row[f'{status}_Records'] = metrics_dict[status]['Records']
                
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv('cyclone_performance_summary.csv', index=False)
            print(f"✓ Performance summary saved to 'cyclone_performance_summary.csv'")

# Fungsi untuk plotting
def plot_lat_lon(cyclone_data, cyclone_name):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Cyclone Trajectory Analysis: {cyclone_name} (Predict)', fontsize=16, y=0.98)
    
    time_points = range(len(cyclone_data))
    
    # Plot 1: Trajectory Comparison
    ax1 = fig.add_subplot(221)
    ax1.plot(cyclone_data['LON'], cyclone_data['LAT'], 'bo-', linewidth=2, markersize=4, label='Actual')
    ax1.plot(cyclone_data['predicted_LON'], cyclone_data['predicted_LAT'], 'ro--', linewidth=2, markersize=4, label='Predicted')
    ax1.plot(cyclone_data['LON'].iloc[0], cyclone_data['LAT'].iloc[0], 'g^', markersize=10, label='Start')
    ax1.plot(cyclone_data['LON'].iloc[-1], cyclone_data['LAT'].iloc[-1], 'rv', markersize=10, label='End')
    ax1.set_xlabel('Longitude', fontsize=9)
    ax1.set_ylabel('Latitude', fontsize=9)
    ax1.set_title('Trajectory Comparison', fontsize=10, pad=10)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Plot 2: Latitude Comparison Over Time
    ax2 = fig.add_subplot(222)
    ax2.plot(time_points, cyclone_data['next_LAT'], 'b-', linewidth=2, label='Actual LAT')
    ax2.plot(time_points, cyclone_data['predicted_LAT'], 'r--', linewidth=2, label='Predicted LAT')
    ax2.set_xlabel('Time Step', fontsize=9)
    ax2.set_ylabel('Latitude', fontsize=9)
    ax2.set_title('Latitude Prediction Over Time', fontsize=10, pad=10)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Plot 3: Longitude Comparison Over Time
    ax3 = fig.add_subplot(223)
    ax3.plot(time_points, cyclone_data['next_LON'], 'g-', linewidth=2, label='Actual LON')
    ax3.plot(time_points, cyclone_data['predicted_LON'], 'orange', linewidth=2, label='Predicted LON')
    ax3.set_xlabel('Time Step', fontsize=9)
    ax3.set_ylabel('Longitude', fontsize=9)
    ax3.set_title('Longitude Prediction Over Time', fontsize=10, pad=10)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Plot 4: Error Analysis
    ax4 = fig.add_subplot(224)
    ax4.plot(time_points, cyclone_data['lat_error'], 'red', linewidth=2, label='Latitude Error')
    ax4.plot(time_points, cyclone_data['lon_error'], 'blue', linewidth=2, label='Longitude Error')
    ax4.set_xlabel('Time Step', fontsize=9)
    ax4.set_ylabel('Absolute Error', fontsize=9)
    ax4.set_title('Prediction Errors Over Time', fontsize=10, pad=10)
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    return fig

def plot_intensity(cyclone_data, cyclone_name):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Cyclone Intensity Analysis: {cyclone_name} (Predict)', fontsize=16, y=0.98)
    
    time_points = range(len(cyclone_data))
    
    # Plot 1: Wind Speed Comparison
    ax1 = fig.add_subplot(221)
    ax1.plot(time_points, cyclone_data['next_WindSpeed'], 'b-', linewidth=2, label='Actual Wind Speed')
    ax1.plot(time_points, cyclone_data['predicted_WindSpeed'], 'r--', linewidth=2, label='Predicted Wind Speed')
    ax1.set_xlabel('Time Step', fontsize=9)
    ax1.set_ylabel('Wind Speed (knots)', fontsize=9)
    ax1.set_title('Wind Speed Prediction', fontsize=10, pad=10)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Plot 2: Pressure Comparison
    ax2 = fig.add_subplot(222)
    ax2.plot(time_points, cyclone_data['next_Pressure'], 'g-', linewidth=2, label='Actual Pressure')
    ax2.plot(time_points, cyclone_data['predicted_Pressure'], 'orange', linewidth=2, label='Predicted Pressure')
    ax2.set_xlabel('Time Step', fontsize=9)
    ax2.set_ylabel('Pressure (mb)', fontsize=9)
    ax2.set_title('Pressure Prediction', fontsize=10, pad=10)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Plot 3: Wind-Pressure Relationship
    ax3 = fig.add_subplot(223)
    ax3.scatter(cyclone_data['next_WindSpeed'], cyclone_data['next_Pressure'], 
               alpha=0.7, label='Actual', s=50, color='blue')
    ax3.scatter(cyclone_data['predicted_WindSpeed'], cyclone_data['predicted_Pressure'], 
               alpha=0.7, label='Predicted', s=50, color='red')
    
    for i in range(len(cyclone_data)):
        ax3.plot([cyclone_data['next_WindSpeed'].iloc[i], cyclone_data['predicted_WindSpeed'].iloc[i]],
                [cyclone_data['next_Pressure'].iloc[i], cyclone_data['predicted_Pressure'].iloc[i]],
                'gray', alpha=0.3, linewidth=0.5)
    
    ax3.set_xlabel('Wind Speed (knots)', fontsize=9)
    ax3.set_ylabel('Pressure (mb)', fontsize=9)
    ax3.set_title('Wind-Pressure Relationship', fontsize=10, pad=10)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Plot 4: Intensity Errors Over Time
    ax4 = fig.add_subplot(224)
    ax4.plot(time_points, cyclone_data['wind_error'], 'red', linewidth=2, label='Wind Speed Error')
    ax4.plot(time_points, cyclone_data['pressure_error'], 'blue', linewidth=2, label='Pressure Error')
    ax4.set_xlabel('Time Step', fontsize=9)
    ax4.set_ylabel('Absolute Error', fontsize=9)
    ax4.set_title('Intensity Prediction Errors', fontsize=10, pad=10)
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    return fig

def plot_all(cyclone_data, cyclone_name):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Complete Cyclone Analysis: {cyclone_name} (Predict)', fontsize=18, y=0.98)
    
    time_points = range(len(cyclone_data))
    
    # LAT-LON Plots
    ax1 = fig.add_subplot(241)
    ax1.plot(cyclone_data['LON'], cyclone_data['LAT'], 'bo-', linewidth=2, markersize=4, label='Actual')
    ax1.plot(cyclone_data['predicted_LON'], cyclone_data['predicted_LAT'], 'ro--', linewidth=2, markersize=4, label='Predicted')
    ax1.plot(cyclone_data['LON'].iloc[0], cyclone_data['LAT'].iloc[0], 'g^', markersize=10, label='Start')
    ax1.plot(cyclone_data['LON'].iloc[-1], cyclone_data['LAT'].iloc[-1], 'rv', markersize=10, label='End')
    ax1.set_xlabel('Longitude', fontsize=8)
    ax1.set_ylabel('Latitude', fontsize=8)
    ax1.set_title('Trajectory Comparison', fontsize=9, pad=10)
    ax1.legend(loc='best', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
    
    ax2 = fig.add_subplot(242)
    ax2.plot(time_points, cyclone_data['next_LAT'], 'b-', linewidth=2, label='Actual LAT')
    ax2.plot(time_points, cyclone_data['predicted_LAT'], 'r--', linewidth=2, label='Predicted LAT')
    ax2.set_xlabel('Time Step', fontsize=8)
    ax2.set_ylabel('Latitude', fontsize=8)
    ax2.set_title('Latitude Prediction', fontsize=9, pad=10)
    ax2.legend(loc='best', fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
    
    ax3 = fig.add_subplot(243)
    ax3.plot(time_points, cyclone_data['next_LON'], 'g-', linewidth=2, label='Actual LON')
    ax3.plot(time_points, cyclone_data['predicted_LON'], 'orange', linewidth=2, label='Predicted LON')
    ax3.set_xlabel('Time Step', fontsize=8)
    ax3.set_ylabel('Longitude', fontsize=8)
    ax3.set_title('Longitude Prediction', fontsize=9, pad=10)
    ax3.legend(loc='best', fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top')
    
    ax4 = fig.add_subplot(244)
    ax4.plot(time_points, cyclone_data['lat_error'], 'red', linewidth=2, label='Latitude Error')
    ax4.plot(time_points, cyclone_data['lon_error'], 'blue', linewidth=2, label='Longitude Error')
    ax4.set_xlabel('Time Step', fontsize=8)
    ax4.set_ylabel('Absolute Error', fontsize=8)
    ax4.set_title('Position Errors', fontsize=9, pad=10)
    ax4.legend(loc='best', fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Intensity Plots
    ax5 = fig.add_subplot(245)
    ax5.plot(time_points, cyclone_data['next_WindSpeed'], 'b-', linewidth=2, label='Actual Wind Speed')
    ax5.plot(time_points, cyclone_data['predicted_WindSpeed'], 'r--', linewidth=2, label='Predicted Wind Speed')
    ax5.set_xlabel('Time Step', fontsize=8)
    ax5.set_ylabel('Wind Speed (knots)', fontsize=8)
    ax5.set_title('Wind Speed Prediction', fontsize=9, pad=10)
    ax5.legend(loc='best', fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.text(0.02, 0.98, '(e)', transform=ax5.transAxes, fontsize=12, fontweight='bold', va='top')
    
    ax6 = fig.add_subplot(246)
    ax6.plot(time_points, cyclone_data['next_Pressure'], 'g-', linewidth=2, label='Actual Pressure')
    ax6.plot(time_points, cyclone_data['predicted_Pressure'], 'orange', linewidth=2, label='Predicted Pressure')
    ax6.set_xlabel('Time Step', fontsize=8)
    ax6.set_ylabel('Pressure (mb)', fontsize=8)
    ax6.set_title('Pressure Prediction', fontsize=9, pad=10)
    ax6.legend(loc='best', fontsize=7)
    ax6.grid(True, alpha=0.3)
    ax6.text(0.02, 0.98, '(f)', transform=ax6.transAxes, fontsize=12, fontweight='bold', va='top')
    
    ax7 = fig.add_subplot(247)
    ax7.scatter(cyclone_data['next_WindSpeed'], cyclone_data['next_Pressure'], 
               alpha=0.7, label='Actual', s=50, color='blue')
    ax7.scatter(cyclone_data['predicted_WindSpeed'], cyclone_data['predicted_Pressure'], 
               alpha=0.7, label='Predicted', s=50, color='red')
    
    for i in range(len(cyclone_data)):
        ax7.plot([cyclone_data['next_WindSpeed'].iloc[i], cyclone_data['predicted_WindSpeed'].iloc[i]],
                [cyclone_data['next_Pressure'].iloc[i], cyclone_data['predicted_Pressure'].iloc[i]],
                'gray', alpha=0.3, linewidth=0.5)
    
    ax7.set_xlabel('Wind Speed (knots)', fontsize=8)
    ax7.set_ylabel('Pressure (mb)', fontsize=8)
    ax7.set_title('Wind-Pressure Relationship', fontsize=9, pad=10)
    ax7.legend(loc='best', fontsize=7)
    ax7.grid(True, alpha=0.3)
    ax7.text(0.02, 0.98, '(g)', transform=ax7.transAxes, fontsize=12, fontweight='bold', va='top')
    
    ax8 = fig.add_subplot(248)
    ax8.plot(time_points, cyclone_data['wind_error'], 'red', linewidth=2, label='Wind Speed Error')
    ax8.plot(time_points, cyclone_data['pressure_error'], 'blue', linewidth=2, label='Pressure Error')
    ax8.set_xlabel('Time Step', fontsize=8)
    ax8.set_ylabel('Absolute Error', fontsize=8)
    ax8.set_title('Intensity Errors', fontsize=9, pad=10)
    ax8.legend(loc='best', fontsize=7)
    ax8.grid(True, alpha=0.3)
    ax8.text(0.02, 0.98, '(h)', transform=ax8.transAxes, fontsize=12, fontweight='bold', va='top')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    
    return fig

# Fungsi untuk menampilkan metrik
def display_metrics(model):
    if model.combined_results is None:
        st.warning("No results available! Please run predictions first.")
        return
    
    test_data = model.combined_results[model.combined_results['status'] == 'test']
    train_data = model.combined_results[model.combined_results['status'] == 'train']
    predict_data = model.combined_results[model.combined_results['status'] == 'predict']
    
    # Test set performance
    if len(test_data) > 0:
        st.subheader("📊 TEST SET PERFORMANCE")
        
        if 'overall_absolute_error' in test_data.columns:
            col1, col2 = st.columns(2)
            col1.metric("Overall MAE", f"{test_data['overall_absolute_error'].mean():.4f}")
            col2.metric("Overall RMSE", f"{np.sqrt(test_data['overall_squared_error'].mean()):.4f}")
        
        if all(col in test_data.columns for col in ['lat_error', 'lon_error', 'wind_error', 'pressure_error']):
            st.write("**Error Breakdown:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Latitude Error", f"{test_data['lat_error'].mean():.4f}")
            col2.metric("Longitude Error", f"{test_data['lon_error'].mean():.4f}")
            col3.metric("Wind Speed Error", f"{test_data['wind_error'].mean():.4f}")
            col4.metric("Pressure Error", f"{test_data['pressure_error'].mean():.4f}")
    
    # Predict data summary
    if len(predict_data) > 0:
        st.subheader("🔮 PREDICT DATA SUMMARY")
        col1, col2 = st.columns(2)
        col1.metric("Total predict records", len(predict_data))
        col2.metric("Cyclones with predict data", predict_data['Name'].nunique())
        
        st.write("**Cyclones with predict data:**")
        for cyclone in sorted(predict_data['Name'].unique()):
            count = len(predict_data[predict_data['Name'] == cyclone])
            st.write(f"- {cyclone}: {count} records")
    
    # Cyclone performance
    if hasattr(model, 'cyclone_metrics'):
        st.subheader("🌀 CYCLONE PERFORMANCE")
        
        cyclone_data = []
        for cyclone, metrics in model.cyclone_metrics.items():
            if 'test' in metrics:
                cyclone_data.append({
                    'Cyclone': cyclone,
                    'MAE': f"{metrics['test']['MAE']:.4f}",
                    'RMSE': f"{metrics['test']['RMSE']:.4f}"
                })
        
        if cyclone_data:
            df_cyclone = pd.DataFrame(cyclone_data)
            st.dataframe(df_cyclone)

# Fungsi utama Streamlit
def main():
    st.set_page_config(
        page_title="Cyclone Prediction System",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌪️ CYCLONE PREDICTION SYSTEM")
    st.markdown("Advanced Machine Learning for Cyclone Trajectory Prediction")
    
    # Sidebar untuk kontrol
    st.sidebar.title("Control Panel")
    
    # Inisialisasi session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'selected_cyclone' not in st.session_state:
        st.session_state.selected_cyclone = None
    
    # Load data
    st.sidebar.subheader("1. Load Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            
            # Update data info
            cyclones = st.session_state.data['Name'].nunique() if 'Name' in st.session_state.data.columns else 0
            status_dist = st.session_state.data['status'].value_counts().to_dict() if 'status' in st.session_state.data.columns else {}
            
            st.sidebar.success(f"✓ Data loaded: {len(st.session_state.data)} rows")
            st.sidebar.write(f"Cyclones: {cyclones}")
            st.sidebar.write(f"Status: {status_dist}")
            
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")
    
    # Train model
    st.sidebar.subheader("2. Train Model")
    train_button = st.sidebar.button("🚀 Train Model", disabled=st.session_state.data is None)
    
    if train_button:
        with st.spinner("Training model..."):
            try:
                st.session_state.model = FixedBoostedCyclonePredictor()
                st.session_state.model.train_with_cross_validation(st.session_state.data)
                st.sidebar.success("Model trained successfully!")
            except Exception as e:
                st.sidebar.error(f"Error during training: {e}")
    
    # Make predictions
    st.sidebar.subheader("3. Make Predictions")
    predict_button = st.sidebar.button("🔮 Predict", disabled=st.session_state.model is None)
    
    if predict_button:
        with st.spinner("Making predictions..."):
            try:
                st.session_state.results = st.session_state.model.predict(st.session_state.data)
                st.sidebar.success("Predictions completed!")
            except Exception as e:
                st.sidebar.error(f"Error during prediction: {e}")
    
    # Select cyclone
    st.sidebar.subheader("4. Select Cyclone")
    
    if st.session_state.results is not None:
        predict_data = st.session_state.results[st.session_state.results['status'] == 'predict']
        if len(predict_data) > 0:
            cyclones = sorted(predict_data['Name'].unique())
            selected_cyclone = st.sidebar.selectbox("Select a cyclone", cyclones)
            
            if selected_cyclone != st.session_state.selected_cyclone:
                st.session_state.selected_cyclone = selected_cyclone
            
            # Get data for selected cyclone
            cyclone_data = predict_data[predict_data['Name'] == selected_cyclone]
            
            if len(cyclone_data) > 0:
                st.sidebar.write(f"✓ Selected: {selected_cyclone}")
                st.sidebar.write(f"Predict records: {len(cyclone_data)}")
        else:
            st.sidebar.warning("No predict data available")
    
    # Generate plots
    st.sidebar.subheader("5. Generate Plots")
    
    plot_type = st.sidebar.radio(
        "Select plot type",
        ["None", "LAT-LON", "Intensity", "All Plots"]
    )
    
    if plot_type != "None" and st.session_state.selected_cyclone is not None:
        predict_data = st.session_state.results[st.session_state.results['status'] == 'predict']
        cyclone_data = predict_data[predict_data['Name'] == st.session_state.selected_cyclone]
        
        if len(cyclone_data) > 0:
            if plot_type == "LAT-LON":
                fig = plot_lat_lon(cyclone_data, st.session_state.selected_cyclone)
                st.pyplot(fig)
            elif plot_type == "Intensity":
                fig = plot_intensity(cyclone_data, st.session_state.selected_cyclone)
                st.pyplot(fig)
            elif plot_type == "All Plots":
                fig = plot_all(cyclone_data, st.session_state.selected_cyclone)
                st.pyplot(fig)
        else:
            st.warning(f"No predict data available for {st.session_state.selected_cyclone}")
    elif plot_type != "None":
        st.warning("Please select a cyclone first!")
    
    # Save results
    st.sidebar.subheader("6. Save Results")
    save_button = st.sidebar.button("💾 Save Results", disabled=st.session_state.results is None)
    
    if save_button:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cyclone_predictions_{timestamp}.csv"
            
            st.session_state.model.combined_results = st.session_state.results
            st.session_state.model.save_results(filename)
            
            # Provide download link
            with open(filename, 'rb') as f:
                bytes_data = f.read()
                b64 = base64.b64encode(bytes_data).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
                st.sidebar.markdown(href, unsafe_allow_html=True)
            
            st.sidebar.success("Results saved successfully!")
        except Exception as e:
            st.sidebar.error(f"Error saving results: {e}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Console", "Results", "Data Preview"])
    
    with tab1:
        st.subheader("Console Output")
        # Placeholder for console output
        console_placeholder = st.empty()
        
        if st.session_state.model is not None:
            with console_placeholder.container():
                st.text("Model training and prediction logs will appear here...")
    
    with tab2:
        st.subheader("Performance Metrics")
        if st.session_state.model is not None:
            display_metrics(st.session_state.model)
        else:
            st.info("Train model and make predictions to see metrics here")
    
    with tab3:
        st.subheader("Data Preview")
        if st.session_state.results is not None:
            st.write("Preview of Results (first 20 rows):")
            preview_columns = ['Name', 'status', 'LAT', 'LON', 'predicted_LAT', 'predicted_LON']
            existing_columns = [col for col in preview_columns if col in st.session_state.results.columns]
            
            if existing_columns:
                preview_df = st.session_state.results[existing_columns].head(20)
                st.dataframe(preview_df)
            else:
                st.write("No preview available")
        else:
            st.info("Make predictions to see data preview here")

if __name__ == "__main__":
    main()