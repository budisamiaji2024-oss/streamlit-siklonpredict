import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading
import sys
import warnings
warnings.filterwarnings('ignore')

# Import your existing model class
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import SelectFromModel

# Your existing model class (copied here for completeness)
class FixedBoostedCyclonePredictor:
    def __init__(self):
        self.models = {}
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.feature_selector = None
        self.feature_columns = []
        self.all_feature_columns = []  # Menyimpan semua fitur asli
        self.selected_feature_columns = []  # Menyimpan fitur yang terpilih
        self.target_columns = ['next_LAT', 'next_LON', 'next_WindSpeed', 'next_Pressure']
        self.combined_results = None
        self.cyclone_metrics = {}
        self.best_params = {}
        
    def create_smart_features(self, df):
        """Membuat features yang lebih selektif dan manageable"""
        print("Creating smart physics-informed features...")
        
        df = df.dropna(subset=['LAT', 'LON', 'WindSpeed Knots', 'Press mb', 'Name', 'Time'])
        df = df.sort_values(['Name', 'Time']).reset_index(drop=True)
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Basic features - tetap pertahankan
        basic_features = ['LAT', 'LON', 'WindSpeed Knots', 'Press mb']
        
        # **HANYA FITUR YANG PALING PENTING**
        # 1. Pressure features
        df['pressure_deficit'] = (1013.25 - df['Press mb']).clip(0, 150)
        
        # 2. Movement features (sangat penting untuk trajectory)
        time_diff = df.groupby('Name')['Time'].diff().dt.total_seconds().fillna(3600) / 3600
        df['movement_lat'] = df.groupby('Name')['LAT'].diff().fillna(0) / time_diff.clip(0.1, 24)
        df['movement_lon'] = df.groupby('Name')['LON'].diff().fillna(0) / time_diff.clip(0.1, 24)
        df['movement_speed'] = np.sqrt(df['movement_lat']**2 + df['movement_lon']**2).clip(0, 50)
        
        # 3. Historical trends (simplified)
        df['wind_trend_3'] = df.groupby('Name')['WindSpeed Knots'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['pressure_trend_3'] = df.groupby('Name')['Press mb'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # 4. Cyclone intensity features
        df['intensity_index'] = df['WindSpeed Knots'] / (df['pressure_deficit'] + 1)
        
        # 5. Geographical features
        df['distance_from_equator'] = np.abs(df['LAT'])
        
        # 6. Time features (simplified)
        df['hour_sin'] = np.sin(2 * np.pi * df['Time'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Time'].dt.hour / 24)
        
        # **HANYA 15 FITUR UTAMA**
        self.all_feature_columns = basic_features + [
            'pressure_deficit', 'movement_lat', 'movement_lon', 'movement_speed',
            'wind_trend_3', 'pressure_trend_3', 'intensity_index', 
            'distance_from_equator', 'hour_sin', 'hour_cos'
        ]
        
        # Filter existing columns
        self.all_feature_columns = [col for col in self.all_feature_columns if col in df.columns]
        self.feature_columns = self.all_feature_columns.copy()  # Initialize dengan semua fitur
        
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
        """Feature selection yang lebih sederhana dan robust"""
        print("Performing smart feature selection...")
        
        # Gunakan feature importance dari RandomForest untuk seleksi
        temp_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit pada target pertama sebagai proxy
        temp_model.fit(X_train, y_train[:, 0])
        
        # Dapatkan feature importance
        importances = temp_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.all_feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Pilih top features (8-12 features untuk menghindari overfitting)
        n_selected = min(10, len(self.all_feature_columns))
        selected_features = feature_importance_df.head(n_selected)['feature'].tolist()
        
        print(f"Selected {len(selected_features)} most important features:")
        for feat in selected_features:
            print(f"  - {feat}")
        
        # Get indices of selected features
        selected_indices = [i for i, col in enumerate(self.all_feature_columns) if col in selected_features]
        
        return selected_features, selected_indices
    
    def train_with_cross_validation(self, df):
        """Training dengan cross-validation untuk mengurangi overfitting"""
        print("Training with robust cross-validation...")
        
        # Prepare data
        df_processed = self.create_smart_features(df)
        train_data = df_processed[df_processed['status'] == 'train']
        
        if len(train_data) == 0:
            raise ValueError("No training data available!")
        
        X_train = train_data[self.all_feature_columns].values
        y_train = train_data[self.target_columns].values
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        y_train_scaled = self.target_scaler.fit_transform(y_train)
        
        # Smart feature selection
        self.selected_feature_columns, selected_indices = self.smart_feature_selection(X_train_scaled, y_train_scaled)
        X_train_selected = X_train_scaled[:, selected_indices]
        self.feature_columns = self.selected_feature_columns  # Update feature columns
        
        print(f"Final training data shape: {X_train_selected.shape}")
        
        # Train models dengan regularisasi kuat
        for i, target_name in enumerate(self.target_columns):
            print(f"\nTraining model for {target_name}...")
            
            # XGBoost dengan regularisasi kuat
            xgb_model = xgb.XGBRegressor(
                n_estimators=300,  # Reduced untuk hindari overfitting
                learning_rate=0.05,
                max_depth=6,      # Shallower trees
                min_child_weight=5,  # More conservative
                subsample=0.7,    # More randomness
                colsample_bytree=0.7,
                reg_alpha=1.0,    # Strong L1 regularization
                reg_lambda=1.0,   # Strong L2 regularization
                random_state=42,
                n_jobs=-1
            )
            
            # Cross-validation untuk monitoring
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(xgb_model, X_train_selected, y_train_scaled[:, i], 
                                      cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
            
            print(f"  CV RMSE: {-cv_scores.mean():.4f} (+/- {-cv_scores.std() * 2:.4f})")
            
            # Train final model
            xgb_model.fit(X_train_selected, y_train_scaled[:, i])
            self.models[target_name] = xgb_model
            
            # Check for overfitting
            train_pred = xgb_model.predict(X_train_selected)
            train_rmse = np.sqrt(mean_squared_error(y_train_scaled[:, i], train_pred))
            print(f"  Train RMSE: {train_rmse:.4f}")
        
        print("All models trained with robust regularization!")
        
        # Evaluate on training and test data
        self.evaluate_and_combine_results(df_processed)
    
    def evaluate_and_combine_results(self, df_processed):
        """Evaluate models dan gabungkan semua hasil"""
        print("\n=== EVALUATING AND COMBINING RESULTS ===")
        
        all_results = []
        
        # Process train data
        train_data = df_processed[df_processed['status'] == 'train']
        if len(train_data) > 0:
            train_results = self._process_data_split(train_data, 'train')
            all_results.append(train_results)
        
        # Process test data
        test_data = df_processed[df_processed['status'] == 'test']
        if len(test_data) > 0:
            test_results = self._process_data_split(test_data, 'test')
            all_results.append(test_results)
        
        # Process predict data
        predict_data = df_processed[df_processed['status'] == 'predict']
        if len(predict_data) > 0:
            predict_results = self._process_data_split(predict_data, 'predict')
            all_results.append(predict_results)
        
        # Combine all results
        if all_results:
            self.combined_results = pd.concat(all_results, ignore_index=True)
            
            # Sort by Name and then status
            status_order = {'train': 0, 'test': 1, 'predict': 2}
            self.combined_results['status_order'] = self.combined_results['status'].map(status_order)
            self.combined_results = self.combined_results.sort_values(['Name', 'status_order']).drop('status_order', axis=1)
            
            print(f"Combined results shape: {self.combined_results.shape}")
            
            # Calculate metrics
            self.calculate_cyclone_metrics()
            self.print_performance_summary()
    
    def _process_data_split(self, data, status):
        """Process data untuk status tertentu - FIXED VERSION"""
        # Gunakan all_feature_columns untuk scaling, lalu pilih features yang dipilih
        X = data[self.all_feature_columns].values
        
        # Scale menggunakan semua fitur
        X_scaled = self.feature_scaler.transform(X)
        
        # Pilih hanya fitur yang terpilih
        selected_indices = [i for i, col in enumerate(self.all_feature_columns) 
                          if col in self.selected_feature_columns]
        X_processed = X_scaled[:, selected_indices]
        
        # Predictions
        predictions_scaled = np.column_stack([
            self.models[target].predict(X_processed) 
            for target in self.target_columns
        ])
        
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        
        # Create results DataFrame
        results = data.copy()
        results['predicted_LAT'] = predictions[:, 0]
        results['predicted_LON'] = predictions[:, 1]
        results['predicted_WindSpeed'] = predictions[:, 2]
        results['predicted_Pressure'] = predictions[:, 3]
        
        # Calculate errors
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
        """Hitung metrik untuk setiap siklon"""
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
        """Print summary performance dengan fokus pada test error"""
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
        
        # Test performance by cyclone
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
        """Membuat predictions"""
        if not self.models:
            raise ValueError("Models not trained!")
        
        print("Making predictions with robust model...")
        
        # Process the data
        df_processed = self.create_smart_features(df)
        self.evaluate_and_combine_results(df_processed)
        
        return self.combined_results
    
    def save_results(self, filename='cyclone_predictions_robust.csv'):
        """Menyimpan semua hasil"""
        if self.combined_results is not None:
            # Select columns to save
            base_columns = ['Name', 'Time', 'status', 'LAT', 'LON', 'WindSpeed Knots', 'Press mb']
            prediction_columns = ['predicted_LAT', 'predicted_LON', 'predicted_WindSpeed', 'predicted_Pressure']
            error_columns = ['lat_error', 'lon_error', 'wind_error', 'pressure_error']
            
            all_columns = base_columns + prediction_columns + error_columns
            
            # Only include columns that exist
            existing_columns = [col for col in all_columns if col in self.combined_results.columns]
            
            self.combined_results[existing_columns].to_csv(filename, index=False)
            print(f"✓ Results saved to '{filename}'")
            print(f"  Total records: {len(self.combined_results)}")
            print(f"  Cyclones: {self.combined_results['Name'].nunique()}")
            
            # Save performance summary
            self.save_performance_summary()
        else:
            print("No results to save!")
    
    def save_performance_summary(self):
        """Menyimpan summary performa"""
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

# Custom Navigation Toolbar with save button
class CustomNavigationToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, parent, **kwargs):
        super().__init__(canvas, parent, **kwargs)
        
    # Add save button to the toolbar
    def _init_toolbar(self):
        self.basedir = os.path.join(os.path.dirname(__file__), "images")
        
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                # Add a spacer
                self._Spacer()
            else:
                self._Button(
                    text=text,
                    tooltip=tooltip_text,
                    image_file=image_file,
                    callback=callback
                )
        
        # Add custom save button
        self._Button(
            text='Save',
            tooltip='Save the figure',
            image_file='filesave',
            callback=self.save_figure
        )
    
    def save_figure(self, *args):
        """Save the current figure"""
        filetypes = [
            ('PNG files', '*.png'),
            ('PDF files', '*.pdf'),
            ('SVG files', '*.svg'),
            ('All files', '*.*')
        ]
        
        filepath = filedialog.asksaveasfilename(
            title='Save figure',
            filetypes=filetypes,
            defaultextension='.png'
        )
        
        if filepath:
            try:
                self.canvas.figure.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Figure saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save figure: {e}")

# Tkinter GUI Application with Latitude-Longitude and Intensity Plots
class CyclonePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cyclone Prediction System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        self.model = None
        self.data = None
        self.results = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🌪️ CYCLONE PREDICTION SYSTEM", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header_frame, text="Advanced Machine Learning for Cyclone Trajectory Prediction", 
                                 font=('Arial', 10), fg='#ecf0f1', bg='#2c3e50')
        subtitle_label.pack(expand=True)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Controls
        control_frame = tk.LabelFrame(main_frame, text="Control Panel", font=('Arial', 12, 'bold'),
                                    bg='#f0f0f0', fg='#2c3e50', padx=15, pady=15)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Data loading section
        data_frame = tk.Frame(control_frame, bg='#f0f0f0')
        data_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(data_frame, text="1. Load Data", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W)
        
        self.load_btn = tk.Button(data_frame, text="📁 Load CSV File", 
                                 command=self.load_data,
                                 font=('Arial', 9), bg='#3498db', fg='white',
                                 width=20, height=2)
        self.load_btn.pack(pady=5)
        
        self.data_info = tk.Label(data_frame, text="No data loaded", 
                                 font=('Arial', 8), bg='#f0f0f0', fg='#7f8c8d')
        self.data_info.pack(anchor=tk.W)
        
        # Model training section
        train_frame = tk.Frame(control_frame, bg='#f0f0f0')
        train_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(train_frame, text="2. Train Model", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W)
        
        self.train_btn = tk.Button(train_frame, text="🚀 Train Model", 
                                  command=self.train_model,
                                  font=('Arial', 9), bg='#27ae60', fg='white',
                                  width=20, height=2, state=tk.DISABLED)
        self.train_btn.pack(pady=5)
        
        # Prediction section
        predict_frame = tk.Frame(control_frame, bg='#f0f0f0')
        predict_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(predict_frame, text="3. Make Predictions", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W)
        
        self.predict_btn = tk.Button(predict_frame, text="🔮 Predict", 
                                    command=self.make_predictions,
                                    font=('Arial', 9), bg='#e74c3c', fg='white',
                                    width=20, height=2, state=tk.DISABLED)
        self.predict_btn.pack(pady=5)
        
        # Cyclone selection section
        cyclone_frame = tk.Frame(control_frame, bg='#f0f0f0')
        cyclone_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(cyclone_frame, text="4. Select Cyclone", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W)
        
        # Cyclone dropdown
        self.cyclone_var = tk.StringVar()
        self.cyclone_dropdown = ttk.Combobox(cyclone_frame, textvariable=self.cyclone_var, 
                                            state="readonly", width=18)
        self.cyclone_dropdown.pack(pady=5)
        self.cyclone_dropdown.bind("<<ComboboxSelected>>", self.on_cyclone_selected)
        
        self.cyclone_info = tk.Label(cyclone_frame, text="No cyclone selected", 
                                   font=('Arial', 8), bg='#f0f0f0', fg='#7f8c8d')
        self.cyclone_info.pack(anchor=tk.W)
        
        # Visualization section
        viz_frame = tk.Frame(control_frame, bg='#f0f0f0')
        viz_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(viz_frame, text="5. Generate Plots", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W)
        
        # Plot buttons frame
        plot_btn_frame = tk.Frame(viz_frame, bg='#f0f0f0')
        plot_btn_frame.pack(fill=tk.X, pady=5)
        
        self.plot_lat_lon_btn = tk.Button(plot_btn_frame, text="📈 Plot LAT-LON", 
                                         command=self.plot_lat_lon,
                                         font=('Arial', 8), bg='#9b59b6', fg='white',
                                         width=15, height=1, state=tk.DISABLED)
        self.plot_lat_lon_btn.pack(side=tk.LEFT, padx=2)
        
        self.plot_intensity_btn = tk.Button(plot_btn_frame, text="💨 Plot Intensity", 
                                          command=self.plot_intensity,
                                          font=('Arial', 8), bg='#e74c3c', fg='white',
                                          width=15, height=1, state=tk.DISABLED)
        self.plot_intensity_btn.pack(side=tk.LEFT, padx=2)
        
        self.plot_all_btn = tk.Button(plot_btn_frame, text="🎨 All Plots", 
                                     command=self.plot_all,
                                     font=('Arial', 8), bg='#f39c12', fg='white',
                                     width=15, height=1, state=tk.DISABLED)
        self.plot_all_btn.pack(side=tk.LEFT, padx=2)
        
        # Results section
        results_frame = tk.Frame(control_frame, bg='#f0f0f0')
        results_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(results_frame, text="6. Save Results", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W)
        
        self.save_btn = tk.Button(results_frame, text="💾 Save Results", 
                                 command=self.save_results,
                                 font=('Arial', 9), bg='#f39c12', fg='white',
                                 width=20, height=2, state=tk.DISABLED)
        self.save_btn.pack(pady=5)
        
        # Progress section
        progress_frame = tk.Frame(control_frame, bg='#f0f0f0')
        progress_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(progress_frame, text="Progress", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0', fg='#2c3e50').pack(anchor=tk.W)
        
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(progress_frame, text="Ready", 
                                   font=('Arial', 8), bg='#f0f0f0', fg='#7f8c8d')
        self.status_label.pack(anchor=tk.W)
        
        # Right panel - Output
        output_frame = tk.LabelFrame(main_frame, text="Output & Results", 
                                   font=('Arial', 12, 'bold'),
                                   bg='#f0f0f0', fg='#2c3e50', padx=15, pady=15)
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(output_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Console tab
        console_frame = tk.Frame(self.notebook, bg='#2c3e50')
        self.console = scrolledtext.ScrolledText(console_frame, 
                                               bg='#2c3e50', fg='#ecf0f1',
                                               font=('Consolas', 9),
                                               wrap=tk.WORD)
        self.console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.console.config(state=tk.DISABLED)
        
        # Results tab
        results_tab = tk.Frame(self.notebook, bg='#f0f0f0')
        
        # Performance metrics frame
        metrics_frame = tk.LabelFrame(results_tab, text="Performance Metrics", 
                                    font=('Arial', 10, 'bold'),
                                    bg='#f0f0f0', fg='#2c3e50', padx=10, pady=10)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, 
                                                    height=10,
                                                    font=('Consolas', 8),
                                                    wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.config(state=tk.DISABLED)
        
        # Data preview frame
        preview_frame = tk.LabelFrame(results_tab, text="Data Preview", 
                                    font=('Arial', 10, 'bold'),
                                    bg='#f0f0f0', fg='#2c3e50', padx=10, pady=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, 
                                                    font=('Consolas', 8),
                                                    wrap=tk.WORD)
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        self.preview_text.config(state=tk.DISABLED)
        
        # Visualization tab
        self.viz_tab = tk.Frame(self.notebook, bg='#f0f0f0')
        self.viz_canvas_frame = tk.Frame(self.viz_tab, bg='#f0f0f0')
        self.viz_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs to notebook
        self.notebook.add(console_frame, text="Console")
        self.notebook.add(results_tab, text="Results")
        self.notebook.add(self.viz_tab, text="Plots")
        
        # Redirect stdout to console
        self.redirect_stdout()
    
    def redirect_stdout(self):
        """Redirect stdout to the console widget"""
        class StdoutRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
            
            def write(self, string):
                self.text_widget.config(state=tk.NORMAL)
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
                self.text_widget.config(state=tk.DISABLED)
                self.text_widget.update_idletasks()
            
            def flush(self):
                pass
        
        sys.stdout = StdoutRedirector(self.console)
    
    def clear_visualization_frame(self):
        """Clear the visualization frame"""
        for widget in self.viz_canvas_frame.winfo_children():
            widget.destroy()
    
    def on_cyclone_selected(self, event):
        """Handle cyclone selection from dropdown"""
        selected_cyclone = self.cyclone_var.get()
        if selected_cyclone and self.results is not None:
            # Get data for selected cyclone with predict status
            cyclone_data = self.results[(self.results['Name'] == selected_cyclone) & 
                                       (self.results['status'] == 'predict')]
            
            if len(cyclone_data) > 0:
                info_text = f"✓ Selected: {selected_cyclone}\n"
                info_text += f"Predict records: {len(cyclone_data)}"
                self.cyclone_info.config(text=info_text)
                
                # Enable plot buttons
                self.plot_lat_lon_btn.config(state=tk.NORMAL)
                self.plot_intensity_btn.config(state=tk.NORMAL)
                self.plot_all_btn.config(state=tk.NORMAL)
            else:
                self.cyclone_info.config(text=f"No predict data for {selected_cyclone}")
                self.plot_lat_lon_btn.config(state=tk.DISABLED)
                self.plot_intensity_btn.config(state=tk.DISABLED)
                self.plot_all_btn.config(state=tk.DISABLED)
    
    def update_cyclone_dropdown(self):
        """Update the cyclone dropdown with predict status cyclones"""
        if self.results is not None:
            # Get unique cyclones with predict status
            predict_data = self.results[self.results['status'] == 'predict']
            if len(predict_data) > 0:
                cyclones = sorted(predict_data['Name'].unique())
                self.cyclone_dropdown['values'] = cyclones
                if cyclones:
                    self.cyclone_dropdown.current(0)
                    self.on_cyclone_selected(None)
            else:
                self.cyclone_dropdown['values'] = []
                self.cyclone_info.config(text="No predict data available")
                self.plot_lat_lon_btn.config(state=tk.DISABLED)
                self.plot_intensity_btn.config(state=tk.DISABLED)
                self.plot_all_btn.config(state=tk.DISABLED)
    
    def plot_lat_lon(self):
        """Plot latitude and longitude comparisons for selected cyclone with predict status"""
        if self.results is None:
            messagebox.showwarning("Warning", "No results available! Please run predictions first.")
            return
        
        selected_cyclone = self.cyclone_var.get()
        if not selected_cyclone:
            messagebox.showwarning("Warning", "Please select a cyclone first!")
            return
        
        self.clear_visualization_frame()
        self.update_status(f"Generating LAT-LON plots for {selected_cyclone} (Predict)...")
        
        try:
            # Create figure with subplots and more spacing
            fig = Figure(figsize=(14, 10), dpi=100)
            fig.suptitle(f'Cyclone Trajectory Analysis: {selected_cyclone} (Predict)', fontsize=16, y=0.98)
            
            # Get data for selected cyclone with predict status
            cyclone_data = self.results[(self.results['Name'] == selected_cyclone) & 
                                       (self.results['status'] == 'predict')]
            
            if len(cyclone_data) > 0:
                # Plot 1: Trajectory Comparison
                ax1 = fig.add_subplot(221)
                
                # Plot actual trajectory
                ax1.plot(cyclone_data['LON'], cyclone_data['LAT'], 'bo-', linewidth=2, markersize=4, label='Actual')
                # Plot predicted trajectory
                ax1.plot(cyclone_data['predicted_LON'], cyclone_data['predicted_LAT'], 'ro--', linewidth=2, markersize=4, label='Predicted')
                
                # Add start and end markers
                ax1.plot(cyclone_data['LON'].iloc[0], cyclone_data['LAT'].iloc[0], 'g^', markersize=10, label='Start')
                ax1.plot(cyclone_data['LON'].iloc[-1], cyclone_data['LAT'].iloc[-1], 'rv', markersize=10, label='End')
                
                ax1.set_xlabel('Longitude', fontsize=9)
                ax1.set_ylabel('Latitude', fontsize=9)
                ax1.set_title('Trajectory Comparison', fontsize=10, pad=10)
                ax1.legend(loc='best', fontsize=8)
                ax1.grid(True, alpha=0.3)
                
                # Add caption
                ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 2: Latitude Comparison Over Time
                ax2 = fig.add_subplot(222)
                time_points = range(len(cyclone_data))
                ax2.plot(time_points, cyclone_data['next_LAT'], 'b-', linewidth=2, label='Actual LAT')
                ax2.plot(time_points, cyclone_data['predicted_LAT'], 'r--', linewidth=2, label='Predicted LAT')
                ax2.set_xlabel('Time Step', fontsize=9)
                ax2.set_ylabel('Latitude', fontsize=9)
                ax2.set_title('Latitude Prediction Over Time', fontsize=10, pad=10)
                ax2.legend(loc='best', fontsize=8)
                ax2.grid(True, alpha=0.3)
                
                # Add caption
                ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 3: Longitude Comparison Over Time
                ax3 = fig.add_subplot(223)
                ax3.plot(time_points, cyclone_data['next_LON'], 'g-', linewidth=2, label='Actual LON')
                ax3.plot(time_points, cyclone_data['predicted_LON'], 'orange', linewidth=2, label='Predicted LON')
                ax3.set_xlabel('Time Step', fontsize=9)
                ax3.set_ylabel('Longitude', fontsize=9)
                ax3.set_title('Longitude Prediction Over Time', fontsize=10, pad=10)
                ax3.legend(loc='best', fontsize=8)
                ax3.grid(True, alpha=0.3)
                
                # Add caption
                ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 4: Error Analysis
                ax4 = fig.add_subplot(224)
                ax4.plot(time_points, cyclone_data['lat_error'], 'red', linewidth=2, label='Latitude Error')
                ax4.plot(time_points, cyclone_data['lon_error'], 'blue', linewidth=2, label='Longitude Error')
                ax4.set_xlabel('Time Step', fontsize=9)
                ax4.set_ylabel('Absolute Error', fontsize=9)
                ax4.set_title('Prediction Errors Over Time', fontsize=10, pad=10)
                ax4.legend(loc='best', fontsize=8)
                ax4.grid(True, alpha=0.3)
                
                # Add caption
                ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Adjust layout with more spacing
                fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make room for suptitle
                plt.subplots_adjust(wspace=0.3, hspace=0.4)  # More space between subplots
                
                # Embed in tkinter
                canvas = FigureCanvasTkAgg(fig, self.viz_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, side=tk.TOP)
                
                # Add custom toolbar with save button
                toolbar_frame = tk.Frame(self.viz_canvas_frame)
                toolbar_frame.pack(fill=tk.X)
                
                toolbar = CustomNavigationToolbar(canvas, toolbar_frame)
                toolbar.update()
                
                self.notebook.select(2)  # Switch to plots tab
                self.update_status(f"LAT-LON plots generated for {selected_cyclone} (Predict)")
                
            else:
                messagebox.showwarning("Warning", f"No predict data available for {selected_cyclone}")
                self.update_status("No predict data available")
                
        except Exception as e:
            self.log_message(f"✗ Error generating LAT-LON plots: {e}")
            self.update_status("Error generating plots")
    
    def plot_intensity(self):
        """Plot cyclone intensity analysis for selected cyclone with predict status"""
        if self.results is None:
            messagebox.showwarning("Warning", "No results available! Please run predictions first.")
            return
        
        selected_cyclone = self.cyclone_var.get()
        if not selected_cyclone:
            messagebox.showwarning("Warning", "Please select a cyclone first!")
            return
        
        self.clear_visualization_frame()
        self.update_status(f"Generating intensity plots for {selected_cyclone} (Predict)...")
        
        try:
            # Create figure with subplots and more spacing
            fig = Figure(figsize=(14, 10), dpi=100)
            fig.suptitle(f'Cyclone Intensity Analysis: {selected_cyclone} (Predict)', fontsize=16, y=0.98)
            
            # Get data for selected cyclone with predict status
            cyclone_data = self.results[(self.results['Name'] == selected_cyclone) & 
                                       (self.results['status'] == 'predict')]
            
            if len(cyclone_data) > 0:
                # Plot 1: Wind Speed Comparison
                ax1 = fig.add_subplot(221)
                time_points = range(len(cyclone_data))
                ax1.plot(time_points, cyclone_data['next_WindSpeed'], 'b-', linewidth=2, label='Actual Wind Speed')
                ax1.plot(time_points, cyclone_data['predicted_WindSpeed'], 'r--', linewidth=2, label='Predicted Wind Speed')
                ax1.set_xlabel('Time Step', fontsize=9)
                ax1.set_ylabel('Wind Speed (knots)', fontsize=9)
                ax1.set_title('Wind Speed Prediction', fontsize=10, pad=10)
                ax1.legend(loc='best', fontsize=8)
                ax1.grid(True, alpha=0.3)
                
                # Add caption
                ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 2: Pressure Comparison
                ax2 = fig.add_subplot(222)
                ax2.plot(time_points, cyclone_data['next_Pressure'], 'g-', linewidth=2, label='Actual Pressure')
                ax2.plot(time_points, cyclone_data['predicted_Pressure'], 'orange', linewidth=2, label='Predicted Pressure')
                ax2.set_xlabel('Time Step', fontsize=9)
                ax2.set_ylabel('Pressure (mb)', fontsize=9)
                ax2.set_title('Pressure Prediction', fontsize=10, pad=10)
                ax2.legend(loc='best', fontsize=8)
                ax2.grid(True, alpha=0.3)
                
                # Add caption
                ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 3: Wind-Pressure Relationship
                ax3 = fig.add_subplot(223)
                # Scatter plot of actual wind vs pressure
                ax3.scatter(cyclone_data['next_WindSpeed'], cyclone_data['next_Pressure'], 
                           alpha=0.7, label='Actual', s=50, color='blue')
                ax3.scatter(cyclone_data['predicted_WindSpeed'], cyclone_data['predicted_Pressure'], 
                           alpha=0.7, label='Predicted', s=50, color='red')
                
                # Add lines connecting corresponding points
                for i in range(len(cyclone_data)):
                    ax3.plot([cyclone_data['next_WindSpeed'].iloc[i], cyclone_data['predicted_WindSpeed'].iloc[i]],
                            [cyclone_data['next_Pressure'].iloc[i], cyclone_data['predicted_Pressure'].iloc[i]],
                            'gray', alpha=0.3, linewidth=0.5)
                
                ax3.set_xlabel('Wind Speed (knots)', fontsize=9)
                ax3.set_ylabel('Pressure (mb)', fontsize=9)
                ax3.set_title('Wind-Pressure Relationship', fontsize=10, pad=10)
                ax3.legend(loc='best', fontsize=8)
                ax3.grid(True, alpha=0.3)
                
                # Add caption
                ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 4: Intensity Errors Over Time
                ax4 = fig.add_subplot(224)
                ax4.plot(time_points, cyclone_data['wind_error'], 'red', linewidth=2, label='Wind Speed Error')
                ax4.plot(time_points, cyclone_data['pressure_error'], 'blue', linewidth=2, label='Pressure Error')
                ax4.set_xlabel('Time Step', fontsize=9)
                ax4.set_ylabel('Absolute Error', fontsize=9)
                ax4.set_title('Intensity Prediction Errors', fontsize=10, pad=10)
                ax4.legend(loc='best', fontsize=8)
                ax4.grid(True, alpha=0.3)
                
                # Add caption
                ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Adjust layout with more spacing
                fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make room for suptitle
                plt.subplots_adjust(wspace=0.3, hspace=0.4)  # More space between subplots
                
                # Embed in tkinter
                canvas = FigureCanvasTkAgg(fig, self.viz_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, side=tk.TOP)
                
                # Add custom toolbar with save button
                toolbar_frame = tk.Frame(self.viz_canvas_frame)
                toolbar_frame.pack(fill=tk.X)
                
                toolbar = CustomNavigationToolbar(canvas, toolbar_frame)
                toolbar.update()
                
                self.notebook.select(2)
                self.update_status(f"Intensity plots generated for {selected_cyclone} (Predict)")
                
            else:
                messagebox.showwarning("Warning", f"No predict data available for {selected_cyclone}")
                self.update_status("No predict data available")
                
        except Exception as e:
            self.log_message(f"✗ Error generating intensity plots: {e}")
            self.update_status("Error generating plots")
    
    def plot_all(self):
        """Generate all plots for selected cyclone with predict status"""
        if self.results is None:
            messagebox.showwarning("Warning", "No results available! Please run predictions first.")
            return
        
        selected_cyclone = self.cyclone_var.get()
        if not selected_cyclone:
            messagebox.showwarning("Warning", "Please select a cyclone first!")
            return
        
        self.update_status(f"Generating all plots for {selected_cyclone} (Predict)...")
        
        # Create a combined figure with all 8 plots
        self.clear_visualization_frame()
        
        try:
            # Create figure with subplots and more spacing
            fig = Figure(figsize=(16, 12), dpi=100)
            fig.suptitle(f'Complete Cyclone Analysis: {selected_cyclone} (Predict)', fontsize=18, y=0.98)
            
            # Get data for selected cyclone with predict status
            cyclone_data = self.results[(self.results['Name'] == selected_cyclone) & 
                                       (self.results['status'] == 'predict')]
            
            if len(cyclone_data) > 0:
                time_points = range(len(cyclone_data))
                
                # LAT-LON Plots
                # Plot 1: Trajectory Comparison
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
                ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 2: Latitude Comparison Over Time
                ax2 = fig.add_subplot(242)
                ax2.plot(time_points, cyclone_data['next_LAT'], 'b-', linewidth=2, label='Actual LAT')
                ax2.plot(time_points, cyclone_data['predicted_LAT'], 'r--', linewidth=2, label='Predicted LAT')
                ax2.set_xlabel('Time Step', fontsize=8)
                ax2.set_ylabel('Latitude', fontsize=8)
                ax2.set_title('Latitude Prediction', fontsize=9, pad=10)
                ax2.legend(loc='best', fontsize=7)
                ax2.grid(True, alpha=0.3)
                ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 3: Longitude Comparison Over Time
                ax3 = fig.add_subplot(243)
                ax3.plot(time_points, cyclone_data['next_LON'], 'g-', linewidth=2, label='Actual LON')
                ax3.plot(time_points, cyclone_data['predicted_LON'], 'orange', linewidth=2, label='Predicted LON')
                ax3.set_xlabel('Time Step', fontsize=8)
                ax3.set_ylabel('Longitude', fontsize=8)
                ax3.set_title('Longitude Prediction', fontsize=9, pad=10)
                ax3.legend(loc='best', fontsize=7)
                ax3.grid(True, alpha=0.3)
                ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 4: Error Analysis
                ax4 = fig.add_subplot(244)
                ax4.plot(time_points, cyclone_data['lat_error'], 'red', linewidth=2, label='Latitude Error')
                ax4.plot(time_points, cyclone_data['lon_error'], 'blue', linewidth=2, label='Longitude Error')
                ax4.set_xlabel('Time Step', fontsize=8)
                ax4.set_ylabel('Absolute Error', fontsize=8)
                ax4.set_title('Position Errors', fontsize=9, pad=10)
                ax4.legend(loc='best', fontsize=7)
                ax4.grid(True, alpha=0.3)
                ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Intensity Plots
                # Plot 5: Wind Speed Comparison
                ax5 = fig.add_subplot(245)
                ax5.plot(time_points, cyclone_data['next_WindSpeed'], 'b-', linewidth=2, label='Actual Wind Speed')
                ax5.plot(time_points, cyclone_data['predicted_WindSpeed'], 'r--', linewidth=2, label='Predicted Wind Speed')
                ax5.set_xlabel('Time Step', fontsize=8)
                ax5.set_ylabel('Wind Speed (knots)', fontsize=8)
                ax5.set_title('Wind Speed Prediction', fontsize=9, pad=10)
                ax5.legend(loc='best', fontsize=7)
                ax5.grid(True, alpha=0.3)
                ax5.text(0.02, 0.98, '(e)', transform=ax5.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 6: Pressure Comparison
                ax6 = fig.add_subplot(246)
                ax6.plot(time_points, cyclone_data['next_Pressure'], 'g-', linewidth=2, label='Actual Pressure')
                ax6.plot(time_points, cyclone_data['predicted_Pressure'], 'orange', linewidth=2, label='Predicted Pressure')
                ax6.set_xlabel('Time Step', fontsize=8)
                ax6.set_ylabel('Pressure (mb)', fontsize=8)
                ax6.set_title('Pressure Prediction', fontsize=9, pad=10)
                ax6.legend(loc='best', fontsize=7)
                ax6.grid(True, alpha=0.3)
                ax6.text(0.02, 0.98, '(f)', transform=ax6.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 7: Wind-Pressure Relationship
                ax7 = fig.add_subplot(247)
                ax7.scatter(cyclone_data['next_WindSpeed'], cyclone_data['next_Pressure'], 
                           alpha=0.7, label='Actual', s=50, color='blue')
                ax7.scatter(cyclone_data['predicted_WindSpeed'], cyclone_data['predicted_Pressure'], 
                           alpha=0.7, label='Predicted', s=50, color='red')
                
                # Add lines connecting corresponding points
                for i in range(len(cyclone_data)):
                    ax7.plot([cyclone_data['next_WindSpeed'].iloc[i], cyclone_data['predicted_WindSpeed'].iloc[i]],
                            [cyclone_data['next_Pressure'].iloc[i], cyclone_data['predicted_Pressure'].iloc[i]],
                            'gray', alpha=0.3, linewidth=0.5)
                
                ax7.set_xlabel('Wind Speed (knots)', fontsize=8)
                ax7.set_ylabel('Pressure (mb)', fontsize=8)
                ax7.set_title('Wind-Pressure Relationship', fontsize=9, pad=10)
                ax7.legend(loc='best', fontsize=7)
                ax7.grid(True, alpha=0.3)
                ax7.text(0.02, 0.98, '(g)', transform=ax7.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Plot 8: Intensity Errors Over Time
                ax8 = fig.add_subplot(248)
                ax8.plot(time_points, cyclone_data['wind_error'], 'red', linewidth=2, label='Wind Speed Error')
                ax8.plot(time_points, cyclone_data['pressure_error'], 'blue', linewidth=2, label='Pressure Error')
                ax8.set_xlabel('Time Step', fontsize=8)
                ax8.set_ylabel('Absolute Error', fontsize=8)
                ax8.set_title('Intensity Errors', fontsize=9, pad=10)
                ax8.legend(loc='best', fontsize=7)
                ax8.grid(True, alpha=0.3)
                ax8.text(0.02, 0.98, '(h)', transform=ax8.transAxes, 
                        fontsize=12, fontweight='bold', va='top')
                
                # Adjust layout with more spacing
                fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make room for suptitle
                plt.subplots_adjust(wspace=0.3, hspace=0.5)  # More space between subplots
                
                # Embed in tkinter
                canvas = FigureCanvasTkAgg(fig, self.viz_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, side=tk.TOP)
                
                # Add custom toolbar with save button
                toolbar_frame = tk.Frame(self.viz_canvas_frame)
                toolbar_frame.pack(fill=tk.X)
                
                toolbar = CustomNavigationToolbar(canvas, toolbar_frame)
                toolbar.update()
                
                self.notebook.select(2)
                self.update_status(f"All plots generated for {selected_cyclone} (Predict)")
                
            else:
                messagebox.showwarning("Warning", f"No predict data available for {selected_cyclone}")
                self.update_status("No predict data available")
                
        except Exception as e:
            self.log_message(f"✗ Error generating all plots: {e}")
            self.update_status("Error generating plots")
    
    def log_message(self, message):
        """Add message to console"""
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def load_data(self):
        """Load CSV data file"""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.update_status("Loading data...")
                self.progress.start()
                
                self.data = pd.read_csv(filename)
                
                # Update data info
                cyclones = self.data['Name'].nunique() if 'Name' in self.data.columns else 0
                status_dist = self.data['status'].value_counts().to_dict() if 'status' in self.data.columns else {}
                
                info_text = f"✓ Data loaded: {len(self.data)} rows\n"
                info_text += f"Cyclones: {cyclones}\n"
                info_text += f"Status: {status_dist}"
                
                self.data_info.config(text=info_text)
                
                # Enable train button
                self.train_btn.config(state=tk.NORMAL)
                
                self.log_message(f"✓ Data loaded successfully from {filename}")
                self.log_message(f"  Rows: {len(self.data)}, Cyclones: {cyclones}")
                
                self.progress.stop()
                self.update_status("Data loaded successfully")
                
            except Exception as e:
                self.log_message(f"✗ Error loading data: {e}")
                messagebox.showerror("Error", f"Failed to load data: {e}")
                self.progress.stop()
                self.update_status("Error loading data")
    
    def train_model(self):
        """Train the model in a separate thread"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        def train_thread():
            try:
                self.update_status("Training model...")
                self.progress.start()
                self.train_btn.config(state=tk.DISABLED)
                
                # Initialize and train model
                self.model = FixedBoostedCyclonePredictor()
                self.model.train_with_cross_validation(self.data)
                
                # Enable predict button
                self.predict_btn.config(state=tk.NORMAL)
                
                self.progress.stop()
                self.update_status("Model trained successfully")
                
            except Exception as e:
                self.log_message(f"✗ Error during training: {e}")
                self.progress.stop()
                self.update_status("Error during training")
                self.train_btn.config(state=tk.NORMAL)
        
        # Start training in separate thread
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
    
    def make_predictions(self):
        """Make predictions in a separate thread"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train model first!")
            return
        
        def predict_thread():
            try:
                self.update_status("Making predictions...")
                self.progress.start()
                self.predict_btn.config(state=tk.DISABLED)
                
                # Make predictions
                self.results = self.model.predict(self.data)
                
                # Update results display
                self.display_results()
                
                # Update cyclone dropdown
                self.update_cyclone_dropdown()
                
                # Enable save button
                self.save_btn.config(state=tk.NORMAL)
                
                self.progress.stop()
                self.update_status("Predictions completed")
                
            except Exception as e:
                self.log_message(f"✗ Error during prediction: {e}")
                self.progress.stop()
                self.update_status("Error during prediction")
                self.predict_btn.config(state=tk.NORMAL)
        
        # Start prediction in separate thread
        thread = threading.Thread(target=predict_thread)
        thread.daemon = True
        thread.start()
    
    def display_results(self):
        """Display results in the results tab"""
        if self.results is not None and self.model is not None:
            # Switch to results tab
            self.notebook.select(1)
            
            # Update metrics text
            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete(1.0, tk.END)
            
            # Add performance summary
            test_data = self.results[self.results['status'] == 'test']
            train_data = self.results[self.results['status'] == 'train']
            predict_data = self.results[self.results['status'] == 'predict']
            
            if len(test_data) > 0:
                self.metrics_text.insert(tk.END, "📊 TEST SET PERFORMANCE:\n")
                self.metrics_text.insert(tk.END, "="*40 + "\n")
                
                if 'overall_absolute_error' in test_data.columns:
                    self.metrics_text.insert(tk.END, f"Overall MAE:  {test_data['overall_absolute_error'].mean():.4f}\n")
                    self.metrics_text.insert(tk.END, f"Overall RMSE: {np.sqrt(test_data['overall_squared_error'].mean()):.4f}\n\n")
                
                if all(col in test_data.columns for col in ['lat_error', 'lon_error', 'wind_error', 'pressure_error']):
                    self.metrics_text.insert(tk.END, f"Latitude Error:  {test_data['lat_error'].mean():.4f}\n")
                    self.metrics_text.insert(tk.END, f"Longitude Error: {test_data['lon_error'].mean():.4f}\n")
                    self.metrics_text.insert(tk.END, f"Wind Speed Error: {test_data['wind_error'].mean():.4f}\n")
                    self.metrics_text.insert(tk.END, f"Pressure Error:  {test_data['pressure_error'].mean():.4f}\n\n")
            
            # Add predict data summary
            if len(predict_data) > 0:
                self.metrics_text.insert(tk.END, "🔮 PREDICT DATA SUMMARY:\n")
                self.metrics_text.insert(tk.END, "="*40 + "\n")
                self.metrics_text.insert(tk.END, f"Total predict records: {len(predict_data)}\n")
                self.metrics_text.insert(tk.END, f"Cyclones with predict data: {predict_data['Name'].nunique()}\n\n")
                
                # List cyclones with predict data
                self.metrics_text.insert(tk.END, "Cyclones with predict data:\n")
                for cyclone in sorted(predict_data['Name'].unique()):
                    count = len(predict_data[predict_data['Name'] == cyclone])
                    self.metrics_text.insert(tk.END, f"  - {cyclone}: {count} records\n")
                self.metrics_text.insert(tk.END, "\n")
            
            # Add cyclone metrics
            if hasattr(self.model, 'cyclone_metrics'):
                self.metrics_text.insert(tk.END, "🌀 CYCLONE PERFORMANCE:\n")
                self.metrics_text.insert(tk.END, "="*40 + "\n")
                
                for cyclone, metrics in self.model.cyclone_metrics.items():
                    if 'test' in metrics:
                        self.metrics_text.insert(tk.END, f"{cyclone}: MAE = {metrics['test']['MAE']:.4f}\n")
            
            self.metrics_text.config(state=tk.DISABLED)
            
            # Update preview text
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete(1.0, tk.END)
            
            # Show preview of results
            preview_columns = ['Name', 'status', 'LAT', 'LON', 'predicted_LAT', 'predicted_LON']
            existing_columns = [col for col in preview_columns if col in self.results.columns]
            
            if existing_columns:
                preview_df = self.results[existing_columns].head(20)
                self.preview_text.insert(tk.END, "Preview of Results (first 20 rows):\n")
                self.preview_text.insert(tk.END, "="*50 + "\n")
                self.preview_text.insert(tk.END, preview_df.to_string(index=False))
            else:
                self.preview_text.insert(tk.END, "No preview available")
            
            self.preview_text.config(state=tk.DISABLED)
    
    def save_results(self):
        """Save results to file"""
        if self.model is None or self.results is None:
            messagebox.showwarning("Warning", "No results to save!")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="Save results as",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                self.update_status("Saving results...")
                self.model.combined_results = self.results
                self.model.save_results(filename)
                self.update_status("Results saved successfully")
                messagebox.showinfo("Success", f"Results saved to {filename}")
                
        except Exception as e:
            self.log_message(f"✗ Error saving results: {e}")
            messagebox.showerror("Error", f"Failed to save results: {e}")
            self.update_status("Error saving results")

def main():
    root = tk.Tk()
    app = CyclonePredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()