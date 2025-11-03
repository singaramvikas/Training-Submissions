"""
Applying NumPy Fundamentals on a Real Dataset
Analyzing weather data using NumPy operations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def load_weather_data():
    """
    Load and prepare weather dataset
    """
    try:
        # Using California housing dataset as a proxy for weather-like data
        california = fetch_openml(name='california_housing', as_frame=True)
        df = california.frame
        
        # Let's treat some columns as weather-related metrics
        # MedInc -> Temperature-like, HouseAge -> Humidity-like, etc.
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Create synthetic weather data as fallback
        print("Creating synthetic weather data...")
        return create_synthetic_weather_data()

def create_synthetic_weather_data():
    """
    Create synthetic weather data for analysis
    """
    np.random.seed(42)
    n_days = 365
    
    # Generate synthetic weather data
    temperatures = 15 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.normal(0, 3, n_days)
    humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(n_days) / 365 + 1) + np.random.normal(0, 10, n_days)
    precipitation = np.random.exponential(2, n_days)
    wind_speed = np.random.gamma(2, 2, n_days)
    
    data = {
        'temperature': temperatures,
        'humidity': humidity,
        'precipitation': precipitation,
        'wind_speed': wind_speed
    }
    
    df = pd.DataFrame(data)
    return df

def analyze_weather_data(df):
    """
    Perform comprehensive analysis using NumPy fundamentals
    """
    print("\n=== Weather Data Analysis using NumPy ===\n")
    
    # Convert to NumPy arrays for analysis
    temperature = df['temperature'].values
    humidity = df['humidity'].values
    precipitation = df['precipitation'].values
    wind_speed = df['wind_speed'].values
    
    # 1. Basic statistics
    print("1. BASIC STATISTICS:")
    print(f"   Temperature - Mean: {np.mean(temperature):.2f}°C, "
          f"Std: {np.std(temperature):.2f}, "
          f"Range: [{np.min(temperature):.2f}, {np.max(temperature):.2f}]")
    
    print(f"   Humidity - Mean: {np.mean(humidity):.2f}%, "
          f"Std: {np.std(humidity):.2f}")
    
    print(f"   Precipitation - Mean: {np.mean(precipitation):.2f}mm, "
          f"Max: {np.max(precipitation):.2f}mm")
    
    # 2. Seasonal analysis
    print("\n2. SEASONAL ANALYSIS:")
    # Split data into seasons (assuming 365 days)
    spring = temperature[80:172]  # Mar 21 - Jun 21
    summer = temperature[172:264] # Jun 21 - Sep 21
    fall = temperature[264:355]   # Sep 21 - Dec 21
    winter = np.concatenate([temperature[0:80], temperature[355:]]) # Dec 21 - Mar 21
    
    print(f"   Spring avg temp: {np.mean(spring):.2f}°C")
    print(f"   Summer avg temp: {np.mean(summer):.2f}°C")
    print(f"   Fall avg temp: {np.mean(fall):.2f}°C")
    print(f"   Winter avg temp: {np.mean(winter):.2f}°C")
    
    # 3. Correlation analysis
    print("\n3. CORRELATION ANALYSIS:")
    # Create correlation matrix
    weather_data = np.column_stack([temperature, humidity, precipitation, wind_speed])
    correlation_matrix = np.corrcoef(weather_data.T)
    
    print("   Correlation Matrix:")
    print("   Temp     Humid    Precip   Wind")
    labels = ['Temp', 'Humid', 'Precip', 'Wind']
    for i, row in enumerate(correlation_matrix):
        print(f"   {labels[i]}", end="")
        for val in row:
            print(f"   {val:6.3f}", end="")
        print()
    
    # 4. Extreme weather events
    print("\n4. EXTREME WEATHER EVENTS:")
    hot_days = np.sum(temperature > np.percentile(temperature, 90))
    rainy_days = np.sum(precipitation > np.percentile(precipitation, 75))
    windy_days = np.sum(wind_speed > np.percentile(wind_speed, 80))
    
    print(f"   Hot days (top 10%): {hot_days} days")
    print(f"   Rainy days (top 25%): {rainy_days} days")
    print(f"   Windy days (top 20%): {windy_days} days")
    
    # 5. Moving averages (smoothing)
    print("\n5. TREND ANALYSIS:")
    window_size = 7  # 7-day moving average
    temp_moving_avg = np.convolve(temperature, np.ones(window_size)/window_size, mode='valid')
    print(f"   7-day moving average - Range: [{np.min(temp_moving_avg):.2f}, {np.max(temp_moving_avg):.2f}]")
    
    # 6. Data normalization
    print("\n6. DATA NORMALIZATION:")
    temp_normalized = (temperature - np.mean(temperature)) / np.std(temperature)
    humidity_normalized = (humidity - np.mean(humidity)) / np.std(humidity)
    
    print(f"   Normalized temperature - Mean: {np.mean(temp_normalized):.3f}, "
          f"Std: {np.std(temp_normalized):.3f}")
    print(f"   Normalized humidity - Mean: {np.mean(humidity_normalized):.3f}, "
          f"Std: {np.std(humidity_normalized):.3f}")
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'precipitation': precipitation,
        'wind_speed': wind_speed,
        'correlation_matrix': correlation_matrix,
        'moving_average': temp_moving_avg
    }

def visualize_analysis(results):
    """
    Create visualizations of the analysis results
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Temperature distribution
    plt.subplot(2, 3, 1)
    plt.hist(results['temperature'], bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    
    # 2. Correlation heatmap
    plt.subplot(2, 3, 2)
    labels = ['Temp', 'Humid', 'Precip', 'Wind']
    im = plt.imshow(results['correlation_matrix'], cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(4), labels)
    plt.yticks(range(4), labels)
    plt.title('Correlation Matrix')
    
    # 3. Time series with moving average
    plt.subplot(2, 3, 3)
    plt.plot(results['temperature'], alpha=0.5, label='Daily Temp')
    plt.plot(range(len(results['moving_average'])), results['moving_average'], 
             'r-', linewidth=2, label='7-day Avg')
    plt.title('Temperature Trend')
    plt.xlabel('Days')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    
    # 4. Scatter plot: Temperature vs Humidity
    plt.subplot(2, 3, 4)
    plt.scatter(results['temperature'], results['humidity'], alpha=0.6)
    plt.title('Temperature vs Humidity')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Humidity (%)')
    
    # 5. Precipitation distribution
    plt.subplot(2, 3, 5)
    plt.hist(results['precipitation'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Precipitation Distribution')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Frequency')
    
    # 6. Wind speed distribution
    plt.subplot(2, 3, 6)
    plt.hist(results['wind_speed'], bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title('Wind Speed Distribution')
    plt.xlabel('Wind Speed')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('weather_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the weather data analysis
    """
    print("Loading weather data...")
    df = load_weather_data()
    
    print(f"\nDataset preview:")
    print(df.head())
    print(df.describe())
    
    # Perform analysis
    results = analyze_weather_data(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_analysis(results)
    
    print("\n=== Analysis Complete ===")
    print("Key insights:")
    print("- NumPy enables efficient statistical analysis of large datasets")
    print("- Vectorized operations make seasonal analysis straightforward")
    print("- Correlation analysis reveals relationships between weather variables")
    print("- Moving averages help identify trends in time series data")

if __name__ == "__main__":
    main()