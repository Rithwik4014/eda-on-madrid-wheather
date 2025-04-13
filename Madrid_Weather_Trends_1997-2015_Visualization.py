import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r"C:\Users\rithw\Downloads\Madrid Daily Weather 1997-2015.csv", parse_dates=['CET'])

# Fix column names with extra spaces
df.columns = df.columns.str.strip()

# Set visual style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Distribution of Continuous Variables
variables = ['Max TemperatureC', 'Min TemperatureC', 'Dew PointC']
plt.figure(figsize=(15, 5))
for i, var in enumerate(variables, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[var], kde=True, bins=30)
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('distributions.png')
plt.show()

# 2. Time Trends in Temperature and Humidity
df['Year'] = df['CET'].dt.year
df['Month'] = df['CET'].dt.month

monthly_avg = df.groupby(['Year', 'Month'])[['Max TemperatureC', 'Mean Humidity']].mean().reset_index()
monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(day=1))

plt.figure(figsize=(12, 6))
plt.plot(monthly_avg['Date'], monthly_avg['Max TemperatureC'], label='Max Temperature (°C)')
plt.plot(monthly_avg['Date'], monthly_avg['Mean Humidity'], label='Mean Humidity (%)')
plt.title('Monthly Average Max Temperature and Mean Humidity Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.savefig('time_trends.png')
plt.show()

# 3. Summary Statistics
stats = df[['Max TemperatureC', 'Min TemperatureC', 'Dew PointC', 'Max Wind SpeedKm/h']].describe()
stats.loc['range'] = stats.loc['max'] - stats.loc['min']
print("Summary Statistics:")
print(stats)

# 4. Outlier Detection with Boxplots
plt.figure(figsize=(15, 5))
for i, var in enumerate(['Max TemperatureC', 'Min TemperatureC', 'Max Wind SpeedKm/h'], 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[var])
    plt.title(f'Boxplot of {var}')
plt.tight_layout()
plt.savefig('boxplots.png')
plt.show()

# 5. Correlation Heatmap
num_features = ['Max TemperatureC', 'Min TemperatureC', 'Dew PointC', 'Mean Humidity', 'Max Wind SpeedKm/h', 'Mean Sea Level PressurehPa']
corr_matrix = df[num_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Weather Variables')
plt.savefig('correlation_heatmap.png')
plt.show()

# 6. Scatter Plot: Max Temperature vs Dew Point
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Max TemperatureC', y='Dew PointC', data=df)
plt.title('Max Temperature vs Dew Point')
plt.xlabel('Max Temperature (°C)')
plt.ylabel('Dew Point (°C)')
plt.savefig('scatter_temp_dewpoint.png')
plt.show()
