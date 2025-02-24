import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the cleaned datasets
historic_data = pd.read_csv('./data/clean-data/historic_data_cleaned.csv')
nearterm_data = pd.read_csv('./data/clean-data/nearterm_data_cleaned.csv')

# Combine datasets for some visualizations
combined_data = pd.concat([historic_data, nearterm_data])

# Group by year and calculate the average values for Tmax_Summer and VWC_Summer_whole
average_data = combined_data.groupby('year').agg({'Tmax_Summer': 'mean', 'VWC_Summer_whole': 'mean'}).reset_index()

# Create the scatter plot to show the relationship between Tmax_Summer and VWC_Summer_whole
fig = px.scatter(average_data, x='Tmax_Summer', y='VWC_Summer_whole',
                 trendline='ols',  # Add a trendline
                 title='Relationship Between Summer Maximum Temperature and Soil Moisture',
                 labels={'Tmax_Summer': 'Maximum Summer Temperature (°C)', 'VWC_Summer_whole': 'Average Soil Moisture (%)'})

# Update layout for better aesthetics
fig.update_layout(
    title_font=dict(size=18, family='Arial, sans-serif'),
    xaxis_title='Maximum Summer Temperature (°C)',
    yaxis_title='Average Soil Moisture (%)',
    width=800,
    height=600,
    title_x=0.5
)