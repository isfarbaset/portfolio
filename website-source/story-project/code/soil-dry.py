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

# Extracting the relevant column for DrySoilDays_Summer_whole analysis
historic_soil_data_dry = historic_data[['year', 'DrySoilDays_Summer_whole']]
nearterm_soil_data_dry = nearterm_data[['year', 'DrySoilDays_Summer_whole']]
combined_soil_data_dry = pd.concat([historic_soil_data_dry, nearterm_soil_data_dry])
combined_soil_data_dry_aggregated = combined_soil_data_dry.groupby('year').mean().reset_index()

# Function to calculate trendline
def calculate_trendline(x, y):
    model = LinearRegression()
    x_reshaped = np.array(x).reshape(-1, 1)
    model.fit(x_reshaped, y)
    trend = model.predict(x_reshaped)
    return trend

# Calculate trendline for DrySoilDays_Summer_whole
dry_soil_trend = calculate_trendline(combined_soil_data_dry_aggregated['year'], combined_soil_data_dry_aggregated['DrySoilDays_Summer_whole'])

# Create the second area chart for DrySoilDays_Summer_whole with trendline
fig2 = go.Figure(
    data=[
        go.Scatter(
            x=combined_soil_data_dry_aggregated['year'],
            y=combined_soil_data_dry_aggregated['DrySoilDays_Summer_whole'],
            fill='tozeroy',
            name='Dry Soil Days in Summer',
            mode='none',
            fillcolor='rgba(165,42,42,0.5)'
        ),
        go.Scatter(
            x=combined_soil_data_dry_aggregated['year'],
            y=dry_soil_trend,
            mode='lines',
            name='Trendline',
            line=dict(color='blue', dash='dash')
        )
    ],
    layout=dict(
        title='Dry Soil Days in Summer Trend',
        xaxis_title='Year',
        yaxis_title='Dry Soil Days',
        title_font=dict(size=18, family='Arial, sans-serif'),
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)),
        width=800,
        height=600,
        plot_bgcolor='#f7f7f7',
        hoverlabel=dict(font_size=16),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        title_x=0.5
    )
)

# Display the second plot
fig2.show()