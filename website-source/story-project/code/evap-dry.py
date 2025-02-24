import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the cleaned datasets
historic_data = pd.read_csv('./data/clean-data/historic_data_cleaned.csv')
nearterm_data = pd.read_csv('./data/clean-data/nearterm_data_cleaned.csv')

# Combine datasets for some visualizations
combined_data = pd.concat([historic_data, nearterm_data])

# Aggregating the data by year using the mean
combined_evap_aggregated = combined_evap_data.groupby('year').mean().reset_index()

# Calculate trendline for Evap_Summer
def calculate_trendline(x, y):
    model = LinearRegression()
    x_reshaped = np.array(x).reshape(-1, 1)
    model.fit(x_reshaped, y)
    trend = model.predict(x_reshaped)
    return trend

# Extracting and merging the relevant columns for ExtremeShortTermDryStress_Summer_whole analysis
combined_dry_stress_data = combined_data[['year', 'ExtremeShortTermDryStress_Summer_whole']]

# Aggregating the data by year using the mean
combined_dry_stress_data_aggregated = combined_dry_stress_data.groupby('year').mean().reset_index()

# Calculate trendline for ExtremeShortTermDryStress_Summer_whole
dry_stress_trend = calculate_trendline(combined_dry_stress_data_aggregated['year'], combined_dry_stress_data_aggregated['ExtremeShortTermDryStress_Summer_whole'])

# Create the Extreme Short Term Dry Stress in Summer trend plot in one step
fig2 = go.Figure(
    data=[
        go.Scatter(
            x=combined_dry_stress_data_aggregated['year'],
            y=combined_dry_stress_data_aggregated['ExtremeShortTermDryStress_Summer_whole'],
            mode='lines+markers',
            line=dict(color='red'),
            hovertemplate='Year: %{x}<br>Dry Stress: %{y}<extra></extra>',
            name='Extreme Short Term Dry Stress'
        ),
        go.Scatter(
            x=combined_dry_stress_data_aggregated['year'],
            y=dry_stress_trend,
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Trendline'
        )
    ],
    layout=dict(
        title='Extreme Short Term Dry Stress in Summer Trend',
        xaxis_title='Year',
        yaxis_title='Extreme Short Term Dry Stress in Summer',
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
        )
    )
)

# Display the second plot
fig2.show()