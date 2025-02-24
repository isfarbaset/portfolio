import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the cleaned datasets
historic_data = pd.read_csv('./data/clean-data/historic_data_cleaned.csv')
nearterm_data = pd.read_csv('./data/clean-data/nearterm_data_cleaned.csv')

# Combine datasets for some visualizations
combined_data = pd.concat([historic_data, nearterm_data])

# Extracting and merging the relevant columns for Evap_Summer analysis
combined_evap_data = combined_data[['year', 'Evap_Summer']]

# Aggregating the data by year using the mean
combined_evap_aggregated = combined_evap_data.groupby('year').mean().reset_index()

# Calculate trendline for Evap_Summer
def calculate_trendline(x, y):
    model = LinearRegression()
    x_reshaped = np.array(x).reshape(-1, 1)
    model.fit(x_reshaped, y)
    trend = model.predict(x_reshaped)
    return trend

evap_trend = calculate_trendline(combined_evap_aggregated['year'], combined_evap_aggregated['Evap_Summer'])

# Create the Evaporation in Summer trend plot in one step
fig1 = go.Figure(
    data=[
        go.Scatter(
            x=combined_evap_aggregated['year'],
            y=combined_evap_aggregated['Evap_Summer'],
            mode='lines+markers',
            line=dict(color='orange'),
            hovertemplate='Year: %{x}<br>Evaporation: %{y}<extra></extra>',
            name='Evaporation in Summer'
        ),
        go.Scatter(
            x=combined_evap_aggregated['year'],
            y=evap_trend,
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Trendline'
        )
    ],
    layout=dict(
        title='Evaporation in Summer Trend',
        xaxis_title='Year',
        yaxis_title='Evaporation in Summer',
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

# Display the first plot
fig1.show()