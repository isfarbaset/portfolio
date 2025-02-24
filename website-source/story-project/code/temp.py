import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the cleaned datasets
historic_data = pd.read_csv('./data/clean-data/historic_data_cleaned.csv')
nearterm_data = pd.read_csv('./data/clean-data/nearterm_data_cleaned.csv')

# Combine datasets for some visualizations
combined_data = pd.concat([historic_data, nearterm_data])

# Extracting and merging the relevant columns for T_Summer analysis
combined_t_summer_data = combined_data[['year', 'T_Summer']]

# Aggregating the data by year using the mean
combined_t_summer_aggregated = combined_t_summer_data.groupby('year').mean().reset_index()

# Calculate trendline for combined data
def calculate_trendline(x, y):
    model = LinearRegression()
    x_reshaped = np.array(x).reshape(-1, 1)
    model.fit(x_reshaped, y)
    trend = model.predict(x_reshaped)
    return trend

t_summer_trend = calculate_trendline(combined_t_summer_aggregated['year'], combined_t_summer_aggregated['T_Summer'])

# Create a custom colorscale 
custom_colorscale = [
    [0.0, 'rgb(255,192,12)'],  # yellow ochre
    [1.0,  'rgb(150,0,0)']  # deep red
]

# Create bubble chart with trendline and update layout
fig = go.Figure(
    data=[
        go.Scatter(
            x=combined_t_summer_aggregated['year'],
            y=combined_t_summer_aggregated['T_Summer'],
            mode='markers',
            marker=dict(
                size=combined_t_summer_aggregated['T_Summer'],
                color=combined_t_summer_aggregated['T_Summer'],
                colorscale=custom_colorscale,
                sizemode='area',
                sizeref=2.*max(combined_t_summer_aggregated['T_Summer'])/(40.**2),  # Adjusted sizeref to make bubbles smaller
                sizemin=3  # Adjusted sizemin to make the smallest bubbles smaller
            ),
            hovertemplate='Year: %{x}<br>Temperature: %{y:.2f} °C<extra></extra>',
            name='Summer Temperature'
        ),
        go.Scatter(
            x=combined_t_summer_aggregated['year'],
            y=t_summer_trend,
            mode='lines',
            line=dict(color='red', dash='dash'),
            hovertemplate='Year: %{x}<br>Trend: %{y:.2f} °C<extra></extra>',
            name='Trendline'
        )
    ],
    layout=dict(
        title='Summer Temperature Trend',
        xaxis_title='Year',
        yaxis_title='Summer Temperature (°C)',
        title_font=dict(size=18, family='Arial, sans-serif'),
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)),
        width=800,
        height=600,
        plot_bgcolor='#f7f7f7',
        hoverlabel=dict(font_size=16),
        showlegend=False,
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

# Display the plot
fig.show()