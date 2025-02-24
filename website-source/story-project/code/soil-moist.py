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

# Extracting the relevant column for VWC_Summer_whole analysis
historic_soil_data_vwc = historic_data[['year', 'VWC_Summer_whole']]
nearterm_soil_data_vwc = nearterm_data[['year', 'VWC_Summer_whole']]
combined_soil_data_vwc = pd.concat([historic_soil_data_vwc, nearterm_soil_data_vwc])
combined_soil_data_vwc_aggregated = combined_soil_data_vwc.groupby('year').mean().reset_index()


# Function to calculate trendline
def calculate_trendline(x, y):
    model = LinearRegression()
    x_reshaped = np.array(x).reshape(-1, 1)
    model.fit(x_reshaped, y)
    trend = model.predict(x_reshaped)
    return trend

# Calculate trendline for VWC_Summer_whole
vwc_trend = calculate_trendline(combined_soil_data_vwc_aggregated['year'], combined_soil_data_vwc_aggregated['VWC_Summer_whole'])

# Determine the y-axis range to better show changes in moisture levels
y_min = combined_soil_data_vwc_aggregated['VWC_Summer_whole'].min() - 0.005
y_max = combined_soil_data_vwc_aggregated['VWC_Summer_whole'].max() + 0.005

# Create the first area chart for VWC_Summer_whole with trendline
fig1 = go.Figure(
    data=[
        go.Scatter(
            x=combined_soil_data_vwc_aggregated['year'],
            y=combined_soil_data_vwc_aggregated['VWC_Summer_whole'],
            fill='tozeroy',
            name='Soil Moisture in Summer',
            mode='none',
            fillcolor='#003c54'
        ),
        go.Scatter(
            x=combined_soil_data_vwc_aggregated['year'],
            y=vwc_trend,
            mode='lines',
            name='Trendline',
            line=dict(color='red', dash='dash')
        )
    ],
    layout=dict(
        title='Soil Moisture in Summer Trend',
        xaxis_title='Year',
        yaxis_title='Soil Moisture (%)',
        yaxis=dict(range=[y_min, y_max], tickfont=dict(size=14)),  # Set the y-axis range
        title_font=dict(size=18, family='Arial, sans-serif'),
        xaxis=dict(tickfont=dict(size=14)),
        width=800,
        height=600,
        title_x=0.5,
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

fig1.show()