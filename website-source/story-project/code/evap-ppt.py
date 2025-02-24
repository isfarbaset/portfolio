import pandas as pd
import plotly.express as px

# Load the cleaned datasets
historic_data = pd.read_csv('./data/clean-data/historic_data_cleaned.csv')
nearterm_data = pd.read_csv('./data/clean-data/nearterm_data_cleaned.csv')

# Combine datasets for some visualizations
combined_data = pd.concat([historic_data, nearterm_data])

# Group by year and calculate the average values for PPT_Summer and Evap_Summer
average_data = combined_data.groupby('year').agg({'PPT_Summer': 'mean', 'Evap_Summer': 'mean'}).reset_index()

# Create the scatter plot to show the relationship between PPT_Summer and Evap_Summer and update layout for better aesthetics
fig = px.scatter(average_data, x='PPT_Summer', y='Evap_Summer',
                 trendline='ols',  # Add a trendline
                 title='Relationship Between Summer Precipitation and Evaporation',
                 labels={'PPT_Summer': 'Summer Precipitation (mm)', 'Evap_Summer': 'Summer Evaporation (mm)'})

fig.update_layout(
    title_font=dict(size=18, family='Arial, sans-serif'),
    xaxis_title='Summer Precipitation (mm)',
    # xaxis=dict(autorange='reversed'),  # Reverse x-axis
    yaxis_title='Summer Evaporation (mm)',
    width=800,
    height=600,
    title_x=0.5
)
