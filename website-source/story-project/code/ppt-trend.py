import pandas as pd
import plotly.express as px
import numpy as np
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load the cleaned datasets
historic_data_1 = pd.read_csv('./data/clean-data/historic_data_cleaned.csv')
nearterm_data_1 = pd.read_csv('./data/clean-data/nearterm_data_cleaned.csv')

# Combine datasets for some visualizations
combined_data_1 = pd.concat([historic_data_1, nearterm_data_1])

# Extracting the relevant column for analysis
historic_ppt_data = historic_data_1[['year', 'PPT_Summer']]
nearterm_ppt_data = nearterm_data_1[['year', 'PPT_Summer']]

# Merging both datasets for a comprehensive analysis
combined_ppt_data = pd.concat([historic_ppt_data, nearterm_ppt_data])

# Aggregating the data by year using the mean
combined_ppt_data_aggregated = combined_ppt_data.groupby('year').mean().reset_index()

# Setting the style of the plot
sns.set_theme(style="whitegrid")

# Plotting the PPT_Summer over the years with enhancements and a trendline
plt.figure(figsize=(16, 9))
sns.lineplot(data=combined_ppt_data_aggregated, x='year', y='PPT_Summer', marker='o', color='navy', linewidth=2.5, label='Summer Precipitation')
sns.regplot(data=combined_ppt_data_aggregated, x='year', y='PPT_Summer', scatter=False, color='#4682B4', ci=None, line_kws={"linewidth":2.5, "linestyle":"--"}, label='Trendline')

# Adding titles and labels with enhanced aesthetics
plt.title('Summer Precipitation Trend\n in Southeastern Utah', fontsize=32)
plt.xlabel('Year', fontsize=25, weight='bold')
plt.ylabel('Summer Precipitation (inches)', fontsize=25, weight='bold')

# Setting the x-axis range from 1980 to 2024
_ = plt.xlim(1980, 2024)

# Adding a grid with customized appearance
_ = plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

# Enhancing the tick labels for better readability
_ = plt.xticks(fontsize=20)
_ = plt.yticks(fontsize=20)

# Adding a light background color to the plot area
plt.gca().set_facecolor('#f7f7f7')

# Removing the top and right spines for a cleaner look
sns.despine()

# Adding legend
_ = plt.legend(fontsize=20)

# Display the plot
plt.show()