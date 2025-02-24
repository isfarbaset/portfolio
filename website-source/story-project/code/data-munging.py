# Import necessary libraries
import pandas as pd

# Load the datasets
historic_data = pd.read_csv('./data/raw-data/NABR_historic.csv')
nearterm_data = pd.read_csv('./data/raw-data/nearterm_data_2020-2024.csv')

historic_data.head()

nearterm_data.head()

historic_data.columns

nearterm_data.columns

# Clean up column names 
historic_data.columns = historic_data.columns.str.strip()
nearterm_data.columns = nearterm_data.columns.str.strip()

# Examine if there are duplicate entries for these identifiers
print(nearterm_data.duplicated(subset=['long', 'lat', 'year']).sum())

import pandas as pd

# Define a function to clean and convert data types
def clean_and_convert(df):
    
    # List of numeric columns expected to be in the data
    numeric_columns = [
        'Tmax_Summer', 'T_Annual', 'treecanopy', 'Ann_Herb', 'Bare', 'Herb', 'Litter', 'Shrub',
        'DrySoilDays_Summer_whole', 'Evap_Summer', 'ExtremeShortTermDryStress_Summer_whole', 
        'FrostDays_Winter', 'NonDrySWA_Summer_whole', 'PPT_Winter', 'PPT_Summer', 'PPT_Annual',
        'T_Winter', 'T_Summer', 'Tmax_Summer', 'Tmin_Winter', 'VWC_Winter_whole', 
        'VWC_Spring_whole', 'VWC_Summer_whole', 'VWC_Fall_whole'
    ]
    
    # Convert all listed numeric columns to float, handling non-convertible values
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert 'year' to datetime and extract the year
    df['year'] = pd.to_datetime(df['year'], format='%Y', errors='coerce').dt.year

    # # Drop rows with NaN values in these important columns to ensure accurate calculations
    # df = df.dropna(subset=numeric_columns + ['year'])
    
    return df

# Clean and convert each dataset separately
historic_data = clean_and_convert(historic_data)
nearterm_data = clean_and_convert(nearterm_data)

# Aggregate data 
nearterm_data = nearterm_data.groupby(['long', 'lat', 'year']).agg({
    'Tmax_Summer': 'mean',  # Replace 'mean' with 'median' or 'mode' as appropriate
    'T_Annual': 'mean',
    'treecanopy': 'mean',
    'Ann_Herb': 'mean', 
    'Bare': 'mean',
    'Herb': 'mean',
    'Litter': 'mean',
    'Shrub': 'mean',
    'DrySoilDays_Summer_whole': 'max',
    'Evap_Summer': 'mean',
    'ExtremeShortTermDryStress_Summer_whole': 'max',
    'FrostDays_Winter': 'max',
    'NonDrySWA_Summer_whole': 'mean',
    'PPT_Winter': 'mean',
    'PPT_Summer': 'mean',
    'PPT_Annual': 'mean',
    'T_Winter': 'mean',
    'T_Summer': 'mean',
    'Tmax_Summer': 'mean',
    'Tmin_Winter': 'mean',
    'VWC_Winter_whole': 'mean',
    'VWC_Spring_whole': 'mean',
    'VWC_Summer_whole': 'mean',
    'VWC_Fall_whole': 'mean'
}).reset_index()

# Handling missing data, example filling with median
nearterm_data.fillna(nearterm_data.median(), inplace=True)


nearterm_data.head()

# Examine now if there are duplicate entries for these identifiers
print(nearterm_data.duplicated(subset=['long', 'lat', 'year']).sum())

# Aggregate data by grouping on 'long', 'lat', 'year'
historic_data = historic_data.groupby(['long', 'lat', 'year']).agg({
    'Tmax_Summer': 'mean',
    'T_Annual': 'mean',
    'treecanopy': 'mean',
    'Ann_Herb': 'mean', 
    'Bare': 'mean',
    'Herb': 'mean',
    'Litter': 'mean',
    'Shrub': 'mean',
    'DrySoilDays_Summer_whole': 'max',
    'Evap_Summer': 'mean',
    'ExtremeShortTermDryStress_Summer_whole': 'max',
    'FrostDays_Winter': 'max',
    'NonDrySWA_Summer_whole': 'mean',
    'PPT_Winter': 'mean',
    'PPT_Summer': 'mean',
    'PPT_Annual': 'mean',
    'T_Winter': 'mean',
    'T_Summer': 'mean',
    'Tmax_Summer': 'mean',
    'Tmin_Winter': 'mean',
    'VWC_Winter_whole': 'mean',
    'VWC_Spring_whole': 'mean',
    'VWC_Summer_whole': 'mean',
    'VWC_Fall_whole': 'mean'
}).reset_index()


historic_data.head()

# Handling missing data, example filling with median
historic_data.fillna(historic_data.median(), inplace=True)

# Examine now if there are duplicate entries for these identifiers
print(historic_data.duplicated(subset=['long', 'lat', 'year']).sum())

# Calculate the number of NaN values in each column
nan_counts_nearterm = nearterm_data.isna().sum()

# Display the NaN counts
print(nan_counts_nearterm)

# Calculate the number of NaN values in each column
nan_counts_historic = historic_data.isna().sum()

# Display the NaN counts
print(nan_counts_historic)

nearterm_data.to_csv('./data/clean-data/nearterm_data_cleaned.csv', index=False)
historic_data.to_csv('./data/clean-data/historic_data_cleaned.csv', index=False)