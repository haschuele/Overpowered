# Data Cleaning
```python
# Pull in CAISO Queue data
caiso_queue_df = spark.read.csv(f"{blob_url}/capstone_data/strippedCaisoQueueData.csv", header = False).toPandas()

# Clean up columns
project_columns = [
    "Project Name",
    "Queue Position",
    "Interconnection Request Receive Date",
    "Application Status",
    "Study Process",
    "Type-1",
    "Type-2",
    "Type-3",
    "Fuel-1",
    "Fuel-2",
    "Fuel-3",
    "MW-1",
    "MW-2",
    "MW-3",
    "Net MWs to Grid",
    "MWh-1",
    "MWh-2",
    "MWh-3",
    "Full Capacity, Partial or Energy Only (FC/P/EO)",
    "TPD Allocation Percentage",
    "Off-Peak Deliverability and Economic Only",
    "TPD Allocation Group",
    "County",
    "State",
    "Utility",
    "PTO Study Region",
    "Station or Transmission Line",
    "Proposed On-line Date (as filed with IR)",
    "Current On-line Date",
    "Suspension Status",
    "Feasibility Study or Supplemental Review",
    "System Impact Study or Phase I Cluster Study",
    "Facilities Study (FAS) or Phase II Cluster Study",
    "Optional Study (OS)",
    "Interconnection Agreement Status"
]

caiso_queue_df.columns = project_columns
# caiso_queue_df = caiso_queue_df.set_index("Project Name")

# Keep only projects in CA
caiso_queue_df = caiso_queue_df[caiso_queue_df['State'].str.contains('CA', case=False)]

# Clean variables
caiso_queue_df['Net MWs to Grid'] = caiso_queue_df['Net MWs to Grid'].astype(float)
caiso_queue_df['Project Name'] = caiso_queue_df['Project Name'].str.strip()

# Cleaned substation name
regex_pattern = r'\b(?:substation|kv|bus|ValleySubstation|\b\d+\s*kV?)\b'

caiso_queue_df['clean substation name'] = caiso_queue_df['Station or Transmission Line'].apply(lambda x: re.sub(regex_pattern, '', x, flags=re.IGNORECASE).strip() if 'Substation' in x else '')

# Fix substation spot checks
caiso_queue_df.loc[caiso_queue_df['clean substation name'] == 'Litehipe', 'clean substation name'] = 'Lighthipe'
caiso_queue_df.loc[caiso_queue_df['clean substation name'] == 'Coolwater', 'clean substation name'] = 'Cool Water'

# Display
caiso_queue_df
```

# Fuzzy Match CAISO Substation data to GIS Data 
```python
# Filter caiso queue to substations
caiso_queue_substations_df = caiso_queue_df[caiso_queue_df["Station or Transmission Line"].str.contains("Substation", case=False, na=False)]

# Create an empty list to store the matched pairs
matched_substations = []

# Iterate over each row in caiso_queue_substations_df
for idx, row in caiso_queue_substations_df.iterrows():
    caiso_station_name = row['clean substation name']
    
    # Initialize variables to keep track of the best match and its score
    best_match = None
    best_score = -1
    
    # Iterate over each row in gis_substation_df to find the best match
    for _, substation_row in gis_substation_df.iterrows():
        gis_substation_name = substation_row["Name"]
        
        # Calculate the similarity score between the two strings
        score = fuzz.ratio(caiso_station_name.lower(), gis_substation_name.lower())
        
        # Update the best match if the current score is higher
        if score > best_score:
            best_match = gis_substation_name
            best_score = score
            gis_substation_lat = substation_row["Lat"]
            gis_substation_long = substation_row["Lon"]
            
    # Add the best match and its score to the matched_substations list
    matched_substations.append((caiso_station_name, best_match, best_score, gis_substation_lat, gis_substation_long))
    print((caiso_station_name, best_match, best_score))

# Convert the list of matched pairs to a DataFrame
matched_substations_df = pd.DataFrame(matched_substations, columns=["Caiso Station", "Best GIS Substation Match", "Similarity Score", "GIS Lat", "GIS Long"])

matched_substations_df
```

# Calculate Inverse Haversine Distance from Each Project to Nearest Retired Power Plant
```python
# !pip install haversine
from haversine import haversine

# Calculate distances between each queue project and all retired plants
distances = []

for index, row1 in caiso_queue_df.iterrows():
    row_distances = []
    for index, row2 in retired_df.iterrows():
        distance = haversine((row1['GIS Lat'], row1['GIS Long']), (row2['Retired Lat'], row2['Retired Long']))
        row_distances.append(distance)
    distances.append(row_distances)

# Append the inverse min distance to caiso_queue_df
for i, row_distances in enumerate(distances):
    min_distance = min(row_distances)
    inverse_distance = 1/min_distance
    caiso_queue_df.loc[i, 'Retired Plant Inverse Distance'] = inverse_distance

# Find the index of the closest point in retired_df for each row in caiso_queue_df
closest_retired_indices = []

for row in distances:
    # Find the index of the minimum distance in the current row
    closest_retired_idx = np.argmin(row)
    # Append the index to the list
    closest_retired_indices.append(closest_retired_idx)

# Merge the relevant columns from retired_df into caiso_queue_df
for i, retired_idx in enumerate(closest_retired_indices):
    retired_row = retired_df.loc[retired_idx]  # Get the retired row corresponding to the index
    caiso_queue_df.loc[i, 'Retired Lat'] = retired_row['Retired Lat']
    caiso_queue_df.loc[i, 'Retired Long'] = retired_row['Retired Long']
    caiso_queue_df.loc[i, 'Retired Plant Name'] = retired_row['Retired Plant Name']

# Display
caiso_queue_df
```

# Use Historical Queue Data to Model Likelihood of Approval for Current Projects
```python
# Preprocess data for modeling
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Organize columns
categorical_cols = caiso_ml_input.select_dtypes(include=['object', 'category']).columns
numerical_cols = caiso_ml_input.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = numerical_cols.drop('Application Status - Outcome')

# Set up preprocessing
# Preprocessing for numerical data: Impute missing values with the mean and then scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: Impute missing values with a placeholder and apply one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Resample the data so completed projects are evenly represented
X = caiso_ml_input.drop('Application Status - Outcome', axis=1)  
y = caiso_ml_input['Application Status - Outcome']  # Target variable

oversampler = RandomOverSampler(random_state=0)

X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Splitting the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

# Train Logistic Regression model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Create a preprocessing and modeling pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=0, class_weight='balanced'))
])

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model performance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Accessing coefficients (weights)
coefficients = model.named_steps['classifier'].coef_[0]
intercept = model.named_steps['classifier'].intercept_

# Initialize an empty dictionary to store feature names and coefficients
coef_dict = {'Feature': [], 'Coefficient': []}

# Store feature names and coefficients in the dictionary
for feature, coef in zip(X_train.columns, coefficients):
    coef_dict['Feature'].append(feature)
    coef_dict['Coefficient'].append(coef)

# Convert the dictionary to a DataFrame
coef_df = pd.DataFrame(coef_dict)

# Display the DataFrame
coef_df.sort_values(by='Coefficient', ascending=False)
```
