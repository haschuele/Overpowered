# Fuzzy Match CAISO Substation Data to GIS Data 
ISOs have access to detailed grid mappings, but these are not publicly available. In order to identify the geolocations of the CAISO Interconnection Queue "Point of Interconnection" variable, we used fuzzy matching. The code below shows a snippet in which we pull latitude and longitude into the Queue data from a separate GIS data source by matching on substation name. This allowed us to plot the Queue projects on a map as well as calculate distance weighting for the cluster algorithm.

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
One of the biggest challenges with renewables is the Duck Curve problem - Solar generates the most electricity mid-day when demand is low. As such, it's desirable for Queue applicants to build or be near energy storage so this power can be conserved. From our research and subject matter expert interviews, we determined that retired power plants are great candidates for energy storage facilities as they're already connected to the grid and zoned for industry usage. In the snippet below, we calculate the distance between each Queue project and its nearest retired power plant. Then we apply inverse distance weighting so Queue projects near a retired plant are weighted exponentially higher than projects that are far away.

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
```
```python
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
```
```python
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
```
```python
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

# Create Project Clusters based on Weighted Features
```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def find_top_k_similarities_for_rows_below_with_weights_and_scores(df, k, weights):
    # Normalize the DataFrame using MinMaxScaler to ensure fair comparison.
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df)
    
    # Apply the weights to the scaled features. This is a simple element-wise multiplication.
    weighted_features = scaled_features * weights
    
    # Dictionary to hold the tuples of indices and scores of the top k similar rows for each row.
    top_k_similar_indices_and_scores = {}
    
    # Calculate cosine similarity for each row against all others, using the weighted features.
    cosine_sim_matrix = cosine_similarity(weighted_features)
    
    for i in range(len(df) - 1):  # Exclude the last row since there are no rows below it to compare.
        # Filter out the current and above rows' similarities for the current row 'i'.
        current_row_similarities = cosine_sim_matrix[i, i+1:]
        
        # Get the indices of the top k values. Since we're looking at a sliced array, add i+1 to correct the indices.
        top_k_indices = np.argsort(-current_row_similarities)[:k]
        
        # Now also retrieve the top k similarity scores using the sorted indices.
        top_k_scores = current_row_similarities[top_k_indices]
        
        # Store these indices and scores as tuples in the dictionary.
        top_k_similar_indices_and_scores[i] = [(i+1+idx, score) for idx, score in zip(top_k_indices, top_k_scores)]
    
    return top_k_similar_indices_and_scores

# Example weights for each feature in your DataFrame.
weights = np.array([15, 1, 1, 1, 1])  # Adjust these weights according to the importance you assign to each feature.

k = 5  # Define how many top similarities you're interested in.

# Assuming `caiso_mvp_input_encoded_df` is your DataFrame.
top_k_similarities_with_weights_and_scores = find_top_k_similarities_for_rows_below_with_weights_and_scores(caiso_mvp_input_encoded_df, k, weights)

# Assuming `caiso_mvp_input` is your original DataFrame with project names or IDs as the index
project_names_mapping = caiso_mvp_input.index

# Create a new dictionary to hold the project names and scores instead of indices
top_k_similarities_with_project_names_and_scores = {}

for row_index, similar_tuples in top_k_similarities_with_weights_and_scores.items():
    # Map the row index to its corresponding project name
    project_name = project_names_mapping[row_index]
    
    # For each tuple in the list (which contains an index and a score),
    # map the index to the project name and keep the score
    similar_project_names_and_scores = [(project_names_mapping[idx], score) for idx, score in similar_tuples]
    
    top_k_similarities_with_project_names_and_scores[project_name] = similar_project_names_and_scores
```
