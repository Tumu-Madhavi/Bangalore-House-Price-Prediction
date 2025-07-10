import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("bangalore data.csv")  # rename your file as needed

# Keep only useful columns
df = df[['location', 'total_sqft', 'bath', 'bhk', 'price']]

# Handle rare locations
df['location'] = df['location'].apply(lambda x: x.strip())
location_counts = df['location'].value_counts()
rare_locations = location_counts[location_counts <= 10]
df['location'] = df['location'].apply(lambda x: 'other' if x in rare_locations else x)

# One-hot encoding
dummies = pd.get_dummies(df['location'])
df = pd.concat([df, dummies.drop('other', axis=1)], axis=1)
df = df.drop('location', axis=1)

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and column names
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(X.columns, open('columns.pkl', 'wb'))
