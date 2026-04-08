import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load the dataset
df = pd.read_csv('Realistic_Flightdata_200.csv')

# Select the relevant features and the target variable
features = ['Approach Speed (knots)', 'Vertical Speed at Threshold (feet/min)', 'Wind Speed (knots)', 
            'Wind Direction (degrees)', 'Runway Condition', 'Flaps Setting (degrees)', 
            'Aircraft Weight (tons)', 'Pilot Experience (years)']
target = 'Hard Landing (Yes=1/No=0)'

X = df[features]
y = df[target]

# Preprocess the data: Encode categorical variables and scale numerical variables
categorical_features = ['Runway Condition']
numerical_features = list(set(features) - set(categorical_features))

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# Reshape data for CNN: (samples, timesteps, features). Here timesteps=1 since we don't have time series data
X_reshaped = np.reshape(X_processed.toarray(), (X_processed.shape[0], 1, X_processed.shape[1]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(1, X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Save the model
model.save('hard_landing_predictor_model.h5')
