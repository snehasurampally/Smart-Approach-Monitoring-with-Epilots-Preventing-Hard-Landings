import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

def run():
    # Load the dataset
    file_path = 'Realistic_Flightdata_200.csv'  # Update this path
    data = pd.read_csv(file_path)

    # Selecting the features and target
    features = ['Approach Speed (knots)', 'Altitude at Threshold (feet)', 'Wind Speed (knots)',
                'Runway Condition', 'Aircraft Weight (tons)', 'Pilot Experience (years)']
    target = 'Hard Landing (Yes=1/No=0)'

    X = data[features]
    y = data[target]

    # Preprocessing pipelines for numeric and categorical data
    numeric_features = ['Approach Speed (knots)', 'Altitude at Threshold (feet)', 'Wind Speed (knots)',
                        'Aircraft Weight (tons)', 'Pilot Experience (years)']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_features = ['Runway Condition']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Applying the preprocessing
    X_preprocessed = preprocessor.fit_transform(X)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    # Reshaping the data for CNN input
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Defining the CNN model
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=32, kernel_size=2, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])

    # Compiling the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Training the model
    history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
 
    # Plotting the training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

    # Saving the trained model
    model.save('hard_landing_CNN_model.h5')

if __name__ == "__main__":
    run()
