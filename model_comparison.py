import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim

import streamlit as st

# Load the data
data = pd.read_csv('balanced_churn.csv')

# Drop irrelevant columns
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Define categorical columns
categorical_cols = [cname for cname in data.columns if 
                    data[cname].dtype == "object"]

# Define numerical columns
numerical_cols = [cname for cname in data.columns if 
                  data[cname].dtype in ['int64', 'float64']]

# Exclude target column from numerical columns
numerical_cols.remove("Exited")

# Preprocessing for numerical data: standard scaling
numerical_transformer = StandardScaler()

# Preprocessing for categorical data: one-hot encoding
categorical_transformer = OneHotEncoder(drop='first')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Separate target from predictors
y = data.Exited
X = data.drop('Exited', axis=1)

# Split data into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Split train data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=0)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
X_test = preprocessor.transform(X_test)

# Logistic Regression

def train_linear_regression(X_train, y_train):
    """Train a logistic regression model and return the model and training time."""
    start_time = time.time()
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

def evaluate_linear_regression(model, X_valid, y_valid):
    """Evaluate the logistic regression model and return accuracy and prediction time."""
    start_time = time.time()
    
    predictions = model.predict(X_valid)
    
    end_time = time.time()
    prediction_time = end_time - start_time
    accuracy = accuracy_score(y_valid, predictions)
    return accuracy, prediction_time

# XG BOOST

def train_xgboost(X_train, y_train):
    """Train an XGBoost model and return the model and training time."""
    start_time = time.time()
    
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

def evaluate_xgboost(model, X_valid, y_valid):
    """Evaluate the XGBoost model and return accuracy and prediction time."""
    start_time = time.time()
    
    predictions = model.predict(X_valid)
    
    end_time = time.time()
    prediction_time = end_time - start_time
    accuracy = accuracy_score(y_valid, predictions)
    return accuracy, prediction_time

# Neural network

# Define the Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

def train_neural_network(X_train, y_train, epochs=10):
    """Train a Neural Network and return the model and training time."""
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

def evaluate_neural_network(model, X_valid, y_valid):
    """Evaluate the Neural Network and return accuracy and prediction time."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        start_time = time.time()
        
        X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
        outputs = model(X_valid_tensor)
        predicted = (outputs.cpu().numpy() > 0.5).astype(int)
        
        end_time = time.time()

    prediction_time = end_time - start_time
    accuracy = accuracy_score(y_valid, predicted)
    return accuracy, prediction_time


# Streamlit app
def main():
    st.title("Model Speed and Accuracy Comparison")

    sample_size = st.slider("Select the size of the sample from the dataset", 100, len(data), 5000)
    X_sample, y_sample = X[:sample_size], y[:sample_size]
    X_sample = preprocessor.transform(X_sample)

    st.write(f"Selected a sample of {sample_size} data points.")
    
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False

    if st.button("Train All Models"):
        st.write("Training models...")
        
        lr_model, lr_training_time = train_linear_regression(X_sample, y_sample)
        st.write(f"Logistic Regression trained in {lr_training_time:.2f} seconds.")
        st.session_state.lr_model = lr_model
        
        xgb_model, xgb_training_time = train_xgboost(X_sample, y_sample)
        st.write(f"XGBoost trained in {xgb_training_time:.2f} seconds.")
        st.session_state.xgb_model = xgb_model
        
        nn_model, nn_training_time = train_neural_network(X_sample, y_sample)
        st.write(f"Neural Network trained in {nn_training_time:.2f} seconds.")
        st.session_state.nn_model = nn_model

        st.session_state.models_trained = True

    if st.button("Make Predictions and Evaluate"):
        if not st.session_state.models_trained:
            st.warning("Please train the models first!")
            return
        if 'lr_model' not in st.session_state or 'xgb_model' not in st.session_state or 'nn_model' not in st.session_state:
            st.warning("Models are not available in the session. Please train the models first!")
            return

        st.write("Evaluating models on validation data...")

        lr_accuracy, lr_prediction_time = evaluate_linear_regression(st.session_state.lr_model, X_valid, y_valid)
        xgb_accuracy, xgb_prediction_time = evaluate_xgboost(st.session_state.xgb_model, X_valid, y_valid)
        nn_accuracy, nn_prediction_time = evaluate_neural_network(st.session_state.nn_model, X_valid, y_valid)

        # Create a dictionary to hold the results
        results = {
            "Model": ["Logistic Regression", "XGBoost", "Neural Network"],
            "Training Time (s)": [st.session_state.lr_training_time, st.session_state.xgb_training_time, st.session_state.nn_training_time],
            "Accuracy (%)": [lr_accuracy*100, xgb_accuracy*100, nn_accuracy*100],
            "Prediction Time (s)": [lr_prediction_time, xgb_prediction_time, nn_prediction_time]
        }

        # Convert the dictionary to a DataFrame and display it as a table
        results_df = pd.DataFrame(results)
        st.table(results_df.set_index("Model"))


if __name__ == "__main__":
    main()


