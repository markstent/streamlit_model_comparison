
# Model Speed and Accuracy Comparison App

This project provides a Streamlit-based web app to compare the training and prediction performance of three machine learning models: Logistic Regression, XGBoost, and a Neural Network (implemented with PyTorch). The app allows users to select a sample size, train all models on this sample, and then evaluate their accuracy and prediction time on unseen data.

View the app: https://modelcomparison.streamlit.app/

## Features

1. **Data Sample Selection**: Choose the size of the data sample to be used for training.
2. **Model Training**: Train all three models on the selected data sample and display training times.
3. **Model Evaluation**: Evaluate the models on validation data, showing both accuracy and prediction time.

## Installation and Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/markstent/streamlit_model_comparison
    cd streamlit_model_comparison
    ```

2. **Set Up a Virtual Environment (Optional but Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit App**:
    ```bash
    streamlit run streamlit_model_comparison.py
    ```

## Data

The app uses a dataset named 'balanced_churn.csv', which contains information about customers and whether they churned or not.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
