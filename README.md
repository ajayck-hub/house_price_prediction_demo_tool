# House Price Prediction Demo Tool

This is a simple demo tool for predicting house prices using the California Housing dataset and a Multiple Linear Regression model. It is designed to demonstrate basic machine learning workflows such as data loading, preprocessing, model training, evaluation, and inference.

## Features

- **Data Loading**: Automatically fetches the California Housing dataset using `scikit-learn`.
- **Model Training**: Uses a Linear Regression model to fit the data.
- **Evaluation**: Calculates Mean Squared Error (MSE) and R-squared (R²) score to evaluate model performance on test data.
- **Feature Coefficients**: Displays the learned coefficients for each independent variable.
- **Custom Predictions**: Demonstrates how to make predictions for new, unseen data points.

## Requirements

Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Simply run the python script:

```bash
python House_price_preidction.py
```

### Example Output

The script outputs the loaded dataframe, the calculated MSE and R2 scores, the feature coefficients, and finally a predicted price for an example set of features.

## Technologies Used

- Python
- Scikit-learn
- Pandas
- Numpy
