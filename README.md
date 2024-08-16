# Stock Price Prediction Using LSTM Neural Networks

This project focuses on predicting stock prices using a Long Short-Term Memory (LSTM) neural network, a type of recurrent neural network (RNN) particularly effective for time series prediction tasks. The primary goal is to leverage historical stock data to predict future price trends, providing insights that could be valuable for investors and financial analysts.

## Project Overview

Predicting stock prices is a challenging task due to the inherently volatile and dynamic nature of financial markets. This project aims to address this challenge by building a predictive model based on deep learning techniques. The model uses historical stock prices from the CAC 40 index to predict future closing prices. We have selected the LSTM model due to its ability to capture temporal dependencies and patterns within sequential data.

## Data and Methodology

The dataset used in this project is `preprocessed_CAC40.csv`, which contains daily historical stock data for companies listed in the CAC 40 index. The data includes features such as the opening price, closing price, daily high, daily low, and trading volume. The dataset has been preprocessed to focus on the closing price, which serves as the target variable for prediction.

### Key Steps:

1. **Data Preprocessing**: The data is cleaned and normalized to ensure consistent input for the model. Normalization is crucial to scale the feature values and improve the model's convergence.

2. **Feature Engineering**: We use a sequence of past closing prices over a window of 60 days to predict the price for the next day, capturing temporal trends effectively.

3. **Model Architecture**: The model employs an LSTM network with multiple layers, including dropout layers for regularization to prevent overfitting. The network is trained to minimize the Mean Squared Error (MSE) between predicted and actual prices.

4. **Training and Evaluation**: The model is trained on historical data, with performance metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and directional accuracy being used to evaluate its effectiveness.

## Results and Performance

The LSTM model demonstrates its ability to predict stock price movements with the following performance metrics:

- **Mean Absolute Error (MAE)**: 3.43
- **Root Mean Squared Error (RMSE)**: 4.33
- **R-squared (R²)**: 0.54
- **Accuracy within 5.0% tolerance**: 82.5%

These results highlight the model's proficiency in capturing stock price trends, with a directional accuracy of 82.5% within a 5% tolerance range, indicating that the model correctly predicts the direction of stock price movement in the majority of cases.

## Future Work

Future improvements could involve integrating additional features such as macroeconomic indicators, sentiment analysis from news and social media, and experimenting with more advanced neural network architectures.

## How to Run

### Prerequisites

- Python 3.x
- Required libraries listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
1. Run the script:
```bash
python stock_prediction.py
```
2. The script will train the model and output the performance metrics.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License.
