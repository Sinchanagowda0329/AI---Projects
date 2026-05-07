# LSTM Supply Chain Demand Forecasting

A time-series forecasting project that predicts supply chain demand using a Long Short-Term Memory (LSTM) neural network built with TensorFlow/Keras.

---

## Project Overview

Accurate demand forecasting is a critical challenge in supply chain management. Over-forecasting leads to excess inventory and storage costs; under-forecasting causes stockouts and lost revenue. This project demonstrates how LSTM networks — a class of recurrent neural networks designed to learn patterns in sequential data — can be applied to forecast demand over time.

---

## Dataset

This project uses synthetically generated demand data for demonstration purposes. In a real-world deployment, replace this with historical demand records from your ERP or warehouse management system.

| Property | Value |
|---|---|
| Data Points | 200 days |
| Feature | Daily demand (integer, range 50–200) |
| Sequence Length | 5 time steps |
| Train / Test Split | 80% / 20% |

---

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

## How It Works

### 1. Data Generation
Synthetic daily demand values are generated using NumPy. For production use, replace this with real historical data loaded from a CSV or database.

### 2. Normalization
Data is scaled to the range [0, 1] using `MinMaxScaler`. This is essential for LSTM training stability since the network is sensitive to the magnitude of input values.

### 3. Sequence Creation
The raw time series is transformed into supervised learning format using a sliding window of `time_steps = 5`. Each input sample is a window of 5 consecutive demand values, and the target is the next value.

```
Input:  [day1, day2, day3, day4, day5]  →  Target: day6
Input:  [day2, day3, day4, day5, day6]  →  Target: day7
...
```

### 4. LSTM Model Architecture

| Layer | Type | Units |
|---|---|---|
| Input | LSTM | 50 units |
| Output | Dense | 1 unit |

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Epochs:** 20
- **Batch Size:** 8

### 5. Evaluation
Predictions are inverse-transformed back to the original demand scale and evaluated using:

- **MAE (Mean Absolute Error)** — average absolute difference between predicted and actual demand
- **RMSE (Root Mean Squared Error)** — penalizes large errors more heavily than MAE

---

## Known Limitations & Recommended Improvements

| Issue | Recommendation |
|---|---|
| `activation='relu'` in LSTM | Use default `tanh` — relu can cause exploding gradients in LSTMs |
| Pure random data has no pattern | Replace with real demand data with seasonality/trends |
| No validation during training | Add `validation_data=(X_test, y_test)` to `model.fit()` |
| Single-feature input | Add external features like promotions, holidays, price |

---

## Project Structure

```
├── lstm_forecasting.ipynb    # Main notebook
└── README.md                 # This file
```

---

## Real-World Applications

- **Retail:** Forecast product demand to optimize stock replenishment
- **Manufacturing:** Predict raw material needs to reduce production delays
- **Logistics:** Anticipate shipment volumes for fleet and warehouse planning
- **E-commerce:** Align inventory levels with predicted order volumes

---

## License

This project uses synthetically generated data. No external data license applies. For real deployment, ensure compliance with your data provider's terms.
