# NYC Uber Fare Prediction – EDA & Modeling

This project explores one year of Uber trip data in Manhattan using exploratory data analysis (EDA) and a simple neural network to predict base passenger fares. Data comes from NYC's Taxi & Limousine Commission (TLC) 2024–2025 trip records.

---

## Project Structure

- **EDA**: Initial exploration and filtering
- **Feature Engineering**: Data cleaning and preparation for modeling
- **Modeling**: Neural network prediction for total fare price

---

## Data Overview

- **Source**: NYC TLC (Feb 2024 – Jan 2025)
- **Focus**: Trips starting and ending in Manhattan
- **Records analyzed**: Approximately 1.8 million, sampled down to 130,000 for modeling

---

## Data Cleaning and Filtering

Filtered for Uber rides (`hvfhs_license_num == 'HV0003'`)

**Removed**:
- Base fares equal to 0
- Base fares below $5 or above $120
- Trips where pickup and dropoff location were the same
- Shared rides (`shared_match_flag` and `shared_request_flag` both set to 'N')

**Kept**:
- WAV requests
- Access-A-Ride (MTA-subsidized) trips

---

## Feature Engineering

Extracted time-based features from datetime columns:
- `pickup_year`, `pickup_month`, `pickup_day`, `pickup_hour`, `pickup_minute`, `pickup_second`
- Similar fields for request, on-scene, and dropoff timestamps

Created:
- `total_fare = base_passenger_fare + tolls + tax + fees + tips + driver pay`

---

## Visualizations

- Histograms of fare distributions
- Scatter plots of trip volume (miles × time) vs. base passenger fare
- Bar charts of prediction error by pickup zones
- Distribution analysis for WAV and Access-A-Ride trip types

---

## Modeling: Neural Network

Used a basic feed-forward neural network with Keras:

- 1 hidden layer with 40 units (ReLU activation)
- Linear output to predict continuous total fare
- Optimizer: Adam with learning rate = 0.01

**Evaluation Metrics**:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

```python
model = Sequential()
model.add(Dense(40, input_dim=28, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['root_mean_squared_error', 'mean_absolute_error'])
