# NYC Taxi Trip Time Prediction

This project analyzes millions of New York City taxi trips using machine learning to predict trip duration and drop-off locations. By combining neural network techniques with geographical features, the model achieves strong predictive performance and provides insights into the factors affecting taxi trip times in Manhattan.

---

## Project Overview

- **Data Source:** NYC Taxi & Limousine Commission (TLC) trip records (2024–2025) and Kaggle (2009)  
- **Geographic Focus:** Manhattan (60+ taxi zones)  
- **Records Analyzed:** Several million trips, with focused sampling for modeling  
- **Prediction Target:** Subzones within larger TLC taxi zones  

---

## Key Features

- **Time and Location-Based Analysis:** Leveraging Manhattan’s taxi zone system to analyze travel patterns  
- **Neural Network Model:** 3-layer architecture (64 → 32 → 16 → 1) with ReLU activation  
- **Feature Engineering:** Advanced cyclical encoding for time-based features (e.g., hour, weekday)  
- **Historical Comparison:** Model trained on 2009 data, applied to 20025 trip records to evaluate drop-off locationz 
