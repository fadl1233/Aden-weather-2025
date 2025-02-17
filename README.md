Weather Forecasting for 2025
Weather Forecasting

This project aims to analyze historical weather data, identify trends, and forecast average temperatures for the year 2025 using advanced machine learning techniques, specifically Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) layers.

Table of Contents
1.
Project Overview
2.
Features
3.
Installation
4.
Usage
5.
Data Description
6.
Model Architecture
7.
Training and Evaluation
8.
Results
9.
Contributing
10.
License
11.
Acknowledgments
12.
Contact
Project Overview
The "Weather Forecasting for 2025" project leverages historical weather data to predict future weather conditions. The primary objective is to forecast the average temperature for the year 2025. The project encompasses data preprocessing, exploratory data analysis (EDA), model training, and visualization of the forecasted results.

Key technologies used include:

Python: Programming language
TensorFlow & Keras: For building and training the LSTM model
Pandas & NumPy: For data manipulation
Matplotlib & Seaborn: For data visualization
Scikit-learn: For data preprocessing and evaluation
Features
Data Loading and Preprocessing

Load data from Excel files.
Handle missing values and data cleaning.
Feature engineering and scaling.
Exploratory Data Analysis (EDA)

Visualize historical temperature trends.
Analyze precipitation patterns.
Identify correlations between different weather variables.
Predictive Modeling

Build an LSTM-based RNN model for time series forecasting.
Train the model using historical data.
Tune hyperparameters for optimal performance.
Forecasting

Predict average temperatures for the year 2025.
Generate forecasts for other weather variables if needed.
Visualization

Plot forecasted temperatures alongside historical data.
Visualize other weather variables as needed.
Installation
Prerequisites
Python 3.7+
Git
Steps
1.
Clone the Repository
bash
git clone https://github.com/yourusername/weather-forecasting-2025.git
cd weather-forecasting-2025
2.
Create a Virtual Environment (Recommended)
bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
3.
Install Dependencies
bash
pip install -r requirements.txt
Dependencies:
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
openpyxl
joblib
Usage
1. Prepare the Data
Ensure that the export.xlsx file is placed in the data/ directory or provide the correct path to the file in the script.
2. Run the Script
bash
python waeher_aden.py
What the Script Does:

1.
Data Loading and Preprocessing:
Loads data from the Excel file.
Cleans and preprocesses the data.
Splits the data into training and testing sets.
2.
Model Training:
Builds the LSTM model.
Trains the model on the training data.
Evaluates the model on the test set.
3.
Forecasting:
Generates forecasts for the year 2025.
Saves the forecast results to forecast_2025.csv.
4.
Visualization:
Displays plots of the forecasted temperatures.
Saves the plots to the plots/ directory.
3. View the Results
The forecast data will be saved in forecast_2025.csv.
Visualization plots will be saved in the plots/ directory and displayed automatically.
Data Description
The weather data used in this project is provided in the data/export.xlsx file. The dataset includes the following columns:

date: Date of the observation (YYYY-MM-DD)
tavg: Average temperature (°C)
tmin: Minimum temperature (°C)
tmax: Maximum temperature (°C)
prcp: Precipitation (mm)
wdir: Wind direction (degrees)
wspd: Wind speed (m/s)
pres: Atmospheric pressure (hPa)
Model Architecture
The model is a Sequential model with the following architecture:

1.
LSTM Layer
Units: 50
Return Sequences: True
Activation: tanh
2.
Dropout Layer
Rate: 20%
3.
LSTM Layer
Units: 50
Return Sequences: False
Activation: tanh
4.
Dropout Layer
Rate: 20%
5.
Dense Layer
Units: 1
Activation: linear
Compilation:

Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Training and Evaluation
Epochs: 20
Batch Size: 32
Training Time: Approximately 5 minutes on a standard CPU
Evaluation Metric: Mean Squared Error (MSE)
Sample Output:

Epoch 1/20
100/100 [==============================] - 10s 100ms/step - loss: 0.0500
...
Epoch 20/20
100/100 [==============================] - 5s 50ms/step - loss: 0.0100
Test MSE: 0.0123
Results
The model forecasts for the year 2025 are saved in the forecast_2025.csv file. The forecasted temperatures are visualized in a plot, which is saved in the plots/ directory.

Visualization:

Forecasted Temperatures

Contributing
Contributions are welcome! If you have any ideas, suggestions, or improvements, please follow these steps:

1.
Fork the Repository
2.
Create a New Branch
bash
git checkout -b feature/your-feature-name
3.
Commit Your Changes
bash
git commit -m 'Add some feature'
4.
Push to the Branch
bash
git push origin feature/your-feature-name
5.
Submit a Pull Request
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Thanks to the open-source community for providing the tools and libraries used in this project.
Special thanks to the contributors of TensorFlow, pandas, and other libraries for their excellent work.
Contact
For any questions or inquiries, please contact:

Email: [fadlcom.2025@gmail.com]
