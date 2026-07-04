# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib
# import pandas as pd


# matplotlib.use('Agg')
# # Function for Linear Regression
# def fit_linear_regression(x_data, y_data):
#     """ Fits a linear regression model to the latency data """
#     model = LinearRegression()
#     model.fit(x_data.reshape(-1, 1), y_data)
#     y_pred = model.predict(x_data.reshape(-1, 1))
    
#     return model, y_pred


# # Function for Polynomial Regression
# def fit_polynomial_regression(x_data, y_data, degree=2):
#     """ Fits a polynomial regression model of specified degree to the latency data """
#     poly = PolynomialFeatures(degree=degree)
#     x_poly = poly.fit_transform(x_data.reshape(-1, 1))
#     model = LinearRegression()
#     model.fit(x_poly, y_data)
#     y_pred = model.predict(x_poly)
    
#     return model, poly, y_pred


# # Function to plot the regression results
# def plot_regression(x_data, y_data, model, poly=None, degree=2, title='Regression'):
#     """ Plot the regression results for latency """
#     plt.scatter(x_data, y_data, color='blue', label='Original Data')
    
#     if poly:  # For polynomial regression
#         x_range = np.linspace(min(x_data), max(x_data), 1000)
#         x_range_poly = poly.transform(x_range.reshape(-1, 1))
#         plt.plot(x_range, model.predict(x_range_poly), color='red', label=f'Polynomial Degree {degree} Fit')
#     else:  # For linear regression
#         plt.plot(x_data, model.predict(x_data.reshape(-1, 1)), color='green', label='Linear Fit')
    
#     plt.xlabel('CPU vCPU')
#     plt.ylabel('Latency (ms)')
#     plt.title(title)
#     plt.legend()
#     plt.show()


# # Function to calculate regression performance
# def evaluate_regression(y_true, y_pred):
#     """ Calculates performance metrics for the regression model """
#     mse = mean_squared_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     return mse, r2


# # --- Main code for regression modeling ---
# def regression_analysis(vcpu_data, latency_data):
#     # 1. Linear Regression
#     linear_model, linear_y_pred = fit_linear_regression(vcpu_data, latency_data)
#     linear_mse, linear_r2 = evaluate_regression(latency_data, linear_y_pred)

#     print("\nLinear Regression Results:")
#     print(f"MSE: {linear_mse}")
#     print(f"R²: {linear_r2}")
    
#     # Plot Linear Regression
#     plot_regression(vcpu_data, latency_data, linear_model, title="Linear Regression")

    
#     # 2. Polynomial Regression
#     poly_model, poly, poly_y_pred = fit_polynomial_regression(vcpu_data, latency_data, degree=2)
#     poly_mse, poly_r2 = evaluate_regression(latency_data, poly_y_pred)

#     print("\nPolynomial Regression Results:")
#     print(f"MSE: {poly_mse}")
#     print(f"R²: {poly_r2}")
    
#     # Plot Polynomial Regression
#     plot_regression(vcpu_data, latency_data, poly_model, poly, degree=2, title="Polynomial Regression")


# # --- Example usage with sample profiling data ---
# # Replace this with your profiling data
# vcpu_data = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # Example CPU configurations (vCPU)
# latency_data = np.array([1200, 950, 800, 700, 650])  # Example latency data for each vCPU configuration

# file_path = '/mnt/data/HarmonyBatch/CPU_profiling_v3.csv'  # replace with the actual file path
# df = pd.read_csv(file_path)
# # print(df.head())
# # Perform regression analysis
# regression_analysis(np.array(df["vCPU"]), np.array(df["Min_latency(ms)"]))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import pandas as pd

# Load the data (replace this with your CSV data loading method)
df = pd.read_csv('/mnt/data/HarmonyBatch/CPU_profiling_v3.csv')  # Update with your CSV file path
df = df[df['Batch_size'] == 1]  # Filter only batch size = 1

# Group data by 'vCPU' and calculate the average of 'Min_latency(ms)' across different 'Memory' configurations
df_avg_latency = df.groupby('vCPU')['Min_latency(ms)'].mean().reset_index()

# Now you have the average minimum latency for each vCPU across different memory configurations
X = df_avg_latency['vCPU'].values.reshape(-1, 1)
y = df_avg_latency['Min_latency(ms)'].values

# Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

# Exponential Regression (using curve fitting)
def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

params, _ = curve_fit(exponential, X.flatten(), y, p0=[4, -1, 1], maxfev=10000)
y_pred_exp = exponential(X.flatten(), *params)

# Evaluate the models
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

mse_exp = mean_squared_error(y, y_pred_exp)
r2_exp = r2_score(y, y_pred_exp)

# Print the results
print("Polynomial Regression Results:")
print(f"MSE: {mse_poly}")
print(f"R²: {r2_poly}")

print("\nExponential Regression Results:")
print(f"MSE: {mse_exp}")
print(f"R²: {r2_exp}")

# Plot the results
plt.figure(figsize=(10, 6))

# Plot original data points (average minimum latency for each vCPU)
plt.scatter(X, y, color='black', label='Data', zorder=5)

# Plot Polynomial Regression (degree 2) result
plt.plot(X, y_pred_poly, color='green', label=f'Poly Degree 2 (R²={r2_poly:.2f})', zorder=3)

# Plot Exponential Regression result
plt.plot(X, y_pred_exp, color='red', label=f'Exponential Regression (R²={r2_exp:.2f})', zorder=2)

plt.xlabel('vCPU')
plt.ylabel('Min Latency (ms)')
plt.title('Polynomial and Exponential Regression for Latency Prediction')
plt.legend()
plt.grid(True)
plt.show()
