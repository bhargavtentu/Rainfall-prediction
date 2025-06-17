import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def safe_mape(y_true, y_pred):
    """Calculate MAPE, handling zeros and NaNs."""
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def ensemble_predict(models, X, is_classifier=False):
    """Averages predictions from multiple models."""
    if is_classifier:
        predictions = np.mean([model.predict_proba(X)[:, 1] for model in models], axis=0)
    else:
        predictions = np.mean([model.predict(X) for model in models], axis=0)
    return predictions

def parse_user_query(query):
    """Parses user query to determine date range and horizon type."""
    query = query.lower()
    if "next week" in query:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=7)
        return pd.date_range(start=start_date, end=end_date, freq='D'), 'short'
    elif "month" in query or "months" in query:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(weeks=4)
        return pd.date_range(start=start_date, end=end_date, freq='W'), 'medium'
    elif "year" in query:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + pd.offsets.MonthEnd(12)
        return pd.date_range(start=start_date, end=end_date, freq='ME'), 'long'
    return None, None

def create_future_df(dates, farm_df, X_clf_features=None):
    """Creates a DataFrame for future predictions."""
    future_df = pd.DataFrame({'ds': dates})
    recent_data = farm_df[farm_df['ds'] >= farm_df['ds'].max() - timedelta(days=30)]
    for col in ['tmin', 'inversedistance', 'windspeed', 'vaporpressure', 'humidity']:
        future_df[col] = recent_data[col].mean()
    future_df['month'] = future_df['ds'].dt.month
    future_df['year'] = future_df['ds'].dt.year
    future_df['y_lag1'] = recent_data['y'].mean()
    future_df['tmin_7d_mean'] = recent_data['tmin'].mean()
    prophet_features = ['yhat', 'trend', 'prophet_residual']
    for feature in X_clf_features:
        if feature not in future_df.columns and feature not in prophet_features:
            future_df[feature] = 0.0
    future_df['ds'] = pd.to_datetime(future_df['ds']).dt.tz_localize(None)
    return future_df

def generate_forecast_output(predictions_df, horizon_type):
    """Generates a formatted forecast output based on the prediction horizon."""
    if horizon_type == "short":
        # Daily rain chance and amount
        results = []
        for _, row in predictions_df.iterrows():
            chance = row['rain_probability'] * 100
            # Use 'precipitation' column which is already the final un-transformed value
            amount = row['precipitation']
            results.append(f"{row['ds'].date()}: {chance:.1f}% chance of rain, {amount:.2f} mm expected")
        return "\n".join(results)

    elif horizon_type == "medium":
        # Aggregate stats for a medium term (e.g., a month)
        output = [f"*Monthly Summary (over {len(predictions_df)} weeks):*"]
        output.append(f"Total expected rainfall: {predictions_df['precipitation'].sum():.2f} mm")
        output.append(f"Expected rainy days (at >10% chance): {(predictions_df['rain_probability'] > 0.1).sum()} out of {len(predictions_df)} weekly intervals\n")
        output.append("*Weekly Precipitation Breakdown:*")
        # Assuming predictions_df already has weekly intervals as per parse_user_query(freq='W')
        for i, row in predictions_df.iterrows():
            # Adjust 'ds' for display if it's the end of the week, to say "Week of..."
            output.append(f"  Week of {row['ds'].date()}: {row['precipitation']:.2f} mm")
        return "\n".join(output)

    elif horizon_type == "long":
        # Long-term trend based on Prophet's output
        output = [f"*Yearly Summary (over {len(predictions_df)} months):*"]
        output.append(f"Total expected rainfall: {predictions_df['precipitation'].sum():.2f} mm\n")
        output.append("*Monthly Precipitation Breakdown:*")
        # Assuming predictions_df already has monthly intervals as per parse_user_query(freq='ME')
        for i, row in predictions_df.iterrows():
            output.append(f"  Month ending {row['ds'].date()}: {row['precipitation']:.2f} mm")
        # Prophet's base forecast and confidence range (log-transformed sums)
        output.append(f"\nProphet's total baseline forecast (sum of monthly log-transformed yhat): {predictions_df['yhat'].sum():.2f}")
        # The sum of daily confidence interval widths, also log-transformed
        conf_range_sum_diff = (predictions_df['yhat_upper'] - predictions_df['yhat_lower']).sum()
        output.append(f"Sum of monthly forecast confidence range (log-transformed): ±{conf_range_sum_diff:.2f}")
        return "\n".join(output)

    else:
        return "Unable to determine forecast horizon."

def predict_weather(query, prophet_model, clf_models, reg_models, farm_df, scaler=None):
    """Predicts weather based on user query using pre-trained models."""
    dates, horizon_type = parse_user_query(query)
    if dates is None:
        return "Invalid query. Please specify a time range like 'next week', 'month', or 'year'."

    # Define features
    X_clf_features = ['year', 'month', 'yhat', 'trend', 'tmin', 'inversedistance', 'windspeed',
                      'vaporpressure', 'humidity', 'prophet_residual', 'y_lag1', 'tmin_7d_mean']
    X_reg_features = ['year', 'month', 'yhat', 'trend', 'tmin', 'inversedistance', 'windspeed',
                      'vaporpressure', 'humidity', 'prophet_residual', 'y_lag1', 'tmin_7d_mean']

    # Create future dataframe
    future_df = create_future_df(dates, farm_df, X_clf_features)

    # Ensure ds is timezone-naive in future_df
    future_df['ds'] = pd.to_datetime(future_df['ds']).dt.tz_localize(None)

    # Predict with Prophet and include confidence intervals
    future_forecast = prophet_model.predict(future_df[['ds']])
    future_forecast['ds'] = pd.to_datetime(future_forecast['ds']).dt.tz_localize(None)

    # Perform merge, now including yhat_lower and yhat_upper
    future_df = pd.merge(future_df, future_forecast[['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper']],
                         on='ds', how='left', validate='one_to_one')

    # Check for merge success
    if 'yhat' not in future_df.columns or 'trend' not in future_df.columns:
        print("Error: 'yhat' or 'trend' not added after merge.")
        # Fallback merge if primary merge failed (e.g., due to index issues)
        future_df = future_df.set_index('ds').join(future_forecast[['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper']].set_index('ds'), how='left').reset_index()
        print("Fallback merge applied. New columns:", future_df.columns.tolist())

    # Check for missing values and fill them
    if future_df['yhat'].isna().any() or future_df['trend'].isna().any() or \
       future_df['yhat_lower'].isna().any() or future_df['yhat_upper'].isna().any():
        print("Warning: Missing values in yhat, trend, yhat_lower, or yhat_upper after merge.")
        print("Rows with missing Prophet forecast components:\n", future_df[future_df['yhat'].isna() | future_df['trend'].isna()][['ds']])
        # Fill missing values for Prophet components (0.0 is a placeholder, consider better imputation if common)
        future_df['yhat'] = future_df['yhat'].fillna(0.0)
        future_df['trend'] = future_df['trend'].fillna(0.0)
        future_df['yhat_lower'] = future_df['yhat_lower'].fillna(0.0)
        future_df['yhat_upper'] = future_df['yhat_upper'].fillna(0.0)

    # Initialize prophet_residual
    if 'prophet_residual' not in future_df.columns:
        future_df['prophet_residual'] = 0.0

    # Apply scaler transformation
    if scaler:
        try:
            # Ensure all X_clf_features are present before scaling
            missing_cols = [col for col in X_clf_features if col not in future_df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in future_df for scaling: {missing_cols}")
            future_df[X_clf_features] = scaler.transform(future_df[X_clf_features])
        except ValueError as e:
            print(f"Scaler transform error for X_clf_features: {e}")
            print("Available columns:", future_df.columns.tolist())
            raise

    # Prepare features for classifier prediction
    X_clf_future = future_df[X_clf_features]

    future_df['rain_probability'] = ensemble_predict(clf_models, X_clf_future, is_classifier=True)

    # Adjust threshold for is_rain
    rain_threshold = 0.1  # Lowered from 0.5 due to low probabilities
    future_df['is_rain'] = (future_df['rain_probability'] >= rain_threshold).astype(int)

    # Prepare features for regressor prediction
    X_reg_future_base = future_df[future_df['is_rain'] == 1]
    X_reg_future = X_reg_future_base[X_reg_features] # Select only required columns

    future_df['residuals'] = 0.0
    if not X_reg_future.empty:
        if scaler:
            try:
                # Ensure all X_reg_features are present before scaling
                missing_cols_reg = [col for col in X_reg_features if col not in X_reg_future.columns]
                if missing_cols_reg:
                    raise ValueError(f"Missing columns in X_reg_future for scaling: {missing_cols_reg}")
                scaled_data = scaler.transform(X_reg_future)
                X_reg_future_scaled = pd.DataFrame(scaled_data, columns=X_reg_features, index=X_reg_future.index)
            except ValueError as e:
                print(f"Scaler transform error for X_reg_future: {e}")
                print("Available columns in X_reg_future:", X_reg_future.columns.tolist())
                raise
        else:
            X_reg_future_scaled = X_reg_future
        future_df.loc[future_df['is_rain'] == 1, 'residuals'] = ensemble_predict(reg_models, X_reg_future_scaled, is_classifier=False)

    future_df['precipitation'] = np.expm1(future_df['yhat'] + future_df['residuals'])
    future_df['precipitation'] = future_df['precipitation'].clip(lower=0)

    # Use the new generate_forecast_output function
    return generate_forecast_output(future_df, horizon_type)