import os
import pandas as pd
from django.shortcuts import render
from prophet.serialize import model_from_json
import joblib
from .utils.utils import predict_weather

def index(request):
    return render(request, 'copilot.html')

def forecast(request):
    if request.method == 'POST':
        query = request.POST.get('query', '')
        # Load data and models
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
        farm_df = pd.read_excel(os.path.join(base_dir, 'forecast', 'data', 'pluvial_farm_singleID_cleaned.xlsx'))
        farm_df['ds'] = pd.to_datetime(farm_df['Date']).dt.tz_localize(None)
        farm_df = farm_df.rename(columns={'precipitation': 'y'})
        farm_df = farm_df[['ds', 'tmin', 'inversedistance', 'windspeed', 'vaporpressure', 'humidity', 'y']]
        for col in farm_df.select_dtypes(include=['float64']).columns:
            farm_df[col] = farm_df[col].astype('float32')

        with open(os.path.join(base_dir, 'forecast', 'models', 'prophet_model.json'), 'r') as f:
            prophet_model = model_from_json(f.read())
        clf_models = joblib.load(os.path.join(base_dir, 'forecast', 'models', 'clf_models.pkl'))
        reg_models = joblib.load(os.path.join(base_dir, 'forecast', 'models', 'reg_models.pkl'))
        scaler = joblib.load(os.path.join(base_dir, 'forecast', 'models', 'scaler.pkl'))

        # Get prediction
        try:
            result = predict_weather(query, prophet_model, clf_models, reg_models, farm_df, scaler)
            return render(request, 'results.html', {'result': result, 'query': query})
        except Exception as e:
            return render(request, 'results.html', {'error': f"Error processing query: {str(e)}", 'query': query})
    return render(request, 'index.html')

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import os
import pandas as pd
import joblib
from prophet.serialize import model_from_json
from .utils.utils import predict_weather

@csrf_exempt
def forecast_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query = data.get('query', '')

            # Load models and data
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            farm_df = pd.read_excel(os.path.join(base_dir, 'forecast', 'data', 'pluvial_farm_singleID_cleaned.xlsx'))
            farm_df['ds'] = pd.to_datetime(farm_df['Date']).dt.tz_localize(None)
            farm_df = farm_df.rename(columns={'precipitation': 'y'})
            farm_df = farm_df[['ds', 'tmin', 'inversedistance', 'windspeed', 'vaporpressure', 'humidity', 'y']]
            for col in farm_df.select_dtypes(include=['float64']).columns:
                farm_df[col] = farm_df[col].astype('float32')

            with open(os.path.join(base_dir, 'forecast', 'models', 'prophet_model.json'), 'r') as f:
                prophet_model = model_from_json(f.read())
            clf_models = joblib.load(os.path.join(base_dir, 'forecast', 'models', 'clf_models.pkl'))
            reg_models = joblib.load(os.path.join(base_dir, 'forecast', 'models', 'reg_models.pkl'))
            scaler = joblib.load(os.path.join(base_dir, 'forecast', 'models', 'scaler.pkl'))

            result = predict_weather(query, prophet_model, clf_models, reg_models, farm_df, scaler)

            return JsonResponse({"result": result}, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST allowed"}, status=405)

@csrf_exempt
def test_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            return JsonResponse({"received": data}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Only POST allowed"}, status=405)
