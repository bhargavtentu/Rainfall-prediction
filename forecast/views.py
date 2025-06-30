# ✅ All imports at top
import io
import json
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import uuid 
import os

from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from prophet.serialize import model_from_json
from .models import UserInfo
from .serializers import UserInfoSerializer
from .utils.utils import predict_weather, parse_user_query, create_future_df

# ✅ Web Page View
def index(request):
    return render(request, 'copilot.html')


# ✅ Django Template Forecast
def forecast(request):
    if request.method == 'POST':
        query = request.POST.get('query', '')
        base_dir = settings.BASE_DIR
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

        try:
            result = predict_weather(query, prophet_model, clf_models, reg_models, farm_df, scaler)
            return render(request, 'results.html', {'result': result, 'query': query})
        except Exception as e:
            return render(request, 'results.html', {'error': f"Error: {str(e)}", 'query': query})
    return render(request, 'index.html')


# ✅ JSON API Endpoint with Graph
@csrf_exempt
def forecast_api(request):
    if request.method == 'POST':
        auth_header = request.headers.get('Authorization', '')
        expected_token = 'Bearer abc123xyz'

        if auth_header != expected_token:
            return JsonResponse({'error': 'Unauthorized'}, status=401)

        try:
            data = json.loads(request.body)
            query = data.get('query', '').strip()
            if not query:
                return JsonResponse({"error": "Missing query field"}, status=400)

            dates, horizon_type = parse_user_query(query)
            if dates is None or horizon_type is None:
                return JsonResponse({
                    "error": "Invalid query. Use phrases like 'next week', 'month', or 'year'."
                }, status=400)

            base_dir = settings.BASE_DIR
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

            result_text = predict_weather(query, prophet_model, clf_models, reg_models, farm_df, scaler)

            X_clf_features = ['year', 'month', 'yhat', 'trend', 'tmin', 'inversedistance',
                              'windspeed', 'vaporpressure', 'humidity', 'prophet_residual',
                              'y_lag1', 'tmin_7d_mean']

            future_df = create_future_df(dates, farm_df, X_clf_features)
            future_forecast = prophet_model.predict(future_df[['ds']])
            future_forecast['ds'] = pd.to_datetime(future_forecast['ds']).dt.tz_localize(None)

            future_df = pd.merge(
                future_df,
                future_forecast[['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper']],
                on='ds', how='left'
            )

            future_df['precipitation'] = future_df['yhat'].apply(lambda x: max(0, np.expm1(x)))

            def generate_chart(df, horizon, path):
                fig, ax = plt.subplots(figsize=(10, 4))
                df = df.dropna(subset=['ds', 'precipitation'])
                if df.empty:
                    return
                if df['precipitation'].max() == 0:
                    df['precipitation'] += 0.01
                    ax.set_ylim(0, 0.02)
                if horizon == "short":
                    ax.plot(df['ds'], df['precipitation'], marker='o', label='Rainfall (mm)')
                    ax.set_title('Rainfall Forecast (Daily)')
                elif horizon == "medium":
                    ax.bar(df['ds'].dt.strftime('%Y-%m-%d'), df['precipitation'], color='skyblue')
                    ax.set_title('Rainfall Forecast (Weekly)')
                elif horizon == "long":
                    df = df.dropna(subset=['yhat_lower', 'yhat_upper'])
                    ax.plot(df['ds'], df['precipitation'], marker='s', linestyle='-', label='Monthly Rainfall')
                    ax.fill_between(df['ds'], df['yhat_lower'], df['yhat_upper'], alpha=0.3, color='gray', label='Confidence Interval')
                    ax.set_title('Rainfall Forecast (Monthly)')
                ax.set_xlabel('Date')
                ax.set_ylabel('Precipitation (mm)')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(path, format='png')
                plt.close(fig)

            charts_dir = os.path.join(settings.MEDIA_ROOT, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            filename = f"forecast_chart_{uuid.uuid4().hex[:8]}.png"
            full_path = os.path.join(charts_dir, filename)
            generate_chart(future_df, horizon_type, full_path)

            chart_url = f"{request.scheme}://{request.get_host()}/media/charts/{filename}"

            # 🌟 Final markdown-style output for chatbot
            markdown_result = (
                f"🌧️ **Rainfall Forecast ({horizon_type.capitalize()})**\n\n"
                f"{result_text}\n\n"
                f"🖼️ **Rainfall Forecast Chart**\n\n"
                f"![Rainfall Chart]({chart_url})\n\n"
                f"🔍 [View full-size chart]({chart_url})"
            )

            return JsonResponse({
                "result": markdown_result,
                "chart_url": chart_url
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST allowed"}, status=405)

@api_view(['POST'])
def add_user(request):
    serializer = UserInfoSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response({"message": "User info saved successfully", "data": serializer.data}, status=201)
    return Response(serializer.errors, status=400)


@api_view(['GET'])
def get_users(request):
    auth_header = request.headers.get('Authorization', '')
    expected_token = 'Bearer abc123xyz'
    if auth_header != expected_token:
        return Response({'error': 'Unauthorized'}, status=401)

    name = request.GET.get('name', None)
    users = UserInfo.objects.filter(name__iexact=name) if name else UserInfo.objects.all()
    if not users.exists():
        return Response({'message': 'No user(s) found'}, status=404)
    serializer = UserInfoSerializer(users, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def create_sample_users(request):
    if not UserInfo.objects.exists():
        UserInfo.objects.create(name="Bhargav", place="Hyderabad", phone_number="9876543210")
        UserInfo.objects.create(name="Mani", place="Hyderabad", phone_number="9876543212")
        UserInfo.objects.create(name="Harish", place="Hyderabad", phone_number="9876003212")
        UserInfo.objects.create(name="Gupta", place="Kaikalur", phone_number="9872303212")
        UserInfo.objects.create(name="Jeevitha", place="Kurnool", phone_number="9870453210")
        return Response({"message": "Sample users added"})
    return Response({"message": "Users already exist"})

