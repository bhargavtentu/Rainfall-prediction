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
        # Check for Bearer Token Authorization
        auth_header = request.headers.get('Authorization', '')
        expected_token = 'Bearer abc123xyz'  # Set your custom token

        if auth_header != expected_token:
            return JsonResponse({'error': 'Unauthorized'}, status=401)

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
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import UserInfo
from.serializers import UserInfoSerializer


@api_view(['POST'])
def add_user(request):
    serializer = UserInfoSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response({"message": "User info saved successfully", "data": serializer.data}, status=201)
    return Response(serializer.errors, status=400)

@api_view(['GET'])
def get_users(request):
    # Authorization check
    auth_header = request.headers.get('Authorization', '')
    expected_token = 'Bearer abc123xyz'

    if auth_header != expected_token:
        return Response({'error': 'Unauthorized'}, status=401)

    name = request.GET.get('name', None)
    if name:
        users = UserInfo.objects.filter(name__iexact=name)
    else:
        users = UserInfo.objects.all()

    if not users.exists():
        return Response({'message': 'No user(s) found'}, status=404)

    serializer = UserInfoSerializer(users, many=True)
    return Response(serializer.data)

from .models import User
from rest_framework.response import Response
from rest_framework.decorators import api_view

@api_view(['GET'])
def create_sample_users(request):
    if not User.objects.exists():
        User.objects.create(name="Bhargav", place="Hyderabad", phone_number="9876543210")
        User.objects.create(name="Mani", place="Hyderabad", phone_number="9876543212")
        User.objects.create(name="Harish", place="Hyderabad", phone_number="9876003212")
        User.objects.create(name="Gupta", place="Kaikalur", phone_number="9872303212")
        User.objects.create(name="Jeevitha", place="Kurnool", phone_number="9870453210")
        return Response({"message": "Sample users added"})
    return Response({"message": "Users already exist"})

