import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Список координат Москвы
moscow_locations = [
    (55.7558, 37.6176),  
    (55.8314, 37.6306),  
    (55.6331, 37.8606),  
    (55.5740, 37.6566),  
    (55.7870, 37.4346),  
]

all_data = []  # Для хранения данных

url = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Проходим по всем локациям
for lat, lon in moscow_locations:
    print(f"Запрос данных для: {lat}°N, {lon}°E")
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "aerosol_optical_depth", "dust", "uv_index", "uv_index_clear_sky", "methane"],
        "timezone": "Europe/Moscow",
        "start_date": "2014-01-01",
        "end_date": "2025-01-01",
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
    hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
    hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
    hourly_carbon_dioxide = hourly.Variables(3).ValuesAsNumpy()
    hourly_nitrogen_dioxide = hourly.Variables(4).ValuesAsNumpy()
    hourly_sulphur_dioxide = hourly.Variables(5).ValuesAsNumpy()
    hourly_ozone = hourly.Variables(6).ValuesAsNumpy()
    hourly_aerosol_optical_depth = hourly.Variables(7).ValuesAsNumpy()
    hourly_dust = hourly.Variables(8).ValuesAsNumpy()
    hourly_uv_index = hourly.Variables(9).ValuesAsNumpy()
    hourly_uv_index_clear_sky = hourly.Variables(10).ValuesAsNumpy()
    hourly_methane = hourly.Variables(11).ValuesAsNumpy()
    
    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    
    hourly_data["pm10"] = hourly_pm10
    hourly_data["pm2_5"] = hourly_pm2_5
    hourly_data["carbon_monoxide"] = hourly_carbon_monoxide
    hourly_data["carbon_dioxide"] = hourly_carbon_dioxide
    hourly_data["nitrogen_dioxide"] = hourly_nitrogen_dioxide
    hourly_data["sulphur_dioxide"] = hourly_sulphur_dioxide
    hourly_data["ozone"] = hourly_ozone
    hourly_data["aerosol_optical_depth"] = hourly_aerosol_optical_depth
    hourly_data["dust"] = hourly_dust
    hourly_data["uv_index"] = hourly_uv_index
    hourly_data["uv_index_clear_sky"] = hourly_uv_index_clear_sky
    hourly_data["methane"] = hourly_methane
    
    # Добавляем координаты
    hourly_data["latitude"] = lat
    hourly_data["longitude"] = lon
    
    hourly_dataframe = pd.DataFrame(data = hourly_data)
    all_data.append(hourly_dataframe)
    print(f"  Получено записей: {len(hourly_dataframe)}")

# Объединяем все данные
if all_data:
    combined_dataframe = pd.concat(all_data, ignore_index=True)
    combined_dataframe.to_csv("hourly_data.csv")
    print(f"\nВсего записей: {len(combined_dataframe)}")
    print(f"Сохранено в hourly_data.csv")
else:
    print("Нет данных для сохранения")