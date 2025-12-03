import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib  # для сохранения моделей

# Загрузка данных
df = pd.read_csv('hourly_data.csv', parse_dates=['date'])
# Создаем временные признаки (сезонность по времени)
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['hour'] = df['date'].dt.hour
# Список параметров для предсказания
parameters = ["pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", "nitrogen_dioxide",
              "sulphur_dioxide", "ozone", "aerosol_optical_depth", "dust", "uv_index",
              "uv_index_clear_sky", "methane"]
for param in parameters:
    df[param] = df[param].fillna(df[param].mean())
# Создаем модели для каждого параметра
models = {}
for param in parameters:
    print(f"Обучение модели для {param}...")
    features = parameters.copy()
    features.remove(param)  # исключаем целевой параметр из признаков

    # Добавляем временные признаки
    X = df[features + ['day_of_week', 'month', 'hour']]
    y = df[param]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)  # shuffle=False для временных данных

    # Обучение модели
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Оценка
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{param} - MAE: {mae:.2f}")

    # Сохраняем модель
    models[param] = model
    joblib.dump(model, f"{param}_model.pkl")

# Теперь делаем прогноз на будущие 30 дней (пример)
future_hours = 30 * 24  # 30 дней по часам
last_date = df['date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=future_hours, freq='h')
future_df = pd.DataFrame({'date': future_dates})

# Создаем временные признаки для будущих данных
future_df['day_of_week'] = future_df['date'].dt.dayofweek
future_df['month'] = future_df['date'].dt.month
future_df['hour'] = future_df['date'].dt.hour

# Предсказание каждого параметра
predictions = pd.DataFrame({'date': future_dates})
for param in parameters:
    print(f"Предсказание для {param}...")
    model = joblib.load(f"{param}_model.pkl")
    # Используем средние значения остальных признаков
    features = parameters.copy()
    features.remove(param)
    feature_means = df[features].mean()
    X_future = pd.DataFrame([feature_means.values] * len(future_df), columns=features)
    X_future['day_of_week'] = future_df['day_of_week']
    X_future['month'] = future_df['month']
    X_future['hour'] = future_df['hour']
    predictions[param] = model.predict(X_future)

# Сохраняем предсказания
predictions.to_csv('future_predictions.csv', index=False)
print("Предсказания сохранены в 'future_predictions.csv'.")