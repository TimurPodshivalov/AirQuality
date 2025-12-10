import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('hourly_data.csv')

features = ["pm10", "carbon_monoxide", "carbon_dioxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "aerosol_optical_depth", "dust",
            "uv_index", "uv_index_clear_sky", "methane"]
for param in features:
    df[param] = df[param].fillna(df[param].mean())
X = df[features]

df['pm2_5_level'] = (df['pm2_5'] > df['pm2_5'].median()).astype(int)
y = df['pm2_5_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели KNN: {accuracy:.2f}')

print('Отчёт по классификации:')
print(classification_report(y_test, y_pred))