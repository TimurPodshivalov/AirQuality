from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('hourly_data.csv')
features_for_clustering = ['pm10', 'pm2_5']
for param in features_for_clustering:
    df[param] = df[param].fillna(df[param].mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_for_clustering])

sse = []
k_range = range(1, 100)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.plot(k_range, sse, 'bx-')
plt.xlabel('Количество кластеров')
plt.ylabel('SSE')
plt.title('Метод локтя для определения оптимального числа кластеров')
plt.show()

optimal_k = int(input("Введите число кластеров: "))

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(df['cluster'].value_counts())

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel('pm10 (стандартизированный)')
plt.ylabel('pm2_5 (стандартизированный)')
plt.title('Кластеризация')
plt.show()