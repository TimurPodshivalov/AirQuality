import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


df_history = pd.read_csv('hourly_data.csv')
df_future = pd.read_csv('future_predictions.csv')
df_future['date'] = pd.to_datetime(df_future['date'])

# Список газов для анализа
cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "dust", "uv_index"]

# Заполнение пропусков средними значениями из истории
for c in cols:
    median_val = df_history[c].median()
    df_history[c] = df_history[c].fillna(median_val)
    df_future[c] = df_future[c].fillna(median_val)


#МЕТОД ОПРЕДЕЛЕНИЯ ПРИЧИН (Z-SCORE)
stats = {col: {'median': df_history[col].median(), 'std': df_history[col].std() or 1} for col in cols}

def get_dominant_source(row):
    scores = {col: (row[col] - stats[col]['median']) / stats[col]['std'] for col in cols}
    # Формулы влияния (веса) для разных источников
    source_power = {
        "🚗 Трафик": (scores["nitrogen_dioxide"] + scores["carbon_monoxide"]) / 2,
        "🏭 Промзона": scores["sulphur_dioxide"] * 1.5,
        "☀️ Смог": (scores["ozone"] + scores["uv_index"]) if row["uv_index"] > 0.5 else -10,
        "🌪️ Пыль": (scores["dust"] + scores["pm10"]) / 2,
        "🔥 Сжигание": (scores["pm2_5"] + scores["carbon_monoxide"]) / 2.5
    }
    best_source = max(source_power, key=source_power.get)
    
    # Если отклонение меньше 0.3 — воздух считается чистым (Норма)
    return best_source if source_power[best_source] > 0.3 else "✅ Норма"


df_future['smart_cause'] = df_future.apply(get_dominant_source, axis=1)

df_future.to_csv('cause_hourly_report.csv', index=False)
print("\n[ВЫГРУЗКА] Почасовой отчет сохранен в: cause_hourly_report.csv")


sns.set_theme(style="whitegrid")

df_future['day'] = df_future['date'].dt.date
pivot_df = df_future.groupby(['day', 'smart_cause']).size().unstack(fill_value=0)

fig1, ax1 = plt.subplots(figsize=(14, 7))
pivot_df.plot(kind='bar', stacked=True, ax=ax1, colormap='Set2', edgecolor='white')



plt.title('Доминирующие причины загрязнения по дням', fontsize=16)
plt.xlabel('Дата')
plt.ylabel('Часы в сутках')
plt.legend(title="Причина", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('pollution_daily_analysis.png', dpi=300)
print("График причин сохранен в: pollution_daily_analysis.png")

fig2, ax2 = plt.subplots(figsize=(14, 5))
for gas, color in zip(['nitrogen_dioxide', 'sulphur_dioxide', 'pm10'], ['red', 'purple', 'blue']):
    vals = df_future[gas]
    norm_val = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
    ax2.plot(df_future['date'], norm_val, label=gas, color=color, alpha=0.7)



plt.title('Динамика показателей (проверка логики корреляции)')
plt.legend()
plt.tight_layout()

plt.savefig('trends_check.png', dpi=300)
print("График трендов сохранен в: trends_check.png")

print("\nОткрываю окна с графиками...")
plt.show()