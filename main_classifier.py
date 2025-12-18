import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_history = pd.read_csv('hourly_data.csv')
df_future = pd.read_csv('future_predictions.csv')
df_future['date'] = pd.to_datetime(df_future['date'])

# Список газов для анализа
cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", 
        "sulphur_dioxide", "ozone", "dust", "uv_index"]

# Заполнение пропусковФ
for c in cols:
    median_val = df_history[c].median()
    df_history[c] = df_history[c].fillna(median_val)
    df_future[c] = df_future[c].fillna(median_val)

# МЕТОД ОПРЕДЕЛЕНИЯ ПРИЧИН
stats = {col: {'median': df_history[col].median(), 'std': df_history[col].std() or 1} for col in cols}

def get_dominant_source(row):
    hour = row['date'].hour
    scores = {col: (row[col] - stats[col]['median']) / stats[col]['std'] for col in cols}
    
    # Комбинированные индикаторы
    traffic_score = (scores["nitrogen_dioxide"] * 1.3 + 
                    scores["carbon_monoxide"] * 0.7) / 2
    
    industry_score = (scores["sulphur_dioxide"] * 2.0 + 
                     scores["nitrogen_dioxide"] * 0.5) / 2.5
    
    smog_score = (scores["ozone"] * 1.5 + 
                 scores["uv_index"] * 0.8 + 
                 scores["nitrogen_dioxide"] * 0.3) / 2.6
    
    dust_score = (scores["pm10"] * 1.2 + 
                 scores["dust"] * 1.0 + 
                 scores["pm2_5"] * 0.5) / 2.7
    
    burning_score = (scores["pm2_5"] * 1.5 + 
                    scores["carbon_monoxide"] * 0.8) / 2.3
    
    source_power = {
        "Трафик": traffic_score * (1.5 if 7 <= hour <= 9 or 16 <= hour <= 20 else 1.0),
        "Промзона": industry_score * (1.4 if hour <= 6 or hour >= 22 else 0.8),
        "Смог": smog_score * (1.6 if 9 <= hour <= 17 and row["uv_index"] > 0.2 else 0.5),
        "Пыль": dust_score * (1.3 if 11 <= hour <= 19 else 0.9),
        "Сжигание": burning_score
    }
    
    best_source = max(source_power, key=source_power.get)
    threshold = 0.25 if hour >= 8 and hour <= 22 else 0.15  # разный порог днем/ночью
    return best_source if source_power[best_source] > threshold else "Норма"

df_future['smart_cause'] = df_future.apply(get_dominant_source, axis=1)


df_future['day'] = df_future['date'].dt.date

# ИСКЛЮЧАЕМ ПЕРВЫЙ ДЕНЬ
all_days = sorted(df_future['day'].unique())
days_to_exclude = [all_days[0]]  # первый день

# Фильтруем данные, исключая выбранные дни
df_filtered = df_future[~df_future['day'].isin(days_to_exclude)].copy()

print(f"\n[ФИЛЬТР] Исключаем дни: {days_to_exclude}")
print(f"Было {len(df_future)} записей, стало {len(df_filtered)} записей")

# Группируем по дням и причинам (на отфильтрованных данных)
cause_counts = df_filtered.groupby(['day', 'smart_cause']).size().reset_index(name='hours')
# ========== КОНЕЦ ВСТАВКИ ==========

# СОХРАНЕНИЕ ОТЧЕТА (сохраняем полные данные)
df_future.to_csv('cause_hourly_report.csv', index=False)
print("\n[ВЫГРУЗКА] Почасовой отчет сохранен в: cause_hourly_report.csv")

# ВИЗУАЛИЗАЦИЯ
sns.set_theme(style="whitegrid")

# 1. Проверяем количество записей в день
print("\n[ПРОВЕРКА] Количество часов в каждом дне:")
daily_counts = df_filtered.groupby(df_filtered['date'].dt.date).size()
print(daily_counts)
print(f"\nОжидается 24 часа в сутках, но есть: {daily_counts.unique()}")

# 2. Исправляем проблему с 25 часами (если есть)
# Удаляем дубликаты по дате+час (если есть)
df_filtered['hour'] = df_filtered['date'].dt.hour
duplicates = df_filtered.duplicated(subset=['date', 'hour'], keep=False)
if duplicates.any():
    print(f"\n[ОЧИСТКА] Найдено {duplicates.sum()} дубликатов, удаляем...")
    df_filtered = df_filtered.drop_duplicates(subset=['date', 'hour'], keep='first')

# Фильтруем ТОЛЬКО причины, которые есть в данные (убираем нулевые)
active_causes = cause_counts['smart_cause'].unique()
print(f"\n[ПРИЧИНЫ] Активные причины в данных: {list(active_causes)}")

# Создаем сводную таблицу
pivot_df = cause_counts.pivot(index='day', columns='smart_cause', values='hours').fillna(0)

# 4. График причин по дням (ИСПРАВЛЕННЫЙ)
fig1, ax1 = plt.subplots(figsize=(14, 7))

# Используем только столбцы, которые есть в данных
colors = sns.color_palette('Set2', len(pivot_df.columns))
bottom = np.zeros(len(pivot_df))

for i, cause in enumerate(pivot_df.columns):
    values = pivot_df[cause].values
    ax1.bar(pivot_df.index, values, bottom=bottom, 
            label=cause, color=colors[i], edgecolor='white')
    bottom += values

# Настраиваем ось Y - максимум 24 часа
ax1.set_ylim(0, 24)
ax1.set_yticks(range(0, 25, 4))

plt.title('Доминирующие причины загрязнения по дням', fontsize=16)
plt.xlabel('Дата')
plt.ylabel('Часы')
plt.legend(title="Причина", bbox_to_anchor=(1, 1), loc='upper left')
plt.xticks(rotation=45)


for i, day in enumerate(pivot_df.index):
    total_hours = bottom[i]
    if total_hours != 24:
        ax1.text(i, total_hours + 0.5, f'{int(total_hours)}ч', 
                ha='center', va='bottom', fontsize=9, color='red')

plt.tight_layout()
plt.savefig('pollution_daily_analysis.png', dpi=300)
print("\n[ГРАФИК] График причин сохранен в: pollution_daily_analysis.png")




print("\n" + "="*60)
print("СТАТИСТИКА ПО ПРИЧИНАМ (первый день исключен):")
print("="*60)
total_hours = len(df_filtered)
for cause in active_causes:
    count = (df_filtered['smart_cause'] == cause).sum()
    percent = (count / total_hours) * 100
    print(f"{cause:15s}: {count:4d} часов ({percent:5.1f}%)")

print(f"\nВсего записей: {total_hours}")
print(f"Дней в данных: {len(pivot_df)}")


expected_hours = len(pivot_df) * 24
completeness = (total_hours / expected_hours) * 100
print(f"Полнота данных: {completeness:.1f}%")

print("\nОткрываю окна с графиками...")
plt.show()


