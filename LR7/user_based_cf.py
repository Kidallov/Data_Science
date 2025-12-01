from surprise import Dataset, Reader
from surprise import KNNWithMeans
from surprise.model_selection import train_test_split
from surprise import accuracy
import time

# 1. Загрузка данных MovieLens 100k
print("="*60)
print("ЗАГРУЗКА ДАННЫХ MOVIELENS 100K")
print("="*60)

# Reader нужен, чтобы указать диапазон оценок (от 1 до 5)
data = Dataset.load_builtin('ml-100k')

# Разделение данных на обучающую и тестовую выборки (80% на 20%)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

print(f"Общее количество оценок в наборе данных: 100,000")
print(f"Размер обучающей выборки: {trainset.n_ratings} оценок")
print(f"Размер тестовой выборки: {len(testset)} оценок")
print(f"Количество пользователей: {trainset.n_users}")
print(f"Количество фильмов: {trainset.n_items}")
print("\n" + "="*60 + "\n")

# 2. Настройка и обучение модели User-Based CF с корреляцией Пирсона
print("ОБУЧЕНИЕ МОДЕЛИ USER-BASED CF (ПИРСОН)")
print("="*60)

# Используем корреляцию Пирсона (pearson) и User-Based подход
sim_options_pearson = {
    'name': 'pearson',
    'user_based': True,  # User-Based Collaborative Filtering
    'min_support': 3     # Минимальное количество общих оценок
}

# Инициализация алгоритма
algo_pearson = KNNWithMeans(
    k=40,                # Количество соседей
    min_k=2,             # Минимальное количество соседей
    sim_options=sim_options_pearson,
    verbose=False
)

# Обучение модели
start_time = time.time()
algo_pearson.fit(trainset)
training_time = time.time() - start_time

print(f"Модель User-Based CF (Пирсон) обучена за {training_time:.2f} секунд")
print(f"Количество соседей: {algo_pearson.k}")
print(f"Минимальное количество соседей: {algo_pearson.min_k}")
print("\n" + "="*60 + "\n")

# 3. Оценка качества модели (Пирсон)
print("ОЦЕНКА КАЧЕСТВА МОДЕЛИ (ПИРСОН)")
print("="*60)

# Предсказание на тестовой выборке
start_time = time.time()
predictions_pearson = algo_pearson.test(testset)
prediction_time = time.time() - start_time

# Расчет метрик
rmse_pearson = accuracy.rmse(predictions_pearson, verbose=False)
mae_pearson = accuracy.mae(predictions_pearson, verbose=False)

print(f"Время предсказания: {prediction_time:.2f} секунд")
print(f"RMSE (Среднеквадратичная ошибка): {rmse_pearson:.4f}")
print(f"MAE (Средняя абсолютная ошибка): {mae_pearson:.4f}")
print(f"Количество предсказаний: {len(predictions_pearson)}")

# Анализ распределения ошибок
errors = [abs(pred.r_ui - pred.est) for pred in predictions_pearson[:1000]]
print(f"Средняя ошибка (первые 1000): {sum(errors)/len(errors):.4f}")
print(f"Максимальная ошибка: {max(errors):.4f}")
print(f"Минимальная ошибка: {min(errors):.4f}")

# 4. Сравнение с косинусным сходством
print("\n" + "="*60)
print("СРАВНЕНИЕ: ПИРСОН VS КОСИНУСНОЕ СХОДСТВО")
print("="*60)

# Модель с косинусным сходством
sim_options_cosine = {
    'name': 'cosine',
    'user_based': True,
    'min_support': 3
}

algo_cosine = KNNWithMeans(
    k=40,
    min_k=2,
    sim_options=sim_options_cosine,
    verbose=False
)

algo_cosine.fit(trainset)
predictions_cosine = algo_cosine.test(testset)
rmse_cosine = accuracy.rmse(predictions_cosine, verbose=False)

print(f"RMSE с корреляцией Пирсона: {rmse_pearson:.4f}")
print(f"RMSE с косинусным сходством: {rmse_cosine:.4f}")
print(f"Разница: {abs(rmse_pearson - rmse_cosine):.4f}")
print("\n" + "="*60 + "\n")

# 5. Предсказание оценки для конкретного пользователя
print("ПРЕДСКАЗАНИЕ ОЦЕНКИ ДЛЯ КОНКРЕТНОГО ПОЛЬЗОВАТЕЛЯ")
print("="*60)

user_id = '196'
item_id = '302'  # Фильм "L.A. Confidential"

# Получение предсказания
prediction = algo_pearson.predict(user_id, item_id, verbose=False)

print(f"Пользователь ID: {user_id}")
print(f"Фильм ID: {item_id} ('L.A. Confidential')")
print(f"Предсказанная оценка: {prediction.est:.3f} / 5.0")

# Примеры других предсказаний
print("\nДополнительные примеры предсказаний:")

test_cases = [
    ('196', '50'),    # User 196, Item 50
    ('196', '181'),   # User 196, Item 181
    ('100', '302'),   # User 100, Item 302
    ('100', '50'),    # User 100, Item 50
]

for uid, iid in test_cases:
    pred = algo_pearson.predict(uid, iid, verbose=False)
    print(f"User {uid}, Item {iid}: {pred.est:.3f} / 5.0")

print("\n" + "="*60)
