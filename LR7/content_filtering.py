import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Создание DataFrame
data = {
    'Film_ID': [1, 2, 3, 4, 5],
    'Title': ['Die Hard', 'The Martian', 'Pulp Fiction', 'Guardians of the Galaxy', 'La La Land'],
    'Action': [1, 0, 0, 1, 0],
    'Comedy': [0, 0, 0, 1, 0],
    'Drama': [0, 0, 1, 0, 1],
    'Sci-Fi': [0, 1, 0, 1, 0]
}

movies_df = pd.DataFrame(data).set_index('Film_ID')

# Извлечение матрицы признаков (только жанры)
feature_matrix = movies_df[['Action', 'Comedy', 'Drama', 'Sci-Fi']]
print("Матрица Признаков:\n", feature_matrix)
print("\n" + "="*50 + "\n")

# 2. Расчет матрицы сходства
cosine_sim_matrix = cosine_similarity(feature_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, 
                              index=movies_df['Title'], 
                              columns=movies_df['Title'])
print("Матрица Косинусного Сходства:\n")
print(cosine_sim_df.round(4))
print("\n" + "="*50 + "\n")

# 3. Функция рекомендации
def get_recommendations(title, cosine_sim_df, movies_df, top_n=2):
    """
    Возвращает top_n рекомендаций для заданного фильма
    """
    # Получаем оценки сходства для заданного фильма
    sim_scores = cosine_sim_df[title]
    
    # Сортируем фильмы по убыванию сходства
    sim_scores = sim_scores.sort_values(ascending=False)
    
    # Исключаем сам фильм и берем top_n
    top_recommendations = sim_scores[1:top_n+1]
    
    # Создаем DataFrame с результатами
    result_df = pd.DataFrame({
        'Фильм': top_recommendations.index,
        'Сходство': top_recommendations.values.round(4)
    })
    
    return result_df

# Пример использования
print("Рекомендации для 'Die Hard':")
print(get_recommendations('Die Hard', cosine_sim_df, movies_df, top_n=2))
print("\n" + "="*50 + "\n")

print("Рекомендации для 'Pulp Fiction':")
print(get_recommendations('Pulp Fiction', cosine_sim_df, movies_df, top_n=2))
print("\n" + "="*50 + "\n")

print("Рекомендации для 'The Martian':")
print(get_recommendations('The Martian', cosine_sim_df, movies_df, top_n=2))
print("\n" + "="*50 + "\n")

print("Рекомендации для 'La La Land':")
print(get_recommendations('La La Land', cosine_sim_df, movies_df, top_n=2))
print("\n" + "="*50 + "\n")

# Вывод полной матрицы сходства
print("Полная матрица косинусного сходства для анализа:")
print(cosine_sim_df.round(4))co
