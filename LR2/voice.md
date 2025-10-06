# Лабораторная работа №3 "Деревья решений. Классификатор по голосу."

### Импорт модулей
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
```

### Загрузка данных
```voice_data = pd.read_csv('voice_gender.csv')```

### Разделение данных на признаки (X) и целевую переменную (y)
```
X = voice_data.drop('label', axis=1)
y = voice_data['label']
```

### Преобразование целевой переменной в числовой формат (необходимо для sklearn)
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```
### Разделение данных на обучающую и тестовую выборки
```X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)```

## Задание 1. Решающие пни

### Создание модели дерева решений глубиной 1
```
dt_1 = tree.DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=42)
dt_1.fit(X_train, y_train)
```

### Визуализация дерева
```
plt.figure(figsize=(10, 6))
tree.plot_tree(dt_1, feature_names=X.columns, class_names=['female', 'male'], filled=True)
plt.show()
```
### Вопросы по заданию 1
1. На основе какого фактора будет построено решающее правило в корневой вершине?
Ответ: meanfreq

2. Чему равно оптимальное пороговое значение для данного фактора?
Получаем порог из визуализации дерева или из атрибутов модели
```
threshold = dt_1.tree_.threshold[0]
print(f"Оптимальное пороговое значение: {threshold:.3f}")
```

3. Сколько процентов наблюдений, для которых выполняется заданное в корневой вершине условие, содержится в обучающей выборке?
Это можно вычислить, анализируя результаты работы модели на обучающей выборке.
```
y_pred_train_1 = dt_1.predict(X_train)
meanfreq_values = X_train['meanfreq']
threshold = dt_1.tree_.threshold[0]
```
Находим индексы наблюдений, удовлетворяющих условию в корневой вершине
```
if dt_1.tree_.children_left[0] != -1:  # Проверяем, что есть левый дочерний узел
    indices = np.where(meanfreq_values <= threshold)[0]
else:
    indices = np.array([])  # Если нет левого дочернего узла, то нет наблюдений, удовлетворяющих условию
percentage = len(indices) / len(X_train) * 100
print(f"Процент наблюдений: {percentage:.1f}")
```
4. Сделайте предсказание и рассчитайте значение метрики accuracy на тестовой выборке.
```
y_pred_1 = dt_1.predict(X_test)
accuracy_1 = metrics.accuracy_score(y_test, y_pred_1)
print(f"Accuracy на тестовой выборке: {accuracy_1:.3f}")
```
## Задание 2. Увеличим глубину дерева

### Создание модели дерева решений глубиной 2
```
dt_2 = tree.DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=42)
dt_2.fit(X_train, y_train)
```
### Визуализация дерева
```
plt.figure(figsize=(10, 6))
tree.plot_tree(dt_2, feature_names=X.columns, class_names=['female', 'male'], filled=True)
plt.show()
```
### Вопросы по заданию 2
1. Из приведённых ниже факторов выберите те, что используются при построении данного дерева решений:
Ответ: A, B, D

2. Сколько листьев в построенном дереве содержат в качестве предсказания класс female? Нужно проанализировать визуализацию дерева.
Ответ: 2

3. Сделайте предсказание и рассчитайте значение метрики accuracy на тестовой выборке.
```
y_pred_2 = dt_2.predict(X_test)
accuracy_2 = metrics.accuracy_score(y_test, y_pred_2)
print(f"Accuracy на тестовой выборке: {accuracy_2:.3f}")
```

## Задание 3. Дадим дереву решений б’ольшую свободу

### Создание модели дерева решений без ограничения глубины
```
dt_inf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_inf.fit(X_train, y_train)
```

### 1. Чему равна глубина полученного дерева решения?
```
depth_inf = dt_inf.get_depth()
print(f"Глубина дерева: {depth_inf}")
```
### 2. Чему равно количество листьев в полученном дереве решений?
```
n_leaves_inf = dt_inf.get_n_leaves()
print(f"Количество листьев: {n_leaves_inf}")
```
### 3. Сделайте предсказание для обучающей и тестовой выборок и рассчитайте значение метрики accuracy на каждой из выборок
```
y_pred_train_inf = dt_inf.predict(X_train)
accuracy_train_inf = metrics.accuracy_score(y_train, y_pred_train_inf)
print(f"Accuracy на обучающей выборке: {accuracy_train_inf:.3f}")

y_pred_test_inf = dt_inf.predict(X_test)
accuracy_test_inf = metrics.accuracy_score(y_test, y_pred_test_inf)
print(f"Accuracy на тестовой выборке: {accuracy_test_inf:.3f}")
```
## Задание 4. Попробуем найти оптимальные внешние параметры модели дерева решений

### Задание сетки параметров
```
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [3, 4, 5, 10]
}
```
### Задаём метод кросс-валидации
```cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Добавлено перемешивание```

### Поиск оптимальных параметров с помощью GridSearchCV
```
grid_search = model_selection.GridSearchCV(tree.DecisionTreeClassifier(random_state=0), param_grid, scoring='accuracy', cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)
```
#### 1. Какой критерий информативности использует наилучшая модель?
```
best_criterion = grid_search.best_params_['criterion']
print(f"Лучший критерий информативности: {best_criterion}")
```
#### 2. Чему равна оптимальная найденная автоматически (с помощью GridSearchCV) максимальная глубина?
```
best_max_depth = grid_search.best_params_['max_depth']
print(f"Оптимальная максимальная глубина: {best_max_depth}")
```
#### 3. Чему равно оптимальное минимальное количество объектов, необходимое для разбиения?
```
best_min_samples_split = grid_search.best_params_['min_samples_split']
print(f"Оптимальное минимальное количество объектов для разбиения: {best_min_samples_split}")
```
#### 4. С помощью наилучшей модели сделайте предсказание отдельно для обучающей и тестовой выборок. Рассчитайте значение метрики accuracy на каждой из выборок.
```
best_model = grid_search.best_estimator_
y_pred_train_best = best_model.predict(X_train)
accuracy_train_best = metrics.accuracy_score(y_train, y_pred_train_best)
print(f"Accuracy на обучающей выборке для лучшей модели: {accuracy_train_best:.3f}")
y_pred_test_best = best_model.predict(X_test)
accuracy_test_best = metrics.accuracy_score(y_test, y_pred_test_best)
print(f"Accuracy на тестовой выборке для лучшей модели: {accuracy_test_best:.3f}")
```
## Задание 5. Для оптимального дерева решений, построенного в задании 4, найдите важность каждого из факторов

### Получение важности признаков
```feature_importances = best_model.feature_importances_```

### Визуализация важности признаков
```
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=X.columns)
plt.title('Важность признаков')
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.show()
```
### Выделение топ-3 наиболее важных факторов
```
importances_sorted = sorted(zip(feature_importances, X.columns), reverse=True)
top_3_features = [feature for importance, feature in importances_sorted[:3]]
print(f"Топ-3 наиболее важных факторов: {top_3_features}")
```

Объяснение решения:

1. Импорт библиотек: Импортированы необходимые библиотеки для работы с данными, визуализации, построения моделей и оценки их качества.
2. Загрузка и подготовка данных:
        Данные загружены из CSV файла.
        Целевая переменная ('label') отделена от признаков.
        Целевая переменная закодирована в числовой формат с помощью LabelEncoder, так как sklearn требует числовые значения для классификации.
        Данные разделены на обучающую и тестовую выборки.

3. Задание 1: Решающие пни
        Создана модель дерева решений с максимальной глубиной 1 и критерием энтропии.
        Модель обучена на обучающей выборке.
        Дерево визуализировано.
        Ответы на вопросы задания:
                Определен фактор, используемый в корневой вершине.
                Получено оптимальное пороговое значение.
                Рассчитан процент наблюдений, удовлетворяющих условию в корневой вершине.
                Рассчитана accuracy на тестовой выборке.

4. Задание 2: Увеличение глубины дерева
        Создана модель дерева решений с максимальной глубиной 2.
        Модель обучена и визуализирована.
        Ответы на вопросы задания:
                Определены используемые факторы.
                Определено количество листьев с предсказанием "female".
                Рассчитана accuracy на тестовой выборке.

5. Задание 3: Дадим дереву решений б’ольшую свободу
        Создана модель дерева решений без ограничения глубины.
        Модель обучена.
        Получены глубина и количество листьев дерева.
        Рассчитана accuracy на обучающей и тестовой выборках.
6. Задание 4: Поиск оптимальных гиперпараметров с GridSearchCV
        Определена сетка гиперпараметров для перебора.
        Создан StratifiedKFold для кросс-валидации (добавлено перемешивание данных).
        Использован GridSearchCV для поиска наилучшей модели.
        Выведены лучшие параметры.
        Рассчитана accuracy на обучающей и тестовой выборках для лучшей модели.

7. Задание 5: Анализ важности признаков
        Получена важность признаков из лучшей модели.
        Важность признаков визуализирована в виде столбчатой диаграммы.
        Выделены топ-3 наиболее важных фактора.

### Важные замечания:

1. Нужно убедитсья, что файл voice_gender.csv находится в той же директории, где находится скрипт Python.
2. Установка random_state обеспечивает воспроизводимость результатов.
3. В задании 4 добавлено перемешивание данных в StratifiedKFold для более надежной оценки.
4. В ответах на вопросы задания добавлены пояснения и расчеты, где это необходимо.
