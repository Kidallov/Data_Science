import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
from datetime import timedelta

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

CSV_PATH = 'tovar_moving.csv'

try:
    df = pd.read_csv(CSV_PATH, parse_dates=['date'], index_col='date')
except FileNotFoundError:
    print(f"Ошибка: файл '{CSV_PATH}' не найден.")
    sys.exit(1)
except Exception as e:
    print(f"Ошибка при чтении CSV: {e}")
    sys.exit(1)

df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
df.dropna(subset=['qty'], inplace=True)
df = df.sort_index().asfreq('D')
df['qty'] = df['qty'].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')

if len(df) < 10:
    print("Внимание: слишком короткий ряд, прогнозы могут быть ненадёжны.")

test_value = df['qty'].iloc[-1]
df_train = df.iloc[:-1].copy()

df_train['log_qty'] = np.log1p(df_train['qty'])
test_log_value = np.log1p(test_value)

print(f"Логарифмическая трансформация применена. Пример: qty={df_train['qty'].iloc[-1]:.2f} -> log_qty={df_train['log_qty'].iloc[-1]:.2f}")

SEASONAL_PERIOD = 7

try:
    decomposition = seasonal_decompose(df_train['qty'], model='additive', period=SEASONAL_PERIOD)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.suptitle('Разложение временного ряда (qty)', fontsize=16)
    plt.show()
except Exception:
    print("Не удалось выполнить seasonal_decompose (недостаточно данных).")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plot_acf(df_train['log_qty'], lags=30, ax=plt.gca(), zero=False)
plt.title('ACF (log_qty)')
plt.subplot(1, 2, 2)
plot_pacf(df_train['log_qty'], lags=30, ax=plt.gca(), zero=False, method='ywm')
plt.title('PACF (log_qty)')
plt.tight_layout()
plt.show()

arima_order, sarima_order, seasonal_order = None, None, None

if PMDARIMA_AVAILABLE:
    print("Используется auto_arima для подбора параметров...")
    try:
        stepwise = auto_arima(df_train['log_qty'], start_p=0, start_q=0, max_p=5, max_q=5,
                              seasonal=True, m=SEASONAL_PERIOD, trace=False, stepwise=True)
        arima_order = stepwise.order
        seasonal_order = stepwise.seasonal_order
        print(f"Подобраны параметры: ARIMA{arima_order}x{seasonal_order}")
    except Exception as e:
        print(f"auto_arima не удалось: {e}")
else:
    print("pmdarima не установлена, используются дефолтные параметры.")
    arima_order = (1, 1, 0)
    seasonal_order = (0, 1, 1, SEASONAL_PERIOD)

sarima_order = arima_order  # для согласованности

def fit_forecast_hw(series_log):
    """Holt–Winters, прогноз на лог-шкале"""
    try:
        model = ExponentialSmoothing(series_log, trend='add', seasonal='add', seasonal_periods=SEASONAL_PERIOD)
        fitted = model.fit(optimized=True)
        return fitted, fitted.forecast(1).iloc[0]
    except Exception as e:
        print(f"Holt-Winters ошибка: {e}")
        return None, np.nan

def fit_forecast_arima(series_log, order, seasonal_order):
    """SARIMA на лог-шкале"""
    try:
        model = ARIMA(series_log, order=order, seasonal_order=seasonal_order)
        fitted = model.fit()
        return fitted, fitted.forecast(1).iloc[0]
    except Exception as e:
        print(f"ARIMA ошибка: {e}")
        return None, np.nan

fitted_hw, pred_log_hw = fit_forecast_hw(df_train['log_qty'])
fitted_arima, pred_log_arima = fit_forecast_arima(df_train['log_qty'], arima_order, seasonal_order)

pred_hw = np.expm1(pred_log_hw)
pred_arima = np.expm1(pred_log_arima)

def safe_rmse(true, pred):
    return np.sqrt(mean_squared_error([true], [pred])) if not np.isnan(pred) else np.nan

results = []
for name, pred, fitted, pred_log in [
    ('HoltWinters', pred_hw, fitted_hw, pred_log_hw),
    (f'SARIMA{arima_order}x{seasonal_order}', pred_arima, fitted_arima, pred_log_arima),
]:
    rmse = safe_rmse(test_value, pred)
    mae = mean_absolute_error([test_value], [pred]) if not np.isnan(pred) else np.nan
    results.append({
        'model': name,
        'forecast_qty': pred,
        'true_qty': test_value,
        'RMSE_onepoint': rmse,
        'MAE_onepoint': mae
    })

print("\n=== Прогноз на одно отложенное наблюдение ===")
for r in results:
    print(f"{r['model']}: прогноз={r['forecast_qty']:.2f}, RMSE={r['RMSE_onepoint']:.2f}")

def plot_residuals(fitted, model_name):
    """Рисуем остатки и их ACF, QQ plot"""
    if fitted is None:
        return
    residuals = fitted.resid.dropna()
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.plot(residuals)
    plt.title(f'Остатки ({model_name})')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plot_acf(residuals, lags=30, ax=plt.gca(), zero=False)
    plt.title('ACF остатков')

    plt.subplot(1, 3, 3)
    qqplot(residuals, line='s', ax=plt.gca())
    plt.title('QQ Plot остатков')
    plt.tight_layout()
    plt.show()

print("\n=== Графики остатков ===")
plot_residuals(fitted_hw, 'Holt-Winters')
plot_residuals(fitted_arima, f'SARIMA{arima_order}x{seasonal_order}')

results_df = pd.DataFrame(results)
csv_path = 'forecast_results.csv'
results_df.to_csv(csv_path, index=False)
print(f"\nРезультаты сохранены в файл: {csv_path}")
print(results_df)

LOOKBACK = 60
plt.figure(figsize=(12, 6))
plt.plot(df['qty'].iloc[-LOOKBACK:], label='Фактический qty')
plt.scatter(df.index[-1], test_value, color='black', label='Тест')
plt.scatter(df.index[-1] + timedelta(days=1), pred_hw, color='orange', label='Holt-Winters прогноз')
plt.scatter(df.index[-1] + timedelta(days=1), pred_arima, color='green', label='SARIMA прогноз')
plt.title('Прогноз моделей (лог-трансформация + обратное преобразование)')
plt.xlabel('Дата')
plt.ylabel('qty')
plt.legend()
plt.grid(True)
plt.show()

print("\nСкрипт успешно завершён ✅")
