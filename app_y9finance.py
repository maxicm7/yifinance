import os
import re
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import normaltest, norm # <--- Agregado norm para VaR
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.api import VAR
import yfinance as yf
import warnings
import traceback # For detailed error printing
import json
from huggingface_hub import InferenceClient
import pypdf
import io
import random
import time # Agregado para sleep/rerun



# --- Dependencia para IOL ---
import requests # Para hacer las llamadas a la API de IOL

# --- Dependencias Adicionales ---
try:
    from prophet import Prophet
    prophet_installed = True
except ImportError:
    prophet_installed = False

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
    pypfopt_installed = True
except ImportError:
    pypfopt_installed = False
    

try:
    from tbats import TBATS
    tbats_installed = True
except ImportError:
    tbats_installed = False

# --- NUEVAS DEPENDENCIAS ML ---
try:
    import xgboost as xgb
    xgb_installed = True
except ImportError:
    xgb_installed = False

try:
    import lightgbm as lgb
    lgbm_installed = True
except ImportError:
    lgbm_installed = False

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Stock Analysis & Portfolio Tool")

# --- Constante para el archivo de portafolios ---
PORTFOLIO_FILE = "portfolios_data1.json"

# --- Constantes para IOL API ---
IOL_API_BASE_URL = "https://api.invertironline.com"

# --- NUEVO: Dependencias para Hugging Face y Chat ---
from huggingface_hub import InferenceClient
import pypdf
import io

# --- NUEVO: Funciones para la Integración con Hugging Face ---

def get_hf_response(api_key, model, prompt, temperature=0.7, max_tokens=2048):
    """
    Función universal para llamar a la API de Hugging Face.
    """
    if not api_key or not api_key.startswith("hf_"):
        st.error("Por favor, introduce una Hugging Face API Key válida en la barra lateral.")
        return None
    try:
        client = InferenceClient(token=api_key)
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False, # Importante para un retorno simple
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al contactar la API de Hugging Face.")
        st.info(
            "Esto puede ocurrir por varias razones:\n"
            "1. No tienes acceso a este modelo (visita su página en HF para solicitarlo).\n"
            "2. El modelo está tardando en cargar en los servidores de HF. Espera un minuto y vuelve a intentarlo.\n"
            "3. La API Key es incorrecta o no tiene los permisos necesarios."
        )
        print(f"Detalle del error de HF: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    """
    Lee un archivo PDF subido a través de Streamlit y extrae su texto.
    """
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error al leer el archivo PDF: {e}")
        return ""

# --- FIN de las nuevas funciones de Hugging Face ---

# --- Helper Functions (Forecasting App - Modificado) ---
def clean_file_name(name):
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name

@st.cache_data(show_spinner="Fetching forecasting data from Yahoo Finance...")
def load_stock_data(tickers, start_date, end_date, calculate_returns=True):
    if not tickers:
        st.error("Please enter at least one stock ticker for forecasting analysis.")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if raw_data.empty:
            st.error(f"No data found for the selected tickers and date range. Please check ticker symbols and dates.")
            return None
        price_col_to_use = None
        data_selection = None
        if isinstance(raw_data.columns, pd.MultiIndex):
            available_top_levels = raw_data.columns.get_level_values(0).unique()
            if 'Adj Close' in available_top_levels: price_col_to_use = 'Adj Close'
            elif 'Close' in available_top_levels: price_col_to_use = 'Close'; st.warning("Could not find 'Adj Close'. Using 'Close' price instead for forecasting data.")
            else: st.error(f"Could not find 'Adj Close' or 'Close' columns in downloaded data for forecasting. Available columns: {available_top_levels}"); return None
            data_selection = raw_data[price_col_to_use]
        else: # Single ticker likely
            if 'Adj Close' in raw_data.columns: price_col_to_use = 'Adj Close'
            elif 'Close' in raw_data.columns: price_col_to_use = 'Close'; st.warning("Could not find 'Adj Close'. Using 'Close' price instead for forecasting data.")
            else: st.error(f"Could not find 'Adj Close' or 'Close' columns in downloaded data for forecasting. Available columns: {raw_data.columns}"); return None
            data_selection = raw_data[[price_col_to_use]]
            if len(tickers) == 1:
                 if tickers[0] not in data_selection.columns:
                     data_selection.columns = tickers
        data = data_selection
        missing_tickers = data.columns[data.isnull().all()].tolist()
        if missing_tickers:
            st.warning(f"Forecasting: Could not retrieve price data ({price_col_to_use}) for: {', '.join(missing_tickers)}. These tickers will be excluded.")
            data = data.drop(columns=missing_tickers)
            # active_tickers = data.columns.tolist() # No se usa explicitamente después
            if data.empty: st.error(f"Forecasting: No valid price data remaining after excluding missing tickers using column '{price_col_to_use}'."); return None
        # else: active_tickers = data.columns.tolist() # No se usa explicitamente después
        if data.isnull().values.any(): data = data.ffill().bfill()
        data = data.dropna()
        if data.empty: st.error(f"Forecasting: Dataframe became empty after handling missing values for column '{price_col_to_use}'. Check data quality."); return None
        data_long = data.stack().reset_index()
        data_long.columns = ['Date', 'Entity', 'Value'] if len(data.columns)>1 else ['Date','Value']
        if len(data.columns)==1: data_long['Entity'] = data.columns[0]
        data_long['Date'] = pd.to_datetime(data_long['Date'])
        data_long = data_long.set_index('Date').sort_index()
        if calculate_returns:
            data_long['Actual'] = data_long.groupby('Entity')['Value'].pct_change() * 100
            data_long.replace([np.inf, -np.inf], np.nan, inplace=True)
            data_long = data_long.dropna(subset=['Actual'])
            data_long = data_long[['Entity', 'Actual']]
            if data_long.empty: st.error("Forecasting: Dataframe empty after calculating returns."); return None
            # processed_tickers = data_long['Entity'].unique() # No se usa explicitamente después
        else:
            data_long['Actual'] = data_long['Value']
            data_long = data_long[['Entity', 'Actual']]
            # processed_tickers = data_long['Entity'].unique() # No se usa explicitamente después
        return data_long
    except KeyError as e:
         st.error(f"A KeyError occurred during forecasting data load: {e}. Check ticker symbols and column '{price_col_to_use}'.")
         st.error(traceback.format_exc())
         return None
    except Exception as e:
        st.error(f"An unexpected error during forecasting data fetching/processing: {e}")
        st.error(traceback.format_exc())
        return None

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > 1e-8
    if np.sum(mask) == 0: return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def treat_outliers_iqr(series: pd.Series, k: float = 1.5) -> tuple[pd.Series, int]:
    if series.empty or series.isnull().all():
        return series, 0
    q1 = series.quantile(0.25); q3 = series.quantile(0.75); iqr = q3 - q1
    if iqr == 0: return series, 0
    lower_bound = q1 - k * iqr; upper_bound = q3 + k * iqr
    original_series = series.copy()
    treated_series = series.clip(lower=lower_bound, upper=upper_bound)
    num_outliers_treated = (original_series != treated_series).sum()
    return treated_series, num_outliers_treated

# --- NUEVA FUNCIÓN DE FEATURE ENGINEERING PARA MODELOS ML (XGBoost/LightGBM) ---
def create_ts_features(df, lag_max=5):
    """
    Genera características de tiempo y rezagos para modelos basados en ML.
    df debe contener la columna 'Actual' y, opcionalmente, la columna exógena.
    """
    df = df.copy()
    
    # 1. Características de fecha (deben estar presentes antes de cualquier dropna)
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    # 2. Características de rezago (lags)
    for lag in range(1, lag_max + 1):
        # Siempre rezagamos la variable objetivo ('Actual')
        df[f'lag_{lag}'] = df['Actual'].shift(lag)
    
    # 3. Características de ventana móvil (Ej: media/std de retornos recientes)
    # NOTA: Usamos shift(1) para asegurarnos de que estas features solo usen datos conocidos ANTES del día actual.
    df['rolling_mean_3'] = df['Actual'].shift(1).rolling(window=3).mean()
    df['rolling_std_5'] = df['Actual'].shift(1).rolling(window=5).std()
    
    # Después de generar features, elimina las filas con NaN (las primeras filas por los lags/rollings)
    df = df.dropna() 
    return df

# --- Helper Functions (Portfolio App - Fetch sin cambios) ---
@st.cache_data(show_spinner="Fetching historical portfolio price data...")
def fetch_stock_prices_for_portfolio(tickers, start_date, end_date):
    if not tickers:
        st.error("Portfolio: Ticker list is empty.")
        return pd.DataFrame()
    price_data = None
    price_col_to_use = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if raw_data.empty:
            st.warning(f"Portfolio Fetch: Initial download returned no data for tickers: {', '.join(tickers)} in the specified date range. Check symbols/dates.")
            return pd.DataFrame()
        if isinstance(raw_data.columns, pd.MultiIndex):
            available_top_levels = raw_data.columns.get_level_values(0).unique()
            if 'Adj Close' in available_top_levels: price_col_to_use = 'Adj Close'; price_data = raw_data['Adj Close']; st.caption(f"Portfolio Fetch: Using '{price_col_to_use}' data.")
            elif 'Close' in available_top_levels: price_col_to_use = 'Close'; price_data = raw_data['Close']; st.warning(f"Portfolio Fetch: Using '{price_col_to_use}' price as 'Adj Close' not found.")
            else: st.error(f"Portfolio Fetch: Critical - Neither 'Adj Close' nor 'Close' found. Available fields: {available_top_levels}"); return pd.DataFrame()
        elif isinstance(raw_data, pd.DataFrame):
             if 'Adj Close' in raw_data.columns:
                 price_col_to_use = 'Adj Close'
                 price_data = raw_data[[price_col_to_use] + [col for col in raw_data.columns if col != price_col_to_use]]
                 price_data = price_data.loc[:,~price_data.columns.duplicated()]
                 st.caption(f"Portfolio Fetch: Using '{price_col_to_use}' data.")
             elif 'Close' in raw_data.columns:
                 price_col_to_use = 'Close'
                 price_data = raw_data[['Close'] + [col for col in raw_data.columns if col != 'Close']]
                 price_data = price_data.loc[:,~price_data.columns.duplicated()]
                 st.warning(f"Portfolio Fetch: Using '{price_col_to_use}' price as 'Adj Close' not found.")
             else: st.error(f"Portfolio Fetch: Critical - Neither 'Adj Close' nor 'Close' found. Available columns: {raw_data.columns}"); return pd.DataFrame()
             if len(tickers) == 1 and price_data.shape[1] >= 1:
                 if tickers[0] not in price_data.columns:
                      if price_col_to_use in price_data.columns:
                         price_data = price_data.rename(columns={price_col_to_use: tickers[0]})
                         price_data = price_data[[tickers[0]]]
                      else:
                         price_data = price_data.rename(columns={price_data.columns[0]: tickers[0]})
                         price_data = price_data[[tickers[0]]]
        elif isinstance(raw_data, pd.Series) and len(tickers) == 1:
             series_name = raw_data.name if raw_data.name else ""
             if 'Adj Close' in series_name: price_col_to_use = 'Adj Close'
             elif 'Close' in series_name: price_col_to_use = 'Close'
             else: price_col_to_use = 'Adj Close'; st.warning(f"Portfolio Fetch: Assuming downloaded series is '{price_col_to_use}' for {tickers[0]}.")
             price_data = raw_data.to_frame(name=tickers[0])
             st.caption(f"Portfolio Fetch: Using '{price_col_to_use}' data (from Series).")
        else: st.error(f"Portfolio Fetch: Unexpected data structure from yfinance for {tickers}. Type: {type(raw_data)}"); return pd.DataFrame()
        if price_data is None or price_data.empty: st.warning(f"Portfolio Fetch: No price data extracted after initial download (tried '{price_col_to_use}')."); return pd.DataFrame()
        valid_requested_tickers = [t for t in tickers if t in price_data.columns]
        price_data = price_data[valid_requested_tickers]
        missing_cols = price_data.columns[price_data.isnull().all()].tolist()
        if missing_cols:
            st.warning(f"Portfolio Fetch: No valid '{price_col_to_use}' data found for: {', '.join(missing_cols)} in the period. Excluding them.")
            price_data = price_data.drop(columns=missing_cols)
            if price_data.empty: st.error("Portfolio Fetch: No valid price data remaining after excluding tickers with missing data."); return pd.DataFrame()
        price_data = price_data.ffill().bfill()
        price_data = price_data.dropna()
        if price_data.empty: st.warning("Portfolio Fetch: Price data became empty after handling missing values (NaNs). Check date range or ticker data quality."); return pd.DataFrame()
        st.caption(f"Portfolio Fetch: Successfully processed '{price_col_to_use}' data for: {', '.join(price_data.columns)}")
        return price_data
    except KeyError as e:
        st.error(f"Portfolio Fetch: A KeyError occurred: {e}. Ticker invalid or yfinance structure changed?")
        st.info(f"Tickers attempted: {tickers}. Column trying to access: '{price_col_to_use}' or related.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Portfolio Fetch: An unexpected error occurred: {e}")
        st.error(traceback.format_exc())
        return pd.DataFrame()

def calculate_portfolio_performance(prices_df, weights_dict):
    if prices_df is None or prices_df.empty: return None, None, ""
    if not weights_dict: st.error("Portfolio Calc: Weights dictionary is empty."); return None, None, ""
    valid_tickers = [ticker for ticker in weights_dict.keys() if ticker in prices_df.columns]
    if not valid_tickers:
        st.error(f"Portfolio Calc: None of the specified portfolio tickers ({', '.join(weights_dict.keys())}) have valid price data.")
        return None, None, ""
    prices_filtered = prices_df[valid_tickers]
    weights_array = np.array([weights_dict[ticker] for ticker in valid_tickers])
    weights_normalized = weights_array
    renormalized_info = ""
    original_tickers = set(weights_dict.keys())
    available_tickers = set(valid_tickers)
    if original_tickers != available_tickers:
        dropped_tickers = list(original_tickers - available_tickers)
        if weights_array.sum() > 1e-6:
             weights_normalized = weights_array / weights_array.sum()
             renormalized_weights_disp = {ticker: f"{weight*100:.2f}%" for ticker, weight in zip(valid_tickers, weights_normalized)}
             renormalized_info = (f"**Note:** Tickers `{dropped_tickers}` had no valid price data and were excluded. "
                                  f"Remaining weights renormalized. Using: `{renormalized_weights_disp}`")
        else:
             st.error(f"Portfolio Calc: All tickers ({dropped_tickers}) excluded due to missing data. Cannot calculate.")
             return None, None, f"**Note:** All specified tickers `{dropped_tickers}` had no valid price data and were excluded."
    try:
        daily_returns = prices_filtered.pct_change().dropna()
        if daily_returns.empty: st.warning("Portfolio Calc: Not enough data points to calculate daily returns."); return None, None, renormalized_info
        portfolio_daily_returns = daily_returns.dot(weights_normalized)
        portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod()
        if not daily_returns.empty:
            first_return_date = daily_returns.index[0]
            original_index_subset = prices_df.index[prices_df.index < first_return_date]
            start_idx = original_index_subset[-1] if not original_index_subset.empty else first_return_date - pd.Timedelta(days=1)
            initial_point = pd.Series([1.0], index=[start_idx])
            portfolio_cumulative_returns = pd.concat([initial_point, portfolio_cumulative_returns]).sort_index()
        return portfolio_daily_returns, portfolio_cumulative_returns, renormalized_info
    except Exception as e:
        st.error(f"Error calculating portfolio performance: {e}")
        st.error(traceback.format_exc())
        return None, None, renormalized_info

# --- NUEVAS FUNCIONES PARA GESTIÓN DE RIESGO ---
def calculate_var(returns, confidence_level=0.95):
    """Calculates Value at Risk (VaR) using Historical and Parametric methods."""
    if returns is None or returns.empty or len(returns) < 10: # Require a few points
        return None, None
    
    # Historical VaR: Returns the loss as a positive fractional number
    hist_var = -1 * returns.quantile(1 - confidence_level)

    # Parametric VaR (Variance-Covariance): Assumes a normal distribution of returns.
    mean = returns.mean()
    std_dev = returns.std()
    # Z-score for the left tail. e.g., for 95% confidence, we want the 5th percentile, ppf(0.05) = -1.645
    z_score_ppf = norm.ppf(1 - confidence_level)
    # The return at that percentile is mu + z*sigma. The loss is its negative.
    para_var = -(mean + z_score_ppf * std_dev)

    return hist_var, para_var

def calculate_drawdowns(cumulative_returns_series):
    """Calculates the drawdown series and max drawdown."""
    if cumulative_returns_series is None or cumulative_returns_series.empty:
        return None, 0, None
    
    # 1. Calculate the running maximum (previous high water mark)
    running_max = cumulative_returns_series.cummax()
    
    # 2. Calculate the drawdown as the percentage drop from the running max
    drawdown = (cumulative_returns_series - running_max) / running_max
    
    # 3. Calculate max drawdown
    max_drawdown = drawdown.min()
    
    # 4. Find the date of the max drawdown
    max_drawdown_date = drawdown.idxmin() if pd.notna(max_drawdown) and not drawdown.empty else None
    
    # CORRECCIÓN: Usar max_drawdown_date en lugar de max_dd_date
    return drawdown, max_drawdown, max_drawdown_date


# --- FUNCIONES PARA CARGAR/GUARDAR PORTAFOLIOS (Sin cambios) ---
def load_portfolios_from_file():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                portfolios = json.load(f)
            # st.sidebar.caption(f"Portfolios cargados desde {PORTFOLIO_FILE}") # Puede ser ruidoso
            return portfolios
        except json.JSONDecodeError:
            st.sidebar.error(f"Error al leer {PORTFOLIO_FILE}. Archivo corrupto o mal formateado. Se iniciará con portafolios vacíos.")
            return {}
        except Exception as e:
            st.sidebar.error(f"Error inesperado al cargar {PORTFOLIO_FILE}: {e}. Se iniciará con portafolios vacíos.")
            return {}
    return {}

# Reemplaza tu función original con esta
def save_portfolios_to_file(portfolios_dict):
    """Guarda el diccionario de portafolios en el archivo JSON.
    
    Returns:
        tuple[bool, str]: Un tuple con (True, "") en caso de éxito, 
                          o (False, "mensaje de error") en caso de fallo.
    """
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolios_dict, f, indent=4)
        return True, "" # Éxito
    except Exception as e:
        error_message = f"Error al guardar portafolios en {PORTFOLIO_FILE}: {e}"
        # También imprime el error en la terminal para depuración
        print(error_message) 
        traceback.print_exc()
        return False, error_message # Fallo


# --- Funciones para InvertirOnline API ---
def login_iol(username, password):
    url = f"{IOL_API_BASE_URL}/token"
    payload = {
        "username": username,
        "password": password,
        "grant_type": "password"
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    try:
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        expires_in = token_data.get('expires_in', 3600)
        token_data['expires_at'] = datetime.now() + timedelta(seconds=expires_in)
        token_data['issued_at'] = datetime.now()
        return token_data, None
    except requests.exceptions.HTTPError as errh:
        error_detail = f"Http Error: {errh}"
        try: error_detail += f" - Response: {response.json()}"
        except json.JSONDecodeError: error_detail += f" - Response: {response.text}"
        return None, error_detail
    except requests.exceptions.RequestException as err:
        return None, f"Request Exception: {err}"
    except json.JSONDecodeError:
        return None, f"Error al decodificar JSON de IOL. Respuesta: {response.text if 'response' in locals() else 'No response object'}"

def get_iol_data(endpoint_path, access_token, params=None, method="GET", data=None):
    if not access_token:
        return None, "No hay token de acceso válido."
    url = f"{IOL_API_BASE_URL}{endpoint_path}"
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=15)
        elif method.upper() == "POST":
            headers["Content-Type"] = "application/json"
            response = requests.post(url, headers=headers, json=data, params=params, timeout=15)
        else:
            return None, f"Método HTTP no soportado: {method}"
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as errh:
        error_detail = f"Http Error: {errh}"
        try: error_detail += f" - Response: {response.json()}"
        except json.JSONDecodeError: error_detail += f" - Response: {response.text}"
        return None, error_detail
    except requests.exceptions.RequestException as err:
        return None, f"Request Exception: {err}"
    except json.JSONDecodeError:
        return None, f"Error al decodificar JSON de IOL. Respuesta: {response.text if 'response' in locals() else 'No response object'}"


def get_iol_token_status():
    if not st.session_state.get('iol_logged_in', False) or not st.session_state.get('iol_token_data'):
        return "No autenticado en IOL."
    token_data = st.session_state.iol_token_data
    expires_at = token_data.get('expires_at')
    now = datetime.now()
    if expires_at and now < expires_at:
        remaining_time = expires_at - now
        return f"Autenticado como {st.session_state.iol_user}. Token expira en: {str(remaining_time).split('.')[0]}"
    else:
        st.session_state.iol_logged_in = False
        return "Token de IOL expirado. Por favor, vuelva a iniciar sesión."

# --- NUEVA FUNCIÓN PARA PARSEAR TICKERS ---
def parse_tickers_from_text(text_data):
    tickers_by_sector = {}
    current_sector = "Sin Sector"
    all_tickers_info = []
    ticker_regex = re.compile(r"^(.*?)\s*\(([A-Z0-9]{2,6})\)$")

    for line in text_data.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">") and ":" in line:
            current_sector = line.split(":")[0].replace(">", "").strip()
            if current_sector not in tickers_by_sector:
                tickers_by_sector[current_sector] = []
            continue
        match = ticker_regex.search(line)
        if match:
            company_name = match.group(1).strip()
            ticker = match.group(2).strip()
            if ticker:
                tickers_by_sector.setdefault(current_sector, []).append({
                    "ticker": ticker,
                    "nombre": company_name,
                })
                all_tickers_info.append({
                    "ticker": ticker,
                    "nombre": company_name,
                    "sector": current_sector
                })
    return all_tickers_info

# --- Page Definition Functions (Forecasting App - Modificado) ---
def main_page():
    st.header("Welcome!")
    st.markdown("""
    This application provides tools for **portfolio management**, **risk management**, **individual stock forecasting**, **conceptual event analysis**, and **InvertirOnline API interaction**.
    (Descripción de workflows sin cambios, solo se actualiza la mención a la API de IOL)
    """)
# ... (resto de tus funciones de página de forecasting sin cambios: data_visualization, decomposition, optimal_lags) ...
def data_visualization():
    st.header("Data Visualization (Single Stock)")
    if 'entities' not in st.session_state or not st.session_state.entities:
        st.warning("No single stock data loaded for forecasting/visualization yet. Please select tickers/dates in the sidebar and click 'Load & Process Data (for Forecasting)'.")
        st.info("This page visualizes the data loaded via the sidebar for the forecasting workflow.")
        return
    data_type = st.session_state.get('data_type', 'returns')
    outlier_status = " (Outliers Treated)" if st.session_state.get('apply_outlier_treatment', False) else ""
    st.info(f"Displaying: Daily {data_type.capitalize()}{outlier_status} for selected stock (from sidebar loading).")

    selected_entity_viz = st.selectbox("Select a ticker to visualize", st.session_state.entities, key="viz_entity")
    if selected_entity_viz and selected_entity_viz in st.session_state.dataframes:
        df_viz = st.session_state.dataframes[selected_entity_viz]
        st.subheader(f"Time Series Plot for: {selected_entity_viz}")
        if df_viz.empty or df_viz['Actual'].isnull().all():
             st.warning(f"No valid data to plot for {selected_entity_viz}.")
        else:
             st.line_chart(df_viz['Actual'])
             with st.expander("View Processed Data (Returns/Prices)"):
                 st.dataframe(df_viz)
    else: st.warning("Please select a valid ticker.")

def decomposition():
    st.header("Series Decomposition (Single Stock)")
    if 'entities' not in st.session_state or not st.session_state.entities:
        st.warning("No single stock data loaded for decomposition. Please load data via the sidebar first using 'Load & Process Data (for Forecasting)'.")
        st.info("This page decomposes the data loaded via the sidebar for the forecasting workflow.")
        return
    data_type = st.session_state.get('data_type', 'returns')
    outlier_status = " (Outliers Treated)" if st.session_state.get('apply_outlier_treatment', False) else ""
    st.info(f"Decomposing: Daily {data_type.capitalize()}{outlier_status} for selected stock (from sidebar loading).")

    selected_entity_decomp = st.selectbox("Select a ticker", st.session_state.entities, key="decomp_entity")

    default_period = 5
    # min_data_for_period = 2 * default_period + 1 # No se usa explicitamente

    period = st.number_input("Seasonal Period (days)", min_value=2, value=default_period, step=1, key="decomp_period",
                             help=f"Typical periods for daily data: 5 (trading week), 7 (full week). Requires at least {2*default_period} data points.")

    if selected_entity_decomp and selected_entity_decomp in st.session_state.dataframes:
        df_decomp_full = st.session_state.dataframes[selected_entity_decomp]
        if df_decomp_full.empty: st.warning(f"No data available for ticker {selected_entity_decomp}."); return
        df_decomp = df_decomp_full['Actual'].dropna()
        if df_decomp.empty: st.warning(f"No valid (non-NaN) data for ticker {selected_entity_decomp}."); return

        if len(df_decomp) < 2 * period:
            st.warning(f"Not enough data ({len(df_decomp)} points) for {selected_entity_decomp} for decomposition period={period}. Requires {2 * period} points."); return

        decomposition_type = st.radio("Decomposition Model", ("additive", "multiplicative"), 0, key="decomp_type",
                                      help="Multiplicative requires positive values (may fail for returns or capped prices).")

        st.subheader(f"Seasonal Decomposition for {selected_entity_decomp} ({decomposition_type.capitalize()})")
        try:
            decomp_model_to_use = decomposition_type
            if decomp_model_to_use == 'multiplicative' and (df_decomp <= 1e-8).any():
                 st.warning("Multiplicative requires strictly positive values. Switching to additive.")
                 decomp_model_to_use = 'additive'

            decomposition_result = seasonal_decompose(df_decomp, model=decomp_model_to_use, period=period)
            fig_decomp, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            decomposition_result.observed.plot(ax=axes[0], legend=False); axes[0].set_ylabel("Observed")
            decomposition_result.trend.plot(ax=axes[1], legend=False); axes[1].set_ylabel("Trend")
            decomposition_result.seasonal.plot(ax=axes[2], legend=False); axes[2].set_ylabel("Seasonal")
            decomposition_result.resid.plot(ax=axes[3], legend=False); axes[3].set_ylabel("Residual")
            plt.xlabel("Date"); fig_decomp.suptitle(f"Decomposition of {selected_entity_decomp} ({decomp_model_to_use.capitalize()}, Period={period})", y=1.01); plt.tight_layout()
            st.pyplot(fig_decomp)
            plt.close(fig_decomp)
        except ValueError as ve: st.error(f"Decomposition failed: {ve}. Check period and data.")
        except Exception as e: st.error(f"Unexpected error during decomposition: {e}")
    else: st.warning("Please select a valid ticker.")

def optimal_lags():
    st.header("Stationarity Test and PACF (Single Stock)")
    if 'entities' not in st.session_state or not st.session_state.entities:
        st.warning("No single stock data loaded. Please load data via the sidebar first using 'Load & Process Data (for Forecasting)'.")
        st.info("This page analyzes the data loaded via the sidebar for the forecasting workflow.")
        return
    data_type = st.session_state.get('data_type', 'returns')
    outlier_status = " (Outliers Treated)" if st.session_state.get('apply_outlier_treatment', False) else ""
    st.info(f"Analyzing: Daily {data_type.capitalize()}{outlier_status} for selected stock (from sidebar loading).")

    selected_entity_lags = st.selectbox("Select a ticker", st.session_state.entities, key="lags_entity")
    if selected_entity_lags and selected_entity_lags in st.session_state.dataframes:
        df_lags_full = st.session_state.dataframes[selected_entity_lags]
        if df_lags_full.empty: st.warning(f"No data available for ticker {selected_entity_lags}."); return
        df_lags = df_lags_full['Actual'].dropna()
        if df_lags.empty: st.warning(f"No valid (non-NaN) data for ticker {selected_entity_lags}."); return

        st.subheader(f"Augmented Dickey-Fuller (ADF) Test for {selected_entity_lags}")
        alpha_adf = st.slider("Significance Level (alpha)", 0.01, 0.10, 0.05, 0.01, key="adf_alpha")
        try:
            adf_result = adfuller(df_lags)
            st.write(f"ADF Statistic: {adf_result[0]:.4f}")
            st.write(f"p-value: {adf_result[1]:.4f}")
            st.write(f"Critical Values: {adf_result[4]}")
            is_stationary = adf_result[1] < alpha_adf
            if is_stationary: st.success(f"Result: Likely Stationary (Reject H0 at alpha={alpha_adf}) - Good for returns!")
            else: st.warning(f"Result: Likely Non-Stationary (Fail to reject H0 at alpha={alpha_adf}) - Common for prices.")
        except Exception as e: st.error(f"ADF test failed: {e}")

        st.subheader(f"Partial Autocorrelation Function (PACF) for {selected_entity_lags}")
        n_obs_pacf = len(df_lags)
        max_possible_lags = max(1, n_obs_pacf // 2 - 1) if n_obs_pacf > 4 else 1
        default_lags_pacf = min(40, max_possible_lags) if max_possible_lags > 0 else 1

        if max_possible_lags < 1: st.warning("Not enough data points to plot PACF.")
        else:
            default_lags_pacf_adj = max(1, min(default_lags_pacf, max_possible_lags))
            lags_pacf = st.slider("Number of Lags for PACF", 1, max_possible_lags, default_lags_pacf_adj, key="pacf_lags_slider")
            try:
                if n_obs_pacf <= lags_pacf: lags_pacf = max(1, n_obs_pacf - 1)

                if lags_pacf >= 1:
                     fig_pacf, ax_pacf = plt.subplots(figsize=(10, 5))
                     plot_pacf(df_lags, lags=lags_pacf, ax=ax_pacf, method='ywm', zero=False)
                     ax_pacf.set_title(f"PACF for {selected_entity_lags}"); ax_pacf.set_xlabel("Lag (Days)"); ax_pacf.set_ylabel("PACF"); st.pyplot(fig_pacf)
                     plt.close(fig_pacf)
                     st.info("Significant spikes suggest the order 'p' for AR models.")
                else: st.warning("Not enough data to plot PACF after adjustments.")
            except Exception as e: st.error(f"Could not plot PACF: {e}")
    else: st.warning("Please select a valid ticker.")

def forecast_models():
    st.header("Time Series Forecasting Models (Single Stock)")
    if 'entities' not in st.session_state or not st.session_state.entities:
        st.warning("No single stock data loaded for forecasting. Please load data via the sidebar first using 'Load & Process Data (for Forecasting)'.")
        st.info("This page runs forecasting models on the data loaded via the sidebar.")
        return
    data_type = st.session_state.get('data_type', 'returns')
    outlier_status = " (Outliers Treated)" if st.session_state.get('apply_outlier_treatment', False) else ""
    st.info(f"Modeling: Daily {data_type.capitalize()}{outlier_status} for selected stock (from sidebar loading).")

    selected_entity_model = st.selectbox("Select a ticker for modeling", st.session_state.entities, key="model_entity")
    if not selected_entity_model or selected_entity_model not in st.session_state.dataframes: st.warning("Please select a valid ticker."); return

    df_model_full = st.session_state.dataframes[selected_entity_model]
    if df_model_full.empty: st.error(f"No data available for modeling ticker {selected_entity_model}."); return
    full_series = df_model_full['Actual'].dropna()
    if full_series.empty: st.error(f"No valid (non-NaN) data for modeling ticker {selected_entity_model}."); return

    inferred_freq = pd.infer_freq(full_series.index)
    freq_offset = None; freq_warning = ""
    if inferred_freq:
        try: freq_offset = pd.tseries.frequencies.to_offset(inferred_freq)
        except ValueError: freq_warning = f"Could not convert inferred frequency '{inferred_freq}'. Using fallback 'B'."; freq_offset = pd.tseries.frequencies.to_offset('B')
    else: freq_warning = "Could not infer frequency; using fallback 'B'."; freq_offset = pd.tseries.frequencies.to_offset('B')
    if freq_warning: st.caption(freq_warning)
    st.caption(f"Using frequency offset: {freq_offset.name if freq_offset else 'None'} for date calculations.")
    if freq_offset is None: st.error("Could not determine frequency offset. Cannot proceed."); return

    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    total_days = len(full_series)
    max_test_days = max(1, total_days // 3) if total_days > 3 else 1
    default_test_days = max(1, min(60, max_test_days)) if total_days > 90 else max(1, total_days // 4)
    default_test_days = max(1, min(default_test_days, max_test_days))

    with col1:
        m_future = st.slider("Test Set Size / Future Forecast Horizon (Days)", 1, max_test_days, default_test_days, key="model_future",
                             help="Number of **days** for the test set (from end) AND number of **days** to forecast.")

    min_required_train_days = 30
    max_possible_train_len = max(0, total_days - m_future)
    default_train_days = min(252*2, max_possible_train_len) if max_possible_train_len > 252 else max(min_required_train_days, max_possible_train_len)
    default_train_days_adj = max(min_required_train_days, min(default_train_days, max_possible_train_len))

    with col2:
         if max_possible_train_len < min_required_train_days:
             st.error(f"Insufficient data. Need >= {min_required_train_days} train days + {m_future} test days, have {max_possible_train_len} available ({total_days} total). Reduce test/forecast horizon or get more data."); return
         m_train = st.slider(f"Training Days (before test set)", min_required_train_days, max_possible_train_len, default_train_days_adj, key="model_train",
                             help="Number of recent **days** for training.")

    st.write(f"**Data Split:** Training: {m_train} days, Test/Forecast: {m_future} days")

    col_cv, col_exog = st.columns(2)
    with col_cv: n_splits = st.slider("Cross-Validation Splits", 2, 10, 5, key="model_cv", help="TimeSeriesSplit CV on training history.")
    exog_help_text = "Include Lagged Squared Value (proxy for volatility) as predictor? (Actual.shift(1)**2)"
    with col_exog: use_exog = st.checkbox("Use Lagged Squared Value as Exogenous?", True, key="use_exog", help=exog_help_text)

    # --- NUEVO SLIDER PARA ML FEATURES ---
    if xgb_installed or lgbm_installed:
        lag_features = st.slider("Max Lags for ML Models (XGB/LGBM)", 1, 30, 5, key="ml_max_lags", 
                                 help="Maximum number of historical days (lags) to use as features for XGBoost and LightGBM.")
    else:
        lag_features = 5 # Default value if libraries are missing

    st.markdown("---"); st.subheader("Model Specific Parameters")
    col_arima, col_ets, col_prophet, col_tbats = st.columns(4)
    with col_arima:
         st.markdown("**ARIMA/SARIMAX**");
         m_seasonal = st.number_input("Seasonal Period (m - days)", 1, 365, 1, 1, key='arima_m_seasonal',
                                      help="Seasonality period (days). 1 = non-seasonal. 5 or 7 for weekly. Requires sufficient data.")
         if m_seasonal > 1 and len(full_series) < 2 * m_seasonal: st.warning(f"Data size ({len(full_series)}) may be small for m={m_seasonal}.")
    with col_ets:
         st.markdown("**ETS**");
         trend_ets = st.selectbox("Trend", ("add", "mul", None), index=0, key="ets_trend");
         seasonal_ets = st.selectbox("Seasonality", ("add", "mul", None), index=0, key="ets_seasonal", help="Requires positive data for 'mul'.")
         # ets_seasonal_periods = int(m_seasonal) if seasonal_ets and m_seasonal > 1 else None # No se usa explicitamente
    with col_prophet:
        st.markdown("**Prophet**")
        prophet_seasonality_mode = st.selectbox("Seasonality Mode", ("additive", "multiplicative"), index=0, key="prophet_seas_mode")
        prophet_growth = st.selectbox("Growth Model", ("linear", "logistic"), index=0, key="prophet_growth")
        if prophet_growth == 'logistic':
             st.caption("Logistic growth requires defining capacity ('cap'). Using linear.")
             prophet_growth = 'linear'
             
    with col_tbats:
        st.markdown("**TBATS**")
        tbats_periods_str = st.text_input(
            "Seasonal Periods", 
            "5, 21", 
            key="tbats_periods",
            help="Comma-separated seasonal periods (e.g., 5 for weekly, 21 for monthly trading days)."
        )
        try:
            seasonal_periods_tbats = [int(p.strip()) for p in tbats_periods_str.split(',') if p.strip().isdigit()]
            if not seasonal_periods_tbats:
                st.warning("No valid seasonal periods entered for TBATS.")
        except:
            st.error("Invalid format for TBATS seasonal periods. Use comma-separated numbers.")
            seasonal_periods_tbats = []

    test_data = full_series.iloc[-m_future:]
    train_end_index_pos = len(full_series) - m_future
    train_start_index_pos = max(0, train_end_index_pos - m_train)
    train_data = full_series.iloc[train_start_index_pos:train_end_index_pos]

    if train_data.empty or test_data.empty: st.error("Could not create valid train/test split."); return
    st.info(f"Training: {train_data.index.min():%Y-%m-%d} to {train_data.index.max():%Y-%m-%d} ({len(train_data)} pts). Testing: {test_data.index.min():%Y-%m-%d} to {test_data.index.max():%Y-%m-%d} ({len(test_data)} pts).")

    # train_data_exog, combined_exog_for_prediction = None, None # No se usa explicitamente
    future_prediction_dates = None; final_train_exog_clean = None; combined_exog_clean = None
    exog_name = "Exog_Lag1_Sq"

    if not test_data.empty and test_data.index[-1] is not pd.NaT and freq_offset is not None:
        try:
            future_prediction_dates = pd.date_range(start=test_data.index[-1] + freq_offset, periods=m_future, freq=freq_offset)
        except Exception as e: st.error(f"Could not generate future prediction dates: {e}. Cannot proceed."); return
    else: st.error("Cannot generate future prediction dates (test data empty, invalid end date, or missing frequency). Cannot proceed."); return

    # --- PREPARACIÓN DE VARIABLE EXÓGENA (para modelos de TS y VAR/ML) ---
    if use_exog:
        with st.spinner("Preparing exogenous variable..."):
            try:
                full_series_lagged = full_series.shift(1)
                full_series_exog_sq = (full_series_lagged**2).rename(exog_name)
                
                # Slicing the training part
                train_data_exog_prep = full_series_exog_sq.reindex(train_data.index).ffill().bfill().fillna(0)
                
                # Slicing the test part (for prediction)
                test_data_exog_prep = full_series_exog_sq.reindex(test_data.index).ffill().bfill().fillna(0)
                
                # Future part (assuming last known actual value squared for m_future steps)
                last_known_actual_value = test_data.iloc[-1] if not test_data.empty else train_data.iloc[-1]
                future_exog_list = [(last_known_actual_value**2)] * m_future
                future_exog_prep = pd.Series(future_exog_list, index=future_prediction_dates, name=exog_name)
                
                combined_exog_for_prediction_prep = pd.concat([test_data_exog_prep, future_exog_prep])
                
                final_train_exog_clean = train_data_exog_prep.reindex(train_data.index).fillna(0)
                full_pred_index = test_data.index.union(future_prediction_dates)
                combined_exog_clean = combined_exog_for_prediction_prep.reindex(full_pred_index).fillna(0)
                
                if final_train_exog_clean.isnull().any() or combined_exog_clean.isnull().any():
                     st.warning("NaNs detected in exogenous variable after processing. Filling with 0.")
                st.caption("Exogenous variable prepared.")
            except Exception as e:
                 st.error(f"Failed to create exogenous variable: {e}")
                 use_exog = False; final_train_exog_clean = None; combined_exog_clean = None
                 st.warning("Proceeding without exogenous variable due to error.")

    # --- PREPARACIÓN DE DATOS PARA MODELOS ML (XGBoost/LightGBM) ---
    y_full, X_full = pd.Series(dtype=float), pd.DataFrame()
    X_train_ml, y_train_ml, X_test_ml, y_test_ml = [None] * 4
    
    if xgb_installed or lgbm_installed:
        with st.spinner(f"Generating time series features (lags={lag_features}) for ML models..."):
            full_df_ml = full_series.to_frame(name='Actual')
            
            # Incorporate Exogenous feature into ML DataFrame if requested
            if use_exog and final_train_exog_clean is not None and combined_exog_clean is not None:
                # Usamos solo la parte histórica para generar lags, la parte futura se genera en el bootstrapping
                full_exog_historical = final_train_exog_clean.reindex(full_series.index)
                full_df_ml[exog_name] = full_exog_historical
                
            full_df_ml_features = create_ts_features(full_df_ml, lag_max=lag_features) 

            # Split into X and y for ML
            if 'Actual' in full_df_ml_features.columns:
                y_full = full_df_ml_features['Actual']
                X_full = full_df_ml_features.drop(columns=['Actual'])
            
            # Update train/test splits based on the new featured data indices
            train_indices_ml = train_data.index.intersection(y_full.index)
            test_indices_ml = test_data.index.intersection(y_full.index)
            
            if not y_full.empty and not train_indices_ml.empty and not test_indices_ml.empty:
                y_train_ml = y_full.loc[train_indices_ml]
                X_train_ml = X_full.loc[train_indices_ml]
                y_test_ml = y_full.loc[test_indices_ml]
                X_test_ml = X_full.loc[test_indices_ml]
                st.caption(f"ML Train Data Size: {len(y_train_ml)} (Loss of {len(train_data) - len(y_train_ml)} pts due to lags/rollings)")
            else:
                 st.warning("ML models skipped: Not enough data remaining after feature engineering (lags/rollings).")

    # --- MODEL LIST FOR CV ---
    model_names_cv = ["ARIMA", "ARIMAX", "SARIMAX", "ETS", "VAR", "TBATS"]
    if prophet_installed: model_names_cv.append("Prophet")
    # Solo añadimos modelos ML si tenemos datos suficientes para entrenamiento (y_train_ml)
    if xgb_installed and y_train_ml is not None and not y_train_ml.empty: model_names_cv.append("XGBoost")
    if lgbm_installed and y_train_ml is not None and not y_train_ml.empty: model_names_cv.append("LightGBM")


    st.markdown("---"); st.subheader("Time Series Cross-Validation")
    cv_test_size = max(5, m_future // 2 if m_future > 1 else 1)
    actual_cv_splits = n_splits
    try:
        # min_data_needed_strict = cv_test_size + (n_splits-1) * cv_test_size # No se usa explicitamente
        if len(train_data) < n_splits * cv_test_size + 5: # Estimación umbral
            st.warning(f"Training data ({len(train_data)}) may be small for {n_splits} CV splits with test size {cv_test_size}. Reducing splits if necessary.")
            n_splits = max(2, len(train_data) // (cv_test_size + 1) ) # Ajuste más seguro
            st.info(f"Adjusted CV splits to {n_splits} based on data availability.")
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=cv_test_size)
        actual_cv_splits = tscv.get_n_splits(train_data) # Esto es lo que realmente importa
        if actual_cv_splits < 2: st.error(f"Cannot perform CV with < 2 splits ({actual_cv_splits} available). Increase training data or decrease CV splits/test size."); return
    except ValueError as e: st.error(f"Failed to create TimeSeriesSplit: {e}."); return

    all_rmse_scores = {name: [] for name in model_names_cv}
    progress_bar_cv = st.progress(0); cv_message = st.empty()
    split_data_cv = train_data; split_exog_cv = final_train_exog_clean
    cv_split_count = 0

    for i, (train_idx, val_idx) in enumerate(tscv.split(split_data_cv)):
        cv_split_count += 1
        cv_train, cv_val = split_data_cv.iloc[train_idx], split_data_cv.iloc[val_idx]
        if cv_val.empty:
            st.caption(f"CV Split {cv_split_count}/{actual_cv_splits}: Skipped (empty validation set).")
            [all_rmse_scores[name].append(np.nan) for name in model_names_cv]
            continue
        
        # --- PREPARACIÓN DE EXÓGENAS PARA MODELOS TS/VAR/PROPHET ---
        cv_train_exog, cv_val_exog = None, None
        prophet_cv_exog_ready = False
        if use_exog and split_exog_cv is not None and not split_exog_cv.empty:
             try:
                 temp_train_exog = split_exog_cv.iloc[train_idx].reindex(cv_train.index)
                 temp_val_exog = split_exog_cv.iloc[val_idx].reindex(cv_val.index)
                 
                 # CORRECCIÓN DE DEPRECACIÓN DE PANDAS: Usar ffill() y bfill() en lugar de fillna(method=...)
                 temp_train_exog = temp_train_exog.ffill().bfill().fillna(0)
                 temp_val_exog = temp_val_exog.ffill().bfill().fillna(0)
                 
                 if not temp_train_exog.isnull().any() and not temp_val_exog.isnull().any():
                      cv_train_exog, cv_val_exog = temp_train_exog, temp_val_exog
                      prophet_cv_exog_ready = True
                 else: st.caption(f"CV Split {cv_split_count}: NaN found in Exog, some models skipped.")
             except Exception as e_exog_cv: st.caption(f"CV Split {cv_split_count}: Error slicing Exog ({e_exog_cv}). Some models skipped.")
        
        # --- PREPARACIÓN DE FEATURES PARA MODELOS ML (XGB/LGBM) ---
        X_cv_train_ml, y_cv_train_ml, X_cv_val_ml, y_cv_val_ml = None, None, None, None
        if "XGBoost" in model_names_cv or "LightGBM" in model_names_cv:
            try:
                # Obtenemos los datos necesarios para generar lags, incluyendo las filas anteriores al tren
                # Esto es crucial para que los lags en el inicio del cv_train sean correctos.
                min_cv_idx = train_idx.min() - lag_features 
                # Asegura que el slice no vaya antes del inicio de train_data
                start_ml_slice = max(0, min_cv_idx)
                
                # Usamos el split_data_cv original (solo Actual) y añadimos exógena si está
                cv_df_raw = split_data_cv.iloc[start_ml_slice:val_idx.max()].to_frame(name='Actual')
                
                # Añadir exógena si está siendo usada
                if use_exog and split_exog_cv is not None:
                    # sliced_exog_cv incluye historial pre-train
                    sliced_exog_cv = split_exog_cv.iloc[start_ml_slice:val_idx.max()]
                    cv_df_raw[exog_name] = sliced_exog_cv.reindex(cv_df_raw.index).fillna(0)
                
                cv_features = create_ts_features(cv_df_raw, lag_max=lag_features)

                # Re-split based on feature indices (only use the indices that actually ended up in cv_train and cv_val)
                cv_train_ml_indices = cv_train.index.intersection(cv_features.index)
                cv_val_ml_indices = cv_val.index.intersection(cv_features.index)
                
                if not cv_train_ml_indices.empty and not cv_val_ml_indices.empty:
                    y_cv_train_ml = cv_features.loc[cv_train_ml_indices, 'Actual']
                    X_cv_train_ml = cv_features.loc[cv_train_ml_indices].drop(columns=['Actual'])
                    y_cv_val_ml = cv_features.loc[cv_val_ml_indices, 'Actual']
                    X_cv_val_ml = cv_features.loc[cv_val_ml_indices].drop(columns=['Actual'])
                else:
                    # Esto puede ocurrir si el conjunto de validación es demasiado pequeño después de los lags
                    raise ValueError("Insufficient data remaining after lag generation for ML CV split.")
            
            except Exception as e_ml_cv:
                st.caption(f"CV Split {cv_split_count}: Error preparing ML features ({e_ml_cv}). ML models skipped.")

        progress_text = f"Running CV Split {cv_split_count}/{actual_cv_splits}..."; progress_bar_cv.progress(cv_split_count / actual_cv_splits); cv_message.text(progress_text)
        min_cv_train_len = max(15, 2 * m_seasonal if m_seasonal > 1 else 15)
        
        if len(cv_train) < min_cv_train_len:
            st.caption(f"CV Split {cv_split_count}: Skipped (training data < {min_cv_train_len}).")
            [all_rmse_scores[name].append(np.nan) for name in model_names_cv]
            continue
        
        order_cv, seasonal_order_cv = (1,0,0), (0,0,0,0)
        
        # --- ARIMA / SARIMAX Parameter Selection ---
        if any(m in model_names_cv for m in ["ARIMA", "ARIMAX", "SARIMAX"]):
            try:
                cv_seasonal_flag = (m_seasonal > 1) and (len(cv_train) >= 2 * m_seasonal)
                if len(cv_train) > 10:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model_auto_cv = auto_arima(cv_train, seasonal=cv_seasonal_flag, m=m_seasonal if cv_seasonal_flag else 1,
                                                exogenous=cv_train_exog if use_exog and prophet_cv_exog_ready else None,
                                                suppress_warnings=True, error_action='ignore', stepwise=True, information_criterion='aic', n_jobs=1,
                                                max_p=5, max_q=5, max_P=2, max_Q=2)
                    order_cv, seasonal_order_cv = model_auto_cv.order, model_auto_cv.seasonal_order
                else: st.caption(f"CV Split {cv_split_count}: AutoARIMA skipped (too few obs). Using defaults.")
            except Exception as e_auto_cv: st.caption(f"CV Split {cv_split_count}: AutoARIMA failed ({e_auto_cv}). Using defaults.")
        
        # --- ARIMA ---
        if "ARIMA" in model_names_cv:
            try:
                model=ARIMA(cv_train,order=order_cv).fit()
                pred=model.predict(start=cv_val.index[0],end=cv_val.index[-1])
                all_rmse_scores["ARIMA"].append(np.sqrt(mean_squared_error(cv_val,pred)))
            except Exception: all_rmse_scores["ARIMA"].append(np.nan)
        
        # --- ARIMAX ---
        if "ARIMAX" in model_names_cv:
            if use_exog and cv_train_exog is not None and cv_val_exog is not None:
                try:
                    model=ARIMA(cv_train,order=order_cv,exog=cv_train_exog).fit()
                    pred=model.predict(start=cv_val.index[0],end=cv_val.index[-1],exog=cv_val_exog)
                    all_rmse_scores["ARIMAX"].append(np.sqrt(mean_squared_error(cv_val,pred)))
                except Exception: all_rmse_scores["ARIMAX"].append(np.nan)
            else: all_rmse_scores["ARIMAX"].append(np.nan)
            
        # --- SARIMAX ---
        if "SARIMAX" in model_names_cv:
            try:
                s_order = list(seasonal_order_cv);
                if len(s_order) == 4 and s_order[3] == 0 and m_seasonal > 1 and any(p > 0 for p in s_order[:3]): s_order[3] = m_seasonal
                valid_seasonal_order = tuple(s_order)
                model=SARIMAX(cv_train,order=order_cv,seasonal_order=valid_seasonal_order, exog=cv_train_exog if use_exog else None,
                              enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
                pred=model.predict(start=cv_val.index[0],end=cv_val.index[-1],exog=cv_val_exog if use_exog else None)
                all_rmse_scores["SARIMAX"].append(np.sqrt(mean_squared_error(cv_val,pred)))
            except Exception: all_rmse_scores["SARIMAX"].append(np.nan)
        
        # --- ETS ---
        if "ETS" in model_names_cv:
            try:
                ets_s_cv, ets_t_cv = seasonal_ets, trend_ets
                if (ets_s_cv == 'mul' or ets_t_cv == 'mul') and (cv_train <= 1e-8).any():
                    ets_s_cv = 'add' if ets_s_cv=='mul' else ets_s_cv
                    ets_t_cv = 'add' if ets_t_cv=='mul' else ets_t_cv
                ets_p_cv = int(m_seasonal) if ets_s_cv and m_seasonal>1 and len(cv_train) >= 2 * m_seasonal else None
                if ets_p_cv and len(cv_train) < 2 * ets_p_cv: ets_s_cv=None; ets_p_cv=None
                model=ExponentialSmoothing(cv_train,trend=ets_t_cv,seasonal=ets_s_cv,seasonal_periods=ets_p_cv, initialization_method='estimated').fit()
                pred=model.predict(start=cv_val.index[0],end=cv_val.index[-1])
                all_rmse_scores["ETS"].append(np.sqrt(mean_squared_error(cv_val,pred)))
            except Exception: all_rmse_scores["ETS"].append(np.nan)
            
        # --- VAR ---
        if "VAR" in model_names_cv:
            if use_exog and cv_train_exog is not None and cv_val_exog is not None:
                 try:
                     cv_var_data = pd.DataFrame({'Actual':cv_train, exog_name:cv_train_exog}).dropna()
                     if len(cv_var_data) > 10: # Umbral mínimo para VAR
                         maxlags_cv = min(10, len(cv_var_data)//2 - 1)
                         if maxlags_cv >= 1:
                             with warnings.catch_warnings(): warnings.simplefilter("ignore")
                             var_select = VAR(cv_var_data)
                             try: lag_res = var_select.select_order(maxlags=maxlags_cv); lag_cv = lag_res.selected_orders['aic']
                             except Exception as e_lag_sel: st.caption(f"VAR CV Lag Sel Failed (Split {cv_split_count}): {e_lag_sel}. Using lag 1."); lag_cv = 1
                             if len(cv_var_data) > lag_cv: # Asegurar suficientes observaciones para el lag
                                 model_var = var_select.fit(lag_cv)
                                 fc = model_var.forecast(y=cv_var_data.values[-lag_cv:], steps=len(cv_val))
                                 pred = fc[:, 0] # Asumiendo que 'Actual' es la primera columna
                                 all_rmse_scores["VAR"].append(np.sqrt(mean_squared_error(cv_val,pred)))
                             else: all_rmse_scores["VAR"].append(np.nan)
                         else: all_rmse_scores["VAR"].append(np.nan) # No suficientes datos para lag
                     else: all_rmse_scores["VAR"].append(np.nan) # No suficientes datos para VAR
                 except Exception as e_var_cv: all_rmse_scores["VAR"].append(np.nan); st.caption(f"VAR CV Error (Split {cv_split_count}): {e_var_cv}")
            else: all_rmse_scores["VAR"].append(np.nan) # VAR requiere exógena en esta configuración
        
        # --- TBATS ---
        if "TBATS" in model_names_cv:
            if tbats_installed and seasonal_periods_tbats:
                try:
                    # Filtra periodos que son demasiado grandes para los datos de entrenamiento del CV
                    valid_cv_periods = [p for p in seasonal_periods_tbats if len(cv_train) > 2 * p]
                    if valid_cv_periods:
                        estimator = TBATS(seasonal_periods=valid_cv_periods, use_arma_errors=False, n_jobs=1)
                        model = estimator.fit(cv_train)
                        pred = model.forecast(steps=len(cv_val))
                        all_rmse_scores["TBATS"].append(np.sqrt(mean_squared_error(cv_val, pred)))
                    else:
                        all_rmse_scores["TBATS"].append(np.nan)
                except Exception as e_tbats_cv:
                    st.caption(f"TBATS CV failed on split {cv_split_count}: {e_tbats_cv}")
                    all_rmse_scores["TBATS"].append(np.nan)
            else:
                all_rmse_scores["TBATS"].append(np.nan)    
        
        # --- Prophet ---
        if "Prophet" in model_names_cv:
            if prophet_installed:
                try:
                    df_prophet_train = cv_train.reset_index(); df_prophet_train.columns = ['ds', 'y']
                    df_prophet_future = pd.DataFrame({'ds': cv_val.index})
                    prophet_use_exog_cv = use_exog and prophet_cv_exog_ready
                    m_prophet_cv = Prophet(seasonality_mode=prophet_seasonality_mode, growth=prophet_growth)
                    if prophet_use_exog_cv:
                        df_prophet_train[exog_name] = cv_train_exog.values
                        df_prophet_future[exog_name] = cv_val_exog.values
                        m_prophet_cv.add_regressor(exog_name)
                    with warnings.catch_warnings(): warnings.simplefilter("ignore")
                    m_prophet_cv.fit(df_prophet_train)
                    fcst_cv = m_prophet_cv.predict(df_prophet_future)
                    pred = fcst_cv['yhat'].values
                    all_rmse_scores["Prophet"].append(np.sqrt(mean_squared_error(cv_val, pred)))
                except Exception as e_prophet_cv: st.caption(f"Prophet CV failed on split {cv_split_count}: {e_prophet_cv}"); all_rmse_scores["Prophet"].append(np.nan)
            else: all_rmse_scores["Prophet"].append(np.nan)
            
        # --- XGBoost ---
        if "XGBoost" in model_names_cv:
            if xgb_installed and X_cv_train_ml is not None and not X_cv_train_ml.empty:
                try:
                    model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
                    model.fit(X_cv_train_ml, y_cv_train_ml)
                    pred = model.predict(X_cv_val_ml)
                    all_rmse_scores["XGBoost"].append(np.sqrt(mean_squared_error(y_cv_val_ml, pred)))
                except Exception as e_xgb_cv:
                    st.caption(f"XGBoost CV failed on split {cv_split_count}: {e_xgb_cv}")
                    all_rmse_scores["XGBoost"].append(np.nan)
            else:
                all_rmse_scores["XGBoost"].append(np.nan)
        
        # --- LightGBM ---
        if "LightGBM" in model_names_cv:
            if lgbm_installed and X_cv_train_ml is not None and not X_cv_train_ml.empty:
                try:
                    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
                    model.fit(X_cv_train_ml, y_cv_train_ml)
                    pred = model.predict(X_cv_val_ml)
                    all_rmse_scores["LightGBM"].append(np.sqrt(mean_squared_error(y_cv_val_ml, pred)))
                except Exception as e_lgbm_cv:
                    st.caption(f"LightGBM CV failed on split {cv_split_count}: {e_lgbm_cv}")
                    all_rmse_scores["LightGBM"].append(np.nan)
            else:
                all_rmse_scores["LightGBM"].append(np.nan)

    progress_bar_cv.empty(); cv_message.empty()
    results_cv = {name: np.nanmean([s for s in scores if pd.notna(s)]) for name, scores in all_rmse_scores.items()}
    valid_results_cv = {name: score for name, score in results_cv.items() if pd.notna(score)}
    if valid_results_cv:
        st.write("Mean RMSE from Cross-Validation:");
        results_df_cv = pd.DataFrame(list(valid_results_cv.items()), columns=['Model', 'Mean RMSE']).sort_values('Mean RMSE').reset_index(drop=True)
        st.dataframe(results_df_cv.style.format({'Mean RMSE': '{:,.4f}'}))
        best_cv_model = results_df_cv.iloc[0]['Model'] if not results_df_cv.empty else "N/A"
        st.info(f"Based on CV (RMSE), best performing model might be: **{best_cv_model}**")
    else: st.warning("Could not calculate valid CV scores for any model.")

    st.markdown("---"); st.subheader(f"Final Model Training & Prediction"); st.write(f"Training up to {train_data.index.max():%Y-%m-%d}, Forecasting {m_future} days ahead.")
    final_models={}; test_predictions={}; future_predictions={}; model_residuals={}
    n_total_preds = len(test_data) + m_future
    predict_start_pos = len(train_data); predict_end_pos = len(train_data) + n_total_preds - 1
    final_order, final_seasonal_order = (1,0,0), (0,0,0,0)

    if any(m in model_names_cv for m in ["ARIMA", "ARIMAX", "SARIMAX"]):
        with st.spinner("Running AutoARIMA to determine final orders..."):
            try:
                auto_arima_exog_final = final_train_exog_clean if use_exog else None
                apply_seasonal = (m_seasonal > 1) and (len(train_data) >= 2 * m_seasonal)
                with warnings.catch_warnings(): warnings.simplefilter("ignore")
                final_auto=auto_arima(train_data, exogenous=auto_arima_exog_final, seasonal=apply_seasonal, m=m_seasonal if apply_seasonal else 1,
                                      suppress_warnings=True, error_action='ignore', stepwise=True, information_criterion='aic', n_jobs=1,
                                      max_p=5, max_q=5, max_P=2, max_Q=2)
                final_order, final_seasonal_order=final_auto.order, final_auto.seasonal_order
                st.success(f"AutoARIMA selected Order: {final_order}, Seasonal Order: {final_seasonal_order}"); final_models['AutoARIMA_Params']={'order':final_order,'seasonal':final_seasonal_order}
            except Exception as e: st.error(f"AutoARIMA failed during final order selection: {e}. Using default orders (1,0,0)(0,0,0,0).")

    if test_data.empty or future_prediction_dates is None: st.error("Cannot fit final models: Test data or future dates are invalid."); return
    full_prediction_index = test_data.index.union(future_prediction_dates)

    # --- FINAL ARIMA/SARIMAX/ETS/VAR/TBATS/PROPHET FITTING (Existente) ---
    if "ARIMA" in model_names_cv:
        with st.spinner("Fitting ARIMA..."):
            try:
                model = ARIMA(train_data, order=final_order).fit()
                final_models['ARIMA'] = model
                preds = model.predict(start=predict_start_pos, end=predict_end_pos)
                if isinstance(preds, pd.Series) and not preds.empty:
                    preds.index = full_prediction_index
                    test_predictions['ARIMA'] = preds.reindex(test_data.index)
                    future_predictions['ARIMA'] = preds.reindex(future_prediction_dates)
                    model_residuals['ARIMA'] = model.resid.reindex(train_data.index)
                else: st.warning("ARIMA prediction yielded no results.")
            except Exception as e: st.warning(f"ARIMA failed: {e}")
    if "ARIMAX" in model_names_cv:
        if use_exog and final_train_exog_clean is not None and combined_exog_clean is not None:
            with st.spinner("Fitting ARIMAX..."):
                try:
                    model = ARIMA(train_data, order=final_order, exog=final_train_exog_clean).fit()
                    final_models['ARIMAX'] = model
                    preds = model.predict(start=predict_start_pos, end=predict_end_pos, exog=combined_exog_clean)
                    if isinstance(preds, pd.Series) and not preds.empty:
                        preds.index = full_prediction_index
                        test_predictions['ARIMAX'] = preds.reindex(test_data.index)
                        future_predictions['ARIMAX'] = preds.reindex(future_prediction_dates)
                        model_residuals['ARIMAX'] = model.resid.reindex(train_data.index)
                    else: st.warning("ARIMAX prediction yielded no results.")
                except Exception as e: st.warning(f"ARIMAX failed: {e}")
        elif use_exog: st.warning("ARIMAX skipped: Exogenous variable requested but not prepared.")
    if "SARIMAX" in model_names_cv:
        with st.spinner("Fitting SARIMAX..."):
            try:
                s_order_final = list(final_seasonal_order);
                if len(s_order_final) == 4 and s_order_final[3] == 0 and m_seasonal > 1 and any(p > 0 for p in s_order_final[:3]): s_order_final[3] = m_seasonal
                valid_seasonal_order_final = tuple(s_order_final)
                model = SARIMAX(train_data, order=final_order, seasonal_order=valid_seasonal_order_final, exog=final_train_exog_clean if use_exog else None,
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                final_models['SARIMAX'] = model
                preds = model.predict(start=predict_start_pos, end=predict_end_pos, exog=combined_exog_clean if use_exog else None)
                if isinstance(preds, pd.Series) and not preds.empty:
                    preds.index = full_prediction_index
                    test_predictions['SARIMAX'] = preds.reindex(test_data.index)
                    future_predictions['SARIMAX'] = preds.reindex(future_prediction_dates)
                    model_residuals['SARIMAX'] = model.resid.reindex(train_data.index)
                else: st.warning("SARIMAX prediction yielded no results.")
            except Exception as e: st.warning(f"SARIMAX failed: {e}")
    if "ETS" in model_names_cv:
        with st.spinner("Fitting ETS..."):
            try:
                ets_s_final, ets_t_final = seasonal_ets, trend_ets
                if (ets_s_final == 'mul' or ets_t_final == 'mul') and (train_data <= 1e-8).any():
                    st.caption("ETS: Multiplicative component detected with non-positive training data. Switching to additive.")
                    ets_s_final = 'add' if ets_s_final=='mul' else ets_s_final
                    ets_t_final = 'add' if ets_t_final=='mul' else ets_t_final
                ets_p_final = int(m_seasonal) if ets_s_final and m_seasonal>1 and len(train_data) >= 2 * m_seasonal else None
                if ets_p_final and len(train_data) < 2 * ets_p_final:
                     st.caption(f"ETS: Training data ({len(train_data)}) too short for seasonal period {ets_p_final}. Disabling seasonality.")
                     ets_s_final=None; ets_p_final=None
                model = ExponentialSmoothing(train_data, trend=ets_t_final, seasonal=ets_s_final, seasonal_periods=ets_p_final, initialization_method='estimated').fit()
                final_models['ETS'] = model
                preds = model.predict(start=predict_start_pos, end=predict_end_pos)
                if isinstance(preds, pd.Series) and not preds.empty:
                    preds.index = full_prediction_index
                    test_predictions['ETS'] = preds.reindex(test_data.index)
                    future_predictions['ETS'] = preds.reindex(future_prediction_dates)
                    model_residuals['ETS'] = model.resid.reindex(train_data.index)
                else: st.warning("ETS prediction yielded no results.")
            except Exception as e: st.warning(f"ETS failed: {e}")
    if "VAR" in model_names_cv:
        if use_exog and final_train_exog_clean is not None:
             with st.spinner("Fitting VAR..."):
                 try:
                     final_var_data = pd.DataFrame({'Actual':train_data, exog_name:final_train_exog_clean}).dropna()
                     if len(final_var_data) > 20 : # Umbral
                         maxlags = min(15, len(final_var_data) // 2 - 1)
                         if maxlags >= 1:
                             var_select = VAR(final_var_data)
                             with warnings.catch_warnings(): warnings.simplefilter("ignore")
                             try: lag_res = var_select.select_order(maxlags=maxlags); lag_order = lag_res.selected_orders['aic']
                             except Exception as e_lag_sel_final: st.caption(f"VAR Final Lag Sel Failed: {e_lag_sel_final}. Using lag 1."); lag_order = 1
                             st.info(f"VAR final lag order selected: {lag_order}")
                             if len(final_var_data) > lag_order:
                                 model_var = var_select.fit(lag_order); final_models['VAR'] = model_var
                                 fc = model_var.forecast(y=final_var_data.values[-lag_order:], steps=n_total_preds)
                                 preds_act = fc[:, 0]
                                 preds_ser = pd.Series(preds_act, index=full_prediction_index)
                                 test_predictions['VAR'] = preds_ser.reindex(test_data.index)
                                 future_predictions['VAR'] = preds_ser.reindex(future_prediction_dates)
                                 res_var = model_var.resid; res_idx = final_var_data.index[lag_order:]
                                 model_residuals['VAR'] = {}
                                 if isinstance(res_var, pd.DataFrame) and len(res_idx) == len(res_var):
                                     for col in res_var.columns: model_residuals['VAR'][col] = pd.Series(res_var[col].values, index=res_idx).reindex(train_data.index)
                                 elif isinstance(res_var, np.ndarray) and res_var.shape[0] == len(res_idx): # Ajuste para ndarray
                                     for i_col, col_name in enumerate(final_var_data.columns): model_residuals['VAR'][col_name] = pd.Series(res_var[:, i_col], index=res_idx).reindex(train_data.index)
                                 else: st.warning("Could not align VAR residuals properly.")
                             else: st.warning(f"VAR skipped: Insufficient obs ({len(final_var_data)}) for selected lag {lag_order}.")
                         else: st.warning("VAR skipped: Insufficient obs for lag selection.")
                     else: st.warning(f"VAR skipped: Insufficient valid obs ({len(final_var_data)}) with Exog for training.")
                 except Exception as e: st.warning(f"VAR failed: {e}"); [d.pop('VAR', None) for d in [test_predictions, future_predictions, final_models, model_residuals]]
        elif use_exog: st.warning("VAR skipped: Exogenous variable requested but not prepared.")
    if "Prophet" in model_names_cv:
        if prophet_installed:
             with st.spinner("Fitting Prophet..."):
                try:
                    df_prophet_train_final = train_data.reset_index(); df_prophet_train_final.columns = ['ds', 'y']
                    df_prophet_future_final = pd.DataFrame({'ds': full_prediction_index})
                    m_prophet_final = Prophet(seasonality_mode=prophet_seasonality_mode, growth=prophet_growth)
                    prophet_use_exog_final = use_exog and final_train_exog_clean is not None and combined_exog_clean is not None
                    if prophet_use_exog_final:
                        df_prophet_train_final[exog_name] = final_train_exog_clean.values
                        df_prophet_future_final[exog_name] = combined_exog_clean.values
                        m_prophet_final.add_regressor(exog_name)
                        st.caption("Prophet: Using exogenous regressor.")
                    elif use_exog: st.warning("Prophet: Exogenous regressor specified but not available. Fitting without it.")
                    with warnings.catch_warnings(): warnings.simplefilter("ignore")
                    m_prophet_final.fit(df_prophet_train_final)
                    final_models['Prophet'] = m_prophet_final
                    fcst_final = m_prophet_final.predict(df_prophet_future_final)
                    preds_prophet = pd.Series(fcst_final['yhat'].values, index=full_prediction_index)
                    test_predictions['Prophet'] = preds_prophet.reindex(test_data.index)
                    future_predictions['Prophet'] = preds_prophet.reindex(future_prediction_dates)
                    in_sample_fcst = m_prophet_final.predict(df_prophet_train_final)
                    residuals_prophet = df_prophet_train_final['y'].values - in_sample_fcst['yhat'].values
                    model_residuals['Prophet'] = pd.Series(residuals_prophet, index=train_data.index)
                except Exception as e: st.warning(f"Prophet failed: {e}"); [d.pop('Prophet', None) for d in [test_predictions, future_predictions, final_models, model_residuals]]
        else: st.warning("Prophet model skipped because the 'prophet' library is not installed.")
    
    
    if "TBATS" in model_names_cv:
        if tbats_installed and seasonal_periods_tbats:
            with st.spinner("Fitting TBATS..."):
                try:
                    # Asegúrate de que los periodos no sean demasiado grandes para el conjunto de entrenamiento final
                    valid_final_periods = [p for p in seasonal_periods_tbats if len(train_data) > 2 * p]
                    if not valid_final_periods:
                        st.warning(f"TBATS skipped: Training data ({len(train_data)}) too small for specified seasonal periods: {seasonal_periods_tbats}")
                    else:
                        estimator = TBATS(seasonal_periods=valid_final_periods, use_arma_errors=False, n_jobs=1)
                        model = estimator.fit(train_data)
                        final_models['TBATS'] = model
                        
                        # Predecir para el conjunto de test y el futuro
                        preds_tbats = model.forecast(steps=n_total_preds)
                        preds_ser = pd.Series(preds_tbats, index=full_prediction_index)

                        test_predictions['TBATS'] = preds_ser.reindex(test_data.index)
                        future_predictions['TBATS'] = preds_ser.reindex(future_prediction_dates)
                        
                        # Calcular residuos (y_hat son las predicciones in-sample)
                        if hasattr(model, 'y_hat'):
                           residuals_tbats = train_data - pd.Series(model.y_hat, index=train_data.index)
                           model_residuals['TBATS'] = residuals_tbats.reindex(train_data.index)

                except Exception as e:
                    st.warning(f"TBATS failed: {e}")
                    # Limpiar en caso de error
                    [d.pop('TBATS', None) for d in [test_predictions, future_predictions, final_models, model_residuals]]
        elif tbats_installed:
            st.warning("TBATS skipped: No valid seasonal periods were configured.")
        else:
            st.warning("TBATS skipped because the 'tbats' library is not installed.")

    # --- NUEVOS: FINAL XGBOOST FITTING ---
    if "XGBoost" in model_names_cv and xgb_installed and X_train_ml is not None and not X_train_ml.empty:
        with st.spinner("Fitting XGBoost..."):
            try:
                model = xgb.XGBRegressor(n_estimators=150, objective='reg:squarederror', random_state=42, n_jobs=1)
                model.fit(X_train_ml, y_train_ml)
                final_models['XGBoost'] = model
                
                # Predict Test Set (simple)
                test_preds = model.predict(X_test_ml)
                test_predictions['XGBoost'] = pd.Series(test_preds, index=X_test_ml.index)
                
                # Predict Future (Bootstrapping required)
                
                # FIX: Create template DataFrame for future prediction dates
                X_future_ml_xgb = pd.DataFrame(index=future_prediction_dates, columns=X_train_ml.columns).fillna(0)

                # Populate fixed time features for the future
                X_future_ml_xgb['dayofweek'] = X_future_ml_xgb.index.dayofweek
                X_future_ml_xgb['quarter'] = X_future_ml_xgb.index.quarter
                X_future_ml_xgb['month'] = X_future_ml_xgb.index.month
                X_future_ml_xgb['year'] = X_future_ml_xgb.index.year
                X_future_ml_xgb['dayofyear'] = X_future_ml_xgb.index.dayofyear
                
                # Populate initial lags using the last N historical observations (from y_full)
                # Usamos y_full porque contiene los valores reales después de la generación de features
                initial_lags_series = y_full.tail(lag_features).values[::-1] # Last actual values, reversed (index 0 is lag_1)
                
                # Also initialize rolling features in the first future step (simplification: last known rolling value)
                rolling_cols = [c for c in X_train_ml.columns if c.startswith('rolling_')]
                if rolling_cols and not X_test_ml.empty:
                    last_known_X = X_test_ml.iloc[-1]
                    for col in rolling_cols:
                        if col in last_known_X.index and col in X_future_ml_xgb.columns:
                            X_future_ml_xgb.loc[X_future_ml_xgb.index[0], col] = last_known_X[col]
                            
                # Set initial lag features for the FIRST future step (t=0)
                if not X_future_ml_xgb.empty:
                    for lag_idx in range(1, lag_features + 1):
                        col_name = f'lag_{lag_idx}'
                        if col_name in X_future_ml_xgb.columns:
                            if lag_idx <= len(initial_lags_series):
                                X_future_ml_xgb.loc[X_future_ml_xgb.index[0], col_name] = initial_lags_series[lag_idx - 1]
                            else:
                                X_future_ml_xgb.loc[X_future_ml_xgb.index[0], col_name] = 0 
                
                # Iterative prediction for the future m_future steps
                future_preds = []
                for t in range(m_future):
                    X_step = X_future_ml_xgb.iloc[t:t+1]
                    pred_step = model.predict(X_step)[0]
                    future_preds.append(pred_step)
                    
                    # Update lags for the next step (t+1) using the prediction
                    if t < m_future - 1:
                        # 1. Shift existing lags forward
                        for lag in range(lag_features, 1, -1):
                            lag_col_current = f'lag_{lag}'
                            lag_col_previous = f'lag_{lag-1}'
                            if lag_col_current in X_future_ml_xgb.columns and lag_col_previous in X_future_ml_xgb.columns:
                                X_future_ml_xgb.loc[X_future_ml_xgb.index[t+1], lag_col_current] = X_future_ml_xgb.loc[X_future_ml_xgb.index[t], lag_col_previous]
                        
                        # 2. Set lag_1 to the predicted value
                        if 'lag_1' in X_future_ml_xgb.columns:
                           X_future_ml_xgb.loc[X_future_ml_xgb.index[t+1], 'lag_1'] = pred_step
                        
                        # 3. If exog is based on lag_1 squared, update that too
                        if use_exog and exog_name in X_future_ml_xgb.columns:
                            X_future_ml_xgb.loc[X_future_ml_xgb.index[t+1], exog_name] = pred_step**2
                            
                        # 4. NOTE: Rolling features are complex to update iteratively (they rely on a mean/std of N past steps), 
                        # so we leave them set to the initial estimate or 0, which is a common simplification in ML time series bootstrapping.
                            
                future_predictions['XGBoost'] = pd.Series(future_preds, index=future_prediction_dates)
                model_residuals['XGBoost'] = pd.Series(y_train_ml.values - model.predict(X_train_ml), index=y_train_ml.index)
                
            except Exception as e:
                st.warning(f"XGBoost failed: {e}")
                [d.pop('XGBoost', None) for d in [test_predictions, future_predictions, final_models, model_residuals]]
        
    # --- NUEVOS: FINAL LIGHTGBM FITTING ---
    if "LightGBM" in model_names_cv and lgbm_installed and X_train_ml is not None and not X_train_ml.empty:
        with st.spinner("Fitting LightGBM..."):
             try:
                 model = lgb.LGBMRegressor(n_estimators=150, random_state=42, n_jobs=1)
                 model.fit(X_train_ml, y_train_ml)
                 final_models['LightGBM'] = model
                 
                 # Predict Test Set (simple)
                 test_preds = model.predict(X_test_ml)
                 test_predictions['LightGBM'] = pd.Series(test_preds, index=X_test_ml.index)

                 # Predict Future (Bootstrapping required) - Reusing logic setup for XGBoost
                 # FIX: Create template DataFrame for future prediction dates
                 X_future_ml_lgbm = pd.DataFrame(index=future_prediction_dates, columns=X_train_ml.columns).fillna(0)

                 # Populate fixed time features for the future
                 X_future_ml_lgbm['dayofweek'] = X_future_ml_lgbm.index.dayofweek
                 X_future_ml_lgbm['quarter'] = X_future_ml_lgbm.index.quarter
                 X_future_ml_lgbm['month'] = X_future_ml_lgbm.index.month
                 X_future_ml_lgbm['year'] = X_future_ml_lgbm.index.year
                 X_future_ml_lgbm['dayofyear'] = X_future_ml_lgbm.index.dayofyear

                 # Initialize rolling features in the first future step (simplification: last known rolling value)
                 if rolling_cols and not X_test_ml.empty:
                    last_known_X = X_test_ml.iloc[-1]
                    for col in rolling_cols:
                        if col in last_known_X.index and col in X_future_ml_lgbm.columns:
                            X_future_ml_lgbm.loc[X_future_ml_lgbm.index[0], col] = last_known_X[col]
                            
                 # 1. Populate initial lags for the first future step 
                 if not X_future_ml_lgbm.empty:
                    for lag_idx in range(1, lag_features + 1):
                        col_name = f'lag_{lag_idx}'
                        if col_name in X_future_ml_lgbm.columns:
                            if lag_idx <= len(initial_lags_series):
                                X_future_ml_lgbm.loc[X_future_ml_lgbm.index[0], col_name] = initial_lags_series[lag_idx - 1]
                            else:
                                X_future_ml_lgbm.loc[X_future_ml_lgbm.index[0], col_name] = 0 


                 future_preds_lgbm = []
                 for t in range(m_future):
                     X_step = X_future_ml_lgbm.iloc[t:t+1]
                     pred_step = model.predict(X_step)[0]
                     future_preds_lgbm.append(pred_step)

                     if t < m_future - 1:
                         # 1. Shift lags forward
                         for lag in range(lag_features, 1, -1):
                             lag_col_current = f'lag_{lag}'
                             lag_col_previous = f'lag_{lag-1}'
                             if lag_col_current in X_future_ml_lgbm.columns and lag_col_previous in X_future_ml_lgbm.columns:
                                X_future_ml_lgbm.loc[X_future_ml_lgbm.index[t+1], lag_col_current] = X_future_ml_lgbm.loc[X_future_ml_lgbm.index[t], lag_col_previous]
                         
                         # 2. Set lag_1 to the predicted value
                         if 'lag_1' in X_future_ml_lgbm.columns:
                            X_future_ml_lgbm.loc[X_future_ml_lgbm.index[t+1], 'lag_1'] = pred_step
                         
                         # 3. If exog is based on lag_1 squared, update that too
                         if use_exog and exog_name in X_future_ml_lgbm.columns:
                            X_future_ml_lgbm.loc[X_future_ml_lgbm.index[t+1], exog_name] = pred_step**2
                            
                 future_predictions['LightGBM'] = pd.Series(future_preds_lgbm, index=future_prediction_dates)
                 model_residuals['LightGBM'] = pd.Series(y_train_ml.values - model.predict(X_train_ml), index=y_train_ml.index)
                 
             except Exception as e:
                 st.warning(f"LightGBM failed: {e}")
                 [d.pop('LightGBM', None) for d in [test_predictions, future_predictions, final_models, model_residuals]]
    # --- END NEW ML FITTING ---

    st.markdown("---"); st.subheader("Test Set Performance & Future Forecast Visualization")
    fig_fc, ax_fc = plt.subplots(figsize=(14, 7))
    ax_fc.plot(train_data.index, train_data, label='Training Data', color='dimgray', lw=1, alpha=0.7)
    # Ajustar para que el test data plot solo use los puntos que realmente tienen features para ML, si es relevante
    if X_test_ml is not None and not X_test_ml.empty:
        test_data_actual = test_data.reindex(X_test_ml.index)
    else:
        test_data_actual = test_data
        
    ax_fc.plot(test_data_actual.index, test_data_actual, label=f'Actual Test ({data_type.capitalize()})', color='blue', marker='.', linestyle='-', ms=5)
    
    colors = {'ARIMA':'#1f77b4','ARIMAX':'#ff7f0e','SARIMAX':'#2ca02c','ETS':'#d62728','VAR':'#9467bd', 'Prophet':'#8c564b', 'TBATS':'#e377c2', 'XGBoost': '#17becf', 'LightGBM': '#bcbd22'}
    plot_successful = False
    
    if test_predictions:
        for name, preds in test_predictions.items():
            if preds is not None and isinstance(preds, pd.Series) and not preds.empty:
                preds_aligned = preds.reindex(test_data_actual.index).dropna() # Usar test_data_actual index para ML models
                if not preds_aligned.empty: ax_fc.plot(preds_aligned.index, preds_aligned, label=f'{name} Test', linestyle='--', color=colors.get(name, '#7f7f7f'), lw=1.5); plot_successful = True
    
    if future_predictions:
        for name, preds in future_predictions.items():
             if preds is not None and isinstance(preds, pd.Series) and not preds.empty:
                 preds_aligned = preds.reindex(future_prediction_dates).dropna()
                 if not preds_aligned.empty: ax_fc.plot(preds_aligned.index, preds_aligned, label=f'{name} Fcst', linestyle=':', color=colors.get(name, '#7f7f7f'), lw=1.5); plot_successful = True
    
    if not plot_successful: st.warning("No valid model predictions were generated to plot.")
    
    ax_fc.set_title(f"Forecasts for {selected_entity_model} ({data_type.capitalize()})"); ax_fc.set_xlabel("Date"); ax_fc.set_ylabel(f"Daily {data_type.capitalize()}");
    ax_fc.legend(loc='center left', bbox_to_anchor=(1.02, 0.5)); ax_fc.grid(True,ls='--',lw=0.5);
    if not train_data.empty: ax_fc.axvline(train_data.index[-1], color='gray', linestyle=':', lw=2, label='_nolegend_');
    plt.tight_layout(rect=[0,0,0.85,1]); st.pyplot(fig_fc); plt.close(fig_fc)

    st.subheader("Detailed Results: Test Performance & Future Forecast")
    results_df = pd.DataFrame()
    format_str = "{:,.4f}%" if data_type == 'returns' else "{:,.2f}"
    
    # Usamos el índice ajustado para la columna Actual
    if not test_data_actual.empty and future_prediction_dates is not None:
         combined_index = test_data_actual.index.union(future_prediction_dates)
         results_df = pd.DataFrame(index=combined_index);
         results_df['Actual'] = test_data_actual.reindex(combined_index)
         all_preds = {}
         for name, p in test_predictions.items():
             if p is not None: all_preds[f'{name}_Test'] = p
         for name, p in future_predictions.items():
             if p is not None: all_preds[f'{name}_Fcst'] = p
         for col_name, preds_series in all_preds.items():
             if isinstance(preds_series, pd.Series): results_df[col_name] = preds_series.reindex(combined_index)
         st.dataframe(results_df.style.format(format_str, na_rep="").highlight_null(color='rgba(0,0,0,0.05)'))
    else: st.warning("Could not generate combined results table (missing test data or future dates).")

    st.subheader(f"Forecast Accuracy (MAPE on Test Set - {data_type.capitalize()})")
    mape_res = {}; mape_df = pd.DataFrame()
    if test_predictions and not test_data_actual.empty:
        for name, preds in test_predictions.items():
            if preds is not None and isinstance(preds, pd.Series):
                 # Aseguramos que la alineación se haga con el test_data_actual
                 preds_aligned = preds.reindex(test_data_actual.index).dropna()
                 actuals_aligned = test_data_actual.reindex(preds_aligned.index).dropna()
                 if not preds_aligned.empty and len(preds_aligned) == len(actuals_aligned):
                     try: mape_val = mape(actuals_aligned, preds_aligned); mape_res[name] = mape_val if pd.notna(mape_val) else np.nan
                     except Exception as e_mape: st.caption(f"MAPE calculation failed for {name}: {e_mape}"); mape_res[name] = np.nan
                 elif not preds_aligned.empty: st.caption(f"MAPE skipped for {name}: Mismatch in lengths (Preds: {len(preds_aligned)}, Actuals: {len(actuals_aligned)})."); mape_res[name] = np.nan
        valid_mape = {k:v for k,v in mape_res.items() if pd.notna(v)}
        if valid_mape:
            mape_df = pd.DataFrame(list(valid_mape.items()),columns=['Model','Test MAPE (%)']).sort_values('Test MAPE (%)').reset_index(drop=True)
            st.dataframe(mape_df.style.format({'Test MAPE (%)':'{:.2f}%'}))
            if not mape_df.empty: st.success(f"Best model based on Test MAPE: **{mape_df.iloc[0]['Model']}**")
            else: st.info("Could not determine best model from MAPE.")
        else: st.info("Could not calculate valid Test MAPE for any model.")
    else: st.info("MAPE calculation skipped (no test predictions or test data).")

    st.subheader("Future Forecast Summary")
    forecast_summary_df = pd.DataFrame()
    if future_predictions and future_prediction_dates is not None:
        forecast_summary_df = pd.DataFrame(index=future_prediction_dates)
        models_with_future_fc = 0
        for name, preds in future_predictions.items():
             if preds is not None and isinstance(preds, pd.Series):
                  aligned_preds = preds.reindex(future_prediction_dates)
                  if not aligned_preds.isnull().all():
                     forecast_summary_df[name] = aligned_preds; models_with_future_fc += 1
        if models_with_future_fc > 0: st.dataframe(forecast_summary_df.style.format(format_str, na_rep=""))
        else: st.info("No successful future forecasts available to display.")
    else: st.info("Could not generate future forecast summary (missing predictions or dates).")

    st.markdown("---"); st.subheader("Residual Diagnostics")
    def run_residual_diagnostics(model_name, residuals, train_idx):
        if residuals is None: return
        if isinstance(residuals, dict): # Para VAR
            st.markdown(f"--- **{model_name} Residuals (VAR)** ---")
            if not residuals: st.warning(f"No residuals found for VAR model {model_name}.")
            else: [run_residual_diagnostics(f"{model_name} Var({var_name})", r, train_idx) for var_name, r in residuals.items() if r is not None]
            return
        res_aligned = None
        if isinstance(residuals, pd.Series):
             try: 
                 # Ajuste de índice para los residuos de ML, que pueden ser más cortos
                 if model_name in ["XGBoost", "LightGBM"]:
                      res_aligned = residuals.reindex(y_train_ml.index).dropna()
                 else:
                      res_aligned = residuals.reindex(train_idx).dropna()
             except Exception: pass # Silenciar si falla el reindex
        if res_aligned is None or res_aligned.empty: st.warning(f"Could not align or find valid residuals for {model_name}."); return
        st.markdown(f"--- **{model_name} Residuals** ---")
        if res_aligned.nunique() <= 1: st.warning(f"{model_name} residuals appear to be constant."); fig, ax = plt.subplots(); res_aligned.plot(ax=ax, title=f"{model_name} Constant Residuals"); st.pyplot(fig); plt.close(fig); return
        try:
            fig_res, ax_res = plt.subplots(1, 2, figsize=(10, 3.5));
            res_aligned.plot(title="Residuals Over Time", ax=ax_res[0], grid=True);
            res_aligned.plot(kind='hist', bins=30, title="Residual Distribution", ax=ax_res[1], grid=True, density=True);
            try: res_aligned.plot(kind='kde', ax=ax_res[1], secondary_y=False, color='red', lw=1.5)
            except Exception: pass # Silenciar si falla KDE
            plt.tight_layout(); st.pyplot(fig_res); plt.close(fig_res)
        except Exception as e_res_plot: st.error(f"Could not plot residuals for {model_name}: {e_res_plot}")
        min_res_len_diag = 20
        if len(res_aligned) < min_res_len_diag: st.warning(f"Fewer than {min_res_len_diag} residuals for {model_name}, diagnostic tests skipped."); return
        try:
             lags_lb = min(int(len(res_aligned)*0.2), 40); lags_lb = max(1, lags_lb)
             lb_test = acorr_ljungbox(res_aligned, lags=[lags_lb], return_df=True)
             p_val_lb = lb_test['lb_pvalue'].iloc[0] if not lb_test.empty and pd.notna(lb_test['lb_pvalue'].iloc[0]) else np.nan
             lb_result = f"p={p_val_lb:.4f} (lag {lags_lb}). {'Likely Independent (p>0.05)' if p_val_lb>0.05 else 'Likely Dependent (p<=0.05)' if pd.notna(p_val_lb) else 'Test Failed/Skipped'}"
             st.write(f"**Ljung-Box (Autocorrelation):** {lb_result}")
        except Exception as e_lb: st.error(f"Ljung-Box test failed for {model_name}: {e_lb}")
        try:
             if res_aligned.var() < 1e-10: st.write("**Normality:** Skipped (Residuals variance near zero).")
             else:
                  stat_norm, p_val_norm = normaltest(res_aligned)
                  norm_result = f"p={p_val_norm:.4f}. {'Likely Normal (p>0.05)' if p_val_norm>0.05 else 'Likely Not Normal (p<=0.05)' if pd.notna(p_val_norm) else 'Test Failed/Skipped'}"
                  st.write(f"**Normality (D'Agostino-Pearson):** {norm_result}")
        except Exception as e_norm: st.error(f"Normality test failed for {model_name}: {e_norm}")
    if not model_residuals: st.warning("No model residuals available for diagnostics.")
    else:
        for name, res in model_residuals.items():
            if res is not None: run_residual_diagnostics(name, res, train_data.index)
            else: st.caption(f"Residuals not available for model: {name}")

    st.markdown("---"); st.subheader("Download Results")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    file_prefix = clean_file_name(selected_entity_model); data_suffix = data_type
    outlier_suffix = "_outliers_treated" if st.session_state.get('apply_outlier_treatment', False) else ""
    with col_dl1:
        if not results_df.empty:
             csv_buf_comb=io.StringIO(); results_df.to_csv(csv_buf_comb);
             st.download_button(label="DL Combined Results", data=csv_buf_comb.getvalue(),
                                file_name=f"{file_prefix}_{data_suffix}{outlier_suffix}_combined_results.csv", mime="text/csv", key="dl_combined")
        else: st.caption("No combined results.")
    with col_dl2:
        if not mape_df.empty:
            csv_buf_mape=io.StringIO(); mape_df.to_csv(csv_buf_mape, index=False);
            st.download_button(label="DL Test MAPE", data=csv_buf_mape.getvalue(),
                               file_name=f"{file_prefix}_{data_suffix}{outlier_suffix}_test_mape.csv", mime="text/csv", key="dl_mape")
        else: st.caption("No MAPE results.")
    with col_dl3:
        if not forecast_summary_df.empty:
             csv_buf_fc=io.StringIO(); forecast_summary_df.to_csv(csv_buf_fc);
             st.download_button(label="DL Future Forecasts", data=csv_buf_fc.getvalue(),
                                file_name=f"{file_prefix}_{data_suffix}{outlier_suffix}_future_forecasts.csv", mime="text/csv", key="dl_forecasts")
        else: st.caption("No future forecasts.")


# --- Page Definition Functions (Portfolio App - Create/Edit) ---
# --- PÁGINA "Create/Edit Portfolios" - VERSIÓN CON RENOMBRADO Y ELIMINACIÓN CORREGIDOS ---

def page_create_portfolio():
    st.header("📈 Create or Edit Portfolios")

    # Inicializar claves de estado necesarias para esta página si no existen
    if 'delete_confirmation' not in st.session_state:
        st.session_state.delete_confirmation = None
    if 'portfolio_name_input' not in st.session_state:
        st.session_state.portfolio_name_input = ""
    if 'portfolio_tickers_input' not in st.session_state:
        st.session_state.portfolio_tickers_input = ""
    
    def sync_form_to_selection():
        # Función de callback que se ejecuta al cambiar la selección del dropdown.
        # Su única misión es poblar el formulario con los datos del portafolio seleccionado.
        st.session_state.delete_confirmation = None # Resetear confirmación de borrado
        
        selected = st.session_state.portfolio_select
        if selected != "(Create New Portfolio)" and selected in st.session_state.portfolios:
            # Sincronizar el nombre en el campo de texto editable
            st.session_state.portfolio_name_input = selected
            # Cargar los tickers y pesos en el área de texto
            weights_data = st.session_state.portfolios[selected]
            tickers_str = "\n".join(f"{ticker}: {weight:.4f}" for ticker, weight in weights_data.items())
            st.session_state.portfolio_tickers_input = tickers_str
        else:
            # Limpiar campos si se va a crear uno nuevo
            st.session_state.portfolio_name_input = ""
            st.session_state.portfolio_tickers_input = ""

    portfolio_names = list(st.session_state.portfolios.keys())
    options = ["(Create New Portfolio)"] + sorted(portfolio_names)
    
    st.selectbox("Select Portfolio to Edit, or Create New:", options, key="portfolio_select", on_change=sync_form_to_selection)

    st.markdown("---")

    col_form, col_display = st.columns([2, 1])

    with col_form:
        # La caja de texto del nombre AHORA es independiente y mantiene su propio estado
        current_name = st.text_input("Portfolio Name:", key="portfolio_name_input").strip()

        st.subheader(f"Define/Edit Assets & Weights for: '{current_name or '(no name)'}'")

        ticker_input_str = st.text_area(
            "Enter Assets (format: TICKER: WEIGHT, one per line or comma-separated):",
            key="portfolio_tickers_input",
            height=250,
            help="Example:\nAAPL: 0.4\nMSFT: 0.6"
        )
        
        # Parseo de los tickers y pesos en tiempo real para feedback del usuario
        weights_parsed = {}
        parse_errors = []
        if ticker_input_str:
            entries = re.split(r'[,\n]+', ticker_input_str)
            for entry in entries:
                if ':' in entry:
                    parts = entry.split(':')
                    ticker = parts[0].strip().upper()
                    try:
                        weight = float(parts[1].strip())
                        if ticker and weight >= 0:
                            weights_parsed[ticker] = weight
                    except (ValueError, IndexError):
                        if entry.strip(): parse_errors.append(f"Could not parse '{entry.strip()}'.")
                elif entry.strip(): parse_errors.append(f"Missing ':' in '{entry.strip()}'.")
        
        for error in parse_errors: st.warning(error, icon="⚠️")

        with st.form("portfolio_form"):
            total_weight = sum(weights_parsed.values())
            st.caption(f"Parsed {len(weights_parsed)} assets. Current total weight: {total_weight:.4f}")

            # --- Botones de Acción ---
            save_col, delete_col = st.columns(2)
            with save_col:
                submitted_save = st.form_submit_button("💾 Save Portfolio", type="primary", use_container_width=True, disabled=not (current_name and weights_parsed))
            with delete_col:
                is_existing_portfolio = st.session_state.portfolio_select != "(Create New Portfolio)"
                submitted_delete = st.form_submit_button(f"🗑️ Delete '{st.session_state.portfolio_select}'", use_container_width=True, disabled=not is_existing_portfolio)

            # --- Lógica de Guardado (corregida) ---
            if submitted_save:
                original_name = st.session_state.portfolio_select
                is_renaming = is_existing_portfolio and current_name != original_name

                # Validaciones
                if not current_name:
                    st.error("Portfolio name cannot be empty.")
                elif not is_existing_portfolio and current_name in st.session_state.portfolios:
                    st.error(f"A portfolio named '{current_name}' already exists.")
                elif is_renaming and current_name in st.session_state.portfolios:
                    st.error(f"Cannot rename to '{current_name}'. A portfolio with that name already exists.")
                elif not np.isclose(total_weight, 1.0, atol=0.001):
                    st.error(f"❌ Weights must sum to 1.0. Current sum is {total_weight:.4f}. Please adjust.")
                else:
                    # Acciones
                    if is_renaming:
                        del st.session_state.portfolios[original_name]
                    
                    normalized_weights = {t: w / total_weight for t, w in weights_parsed.items()}
                    st.session_state.portfolios[current_name] = normalized_weights
                    success, error_message = save_portfolios_to_file(st.session_state.portfolios)

                    if success:
                        st.success(f"✅ Portfolio '{current_name}' saved successfully!")
                        st.session_state.portfolio_select = current_name
                        time.sleep(1)
                        st.rerun()
                    else: st.error(f"🚨 SAVE FAILED: {error_message}")
            
            # --- Lógica de Eliminación ---
            if submitted_delete:
                portfolio_to_delete = st.session_state.portfolio_select
                if st.session_state.delete_confirmation != portfolio_to_delete:
                    st.session_state.delete_confirmation = portfolio_to_delete
                    st.warning(f"Are you sure? Click delete again to confirm deletion of '{portfolio_to_delete}'.")
                else:
                    del st.session_state.portfolios[portfolio_to_delete]
                    success, error_message = save_portfolios_to_file(st.session_state.portfolios)
                    
                    if success:
                        st.success(f"✅ Portfolio '{portfolio_to_delete}' has been deleted.")
                        st.session_state.delete_confirmation = None
                        st.session_state.portfolio_select = "(Create New Portfolio)"
                        time.sleep(1)
                        st.rerun()
                    else: st.error(f"🚨 DELETE FAILED: {error_message}")
    
    # Columna de la derecha para mostrar los portafolios guardados
    with col_display:
        st.subheader("Saved Portfolios")
        if not st.session_state.portfolios:
            st.info("No portfolios defined yet.")
        else:
            for name in sorted(st.session_state.portfolios.keys()):
                with st.expander(f"**{name}**", expanded=(name == st.session_state.portfolio_select)):
                    df = pd.DataFrame(st.session_state.portfolios[name].items(), columns=['Ticker', 'Weight'])
                    st.dataframe(df.style.format({'Weight': '{:.2%}'}), hide_index=True, use_container_width=True)

# --- FUNCIÓN DE PÁGINA DE RETORNOS DE PORTAFOLIO MODIFICADA ---
def page_view_portfolio_returns():
    st.header("📊 Portfolio Performance Viewer & Optimizer")
    if 'portfolios' not in st.session_state or not st.session_state.portfolios:
        st.warning("⚠️ No portfolios defined. Please go to the 'Create/Edit Portfolios' page first.")
        return
    portfolio_names = sorted(list(st.session_state.portfolios.keys()))
    selected_names = st.multiselect(
        "Select Portfolios to Analyze (select one for optimization):",
        options=portfolio_names,
        key="portfolio_view_select"
    )
    if not selected_names:
        st.info("Select one or more portfolios above to view their performance. Select **only one** to enable optimization.")
        return
    st.markdown("---")
    st.subheader("Select Analysis Period")
    today_pf = datetime.today().date()
    years_ago_pf = today_pf - timedelta(days=5*365)
    col_date1_pf, col_date2_pf = st.columns(2)
    with col_date1_pf:
        start_date_pf = st.date_input("Start Date", years_ago_pf, key="pf_view_start_date")
    with col_date2_pf:
        end_date_pf = st.date_input("End Date", today_pf, key="pf_view_end_date")
    if start_date_pf >= end_date_pf:
        st.error("Error: Start date must be before end date.")
        return
    combined_tickers = set()
    aggregate_weights = {}
    individual_portfolio_weights = {}
    analysis_title = ""
    portfolio_to_optimize = None
    if len(selected_names) == 1:
        portfolio_name = selected_names[0]
        analysis_title = f"Portfolio: {portfolio_name}"
        st.subheader(f"Performance Analysis for: '{portfolio_name}'")
        individual_portfolio_weights = st.session_state.portfolios[portfolio_name]
        combined_tickers = set(individual_portfolio_weights.keys())
        aggregate_weights = individual_portfolio_weights
        portfolio_to_optimize = portfolio_name
        st.dataframe(pd.DataFrame(list(aggregate_weights.items()), columns=['Ticker', 'Weight']).style.format({'Weight': '{:.2%}'}))
    elif len(selected_names) > 1:
        analysis_title = f"Aggregated: {', '.join(selected_names)}"
        st.subheader(f"Aggregated Performance Analysis for: {', '.join(selected_names)}")
        num_selected = len(selected_names)
        portfolio_contribution = 1.0 / num_selected
        temp_aggregate_weights = {}
        for name in selected_names:
            portfolio_def = st.session_state.portfolios.get(name, {})
            individual_portfolio_weights[name] = portfolio_def
            for ticker, weight in portfolio_def.items():
                combined_tickers.add(ticker)
                temp_aggregate_weights[ticker] = temp_aggregate_weights.get(ticker, 0) + (weight * portfolio_contribution)
        aggregate_weights = temp_aggregate_weights
        st.caption(f"Displaying aggregated performance with equal weight ({portfolio_contribution:.1%}) given to each selected portfolio.")
        agg_weights_df = pd.DataFrame(list(aggregate_weights.items()), columns=['Ticker', 'Aggregate Weight']).sort_values('Aggregate Weight', ascending=False)
        st.dataframe(agg_weights_df.style.format({'Aggregate Weight': '{:.2%}'}))
        agg_sum = sum(aggregate_weights.values())
        if not np.isclose(agg_sum, 1.0): st.warning(f"Aggregate Weight Sum: {agg_sum:.4f}")
        st.info("Optimization is only available when a single portfolio is selected.")
    tickers_to_fetch_pf = list(combined_tickers)
    if not tickers_to_fetch_pf:
         st.warning("No tickers found in the selected portfolio(s) to fetch data for.")
         return
    st.markdown("---")
    prices_df_pf = fetch_stock_prices_for_portfolio(tickers_to_fetch_pf, start_date_pf, end_date_pf)
    if prices_df_pf is None: st.warning("Price fetching returned None."); return # Cambio para manejar None
    if prices_df_pf.empty:
         st.error("Failed to retrieve valid price data for *any* required tickers in the selected period. Cannot calculate performance or optimize.")
         st.info("Troubleshooting: \n1. Verify ticker symbols (e.g. CEPU.BA for BCBA). \n2. Check date range. \n3. Try a shorter, more recent date range.")
         return
    st.markdown("---")
    portfolio_daily_returns_pf, portfolio_cumulative_returns_pf, renormalized_info = calculate_portfolio_performance(
        prices_df_pf,
        aggregate_weights
    )
    
    # Initialize metric variables outside the 'if' block for the chat
    total_return, annualized_return, annualized_volatility, hist_var, max_dd = 0.0, 0.0, 0.0, 0.0, 0.0

    if renormalized_info: st.warning(renormalized_info)
    if portfolio_cumulative_returns_pf is not None:
        st.subheader(f"Performance Chart: '{analysis_title}'")
        st.line_chart(portfolio_cumulative_returns_pf)
        chart_start_date_str = portfolio_cumulative_returns_pf.index.min().strftime('%Y-%m-%d') if not portfolio_cumulative_returns_pf.empty else "N/A"
        st.caption(f"Cumulative Return (1 = Initial Value on {chart_start_date_str})")
        st.subheader("Key Performance Metrics")
        col_m1, col_m2, col_m3 = st.columns(3)
        total_return = (portfolio_cumulative_returns_pf.iloc[-1] - 1) * 100 if not portfolio_cumulative_returns_pf.empty else 0
        num_days = (end_date_pf - start_date_pf).days; num_years = num_days / 365.25
        last_cum_return = portfolio_cumulative_returns_pf.iloc[-1] if not portfolio_cumulative_returns_pf.empty else 1.0
        annualized_return = ((last_cum_return)**(1/num_years) - 1) * 100 if num_years > (1/365.25) and last_cum_return > 0 else 0
        annualized_volatility = portfolio_daily_returns_pf.std() * np.sqrt(252) * 100 if portfolio_daily_returns_pf is not None and not portfolio_daily_returns_pf.empty else 0
        with col_m1: st.metric(label="Total Return", value=f"{total_return:.2f}%")
        with col_m2: st.metric(label="Annualized Return (CAGR)", value=f"{annualized_return:.2f}%")
        with col_m3: st.metric(label="Annualized Volatility (Std Dev)", value=f"{annualized_volatility:.2f}%", help="Based on daily returns, 252 trading days/year.")
        
        # --- SECCIÓN DE VAR AGREGADA ---
        st.subheader("Value at Risk (VaR)")
        confidence_level_var = st.slider(
            "Confidence Level for VaR", 
            min_value=0.90, 
            max_value=0.99, 
            value=0.95, 
            step=0.01,
            format="%.2f",
            key="var_confidence_slider",
            help="The confidence level for the VaR calculation. A 95% level means we are 95% confident the daily loss will not exceed the calculated VaR."
        )
        if portfolio_daily_returns_pf is not None and not portfolio_daily_returns_pf.empty:
            hist_var, para_var = calculate_var(portfolio_daily_returns_pf, confidence_level=confidence_level_var)
            if hist_var is not None and para_var is not None:
                col_var1, col_var2 = st.columns(2)
                with col_var1:
                    st.metric(
                        label=f"Historical VaR ({confidence_level_var:.0%})",
                        value=f"{hist_var*100:.2f}%",
                        help="Based on the actual worst-case daily returns from the historical data for this period."
                    )
                with col_var2:
                    st.metric(
                        label=f"Parametric VaR ({confidence_level_var:.0%})",
                        value=f"{para_var*100:.2f}%",
                        help="Calculated assuming returns follow a normal distribution. May underestimate risk in volatile markets."
                    )
                st.caption(f"Interpretación: Con {confidence_level_var:.0%} de confianza, el portfolio's maximum loss in a single day is not expected to exceed these values.")
            else:
                st.warning("Could not calculate VaR. Not enough daily return data available for the selected period.")
        
        # Calculate Drawdowns here so 'max_dd' is available for the Chat section
        drawdown_series, max_dd, max_dd_date = calculate_drawdowns(portfolio_cumulative_returns_pf)

        with st.expander("View Detailed Data Tables"):
            if portfolio_daily_returns_pf is not None:
                st.markdown("**Calculated Daily Returns** (Individual or Aggregated)")
                st.dataframe(portfolio_daily_returns_pf.to_frame(name="Daily Return").style.format("{:.4%}"))
            st.markdown("**Calculated Cumulative Returns** (Individual or Aggregated)")
            st.dataframe(portfolio_cumulative_returns_pf.to_frame(name="Cumulative Return").style.format("{:.4f}"))
            st.markdown("**Fetched Stock Prices (Used for Calculation)**")
            st.dataframe(prices_df_pf.style.format("{:.2f}"))
            if len(selected_names) > 1:
                st.markdown("**Original Portfolio Definitions (Selected for Aggregation)**")
                for name, weights in individual_portfolio_weights.items():
                    st.caption(f"**{name}**")
                    df_orig = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Weight'])
                    st.dataframe(df_orig.style.format({'Weight': '{:.2%}'}), hide_index=True)
    else:
        st.error("Could not calculate portfolio performance after fetching data.")
        if prices_df_pf is not None and not prices_df_pf.empty:
             with st.expander("View Fetched Price Data (for debugging)"): st.dataframe(prices_df_pf)
        
        # Set dummy values for chat context if calculation failed
        total_return, annualized_return, annualized_volatility = 0.0, 0.0, 0.0
        hist_var, max_dd = 0.0, 0.0 # Using 0.0 ensures variables exist for the Chat section.


    st.markdown("---")
    st.subheader("🚀 Portfolio Optimization")
    if not pypfopt_installed:
        st.error("Optimization requires the 'PyPortfolioOpt' library.")
        st.code("pip install PyPortfolioOpt")
        return
    if portfolio_to_optimize and not prices_df_pf.empty:
        st.write(f"Optimize weights for portfolio: **'{portfolio_to_optimize}'**")
        st.caption(f"Using historical price data from {start_date_pf.strftime('%Y-%m-%d')} to {end_date_pf.strftime('%Y-%m-%d')}.")
        prices_for_opt = prices_df_pf.copy()
        if len(prices_for_opt.columns) < 2:
             st.warning("Optimization requires at least 2 assets with valid price data.")
             return
        opt_objective = st.selectbox(
            "Optimization Objective:",
            options=[
                "Maximize Sharpe Ratio (Best Risk-Adjusted Return)",
                "Minimize Volatility (Lowest Risk)",
                "Maximize Quadratic Utility (Balance Risk/Return)",
            ],
            key="opt_objective_select",
            index=0
        )
        risk_free_rate = st.number_input(
            "Risk-Free Rate (Annualized, e.g., 0.02 for 2%)",
            min_value=0.0, max_value=0.2, value=0.02, step=0.005, format="%.4f",
            key="opt_risk_free",
            help="Used primarily for Sharpe Ratio calculation."
        )
        run_optimization = st.button("Calculate Optimized Weights", key="run_opt_button")
        if run_optimization:
            with st.spinner("Optimizing portfolio..."):
                try:
                    mu = expected_returns.mean_historical_return(prices_for_opt, frequency=252)
                    S = risk_models.sample_cov(prices_for_opt, frequency=252)
                    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                    if opt_objective == "Maximize Sharpe Ratio (Best Risk-Adjusted Return)":
                        ef.max_sharpe(risk_free_rate=risk_free_rate)
                    elif opt_objective == "Minimize Volatility (Lowest Risk)":
                        ef.min_volatility()
                    elif opt_objective == "Maximize Quadratic Utility (Balance Risk/Return)":
                        risk_aversion_param = 2
                        ef.max_quadratic_utility(risk_aversion=risk_aversion_param)
                        st.caption(f"(Using Risk Aversion = {risk_aversion_param})")
                    weights_optimized = ef.clean_weights()
                    st.success("Optimization complete!")
                    st.subheader("Optimized Portfolio Weights")
                    weights_df = pd.DataFrame(list(weights_optimized.items()), columns=['Ticker', 'Optimized Weight'])
                    weights_df = weights_df[weights_df['Optimized Weight'] > 1e-5] # Filtrar pesos muy pequeños
                    weights_df = weights_df.sort_values('Optimized Weight', ascending=False).reset_index(drop=True)
                    st.dataframe(weights_df.style.format({'Optimized Weight': '{:.2%}'}))
                    st.subheader("Expected Performance of Optimized Portfolio")
                    try:
                        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                        perf_cols = st.columns(3)
                        with perf_cols[0]: st.metric("Expected Annual Return", f"{expected_annual_return:.2%}")
                        with perf_cols[1]: st.metric("Annual Volatility", f"{annual_volatility:.2%}")
                        with perf_cols[2]: st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    except Exception as e_perf:
                        st.warning(f"Could not calculate expected performance metrics: {e_perf}")
                except ValueError as ve:
                     st.error(f"Optimization Error: {ve}")
                     st.info("This often happens if price data has issues (e.g., constant price for an asset) or if the period is too short.")
                except Exception as e:
                     st.error(f"An unexpected error occurred during optimization: {e}")
                     st.error(traceback.format_exc())
    elif len(selected_names) > 1:
        st.info("Select only **one** portfolio from the top multiselect box to enable optimization features.")
    elif prices_df_pf.empty:
         st.warning("Cannot optimize because no valid price data was loaded for the selected portfolio and period.")
    else:
        st.info("Optimization is available when a single portfolio with valid price data is selected.")
    
    # --- NUEVO: Chat de Análisis de Portafolio ---
    st.markdown("---")
    st.header("🤖 Chat de Análisis de Portafolio")

    if not selected_names or portfolio_daily_returns_pf is None:
        st.info("Selecciona un portafolio arriba y asegura que los datos de rendimiento se hayan cargado para activar el chat de análisis.")
        return

    # Usamos el primer portafolio seleccionado como clave para el historial de chat
    chat_key = selected_names[0]
    if chat_key not in st.session_state.portfolio_chat_messages:
        st.session_state.portfolio_chat_messages[chat_key] = []

    st.markdown(f"Conversando sobre **'{analysis_title}'**. Puedes subir documentos adicionales para contextualizar tus preguntas.")

    # Widgets para subir contexto adicional
    chat_uploaded_pdfs = st.file_uploader(
        "Sube PDFs (informes, noticias relevantes)", type="pdf", accept_multiple_files=True, key=f"chat_pdf_{chat_key}"
    )

    # Mostrar historial de chat para este portafolio
    for message in st.session_state.portfolio_chat_messages[chat_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Pregunta sobre el rendimiento de '{analysis_title}'..."):
        st.session_state.portfolio_chat_messages[chat_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Combinando datos y analizando..."):
                # 1. Recopilar contexto CUANTITATIVO (los datos que ya calculaste)
                quantitative_context = f"**Resumen de Rendimiento del Portafolio '{analysis_title}'**\n"
                
                # Aseguramos que las variables de métricas existan
                if 'total_return' in locals():
                    quantitative_context += f"- Retorno Total: {total_return:.2f}%\n"
                    quantitative_context += f"- Retorno Anualizado (CAGR): {annualized_return:.2f}%\n"
                    quantitative_context += f"- Volatilidad Anualizada: {annualized_volatility:.2f}%\n"
                
                # Check for VaR and Drawdown results
                if 'hist_var' in locals() and hist_var is not None and hist_var != 0.0:
                    quantitative_context += f"- VaR Histórico (95%): Pérdida diaria no debería exceder {hist_var*100:.2f}%\n"
                
                if 'max_dd' in locals() and max_dd != 0:
                    quantitative_context += f"- Máximo Drawdown: {max_dd:.2%}\n"
                
                quantitative_context += "**Composición del Portafolio:**\n"
                for ticker, weight in aggregate_weights.items():
                    quantitative_context += f"- {ticker}: {weight:.2%}\n"

                # 2. Recopilar contexto CUALITATIVO (nuevos documentos)
                qualitative_context = ""
                if chat_uploaded_pdfs:
                    for pdf in chat_uploaded_pdfs:
                        qualitative_context += f"--- INICIO DOCUMENTO ADICIONAL: {pdf.name} ---\n"
                        qualitative_context += extract_text_from_pdf(pdf)
                        qualitative_context += f"\n--- FIN DOCUMENTO ADICIONAL: {pdf.name} ---\n\n"

                # 3. Construir el prompt final
                system_prompt = """
                Eres un asesor de inversiones IA. Tu tarea es responder a la pregunta del usuario combinando dos fuentes de información:
                1.  **Datos Cuantitativos:** El rendimiento histórico y la composición del portafolio que te proporcionaré.
                2.  **Datos Cualitativos:** El contenido de cualquier documento adicional que el usuario haya subido.
                Cuando el usuario pregunte '¿qué hago con la acción X?', debes analizar su peso en el portafolio, su rendimiento implícito en los datos y cualquier noticia relevante en los documentos.
                Proporciona una recomendación razonada (ej. 'considerar mantener', 'revisar debido a X riesgo', 'parece fuerte por Y motivo').
                **IMPORTANTE:** Siempre incluye un descargo de responsabilidad de que esto no es un consejo financiero financiero real.
                """
                final_prompt = f"{system_prompt}\n\n**DATOS CUANTITATIVOS DEL PORTAFOLIO:**\n{quantitative_context}\n\n**DATOS CUALITATIVOS ADICIONALES:**\n{qualitative_context}\n\n**Pregunta del Usuario:**\n{prompt}"

                # 4. Llamar a la API
                response = get_hf_response(
                    st.session_state.hf_api_key,
                    st.session_state.hf_model,
                    final_prompt,
                    st.session_state.hf_temp,
                    max_tokens=4096 # Damos más espacio para respuestas complejas
                )

                # 5. Mostrar respuesta
                if response:
                    message_placeholder.markdown(response)
                    st.session_state.portfolio_chat_messages[chat_key].append({"role": "assistant", "content": response})
                else:
                    message_placeholder.markdown("No pude procesar la solicitud. Revisa la API Key.")
                    st.session_state.portfolio_chat_messages[chat_key].append({"role": "assistant", "content": "Error."})
    
    
    
    
    
    
    
    
    
    
    

# --- NUEVA PÁGINA: GESTIÓN DE RIESGO ---
def page_risk_management():
    st.header("🛡️ Gestión de Riesgo del Portafolio")
    st.markdown("""
    Esta sección proporciona herramientas para analizar el riesgo de sus portafolios guardados. 
    Analice el Valor en Riesgo (VaR), las caídas (drawdowns) y la correlación entre activos.
    """)

    if 'portfolios' not in st.session_state or not st.session_state.portfolios:
        st.warning("⚠️ No hay portafolios definidos. Por favor, vaya a la página 'Crear/Editar Portafolios' primero.")
        return

    portfolio_names = sorted(list(st.session_state.portfolios.keys()))
    selected_name = st.selectbox(
        "Seleccione un Portafolio para Analizar:",
        options=portfolio_names,
        key="risk_portfolio_select"
    )
    
    if not selected_name:
        st.info("Seleccione un portafolio para comenzar el análisis de riesgo.")
        return

    st.markdown("---")
    st.subheader("Seleccionar Período de Análisis")
    today_risk = datetime.today().date()
    years_ago_risk = today_risk - timedelta(days=3*365)
    
    col_date1_risk, col_date2_risk = st.columns(2)
    with col_date1_risk:
        start_date_risk = st.date_input("Fecha de Inicio", years_ago_risk, key="risk_start_date")
    with col_date2_risk:
        end_date_risk = st.date_input("Fecha de Fin", today_risk, key="risk_end_date")

    if start_date_risk >= end_date_risk:
        st.error("Error: La fecha de inicio debe ser anterior a la fecha de fin.")
        return

    weights_dict = st.session_state.portfolios[selected_name]
    tickers_to_fetch = list(weights_dict.keys())

    if not tickers_to_fetch:
         st.warning("El portafolio seleccionado no tiene tickers.")
         return

    # Fetch data and calculate returns
    prices_df = fetch_stock_prices_for_portfolio(tickers_to_fetch, start_date_risk, end_date_risk)
    
    if prices_df is None or prices_df.empty:
         st.error("No se pudieron obtener datos de precios para los tickers en el período seleccionado. No se puede realizar el análisis de riesgo.")
         return
         
    daily_returns_portfolio, cumulative_returns_portfolio, renormalized_info = calculate_portfolio_performance(
        prices_df,
        weights_dict
    )

    if renormalized_info:
        st.warning(renormalized_info)

    if daily_returns_portfolio is None or cumulative_returns_portfolio is None:
        st.error("No se pudo calcular el rendimiento del portafolio. Verifique los datos.")
        return

    # --- 1. Value at Risk (VaR) Section ---
    st.markdown("---")
    st.subheader(f"1. Valor en Riesgo (VaR) para '{selected_name}'")
    st.markdown("""
    El VaR estima la pérdida potencial máxima de un portafolio en un período de tiempo (aquí, un día) dentro de un nivel de confianza dado.
    - **VaR Histórico:** Se basa en los peores rendimientos diarios que ocurrieron realmente en el pasado.
    - **VaR Paramétrico:** Asume que los rendimientos siguen una distribución normal (una campana de Gauss). Puede ser menos preciso si los mercados tienen "colas anchas" (eventos extremos más frecuentes de lo normal).
    """)
    
    confidence_level_risk = st.slider(
        "Nivel de Confianza para VaR", 
        min_value=0.90, 
        max_value=0.99, 
        value=0.95, 
        step=0.01,
        format="%.2f",
        key="risk_var_confidence_slider"
    )

    hist_var_risk, para_var_risk = calculate_var(daily_returns_portfolio, confidence_level=confidence_level_risk)

    if hist_var_risk is not None and para_var_risk is not None:
        col_var_risk1, col_var_risk2 = st.columns(2)
        with col_var_risk1:
            st.metric(
                label=f"VaR Histórico ({confidence_level_risk:.0%})",
                value=f"{hist_var_risk*100:.2f}%"
            )
        with col_var_risk2:
             st.metric(
                label=f"VaR Paramétrico ({confidence_level_risk:.0%})",
                value=f"{para_var_risk*100:.2f}%"
            )
        st.caption(f"Interpretación: Con un {confidence_level_risk:.0%} de confianza, no se espera que la pérdida máxima del portafolio en un solo día supere estos valores.")
    else:
        st.warning("No se pudo calcular el VaR. No hay suficientes datos de rendimiento diario para el período seleccionado.")

    # --- 2. Drawdown Analysis ---
    st.markdown("---")
    st.subheader(f"2. Análisis de Caídas (Drawdowns) para '{selected_name}'")
    st.markdown("""
    Un drawdown es una caída desde un pico hasta un valle en el valor de un portafolio. El **Máximo Drawdown** es la mayor pérdida porcentual que el portafolio ha experimentado en el período. Es un indicador clave del riesgo de pérdida que un inversor habría enfrentado.
    """)
    
    drawdown_series, max_dd, max_dd_date = calculate_drawdowns(cumulative_returns_portfolio)

    if drawdown_series is not None and not drawdown_series.empty:
        col_dd1, col_dd2 = st.columns(2)
        with col_dd1:
            st.metric(
                label="Máximo Drawdown",
                value=f"{max_dd:.2%}",
                help=f"La peor caída de pico a valle ocurrió alrededor de {max_dd_date.strftime('%Y-%m-%d') if max_dd_date else 'N/A'}."
            )
        
        st.line_chart(drawdown_series.multiply(100)) # Multiply by 100 for better display
        st.caption("Gráfico de Drawdown (%) a lo largo del tiempo. Valores más bajos indican mayores pérdidas desde el último pico.")
    else:
        st.warning("No se pudo calcular el análisis de Drawdown.")


    # --- 3. Correlation Matrix ---
    st.markdown("---")
    st.subheader(f"3. Matriz de Correlación de Activos en '{selected_name}'")
    st.markdown("""
    La matriz de correlación muestra cómo se mueven los precios de los activos entre sí. Es fundamental para entender la diversificación.
    - **Valores cercanos a +1 (verde oscuro):** Los activos se mueven fuertemente en la misma dirección.
    - **Valores cercanos a -1 (rojo oscuro):** Los activos se mueven en direcciones opuestas (buena diversificación).
    - **Valores cercanos a 0 (blanco/amarillo):** Hay poca o ninguna relación en sus movimientos.
    """)
    
    # Use only tickers that are actually in the prices_df columns
    valid_tickers_corr = [t for t in tickers_to_fetch if t in prices_df.columns]
    if len(valid_tickers_corr) > 1:
        asset_returns = prices_df[valid_tickers_corr].pct_change().dropna()
        corr_matrix = asset_returns.corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(max(6, len(corr_matrix.columns)*0.8), max(5, len(corr_matrix.columns)*0.6)))
        cax = ax_corr.matshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        fig_corr.colorbar(cax)
        
        ax_corr.set_xticks(np.arange(len(corr_matrix.columns)))
        ax_corr.set_yticks(np.arange(len(corr_matrix.columns)))
        ax_corr.set_xticklabels(corr_matrix.columns, rotation=90, ha="center")
        ax_corr.set_yticklabels(corr_matrix.columns)
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax_corr.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black", fontsize=8)
        
        ax_corr.set_title(f"Matriz de Correlación de Rendimientos Diarios\n({start_date_risk.strftime('%Y-%m-%d')} a {end_date_risk.strftime('%Y-%m-%d')})", pad=20)
        fig_corr.tight_layout(pad=1.5)
        st.pyplot(fig_corr)
        plt.close(fig_corr) # Close plot to free memory
    else:
        st.info("La matriz de correlación requiere al menos 2 activos en el portafolio con datos de precios válidos.")


# --- NUEVA PÁGINA DE CHAT (REEMPLAZA A EVENT ANALYZER) ---
def page_investment_insights_chat():
    st.header("🤖 Chat de Análisis Cualitativo")
    st.markdown("""
    Sube informes, noticias o pega texto para analizar. Pregunta a la IA sobre el sentimiento, los puntos clave,
    los riesgos mencionados o el posible impacto en el valor de una acción.
    """)

    # Inicializar historial de chat si no existe
    if "insights_messages" not in st.session_state:
        st.session_state.insights_messages = []

    # Widgets para subir documentos y pegar texto (fuera del bucle de chat)
    with st.sidebar:
        st.subheader("Contexto para el Chat de Insights")
        uploaded_pdfs = st.file_uploader(
            "Sube PDFs (noticias, informes)", type="pdf", accept_multiple_files=True, key="insights_pdf"
        )
        pasted_text = st.text_area("O pega texto aquí", height=150, key="insights_text")

    # Mostrar historial de chat
    for message in st.session_state.insights_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pregunta sobre los documentos o el texto..."):
        # Añadir mensaje del usuario al historial
        st.session_state.insights_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Preparar la respuesta de la IA
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analizando y pensando..."):
                # 1. Recopilar todo el contexto de texto
                context_text = ""
                if uploaded_pdfs:
                    for pdf in uploaded_pdfs:
                        context_text += f"--- INICIO DEL DOCUMENTO: {pdf.name} ---\n"
                        context_text += extract_text_from_pdf(pdf)
                        context_text += f"\n--- FIN DEL DOCUMENTO: {pdf.name} ---\n\n"
                if pasted_text:
                    context_text += f"--- INICIO DEL TEXTO PEGADO ---\n"
                    context_text += pasted_text
                    context_text += f"\n--- FIN DEL TEXTO PEGADO ---\n\n"

                # 2. Construir el prompt final
                system_prompt = """
                Eres un analista financiero experto. Tu tarea es analizar los documentos y textos proporcionados
                para responder a la pregunta del usuario. Enfócate en identificar el sentimiento (positivo, negativo, neutro),
                los puntos clave, los riesgos potenciales y cómo la información podría afectar el valor de las acciones mencionadas.
                Sé conciso, objetivo y basa tus respuestas únicamente en la información proporcionada.
                """
                final_prompt = f"{system_prompt}\n\n**Documentos y Textos de Contexto:**\n{context_text}\n\n**Pregunta del Usuario:**\n{prompt}"

                # 3. Llamar a la API
                response = get_hf_response(
                    st.session_state.hf_api_key,
                    st.session_state.hf_model,
                    final_prompt,
                    st.session_state.hf_temp
                )

                # 4. Mostrar respuesta
                if response:
                    message_placeholder.markdown(response)
                    st.session_state.insights_messages.append({"role": "assistant", "content": response})
                else:
                    message_placeholder.markdown("No pude procesar la solicitud. Revisa la API Key y el error en la consola.")
                    st.session_state.insights_messages.append({"role": "assistant", "content": "Error en la solicitud."})


# --- PÁGINA: InvertirOnline API ---
def page_invertir_online():
    st.header("🏦 InvertirOnline (IOL) API Interaction")
    st.markdown("""
    Esta sección te permite interactuar con la API de InvertirOnline.
    **Importante:** Tus credenciales se usan para obtener un token de acceso y no se almacenan permanentemente por esta aplicación más allá de la sesión actual.
    El token de acceso tiene una duración limitada.
    """)

    # --- Sección de Login IOL ---
    st.subheader("1. Autenticación en IOL")
    if not st.session_state.get('iol_logged_in', False):
        with st.form("iol_login_form"):
            iol_username = st.text_input("Usuario IOL", key="iol_user_input", autocomplete="username", placeholder="TuUsuarioDeIOL")
            iol_password = st.text_input("Contraseña IOL", type="password", key="iol_pass_input", autocomplete="current-password", placeholder="TuContraseñaDeIOL")
            submitted_login = st.form_submit_button("Iniciar Sesión en IOL")

            if submitted_login:
                if not iol_username or not iol_password:
                    st.error("Por favor, ingresa tu usuario y contraseña de IOL.")
                else:
                    with st.spinner("Autenticando con IOL..."):
                        token_data, error = login_iol(iol_username, iol_password)
                        if token_data:
                            st.session_state.iol_token_data = token_data
                            st.session_state.iol_access_token = token_data.get('access_token')
                            st.session_state.iol_refresh_token = token_data.get('refresh_token')
                            st.session_state.iol_user = iol_username
                            st.session_state.iol_logged_in = True
                            st.session_state.iol_last_error = None
                            st.success(f"¡Autenticación exitosa! Bienvenido {iol_username}.")
                            st.info(get_iol_token_status())
                            st.rerun()
                        else:
                            st.session_state.iol_logged_in = False
                            st.session_state.iol_last_error = f"Error de autenticación: {error}"
                            st.error(st.session_state.iol_last_error)
    else:
        st.success(get_iol_token_status())
        if st.button("Cerrar Sesión de IOL"):
            st.session_state.iol_logged_in = False
            st.session_state.iol_access_token = None
            st.session_state.iol_token_data = None
            st.session_state.iol_user = None
            st.session_state.iol_last_data = None
            st.session_state.iol_last_error = None
            st.info("Sesión cerrada.")
            st.rerun()

    # --- Sección de Consulta de Endpoints ---
    if st.session_state.get('iol_logged_in', False):
        st.markdown("---")
        st.subheader("2. Consultar Endpoints de IOL")

        token_status = get_iol_token_status()
        if "expirado" in token_status.lower() or "no autenticado" in token_status.lower():
            st.warning(f"Estado del token: {token_status}. Por favor, vuelve a iniciar sesión si es necesario.")
            st.session_state.iol_logged_in = False
            st.rerun()
            return

        available_endpoints = {
            "Estado de Cuenta": "/api/v2/estadocuenta",
            "Portafolio (Argentina)": "/api/v2/portafolio/argentina",
            "Portafolio (EEUU)": "/api/v2/portafolio/estados_unidos",
            "Operaciones": "/api/v2/operaciones",
            "Notificaciones": "/api/v2/Notificacion",
            "Tipos de Fondos FCI": "/api/v2/Titulos/FCI/TipoFondos",
            "Administradoras FCI": "/api/v2/Titulos/FCI/Administradoras",
            "Datos de Perfil": "/api/v2/datos-perfil",
            "Cotización Título (BCBA)": "/api/v2/BCBA/Titulos/{simbolo}/Cotizacion",
            "Detalle Cotización Título (BCBA)": "/api/v2/BCBA/Titulos/{simbolo}/CotizacionDetalle",
            "Serie Histórica Título (BCBA)": "/api/v2/BCBA/Titulos/{simbolo}/Cotizacion/seriehistorica/{fechaDesde}/{fechaHasta}/{ajustada}"
        }
        selected_endpoint_name = st.selectbox(
            "Selecciona un endpoint para consultar:",
            options=list(available_endpoints.keys()),
            key="iol_endpoint_select"
        )

        endpoint_path = available_endpoints[selected_endpoint_name]
        params_input = {} 
        final_endpoint_path = endpoint_path # Inicializar con el path que puede tener placeholders

        # Variable para controlar si la UI está lista para hacer la llamada
        can_fetch_data = True 

        if "{simbolo}" in endpoint_path:
            simbolo_seleccionado_ui = None
            if "iol_parsed_tickers_list" in st.session_state and st.session_state.iol_parsed_tickers_list:
                opciones_simbolos = ["(Ingresar manualmente)"] + st.session_state.iol_parsed_tickers_list
                simbolo_choice = st.selectbox(
                    "Símbolo (BCBA):", options=opciones_simbolos, key="iol_simbolo_select_parsed"
                )
                if simbolo_choice == "(Ingresar manualmente)":
                    simbolo_seleccionado_ui = st.text_input("Ingresa Símbolo:", key="iol_simbolo_manual_input").upper()
                else:
                    simbolo_seleccionado_ui = simbolo_choice
            else:
                simbolo_seleccionado_ui = st.text_input("Símbolo del Título (BCBA):", key="iol_simbolo_generic_input").upper()

            if not simbolo_seleccionado_ui:
                st.warning("Por favor, selecciona o ingresa un símbolo.")
                can_fetch_data = False # No se puede continuar sin símbolo
            else:
                # Reemplazar {simbolo} solo si tenemos uno válido
                final_endpoint_path = endpoint_path.replace("{simbolo}", simbolo_seleccionado_ui)

                # Procesar parámetros adicionales solo si el símbolo está presente
                if selected_endpoint_name == "Serie Histórica Título (BCBA)":
                    today_date = datetime.today().date() # Obtener fecha de hoy
                    one_year_ago = today_date - timedelta(days=365) # Default 1 año atrás

                    col_fecha1, col_fecha2, col_ajustada = st.columns(3)
                    with col_fecha1:
                        fecha_desde_hist = st.date_input("Fecha Desde", one_year_ago, key="iol_hist_desde") # Default 1 año atrás
                    with col_fecha2:
                        # Asegurar que la fecha hasta no sea mayor que hoy
                        fecha_hasta_hist = st.date_input(
                            "Fecha Hasta",
                            today_date, # Default hoy
                            max_value=today_date, # Limitar a hoy como máximo
                            key="iol_hist_hasta"
                         )
                    with col_ajustada:
                        ajustada_hist = st.checkbox("Ajustada", value=True, key="iol_hist_ajustada")

                    if fecha_desde_hist >= fecha_hasta_hist:
                        st.error("La 'Fecha Desde' debe ser anterior a 'Fecha Hasta'.")
                        can_fetch_data = False # No se puede continuar con fechas inválidas
                    else:
                        # Reemplazar el resto de placeholders solo si las fechas son válidas
                        final_endpoint_path = final_endpoint_path.replace("{fechaDesde}", fecha_desde_hist.strftime("%Y-%m-%d"))
                        final_endpoint_path = final_endpoint_path.replace("{fechaHasta}", fecha_hasta_hist.strftime("%Y-%m-%d"))
                        final_endpoint_path = final_endpoint_path.replace("{ajustada}", str(ajustada_hist).lower())

        # El botón de obtener datos ahora verifica 'can_fetch_data'
        if st.button(f"Obtener Datos de '{selected_endpoint_name}'", key="iol_get_data_button", disabled=not can_fetch_data):
            # Volver a verificar si quedan placeholders (doble chequeo)
            if "{" in final_endpoint_path and "}" in final_endpoint_path:
                 st.error(f"Error interno: Faltan parámetros por completar en la ruta final: {final_endpoint_path}. Revisa la lógica del código.")
            else:
                with st.spinner(f"Consultando {selected_endpoint_name}..."):
                    access_token = st.session_state.get('iol_access_token')
                    data, error = get_iol_data(final_endpoint_path, access_token, params=params_input)
                    if data:
                        st.session_state.iol_last_data = data
                        st.session_state.iol_last_error = None
                        st.success("Datos obtenidos exitosamente.")
                    else:
                        st.session_state.iol_last_data = None
                        st.session_state.iol_last_error = f"Error al obtener datos de {selected_endpoint_name}: {error}"
                        st.error(st.session_state.iol_last_error)
                        if error and ("401" in str(error) or "Unauthorized" in str(error)):
                            st.warning("El token de acceso puede haber expirado. Intenta cerrar sesión y volver a iniciarla.")
                            st.session_state.iol_logged_in = False
                            st.rerun()

    st.markdown("---")
    st.subheader("3. Resultados de la API de IOL")
    if st.session_state.get('iol_last_error'):
        st.error(st.session_state.iol_last_error)
    if st.session_state.get('iol_last_data'):
        st.json(st.session_state.iol_last_data)
        # Intenta mostrar como DataFrame si es posible
        processed_data_for_df = None
        if isinstance(st.session_state.iol_last_data, list) and all(isinstance(item, dict) for item in st.session_state.iol_last_data):
             processed_data_for_df = st.session_state.iol_last_data
        # Ejemplo específico para la serie histórica que puede venir anidada
        elif isinstance(st.session_state.iol_last_data, dict) and 'datos' in st.session_state.iol_last_data and isinstance(st.session_state.iol_last_data['datos'], list):
             processed_data_for_df = st.session_state.iol_last_data['datos']

        if processed_data_for_df is not None and len(processed_data_for_df) > 0:
            try:
                df_iol = pd.DataFrame(processed_data_for_df)
                # Intentar convertir columnas comunes a tipos adecuados
                if 'fechaHora' in df_iol.columns:
                    df_iol['fechaHora'] = pd.to_datetime(df_iol['fechaHora'], errors='coerce')
                if 'ultimoPrecio' in df_iol.columns:
                    df_iol['ultimoPrecio'] = pd.to_numeric(df_iol['ultimoPrecio'], errors='coerce')
                # ... otras conversiones según sea necesario ...
                st.dataframe(df_iol)
            except Exception as e:
                st.caption(f"No se pudo convertir la respuesta a DataFrame o procesar columnas: {e}")
                
# --- NUEVAS FUNCIONES DE BACKEND PARA EL GENERADOR HOMEOSTÁTICO ---
@st.cache_data
def get_asset_metrics(_tickers):
    """
    Obtiene métricas financieras clave (Beta, P/E) para una lista de tickers usando yfinance.
    _tickers es un tuple para que la caché funcione correctamente.
    """
    tickers = list(_tickers)
    metrics = {}
    
    progress_bar = st.progress(0, text="Obteniendo métricas de activos...")
    
    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Obteniendo métricas para {ticker}...")
        try:
            info = yf.Ticker(ticker).info
            # CORRECCIÓN: Nos aseguramos de que info sea un diccionario válido y tenga al menos una clave
            # yfinance a veces puede devolver None o un dict vacío en caso de error.
            if info and 'symbol' in info: 
                metrics[ticker] = {
                    'beta': info.get('beta', 1.0),
                    'pe': info.get('trailingPE', 30.0),
                }
            else:
                # Si yfinance no devuelve nada útil, asignamos valores por defecto explícitamente.
                # Esto previene KeyErrors si info es None.
                metrics[ticker] = {'beta': 1.0, 'pe': 30.0}
        except Exception as e:
            # CORRECCIÓN: Nos aseguramos de asignar valores por defecto también en caso de una excepción
            metrics[ticker] = {'beta': 1.0, 'pe': 30.0}
            print(f"Advertencia: No se pudieron obtener métricas para {ticker} debido a un error. Usando valores por defecto. Error: {e}")
    
    progress_bar.empty()
    return metrics

def run_homeostatic_model(params):
    (universe, k, n_portfolios, beta_range, pe_range, hhi_max) = params
    
    asset_metrics = get_asset_metrics(tuple(universe))
    
    plausible_portfolios = []
    max_attempts = n_portfolios * 500 

    for _ in range(max_attempts):
        if len(plausible_portfolios) >= n_portfolios:
            break
            
        portfolio_tickers = random.sample(universe, k)
        
        # CORRECCIÓN: Chequeo de seguridad. Verificar si todos los tickers del candidato
        # tienen una entrada en el diccionario de métricas ANTES de acceder a ellas.
        if not all(ticker in asset_metrics for ticker in portfolio_tickers):
            # Si un ticker no está, este portafolio es inválido. Lo saltamos.
            continue

        betas = [asset_metrics[t]['beta'] for t in portfolio_tickers]
        pes = [asset_metrics[t]['pe'] for t in portfolio_tickers]
        
        portfolio_beta = np.mean(betas)
        portfolio_pe = np.mean(pes)
        
        if not (beta_range[0] <= portfolio_beta <= beta_range[1]):
            continue
        if not (pe_range[0] <= portfolio_pe <= pe_range[1]):
            continue
            
        plausible_portfolios.append(sorted(portfolio_tickers))

    unique_portfolios = list(set(tuple(p) for p in plausible_portfolios))
    
    return unique_portfolios


# --- CÓDIGO DE LA NUEVA PÁGINA ---
def page_homeostatic_generator():
    st.header("🛠️ Generador de Portafolios Homeostáticos")
    st.markdown("""
    Esta herramienta utiliza el **Modelo Homeostático** para descubrir portafolios de inversión que son estructuralmente robustos y equilibrados.
    En lugar de buscar el máximo retorno, busca carteras que se mantengan dentro de rangos de equilibrio en múltiples dimensiones de riesgo.
    """)
    
    # --- 1. Definir Universo de Activos ---
    st.subheader("1. Definir Universo de Activos")
    default_tickers = "AAPL, MSFT, GOOG, AMZN, NVDA, META, JPM, JNJ, V, PG, MA, UNH, HD, BAC, PFE, KO, XOM, T, CSCO, INTC"
    universe_input = st.text_area("Pega una lista de tickers (separados por coma o espacio) para el universo de búsqueda:", value=default_tickers, height=100)
    
    # CORRECCIÓN: Usar r'' para la expresión regular
    universe = sorted(list(set([t.strip().upper() for t in re.split(r'[,\s]+', universe_input) if t.strip()])))
    
    if len(universe) < 6:
        st.warning("Por favor, introduce al menos 6 tickers para formar un portafolio.")
        return
        
    st.info(f"Universo de búsqueda definido con **{len(universe)}** activos únicos.")

    # --- 2. Configurar Parámetros del Portafolio y Homeostasis ---
    st.subheader("2. Configurar Filtros de Homeostasis")
    
    col_params1, col_params2 = st.columns(2)
    with col_params1:
        k_assets = st.slider("Número de Activos por Portafolio:", min_value=5, max_value=min(30, len(universe)), value=10)
        n_portfolios_to_gen = st.number_input("Número de portafolios plausibles a generar:", min_value=10, max_value=10000, value=100)
    
    with col_params2:
        st.info("Configura los rangos aceptables para las 'señales vitales' de los portafolios.")
        
    col_filters1, col_filters2 = st.columns(2)
    with col_filters1:
        beta_range = st.slider("Rango de Beta del Portafolio (Riesgo de Mercado):", 0.0, 2.0, (0.8, 1.2), step=0.05)
    with col_filters2:
        pe_range = st.slider("Rango de P/E Ratio Promedio (Valuación):", 5.0, 100.0, (15.0, 40.0), step=1.0)
    
    # El HHI para un portafolio equiponderado de k activos es 1/k.
    hhi_max = 1.0 / k_assets 
    st.caption(f"Filtro de Diversificación (HHI) implícito: Se buscarán portafolios equiponderados de {k_assets} activos, lo que garantiza una diversificación estructural máxima (HHI de {hhi_max:.3f}).")
    
    # --- Explicación Matemática ---
    with st.expander("📖 Explicación Matemática del Modelo Homeostático de Portafolios"):
        st.markdown("""
        El modelo busca descubrir un conjunto de portafolios robustos `P_robusto` que cumplan con una serie de restricciones de homeostasis. Se define de la siguiente manera:
        
        **1. Definiciones:**
        - **A**: Universo de activos disponibles.
        - **P**: Un portafolio `P = {a₁, ..., a_k}`.
        - **V(P)**: Una función que calcula las "señales vitales" de un portafolio: `V(P) → [β_p(P), P/E_p(P), HHI(P), ...]`
        - **R**: El conjunto de restricciones o rangos homeostáticos definidos por el usuario.

        **2. La Fórmula Canónica:**
        El conjunto `P_robusto` se define como:
        
        `P_robusto = { P ⊆ A | |P| = k ∧ (V(P) ∈ R) }`
        
        Esto se lee como: "El conjunto de todos los portafolios P que son un subconjunto del universo de activos, tienen un tamaño k, y cuyo vector de señales vitales V(P) se encuentra dentro de los rangos de homeostasis R."

        **Métricas de Homeostasis Implementadas:**
        - **Filtro de Riesgo Sistémico:** El **Beta ponderado (`β_p`)** del portafolio se mantiene en un rango, evitando exposiciones extremas al mercado.
        - **Filtro de Valuación:** El **Price-to-Earnings (P/E) Ratio ponderado** se mantiene en un rango saludable para evitar burbujas o trampas de valor.
        - **Filtro de Diversificación:** Se exige una estructura equiponderada, lo que maximiza la diversificación y minimiza el **Índice de Herfindahl-Hirschman (HHI)**.
        """)
        
    # --- Ejecución del Modelo ---
    if st.button("🧬 Generar Portafolios Homeostáticos", type="primary"):
        params = (universe, k_assets, n_portfolios_to_gen, beta_range, pe_range, hhi_max)
        with st.spinner("Ejecutando Modelo Homeostático Híbrido... (Esto puede tardar varios minutos la primera vez mientras se descargan las métricas)"):
            generated_portfolios = run_homeostatic_model(params)
        
        st.session_state.generated_portfolios = generated_portfolios
        
        if not generated_portfolios:
            st.error("No se encontraron portafolios que cumplieran con los estrictos filtros de homeostasis. Intenta ampliar los rangos de Beta o P/E, o aumentar el universo de activos.")
        else:
            st.success(f"¡Éxito! Se generaron **{len(generated_portfolios)}** portafolios robustos.")
    
    # --- Resultados y Opción de Guardar ---
    if 'generated_portfolios' in st.session_state and st.session_state.generated_portfolios:
        st.subheader("Portafolios Generados")
        portfolios_to_display = st.session_state.generated_portfolios

        for i, portfolio in enumerate(portfolios_to_display[:50]): # Mostramos un máximo de 50
            col_portfolio, col_button = st.columns([4, 1])
            with col_portfolio:
                st.code(", ".join(portfolio))
            with col_button:
                portfolio_name = f"Homeostatico-{i+1}-{datetime.now().strftime('%Y%m%d')}"
                if st.button(f"Guardar como '{portfolio_name}'", key=f"save_btn_{i}"):
                    # Guardar con pesos iguales
                    weights = {ticker: 1.0 / len(portfolio) for ticker in portfolio}
                    st.session_state.portfolios[portfolio_name] = weights
                    success, error_msg = save_portfolios_to_file(st.session_state.portfolios)
                    if success:
                        st.toast(f"✅ Portafolio '{portfolio_name}' guardado exitosamente!")
                    else:
                        st.error(error_msg)
                        
        if len(portfolios_to_display) > 50:
            st.info("Se muestran los primeros 50 portafolios generados.")
            
        st.markdown("---")
        st.info("Los portafolios guardados ahora están disponibles en las páginas 'Create/Edit Portfolios', 'View Portfolio Returns' y 'Gestión de Riesgo' para un análisis detallado.")


# --- INTEGRACIÓN CON LA IA: EL ASISTENTE DE ESTRATEGIA ---
def page_ai_strategy_assistant():
    st.header("🧠 Asistente de Estrategia IA")
    st.markdown("""
    Describe tu objetivo de inversión en lenguaje natural, y la IA lo traducirá en un conjunto de **restricciones cuantitativas**
    para el **Generador de Portafolios Homeostáticos**.
    """)
    
    user_strategy_prompt = st.text_area(
        "Describe tu estrategia de inversión:",
        height=150,
        placeholder="Ej: 'Busco un portafolio de 15 acciones tecnológicas, pero no muy riesgoso. Quiero evitar empresas sobrevaloradas (burbujas) y enfocarme en las más grandes y establecidas del S&P 500. No quiero apostar todo a la IA, busco algo de diversificación dentro del sector tech.'"
    )

    if st.button("Traducir Estrategia a Filtros", type="primary"):
        if not user_strategy_prompt:
            st.warning("Por favor, describe tu estrategia.")
        else:
            system_prompt = """
            Eres un experto en finanzas cuantitativas (Quant). Tu tarea es traducir la estrategia de inversión de un usuario en un conjunto de restricciones JSON para un modelo de optimización. Las claves del JSON que puedes generar son: 'k_assets', 'beta_range', 'pe_range', 'universe'.

            - 'k_assets' (integer): El número de activos en el portafolio.
            - 'beta_range' (array of two floats): [min_beta, max_beta]. Un portafolio de bajo riesgo tiene un Beta cercano a 0.8. Uno de mercado es 1.0. Uno agresivo es > 1.2.
            - 'pe_range' (array of two floats): [min_pe, max_pe]. Estrategias "Value" (evitar sobrevaloración) tienen un P/E bajo (ej. 10-25). Estrategias "Growth" aceptan P/E altos (ej. 30-80).
            - 'universe' (string): Una lista de tickers de acciones, separados por coma, que se ajusten a la descripción del usuario (ej. si pide 'grandes tecnológicas', lista los tickers de las FAANG y otras similares).

            Analiza la descripción del usuario y responde **SOLAMENTE CON UN BLOQUE DE CÓDIGO JSON VÁLIDO Y NADA MÁS**.
            """
            
            final_prompt = f"{system_prompt}\n\n**Descripción del Usuario:**\n{user_strategy_prompt}"
            
            with st.spinner("IA Quant analizando tu estrategia..."):
                response = get_hf_response(
                    st.session_state.hf_api_key,
                    st.session_state.hf_model,
                    final_prompt,
                    temperature=0.1 # Queremos una respuesta precisa y estructurada
                )
            
            if response:
                st.subheader("Filtros Cuantitativos Sugeridos por la IA")
                st.code(response, language="json")
                
                try:
                    # Limpiar la respuesta para que sea un JSON válido
                    # Algunos modelos añaden ```json y ``` al principio/final
                    json_text = re.search(r'\{.*\}', response, re.DOTALL).group(0)
                    suggested_params = json.loads(json_text)
                    
                    st.subheader("Aplicación Práctica")
                    st.markdown(
                        "Puedes copiar y pegar el **Universo** sugerido y ajustar los **Filtros** "
                        "en la página **'Generador Homeostático'** para ejecutar el modelo con esta estrategia."
                    )
                except (json.JSONDecodeError, AttributeError):
                    st.error("La IA generó una respuesta, pero no pude interpretarla como un JSON válido. Inténtalo de nuevo, quizás reformulando tu estrategia.")
                    st.text(response)
            else:
                st.error("No se pudo obtener una respuesta de la IA.")


# --- Initialize Session State ---
default_session_values = {
    'selected_page': "Welcome Page",
    'entities': [], 'dataframes': {}, 'data_type': 'returns',
    'apply_outlier_treatment': False, 'iqr_factor': 1.5,
    'iol_logged_in': False, 'iol_access_token': None, 'iol_token_data': None,
    'iol_user': None, 'iol_last_data': None, 'iol_last_error': None,
    'iol_parsed_tickers_info': [], 'iol_parsed_tickers_list': [],
    # --- NUEVAS CLAVES A AÑADIR ---
    'hf_api_key': "",
    'insights_messages': [], # Historial para el chat de insights
    'portfolio_chat_messages': {}, # Un historial por portafolio
}
for key, value in default_session_values.items():
    if key not in st.session_state:
        st.session_state[key] = value
        
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = load_portfolios_from_file()
elif not st.session_state.portfolios and os.path.exists(PORTFOLIO_FILE): # Cargar si está vacío pero existe el archivo
    loaded_portfolios_on_empty = load_portfolios_from_file()
    if loaded_portfolios_on_empty:
        st.session_state.portfolios = loaded_portfolios_on_empty


# --- Sidebar Setup ---
st.sidebar.title("Configuración General")

# --- NUEVO: Configuración de la IA ---
st.sidebar.header("🤖 Asistente IA (Hugging Face)")
hf_api_key = st.sidebar.text_input(
    "Hugging Face API Key",
    type="password",
    value=st.session_state.get('hf_api_key', ''),
    help="Introduce tu clave aquí. No se guarda permanentemente."
)
# Guardamos la clave en el session state para que persista entre páginas
if hf_api_key:
    st.session_state.hf_api_key = hf_api_key

st.session_state.hf_model = st.sidebar.selectbox(
    "Modelo de Lenguaje",
    ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct", "google/gemma-7b-it"],
    help="Elige el modelo de IA para las tareas de análisis de texto."
)
st.session_state.hf_temp = st.sidebar.slider(
    "Temperatura (Creatividad)", 0.1, 1.0, 0.5, 0.1,
    help="Bajo=preciso y repetitivo. Alto=creativo y variado."
)


st.sidebar.header("Forecasting: Data Selection")
# ... (resto de la configuración de la sidebar de forecasting sin cambios) ...
ticker_input = st.sidebar.text_area("Enter Tickers (for Forecasting)", "AAPL, MSFT, GOOG", help="e.g., AAPL, MSFT GOOG TSLA (Used for forecasting pages)")
today = datetime.today().date(); three_years_ago = today - timedelta(days=3*365)
start_date = st.sidebar.date_input("Start Date (for Forecasting)", three_years_ago, key="forecast_start_date")
end_date = st.sidebar.date_input("End Date (for Forecasting)", today, key="forecast_end_date")
data_type_choice = st.sidebar.radio("Forecast Target", ('Daily Returns (%)', 'Adjusted Close Price'), index=0, key="data_choice", help="Forecast returns (stationary) or prices.")
st.sidebar.markdown("---")
st.sidebar.subheader("Forecasting: Preprocessing")
apply_outlier_treatment_ui = st.sidebar.checkbox(
    "Apply Outlier Treatment (IQR)", value=st.session_state.get('apply_outlier_treatment', False),
    key="apply_outlier_treatment_cb", help="Cap extreme values using IQR before forecasting analysis. Affects forecasting pages only." )
iqr_k_factor_ui = st.sidebar.number_input(
    "IQR Factor (k)", min_value=1.0, max_value=5.0, value=st.session_state.get('iqr_factor', 1.5),
    step=0.1, key="iqr_factor_input", help="Multiplier for IQR bounds (higher k treats fewer points). Affects forecasting pages only.",
    disabled=not apply_outlier_treatment_ui )
st.sidebar.markdown("---")
calculate_returns_flag = (data_type_choice == 'Daily Returns (%)')
load_button = st.sidebar.button("Load & Process Data (for Forecasting)", key="load_data_button")

if ticker_input: 
    # CORRECCIÓN: Usar r'' para la expresión regular
    tickers_forecast = [t.strip().upper() for t in re.split(r'[,\s]+', ticker_input) if t.strip()] 
else: 
    tickers_forecast = []

if load_button and tickers_forecast:
    if start_date >= end_date: st.sidebar.error("Forecast Start date must be before End date.")
    else:
        st.session_state.apply_outlier_treatment = apply_outlier_treatment_ui
        st.session_state.iqr_factor = iqr_k_factor_ui
        st.session_state.entities = []; st.session_state.dataframes = {}
        st.session_state.data_type = 'returns' if calculate_returns_flag else 'prices'
        df_loaded = load_stock_data(tickers_forecast, start_date, end_date, calculate_returns=calculate_returns_flag)

        if df_loaded is not None and not df_loaded.empty:
            with st.spinner("Processing forecasting data..."):
                grouped_data = df_loaded
                all_entities = sorted(grouped_data['Entity'].unique())
                temp_dfs = {}
                statuses = {}
                outlier_info = {}
                for i, entity in enumerate(all_entities):
                    statuses[entity] = f"Processing {entity}..."
                    # num_treated = 0 # No se usa explicitamente
                    try:
                        entity_df = grouped_data[grouped_data['Entity'] == entity][['Actual']].sort_index()
                        entity_df.index = pd.to_datetime(entity_df.index)
                        if not entity_df['Actual'].isnull().all():
                            if st.session_state.apply_outlier_treatment:
                                target_series = entity_df['Actual'].dropna()
                                if not target_series.empty:
                                    treated_series, num_treated_val = treat_outliers_iqr(target_series, k=st.session_state.iqr_factor)
                                    entity_df['Actual'].update(treated_series)
                                    outlier_info[entity] = num_treated_val
                                else: outlier_info[entity] = 0
                            else: outlier_info[entity] = 0
                        else: outlier_info[entity] = 0
                        entity_df_clean_post_outlier = entity_df['Actual'].dropna()
                        min_pts = 30
                        if len(entity_df_clean_post_outlier) >= min_pts:
                            if not entity_df.index.is_unique: entity_df = entity_df[~entity_df.index.duplicated(keep='first')]
                            inf_freq = pd.infer_freq(entity_df.index); curr_freq = None
                            try:
                                freq_to_set = inf_freq if inf_freq else 'B'
                                entity_df = entity_df.asfreq(freq_to_set)
                                entity_df['Actual'] = entity_df['Actual'].interpolate(method='time', limit=3)
                                curr_freq = freq_to_set
                                statuses[entity] += f" (Freq: {curr_freq})"
                            except Exception as freq_e : statuses[entity] += f" (Freq Err: {freq_e})"; entity_df.dropna(subset=['Actual'], inplace=True); curr_freq = "Error/Irregular"
                            entity_df.dropna(subset=['Actual'], inplace=True)
                            if len(entity_df) >= min_pts:
                                temp_dfs[entity] = entity_df
                                status_outlier = f", {outlier_info.get(entity, 0)} outliers capped" if st.session_state.apply_outlier_treatment and outlier_info.get(entity, 0) > 0 else ""
                                statuses[entity] = f"OK {entity} ({len(entity_df)} pts, Freq: {curr_freq if curr_freq else 'Check'}){status_outlier}"
                            else: statuses[entity] = f"Skip '{entity}': < {min_pts} pts after final cleaning."
                        else: statuses[entity] = f"Skip '{entity}': < {min_pts} pts after initial load/outlier step."
                    except Exception as e: statuses[entity] = f"ERR '{entity}': {e}"; st.sidebar.error(f"Error processing {entity}: {e}\n{traceback.format_exc()}")
                st.sidebar.markdown("---"); st.sidebar.subheader("Forecasting: Processing Status")
                for status_msg in statuses.values(): # Renombrar para evitar conflicto
                    if "Skip" in status_msg or "ERR" in status_msg: st.sidebar.warning(status_msg)
                    elif "OK" in status_msg: st.sidebar.caption(status_msg)
                st.session_state.dataframes = temp_dfs
                st.session_state.entities = sorted(list(temp_dfs.keys()))
                if not st.session_state.entities:
                    st.error("No tickers had sufficient data after processing for forecasting.")
                    st.session_state.selected_page = "Welcome Page"
                else:
                    st.sidebar.success(f"Successfully processed {len(st.session_state.entities)} tickers for forecasting.")
                    st.session_state.selected_page = "Data Visualization" if st.session_state.selected_page =="Welcome Page" else st.session_state.selected_page
        elif df_loaded is None: st.error("Forecasting data loading from Yahoo Finance failed. Check tickers and connection.")
        st.rerun()

# --- Carga de Tickers para IOL en Sidebar ---
st.sidebar.markdown("---")
st.sidebar.subheader("IOL: Tickers (BCBA)")
ocr_text_input_sidebar = st.sidebar.text_area(
    "Pega aquí tu lista de tickers (formato: Nombre (TICKER)):",
    height=150,
    key="iol_ocr_text_input_sidebar",
    value="""> ENERGÍA:
Central Puerto (CEPU)
SC del Plata (COME)
Edenor (EDN)
Pampa Energía (PAMP)
Transp. Gas del Norte (TGNO4)
Transp. Gas del Sur (TGSU2)
Transener (TRAN)
YPF (YPFD)
> Bancario:
Banco BBVA (BBVA)
Banco Macro (BMA)
Grupo Financiero Galicia (GGAL)
Grupo Supervielle (SUPV)
Grupo Financiero Valores (VALO)
> Bienes de consumo:
Cresud (CRES)
Mirgor (MIRG)
Cablevisión (CVH)
Telecom (TECO2)
Havanna Holding S.A. (HAVA)
Grupo Clarín S.A. (GCLA)
> Materiales:
Aluar (ALUA)
Ternium (TXAR)
Loma Negra (LOMA)
Holcim S.A. (HARG)
Celulosa Argentina S.A. (CELU)
Ferrum S.A. (FERR)
Agrometal S.A. (AGRO)"""
)
if st.sidebar.button("Parsear Tickers de Lista IOL", key="iol_parse_ocr_button_sidebar"):
    if ocr_text_input_sidebar:
        parsed_data_sidebar = parse_tickers_from_text(ocr_text_input_sidebar)
        if parsed_data_sidebar:
            st.session_state.iol_parsed_tickers_info = parsed_data_sidebar
            st.session_state.iol_parsed_tickers_list = sorted(list(set(info["ticker"] for info in parsed_data_sidebar)))
            st.sidebar.success(f"{len(st.session_state.iol_parsed_tickers_list)} tickers parseados.")
            st.sidebar.info("Tickers para copiar (para portafolios de Yahoo Finance, añade '.BA'):")
            tickers_con_ba = [f"{t}.BA" for t in st.session_state.iol_parsed_tickers_list]
            st.sidebar.code(", ".join(tickers_con_ba))
        else:
            st.sidebar.warning("No se pudieron parsear tickers del texto.")
    else:
        st.sidebar.warning("El área de texto de IOL Tickers está vacía.")


# --- Sidebar Navigation ---
st.sidebar.markdown("---"); st.sidebar.title("Navigation")
page_options = [
    "Welcome Page",
    "IA Strategy Assistant",       
    "Homeostatic Generator", # <-- NUEVA PÁGINA    
    "Create/Edit Portfolios",
    "View Portfolio Returns", # El chat de portafolio está aquí dentro
    "Gestión de Riesgo",
    "InvertirOnline API",
    "Chat de Análisis Cualitativo", # <-- NUEVO NOMBRE DE PÁGINA
    "--- Forecasting ---",
    "Data Visualization",
    "Series Decomposition",
    "Stationarity & Lags",
    "Forecasting Models"
]
selectable_page_options = [p for p in page_options if not p.startswith("---")]

if 'selected_page' not in st.session_state or st.session_state.selected_page not in selectable_page_options:
    st.session_state.selected_page = "Welcome Page"

try:
    if st.session_state.selected_page not in page_options:
        st.session_state.selected_page = "Welcome Page"
    page_idx = page_options.index(st.session_state.selected_page)
except ValueError:
     st.session_state.selected_page = "Welcome Page"
     page_idx = page_options.index(st.session_state.selected_page)

page = st.sidebar.radio("Select Section", page_options, index=page_idx, key="nav_radio")
if page != st.session_state.selected_page and not page.startswith("---"):
    st.session_state.selected_page = page
    st.rerun()

st.sidebar.markdown("---")
if prophet_installed: st.sidebar.caption("✅ Prophet library is installed.")
else:
    st.sidebar.warning("⚠️ Prophet library not installed. Prophet model will be skipped.")
    st.sidebar.caption("Install via: `pip install prophet`")
if pypfopt_installed: st.sidebar.caption("✅ PyPortfolioOpt library is installed.")
else:
    st.sidebar.warning("⚠️ PyPortfolioOpt not installed. Portfolio Optimization disabled.")
    st.sidebar.caption("Install via: `pip install PyPortfolioOpt`")

if tbats_installed: st.sidebar.caption("✅ TBATS library is installed.")
else:
    st.sidebar.warning("⚠️ TBATS library not installed. TBATS model will be disabled.")
    st.sidebar.caption("Install via: `pip install tbats`")

if xgb_installed: st.sidebar.caption("✅ XGBoost library is installed.")
else:
    st.sidebar.warning("⚠️ XGBoost library not installed. XGBoost model will be disabled.")
    st.sidebar.caption("Install via: `pip install xgboost`")

if lgbm_installed: st.sidebar.caption("✅ LightGBM library is installed.")
else:
    st.sidebar.warning("⚠️ LightGBM library not installed. LightGBM model will be disabled.")
    st.sidebar.caption("Install via: `pip install lightgbm`")
    
# --- Page Routing ---
current_page_to_display = st.session_state.selected_page
if current_page_to_display == "Welcome Page":
    main_page()
elif current_page_to_display == "IA Strategy Assistant":
    page_ai_strategy_assistant()
elif current_page_to_display == "Homeostatic Generator": # <-- RUTA A LA NUEVA PÁGINA
    page_homeostatic_generator()
elif current_page_to_display == "Create/Edit Portfolios":
    page_create_portfolio()
elif current_page_to_display == "View Portfolio Returns":
    page_view_portfolio_returns()
elif current_page_to_display == "Gestión de Riesgo": # <-- ROUTING A LA NUEVA PÁGINA
    page_risk_management()
elif current_page_to_display == "InvertirOnline API":
    page_invertir_online()
elif current_page_to_display == "Chat de Análisis Cualitativo": # <-- NUEVA RUTA
    page_investment_insights_chat()
elif not st.session_state.get('entities', []) and current_page_to_display in ["Data Visualization", "Series Decomposition", "Stationarity & Lags", "Forecasting Models"]:
    st.warning("Forecasting data not loaded yet.")
    st.info("👈 Load data using the sidebar controls ('Load & Process Data (for Forecasting)') to enable these analysis pages.")
elif current_page_to_display == "Data Visualization":
    data_visualization()
elif current_page_to_display == "Series Decomposition":
    decomposition()
elif current_page_to_display == "Stationarity & Lags":
    optimal_lags()
elif current_page_to_display == "Forecasting Models":
    forecast_models()
else:
    st.error(f"Invalid page selected: {current_page_to_display}. Returning to Welcome Page.")
    main_page()
