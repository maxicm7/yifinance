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
from scipy.stats import normaltest
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.api import VAR
import yfinance as yf
import warnings
import traceback # For detailed error printing
import json

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

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Stock Analysis & Portfolio Tool")

# --- Constante para el archivo de portafolios ---
PORTFOLIO_FILE = "portfolios_data1.json"

# --- Constantes para IOL API ---
IOL_API_BASE_URL = "https://api.invertironline.com"

# --- Helper Functions (Forecasting App - Sin cambios) ---
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

def save_portfolios_to_file(portfolios_dict):
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolios_dict, f, indent=4)
    except Exception as e:
        st.sidebar.error(f"Error al guardar portafolios en {PORTFOLIO_FILE}: {e}")


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

# --- Page Definition Functions (Forecasting App - Sin cambios) ---
def main_page():
    st.header("Welcome!")
    st.markdown("""
    This application provides tools for **portfolio management**, **individual stock forecasting**, **conceptual event analysis**, and **InvertirOnline API interaction**.
    (Descripción de workflows sin cambios, solo se actualiza la mención a la API de IOL)
    """)
# ... (resto de tus funciones de página de forecasting sin cambios: data_visualization, decomposition, optimal_lags, forecast_models) ...
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

    st.markdown("---"); st.subheader("Model Specific Parameters")
    col_arima, col_ets, col_prophet = st.columns(3)
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

    if use_exog:
        with st.spinner("Preparing exogenous variable..."):
            try:
                full_series_lagged = full_series.shift(1)
                full_series_exog_sq = (full_series_lagged**2).rename(exog_name)
                train_data_exog_prep = full_series_exog_sq.reindex(train_data.index).fillna(method='bfill').fillna(method='ffill').fillna(0)
                test_data_exog_prep = full_series_exog_sq.reindex(test_data.index).fillna(method='bfill').fillna(method='ffill').fillna(0)
                last_known_actual_value = test_data.iloc[-1] if not test_data.empty else train_data.iloc[-1]
                future_exog_list = [(last_known_actual_value**2)] * m_future
                future_exog_prep = pd.Series(future_exog_list, index=future_prediction_dates, name=exog_name)
                combined_exog_for_prediction_prep = pd.concat([test_data_exog_prep, future_exog_prep])
                final_train_exog_clean = train_data_exog_prep.reindex(train_data.index)
                full_pred_index = test_data.index.union(future_prediction_dates)
                combined_exog_clean = combined_exog_for_prediction_prep.reindex(full_pred_index)
                nan_in_train_exog = final_train_exog_clean.isnull().any()
                nan_in_pred_exog = combined_exog_clean.isnull().any()
                if nan_in_train_exog or nan_in_pred_exog:
                     st.warning(f"NaNs detected in exogenous variable after processing (Train: {nan_in_train_exog}, Pred: {nan_in_pred_exog}). Attempting to fill with 0.")
                     if final_train_exog_clean is not None: final_train_exog_clean = final_train_exog_clean.fillna(0)
                     if combined_exog_clean is not None: combined_exog_clean = combined_exog_clean.fillna(0)
                st.caption("Exogenous variable prepared.")
            except Exception as e:
                 st.error(f"Failed to create exogenous variable: {e}")
                 use_exog = False; final_train_exog_clean = None; combined_exog_clean = None
                 st.warning("Proceeding without exogenous variable due to error.")

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

    model_names_cv = ["ARIMA", "ARIMAX", "SARIMAX", "ETS", "VAR"]
    if prophet_installed: model_names_cv.append("Prophet")

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
        cv_train_exog, cv_val_exog = None, None
        prophet_cv_exog_ready = False
        if use_exog and split_exog_cv is not None and not split_exog_cv.empty:
             try:
                 temp_train_exog = split_exog_cv.iloc[train_idx].reindex(cv_train.index)
                 temp_val_exog = split_exog_cv.iloc[val_idx].reindex(cv_val.index)
                 temp_train_exog = temp_train_exog.fillna(method='ffill').fillna(method='bfill').fillna(0)
                 temp_val_exog = temp_val_exog.fillna(method='ffill').fillna(method='bfill').fillna(0)
                 if not temp_train_exog.isnull().any() and not temp_val_exog.isnull().any():
                      cv_train_exog, cv_val_exog = temp_train_exog, temp_val_exog
                      prophet_cv_exog_ready = True
                 else: st.caption(f"CV Split {cv_split_count}: NaN found in Exog, some models skipped.")
             except Exception as e_exog_cv: st.caption(f"CV Split {cv_split_count}: Error slicing Exog ({e_exog_cv}), some models skipped.")
        progress_text = f"Running CV Split {cv_split_count}/{actual_cv_splits}..."; progress_bar_cv.progress(cv_split_count / actual_cv_splits); cv_message.text(progress_text)
        min_cv_train_len = max(15, 2 * m_seasonal if m_seasonal > 1 else 15)
        if len(cv_train) < min_cv_train_len:
            st.caption(f"CV Split {cv_split_count}: Skipped (training data < {min_cv_train_len}).")
            [all_rmse_scores[name].append(np.nan) for name in model_names_cv]
            continue
        order_cv, seasonal_order_cv = (1,0,0), (0,0,0,0)
        # ran_auto_arima_cv = False # No se usa explicitamente
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
                    # ran_auto_arima_cv = True # No se usa explicitamente
                else: st.caption(f"CV Split {cv_split_count}: AutoARIMA skipped (too few obs). Using defaults.")
            except Exception as e_auto_cv: st.caption(f"CV Split {cv_split_count}: AutoARIMA failed ({e_auto_cv}). Using defaults.")
        if "ARIMA" in model_names_cv:
            try:
                model=ARIMA(cv_train,order=order_cv).fit()
                pred=model.predict(start=cv_val.index[0],end=cv_val.index[-1])
                all_rmse_scores["ARIMA"].append(np.sqrt(mean_squared_error(cv_val,pred)))
            except Exception: all_rmse_scores["ARIMA"].append(np.nan)
        if "ARIMAX" in model_names_cv:
            if use_exog and cv_train_exog is not None and cv_val_exog is not None:
                try:
                    model=ARIMA(cv_train,order=order_cv,exog=cv_train_exog).fit()
                    pred=model.predict(start=cv_val.index[0],end=cv_val.index[-1],exog=cv_val_exog)
                    all_rmse_scores["ARIMAX"].append(np.sqrt(mean_squared_error(cv_val,pred)))
                except Exception: all_rmse_scores["ARIMAX"].append(np.nan)
            else: all_rmse_scores["ARIMAX"].append(np.nan)
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

    st.markdown("---"); st.subheader("Test Set Performance & Future Forecast Visualization")
    fig_fc, ax_fc = plt.subplots(figsize=(14, 7))
    ax_fc.plot(train_data.index, train_data, label='Training Data', color='dimgray', lw=1, alpha=0.7)
    ax_fc.plot(test_data.index, test_data, label=f'Actual Test ({data_type.capitalize()})', color='blue', marker='.', linestyle='-', ms=5)
    colors = {'ARIMA':'#1f77b4','ARIMAX':'#ff7f0e','SARIMAX':'#2ca02c','ETS':'#d62728','VAR':'#9467bd', 'Prophet':'#8c564b'}
    plot_successful = False
    if test_predictions:
        for name, preds in test_predictions.items():
            if preds is not None and isinstance(preds, pd.Series) and not preds.empty:
                preds_aligned = preds.reindex(test_data.index).dropna()
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
    if not test_data.empty and future_prediction_dates is not None:
         combined_index = test_data.index.union(future_prediction_dates)
         results_df = pd.DataFrame(index=combined_index);
         results_df['Actual'] = test_data.reindex(combined_index)
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
    if test_predictions and not test_data.empty:
        for name, preds in test_predictions.items():
            if preds is not None and isinstance(preds, pd.Series):
                 preds_aligned = preds.reindex(test_data.index).dropna()
                 actuals_aligned = test_data.reindex(preds_aligned.index).dropna()
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
             try: res_aligned = residuals.reindex(train_idx).dropna()
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
def page_create_portfolio():
    st.header("📈 Create or Edit Portfolios")
    st.markdown("""
    Define multiple portfolios by name, entering stock tickers and their weights.
    **Important:**
    *   Each portfolio's weights **must sum to 1.0** (or 100%).
    *   Use official ticker symbols (e.g., **AAPL**, **MSFT**, **CEPU.BA**). Check spelling carefully.
    *   Saving overwrites portfolios with the same name.
    *   **Portfolios are saved to `portfolios_data.json` in the same directory as the script.**
    """)
    portfolio_names = list(st.session_state.portfolios.keys())
    options = ["(Create New Portfolio)"] + sorted(portfolio_names)
    col_select, col_name_input = st.columns([1, 2])
    with col_select:
        selected_portfolio_name = st.selectbox(
            "Select Portfolio:", options, key="portfolio_select",
            help="Choose a portfolio to edit or select '(Create New Portfolio)'"
        )
    current_name = ""
    with col_name_input:
        if selected_portfolio_name == "(Create New Portfolio)":
            current_name = st.text_input("Enter Name for New Portfolio:", key="new_portfolio_name", placeholder="e.g., My Tech Stocks").strip()
        else:
            current_name = st.text_input("Portfolio Name (can be edited):", value=selected_portfolio_name, key="edit_portfolio_name").strip()
    st.markdown("---")
    initial_tickers_str = ""
    initial_weights = {}
    if selected_portfolio_name != "(Create New Portfolio)" and selected_portfolio_name in st.session_state.portfolios:
        initial_weights = st.session_state.portfolios[selected_portfolio_name]
        initial_tickers_str = ", ".join(initial_weights.keys())
    col_form, col_display = st.columns([2, 1])
    with col_form:
        if current_name: st.subheader(f"Define/Edit Assets & Weights for: '{current_name}'")
        else:
             st.subheader("Define Assets & Weights")
             if selected_portfolio_name == "(Create New Portfolio)": st.warning("Please enter a name for the new portfolio.")
        with st.form("portfolio_form"):
            ticker_input_str = st.text_area(
                "Enter Tickers (comma or space separated)", value=initial_tickers_str,
                help="e.g., AAPL, MSFT, GOOG. For BCBA use format: TICKER.BA (e.g. CEPU.BA)" )
            tickers = []
            if ticker_input_str: tickers = sorted([t.strip().upper() for t in re.split('[,\s]+', ticker_input_str) if t.strip()])
            weights_input = {}
            if tickers:
                st.markdown("**Enter Weights (as decimals, e.g., 0.4 for 40%):**")
                max_cols = 4; num_rows = (len(tickers) + max_cols - 1) // max_cols; idx = 0
                for r in range(num_rows):
                    weight_cols = st.columns(min(len(tickers) - idx, max_cols))
                    for i in range(len(weight_cols)):
                        if idx < len(tickers):
                            ticker = tickers[idx]
                            with weight_cols[i]:
                                default_val = initial_weights.get(ticker, 0.0 if len(initial_weights)>0 else (1.0/len(tickers) if len(tickers)>0 else 0.0))
                                weights_input[ticker] = st.number_input(
                                    f"Wt {ticker}", min_value=0.0, max_value=1.0, value=float(default_val),
                                    step=0.01, format="%.4f", key=f"weight_{current_name}_{ticker}" )
                            idx += 1
            else: st.info("Enter ticker symbols above to define weights.")
            form_cols = st.columns(2)
            with form_cols[0]: submitted_save = st.form_submit_button(f"💾 Save Portfolio '{current_name or '(Enter Name)'}'", use_container_width=True, type="primary" if current_name else "secondary")
            with form_cols[1]: submitted_delete = st.form_submit_button(f"🗑️ Delete '{selected_portfolio_name}'", use_container_width=True, disabled=(selected_portfolio_name == "(Create New Portfolio)"))
            if submitted_save:
                if not current_name: st.error("Please enter a portfolio name.")
                elif selected_portfolio_name == "(Create New Portfolio)" and current_name in st.session_state.portfolios: st.error(f"A portfolio named '{current_name}' already exists.")
                elif not tickers: st.error("Please enter at least one ticker symbol.")
                elif not weights_input: st.error("Weights could not be determined.")
                else:
                    filtered_weights = {t: w for t, w in weights_input.items() if t in tickers}
                    total_weight = sum(filtered_weights.values())
                    if not np.isclose(total_weight, 1.0, atol=0.001): st.warning(f"⚠️ Weights for '{current_name}' sum to {total_weight:.4f}, which is not close to 1.0. Please adjust.")
                    else:
                        final_weights = {t: max(0.0, w) for t, w in filtered_weights.items()}
                        final_sum = sum(final_weights.values())
                        if not np.isclose(final_sum, 1.0) and final_sum > 1e-6 : final_weights = {t: w / final_sum for t, w in final_weights.items()}
                        renamed = False
                        if selected_portfolio_name != "(Create New Portfolio)" and current_name != selected_portfolio_name:
                             if current_name in st.session_state.portfolios: st.error(f"Cannot rename to '{current_name}' as another portfolio with that name already exists."); st.stop()
                             else: del st.session_state.portfolios[selected_portfolio_name]; renamed = True
                        st.session_state.portfolios[current_name] = final_weights
                        save_portfolios_to_file(st.session_state.portfolios)
                        if renamed: st.success(f"✅ Portfolio '{selected_portfolio_name}' successfully renamed and saved as '{current_name}'!")
                        else: st.success(f"✅ Portfolio '{current_name}' saved successfully!")
                        st.balloons(); st.rerun()
            if submitted_delete:
                 if selected_portfolio_name != "(Create New Portfolio)" and selected_portfolio_name in st.session_state.portfolios:
                      del st.session_state.portfolios[selected_portfolio_name]
                      save_portfolios_to_file(st.session_state.portfolios)
                      st.success(f"Portfolio '{selected_portfolio_name}' deleted.")
                      st.session_state.portfolio_select = "(Create New Portfolio)"; st.rerun()
                 elif selected_portfolio_name == "(Create New Portfolio)": st.warning("Cannot delete '(Create New Portfolio)'.")
                 else: st.error(f"Portfolio '{selected_portfolio_name}' not found.")
    with col_display:
        st.subheader("Saved Portfolios")
        if not st.session_state.portfolios: st.info("No portfolios defined yet.")
        else:
            st.markdown("**List of Saved Portfolios:**")
            sorted_names = sorted(st.session_state.portfolios.keys())
            if len(sorted_names) > 10: st.info("Showing first 10 portfolios. Scroll down for more.")
            displayed_count = 0
            for name in sorted_names:
                with st.expander(f"{name}", expanded=(displayed_count < 5)):
                    portfolio_data = st.session_state.portfolios[name]
                    df = pd.DataFrame(list(portfolio_data.items()), columns=['Ticker', 'Weight'])
                    df['Weight %'] = (df['Weight'] * 100).map('{:.2f}%'.format)
                    st.dataframe(df[['Ticker', 'Weight %']], hide_index=True, use_container_width=True)
                    sum_w = df['Weight'].sum()
                    sum_text = f"Total Weight: {sum_w:.4f}"
                    if np.isclose(sum_w, 1.0): st.caption(f"✅ {sum_text}")
                    else: st.caption(f"⚠️ {sum_text} (Should be 1.0)")
                displayed_count += 1

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
        # available_opt_tickers = prices_for_opt.columns.tolist() # No se usa explicitamente
        if len(prices_for_opt.columns) < 2: # Chequeo directo en vez de usar variable
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
                    # weights_optimized = None # No es necesario inicializar a None
                    if opt_objective == "Maximize Sharpe Ratio (Best Risk-Adjusted Return)":
                        ef.max_sharpe(risk_free_rate=risk_free_rate)
                    elif opt_objective == "Minimize Volatility (Lowest Risk)":
                        ef.min_volatility()
                    elif opt_objective == "Maximize Quadratic Utility (Balance Risk/Return)":
                        risk_aversion_param = 2 # Podría ser un input del usuario
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

def page_event_analyzer():
    st.header("📰 Event Analyzer (Simple Demo)")
    st.warning("""
    **DISCLAIMER:** This tool performs a **very basic keyword search**.
    It **DOES NOT** understand context, nuance, sarcasm, or the actual market impact of news.
    Results are **NOT** financial advice and should **NOT** be used for investment decisions.
    This is purely a conceptual demonstration.
    """)
    st.markdown("---")
    positive_keywords = [
        "crecimiento", "supera", "acuerdo", "beneficio", "ganancia", "upgrade",
        "expansión", "éxito", "innovación", "alianza", "optimista", "demanda alta",
        "récord", "lanzamiento exitoso", "aumento", "mejora"
    ]
    negative_keywords = [
        "caída", "pérdida", "impuesto", "arancel", "investigación", "demanda",
        "retraso", "multa", "riesgo", "downgrade", "incertidumbre", "problema",
        "descenso", "crisis", "recorte", "competencia", "regulación", "trump",
        "aumento de impuestos"
    ]
    positive_keywords = [k.lower() for k in positive_keywords]
    negative_keywords = [k.lower() for k in negative_keywords]
    st.subheader("Input Data")
    news_text = st.text_area("Paste News Snippet Here:", height=150, key="event_text")
    affected_tickers_str = st.text_input("Enter Ticker(s) Potentially Affected (comma-separated):", key="event_tickers", placeholder="e.g., AAPL, TSLA")
    analyze_button = st.button("Analyze Text", key="event_analyze_button")
    st.markdown("---")
    st.subheader("Analysis Results")
    if analyze_button and news_text and affected_tickers_str:
        tickers = [t.strip().upper() for t in affected_tickers_str.split(',') if t.strip()]
        if not tickers:
            st.warning("Please enter at least one ticker symbol.")
        else:
            text_lower = news_text.lower()
            positive_score = 0
            negative_score = 0
            found_pos_keywords = []
            found_neg_keywords = []
            for keyword in positive_keywords:
                if keyword in text_lower:
                    positive_score += 1
                    found_pos_keywords.append(keyword)
            for keyword in negative_keywords:
                 if keyword in text_lower:
                    negative_score += 1
                    found_neg_keywords.append(keyword)
            signal = "❓ **UNCLEAR SIGNAL**"
            signal_color = "orange"
            if positive_score > negative_score:
                signal = "📈 **POTENTIAL UP SIGNAL**"
                signal_color = "green"
            elif negative_score > positive_score:
                signal = "📉 **POTENTIAL DOWN SIGNAL**"
                signal_color = "red"
            for ticker in tickers:
                st.markdown(f"### Signal for {ticker}:")
                st.markdown(f"<span style='color:{signal_color}; font-size: 1.2em;'>{signal}</span>", unsafe_allow_html=True)
            st.markdown("---")
            col_kw1, col_kw2 = st.columns(2)
            with col_kw1:
                st.metric("Positive Keyword Hits", positive_score)
                if found_pos_keywords: st.caption(f"Found: {', '.join(list(set(found_pos_keywords)))}")
            with col_kw2:
                st.metric("Negative Keyword Hits", negative_score)
                if found_neg_keywords: st.caption(f"Found: {', '.join(list(set(found_neg_keywords)))}")
    elif analyze_button:
        st.warning("Please provide both the news text and at least one ticker symbol.")
    else:
        st.info("Enter text and tickers above and click 'Analyze Text'.")


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

# --- Initialize Session State ---
default_session_values = {
    'selected_page': "Welcome Page",
    'entities': [], 'dataframes': {}, 'data_type': 'returns',
    'apply_outlier_treatment': False, 'iqr_factor': 1.5,
    'iol_logged_in': False, 'iol_access_token': None, 'iol_token_data': None,
    'iol_user': None, 'iol_last_data': None, 'iol_last_error': None,
    'iol_parsed_tickers_info': [], 'iol_parsed_tickers_list': []
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

if ticker_input: tickers_forecast = [t.strip().upper() for t in re.split('[,\s]+', ticker_input) if t.strip()] # Renombrar para evitar conflicto
else: tickers_forecast = []

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
st.sidebar.subheader("IOL: Cargar Tickers (BCBA)")
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
    "Create/Edit Portfolios",
    "View Portfolio Returns",
    "InvertirOnline API",
    "Event Analyzer (Simple Demo)",
    "--- Forecasting ---",
    "Data Visualization",
    "Series Decomposition",
    "Stationarity & Lags",
    "Forecasting Models"
]
selectable_page_options = [p for p in page_options if not p.startswith("---")]
# forecasting_data_loaded = st.session_state.get('entities', []) # No se usa explicitamente aqui

if 'selected_page' not in st.session_state or st.session_state.selected_page not in selectable_page_options:
    st.session_state.selected_page = "Welcome Page"

try:
    if st.session_state.selected_page not in page_options: # Asegurar que es una opción válida
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

# --- Page Routing ---
current_page_to_display = st.session_state.selected_page
if current_page_to_display == "Welcome Page":
    main_page()
elif current_page_to_display == "Create/Edit Portfolios":
    page_create_portfolio()
elif current_page_to_display == "View Portfolio Returns":
    page_view_portfolio_returns()
elif current_page_to_display == "InvertirOnline API":
    page_invertir_online()
elif current_page_to_display == "Event Analyzer (Simple Demo)":
    page_event_analyzer()
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


