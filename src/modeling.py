import pandas as pd
import numpy as np
import joblib
import os
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
# Importa do módulo local
try:
    from src.data_processing import ensure_ms_index
except ModuleNotFoundError:
    # Fallback para rodar direto da pasta raiz
    from data_processing import ensure_ms_index

warnings.filterwarnings("ignore")

def train_sarimax_model(s_cesta, s_ipca, capital_name):
    """Treina um modelo SARIMAX para uma capital específica."""
    
    # Feature Engineering (Exógenas)
    # Criamos um Salário Mínimo dummy constante apenas para viabilizar o fit
    # Num cenário real, você carregaria o histórico do salário mínimo aqui.
    sal_min_dummy = pd.Series(1320.0, index=s_cesta.index)
    
    X = pd.DataFrame(index=s_cesta.index)
    X['ipca_alimentos_mom'] = s_ipca.reindex(s_cesta.index).fillna(0.0)
    X['sal_min'] = sal_min_dummy
    
    y = s_cesta
    
    # Split Treino/Teste
    if len(y) < 12:
        return None, None # Dados insuficientes (ex: capitais novas 2025)
        
    cut = -6
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    
    try:
        # Ordem fixa para simplificação no pipeline automatizado
        model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        
        # Avaliação rápida
        forecast = model_fit.get_forecast(steps=len(y_test), exog=X_test)
        mae = mean_absolute_error(y_test, forecast.predicted_mean)
        
        # Retreina com TODOS os dados para salvar o modelo final de produção
        final_model = SARIMAX(y, exog=X, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
                             enforce_stationarity=False, enforce_invertibility=False)
        final_fit = final_model.fit(disp=False)
        
        return final_fit, mae
    except Exception as e:
        print(f"Erro treinando {capital_name}: {e}")
        return None, None

def run_training_pipeline(df_analitico, df_dieese):
    print("--- [Modeling] Iniciando treinamento de modelos...")
    os.makedirs('models', exist_ok=True)
    
    if df_analitico.empty or df_dieese.empty:
        print("Dados insuficientes para modelagem.")
        return

    s_ipca = ensure_ms_index(df_analitico['ipca_alimentos_mom'].dropna())
    capitais = df_dieese['capital'].unique()
    
    results = {}
    
    print(f"Capitais encontradas: {len(capitais)}")
    
    for cap in capitais:
        print(f"Treinando: {cap}...", end=" ")
        df_cap = df_dieese[df_dieese['capital'] == cap].set_index('data')['valor_cesta'].sort_index()
        s_cesta = ensure_ms_index(df_cap)
        
        model, mae = train_sarimax_model(s_cesta, s_ipca, cap)
        
        if model:
            print(f"OK (MAE: {mae:.2f})")
            results[cap] = {
                'model': model,
                'mae': mae,
                'last_date': s_cesta.index.max(),
                'last_exog': model.model.exog[-1] # Salva os últimos exógenos para projeção futura
            }
        else:
            print("Pulado (Dados insuficientes/Erro)")
            
    # Serialização: Salva o dicionário com todos os modelos treinados
    # Nível de compressão 3 é um ótimo equilíbrio entre tamanho e velocidade
    joblib.dump(results, 'models/all_capitals_models.pkl', compress=3)
    print("--- [Modeling] Pipeline finalizado. Modelos salvos em 'models/all_capitals_models.pkl'")

if __name__ == "__main__":
    # Para rodar independente:
    from data_ingestion import load_raw_data
    from data_processing import process_pipeline
    
    raw_ana, raw_dieese = load_raw_data()
    proc_ana, proc_dieese = process_pipeline(raw_ana, raw_dieese)
    run_training_pipeline(proc_ana, proc_dieese)