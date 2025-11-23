import pandas as pd
import numpy as np
import re
from datetime import datetime

def parse_month_label(x: str) -> pd.Timestamp:
    """Converte datas no formato MM-YYYY (ex: 01-2019)."""
    if isinstance(x, pd.Timestamp): return pd.Timestamp(x.year, x.month, 1)
    s = str(x).strip()
    if not s or s.lower() in ('nan', 'none', 'nat'): return pd.NaT

    # Padrão principal do seu arquivo: MM-YYYY
    if m := re.match(r'^(\d{1,2})[-/](\d{4})$', s):
        return pd.Timestamp(int(m.group(2)), int(m.group(1)), 1)
    
    # Fallback
    if m := re.match(r'^(\d{4})[-/](\d{1,2})$', s): return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    
    return pd.to_datetime(s, errors='coerce')

def normalize_date_column(df: pd.DataFrame, col='data') -> pd.DataFrame:
    if df.empty or col not in df.columns: return df
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated()]
    out[col] = out[col].apply(parse_month_label)
    out = out.dropna(subset=[col])
    if out.empty: return out
    out[col] = pd.to_datetime(out[col])
    out = out[out[col].dt.year > 1990] # Filtro de segurança
    return out.sort_values(col).reset_index(drop=True)

def ensure_ms_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors='coerce')
    s = s.sort_index()
    return s.asfreq('MS')

def clean_and_format_dieese(df):
    """Transforma colunas de cidades em linhas (Formato Longo)."""
    if df.empty: return pd.DataFrame()
    
    # Garante que temos a coluna data
    if 'data' not in df.columns:
        # Se por acaso o ingestion falhar, pega a primeira coluna
        df = df.rename(columns={df.columns[0]: 'data'})

    # Transforma de Wide (Colunas) para Long (Linhas)
    # id_vars=['data'] trava a data, o resto vira 'capital' e 'valor_cesta'
    df_long = df.melt(id_vars=['data'], var_name='capital', value_name='valor_cesta')
    
    # Normaliza data
    df_long = normalize_date_column(df_long, col='data')
    
    # Garante numérico (Seu arquivo já vem com ponto, então é direto)
    if 'valor_cesta' in df_long.columns:
        df_long['valor_cesta'] = pd.to_numeric(df_long['valor_cesta'], errors='coerce')
    
    # Remove nulos (ex: cidades que não tinham dados em 2019)
    return df_long.dropna(subset=['data', 'capital', 'valor_cesta']).sort_values(['capital', 'data'])

def process_pipeline(df_analitico, df_dieese):
    print("--- [Processing] Processando dados...")
    
    # Analítico
    df_analitico = normalize_date_column(df_analitico, col='data')
    if not df_analitico.empty:
        df_analitico = df_analitico.set_index('data')
        for c in ['IE_essenciais_mom', 'inpc_mom', 'ipca_alimentos_mom']:
            if c in df_analitico.columns:
                df_analitico[c] = pd.to_numeric(df_analitico[c], errors='coerce')
    
    # Dieese
    df_dieese = clean_and_format_dieese(df_dieese)
    
    print(f"    Dados Prontos: {len(df_dieese)} registros consolidados.")
    return df_analitico, df_dieese