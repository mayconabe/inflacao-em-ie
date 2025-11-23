import pandas as pd
import os

def load_raw_data(base_path='data/raw'):
    """Carrega os datasets brutos (Analítico e Novo Exporta.csv)."""
    print("--- [Ingestion] Carregando dados brutos...")
    
    # 1. Carrega Analítico
    path_analitico = os.path.join(base_path, 'dataset_analitico.csv')
    if os.path.exists(path_analitico):
        df_analitico = pd.read_csv(path_analitico, dtype=str)
        print(f"    ✔ Analítico carregado: {len(df_analitico)} registros.")
    else:
        print(f"    ❌ Analítico não encontrado: {path_analitico}")
        df_analitico = pd.DataFrame()

    # 2. Carrega DIEESE (exporta.csv)
    path_dieese = os.path.join(base_path, 'exporta.csv')
    
    if os.path.exists(path_dieese):
        # O Pandas detecta automaticamente que a primeira coluna é índice (datas)
        df_dieese = pd.read_csv(path_dieese)
        
        # Correção: Tira a data do índice e transforma em coluna normal chamada 'data'
        df_dieese = df_dieese.reset_index()
        df_dieese = df_dieese.rename(columns={'index': 'data'})
        
        print(f"    ✔ DIEESE carregado: {len(df_dieese)} meses x {len(df_dieese.columns)} colunas.")
    else:
        print(f"    ❌ Arquivo 'exporta.csv' não encontrado em {base_path}.")
        df_dieese = pd.DataFrame()
            
    return df_analitico, df_dieese

if __name__ == "__main__":
    load_raw_data()