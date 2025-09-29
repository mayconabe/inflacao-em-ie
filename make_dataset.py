# make_dataset.py — versão simplificada (só alimentos + INPC, sem pesos)
import pandas as pd, ipeadatapy as idpy

ANOS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

def _as_datetime(df):
    if 'DATE' in df.columns:
        return pd.to_datetime(df['DATE'])
    return pd.to_datetime(dict(year=df['YEAR'], month=df['MONTH'], day=1))

def _val_col(df):
    return [c for c in df.columns if str(c).upper().startswith('VALUE')][0]

def ipca_alimentos(ano):
    df = idpy.timeseries('PRECOS12_IPCAAB12', year=ano)
    d = _as_datetime(df)
    v = pd.to_numeric(df[_val_col(df)], errors='coerce')
    return pd.DataFrame({'data': d, 'ipca_alimentos_mom': v})

def inpc_all():
    df = idpy.timeseries('PRECOS12_INPC12')
    d = _as_datetime(df)
    v = pd.to_numeric(df[_val_col(df)], errors='coerce')
    base = pd.DataFrame({'data': d, 'inpc_indice': v}).sort_values('data')
    base['inpc_mom'] = base['inpc_indice'].pct_change()*100
    return base[['data','inpc_mom']]

def monta(ano, inpc):
    alim = ipca_alimentos(ano)
    df = alim.merge(inpc, on='data', how='left')
    df['peso_alimentos'] = 100
    df['IE_essenciais_mom'] = df['ipca_alimentos_mom']
    df['contrib_ipca_alimentos_pp'] = df['ipca_alimentos_mom']
    return df[['data','ipca_alimentos_mom','peso_alimentos',
               'IE_essenciais_mom','inpc_mom','contrib_ipca_alimentos_pp']]

def main():
    inpc = inpc_all()
    frames=[]
    for a in ANOS:
        try:
            frames.append(monta(a, inpc))
        except Exception as e:
            print(f"[WARN] {a}: {e}")
    full=pd.concat(frames).sort_values('data')
    full.to_csv('dataset_analitico.csv',index=False)
    print('Gerado dataset_analitico.csv',full.tail())

if __name__=="__main__":
    main()
