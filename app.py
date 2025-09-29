# app.py
# Dashboard: Inflação de Alimentos (IPCA) vs INPC — visão mensal/anual + Poder de Compra (DIEESE)
# + Previsões (ETS para IPCA e SARIMAX para Cesta) — sem cenários
#
# Requisitos:
#   pip install streamlit pandas numpy plotly statsmodels scikit-learn

import os
import io
import re
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ===== Modelos / métricas =====
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAVE_SM = True
except Exception:
    HAVE_SM = False

try:
    from sklearn.metrics import mean_absolute_error
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# ===== Config Streamlit =====
st.set_page_config(page_title='Alimentos x INPC — Mensal e Anual', layout='wide')


# =========================
# Helpers gerais
# =========================
def load_csv(path: str, parse_dates=('data',)):
    """Carrega CSV se existir; caso contrário, DataFrame vazio. Converte colunas de data se presentes."""
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    for c in parse_dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df


def to_download_button(df: pd.DataFrame, filename: str, label: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding='utf-8')
    st.download_button(label=label, data=buf.getvalue(), file_name=filename, mime='text/csv')


def _fmt_metric(x, suffix=''):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return '—'
    return f'{x:.2f}{suffix}'


# =========================
# Normalização de datas (VERSÃO CORRIGIDA E MAIS ROBUSTA)
# =========================
def parse_month_label(x: str) -> pd.Timestamp:
    """
    Converte vários formatos de data para Timestamp('YYYY-MM-01').
    NOVA LÓGICA: Trata anos com 2 dígitos de forma inteligente.
    """
    if isinstance(x, pd.Timestamp):
        return pd.Timestamp(x.year, x.month, 1)
    s = str(x).strip()
    if not s or s.lower() in ('nan', 'none', 'nat'):
        return pd.NaT

    # Padrão: YYYY-MM ou YYYY/MM
    m = re.match(r'^(\d{4})[-/](\d{1,2})$', s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return pd.Timestamp(y, mo, 1)

    # Padrão: MM-YYYY ou MM/YYYY
    m = re.match(r'^(\d{1,2})[-/](\d{4})$', s)
    if m:
        mo, y = int(m.group(1)), int(m.group(2))
        return pd.Timestamp(y, mo, 1)
        
    # >>> INÍCIO DA CORREÇÃO PARA BRASÍLIA <<<
    # Padrão: MM-YY ou MM/YY (ex: '05/99', '10-21')
    m = re.match(r'^(\d{1,2})[-/](\d{2})$', s)
    if m:
        mo, y_short = int(m.group(1)), int(m.group(2))
        current_century_cutoff = (datetime.now().year % 100) + 5
        if y_short <= current_century_cutoff:
            y = 2000 + y_short # 00-29 vira 2000-2029
        else:
            y = 1900 + y_short # 70-99 vira 1970-1999
        return pd.Timestamp(y, mo, 1)
    # >>> FIM DA CORREÇÃO <<<

    # Padrão: YYYYMM ou MMYYYY (6 dígitos)
    m = re.match(r'^(\d{6})$', s)
    if m:
        v = m.group(1)
        if v.startswith(('20', '19')):
            y, mo = int(v[:4]), int(v[4:])
        else:
            mo, y = int(v[:2]), int(v[2:])
        return pd.Timestamp(y, mo, 1)

    # Fallback para outros formatos (ex: '2020-01-15')
    try:
        ts = pd.to_datetime(s, errors='coerce')
        if pd.isna(ts):
            return pd.NaT
        return pd.Timestamp(ts.year, ts.month, 1)
    except Exception:
        return pd.NaT


def normalize_date_column(df: pd.DataFrame, col='data') -> pd.DataFrame:
    """Normaliza df[col] para datetime YYYY-MM-01 e ordena; ignora linhas inválidas.
       Se houver colunas duplicadas, mantém a primeira ocorrência."""
    if df.empty or col not in df.columns:
        return df
    out = df.copy()

    # remove colunas duplicadas (mantém a primeira)
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]

    out[col] = out[col].apply(parse_month_label)
    out = out.dropna(subset=[col])
    out[col] = pd.to_datetime(out[col], errors='coerce')
    # Remove datas muito antigas que podem ser erros de parsing
    out = out[out[col].dt.year > 1980]
    out = out.sort_values(col).reset_index(drop=True)
    return out


def ensure_ms_index(s: pd.Series) -> pd.Series:
    """Garante índice datetime mensal (MS) ordenado."""
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors='coerce')
    s = s.sort_index()
    # Assegura frequência mensal e preenche lacunas se houver
    s = s.asfreq('MS')
    return s


# =========================
# Forecast simples para IPCA (ETS/naive)
# =========================
def monthly_forecast(series: pd.Series, h: int = 6, method: str = 'auto'):
    s = ensure_ms_index(series.dropna())
    if len(s) < 1:
        idx_fut = pd.date_range(start=pd.Timestamp.today().normalize() - pd.DateOffset(months=1), periods=h, freq='MS')
        return pd.DataFrame({'data': idx_fut, 'forecast': np.zeros(h)})

    if HAVE_SM and method in ('auto', 'ets') and len(s) >= 6:
        try:
            model = ExponentialSmoothing(
                s.astype(float), trend=None, seasonal=None, initialization_method='estimated'
            ).fit(optimized=True)
            fc = model.get_forecast(steps=h)
            pm = fc.predicted_mean
            return pd.DataFrame({'data': pm.index, 'forecast': pm.values})
        except Exception:
            pass  # Fallback para o método naive
    
    last_val = s.iloc[-1]
    idx_fut = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(), periods=h, freq='MS')
    return pd.DataFrame({'data': idx_fut, 'forecast': np.repeat(float(last_val), h)})


# =========================
# Carregadores DIEESE
# =========================
def load_dieese_cesta(path='dieese_cesta_2022.csv'):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # normaliza coluna data
    if 'data' not in df.columns:
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'data'})
        elif 'mes' in df.columns:
            df = df.rename(columns={'mes': 'data'})
    
    if 'data' not in df.columns: return pd.DataFrame() # se não encontrou coluna de data
    
    df = normalize_date_column(df, col='data')
    if 'capital' not in df.columns and 'cidade' in df.columns:
        df = df.rename(columns={'cidade': 'capital'})
    if 'valor_cesta' not in df.columns and 'valor' in df.columns:
        df = df.rename(columns={'valor': 'valor_cesta'})
    
    # tipos
    if 'valor_cesta' in df.columns:
        df['valor_cesta'] = (
            df['valor_cesta'].astype(str).str.replace(',', '.', regex=False)
            .str.replace(' ', '', regex=False)
        )
        df['valor_cesta'] = pd.to_numeric(df['valor_cesta'], errors='coerce')
    
    df = df.dropna(subset=['data', 'capital', 'valor_cesta'])
    return df.sort_values(['capital', 'data']).reset_index(drop=True)


def load_dieese_cesta_detalhado(path='dieese_cesta_2022_detalhado.csv'):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'data' not in df.columns:
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'data'})
        elif 'mes' in df.columns:
            df = df.rename(columns={'mes': 'data'})

    if 'data' not in df.columns: return pd.DataFrame()

    df = normalize_date_column(df, col='data')
    if 'capital' not in df.columns and 'cidade' in df.columns:
        df = df.rename(columns={'cidade': 'capital'})
    
    # converter todas colunas numéricas
    for c in [c for c in df.columns if c not in ('data', 'capital')]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(',', '.', regex=False).str.replace(' ', '', regex=False),
            errors='coerce'
        )
    base_cols = {'data', 'capital', 'valor_cesta'}
    if not base_cols.issubset(df.columns):
        return pd.DataFrame()
    return df.dropna(subset=['data', 'capital']).sort_values(['capital', 'data']).reset_index(drop=True)


# =========================
# SARIMAX (Cesta com exógenas)
# =========================
def _build_exog(cesta_idx, ipca_alimentos: pd.Series, sal_min: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=cesta_idx)
    X['ipca_alimentos_mom'] = ensure_ms_index(ipca_alimentos).reindex(cesta_idx).fillna(0.0)
    X['sal_min'] = ensure_ms_index(sal_min).reindex(cesta_idx).ffill().bfill()
    return X


def fit_sarimax_cesta(cesta_series: pd.Series,
                      ipca_alimentos: pd.Series,
                      sal_min: pd.Series,
                      order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
    if not HAVE_SM:
        return None, None, None, {'MAE': None, 'MAPE_%': None}

    y0 = ensure_ms_index(cesta_series.dropna()).rename('y')
    X0 = _build_exog(y0.index, ipca_alimentos, sal_min)
    base = pd.concat([y0, X0], axis=1).dropna()

    if base.empty or len(base) < 24: # Aumenta o mínimo para robustez sazonal
        return None, None, X0, {'MAE': None, 'MAPE_%': None}

    y = base['y']
    X = base.drop(columns=['y'])

    # validação (últimos 12 meses, se houver dados)
    val_len = min(12, max(3, len(y) // 5))
    cut = -val_len
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]

    model = SARIMAX(y, exog=X, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    mae = mape = None
    if HAVE_SK and len(y_te) > 0:
        try:
            val_model = SARIMAX(y_tr, exog=X_tr, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            preds = val_model.get_forecast(steps=len(y_te), exog=X_te).predicted_mean
            mae = mean_absolute_error(y_te, preds)
        except Exception:
            mae = None
    
    return model, res, X, {'MAE': mae}


def forecast_cesta(res, last_train_date: pd.Timestamp, horizon=6):
    """
    Previsão h passos à frente SEM cenários.
    Usa os últimos valores de exógenas do treino, repetidos no horizonte.
    """
    if res is None:
        idx_fut = pd.date_range(pd.Timestamp.today().replace(day=1), periods=horizon, freq='MS')
        return pd.DataFrame({'data': idx_fut, 'cesta_fc': np.nan, 'lo80': np.nan, 'hi80': np.nan, 'sal_min_usado': np.nan})

    if not isinstance(last_train_date, pd.Timestamp):
        last_train_date = pd.to_datetime(last_train_date)

    idx_fut = pd.date_range(last_train_date + pd.offsets.MonthBegin(), periods=horizon, freq='MS')

    # últimos exógenos do treino (ordem: ['ipca_alimentos_mom','sal_min'])
    try:
        ipca_last = float(res.model.exog[-1, 0])
        sal_last = float(res.model.exog[-1, 1])
    except Exception:
        ipca_last, sal_last = 0.0, 0.0

    Xf = pd.DataFrame(index=idx_fut)
    Xf['ipca_alimentos_mom'] = ipca_last
    Xf['sal_min'] = sal_last

    fc = res.get_forecast(steps=horizon, exog=Xf)
    pm = fc.predicted_mean.rename('cesta_fc')
    ci = fc.conf_int(alpha=0.2)  # 80% CI
    ci.columns = ['lo80', 'hi80']
    out = pd.concat([pm, ci], axis=1).reset_index().rename(columns={'index': 'data'})
    out['sal_min_usado'] = sal_last
    return out


# =========================
# Carregar dados
# =========================
df = load_csv('dataset_analitico.csv')
dieese = load_dieese_cesta('dieese_cesta_2022.csv')
dieese_det = load_dieese_cesta_detalhado('dieese_cesta_2022_detalhado.csv')
alims_file = 'ipca_alimentos_2022.csv'
alims_opt = load_csv(alims_file) if os.path.exists(alims_file) else pd.DataFrame()

st.title('Inflação de Alimentos (IPCA) vs INPC — Visão Mensal e Anual')
st.caption('Fontes: Ipeadata/IBGE (INPC(Índice Nacional de Preços ao Consumidor)), DIEESE (Cesta Básica).')
st.markdown(f'Tema: Qual a influência da inflação no preço bens essenciais como alimentos? Qual o impacto da inflação desses preços no poder de compra de familias de baixa renda?')
st.markdown(f'Objetivo: Identificar e explorar o impacto da inflação de bens essenciais (alimentos) sobre o poder de compra das famílias de baixa renda.')

if df.empty:
    st.error('Arquivo `dataset_analitico.csv` não encontrado. Este arquivo é essencial para o dashboard.')
    st.stop()

# Normaliza df principal e define índice
df = normalize_date_column(df, col='data').set_index('data')

# acumulado no ano (jan → mês)
df['IE_acum_no_ano'] = (1 + df['IE_essenciais_mom'] / 100).groupby(df.index.year).cumprod() - 1
df['INPC_acum_no_ano'] = (1 + df['inpc_mom'] / 100).groupby(df.index.year).cumprod() - 1

anos_disponiveis = sorted(df.index.year.unique().tolist(), reverse=True)
ano_foco = st.sidebar.selectbox('Ano foco (gráficos mensais)', anos_disponiveis, index=0)
df_foco = df[df.index.year == ano_foco]

# =========================
# Abas
# =========================
# tab1, tab2, tab3, tab4, tab5 = st.tabs([
#     'Visão Geral', 'Alimentos', 'Poder de Compra', 'Previsões', 'Relatório'
# ])

tab1, tab3, tab4, tab5 = st.tabs([
    'Visão Geral', 'Poder de Compra', 'Previsões', 'Relatório'
])

# ========== Visão Geral ==========
with tab1:
    st.header(f'Análise Comparativa: Inflação de Alimentos (IE) vs. INPC em {ano_foco}')
    
    if not df_foco.empty:
        # --- 1. Funções para gerar os textos de ajuda dinâmicos ---
        def get_help_media_ie(ie_val, inpc_val):
            if ie_val > inpc_val:
                diferenca = (ie_val / inpc_val)
                return f"Neste ano, a inflação média da comida foi {diferenca:.1f}x maior que a inflação geral. Isso mostra que os alimentos foram o principal fator de pressão no custo de vida."
            else:
                return "Neste ano, a inflação média da comida foi menor que a geral. Isso sugere que outros setores (energia, transporte, etc.) pesaram mais no orçamento."

        def get_help_meses_piores(meses_val, total_meses):
            if total_meses == 0: return "Sem dados."
            percentual_meses = (meses_val / total_meses) * 100
            if percentual_meses >= 75:
                return f"Em {percentual_meses:.0f}% dos meses, a alta da comida superou a inflação geral. Isso indica uma pressão persistente e generalizada sobre o poder de compra."
            elif percentual_meses >= 50:
                return f"Em {percentual_meses:.0f}% dos meses (a maior parte do ano), a comida ficou mais cara que a média dos outros itens, indicando um problema consistente."
            else:
                return f"Apesar de não ser na maior parte do tempo, em {percentual_meses:.0f}% dos meses a comida subiu mais que a inflação geral, mostrando picos de volatilidade que afetam o orçamento."

        def get_help_max_contrib(contrib_val):
            return f"No pico do ano, os alimentos foram responsáveis, sozinhos, por {contrib_val:.2f} pontos percentuais de toda a inflação (INPC) daquele mês. Um indicador claro do impacto desproporcional da comida em momentos de crise."

        # --- 2. Cálculo dos valores ---
        media_ie = df_foco['IE_essenciais_mom'].mean()
        media_inpc = df_foco['inpc_mom'].mean()
        meses_piores = int((df_foco['IE_essenciais_mom'] > df_foco['inpc_mom']).sum())
        total_meses_no_ano = len(df_foco)
        max_contrib = df_foco["contrib_ipca_alimentos_pp"].max()
        
        # --- 3. Geração dos textos dinâmicos ---
        help_ie_dinamico = get_help_media_ie(media_ie, media_inpc)
        help_meses_dinamico = get_help_meses_piores(meses_piores, total_meses_no_ano)
        help_contrib_dinamico = get_help_max_contrib(max_contrib)
        
        # --- 4. Criação dos indicadores com os 'helps' dinâmicos ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Média Mensal IE (Alimentos)', f'{media_ie:.2f}%', help=help_ie_dinamico)
        c2.metric('Média Mensal INPC', f'{media_inpc:.2f}%', help="Representa a média da inflação geral para famílias de baixa renda.")
        c3.metric('Meses IE > INPC', f'{meses_piores} de {total_meses_no_ano}', help=help_meses_dinamico)
        c4.metric('Máx. Contrib. Alimentos', f'{max_contrib:.2f} p.p.', help=help_contrib_dinamico)

        ie_acum_pct = ((1 + df_foco['IE_essenciais_mom'] / 100).prod() - 1) * 100
        st.metric(f'IE Acumulado em {ano_foco}', f'{ie_acum_pct:.1f}%')
        
        st.markdown(
            f'Ao longo de **{ano_foco}**, o índice de alimentos (IE) acumulou **{ie_acum_pct:.1f}%**. '
            f'Isso significa que, no fim do ano, uma família de baixa renda precisou gastar **{ie_acum_pct:.1f}% a mais** '
            f'para comprar a **mesma cesta de alimentos**, impactando diretamente seu poder de compra.'
        )

        st.markdown('---')
        st.subheader(f'Variação Mensal ({ano_foco}) — IE vs. INPC')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_foco.index, y=df_foco['IE_essenciais_mom'], mode='lines+markers', name='IE (Alimentos)'))
        fig.add_trace(go.Scatter(x=df_foco.index, y=df_foco['inpc_mom'], mode='lines+markers', name='INPC'))
        fig.update_layout(height=420, xaxis_title='Mês', yaxis_title='% ao mês', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f'Inflação Acumulada no Ano ({ano_foco}) — IE vs. INPC')
        fig_ac = go.Figure()
        fig_ac.add_trace(go.Scatter(x=df_foco.index, y=100 * df_foco['IE_acum_no_ano'], mode='lines+markers', name='IE Acumulado'))
        fig_ac.add_trace(go.Scatter(x=df_foco.index, y=100 * df_foco['INPC_acum_no_ano'], mode='lines+markers', name='INPC Acumulado'))
        fig_ac.update_layout(height=420, xaxis_title='Mês', yaxis_title='% Acumulado', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_ac, use_container_width=True)
    else:
        st.warning(f"Não há dados para o ano {ano_foco}.")

    st.markdown('---')
    st.subheader('Visão Anual Comparativa — IE vs. INPC Acumulados')
    def _acum_por_ano(serie_mensal_pct: pd.Series) -> pd.Series:
        g = (1 + serie_mensal_pct / 100).groupby(serie_mensal_pct.index.year).prod() - 1
        return (g * 100).rename('acumulado_%')

    ie_by_year = _acum_por_ano(df['IE_essenciais_mom'])
    inpc_by_year = _acum_por_ano(df['inpc_mom'])
    tbl = pd.concat([ie_by_year.rename('IE_acum_%'), inpc_by_year.rename('INPC_acum_%')], axis=1)
    
    fig_year = go.Figure()
    fig_year.add_bar(name='IE Acumulado (%)', x=tbl.index.astype(str), y=tbl['IE_acum_%'])
    fig_year.add_bar(name='INPC Acumulado (%)', x=tbl.index.astype(str), y=tbl['INPC_acum_%'])
    fig_year.update_layout(barmode='group', height=420, xaxis_title='Ano', yaxis_title='% no Ano', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_year, use_container_width=True)
    st.dataframe(tbl.round(2).reset_index().rename(columns={'index': 'ano'}), use_container_width=True)

# ========== Alimentos ==========
# with tab2:
#     st.header(f'IPCA - Alimentação e Bebidas (var. % mensal) em {ano_foco}')
#     if alims_opt.empty:
#         st.info('Para esta aba, adicione o arquivo `ipca_alimentos_2022.csv` (colunas: `data`, `valor`).')
#     else:
#         alims_opt = normalize_date_column(alims_opt, col='data')
#         aux = alims_opt.set_index('data').sort_index()
#         aux = aux[aux.index.year == ano_foco] if not aux.empty else aux
#         if not aux.empty:
#             st.line_chart(aux[['valor']].rename(columns={'valor': 'IPCA Alimentos (% m/m)'}))
#         else:
#             st.warning(f"Não há dados de IPCA Alimentos para o ano {ano_foco}.")

# ========== Poder de Compra ==========
with tab3:
    st.header(f'Composição da Cesta de Alimentos em {ano_foco}')
    if dieese_det.empty:
        st.warning('Arquivo `dieese_cesta_2022_detalhado.csv` não encontrado ou inválido.')
    else:
        caps_det = sorted(dieese_det['capital'].unique())
        cap_sel_det = st.selectbox('Capital', caps_det, index=caps_det.index('São Paulo') if 'São Paulo' in caps_det else 0, key='det_cap')
        
        det = dieese_det[(dieese_det['capital'] == cap_sel_det) & (dieese_det['data'].dt.year == ano_foco)].copy()

        if det.empty:
            st.warning(f'Não há dados detalhados para {cap_sel_det} em {ano_foco}.')
        else:
            possiveis_itens = ['carne', 'leite', 'feijao', 'arroz', 'farinha', 'batata', 'tomate', 'pao', 'cafe', 'banana', 'acucar', 'oleo', 'manteiga']
            itens_existentes = [c for c in possiveis_itens if c in det.columns and det[c].notna().any()]
            
            default_selection = itens_existentes[:5] if len(itens_existentes) >= 5 else itens_existentes
            selecao = st.multiselect('Selecione os itens para visualizar', itens_existentes, default=default_selection)

            if selecao:
                det_plot = det.set_index('data')

                # --- INÍCIO DA ALTERAÇÃO DO GRÁFICO ---
                # O gráfico antigo de barras foi substituído por este de linhas.
                st.subheader('Evolução do Custo dos Itens (R$)')
                
                fig_linhas = go.Figure()
                for item in selecao:
                    fig_linhas.add_trace(go.Scatter(
                        x=det_plot.index, 
                        y=det_plot[item],
                        mode='lines+markers',
                        name=item.capitalize()
                    ))
                
                fig_linhas.update_layout(
                    xaxis_title='Mês',
                    yaxis_title='Custo Mensal (R$)',
                    legend_title='Itens'
                )
                st.plotly_chart(fig_linhas, use_container_width=True)

# ========== Previsões ==========
with tab4:
    st.header('Previsões de Curto Prazo')
    st.markdown("As variáveis utilizadas foram ipca_alimentos_mom(Inflação mensal dos alimentos) e valor_cesta(Valor da cesta básica).")

    help_sarimax = """
    O SARIMAX é um modelo estatístico avançado para prever valores futuros em uma série de dados que possui tendências e padrões sazonais (que se repetem em períodos fixos, como anualmente).

    - **S** (Seasonal): Lida com a sazonalidade.
    - **AR** (AutoRegressive): Usa a relação entre uma observação e as observações anteriores.
    - **I** (Integrated): Remove tendências dos dados para estabilizar a série.
    - **MA** (Moving Average): Usa a dependência entre uma observação e os erros de previsão passados.
    - **X** (eXogenous): Permite incluir variáveis externas que influenciam o resultado (neste caso, usamos o IPCA de Alimentos como uma variável externa).
    """
    
    st.subheader('Previsão para Cesta Básica (R$) com SARIMAX', help=help_sarimax)

    if dieese.empty:
        st.info('Adicione `dieese_cesta_2022.csv` para usar a previsão SARIMAX.')
    elif not HAVE_SM:
        st.info('Instale `statsmodels` e `scikit-learn` para usar o modelo SARIMAX.')
    else:
        caps = sorted(dieese['capital'].unique().tolist())
        
        # Define o índice padrão para Brasília se existir, senão São Paulo
        default_ix = 0
        if 'Brasília' in caps:
            default_ix = caps.index('Brasília')
        elif 'São Paulo' in caps:
            default_ix = caps.index('São Paulo')
            
        cap_sel = st.selectbox('Capital', caps, index=default_ix, key='prev_cap_cesta')
        h_sarimax = st.slider('Horizonte de Previsão (meses)', 3, 12, 6, key='h_sarimax')

        # Prepara dados para o modelo
        s_cesta = dieese[dieese['capital'] == cap_sel].set_index('data')['valor_cesta'].sort_index()
        s_cesta = ensure_ms_index(s_cesta.dropna())
        
        if s_cesta.empty:
            st.warning(f"Não há dados históricos de cesta básica para {cap_sel} para treinar o modelo.")
        else:
            s_ipca = ensure_ms_index(df['ipca_alimentos_mom'].dropna())
            
            # Cria uma série dummy de salário mínimo alinhada à cesta
            sal_min_series = pd.Series(1320.0, index=s_cesta.index)

            res = None
            metrics = {'MAE': None}
            try:
                model, res, X_used, metrics = fit_sarimax_cesta(s_cesta, s_ipca, sal_min_series)
            except Exception as e:
                st.error(f'Falha ao treinar o SARIMAX: {e}')

            st.metric('MAE (Erro Absoluto Médio na Validação)', _fmt_metric(metrics.get('MAE'), ' R$'))

            if res is not None and not s_cesta.empty:
                last_train_date = s_cesta.index.max()
                fc_cesta = forecast_cesta(res, last_train_date=last_train_date, horizon=h_sarimax)

                hist = s_cesta.rename('histórico').reset_index()
                figp = go.Figure()
                figp.add_trace(go.Scatter(x=hist['data'], y=hist['histórico'], mode='lines', name='Histórico (Cesta)'))
                figp.add_trace(go.Scatter(x=fc_cesta['data'], y=fc_cesta['cesta_fc'], mode='lines+markers', name='Previsão (Cesta)'))
                figp.add_trace(go.Scatter(x=fc_cesta['data'], y=fc_cesta['hi80'], mode='lines', line=dict(width=0), showlegend=False))
                figp.add_trace(go.Scatter(x=fc_cesta['data'], y=fc_cesta['lo80'], mode='lines', fill='tonexty', name='Intervalo de Confiança 80%', line=dict(width=0)))
                figp.update_layout(height=420, xaxis_title='Mês', yaxis_title='Cesta (R$)', title=f'Previsão do Custo da Cesta Básica em {cap_sel}')
                st.plotly_chart(figp, use_container_width=True)
                
                st.caption(f"A previsão assume que o IPCA de alimentos e o salário mínimo permanecerão nos últimos valores conhecidos pelo modelo: R$ {fc_cesta['sal_min_usado'].iloc[0]:.2f}.")
                st.dataframe(fc_cesta[['data', 'cesta_fc', 'lo80', 'hi80']].set_index('data').round(2), use_container_width=True)
                to_download_button(fc_cesta, f'previsao_cesta_{cap_sel}.csv', '⬇️ Baixar Previsão da Cesta (CSV)')
            else:
                st.warning('Não foi possível gerar a previsão (série de dados muito curta ou erro no ajuste do modelo).')


# ========== Relatório ==========
with tab5:
    st.header(f'Relatório ({ano_foco})')
    
    def pct(x): return f'{x:.1f}%' if pd.notna(x) else 'N/A'
    
    if not df_foco.empty:
        media_ie = df_foco['IE_essenciais_mom'].mean()
        media_inpc = df_foco['inpc_mom'].mean()
        meses_crit_mask = df_foco['IE_essenciais_mom'] > df_foco['inpc_mom']
        n_crit = meses_crit_mask.sum()
        ie_acum_pct = ((1 + df_foco['IE_essenciais_mom'] / 100).prod() - 1) * 100
        inpc_acum_pct = ((1 + df_foco['inpc_mom'] / 100).prod() - 1) * 100

        st.markdown('### **IE (Alimentos) vs. INPC**')
        st.markdown(
            f"""
            - **Acumulado no Ano:** A inflação específica de alimentos (IE) fechou em **{pct(ie_acum_pct)}**, enquanto o INPC geral acumulou **{pct(inpc_acum_pct)}**. A diferença evidencia o peso dos alimentos na inflação sentida pela população de baixa renda.
            - **Análise Mensal:** Em **{n_crit} de {len(df_foco)} meses**, a inflação de alimentos superou o INPC, pressionando o orçamento familiar de forma consistente.
            """
        )

        st.markdown('#### Tabela Resumo do Ano')
        tbl_report = df_foco[['IE_essenciais_mom', 'inpc_mom', 'contrib_ipca_alimentos_pp']].round(2).rename(
            columns={'IE_essenciais_mom': 'IE (Alim) % m/m', 'inpc_mom': 'INPC % m/m', 'contrib_ipca_alimentos_pp': 'Contrib. Alimentos (p.p.)'}
        )
        st.dataframe(tbl_report, use_container_width=True)
        to_download_button(tbl_report.reset_index(), f'relatorio_mensal_{ano_foco}.csv', '⬇️ Baixar Tabela Mensal (CSV)')
    else:
        st.warning(f"Não há dados para gerar o relatório de {ano_foco}.")

    st.markdown('---')
    st.markdown('### 🛒 **Impacto no Poder de Compra (Cesta Básica)**')

    if dieese.empty:
        st.info('Adicione `dieese_cesta_2022.csv` para gerar o relatório de poder de compra.')
    else:
        caps_rel = sorted(dieese['capital'].unique())
        cap_sel_rel = st.selectbox('Capital para o Relatório', caps_rel, index=caps_rel.index('São Paulo') if 'São Paulo' in caps_rel else 0, key='rel_cap')
        sal_rel = st.number_input('Salário Mínimo para o Relatório (R$)', value=1320.0, step=10.0, key='rel_sal', format="%.2f")
        
        s_foco_rel = dieese[(dieese['capital'] == cap_sel_rel) & (dieese['data'].dt.year == ano_foco)].copy()
        
        if s_foco_rel.empty:
            st.warning(f'Não há dados de cesta para {cap_sel_rel} em {ano_foco}.')
        else:
            s_foco_rel['cestas_por_salario'] = sal_rel / s_foco_rel['valor_cesta']
            s_foco_rel['cesta_pct_salario'] = 100 * s_foco_rel['valor_cesta'] / sal_rel
            
            ini_rel, fim_rel = s_foco_rel.iloc[0], s_foco_rel.iloc[-1]
            delta_cestas = fim_rel['cestas_por_salario'] - ini_rel['cestas_por_salario']
            delta_pct_salario = fim_rel['cesta_pct_salario'] - ini_rel['cesta_pct_salario']

            # 1. Calcule a variação percentual e guarde em uma variável
            variacao_percentual = ((fim_rel['valor_cesta'] - ini_rel['valor_cesta']) / ini_rel['valor_cesta']) * 100

            # 2. Defina o texto e o valor a ser exibido
            if variacao_percentual == 0:
                texto_final = "(O valor manteve-se estável)"
            else:
                # Escolhe a palavra "Aumento" ou "Queda"
                status = "Aumento" if variacao_percentual > 0 else "Queda"
                
                # Usa o valor absoluto (abs) para que nunca apareça "Queda de -10%"
                valor_abs_percentual = abs(variacao_percentual)
                
                # Monta a frase final formatando com 1 casa decimal
                texto_final = f"({status} de {valor_abs_percentual:.1f}%)"
            
            st.markdown(f"""
            - **Evolução do Custo:** Em **{cap_sel_rel}**, o custo da cesta básica variou de `R$ {ini_rel['valor_cesta']:.2f}` no início do ano para `R$ {fim_rel['valor_cesta']:.2f}` no final. {texto_final}
            - **Perda de Poder de Compra:** Com um salário de **R$ {sal_rel:.2f}**, o número de cestas que uma pessoa poderia comprar **caiu em {abs(delta_cestas):.2f}** ao longo do ano.
            - **Comprometimento da Renda:** Ao final de {ano_foco}, era necessário comprometer **{fim_rel['cesta_pct_salario']:.1f}%** do salário mínimo para adquirir uma única cesta básica, um aumento de **{delta_pct_salario:.1f} pontos percentuais** em relação ao início do ano.
            """)