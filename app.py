import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
import requests

# Garante acesso aos scripts na pasta src
sys.path.append('src')
try:
    from src.data_ingestion import load_raw_data
    from src.data_processing import process_pipeline
except ImportError:
    # Fallback para execução direta
    from data_ingestion import load_raw_data
    from data_processing import process_pipeline

# ===== Configuração da Página =====
st.set_page_config(
    page_title='Monitor de Inflação e Poder de Compra',
    page_icon="🇧🇷",
    layout='wide'
)

# ===== Mapeamento Capital -> UF (Para o Mapa) =====
CAPITAL_TO_UF = {
    'Rio Branco': 'AC', 'Maceió': 'AL', 'Macapá': 'AP', 'Manaus': 'AM',
    'Salvador': 'BA', 'Fortaleza': 'CE', 'Brasília': 'DF', 'Vitória': 'ES',
    'Goiânia': 'GO', 'São Luís': 'MA', 'Cuiabá': 'MT', 'Campo Grande': 'MS',
    'Belo Horizonte': 'MG', 'Belém': 'PA', 'João Pessoa': 'PB', 'Curitiba': 'PR',
    'Recife': 'PE', 'Teresina': 'PI', 'Rio de Janeiro': 'RJ', 'Natal': 'RN',
    'Porto Alegre': 'RS', 'Porto Velho': 'RO', 'Boa Vista': 'RR', 'Florianópolis': 'SC',
    'São Paulo': 'SP', 'Aracaju': 'SE', 'Palmas': 'TO'
}

# ===== Helper: Histórico de Salário Mínimo =====
def get_salario_minimo(data):
    """Retorna o salário mínimo vigente aproximado para a data informada."""
    ano = data.year
    mes = data.month
    if ano == 2019: return 998.00
    if ano == 2020: return 1045.00 if mes >= 2 else 1039.00
    if ano == 2021: return 1100.00
    if ano == 2022: return 1212.00
    if ano == 2023: return 1320.00 if mes >= 5 else 1302.00
    if ano == 2024: return 1412.00
    if ano >= 2025: return 1518.00 
    return 1412.00

# ===== Carregamento de Dados (Cacheado) =====
@st.cache_data
def get_processed_data():
    raw_ana, raw_dieese = load_raw_data()
    df, dieese = process_pipeline(raw_ana, raw_dieese)
    
    if not dieese.empty:
        dieese['salario_minimo'] = dieese['data'].apply(get_salario_minimo)
        # Cálculo: Cesta / (Salário / 220 horas mensais)
        dieese['horas_trabalho'] = dieese['valor_cesta'] / (dieese['salario_minimo'] / 220)
        dieese['pct_comprometido'] = (dieese['valor_cesta'] / dieese['salario_minimo']) * 100
        # Mapeia UF para o mapa
        dieese['UF'] = dieese['capital'].map(CAPITAL_TO_UF)
        
    return df, dieese

@st.cache_data
def get_geojson_brazil():
    """Baixa o GeoJSON dos estados brasileiros para o mapa."""
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    try:
        r = requests.get(url)
        return r.json()
    except:
        return None

@st.cache_resource
def load_models():
    path = 'models/all_capitals_models.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    return {}

try:
    df, dieese = get_processed_data()
    models = load_models()
    geojson_br = get_geojson_brazil()
except Exception as e:
    st.error(f"Erro no carregamento: {e}")
    df, dieese, models, geojson_br = pd.DataFrame(), pd.DataFrame(), {}, None

# ===== Interface do Dashboard =====
st.title('🛒 Inflação, Poder de Compra e Desigualdade Regional')
st.markdown("""
**Análise Econômica:** Impacto da inflação de alimentos no custo de vida e no tempo de trabalho necessário para subsistência nas capitais brasileiras.
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Visão Geral", 
    "🗺️ Mapa da Desigualdade", 
    "⏳ Poder de Compra", 
    "📈 Análise Avançada", 
    "🤖 Previsões (ML)"
])

# --- ABA 1: Visão Geral ---
with tab1:
    st.header("Dinâmica da Inflação")
    if not df.empty:
        col1, col2 = st.columns([1, 3])
        with col1:
            anos = sorted(df.index.year.unique())
            ano_sel = st.selectbox("Filtrar Ano", anos, index=len(anos)-1)
        with col2:
            d = df[df.index.year == ano_sel]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=d.index, y=d['IE_essenciais_mom'], name='IPCA Alimentos (%)', marker_color='#1f77b4'))
            fig.add_trace(go.Scatter(x=d.index, y=d['inpc_mom'], name='INPC Geral (%)', mode='lines+markers', line=dict(color='red', width=3)))
            # Correção: plotly_chart ainda usa use_container_width=True como padrão em muitas versões
            fig.update_layout(title=f"Alimentos vs Inflação Geral ({ano_sel})", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# --- ABA 2: Mapa (NOVO) ---
with tab2:
    st.header("Geografia do Custo de Vida")
    st.markdown("Comparativo regional do impacto da Cesta Básica.")
    
    if not dieese.empty and geojson_br:
        col_map1, col_map2 = st.columns([1, 3])
        
        with col_map1:
            # Pega o último mês disponível nos dados
            ultima_data = dieese['data'].max()
            mes_mapa = st.date_input("Mês de Referência", value=ultima_data, min_value=dieese['data'].min(), max_value=ultima_data)
            # Converte para inicio do mês para bater com os dados
            mes_mapa = pd.Timestamp(mes_mapa).replace(day=1)
            
            metrica_mapa = st.radio("Visualizar:", ["Valor da Cesta (R$)", "% do Salário Mínimo"])
            col_metrica = 'valor_cesta' if metrica_mapa == "Valor da Cesta (R$)" else 'pct_comprometido'
            
        with col_map2:
            # Filtra dados do mês selecionado
            df_mapa = dieese[dieese['data'] == mes_mapa].copy()
            
            # Garante que temos UFs mapeadas
            df_mapa = df_mapa.dropna(subset=['UF'])
            
            if not df_mapa.empty:
                fig_map = px.choropleth(
                    df_mapa,
                    geojson=geojson_br,
                    locations='UF',
                    featureidkey="properties.sigla",
                    color=col_metrica,
                    color_continuous_scale="Reds",
                    range_color=(df_mapa[col_metrica].min(), df_mapa[col_metrica].max()),
                    hover_name="capital",
                    hover_data={'valor_cesta': ':.2f', 'pct_comprometido': ':.1f', 'UF': False},
                    title=f"{metrica_mapa} por Estado ({mes_mapa.strftime('%m/%Y')})"
                )
                fig_map.update_geos(fitbounds="locations", visible=False)
                fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
                
                # Insight Automático
                pior = df_mapa.loc[df_mapa[col_metrica].idxmax()]
                melhor = df_mapa.loc[df_mapa[col_metrica].idxmin()]
                st.info(f"📍 **Desigualdade:** Em {pior['capital']}, a cesta custa **R$ {pior['valor_cesta']:.2f}** ({pior['pct_comprometido']:.1f}% do salário). Já em {melhor['capital']}, custa **R$ {melhor['valor_cesta']:.2f}**.")
            else:
                st.warning(f"Sem dados para o mês {mes_mapa.strftime('%m/%Y')}.")
    else:
        st.error("Erro ao carregar mapa. Verifique conexão com internet para baixar GeoJSON.")

# --- ABA 3: Poder de Compra ---
with tab3:
    st.header("Esforço Laboral e Impacto Familiar")
    st.markdown("Análise do poder de compra real do Salário Mínimo frente ao custo da alimentação.")

    if not dieese.empty:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Parâmetros")
            capitais = sorted(dieese['capital'].unique())
            idx_padrao = capitais.index('São Paulo') if 'São Paulo' in capitais else 0
            cap_ref = st.selectbox("Selecione a Capital", capitais, index=idx_padrao)
            
            d_cap = dieese[dieese['capital'] == cap_ref].sort_values('data')
            
            if not d_cap.empty:
                atual = d_cap.iloc[-1]
                custo_cesta = atual['valor_cesta']
                salario = atual['salario_minimo']
                horas = atual['horas_trabalho']
                pct_indiv = atual['pct_comprometido']
                
                st.markdown("### 👤 Individual (1 Pessoa)")
                col_a, col_b = st.columns(2)
                col_a.metric("Custo da Cesta", f"R$ {custo_cesta:.2f}")
                col_b.metric("Horas de Trabalho", f"{int(horas)}h {int((horas%1)*60)}min")
                st.metric("% do Salário Mínimo", f"{pct_indiv:.1f}%")
                
                st.divider()
                
                st.markdown("### 👨‍👩‍👧‍👦 Família (2 Adultos + 2 Crianças)")
                st.caption("Estimativa DIEESE: Necessidade de 3 Cestas Básicas.")
                
                custo_familia = custo_cesta * 3
                pct_familia = (custo_familia / salario) * 100
                
                col_c, col_d = st.columns(2)
                col_c.metric("Custo Família", f"R$ {custo_familia:.2f}")
                col_d.metric("Impacto na Renda", f"{pct_familia:.1f}%", 
                             delta="Déficit" if pct_familia > 100 else "Sobra", 
                             delta_color="inverse")
                
                if pct_familia > 100:
                    st.error(f"🚨 **Situação Crítica:** O custo da alimentação básica ({pct_familia:.0f}%) é maior que o salário mínimo inteiro!")

        with c2:
            st.subheader(f"Evolução Histórica em {cap_ref}")
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=d_cap['data'], y=d_cap['horas_trabalho'], 
                fill='tozeroy', 
                name='Horas Necessárias', 
                line=dict(color='#ff7f0e'),
                hovertemplate='%{y:.1f} horas<extra></extra>'
            ))
            fig_h.add_hline(y=110, line_dash="dot", annotation_text="50% da Jornada (110h)", annotation_position="bottom right")
            fig_h.add_hline(y=220, line_dash="solid", line_color="red", annotation_text="Jornada Completa (220h)", annotation_position="top right")
            
            fig_h.update_layout(
                title="Tempo de trabalho necessário para comprar 1 Cesta Básica",
                yaxis_title="Horas de Trabalho (Mensal)",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig_h, use_container_width=True)
            
            st.info("""
            **Nota Metodológica:**
            * **Cálculo de Horas:** Baseado na jornada constitucional de 220 horas mensais.
            * **Custo Família:** Cobre a alimentação de uma família de 4 pessoas (ref: DIEESE).
            """)
    else:
        st.warning("Dados não disponíveis para gerar análise de poder de compra.")

# --- ABA 4: Análise Avançada ---
with tab4:
    st.header("📊 Estatística e Inflação Pessoal")
    
    col_inf1, col_inf2 = st.columns(2)
    
    # 1. Calculadora de Inflação Pessoal
    with col_inf1:
        st.subheader("Simulador: Minha Inflação")
        st.markdown("O IPCA oficial é uma média. Qual é a inflação para quem gasta muito com comida?")
        
        peso_alim = st.slider("Quanto da sua renda vai para Alimentos?", 10, 90, 50, format="%d%%")
        peso_resto = 100 - peso_alim
        
        st.write(f"Sua Cesta: **{peso_alim}% Alimentos** (IPCA Alimentos) + **{peso_resto}% Outros** (INPC Geral)")
        
        if not df.empty:
            # Cria índice sintético
            df['Inflacao_Pessoal'] = (df['IE_essenciais_mom'] * (peso_alim/100)) + (df['inpc_mom'] * (peso_resto/100))
            
            ultimo_ano = df[df.index.year == df.index.year.max()]
            
            if not ultimo_ano.empty:
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(x=ultimo_ano.index, y=ultimo_ano['Inflacao_Pessoal'], name='Minha Inflação'))
                fig_sim.add_trace(go.Scatter(x=ultimo_ano.index, y=ultimo_ano['inpc_mom'], name='INPC Oficial', line=dict(dash='dot')))
                st.plotly_chart(fig_sim, use_container_width=True)

    # 2. Correlação
    with col_inf2:
        st.subheader("Correlação: Cesta vs Inflação")
        cap_corr = st.selectbox("Capital para Correlação", sorted(dieese['capital'].unique()), key='corr_cap')
        
        d_corr = dieese[dieese['capital'] == cap_corr][['data', 'valor_cesta']].set_index('data')
        merged = df.join(d_corr, how='inner')
        
        if not merged.empty:
            corr_val = merged['valor_cesta'].pct_change().corr(merged['inpc_mom'])
            st.metric(f"Correlação (Cesta {cap_corr} vs INPC)", f"{corr_val:.2f}")
            st.caption("Próximo de 1 indica forte relação positiva.")
            
            fig_scat = px.scatter(merged, x='inpc_mom', y=merged['valor_cesta'].pct_change()*100, 
                                 labels={'inpc_mom': 'INPC (%)', 'y': 'Variação Cesta (%)'},
                                 title=f"Dispersão: {cap_corr}")
            st.plotly_chart(fig_scat, use_container_width=True)

# --- ABA 5: Previsões ---
with tab5:
    st.header("Projeções (SARIMAX)")
    if models:
        c1, c2 = st.columns(2)
        cap_target = c1.selectbox("Capital", sorted(models.keys()))
        h = c2.slider("Meses", 1, 12, 6)
        
        m_data = models[cap_target]
        mae = m_data.get('mae', 0)
        
        if st.button("Prever"):
            model = m_data.get('model') or m_data.get('model_obj')
            last_exog = m_data['last_exog']
            try:
                # Tenta desempacotar se for lista/array
                val_ipca = last_exog[0] if len(last_exog) > 0 else 0.5
                val_sal = last_exog[1] if len(last_exog) > 1 else 1412.0
            except:
                val_ipca, val_sal = 0.5, 1412.0
                
            dates = pd.date_range(m_data['last_date'], periods=h+1, freq='MS')[1:]
            X_fut = pd.DataFrame({'ipca': val_ipca, 'sal_min': val_sal}, index=dates)
            
            if 'ipca' not in str(model.model.exog_names):
                X_fut.columns = ['ipca_alimentos_mom', 'sal_min']
            
            fc = model.get_forecast(steps=h, exog=X_fut)
            mean = fc.predicted_mean
            conf = fc.conf_int(alpha=0.2)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=model.data.dates[-12:], y=model.data.endog[-12:], name='Histórico'))
            fig.add_trace(go.Scatter(x=mean.index, y=mean, name='Previsão', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=mean.index, y=conf.iloc[:,0], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=mean.index, y=conf.iloc[:,1], fill='tonexty', name='IC 80%', line=dict(width=0)))
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Previsão ({h} meses): R$ {mean.iloc[-1]:.2f} (MAE: {mae:.2f})")
    else:
        st.warning("Modelos não encontrados.")

# Footer
st.sidebar.info("Projeto P2 - Data Science")
st.sidebar.caption("Mapa carregado via GeoJSON público.")