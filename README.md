# 🛒 Monitor de Inflação e Poder de Compra (MLOps)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://inflacao-em-ie.streamlit.app/)  
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![Status](https://img.shields.io/badge/Status-Concluído-success)

Este projeto é uma solução completa de **Data Science e MLOps** desenvolvida para analisar o impacto da inflação de alimentos no custo de vida e no poder de compra das famílias de baixa renda nas capitais brasileiras.

🔗 **Acesse o Dashboard Online:**  
👉 https://inflacao-em-ie.streamlit.app/

---

## 🎯 Problema de Pesquisa e Objetivo

### **Questão de Pesquisa**
> *"Qual o impacto da inflação do grupo de Alimentos e Bebidas (IPCA) no custo nominal da Cesta Básica e como isso corroeu o poder de compra (horas de trabalho) das famílias de baixa renda?"*

### **Objetivos**
1. **Monitorar:** Comparar inflação oficial (IPCA/INPC) vs custo real da cesta (DIEESE).  
2. **Mensurar:** Estimar horas de trabalho necessárias para comprar a cesta básica.  
3. **Regionalizar:** Mapear desigualdades entre capitais brasileiras.  
4. **Prever:** Projetar custo futuro da cesta usando modelos SARIMAX.

---

## 🛠️ Arquitetura do Projeto (Pipeline MLOps)

O projeto segue boas práticas de MLOps, garantindo reprodutibilidade, modularidade e separação entre backend e frontend.

### 📁 Estrutura de Diretórios

```plaintext
📂 inflacao-em-ie/
│
├── 📂 data/                     # Armazenamento de dados
│   ├── 📂 raw/                  # Dados brutos (IBGE/DIEESE)
│   └── 📂 processed/            # Dados tratados pelo pipeline
│
├── 📂 models/                   # Modelos de Machine Learning
│   └── all_capitals_models.pkl  # Modelos SARIMAX serializados
│
├── 📂 src/                      # Backend (pipeline)
│   ├── data_ingestion.py        # Leitura robusta (.csv/.xls)
│   ├── data_processing.py       # ETL, limpeza, normalização
│   └── modeling.py              # Treinamento + serialização (joblib)
│
├── app.py                       # Dashboard Streamlit (frontend)
├── database_doc.md              # Documentação técnica
└── requirements.txt             # Dependências
```

---

## 📊 Funcionalidades do Dashboard

### **📈 Visão Geral da Inflação**
- Comparação IPCA (Alimentos) × INPC (Geral)  
- Identificação de períodos de pressão inflacionária

### **🗺️ Mapa da Desigualdade (Georreferenciado)**
- Mapa coroplético interativo por capital  
- Exibição do custo da cesta e comprometimento da renda

### **⏱️ Poder de Compra & Horas de Trabalho**
- Cálculo do número de horas necessárias p/ comprar a cesta  
- Indicador familiar (4 pessoas) com alerta quando alimentação > renda

### **🤖 Previsões com IA (SARIMAX)**
- Previsão entre 3 e 12 meses  
- Intervalos de confiança (80%)  
- Inferência em tempo real com modelos pré-treinados

### **📚 Análises Avançadas**
- Simulador de *inflação pessoal*  
- Correlação entre inflação local e nacional

---

## 🚀 Como Rodar Localmente

### **1. Clonar o repositório**
```bash
git clone https://github.com/mayconabe/inflacao-em-ie.git
cd inflacao-em-ie
```

### **2. Instalar dependências**
```bash
pip install -r requirements.txt
```

### **3. Executar pipeline de modelagem**
Processa dados brutos, treina modelos e salva o `.pkl`.

```bash
python src/modeling.py
```

### **4. Iniciar o Dashboard**
```bash
streamlit run app.py
```

---

## 🗂️ Fontes de Dados

| Fonte | Descrição |
|-------|-----------|
| **IBGE (SIDRA)** | IPCA (Alimentos) e INPC Geral |
| **DIEESE** | Cesta Básica de Alimentos – série histórica |
| **GeoJSON** | Malha territorial (CodeForAmerica) |

---

## 📝 Autoria

Desenvolvido como parte da avaliação final de **Data Science**.  
Envolve técnicas de **Engenharia de Dados**, **Séries Temporais**, **Visualização** e **MLOps**, aplicadas para investigar um problema econômico real.
