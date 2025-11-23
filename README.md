# Documentação do Banco de Dados - Projeto Inflação e Poder de Compra

## 1. Visão Geral
* **Nome do Dataset:** Inflação de Alimentos e Cesta Básica (Consolidado)
* **Fontes:** * IBGE (SIDRA - Tabela 7060 e INPC)
    * DIEESE (Pesquisa Nacional da Cesta Básica de Alimentos)
* **Contexto do Negócio:** O conjunto de dados visa permitir a análise do impacto da inflação específica de alimentos (IPCA-Alimentos/Bebidas) no custo de vida, comparando-o com o INPC (inflação para baixa renda) e com o custo nominal da Cesta Básica em diferentes capitais brasileiras. O objetivo é mensurar a perda de poder de compra.

## 2. Modelo Conceitual
O banco de dados é composto por séries temporais mensais unificadas pela data de referência (`data`).
* **Tabela Principal (dataset_analitico.csv):** Contém os índices nacionais de inflação (IPCA Geral, IPCA Alimentos, INPC).
* **Tabela Auxiliar (dieese_cesta_2022.csv):** Contém os valores nominais (R$) da cesta básica desagregados por Capital/UF.

## 3. Dicionário de Dados

| Coluna | Tipo de Dado | Descrição | Exemplo/Valores Válidos |
| :--- | :--- | :--- | :--- |
| `data` | datetime64[ns] | Data de referência do índice (Primeiro dia do mês). Normalizada para YYYY-MM-01. | `2023-01-01`, `2023-02-01` |
| `IE_essenciais_mom` | float64 | Variação mensal (%) do subgrupo Alimentos e Bebidas (IPCA). | `0.5`, `-0.1` |
| `inpc_mom` | float64 | Variação mensal (%) do Índice Nacional de Preços ao Consumidor (INPC). | `0.4` |
| `capital` | object (string) | Nome da Capital (Unidade Geográfica do DIEESE). | `São Paulo`, `Brasília`, `Rio Branco`* |
| `valor_cesta` | float64 | Custo nominal da Cesta Básica calculado pelo DIEESE. | `750.00`, `800.50` |
| `sal_min` | float64 | Valor do Salário Mínimo vigente no mês de referência. | `1212.00`, `1320.00` |

*\*Nota: Novas capitais inseridas na expansão DIEESE/2025 podem apresentar histórico reduzido.*

## 4. Pré-Processamento Realizado
1.  **Normalização de Datas:** Conversão de múltiplos formatos (`MM/YY`, `YYYY-MM`, `YYYYMM`) para o padrão ISO `YYYY-MM-01`.
2.  **Tratamento de Anos Ambíguos:** Datas com ano em dois dígitos (ex: `99`) foram convertidas utilizando um cutoff móvel (anos > 25 são 19xx, anos <= 25 são 20xx).
3.  **Filtragem de Ruído:** Remoção de registros com datas anteriores a 1990 (inconsistentes com o escopo do plano Real/pós-hiperinflação moderna).
4.  **Deduplicação:** Remoção de colunas de data duplicadas geradas na fusão de datasets.