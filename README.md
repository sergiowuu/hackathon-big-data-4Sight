# Hackathon de Previsão de Vendas - Reposição de Estoque

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![Pandas](https://img.shields.io/badge/Pandas-%26%20Polars-yellow.svg) ![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green.svg) ![Optuna](https://img.shields.io/badge/Optuna-3.0%2B-purple.svg)

## 📖 Visão Geral

Este repositório contém a solução desenvolvida para o Hackathon Forecast da Big Data. O objetivo central é criar um modelo de previsão de vendas (*forecast*) para otimizar a reposição de estoque no varejo. O código analisa o histórico de vendas de 2022 para prever a demanda semanal por Ponto de Venda (PDV) e produto (SKU) para as cinco semanas de janeiro de 2023.

O modelo utiliza a técnica de **Gradient Boosting** com a biblioteca **LightGBM**, otimizada com **Optuna** para encontrar os melhores hiperparâmetros, garantindo alta performance e eficiência computacional.

---

## 🎯 O Desafio

O problema proposto consiste em desenvolver uma solução de IA para prever a quantidade de vendas semanais, com base nos seguintes critérios:

* **Objetivo:** Prever a quantidade de vendas por `PDV` e `SKU` para cada uma das 5 semanas de janeiro de 2023.
* **Dados de Treino:** Histórico transacional de vendas de 2022, enriquecido com dados cadastrais de produtos e PDVs.
* **Métrica de Avaliação:** A performance do modelo é medida pelo **WMAPE** (Weighted Mean Absolute Percentage Error), que pondera os erros pela magnitude das vendas.
* **Critérios Adicionais:** Qualidade técnica do código (clareza, organização, documentação), criatividade na modelagem e superação de um modelo de *baseline* interno da empresa organizadora.

---

## 📊 Dados

Foram disponibilizados três conjuntos de dados no formato `.parquet`:

1.  **`transacao.parquet`**: Contém o histórico detalhado de cada transação realizada em 2022, incluindo IDs do PDV e do produto, data, quantidade e valores.
2.  **`produto.parquet`**: Tabela de cadastro com informações sobre cada produto (SKU), como categoria, marca e fabricante.
3.  **`pdv.parquet`**: Tabela de cadastro com informações sobre cada ponto de venda, como localização e tipo de estabelecimento.

---

## ⚙️ Metodologia Aplicada

A solução foi construída seguindo um pipeline estruturado de ponta a ponta, desde a leitura dos dados até a geração do arquivo final de submissão.

#### 1. Limpeza e Pré-processamento de Dados
* Carregamento otimizado dos arquivos `.parquet` utilizando **Polars** e **Pandas**.
* Aplicação de regras de negócio para remover dados inválidos (ex: quantidades e valores negativos ou nulos).
* Tratamento de valores nulos nas colunas categóricas, preenchendo-os com a categoria `"Desconhecido"` para não perder informação.
* Unificação das três bases em um único DataFrame mestre, pronto para a próxima fase.

#### 2. Engenharia de Features (Feature Engineering)
* **Agregação Semanal:** Os dados transacionais são agregados por semana, PDV e produto para alinhar a granularidade dos dados com o objetivo da previsão.
* **Features Temporais:** Mês e semana do ano são extraídos da data para capturar sazonalidades.
* **Lag Features:** Vendas da mesma combinação PDV-SKU em semanas anteriores (lags de 1 a 4 semanas) são criadas para informar ao modelo a tendência recente.
* **Features de Janela Móvel (Rolling Window):** Média e desvio padrão das vendas nas últimas 4 semanas são calculados para suavizar ruídos e capturar a tendência e volatilidade.

#### 3. Otimização de Hiperparâmetros com Optuna
* Antes do treinamento final, uma busca de hiperparâmetros é realizada com **Optuna** para encontrar a melhor configuração para o LightGBM.
* A busca é feita de forma eficiente em uma **amostra representativa** dos dados (10%) para acelerar o processo.
* Técnicas avançadas como **`GOSS` boosting** e **Pruning** (poda de `trials` ruins) são utilizadas para garantir uma otimização rápida e inteligente.

#### 4. Treinamento e Validação do Modelo
* **Modelo:** É utilizado o **LightGBM (Light Gradient Boosting Machine)**, com os parâmetros otimizados pelo Optuna.
* **Divisão dos Dados:** Foi adotada uma estratégia de divisão estratificada por mês. **80% de cada mês** de 2022 é usado para treino e **20% de cada mês** para teste. Isso garante que a sazonalidade esteja presente em ambos os conjuntos, tornando a validação mais robusta.
* **Tratamento de Categóricas:** As variáveis categóricas são convertidas para o tipo `category` do Pandas, que é tratado de forma nativa e altamente eficiente pelo LightGBM.

#### 5. Geração das Previsões
* O modelo final é treinado com **todos os dados de 2022** utilizando os melhores parâmetros encontrados.
* Um "esqueleto" de dados para as 5 semanas de janeiro de 2023 é criado para os pares PDV-SKU ativos.
* As previsões são geradas em um **loop autoregressivo**, onde a previsão da Semana 1 é usada para calcular as features da Semana 2, e assim por diante.
* Os resultados são tratados (valores negativos zerados, arredondados para inteiros) e formatados no arquivo de submissão, respeitando o limite de 1.5M de linhas.

---

## 🛠️ Tecnologias e Bibliotecas

Para executar este projeto, é necessário ter o Python 3.9 (ou superior) instalado, juntamente com as seguintes bibliotecas.

* **`pandas`** e **`polars`**: Para manipulação e análise de dados.
* **`numpy`**: Para operações numéricas.
* **`scikit-learn`**: Para a divisão estratégica dos dados.
* **`lightgbm`**: Para o treinamento do modelo de Gradient Boosting.
* **`optuna`**: Para a otimização de hiperparâmetros.
* **`pyarrow`**: Dependência para ler arquivos `.parquet`.

Você pode instalar todas as dependências de uma vez usando o arquivo `requirements.txt`.
 
```bash
pip install -r requirements.txt
