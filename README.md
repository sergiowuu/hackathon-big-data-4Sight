# Hackathon de Previsão de Vendas - Reposição Inteligente de Estoque

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-yellow.svg) ![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green.svg)

## 📖 Visão Geral

Este repositório contém a solução desenvolvida para um Hackathon de Ciência de Dados e IA. O objetivo central é criar um modelo de previsão de vendas (*forecast*) para otimizar a reposição de estoque no varejo. O código analisa o histórico de vendas de 2022 para prever a demanda semanal por Ponto de Venda (PDV) e produto (SKU) para as cinco semanas de janeiro de 2023.

O modelo utiliza a técnica de **Gradient Boosting** com a biblioteca **LightGBM**, conhecida por sua alta performance e eficiência computacional em dados tabulares.

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

1.  **`transacoes.parquet`**: Contém o histórico detalhado de cada transação realizada em 2022, incluindo IDs do PDV e do produto, data, quantidade e valores.
2.  **`produtos.parquet`**: Tabela de cadastro com informações sobre cada produto (SKU), como categoria, marca e fabricante.
3.  **`pdv.parquet`**: Tabela de cadastro com informações sobre cada ponto de venda, como localização (zipcode) e tipo de estabelecimento.

---

## ⚙️ Metodologia Aplicada

A solução foi construída seguindo um pipeline estruturado de ponta a ponta, desde a leitura dos dados até a geração do arquivo final de submissão.

#### 1. Preparação e Unificação dos Dados
* Os três arquivos `.parquet` são carregados em DataFrames do Pandas.
* As bases são unificadas em um único DataFrame através de `merge`, utilizando os IDs de produto e PDV como chaves.
* É realizada uma limpeza inicial, como a conversão da coluna de data para o formato `datetime` e o tratamento de valores nulos.

#### 2. Agregação Semanal
* Como o objetivo é prever vendas semanais, os dados transacionais (diários) são agregados.
* As transações são agrupadas por semana, PDV e produto. A quantidade de vendas é somada para cada grupo, transformando a base em uma série temporal com frequência semanal.

#### 3. Engenharia de Features (Feature Engineering)
* Para enriquecer o modelo e permitir que ele aprenda padrões complexos, foram criadas novas features:
    * **Features Temporais:** Mês e semana do ano, extraídos da data para capturar sazonalidades.
    * **Lag Features:** Vendas da mesma combinação PDV-produto em semanas anteriores (ex: lag de 1, 2, 3 e 4 semanas). Isso informa ao modelo a tendência mais recente.
    * **Features de Janela Móvel (Rolling Window):** Média e desvio padrão das vendas nas últimas 4 semanas. Ajudam a suavizar ruídos e capturar a tendência e a volatilidade recentes.

#### 4. Pré-processamento e Modelo
* **Codificação de Variáveis Categóricas:** As features de texto (como categoria do produto e do PDV) são convertidas para números usando `LabelEncoder`, pois modelos de Machine Learning exigem entradas numéricas.
* **Otimização de Memória:** Uma função é aplicada para reduzir o uso de RAM, convertendo os tipos de dados para o menor formato possível sem perda de informação.
* **Modelo de Machine Learning:** Foi escolhido o **LightGBM (Light Gradient Boosting Machine)**, um algoritmo de boosting que constrói árvores de decisão de forma sequencial, corrigindo os erros da árvore anterior.

#### 5. Treinamento e Validação
* Para validar a robustez do modelo, foi adotada uma estratégia de validação temporal. Os dados foram divididos em:
    * **Treino:** Dados de Fevereiro a Novembro de 2022.
    * **Validação:** Dados de Dezembro de 2022.
* O modelo foi treinado no conjunto de treino e avaliado no conjunto de validação usando a métrica WMAPE. A técnica de `Early Stopping` foi utilizada para evitar overfitting e encontrar o número ideal de árvores.

#### 6. Geração das Previsões
* Após a validação da abordagem, o modelo final é retreinado com **todos os dados de 2022**.
* Um "esqueleto" de dados para as 5 semanas de janeiro de 2023 é criado, contendo todas as combinações de PDV-produto existentes.
* As mesmas features (lags, médias móveis) são calculadas para este período futuro.
* O modelo treinado é usado para prever as vendas. Os resultados são tratados (valores negativos zerados e arredondados para inteiros) e formatados no arquivo de submissão.

---

## 🛠️ Tecnologias e Bibliotecas

Para executar este projeto, é necessário ter o Python 3.8 (ou superior) instalado, juntamente com as seguintes bibliotecas.

* **`pandas`**: Para manipulação e análise de dados.
* **`numpy`**: Para operações numéricas eficientes.
* **`scikit-learn`**: Para pré-processamento de dados (especificamente o `LabelEncoder`).
* **`lightgbm`**: Para o treinamento do modelo de Gradient Boosting.
* **`pyarrow`**: Dependência necessária para ler arquivos no formato `.parquet` com o pandas.

Você pode instalar todas as dependências de uma vez usando o arquivo `requirements.txt`.

```bash
pip install -r requirements.txt