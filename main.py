# ==============================================================================
# FASE 0: IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
# Aqui, importamos todas as ferramentas necessárias para o projeto.

import pandas as pd  # Principal biblioteca para manipulação e análise de dados (DataFrames).
import numpy as np   # Biblioteca para operações numéricas, especialmente arrays.
import lightgbm as lgb # Biblioteca que contém o modelo de Machine Learning (Light Gradient Boosting Machine).
from sklearn.preprocessing import LabelEncoder # Ferramenta para transformar variáveis categóricas (texto) em números.
from itertools import product # Ferramenta para criar combinações (não utilizada na versão final, mas útil para criar esqueletos de dados).
import gc # Módulo "Garbage Collector", para forçar a limpeza de memória RAM.

print("Bibliotecas importadas com sucesso!")


# ==============================================================================
# FASE 1: CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================
# O primeiro passo é carregar os dados disponibilizados no formato .parquet,
# que é mais eficiente em termos de espaço e velocidade do que o .csv.

try:
    # Leitura dos arquivos parquet para DataFrames do pandas.
    df_transacoes = pd.read_parquet('files-parquet\\transacoes.parquet')
    df_produtos = pd.read_parquet('files-parquet\\produtos.parquet')
    df_pdv = pd.read_parquet('files-parquet\\pdv.parquet')
    print("Arquivos .parquet carregados com sucesso!")
    # Exibimos o formato (linhas, colunas) de cada base para uma verificação inicial.
    print(f"Transações: {df_transacoes.shape}")
    print(f"Produtos: {df_produtos.shape}")
    print(f"PDV: {df_pdv.shape}")
except FileNotFoundError:
    print("Arquivos .parquet não encontrados. Verifique os nomes e caminhos.")

# --- Análise Exploratória Inicial (EDA) ---
# Usamos o método .info() para ter uma visão geral dos dados: tipos de colunas,
# valores não nulos e uso de memória. É crucial para identificar problemas
# como tipos de dados incorretos ou valores ausentes.
print("--- Informações da Base de Transações ---")
df_transacoes.info()
print("\n--- Informações da Base de Produtos ---")
df_produtos.info()
print("\n--- Informações da Base de PDV ---")
df_pdv.info()

print("\n\n--- Amostra das Transações ---")
print(df_transacoes.head())


# --- Unificação e Limpeza dos Dados ---
print("Unificando e limpando os dados...")

# Para unir (merge) as bases, precisamos que as colunas-chave tenham o mesmo nome.
# Renomeamos as colunas de ID para que correspondam à base de transações.
df_produtos.rename(columns={'produto': 'internal_product_id'}, inplace=True)
df_pdv.rename(columns={'pdv': 'internal_store_id'}, inplace=True)

# Unimos as três bases em um único DataFrame.
# Usamos 'how=left' para garantir que todas as transações sejam mantidas,
# e as informações de produtos e PDVs sejam adicionadas a elas.
df_merged = pd.merge(df_transacoes, df_produtos, on='internal_product_id', how='left')
df_final = pd.merge(df_merged, df_pdv, on='internal_store_id', how='left')

# Convertemos a coluna de data para o formato datetime, essencial para qualquer análise de série temporal.
df_final['transaction_date'] = pd.to_datetime(df_final['transaction_date'])
# Preenchemos os valores nulos na coluna 'label' com 'Desconhecido'. Esta é uma
# estratégia simples de imputação para não perder dados.
df_final['label'].fillna('Desconhecido', inplace=True)

# --- Gerenciamento de Memória ---
# Após a unificação, os DataFrames originais não são mais necessários.
# Removemos eles da memória para liberar recursos, o que é uma boa prática
# ao lidar com grandes volumes de dados.
del df_transacoes, df_produtos, df_pdv, df_merged
gc.collect()

print("\nDataFrame unificado criado. Verificando as informações:")
df_final.info()
print("\nAmostra dos dados unificados:")
print(df_final.head())


# ==============================================================================
# FASE 2: AGREGAÇÃO DOS DADOS POR SEMANA
# ==============================================================================
# O desafio pede uma previsão semanal. Nossos dados, no entanto, são transacionais (diários).
# Portanto, precisamos agrupar as vendas por semana para cada combinação de PDV e produto.

print("Agregando transações por semana...")

# Criamos uma nova coluna 'semana_data' que representa o primeiro dia da semana de cada transação.
# Isso padroniza as datas e permite o agrupamento.
df_final['semana_data'] = df_final['transaction_date'].dt.to_period('W').apply(lambda r: r.start_time).dt.date
df_final['semana_data'] = pd.to_datetime(df_final['semana_data'])

# Definimos como cada coluna será agregada.
# A quantidade ('quantity') será somada. Para as outras colunas (categóricas),
# pegamos o primeiro valor ('first'), pois elas são constantes para um mesmo produto/PDV.
agg_dict = {
    'quantity': 'sum',
    'categoria': 'first', 'subcategoria': 'first', 'marca': 'first',
    'fabricante': 'first', 'premise': 'first', 'categoria_pdv': 'first',
}

# Agrupamos os dados por semana, PDV e produto, e aplicamos as agregações definidas.
df_semanal = df_final.groupby(
    ['semana_data', 'internal_store_id', 'internal_product_id']
).agg(agg_dict).reset_index()
df_semanal.rename(columns={'quantity': 'quantidade_total'}, inplace=True)

# Garantimos que os IDs continuem como números inteiros.
df_semanal['internal_store_id'] = df_semanal['internal_store_id'].astype(np.int64)
df_semanal['internal_product_id'] = df_semanal['internal_product_id'].astype(np.int64)

# Liberamos mais memória.
del df_final
gc.collect()

print("Dados agregados por semana:")
print(df_semanal.head())


# ==============================================================================
# FASE 3: ENGENHARIA DE FEATURES (FEATURE ENGINEERING)
# ==============================================================================
# Esta é uma das partes mais importantes. Criamos novas variáveis (features)
# a partir dos dados existentes para ajudar o modelo a encontrar padrões.

print("Criando features de tempo, lag e janela móvel...")
# Ordenar os dados é crucial para calcular lags e janelas móveis corretamente.
df_semanal.sort_values(by=['internal_store_id', 'internal_product_id', 'semana_data'], inplace=True)

# --- Features Temporais ---
# Extraímos informações da data que podem indicar sazonalidade.
df_semanal['mes'] = df_semanal['semana_data'].dt.month
df_semanal['semana_do_ano'] = df_semanal['semana_data'].dt.isocalendar().week.astype(int)
df_semanal['ano'] = df_semanal['semana_data'].dt.year

# --- Features de Lag ---
# O lag é o valor de uma variável em um período de tempo anterior.
# 'lag_1_semanas' será a quantidade vendida na semana anterior. Isso é
# altamente preditivo, pois as vendas de hoje geralmente se parecem com as de ontem.
for lag in range(1, 5):
    df_semanal[f'lag_{lag}_semanas'] = df_semanal.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(lag)

# --- Features de Janela Móvel (Rolling Window) ---
# Calculamos estatísticas (média, desvio padrão) sobre uma janela de tempo passada.
# Isso ajuda a suavizar ruídos e capturar a tendência recente das vendas.
df_semanal['media_movel_4_semanas'] = df_semanal.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(1).rolling(window=4, min_periods=1).mean()
df_semanal['desvio_padrao_movel_4_semanas'] = df_semanal.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(1).rolling(window=4, min_periods=1).std()

# Lags e janelas móveis criam valores nulos no início da série.
# Preenchemos com 0, assumindo que não havia vendas antes do início do histórico.
df_semanal.fillna(0, inplace=True)

print("Features criadas com sucesso. Amostra:")
print(df_semanal[['semana_data', 'quantidade_total', 'lag_1_semanas', 'media_movel_4_semanas']].tail())


# ==============================================================================
# FASE 4: PRÉ-PROCESSAMENTO ADICIONAL
# ==============================================================================

# --- Codificação de Features Categóricas ---
# Modelos de Machine Learning trabalham com números, não com texto.
# O LabelEncoder transforma cada categoria única em um número inteiro (ex: 'Mercearia' -> 0, 'Bar' -> 1).
print("Codificando features categóricas...")
colunas_categoricas = ['categoria', 'subcategoria', 'marca', 'fabricante', 'premise', 'categoria_pdv']
for col in colunas_categoricas:
    le = LabelEncoder()
    df_semanal[col] = le.fit_transform(df_semanal[col].astype(str))

print("Features categóricas codificadas. Amostra:")
print(df_semanal[colunas_categoricas].head())

# --- Otimização de Memória ---
# Esta função ajuda a reduzir o uso de memória do DataFrame, convertendo os tipos
# de dados das colunas para o menor tipo possível que ainda comporte os valores
# (ex: de int64 para int16), sem perda de informação.
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Uso de memória inicial: {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Uso de memória final: {:.2f} MB'.format(end_mem))
    print('Redução de {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

print("Função de otimização de memória definida.")
df_semanal_otimizado = reduce_mem_usage(df_semanal.copy())
del df_semanal
gc.collect()


# ==============================================================================
# FASE 5: TREINAMENTO E VALIDAÇÃO DO MODELO
# ==============================================================================
# Antes de treinar o modelo final, criamos um cenário de validação que simula
# o problema real: usamos dados mais antigos para treinar e dados mais recentes
# para validar. Isso nos dá uma ideia de quão bem o modelo generaliza para o futuro.

print("Dividindo os dados para treino e validação...")
# Descartamos o primeiro mês de dados, pois as features de lag e janela móvel
# ainda não estão bem calculadas (estão preenchidas com zeros).
df_modelo = df_semanal_otimizado[df_semanal_otimizado['semana_data'] >= '2022-02-01'].copy()
del df_semanal_otimizado
gc.collect()

# Definimos uma data de corte. Tudo antes será treino, tudo depois será validação.
data_corte = '2022-12-01'
df_treino = df_modelo[df_modelo['semana_data'] < data_corte]
df_validacao = df_modelo[df_modelo['semana_data'] >= data_corte]

# Separamos as features (variáveis de entrada) do nosso alvo (o que queremos prever).
target = 'quantidade_total'
features = [col for col in df_modelo.columns if col not in [target, 'semana_data']]

X_treino, y_treino = df_treino[features], df_treino[target]
X_validacao, y_validacao = df_validacao[features], df_validacao[target]

print(f"Formato do Treino: {X_treino.shape}")
print(f"Formato da Validação: {X_validacao.shape}")

# --- Configuração do Modelo LightGBM ---
# Hiperparâmetros do modelo. São "configurações" que ajustam como o modelo aprende.
# Estes valores foram escolhidos como um bom ponto de partida.
params = {
    'objective': 'regression_l1', # Objetivo: regressão, usando MAE como função de perda.
    'metric': 'mae', # Métrica de avaliação: Mean Absolute Error (Erro Absoluto Médio).
    'n_estimators': 1500, # Número máximo de "árvores" que o modelo pode criar.
    'learning_rate': 0.02, # Taxa de aprendizado: quão rápido o modelo se ajusta.
    'feature_fraction': 0.8, # Usa 80% das features em cada iteração para evitar overfitting.
    'bagging_fraction': 0.8, # Usa 80% dos dados em cada iteração para evitar overfitting.
    'bagging_freq': 1,
    'lambda_l1': 0.1, # Termos de regularização para evitar overfitting.
    'lambda_l2': 0.1,
    'num_leaves': 31, # Número de folhas por árvore.
    'verbose': -1, 'n_jobs': 4, # Configurações de performance e log.
    'seed': 42, # Semente para garantir que os resultados sejam reprodutíveis.
    'boosting_type': 'gbdt',
}

print("Treinando modelo de validação...")
modelo = lgb.LGBMRegressor(**params)
# Treinamos o modelo usando os dados de treino.
# O 'eval_set' permite que o modelo se avalie nos dados de validação a cada passo.
# O 'early_stopping' para o treinamento se o desempenho na validação não melhorar
# por 100 rodadas, evitando overfitting e economizando tempo.
modelo.fit(X_treino, y_treino,
           eval_set=[(X_validacao, y_validacao)],
           eval_metric='mae',
           callbacks=[lgb.early_stopping(100, verbose=True)])

del X_treino, y_treino, df_treino
gc.collect()

# --- Avaliação da Performance (WMAPE) ---
# WMAPE (Weighted Mean Absolute Percentage Error) é a métrica de avaliação do Hackathon.
print("Avaliando o modelo com WMAPE...")
previsoes_validacao = modelo.predict(X_validacao)
previsoes_validacao[previsoes_validacao < 0] = 0 # Vendas não podem ser negativas.

def wmape(y_true, y_pred):
    # Fórmula: Soma dos erros absolutos / Soma dos valores verdadeiros.
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

wmape_score = wmape(y_validacao, np.round(previsoes_validacao))
print(f"WMAPE na validação: {wmape_score:.4f}")

del X_validacao, y_validacao, df_validacao, previsoes_validacao
gc.collect()


# ==============================================================================
# FASE 6: TREINAMENTO DO MODELO FINAL E GERAÇÃO DA SUBMISSÃO
# ==============================================================================
# Após validar nossa abordagem, retreinamos o modelo usando TODOS os dados
# disponíveis de 2022. Isso permite que o modelo aprenda com o máximo de
# informação possível antes de fazer as previsões para 2023.

print("Retreinando modelo final com todos os dados...")
# Ajustamos n_estimators com base no resultado do early_stopping para otimizar o tempo.
params['n_estimators'] = 800
modelo_final = lgb.LGBMRegressor(**params)
modelo_final.fit(df_modelo[features], df_modelo[target],
                 callbacks=[lgb.log_evaluation(period=100)])
print("Modelo final treinado com sucesso!")

# --- Preparação para a Previsão de Janeiro/2023 ---
print("Criando esqueleto de previsão para Jan/2023...")
# Definimos as datas das 5 semanas de janeiro para as quais queremos prever.
datas_jan_2023 = pd.to_datetime(['2023-01-02', '2023-01-09', '2023-01-16', '2023-01-23', '2023-01-30'])

# Pegamos todas as combinações únicas de PDV e produto que já existiram em 2022.
combinacoes_existentes = df_modelo[['internal_store_id', 'internal_product_id']].drop_duplicates()
print(f"Número de combinações PDV-Produto únicas: {len(combinacoes_existentes)}")

# Criamos um DataFrame "futuro" com todas as combinações para cada uma das 5 semanas de janeiro.
df_futuro = combinacoes_existentes.copy()
df_futuro['key'] = 1
datas_df = pd.DataFrame({'semana_data': datas_jan_2023, 'key': 1})
df_futuro = pd.merge(df_futuro, datas_df, on='key').drop('key', axis=1)

print(f"Tamanho do DataFrame futuro: {len(df_futuro)} linhas")
print(df_futuro.head())

# --- Cálculo das Features para o Futuro ---
# Para prever, o DataFrame futuro precisa ter exatamente as mesmas features
# que usamos para treinar o modelo (lags, médias móveis, etc.).
print("Calculando features para o período de previsão...")
# Concatenamos os dados históricos com o esqueleto futuro.
df_combinado = pd.concat([df_modelo, df_futuro], ignore_index=True)
df_combinado.sort_values(by=['internal_store_id', 'internal_product_id', 'semana_data'], inplace=True)

# Recalculamos as features de lag e janela móvel sobre o DataFrame combinado.
# Desta forma, os lags para a primeira semana de 2023 usarão os dados das últimas semanas de 2022.
for lag in range(1, 5):
    df_combinado[f'lag_{lag}_semanas'] = df_combinado.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(lag)
df_combinado['media_movel_4_semanas'] = df_combinado.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(1).rolling(window=4, min_periods=1).mean()
df_combinado['desvio_padrao_movel_4_semanas'] = df_combinado.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(1).rolling(window=4, min_periods=1).std()

# Criamos as features de tempo para as novas datas.
df_combinado['mes'] = df_combinado['semana_data'].dt.month
df_combinado['semana_do_ano'] = df_combinado['semana_data'].dt.isocalendar().week.astype(int)
df_combinado['ano'] = df_combinado['semana_data'].dt.year

# Adicionamos as informações categóricas (codificadas) às novas linhas de janeiro.
info_categoricas = df_modelo.drop_duplicates(subset=['internal_store_id', 'internal_product_id'])
info_categoricas = info_categoricas[['internal_store_id', 'internal_product_id'] + colunas_categoricas]
df_combinado.drop(columns=colunas_categoricas, inplace=True, errors='ignore')
df_combinado = pd.merge(df_combinado, info_categoricas, on=['internal_store_id', 'internal_product_id'], how='left')
df_combinado.fillna(0, inplace=True)

# Filtramos apenas as linhas que precisamos prever (janeiro de 2023).
df_para_prever = df_combinado[df_combinado['semana_data'].isin(datas_jan_2023)]
X_para_prever = df_para_prever[features]

del df_combinado, df_modelo, df_futuro, info_categoricas
gc.collect()

print("Features para o futuro calculadas. Formato para previsão:", X_para_prever.shape)

# --- Geração das Previsões Finais ---
print("Fazendo previsões finais...")
# Usamos o modelo final treinado para prever as quantidades para janeiro.
previsoes_finais = modelo_final.predict(X_para_prever)
previsoes_finais[previsoes_finais < 0] = 0 # Novamente, garantimos que não haja vendas negativas.
previsoes_finais = np.round(previsoes_finais).astype(int) # Arredondamos para o inteiro mais próximo.

# --- Formatação do Arquivo de Submissão ---
# Criamos o DataFrame final no formato exigido pela competição.
df_submissao = pd.DataFrame({
    'semana_data': df_para_prever['semana_data'],
    'pdv': df_para_prever['internal_store_id'],
    'produto': df_para_prever['internal_product_id'],
    'quantidade': previsoes_finais
})

# Mapeamos as datas de janeiro para os números de semana 1 a 5.
map_semana = {semana: i+1 for i, semana in enumerate(sorted(df_submissao['semana_data'].unique()))}
df_submissao['semana'] = df_submissao['semana_data'].map(map_semana)

# Selecionamos e ordenamos as colunas conforme o modelo de entrega.
df_submissao_final = df_submissao[['semana', 'pdv', 'produto', 'quantidade']]

print("Formatação final concluída. Amostra:")
print(df_submissao_final.head())

# --- Salvamento dos Resultados ---
print("Salvando arquivos de submissão...")
df_submissao_final.to_csv('submissao_hackathon.csv', index=False)
df_submissao_final.to_parquet('submissao_hackathon.parquet', index=False)

print("\n\nPROCESSO CONCLUÍDO COM SUCESSO!")
print("Arquivos 'submissao_hackathon.csv' e 'submissao_hackathon.parquet' gerados.")