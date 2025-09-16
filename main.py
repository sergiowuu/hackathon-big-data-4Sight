# ==============================================================================
# FASE 0: IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from itertools import product
import gc

print("Bibliotecas importadas com sucesso!")

# ==============================================================================
# FASE 1: CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================

try:
    df_transacoes = pd.read_parquet('files-parquet\\transacoes.parquet')
    df_produtos = pd.read_parquet('files-parquet\\produtos.parquet')
    df_pdv = pd.read_parquet('files-parquet\\pdv.parquet')
    print("Arquivos .parquet carregados com sucesso!")
    print(f"Transações: {df_transacoes.shape}")
    print(f"Produtos: {df_produtos.shape}")
    print(f"PDV: {df_pdv.shape}")
except FileNotFoundError:
    print("Arquivos .parquet não encontrados. Verifique os nomes e caminhos.")

print("--- Informações da Base de Transações ---")
df_transacoes.info()
print("\n--- Informações da Base de Produtos ---")
df_produtos.info()
print("\n--- Informações da Base de PDV ---")
df_pdv.info()

print("\n\n--- Amostra das Transações ---")
print(df_transacoes.head())

print("Unificando e limpando os dados...")
df_produtos.rename(columns={'produto': 'internal_product_id'}, inplace=True)
df_pdv.rename(columns={'pdv': 'internal_store_id'}, inplace=True)
df_merged = pd.merge(df_transacoes, df_produtos, on='internal_product_id', how='left')
df_final = pd.merge(df_merged, df_pdv, on='internal_store_id', how='left')

df_final['transaction_date'] = pd.to_datetime(df_final['transaction_date'])
df_final['label'].fillna('Desconhecido', inplace=True)

del df_transacoes, df_produtos, df_pdv, df_merged
gc.collect()

print("\nDataFrame unificado criado. Verificando as informações:")
df_final.info()
print("\nAmostra dos dados unificados:")
print(df_final.head())


# ==============================================================================
# FASE 2: AGREGAÇÃO DOS DADOS POR SEMANA
# ==============================================================================

print("Agregando transações por semana...")

df_final['semana_data'] = df_final['transaction_date'].dt.to_period('W').apply(lambda r: r.start_time).dt.date
df_final['semana_data'] = pd.to_datetime(df_final['semana_data'])

agg_dict = {
    'quantity': 'sum',
    'categoria': 'first', 'subcategoria': 'first', 'marca': 'first',
    'fabricante': 'first', 'premise': 'first', 'categoria_pdv': 'first',
}

df_semanal = df_final.groupby(
    ['semana_data', 'internal_store_id', 'internal_product_id']
).agg(agg_dict).reset_index()
df_semanal.rename(columns={'quantity': 'quantidade_total'}, inplace=True)

df_semanal['internal_store_id'] = df_semanal['internal_store_id'].astype(np.int64)
df_semanal['internal_product_id'] = df_semanal['internal_product_id'].astype(np.int64)

del df_final
gc.collect()

print("Dados agregados por semana:")
print(df_semanal.head())


# ==============================================================================
# FASE 3: ENGENHARIA DE FEATURES (FEATURE ENGINEERING)
# ==============================================================================

print("Criando features de tempo, lag e janela móvel...")
df_semanal.sort_values(by=['internal_store_id', 'internal_product_id', 'semana_data'], inplace=True)

# --- Features Temporais ---
df_semanal['mes'] = df_semanal['semana_data'].dt.month
df_semanal['semana_do_ano'] = df_semanal['semana_data'].dt.isocalendar().week.astype(int)
df_semanal['ano'] = df_semanal['semana_data'].dt.year

# --- Features de Lag ---
for lag in range(1, 5):
    df_semanal[f'lag_{lag}_semanas'] = df_semanal.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(lag)

# --- Features de Janela Móvel (Rolling Window) ---
df_semanal['media_movel_4_semanas'] = df_semanal.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(1).rolling(window=4, min_periods=1).mean()
df_semanal['desvio_padrao_movel_4_semanas'] = df_semanal.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(1).rolling(window=4, min_periods=1).std()

df_semanal.fillna(0, inplace=True)

print("Features criadas com sucesso. Amostra:")
print(df_semanal[['semana_data', 'quantidade_total', 'lag_1_semanas', 'media_movel_4_semanas']].tail())


# ==============================================================================
# FASE 4: PRÉ-PROCESSAMENTO ADICIONAL
# ==============================================================================

print("Codificando features categóricas...")
colunas_categoricas = ['categoria', 'subcategoria', 'marca', 'fabricante', 'premise', 'categoria_pdv']
for col in colunas_categoricas:
    le = LabelEncoder()
    df_semanal[col] = le.fit_transform(df_semanal[col].astype(str))

print("Features categóricas codificadas. Amostra:")
print(df_semanal[colunas_categoricas].head())

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

print("Dividindo os dados para treino e validação...")
df_modelo = df_semanal_otimizado[df_semanal_otimizado['semana_data'] >= '2022-02-01'].copy()
del df_semanal_otimizado
gc.collect()

data_corte = '2022-12-01'
df_treino = df_modelo[df_modelo['semana_data'] < data_corte]
df_validacao = df_modelo[df_modelo['semana_data'] >= data_corte]

target = 'quantidade_total'
features = [col for col in df_modelo.columns if col not in [target, 'semana_data']]

X_treino, y_treino = df_treino[features], df_treino[target]
X_validacao, y_validacao = df_validacao[features], df_validacao[target]

print(f"Formato do Treino: {X_treino.shape}")
print(f"Formato da Validação: {X_validacao.shape}")

# --- Configuração do Modelo LightGBM ---
params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': 1500,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1, 'n_jobs': 4,
    'seed': 42,
    'boosting_type': 'gbdt',
}

print("Treinando modelo de validação...")
modelo = lgb.LGBMRegressor(**params)

modelo.fit(X_treino, y_treino,
           eval_set=[(X_validacao, y_validacao)],
           eval_metric='mae',
           callbacks=[lgb.early_stopping(100, verbose=True)])

del X_treino, y_treino, df_treino
gc.collect()

print("Avaliando o modelo com WMAPE...")
previsoes_validacao = modelo.predict(X_validacao)
previsoes_validacao[previsoes_validacao < 0] = 0

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

wmape_score = wmape(y_validacao, np.round(previsoes_validacao))
print(f"WMAPE na validação: {wmape_score:.4f}")

del X_validacao, y_validacao, df_validacao, previsoes_validacao
gc.collect()


# ==============================================================================
# FASE 6: TREINAMENTO DO MODELO FINAL E GERAÇÃO DA SUBMISSÃO
# ==============================================================================

print("Retreinando modelo final com todos os dados...")
params['n_estimators'] = 800
modelo_final = lgb.LGBMRegressor(**params)
modelo_final.fit(df_modelo[features], df_modelo[target],
                 callbacks=[lgb.log_evaluation(period=100)])
print("Modelo final treinado com sucesso!")

print("Criando esqueleto de previsão para Jan/2023...")
datas_jan_2023 = pd.to_datetime(['2023-01-02', '2023-01-09', '2023-01-16', '2023-01-23', '2023-01-30'])

combinacoes_existentes = df_modelo[['internal_store_id', 'internal_product_id']].drop_duplicates()
print(f"Número de combinações PDV-Produto únicas: {len(combinacoes_existentes)}")

df_futuro = combinacoes_existentes.copy()
df_futuro['key'] = 1
datas_df = pd.DataFrame({'semana_data': datas_jan_2023, 'key': 1})
df_futuro = pd.merge(df_futuro, datas_df, on='key').drop('key', axis=1)

print(f"Tamanho do DataFrame futuro: {len(df_futuro)} linhas")
print(df_futuro.head())

print("Calculando features para o período de previsão...")
df_combinado = pd.concat([df_modelo, df_futuro], ignore_index=True)
df_combinado.sort_values(by=['internal_store_id', 'internal_product_id', 'semana_data'], inplace=True)

for lag in range(1, 5):
    df_combinado[f'lag_{lag}_semanas'] = df_combinado.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(lag)
df_combinado['media_movel_4_semanas'] = df_combinado.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(1).rolling(window=4, min_periods=1).mean()
df_combinado['desvio_padrao_movel_4_semanas'] = df_combinado.groupby(['internal_store_id', 'internal_product_id'])['quantidade_total'].shift(1).rolling(window=4, min_periods=1).std()

df_combinado['mes'] = df_combinado['semana_data'].dt.month
df_combinado['semana_do_ano'] = df_combinado['semana_data'].dt.isocalendar().week.astype(int)
df_combinado['ano'] = df_combinado['semana_data'].dt.year

info_categoricas = df_modelo.drop_duplicates(subset=['internal_store_id', 'internal_product_id'])
info_categoricas = info_categoricas[['internal_store_id', 'internal_product_id'] + colunas_categoricas]
df_combinado.drop(columns=colunas_categoricas, inplace=True, errors='ignore')
df_combinado = pd.merge(df_combinado, info_categoricas, on=['internal_store_id', 'internal_product_id'], how='left')
df_combinado.fillna(0, inplace=True)

df_para_prever = df_combinado[df_combinado['semana_data'].isin(datas_jan_2023)]
X_para_prever = df_para_prever[features]

del df_combinado, df_modelo, df_futuro, info_categoricas
gc.collect()

print("Features para o futuro calculadas. Formato para previsão:", X_para_prever.shape)

print("Fazendo previsões finais...")
previsoes_finais = modelo_final.predict(X_para_prever)
previsoes_finais[previsoes_finais < 0] = 0
previsoes_finais = np.round(previsoes_finais).astype(int)

df_submissao = pd.DataFrame({
    'semana_data': df_para_prever['semana_data'],
    'pdv': df_para_prever['internal_store_id'],
    'produto': df_para_prever['internal_product_id'],
    'quantidade': previsoes_finais
})

map_semana = {semana: i+1 for i, semana in enumerate(sorted(df_submissao['semana_data'].unique()))}
df_submissao['semana'] = df_submissao['semana_data'].map(map_semana)

df_submissao_final = df_submissao[['semana', 'pdv', 'produto', 'quantidade']]

print("Formatação final concluída. Amostra:")
print(df_submissao_final.head())

print("Salvando arquivos de submissão...")
df_submissao_final.to_csv('submissao_hackathon.csv', index=False)
df_submissao_final.to_parquet('submissao_hackathon.parquet', index=False)

print("\n\nPROCESSO CONCLUÍDO COM SUCESSO!")
print("Arquivos 'submissao_hackathon.csv' e 'submissao_hackathon.parquet' gerados.")