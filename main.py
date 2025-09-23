# Instalação de bibliotecas necessárias
# pip install pandas polars pyarrow numpy scikit-learn lightgbm "optuna-integration[lightgbm]"

# ==============================================================================
# FASE 1: CONFIGURAÇÃO, CARGA E LIMPEZA DOS DADOS
# ==============================================================================

# 1.1: Importações e Configurações Globais
import pandas as pd
import polars as pl
import numpy as np
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import train_test_split
from itertools import product
from pathlib import Path
import time
import gc
import warnings
 
# --- Configurações Globais ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
SEED = 42
BASE_PATH = Path('./parquet/')

print("✓ Fase 1.1: Bibliotecas importadas e configurações definidas.")

# 1.2: Carregamento Otimizado dos Dados
print("\n--- Carregando Bases de Dados ---")
df_transacoes = pl.read_parquet(BASE_PATH / 'transacao.parquet').to_pandas()
df_produtos = pl.read_parquet(BASE_PATH / 'produto.parquet').to_pandas().rename(columns={'produto': 'internal_product_id'})
df_pdv = pl.read_parquet(BASE_PATH / 'pdv.parquet').to_pandas().rename(columns={'pdv': 'internal_store_id'})
print("✓ Bases carregadas.")

# 1.3: Otimização de Memória e Limpeza Simples
def reduce_mem_usage(df, name):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                else: df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
                else: df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'   - Uso de memória do {name}: {start_mem:.2f} MB -> {end_mem:.2f} MB')
    return df

df_transacoes = reduce_mem_usage(df_transacoes, 'Transações')
df_produtos = reduce_mem_usage(df_produtos, 'Produtos')
df_pdv = reduce_mem_usage(df_pdv, 'PDV')

df_transacoes = df_transacoes.drop(columns=['reference_date', 'taxes'])
df_produtos = df_produtos.drop(columns=['descricao'])
df_pdv = df_pdv.drop(columns=['zipcode'])
df_transacoes['transaction_date'] = pd.to_datetime(df_transacoes['transaction_date'])
print("✓ Colunas desnecessárias removidas e otimização de memória concluída.")

# 1.4: Filtragem Estratégica de Produtos
print("\n--- Filtrando produtos estrategicamente ---")
df_produtos['label'].fillna('Unknown', inplace=True)
labels_para_remover = ['Discontinued', 'Close Out', 'Clearing', 'Allocated']
produtos_validos_df = df_produtos.query("label not in @labels_para_remover")
produtos_validos_ids = set(produtos_validos_df['internal_product_id'])
df_transacoes = df_transacoes[df_transacoes['internal_product_id'].isin(produtos_validos_ids)]
produtos_finais_ids = set(df_transacoes['internal_product_id'].unique())
df_produtos_final = df_produtos[df_produtos['internal_product_id'].isin(produtos_finais_ids)]
print(f"✓ Produtos filtrados. Restaram {len(df_produtos_final)} produtos únicos com transações.")

# 1.5: Filtragem de Transações por Regras de Negócio
print("\n--- Limpando transações com base nas regras de negócio ---")
query_regras = "quantity > 0 and net_value > 0 and gross_value > 0 and discount >= 0"
df_transacoes = df_transacoes.query(query_regras).copy()
float_cols = df_transacoes.select_dtypes(include=['float']).columns
df_transacoes[float_cols] = df_transacoes[float_cols].round(2)
print("✓ Transações inválidas removidas e valores arredondados.")

# 1.6: Consolidação e Imputação de Nulos
def consolidate_dataframes(df_trans, df_prod, df_pdv):
    master = df_trans.merge(df_prod, on='internal_product_id', how='left').merge(df_pdv, on='internal_store_id', how='left')
    if master.isnull().sum().sum() > 0:
        cols_to_fill = ['label', 'subcategoria', 'marca', 'fabricante', 'premise', 'categoria_pdv', 'categoria', 'tipos']
        for col in cols_to_fill:
            if col in master.columns and master[col].isnull().any():
                master[col].fillna('Desconhecido', inplace=True)
    return master

master_df = consolidate_dataframes(df_transacoes, df_produtos_final, df_pdv)
del df_transacoes, df_produtos_final, df_pdv
gc.collect()
print("✓ Dataframes consolidados e memória liberada.")


# ==============================================================================
# FASE 2: ENGENHARIA DE ATRIBUTOS
# ==============================================================================
print("--- Iniciando Fase 2: Engenharia de Atributos ---")
pl_master = pl.from_pandas(master_df)
del master_df; gc.collect()
pl_master = pl_master.sort("internal_store_id", "internal_product_id", "transaction_date")
weekly_sales_df = pl_master.group_by_dynamic(
    index_column="transaction_date", every="1w", by=["internal_store_id", "internal_product_id"]
).agg([
    pl.sum("quantity").alias("quantity_sum"),
    pl.sum("net_value").alias("net_value_sum"),
    pl.sum("discount").alias("discount_sum"),
    pl.first('transaction_date').alias('first_day_of_week') 
])
weekly_sales_df = weekly_sales_df.with_columns([
    pl.col("first_day_of_week").dt.month().alias("mes"),
    pl.col("first_day_of_week").dt.year().alias("ano"),
    pl.col("first_day_of_week").dt.week().alias("semana_do_ano"),
    pl.col("first_day_of_week").dt.day().alias("dia_do_mes"),
    pl.col("first_day_of_week").dt.ordinal_day().alias("dia_do_ano"),
    pl.col("first_day_of_week").dt.quarter().alias("trimestre"),
])
df_model = weekly_sales_df.to_pandas()
del weekly_sales_df, pl_master; gc.collect()
print("✓ Agregação semanal e features de data criadas.")

df_model = df_model.sort_values(by=['internal_store_id', 'internal_product_id', 'transaction_date'])
lags = [1, 2, 3, 4]; window_size = 4
for lag in lags: df_model[f'quantity_lag_{lag}'] = df_model.groupby(['internal_store_id', 'internal_product_id'])['quantity_sum'].shift(lag)
df_model[f'quantity_rolling_mean_{window_size}'] = df_model.groupby(['internal_store_id', 'internal_product_id'])['quantity_sum'].shift(1).rolling(window=window_size).mean()
df_model[f'quantity_rolling_std_{window_size}'] = df_model.groupby(['internal_store_id', 'internal_product_id'])['quantity_sum'].shift(1).rolling(window=window_size).std()
df_model.fillna(0, inplace=True)
print("✓ Features de Lag e Janela Móvel criadas.")

# ==============================================================================
# FASE 3: PREPARAÇÃO PARA MODELAGEM
# ==============================================================================
print("--- Iniciando Fase 3: Preparação para Modelagem ---")
def wmape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-9)

df_produtos_features = pd.read_parquet(BASE_PATH / 'produto.parquet').rename(columns={'produto': 'internal_product_id'})
df_pdv_features = pd.read_parquet(BASE_PATH / 'pdv.parquet').rename(columns={'pdv': 'internal_store_id'})
product_features = ['internal_product_id', 'categoria', 'label', 'subcategoria', 'marca', 'fabricante']
pdv_features = ['internal_store_id', 'premise', 'categoria_pdv']
df_model = df_model.merge(df_produtos_features[product_features], on='internal_product_id', how='left')
df_model = df_model.merge(df_pdv_features[pdv_features], on='internal_store_id', how='left')
categorical_features = ['internal_store_id', 'internal_product_id', 'mes', 'trimestre', 'dia_do_mes', 'categoria', 'label', 'subcategoria', 'marca', 'fabricante', 'premise', 'categoria_pdv']
for col in categorical_features:
    if col in df_model.columns: df_model[col] = df_model[col].fillna('Desconhecido').astype('category')
numeric_cols = df_model.select_dtypes(include=np.number).columns.tolist()
for col in numeric_cols: df_model[col] = df_model[col].astype(np.float32)

train_indices, test_indices = [], []
for month in df_model['mes'].unique():
    month_df = df_model[df_model['mes'] == month]
    train_idx, test_idx = train_test_split(month_df.index, train_size=0.8, random_state=SEED)
    train_indices.extend(train_idx); test_indices.extend(test_idx)
train_df = df_model.loc[train_indices]; test_df = df_model.loc[test_indices]
TARGET = 'quantity_sum'; cols_to_drop = [TARGET, 'transaction_date', 'first_day_of_week', 'ano']
X_train = train_df.drop(columns=cols_to_drop); y_train = train_df[TARGET].astype(np.float32)
X_test = test_df.drop(columns=cols_to_drop); y_test = test_df[TARGET].astype(np.float32)
print("✓ Dados preparados e divididos em treino/teste com sucesso.")
gc.collect()

# ==============================================================================
# FASE 4: OTIMIZAÇÃO E TREINAMENTO FINAL
# ==============================================================================
print("--- Iniciando Fase 4: Otimização e Treinamento Final ---")
# 4.1: Busca de Hiperparâmetros com Optuna
print("\n--- 4.1: Buscando melhores hiperparâmetros com Optuna ---")
train_sample_size = int(len(X_train) * 0.10); test_sample_size = int(len(X_test) * 0.10)
np.random.seed(SEED)
train_sample_idx = np.random.choice(X_train.index, train_sample_size, replace=False)
test_sample_idx = np.random.choice(X_test.index, test_sample_size, replace=False)
X_train_sample = X_train.loc[train_sample_idx]; y_train_sample = y_train.loc[train_sample_idx]
X_test_sample = X_test.loc[test_sample_idx]; y_test_sample = y_test.loc[test_sample_idx]
categorical_features_in_train = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
lgb_train_sample = lgb.Dataset(X_train_sample, label=y_train_sample, categorical_feature=categorical_features_in_train)
lgb_test_sample = lgb.Dataset(X_test_sample, label=y_test_sample, reference=lgb_train_sample)
def objective_optimized(trial):
    params = {'boosting_type': 'goss', 
              'objective': 'regression_l1', 
              'metric': 'l1', 
              'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.12),
              'num_leaves': trial.suggest_int('num_leaves', 20, 60),
              'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
              'lambda_l1': trial.suggest_float('lambda_l1', 0.001, 10.0, log=True),
              'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 10.0, log=True),
              'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
              'num_threads': -1,
              'seed': SEED,
              'verbosity': -1,
              'feature_pre_filter': False}
    try:
        model = lgb.train(params=params, train_set=lgb_train_sample, num_boost_round=1000, valid_sets=[lgb_test_sample], valid_names=['valid'], callbacks=[lgb.early_stopping(30, verbose=False), LightGBMPruningCallback(trial, "l1", valid_name="valid")])
        score = wmape(y_test_sample, np.maximum(0, np.round(model.predict(X_test_sample, num_iteration=model.best_iteration))).astype(int))
        trial.set_user_attr('n_estimators', model.best_iteration); return score
    except Exception: return float('inf')

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
N_TRIALS = 20
study.optimize(objective_optimized, n_trials=N_TRIALS, n_jobs=-1, gc_after_trial=True)
if study.best_value != float('inf'):
    best_params = study.best_params; best_n_estimators = study.best_trial.user_attrs.get('n_estimators', 500)
    print("✓ Busca do Optuna concluída com sucesso.")
else:
    print("⚠️  Busca do Optuna falhou. Usaremos parâmetros padrão."); best_params = {}; best_n_estimators = 500

# 4.2: Treinamento do Modelo Final
print("\n--- 4.2: Treinando modelo final com parâmetros otimizados ---")
final_params = best_params.copy()
final_params.update({'boosting_type': 'goss',
                     'objective': 'regression_l1',
                     'metric': 'mae',
                     'verbosity': -1,
                     'seed': SEED,
                     'num_threads': -1,
                     'feature_pre_filter': False})
if not isinstance(best_n_estimators, int) or best_n_estimators <= 0: best_n_estimators = 500
final_params['n_estimators'] = best_n_estimators

final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(X_train, y_train)
print("✓ Modelo final treinado.")

# ==============================================================================
# FASE 5: GERAÇÃO DA SUBMISSÃO FINAL
# ==============================================================================
print("--- Iniciando Fase 5: Geração da Submissão Final ---")

# 5.1: Criação da Estrutura da Previsão
df_q4 = df_model[df_model['transaction_date'].dt.quarter == 4]
active_pairs = df_q4[['internal_store_id', 'internal_product_id']].drop_duplicates()
prediction_weeks_start_dates = pd.to_datetime(['2023-01-02', '2023-01-09', '2023-01-16', '2023-01-23', '2023-01-30'])
df_scaffold = pd.DataFrame(list(product(active_pairs.index, prediction_weeks_start_dates)), columns=['pair_index', 'transaction_date'])
df_scaffold = active_pairs.merge(df_scaffold, left_index=True, right_on='pair_index').drop(columns='pair_index')
print(f"✓ Estrutura de previsão criada para {len(active_pairs):,} pares ativos.")

# 5.2: Previsão Autoregressiva
print("\n--- 5.2: Iniciando o processo de previsão autoregressiva ---")
df_scaffold_rich = df_scaffold.merge(df_produtos_features[product_features], on='internal_product_id', how='left')
df_scaffold_rich = df_scaffold_rich.merge(df_pdv_features[pdv_features], on='internal_store_id', how='left')
full_history_df = pd.concat([df_model, df_scaffold_rich], ignore_index=True)
full_history_df = full_history_df.sort_values(by=['internal_store_id', 'internal_product_id', 'transaction_date']).reset_index(drop=True)

full_history_df['mes'] = full_history_df['transaction_date'].dt.month
full_history_df['semana_do_ano'] = full_history_df['transaction_date'].dt.isocalendar().week
full_history_df['dia_do_mes'] = full_history_df['transaction_date'].dt.day
full_history_df['dia_do_ano'] = full_history_df['transaction_date'].dt.dayofyear
full_history_df['trimestre'] = full_history_df['transaction_date'].dt.quarter
for col in categorical_features:
    if col in full_history_df.columns:
        full_history_df[col] = full_history_df[col].fillna('Desconhecido').astype('category')
        if col in X_train.columns: full_history_df[col] = full_history_df[col].cat.set_categories(X_train[col].cat.categories)

all_predictions = []
features_for_prediction = [f for f in X_train.columns]
for week_date in prediction_weeks_start_dates:
    print(f"   - Prevendo para a semana de {week_date.strftime('%Y-%m-%d')}...")
    for lag in lags: full_history_df[f'quantity_lag_{lag}'] = full_history_df.groupby(['internal_store_id', 'internal_product_id'])['quantity_sum'].shift(lag)
    full_history_df[f'quantity_rolling_mean_{window_size}'] = full_history_df.groupby(['internal_store_id', 'internal_product_id'])['quantity_sum'].shift(1).rolling(window=window_size).mean()
    full_history_df[f'quantity_rolling_std_{window_size}'] = full_history_df.groupby(['internal_store_id', 'internal_product_id'])['quantity_sum'].shift(1).rolling(window=window_size).std()
    cols_to_fill = [col for col in full_history_df.columns if 'lag' in col or 'rolling' in col]
    full_history_df[cols_to_fill] = full_history_df[cols_to_fill].fillna(0)
    df_to_predict = full_history_df[full_history_df['transaction_date'] == week_date]
    week_preds = final_model.predict(df_to_predict[features_for_prediction])
    week_preds_rounded = np.maximum(0, np.round(week_preds)).astype(int)
    full_history_df.loc[df_to_predict.index, 'quantity_sum'] = week_preds_rounded
    df_to_predict_final = df_to_predict.copy(); df_to_predict_final['quantity'] = week_preds_rounded
    all_predictions.append(df_to_predict_final[['transaction_date', 'internal_store_id', 'internal_product_id', 'quantity']])
print("✓ Processo de previsão autoregressiva concluído.")

# 5.3: Formatação, Verificação de Limite e Salvamento
print("\n--- 5.3: Formatando e salvando a submissão final ---")
submission_df = pd.concat(all_predictions, ignore_index=True)
week_map = {date: i+1 for i, date in enumerate(prediction_weeks_start_dates)}
submission_df['semana'] = submission_df['transaction_date'].map(week_map)
submission_df = submission_df.rename(columns={'internal_store_id': 'pdv', 'internal_product_id': 'produto', 'quantity': 'quantidade'})
final_submission = submission_df[['semana', 'pdv', 'produto', 'quantidade']].astype(int)

print(f"✓ Shape inicial do arquivo final: {final_submission.shape}")
if len(final_submission) > 1500000:
    print(f"⚠️  ATENÇÃO: O arquivo final com {len(final_submission):,} linhas excede o limite de 1.5M.")
    print("   - Removendo previsões de valor zero como fallback...")
    final_submission = final_submission[final_submission['quantidade'] > 0]
    print(f"✓ Shape após remoção dos zeros: {final_submission.shape}")
    if len(final_submission) > 1500000: print("❌ ERRO: Mesmo após remover os zeros, o arquivo ainda excede o limite.")
    else: print("✅ O número de linhas agora está dentro do limite.")
else: print("✓ O número de linhas está dentro do limite.")

print("\n💾 Salvando os arquivos de submissão...")
final_submission.to_csv('previsao_vendas_janeiro2023.csv', index=False, sep=';')
final_submission.to_parquet('previsao_vendas_janeiro2023.parquet', index=False)
print("✓ Arquivos .csv e .parquet salvos com sucesso.")