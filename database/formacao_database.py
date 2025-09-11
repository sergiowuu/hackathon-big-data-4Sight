import polars as pl
import os

arquivo_produtos = r"E:\jogo\My project (1)\hakaton\dados\local.parquet"
arquivo_transacoes = r"E:\jogo\My project (1)\hakaton\dados\produtos.parquet"
arquivo_pvd = r"E:\jogo\My project (1)\hakaton\dados\vendas.parquet"

# Verificando se os arquivos existem antes de tentar carregá-los.
if not all(os.path.exists(path) for path in [arquivo_produtos, arquivo_transacoes, arquivo_pvd]):
    print("Erro: Um ou mais arquivos não foram encontrados. Verifique os caminhos.")
else:
    # 2. Carregamento dos dados usando Polars.
    # Agora as variáveis são DataFrames.
    df_produtos = pl.read_parquet(arquivo_produtos)
    df_transacoes = pl.read_parquet(arquivo_transacoes)
    df_pvd = pl.read_parquet(arquivo_pvd)
    
    print("Amostra do DataFrame de Produtos: ")
    print(df_produtos.head())
    print("\nAmostra do DataFrame de Transações: ")
    print(df_transacoes.head())
    print("\nAmostra do DataFrame de PVDs: ")
    print(df_pvd.head())
    
    # 3. Unificação dos dados
    df_mestre = df_transacoes.join(
        df_pvd,
        left_on="internal_store_id",
        right_on="pdv",
        how="left"
    )

    df_mestre = df_mestre.join(
        df_produtos,
        left_on="internal_product_id",
        right_on="produto", 
        how="left"
    )

    print("\nAmostra do DataFrame Mestre: ")
    print(df_mestre.head())
    print(f"\nNúmero de linhas no DataFrame Mestre: {len(df_mestre)}")

    # 4. Organizando os dados por semana
    #df_mestre = df_mestre.with_columns(
       # pl.col("transaction_date").str.strptime(pl.Date, "%Y-%m-%d").alias("data")
   # )
    df_agregado = df_mestre.with_columns(
    semana=pl.col("transaction_date").dt.week()
    ).group_by(
    ["semana", "internal_store_id", "internal_product_id"]
    ).agg(
    pl.sum("quantity").alias("quantidade_semanal"),
    pl.sum("gross_value").alias("faturamento_semanal")
    ).sort(
    "semana"
        )

    print("\nAmostra do DataFrame Agregado por Semana: ")
    print(df_agregado.head())
    print(f"\nNúmero de linhas no DataFrame Agregado: {len(df_agregado)}")

  

    # Salvando o DataFrame agregado na pasta 'data/processed'
    caminho_saida = r"E:\jogo\My project (1)\hakaton\saida\saida.parquet"
    df_agregado.write_parquet(caminho_saida)
    
    print(f"\nDataFrame salvo com sucesso em: {caminho_saida}")