def calcular_surpresa(df, coluna_actual="actual", coluna_forecast="forecast"):
    """
    Calcula a surpresa (z-score) entre valor divulgado e consenso.

    Parâmetros:
        df: DataFrame com os eventos do indicador
        coluna_actual: nome da coluna com o valor divulgado
        coluna_forecast: nome da coluna com o valor esperado (consenso)

    Retorna:
        O mesmo DataFrame, com as colunas 'diferenca' e 'surpresa_zscore' adicionadas
    """
    df = df.copy()
    df["diferenca"] = df[coluna_actual] - df[coluna_forecast]
    desvio_padrao = df["diferenca"].std()
    df["surpresa_zscore"] = df["diferenca"] / desvio_padrao
    return df


def calcular_ian(eventos, termos_busca, geo="US", timeframe="2023-01-01 2026-08-06"):
    """
    Calcula o IAN (Índice de Atenção Normalizado) via Google Trends.

    Parâmetros:
        eventos: DataFrame com a coluna 'data' (datas dos eventos, já com surpresa calculada)
        termos_busca: lista de termos de busca relacionados ao indicador (ex.: ["CPI", "inflation report"])
        geo: código do país no Google Trends (ex.: "US", "BR")
        timeframe: período da busca, no formato "AAAA-MM-DD AAAA-MM-DD"

    Retorna:
        O DataFrame de eventos, com as colunas 'atencao_bruta' e 'IAN' adicionadas
    """
    from trendspy import Trends
    import pandas as pd

    tr = Trends()
    tendencias = tr.interest_over_time(termos_busca, timeframe=timeframe, geo=geo)

    tendencias["atencao_bruta"] = tendencias[termos_busca].mean(axis=1)
    tendencias = tendencias.reset_index().rename(columns={"time [UTC]": "data"})
    tendencias["data"] = tendencias["data"].astype("datetime64[ns]")

    eventos = eventos.copy()
    eventos["data"] = pd.to_datetime(eventos["data"]).astype("datetime64[ns]")

    tendencias = tendencias.sort_values("data")
    eventos = eventos.sort_values("data")

    eventos = pd.merge_asof(eventos, tendencias[["data", "atencao_bruta"]], on="data", direction="backward")

    minimo = eventos["atencao_bruta"].min()
    maximo = eventos["atencao_bruta"].max()
    eventos["IAN"] = (eventos["atencao_bruta"] - minimo) / (maximo - minimo)

    return eventos


def calcular_ice(eventos, termos_otimistas, termos_pessimistas, geo="BR", timeframe="2023-01-01 2026-08-06"):
    """
    Calcula o ICE (Índice de Clima Econômico) via Google Trends.

    Parâmetros:
        eventos: DataFrame com a coluna 'data' (datas dos eventos, já com surpresa e IAN calculados)
        termos_otimistas: lista de termos de busca da cesta otimista
        termos_pessimistas: lista de termos de busca da cesta pessimista
        geo: código do país no Google Trends (ex.: "US", "BR")
        timeframe: período da busca, no formato "AAAA-MM-DD AAAA-MM-DD"

    Retorna:
        O DataFrame de eventos, com a coluna 'ICE' adicionada
    """
    from trendspy import Trends
    from statsmodels.tsa.seasonal import STL
    import pandas as pd
    import numpy as np

    tr = Trends()
    cesta_otimista = tr.interest_over_time(termos_otimistas, timeframe=timeframe, geo=geo)
    cesta_pessimista = tr.interest_over_time(termos_pessimistas, timeframe=timeframe, geo=geo)

    cesta_otimista["media_otimista"] = cesta_otimista[termos_otimistas].mean(axis=1)
    cesta_pessimista["media_pessimista"] = cesta_pessimista[termos_pessimistas].mean(axis=1)

    cesta_otimista["z_otimista"] = (cesta_otimista["media_otimista"] - cesta_otimista["media_otimista"].mean()) / cesta_otimista["media_otimista"].std()
    cesta_pessimista["z_pessimista"] = (cesta_pessimista["media_pessimista"] - cesta_pessimista["media_pessimista"].mean()) / cesta_pessimista["media_pessimista"].std()

    ice_bruto = pd.DataFrame({
        "z_otimista": cesta_otimista["z_otimista"],
        "z_pessimista": cesta_pessimista["z_pessimista"]
    })

    stl_otimista = STL(ice_bruto["z_otimista"], period=52, robust=True).fit()
    stl_pessimista = STL(ice_bruto["z_pessimista"], period=52, robust=True).fit()

    ice_bruto["z_otimista_dessaz"] = stl_otimista.trend + stl_otimista.resid
    ice_bruto["z_pessimista_dessaz"] = stl_pessimista.trend + stl_pessimista.resid
    ice_bruto["ICE"] = np.tanh(ice_bruto["z_otimista_dessaz"] - ice_bruto["z_pessimista_dessaz"])

    ice_final = ice_bruto.reset_index().rename(columns={"time [UTC]": "data"})
    ice_final["data"] = pd.to_datetime(ice_final["data"]).astype("datetime64[ns]")

    eventos = eventos.copy()
    eventos["data"] = pd.to_datetime(eventos["data"]).astype("datetime64[ns]")

    eventos = eventos.sort_values("data")
    ice_final = ice_final.sort_values("data")

    eventos = pd.merge_asof(eventos, ice_final[["data", "ICE"]], on="data", direction="backward")

    return eventos