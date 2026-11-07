"""Avalia previsões anuais de FCL com origem móvel e janela expansiva."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data_input"
OUTPUT_DIR = BASE_DIR / "data_output"
FIG_DIR = BASE_DIR / "fig" / "FCL"
TICKERS_PADRAO = ("ITUB3.SA", "FESA4.SA", "EGIE3.SA", "VALE3.SA")
ANOS_MIN_TREINO = 5


def carregar_dados(caminho_fcl, caminho_inpc):
    """Carrega o FCL nominal e o INPC anual sem modificar os arquivos de entrada."""
    fcl = pd.read_csv(caminho_fcl, sep="\t", decimal=",", index_col=0)
    fcl.index = fcl.index.astype(str).str.strip()
    fcl.columns = pd.Index([int(str(coluna).strip()) for coluna in fcl.columns])
    fcl = fcl.apply(pd.to_numeric, errors="coerce")

    inpc = pd.read_csv(caminho_inpc, index_col="ano", decimal=",")
    inpc.index = inpc.index.astype(int)
    inpc = pd.to_numeric(inpc["INPC"], errors="coerce").sort_index()

    if fcl.empty or inpc.empty:
        raise ValueError("Os arquivos de FCL e INPC precisam conter dados.")
    if fcl.columns.has_duplicates or inpc.index.has_duplicates:
        raise ValueError("Os arquivos não podem conter anos duplicados.")
    return fcl.sort_index(axis="columns"), inpc


def converter_para_base_precos(valor, ano_valor, ano_base, inpc):
    """Converte valores nominais para a base de preços indicada usando INPC anual."""
    ano_valor = int(ano_valor)
    ano_base = int(ano_base)
    if ano_valor == ano_base:
        return float(valor)

    primeiro_ano = min(ano_valor, ano_base) + 1
    ultimo_ano = max(ano_valor, ano_base)
    anos_correcao = range(primeiro_ano, ultimo_ano + 1)
    taxas = inpc.reindex(anos_correcao)
    if taxas.isna().any():
        faltantes = taxas[taxas.isna()].index.astype(str).tolist()
        raise ValueError(f"INPC ausente para os anos: {', '.join(faltantes)}")
    if (taxas <= -100).any():
        raise ValueError("As taxas de INPC precisam ser maiores que -100%.")

    fator = float(np.prod(1 + taxas.to_numpy(dtype=float) / 100))
    return float(valor) * fator if ano_valor < ano_base else float(valor) / fator


def gerar_divisoes_temporais(anos, anos_min_treino=ANOS_MIN_TREINO):
    """Gera pares (anos de treino, ano previsto) consecutivos no tempo."""
    anos = np.asarray(sorted({int(ano) for ano in anos}), dtype=int)
    if anos_min_treino < 2:
        raise ValueError("O treino precisa conter pelo menos dois anos.")
    if len(anos) <= anos_min_treino:
        raise ValueError(
            f"São necessários mais de {anos_min_treino} anos completos para avaliar."
        )
    if not np.all(np.diff(anos) == 1):
        raise ValueError("A série anual precisa ser contínua, sem anos ausentes.")

    return [
        (anos[:indice], int(anos[indice]))
        for indice in range(anos_min_treino, len(anos))
    ]


def previsoes_referencia(anos_treino, valores_treino, ano_previsto):
    """Calcula previsões de último valor, média histórica e tendência linear."""
    anos_treino = np.asarray(anos_treino, dtype=float)
    valores_treino = np.asarray(valores_treino, dtype=float)
    if len(anos_treino) != len(valores_treino) or len(anos_treino) < 2:
        raise ValueError("Anos e valores de treino devem ter o mesmo tamanho (>= 2).")
    if not np.isfinite(anos_treino).all() or not np.isfinite(valores_treino).all():
        raise ValueError("Os dados de treino precisam ser finitos.")

    inclinacao, intercepto = np.polyfit(anos_treino, valores_treino, 1)
    return {
        "Último valor": float(valores_treino[-1]),
        "Média histórica": float(valores_treino.mean()),
        "Tendência linear": float(inclinacao * ano_previsto + intercepto),
    }


def calcular_metricas(previsoes):
    """Resume erros pontuais e, quando disponíveis, métricas dos intervalos."""
    erro = previsoes["previsao_media"] - previsoes["observado"]
    resultado = {
        "n_previsoes": int(erro.count()),
        "mae": float(erro.abs().mean()),
        "rmse": float(np.sqrt(np.square(erro).mean())),
        "vies_medio": float(erro.mean()),
    }

    intervalos_validos = previsoes["p10"].notna() & previsoes["p90"].notna()
    if intervalos_validos.any():
        observados = previsoes.loc[intervalos_validos, "observado"]
        resultado["cobertura_80_pct"] = float(
            (
                observados.ge(previsoes.loc[intervalos_validos, "p10"])
                & observados.le(previsoes.loc[intervalos_validos, "p90"])
            ).mean()
            * 100
        )
        resultado["largura_media_intervalo"] = float(
            (
                previsoes.loc[intervalos_validos, "p90"]
                - previsoes.loc[intervalos_validos, "p10"]
            ).mean()
        )
    else:
        resultado["cobertura_80_pct"] = np.nan
        resultado["largura_media_intervalo"] = np.nan
    return resultado


def prever_bayesiano_multivariado(
    anos_treino,
    valores_treino,
    ano_previsto,
    draws=1000,
    tune=1000,
    seed=42,
):
    """Ajusta o modelo multivariado do Portfolio.py e simula o ano seguinte."""
    try:
        import pymc as pm
    except ImportError as erro:
        raise RuntimeError(
            "A avaliação bayesiana requer as dependências de requirements.txt."
        ) from erro

    anos_treino = np.asarray(anos_treino, dtype=float)
    valores_treino = np.asarray(valores_treino, dtype=float)
    media_anos = anos_treino.mean()
    escala_anos = anos_treino.std()
    anos_padronizados = (anos_treino - media_anos) / escala_anos
    ano_padronizado = (float(ano_previsto) - media_anos) / escala_anos
    numero_empresas = valores_treino.shape[1]

    with pm.Model() as modelo:
        alpha = pm.Normal(
            "alpha",
            mu=valores_treino.mean(axis=0),
            sigma=10,
            shape=numero_empresas,
        )
        beta = pm.Normal("beta", mu=0, sigma=10, shape=numero_empresas)
        chol, _, _ = pm.LKJCholeskyCov(
            "covariancia",
            n=numero_empresas,
            eta=2,
            sd_dist=pm.Exponential.dist(1),
            compute_corr=True,
        )
        media_fcl = alpha[None, :] + beta[None, :] * anos_padronizados[:, None]
        pm.MvNormal(
            "FCL_observado",
            mu=media_fcl,
            chol=chol,
            observed=valores_treino,
        )
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=1,
            target_accept=0.90,
            random_seed=seed,
            return_inferencedata=True,
            progressbar=False,
        )

    alpha_amostras = idata.posterior["alpha"].values.reshape(-1, numero_empresas)
    beta_amostras = idata.posterior["beta"].values.reshape(-1, numero_empresas)
    desvios = idata.posterior["covariancia_stds"].values.reshape(
        -1, numero_empresas
    )
    correlacoes = idata.posterior["covariancia_corr"].values.reshape(
        -1, numero_empresas, numero_empresas
    )
    medias = alpha_amostras + beta_amostras * ano_padronizado
    covariancias = (
        desvios[:, :, None] * correlacoes * desvios[:, None, :]
    )

    rng = np.random.default_rng(seed)
    previsoes = np.empty_like(medias)
    for indice, (media, covariancia) in enumerate(zip(medias, covariancias)):
        previsoes[indice] = rng.multivariate_normal(
            mean=media,
            cov=covariancia,
            check_valid="raise",
        )
    return {
        "previsao_media": previsoes.mean(axis=0),
        "p10": np.percentile(previsoes, 10, axis=0),
        "p90": np.percentile(previsoes, 90, axis=0),
    }


def avaliar_previsoes(
    fcl,
    inpc,
    tickers=TICKERS_PADRAO,
    anos_min_treino=ANOS_MIN_TREINO,
    incluir_bayesiano=True,
    draws=1000,
    tune=1000,
    seed=42,
):
    """Executa avaliação temporal e retorna previsões e métricas por empresa."""
    tickers = tuple(tickers)
    ausentes = sorted(set(tickers) - set(fcl.index))
    if ausentes:
        raise ValueError(f"Tickers ausentes no FCL: {', '.join(ausentes)}")

    anos = np.asarray(fcl.columns, dtype=int)
    divisoes = gerar_divisoes_temporais(anos, anos_min_treino)
    dados = fcl.loc[list(tickers), anos].astype(float)
    if dados.isna().any().any() or not np.isfinite(dados.to_numpy()).all():
        raise ValueError("O FCL selecionado contém valores ausentes ou não finitos.")

    registros = []
    for indice_divisao, (anos_treino, ano_previsto) in enumerate(divisoes):
        origem = int(anos_treino[-1])
        valores_treino = np.column_stack(
            [
                [
                    converter_para_base_precos(
                        dados.loc[ticker, ano],
                        ano,
                        origem,
                        inpc,
                    )
                    for ano in anos_treino
                ]
                for ticker in tickers
            ]
        )
        observados = np.asarray(
            [
                converter_para_base_precos(
                    dados.loc[ticker, ano_previsto],
                    ano_previsto,
                    origem,
                    inpc,
                )
                for ticker in tickers
            ]
        )
        previsoes_por_metodo = []
        for indice_ticker, ticker in enumerate(tickers):
            referencias = previsoes_referencia(
                anos_treino,
                valores_treino[:, indice_ticker],
                ano_previsto,
            )
            for metodo, previsao in referencias.items():
                previsoes_por_metodo.append(
                    (ticker, metodo, previsao, np.nan, np.nan)
                )

        if incluir_bayesiano:
            previsao_bayesiana = prever_bayesiano_multivariado(
                anos_treino,
                valores_treino,
                ano_previsto,
                draws=draws,
                tune=tune,
                seed=seed + indice_divisao,
            )
            for indice_ticker, ticker in enumerate(tickers):
                previsoes_por_metodo.append(
                    (
                        ticker,
                        "Bayesiano multivariado",
                        float(previsao_bayesiana["previsao_media"][indice_ticker]),
                        float(previsao_bayesiana["p10"][indice_ticker]),
                        float(previsao_bayesiana["p90"][indice_ticker]),
                    )
                )

        for ticker, metodo, previsao, p10, p90 in previsoes_por_metodo:
            observado = float(observados[tickers.index(ticker)])
            registros.append(
                {
                    "ano_origem": origem,
                    "ano_previsto": ano_previsto,
                    "ticker": ticker,
                    "metodo": metodo,
                    "previsao_media": previsao,
                    "p10": p10,
                    "p90": p90,
                    "observado": observado,
                    "erro": previsao - observado,
                    "erro_absoluto": abs(previsao - observado),
                    "base_precos": origem,
                }
            )

    detalhado = pd.DataFrame(registros)
    resumos = []
    for (ticker, metodo), grupo in detalhado.groupby(["ticker", "metodo"]):
        resumos.append(
            {"ticker": ticker, "metodo": metodo, **calcular_metricas(grupo)}
        )
    return detalhado, pd.DataFrame(resumos)


def salvar_resultados(detalhado, resumos):
    """Salva tabelas e gráficos da avaliação nos diretórios existentes."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    caminho_detalhado = OUTPUT_DIR / "validacao_previsoes_detalhada.csv"
    caminho_resumo = OUTPUT_DIR / "validacao_previsoes_resumo.csv"
    detalhado.to_csv(caminho_detalhado, index=False, decimal=",")
    resumos.to_csv(caminho_resumo, index=False, decimal=",")

    tickers = detalhado["ticker"].drop_duplicates().tolist()
    figura, eixos = plt.subplots(
        len(tickers),
        1,
        figsize=(10, 4 * len(tickers)),
        squeeze=False,
    )
    for indice, ticker in enumerate(tickers):
        eixo = eixos[indice, 0]
        dados_ticker = detalhado.loc[detalhado["ticker"] == ticker]
        observados = (
            dados_ticker[["ano_previsto", "observado"]]
            .drop_duplicates()
            .sort_values("ano_previsto")
        )
        eixo.plot(
            observados["ano_previsto"],
            observados["observado"],
            color="black",
            marker="o",
            label="Observado",
        )
        for metodo, grupo in dados_ticker.groupby("metodo", sort=False):
            grupo = grupo.sort_values("ano_previsto")
            eixo.plot(
                grupo["ano_previsto"],
                grupo["previsao_media"],
                marker=".",
                label=metodo,
            )
            if grupo["p10"].notna().any():
                eixo.fill_between(
                    grupo["ano_previsto"].to_numpy(dtype=float),
                    grupo["p10"].to_numpy(dtype=float),
                    grupo["p90"].to_numpy(dtype=float),
                    alpha=0.18,
                    label=f"{metodo} P10–P90",
                )
        eixo.set_title(ticker)
        eixo.set_xlabel("Ano previsto")
        eixo.set_ylabel(f"FCL (preços do ano de origem)")
        eixo.grid(alpha=0.3)
        eixo.legend()

    figura.suptitle("Avaliação retrospectiva de previsões anuais de FCL")
    figura.tight_layout()
    figura.savefig(FIG_DIR / "validacao_previsoes.png", dpi=200, bbox_inches="tight")
    plt.close(figura)
    return caminho_detalhado, caminho_resumo


def main():
    parser = argparse.ArgumentParser(
        description="Compara previsões anuais de FCL com avaliação temporal."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=TICKERS_PADRAO,
        help="Tickers a avaliar (padrão: quatro empresas do Portfolio.py).",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=1000,
        help="Amostras posteriores por cadeia do modelo bayesiano.",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=1000,
        help="Amostras de ajuste por cadeia do modelo bayesiano.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    parser.add_argument(
        "--somente-referencias",
        action="store_true",
        help="Executa apenas as três referências, sem o modelo bayesiano.",
    )
    argumentos = parser.parse_args()

    if argumentos.draws < 1 or argumentos.tune < 1:
        parser.error("--draws e --tune precisam ser positivos.")
    fcl, inpc = carregar_dados(
        INPUT_DIR / "FCL.csv",
        INPUT_DIR / "INPC_anual.csv",
    )
    detalhado, resumos = avaliar_previsoes(
        fcl,
        inpc,
        tickers=argumentos.tickers,
        incluir_bayesiano=not argumentos.somente_referencias,
        draws=argumentos.draws,
        tune=argumentos.tune,
        seed=argumentos.seed,
    )
    caminho_detalhado, caminho_resumo = salvar_resultados(detalhado, resumos)
    print("Métricas por empresa e método:")
    print(resumos.to_string(index=False))
    print(f"Previsões detalhadas salvas em: {caminho_detalhado}")
    print(f"Resumo das métricas salvo em: {caminho_resumo}")
    print(f"Gráfico salvo em: {FIG_DIR / 'validacao_previsoes.png'}")


if __name__ == "__main__":
    main()
