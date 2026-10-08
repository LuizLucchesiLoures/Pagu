import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytensor.tensor as pt
import pandas as pd
import pymc as pm
import yfinance as yf
from pathlib import Path
from scipy.stats import gaussian_kde, halfnorm, norm
from datetime import date
from urllib.parse import urlencode
from urllib.request import urlopen
import json

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "fig" / "FCL"
OUTPUT_DIR = BASE_DIR / "data_output"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
ano_inicial = 2016
ano_final = 2037
ano_final_plot = 2035
anos_projecao = np.arange(ano_inicial, ano_final)
empresas = ["ITUB3.SA", "FESA4.SA", "EGIE3.SA", "VALE3.SA"]
empresas_multivariadas = empresas
USAR_FCL_CORRIGIDO_capex_expandido = True
USAR_FCL_FUTURO_PROJETADO = False
USAR_FCL_CORRIGIDO_INPC = True
USAR_EMPRESAS_MUTLIVARIADAS = True
USAR_PAYOUT_BAYESIANO = True
TICKERS_YAHOO_ACOES = {
    "ITUB3.SA": ("ITUB3.SA", "ITUB4.SA"),
    "FESA4.SA": ("FESA3.SA", "FESA4.SA"),
    "EGIE3.SA": ("EGIE3.SA",),
    "VALE3.SA": ("VALE3.SA",),
}


def buscar_quantidade_acoes_yahoo(tickers):
    quantidades = {}
    registros = []

    for ticker in tickers:
        tickers_yahoo = TICKERS_YAHOO_ACOES.get(ticker, (ticker,))

        try:
            quantidades_por_classe = []
            for ticker_yahoo in tickers_yahoo:
                informacoes = yf.Ticker(ticker_yahoo).get_info()
                quantidade = informacoes.get("sharesOutstanding")
                if quantidade is None:
                    raise ValueError(
                        f"Yahoo Finance não retornou sharesOutstanding para {ticker_yahoo}"
                    )
                quantidades_por_classe.append(int(quantidade))
            quantidades[ticker] = sum(quantidades_por_classe)
        except Exception as erro:
            quantidades[ticker] = None
            print(
                f"Não foi possível obter a quantidade de ações de {ticker} "
                f"usando {', '.join(tickers_yahoo)}: {erro}"
            )

        registros.append(
            {
                "ticker": ticker,
                "tickers_yahoo_consultados": ";".join(tickers_yahoo),
                "quantidade_acoes": quantidades[ticker],
                "data_consulta": date.today().isoformat(),
            }
        )

    return quantidades, pd.DataFrame(registros)


quantidades_acoes, tabela_quantidade_acoes = buscar_quantidade_acoes_yahoo(empresas)
quantidade_acoes_itub3 = quantidades_acoes["ITUB3.SA"]
quantidade_acoes_fesa4 = quantidades_acoes["FESA4.SA"]
quantidade_acoes_egie3 = quantidades_acoes["EGIE3.SA"]
quantidade_acoes_vale3 = quantidades_acoes["VALE3.SA"]

arquivo_quantidade_acoes = OUTPUT_DIR / "quantidade_acoes_yahoo.csv"
tabela_quantidade_acoes.to_csv(arquivo_quantidade_acoes, index=False)
print("Quantidade de ações em circulação obtida do Yahoo Finance:")
print(tabela_quantidade_acoes)
print(f"Tabela salva em: {arquivo_quantidade_acoes}")

def ler_inpc_anual(ano_inicial):
    """Lê as variações mensais do INPC e calcula o resultado anual.

    Tenta a API do Banco Central; se ela estiver indisponível, usa o arquivo
    local INPC_anual.csv como fallback.
    """
    ano_final = date.today().year - 1
    caminho_csv = BASE_DIR / "data_input" / "INPC_anual.csv"

    try:
        parametros = urlencode(
            {
                "formato": "json",
                "dataInicial": f"01/01/{ano_inicial}",
                "dataFinal": f"31/12/{ano_final}",
            }
        )
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.188/dados?{parametros}"

        with urlopen(url, timeout=30) as resposta:
            dados = json.load(resposta)

        inpc_mensal = pd.DataFrame(dados)
        inpc_mensal["data"] = pd.to_datetime(inpc_mensal["data"], dayfirst=True)
        inpc_mensal["valor"] = (
            inpc_mensal["valor"].str.replace(",", ".", regex=False).astype(float)
        )
        inpc_mensal["ano"] = inpc_mensal["data"].dt.year

        inpc_anual = (
            inpc_mensal.groupby("ano")["valor"]
            .apply(lambda variacoes: ((1 + variacoes / 100).prod() - 1) * 100)
            .rename("INPC")
            .to_frame()
        )
        inpc_anual.index.name = "ano"
        return inpc_anual

    except Exception as erro:
        if caminho_csv.exists():
            print(
                f"API do BCB indisponível ({erro.__class__.__name__}: {erro}). "
                f"Usando fallback do arquivo local: {caminho_csv}"
            )
            inpc_anual = pd.read_csv(
                caminho_csv,
                index_col="ano",
                decimal=",",
            )
            inpc_anual.index = inpc_anual.index.astype(int)
            inpc_anual = inpc_anual.loc[inpc_anual.index >= ano_inicial]
            return inpc_anual

        raise


inpc_anual = ler_inpc_anual(ano_inicial)
inpc_anual.to_csv(OUTPUT_DIR / "INPC_anual.csv", decimal=",")
print("INPC anual carregado:")
print(inpc_anual)

#leitura do FLC
fcl = pd.read_csv(
    BASE_DIR / "data_input" / "FCL.csv",
    sep="\t",
    decimal=",",
    index_col=0,
)

#leitura do Capex de expansão
capex_expansao = pd.read_csv(
    BASE_DIR / "data_input" / "Capex_Expancao.csv",
    sep="\t",
    decimal=",",
    index_col=0,
)

#leitura do FCL futuro projetado dos projetos de desenvolvimento
fcl_futuro_projetado = pd.read_csv(
    BASE_DIR / "data_input" / "FCL_futuro_projetado.csv",
    sep="\t",
    decimal=",",
    index_col=0,
)

#leitura dos proventos anuais por ação, para cálculo do dividend yield futuro
proventos_anuais = pd.read_csv(
    BASE_DIR / "data_input" / "proventos.csv",
    sep="\t",
    decimal=",",
    index_col=0,
)

fcl.columns = fcl.columns.str.strip()
capex_expansao.columns = capex_expansao.columns.str.strip()
fcl_futuro_projetado.columns = fcl_futuro_projetado.columns.str.strip()
fcl_futuro_projetado.index = fcl_futuro_projetado.index.str.strip()
fcl_futuro_projetado.columns = (
    fcl_futuro_projetado.columns.str.extract(r"(\d{4})", expand=False).astype(int)
)
proventos_anuais.columns = proventos_anuais.columns.str.strip()

#-------------------------------------------------------------------
# Ajusta o FCL adicionando o Capex de expansão, se disponível.

if USAR_FCL_CORRIGIDO_capex_expandido:
    fcl = fcl.add(capex_expansao, fill_value=0)

# Corrige os valores históricos do FCL para preços do último ano disponível.
ano_base_fcl = int(inpc_anual.index.max())
anos_fcl = fcl.columns.astype(int)
fatores_correcao_inpc = pd.Series(
    {
        ano: np.prod(
            1 + inpc_anual.loc[ano + 1 : ano_base_fcl, "INPC"].to_numpy() / 100
        )
        for ano in anos_fcl
    },
    name="fator_correcao_inpc",
)
fatores_correcao_inpc.index = fcl.columns
fcl_corrigido = fcl.mul(fatores_correcao_inpc, axis="columns")
fcl_corrigido.to_csv(
    OUTPUT_DIR / "FCL_corrigido_INPC.csv",
    sep="\t",
    decimal=",",
)
print(f"FCL corrigido para preços de {ano_base_fcl}:")
print(fcl_corrigido)

# Escolha os dados usados no modelo e nos gráficos:
# True usa o FCL corrigido pelo INPC; False usa os valores originais.
#USAR_FCL_CORRIGIDO_INPC = True

fcl_utilizado = fcl_corrigido if USAR_FCL_CORRIGIDO_INPC else fcl
nome_serie_fcl = "corrigido pelo INPC" if USAR_FCL_CORRIGIDO_INPC else "original"
print(f"FCL utilizado: {nome_serie_fcl}")


def construir_modelo_bayesiano(ticker, eixos):
    anos = np.asarray(fcl_utilizado.columns, dtype=float)
    fluxo_caixa = fcl_utilizado.loc[ticker].astype(float).to_numpy()
    anos_padronizados = (anos - anos.mean()) / anos.std()

    with pm.Model() as modelo:
        alpha = pm.Normal("alpha", mu=fluxo_caixa.mean(), sigma=10 * fluxo_caixa.std())
        beta = pm.Normal("beta", mu=0, sigma=10 * fluxo_caixa.std())
        sigma = pm.HalfNormal("sigma", sigma=fluxo_caixa.std())
        media_fcl = alpha + beta * anos_padronizados
        pm.Normal("FCL_observado", mu=media_fcl, sigma=sigma, observed=fluxo_caixa)
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=1,
            target_accept=0.90,
            random_seed=42,
            return_inferencedata=True,
        )

    alpha_amostras = idata.posterior["alpha"].values.flatten()
    beta_amostras = idata.posterior["beta"].values.flatten()
    sigma_amostras = idata.posterior["sigma"].values.flatten()

    #anos_projecao = np.arange(2016, 2037)
    anos_projecao_padronizados = (anos_projecao - anos.mean()) / anos.std()
    media_previsao = (
        alpha_amostras[:, None]
        + beta_amostras[:, None] * anos_projecao_padronizados[None, :]
    )
    previsoes_fcl = np.random.default_rng(42).normal(
        loc=media_previsao,
        scale=sigma_amostras[:, None],
    )
    fcl_medio = np.mean(previsoes_fcl, axis=0)
    fcl_p90 = np.percentile(previsoes_fcl, 90, axis=0)
    fcl_p10 = np.percentile(previsoes_fcl, 10, axis=0)
    if USAR_FCL_FUTURO_PROJETADO:
        fcl_p90 += (
            fcl_futuro_projetado.loc[ticker]
            .reindex(anos_projecao, fill_value=0)
            .to_numpy(dtype=float)
        )
        fcl_medio += (
            fcl_futuro_projetado.loc[ticker]
            .reindex(anos_projecao, fill_value=0)
            .to_numpy(dtype=float)
        )/2

    parametros = ["alpha", "beta", "sigma"]
    resumo = az.summary(idata, var_names=parametros)
    print(f"\nResumo do modelo bayesiano: {ticker}")
    print(resumo)

    alpha_medio = alpha_amostras.mean()
    beta_medio = beta_amostras.mean()
    sigma_medio = sigma_amostras.mean()
    alpha_valores = np.linspace(alpha_amostras.min(), alpha_amostras.max(), 500)
    beta_valores = np.linspace(beta_amostras.min(), beta_amostras.max(), 500)
    sigma_valores = np.linspace(max(0.0001, sigma_amostras.min()), sigma_amostras.max(), 500)

    def calcular_likelihood(parametros_grafico, nome_parametro):
        valores = []
        for valor in parametros_grafico:
            if nome_parametro == "alpha":
                media = valor + beta_medio * anos_padronizados
                desvio = sigma_medio
            elif nome_parametro == "beta":
                media = alpha_medio + valor * anos_padronizados
                desvio = sigma_medio
            else:
                media = alpha_medio + beta_medio * anos_padronizados
                desvio = valor
            valores.append(np.sum(norm.logpdf(fluxo_caixa, loc=media, scale=desvio)))
        valores = np.asarray(valores)
        valores -= valores.max()
        return np.exp(valores)

    def normalizar(curva):
        return curva / curva.max()

    curvas = [
        (alpha_valores, norm.pdf(alpha_valores, fluxo_caixa.mean(), 10 * fluxo_caixa.std()), calcular_likelihood(alpha_valores, "alpha"), gaussian_kde(alpha_amostras)(alpha_valores), "alpha"),
        (beta_valores, norm.pdf(beta_valores, 0, 10 * fluxo_caixa.std()), calcular_likelihood(beta_valores, "beta"), gaussian_kde(beta_amostras)(beta_valores), "beta"),
        (sigma_valores, halfnorm.pdf(sigma_valores, scale=fluxo_caixa.std()), calcular_likelihood(sigma_valores, "sigma"), gaussian_kde(sigma_amostras)(sigma_valores), "sigma"),
    ]

    eixo_fcl = eixos[0]
    eixo_fcl.scatter(anos, fluxo_caixa, color="black", label="FCL observado")
    eixo_fcl.plot(anos_projecao, fcl_medio, color="blue", label="FCL médio")
    eixo_fcl.plot(anos_projecao, fcl_p90, color="green", linestyle="--", label="FCL P90")
    eixo_fcl.plot(anos_projecao, fcl_p10, color="red", linestyle="--", label="FCL P10")
    eixo_fcl.set_xlim(ano_inicial, ano_final_plot)
    eixo_fcl.set_title(f"{ticker} - FCL até {ano_final_plot}")
    eixo_fcl.set_xlabel("Ano")
    eixo_fcl.set_ylabel("Fluxo de caixa livre")
    eixo_fcl.legend()
    eixo_fcl.grid(alpha=0.3)

    for eixo, (valores, prior, likelihood, posterior, nome) in zip(eixos[1:], curvas):
        eixo.plot(valores, normalizar(prior), label="Prior", linewidth=2)
        eixo.plot(valores, normalizar(likelihood), label="Likelihood", linewidth=2)
        eixo.plot(valores, normalizar(posterior), label="Posterior", linewidth=2)
        eixo.set_title(f"{ticker} - {nome}")
        eixo.set_xlabel(f"Valor de {nome}")
        eixo.set_ylabel("Densidade normalizada")
        eixo.legend()
        eixo.grid(alpha=0.3)

    nome_tabela = OUTPUT_DIR / f"{ticker}_posterior_summary.csv"
    resumo.to_csv(nome_tabela)
    print(f"Tabela salva em: {nome_tabela}")
    previsao = {
        "medio": fcl_medio,
        "p90": fcl_p90,
        "p10": fcl_p10,
    }
    return modelo, idata, previsao


modelos = {}
resultados = {}
previsoes_individuais = {}
if not USAR_EMPRESAS_MUTLIVARIADAS:
    figura_final, eixos_finais = plt.subplots(
        len(empresas), 4, figsize=(20, 5 * len(empresas)), squeeze=False
    )

    for indice, empresa in enumerate(empresas):
        modelo, idata, previsao = construir_modelo_bayesiano(
            empresa, eixos_finais[indice]
        )
        modelos[empresa] = modelo
        resultados[empresa] = idata
        previsoes_individuais[empresa] = previsao

    figura_final.suptitle(
        "FCL e distribuições Prior, Likelihood e Posterior", fontsize=16
    )
    figura_final.tight_layout()
    plot_name = FIG_DIR / "FCL_modelos_bayesianos.png"
    figura_final.savefig(plot_name, dpi=200, bbox_inches="tight")
    plt.close(figura_final)
    print(f"Figura única salva em: {plot_name}")


################################################################################333
#
# Modelo bayesiano multivariado para as três empresas


# Matriz: linhas = anos; colunas = empresas
dados_fcl = fcl_utilizado.loc[empresas_multivariadas].T

anos = dados_fcl.index.astype(float).to_numpy()
fcl_observado = dados_fcl.to_numpy(dtype=float)

# Padronização dos anos
anos_padronizados = (anos - anos.mean()) / anos.std()

numero_empresas = len(empresas_multivariadas)

with pm.Model() as modelo_multivariado:

    # Intercepto de cada empresa
    alpha = pm.Normal(
        "alpha",
        mu=fcl_observado.mean(axis=0),
        sigma=10,
        shape=numero_empresas,
    )

    # Inclinação da regressão de cada empresa
    beta = pm.Normal(
        "beta",
        mu=0,
        sigma=10,
        shape=numero_empresas,
    )

    # Liga ou desliga a correlação entre os resíduos das empresas.
    if USAR_EMPRESAS_MUTLIVARIADAS:
        chol, correlacao, desvios = pm.LKJCholeskyCov(
            "covariancia",
            n=numero_empresas,
            eta=2,
            sd_dist=pm.Exponential.dist(1),
            compute_corr=True,
        )
    else:
        desvios = pm.Exponential(
            "covariancia_stds",
            lam=1,
            shape=numero_empresas,
        )
        correlacao = pt.eye(numero_empresas)
        chol = pt.diag(desvios)

    # Média esperada do FCL para cada ano e empresa
    media_fcl = (
        alpha[None, :]
        + beta[None, :] * anos_padronizados[:, None]
    )

    # Likelihood multivariada
    pm.MvNormal(
        "FCL_observado",
        mu=media_fcl,
        chol=chol,
        observed=fcl_observado,
    )

    # Inferência bayesiana
    idata_multivariado = pm.sample(
        draws=1000,
        tune=1000,
        chains=2,
        cores=1,
        target_accept=0.90,
        random_seed=42,
        return_inferencedata=True,
    )


# Resumo dos parâmetros individuais
resumo_parametros = az.summary(
    idata_multivariado,
    var_names=[
        "alpha",
        "beta",
        "covariancia_stds",
    ],
)

print("\nResumo dos parâmetros:")
print(resumo_parametros)


# Resumo das correlações entre os resíduos das empresas
if USAR_EMPRESAS_MUTLIVARIADAS:
    resumo_correlacao = az.summary(
        idata_multivariado,
        var_names=["covariancia_corr"],
    )
else:
    resumo_correlacao = pd.DataFrame(
        np.eye(numero_empresas),
        index=empresas_multivariadas,
        columns=empresas_multivariadas,
    )

print("\nCorrelação posterior ou matriz usada:")
print(resumo_correlacao)


# Salva os resumos em arquivos CSV
resumo_parametros.to_csv(
    OUTPUT_DIR / "modelo_multivariado_parametros.csv"
)

resumo_correlacao.to_csv(
    OUTPUT_DIR / "modelo_multivariado_correlacoes.csv"
)


# Extrai as amostras posteriores dos parâmetros
alpha_amostras = (
    idata_multivariado.posterior["alpha"]
    .stack(amostra=("chain", "draw"))
    .values
)

beta_amostras = (
    idata_multivariado.posterior["beta"]
    .stack(amostra=("chain", "draw"))
    .values
)

desvios_amostras = (
    idata_multivariado.posterior["covariancia_stds"]
    .stack(amostra=("chain", "draw"))
    .values
)

if USAR_EMPRESAS_MUTLIVARIADAS:
    correlacao_amostras = (
        idata_multivariado.posterior["covariancia_corr"]
        .stack(amostra=("chain", "draw"))
        .values
    )
else:
    correlacao_amostras = np.broadcast_to(
        np.eye(numero_empresas)[:, :, None],
        (
            numero_empresas,
            numero_empresas,
            desvios_amostras.shape[1],
        ),
    ).copy()


# Matriz de correlação média posterior
correlacao_media = correlacao_amostras.mean(axis=2)

print("\nMatriz de correlação média posterior:")
print(
    pd.DataFrame(
        correlacao_media,
        index=empresas_multivariadas,
        columns=empresas_multivariadas,
    )
)


# Calcula a matriz de covariância para cada amostra posterior
covariancia_amostras = np.empty(
    (
        desvios_amostras.shape[1],
        numero_empresas,
        numero_empresas,
    )
)

for indice_amostra in range(desvios_amostras.shape[1]):
    matriz_desvios = np.diag(
        desvios_amostras[:, indice_amostra]
    )

    covariancia_amostras[indice_amostra] = (
        matriz_desvios
        @ correlacao_amostras[:, :, indice_amostra]
        @ matriz_desvios
    )


# Matriz de covariância média posterior
covariancia_media = covariancia_amostras.mean(axis=0)

print("\nMatriz de covariância média posterior:")
print(
    pd.DataFrame(
        covariancia_media,
        index=empresas_multivariadas,
        columns=empresas_multivariadas,
    )
)

pd.DataFrame(
    correlacao_media,
    index=empresas_multivariadas,
    columns=empresas_multivariadas,
).to_csv(
    OUTPUT_DIR / "matriz_correlacao_posterior.csv"
)

pd.DataFrame(
    covariancia_media,
    index=empresas_multivariadas,
    columns=empresas_multivariadas,
).to_csv(
    OUTPUT_DIR / "matriz_covariancia_posterior.csv"
)


# Gráfico da matriz de correlação posterior média
figura_correlacao, eixo_correlacao = plt.subplots(
    figsize=(7, 6)
)

imagem = eixo_correlacao.imshow(
    correlacao_media,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)

eixo_correlacao.set_xticks(range(numero_empresas))
eixo_correlacao.set_yticks(range(numero_empresas))
eixo_correlacao.set_xticklabels(empresas_multivariadas)
eixo_correlacao.set_yticklabels(empresas_multivariadas)

for linha in range(numero_empresas):
    for coluna in range(numero_empresas):
        eixo_correlacao.text(
            coluna,
            linha,
            f"{correlacao_media[linha, coluna]:.2f}",
            ha="center",
            va="center",
            color="black",
        )

eixo_correlacao.set_title(
    "Correlação posterior média entre os FCLs"
)

figura_correlacao.colorbar(
    imagem,
    ax=eixo_correlacao,
    label="Correlação",
)

figura_correlacao.tight_layout()

figura_correlacao.savefig(
    FIG_DIR / "matriz_correlacao_posterior.png",
    dpi=200,
)

plt.close(figura_correlacao)


# Figura do modelo multivariado, no mesmo formato da figura individual.
anos_projecao = np.arange(2016, 2037)
anos_projecao_padronizados = (anos_projecao - anos.mean()) / anos.std()
rng = np.random.default_rng(42)
previsoes_multivariadas = np.empty(
    (alpha_amostras.shape[1], len(anos_projecao), numero_empresas)
)

for indice_amostra in range(alpha_amostras.shape[1]):
    media_projecao = (
        alpha_amostras[:, indice_amostra][None, :]
        + beta_amostras[:, indice_amostra][None, :]
        * anos_projecao_padronizados[:, None]
    )
    erros_multivariados = rng.multivariate_normal(
        mean=np.zeros(numero_empresas),
        cov=covariancia_amostras[indice_amostra],
        size=len(anos_projecao),
    )
    previsoes_multivariadas[indice_amostra] = media_projecao + erros_multivariados

fcl_medio_multivariado = previsoes_multivariadas.mean(axis=0)
fcl_p90_multivariado = np.percentile(previsoes_multivariadas, 90, axis=0)
fcl_p10_multivariado = np.percentile(previsoes_multivariadas, 10, axis=0)
fcl_medio_multivariado_sem_futuro = fcl_medio_multivariado.copy()
fcl_p90_multivariado_sem_futuro = fcl_p90_multivariado.copy()
if USAR_FCL_FUTURO_PROJETADO:
    fcl_p90_multivariado += (
        fcl_futuro_projetado.reindex(
            index=empresas_multivariadas,
            columns=anos_projecao,
            fill_value=0,
        )
        .to_numpy(dtype=float)
        .T
    )
    fcl_medio_multivariado += (
        fcl_futuro_projetado.reindex(
            index=empresas_multivariadas,
            columns=anos_projecao,
            fill_value=0,
        )
        .to_numpy(dtype=float)
        .T
    )/2

figura_fcl, eixos_fcl = plt.subplots(
    len(empresas_multivariadas),
    1 if USAR_EMPRESAS_MUTLIVARIADAS else 2,
    figsize=(7 if USAR_EMPRESAS_MUTLIVARIADAS else 14, 4 * len(empresas_multivariadas)),
    sharex=True,
    squeeze=False,
)

for indice_empresa, empresa in enumerate(empresas_multivariadas):
    previsoes_por_modelo = []
    if not USAR_EMPRESAS_MUTLIVARIADAS:
        previsoes_por_modelo.append(
            ("Individual", previsoes_individuais[empresa])
        )
    previsoes_por_modelo.append(
        (
            "Multivariado" if USAR_EMPRESAS_MUTLIVARIADAS else "Independente",
            {
                "medio": fcl_medio_multivariado[:, indice_empresa],
                "p90": fcl_p90_multivariado[:, indice_empresa],
                "p10": fcl_p10_multivariado[:, indice_empresa],
            },
        )
    )

    for indice_modelo, (nome_modelo, previsao) in enumerate(
        previsoes_por_modelo
    ):
        eixo = eixos_fcl[indice_empresa, indice_modelo]
        eixo.scatter(
            anos,
            fcl_observado[:, indice_empresa],
            color="black",
            label="FCL observado",
        )
        eixo.plot(
            anos_projecao,
            previsao["medio"],
            color="blue",
            label="FCL médio",
        )
        eixo.plot(
            anos_projecao,
            previsao["p90"],
            color="green",
            linestyle="--",
            label="FCL P90",
        )
        eixo.plot(
            anos_projecao,
            previsao["p10"],
            color="red",
            linestyle="--",
            label="FCL P10",
        )
        eixo.set_xlim(ano_inicial, ano_final_plot)
        eixo.set_title(f"{empresa} - Modelo {nome_modelo.lower()}")
        eixo.set_xlabel("Ano")
        eixo.set_ylabel("Fluxo de caixa livre")
        eixo.legend()
        eixo.grid(alpha=0.3)

figura_fcl.suptitle(
    "Projeções de FCL: "
    + ("modelo multivariado" if USAR_EMPRESAS_MUTLIVARIADAS else "modelos individual e independente")
)
figura_fcl.tight_layout(rect=(0, 0, 1, 0.96))
nome_arquivo_fcl = (
    "FCL_graficos_multivariado.png"
    if USAR_EMPRESAS_MUTLIVARIADAS
    else "FCL_graficos_individual_e_independente.png"
)
nome_figura_fcl = FIG_DIR / nome_arquivo_fcl
figura_fcl.savefig(nome_figura_fcl, dpi=200, bbox_inches="tight")
plt.close(figura_fcl)
print(f"Figura somente com gráficos de FCL salva em: {nome_figura_fcl}")

figura_multivariada, eixos_multivariados = plt.subplots(
    numero_empresas,
    4,
    figsize=(20, 5 * numero_empresas),
    squeeze=False,
)

for indice_empresa, empresa in enumerate(empresas_multivariadas):
    eixo_fcl = eixos_multivariados[indice_empresa, 0]
    eixo_fcl.scatter(
        anos,
        fcl_observado[:, indice_empresa],
        color="black",
        label="FCL observado",
    )
    eixo_fcl.plot(
        anos_projecao,
        fcl_medio_multivariado[:, indice_empresa],
        color="blue",
        label="FCL médio",
    )
    eixo_fcl.plot(
        anos_projecao,
        fcl_p90_multivariado[:, indice_empresa],
        color="green",
        linestyle="--",
        label="FCL P90",
    )
    eixo_fcl.plot(
        anos_projecao,
        fcl_p10_multivariado[:, indice_empresa],
        color="red",
        linestyle="--",
        label="FCL P10",
    )
    if USAR_FCL_FUTURO_PROJETADO:
        eixo_fcl.plot(
            anos_projecao,
            fcl_medio_multivariado_sem_futuro[:, indice_empresa],
            color="black",
            linestyle=":",
            linewidth=0.8,
            label="FCL médio sem fluxo futuro",
        )
        eixo_fcl.plot(
            anos_projecao,
            fcl_p90_multivariado_sem_futuro[:, indice_empresa],
            color="black",
            linestyle=":",
            linewidth=0.8,
            label="FCL P90 sem fluxo futuro",
        )
    eixo_fcl.set_xlim(ano_inicial, ano_final_plot)
    tipo_modelo = (
        "multivariado" if USAR_EMPRESAS_MUTLIVARIADAS else "independente"
    )
    eixo_fcl.set_title(f"{empresa} - FCL {tipo_modelo} até {ano_final_plot}")
    eixo_fcl.set_xlabel("Ano")
    eixo_fcl.set_ylabel("Fluxo de caixa livre")
    eixo_fcl.legend()
    eixo_fcl.grid(alpha=0.3)

    parametros_empresa = [
        (
            alpha_amostras[indice_empresa],
            "alpha",
            norm.pdf(
                np.linspace(
                    alpha_amostras[indice_empresa].min(),
                    alpha_amostras[indice_empresa].max(),
                    500,
                ),
                fcl_observado[:, indice_empresa].mean(),
                10 * fcl_observado[:, indice_empresa].std(),
            ),
        ),
        (
            beta_amostras[indice_empresa],
            "beta",
            None,
        ),
        (
            desvios_amostras[indice_empresa],
            "desvio residual",
            None,
        ),
    ]

    for indice_parametro, (amostras, nome, prior) in enumerate(parametros_empresa, 1):
        eixo = eixos_multivariados[indice_empresa, indice_parametro]
        valores = np.linspace(amostras.min(), amostras.max(), 500)
        posterior = gaussian_kde(amostras)(valores)
        eixo.plot(valores, posterior / posterior.max(), label="Posterior", linewidth=2)

        if nome == "alpha":
            eixo.plot(valores, prior / prior.max(), label="Prior", linewidth=2)
        elif nome == "beta":
            beta_prior = norm.pdf(valores, 0, 10)
            eixo.plot(valores, beta_prior / beta_prior.max(), label="Prior", linewidth=2)
        else:
            desvio_prior = halfnorm.pdf(valores, scale=1)
            eixo.plot(
                valores,
                desvio_prior / desvio_prior.max(),
                label="Prior",
                linewidth=2,
            )

        eixo.set_title(f"{empresa} - {nome}")
        eixo.set_xlabel("Valor")
        eixo.set_ylabel("Densidade normalizada")
        eixo.legend()
        eixo.grid(alpha=0.3)

figura_multivariada.suptitle(
    "Modelo "
    + ("multivariado" if USAR_EMPRESAS_MUTLIVARIADAS else "independente")
    + ": FCL e distribuições posteriores",
    fontsize=16,
)
figura_multivariada.tight_layout()
nome_figura_multivariada = FIG_DIR / "FCL_modelos_multivaraible_bayesianos.png"
figura_multivariada.savefig(
    nome_figura_multivariada,
    dpi=200,
    bbox_inches="tight",
)
plt.close(figura_multivariada)
print(f"Figura multivariada salva em: {nome_figura_multivariada}")


#-----------------------------------------------------------------
#calculo de dividendos futuros com base no FCL projetado

#calculo de pay-out médio
anos_payout = pd.Index(range(2016, 2026), dtype=int)

proventos_anuais.index = proventos_anuais.index.str.strip()
proventos_anuais.columns = proventos_anuais.columns.astype(int)
proventos_anuais = proventos_anuais.apply(pd.to_numeric, errors="coerce")
proventos_por_acao = proventos_anuais.reindex(
    index=empresas,
    columns=anos_payout,
)

fcl_payout = fcl.reindex(index=empresas).copy()
fcl_payout.columns = fcl_payout.columns.astype(int)
fcl_payout = fcl_payout.reindex(columns=anos_payout)
fcl_payout = fcl_payout.where(fcl_payout > 0)

quantidades_acoes_payout = pd.Series(quantidades_acoes, dtype=float).reindex(empresas)
proventos_totais_estimados = proventos_por_acao.mul(
    quantidades_acoes_payout,
    axis="index",
) / 1_000_000_000

# Dividendo total pago (R$ bilhões, mesma unidade do FCL), preenchido à mão.
# Onde houver valor, ele substitui a estimativa provento/ação x ações atuais,
# evitando o erro causado por mudanças no número de ações ao longo dos anos.
proventos_totais_informados = pd.read_csv(
    BASE_DIR / "data_input" / "proventos_totais.csv",
    sep=r"\s+",
    decimal=".",
    index_col=0,
)
proventos_totais_informados.index = proventos_totais_informados.index.str.strip()
proventos_totais_informados.columns = proventos_totais_informados.columns.astype(int)
proventos_totais_informados = (
    proventos_totais_informados.apply(pd.to_numeric, errors="coerce")
    .reindex(index=empresas, columns=anos_payout)
)
proventos_totais_estimados = proventos_totais_informados.combine_first(
    proventos_totais_estimados
)
print("\nProventos totais informados (R$ bi; NaN = usa provento/ação x ações atuais):")
print(proventos_totais_informados)
payout_anual_percentual = proventos_totais_estimados.div(
    fcl_payout.replace(0, np.nan)
).mul(100)
payout_medio_por_empresa = payout_anual_percentual.mean(axis="columns")
payout_resumo_bayesiano = None

if USAR_PAYOUT_BAYESIANO:
    figura_payout, eixos_payout = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        squeeze=False,
    )
    resumos_posteriores = []
    densidades_posteriores = []

    for indice_empresa, empresa in enumerate(empresas):
        eixo = eixos_payout.flat[indice_empresa]
        observacoes_payout = (
            payout_anual_percentual.loc[empresa]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
        # A log-normal exige payout estritamente positivo (anos sem proventos saem).
        observacoes_payout = observacoes_payout[observacoes_payout > 0]

        if observacoes_payout.size < 2:
            payout_medio_por_empresa.loc[empresa] = np.nan
            eixo.set_title(f"{empresa} - dados insuficientes")
            eixo.set_axis_off()
            resumos_posteriores.append(
                {
                    "ticker": empresa,
                    "anos_observados": observacoes_payout.size,
                    "payout_medio_posterior_pct": np.nan,
                    "payout_mediano_posterior_pct": np.nan,
                    "payout_p10_posterior_pct": np.nan,
                    "payout_p50_posterior_pct": np.nan,
                    "payout_p90_posterior_pct": np.nan,
                    "payout_p05_posterior_pct": np.nan,
                    "payout_p95_posterior_pct": np.nan,
                    "payout_previsto_p10_pct": np.nan,
                    "payout_previsto_p50_pct": np.nan,
                    "payout_previsto_p90_pct": np.nan,
                }
            )
            continue

        # O payout é uma razão positiva e assimétrica: usa-se um modelo
        # log-normal, com prior fraca centrada em 50% (desvio de 1 em log).
        with pm.Model() as modelo_payout:
            payout_mu_log = pm.Normal("payout_mu_log", mu=np.log(50.0), sigma=1.0)
            payout_sigma_log = pm.HalfNormal("payout_sigma_log", sigma=1.0)
            pm.LogNormal(
                "payout_anual_observado",
                mu=payout_mu_log,
                sigma=payout_sigma_log,
                observed=observacoes_payout,
            )
            # Mediana do payout anual: payout central usado nos resultados.
            pm.Deterministic("payout_mediano", pt.exp(payout_mu_log))
            # Média da log-normal: só informativa, é puxada para cima por sigma.
            pm.Deterministic(
                "payout_medio",
                pt.exp(payout_mu_log + payout_sigma_log**2 / 2),
            )
            idata_payout = pm.sample(
                draws=2000,
                tune=1000,
                chains=2,
                cores=1,
                target_accept=0.93,
                random_seed=42 + indice_empresa,
                return_inferencedata=True,
                progressbar=False,
            )

        amostras_mediana_posterior = (
            idata_payout.posterior["payout_mediano"].values.flatten()
        )
        media_posterior = float(
            idata_payout.posterior["payout_medio"].values.mean()
        )
        payout_p10, payout_p50, payout_p90 = np.percentile(
            amostras_mediana_posterior,
            [10, 50, 90],
        )
        mediana_posterior = float(payout_p50)
        # Payout de um ano futuro: inclui a variação anual (sigma), não só a
        # incerteza sobre a média.
        mu_log_amostras = idata_payout.posterior["payout_mu_log"].values.flatten()
        sigma_log_amostras = idata_payout.posterior["payout_sigma_log"].values.flatten()
        payout_previsto = np.exp(
            np.random.default_rng(42 + indice_empresa).normal(
                mu_log_amostras, sigma_log_amostras
            )
        )
        previsto_p10, previsto_p50, previsto_p90 = np.percentile(
            payout_previsto, [10, 50, 90]
        )
        payout_p05, payout_p95 = np.percentile(
            amostras_mediana_posterior,
            [5, 95],
        )
        payout_medio_por_empresa.loc[empresa] = mediana_posterior

        grade_payout = np.linspace(
            np.percentile(amostras_mediana_posterior, 0.5),
            np.percentile(amostras_mediana_posterior, 99.5),
            500,
        )
        densidade_payout = gaussian_kde(amostras_mediana_posterior)(grade_payout)
        eixo.plot(grade_payout, densidade_payout, color="teal", linewidth=2)
        eixo.fill_between(
            grade_payout,
            0,
            densidade_payout,
            where=(grade_payout >= payout_p05) & (grade_payout <= payout_p95),
            color="teal",
            alpha=0.18,
            label="Intervalo de credibilidade 90%",
        )
        eixo.axvline(
            mediana_posterior,
            color="black",
            linestyle="--",
            label="Mediana posterior",
        )
        for percentil, valor, cor in (
            ("P10", payout_p10, "#d55e00"),
            ("P90", payout_p90, "#009e73"),
        ):
            eixo.axvline(
                valor,
                color=cor,
                linestyle=":",
                linewidth=1.8,
                label=f"{percentil}: {valor:.2f}%",
            )
        eixo.set_title(f"{empresa} - payout mediano posterior")
        eixo.set_xlabel("Payout (%)")
        eixo.set_ylabel("Densidade posterior")
        eixo.legend()
        eixo.grid(alpha=0.3)

        resumos_posteriores.append(
            {
                "ticker": empresa,
                "anos_observados": observacoes_payout.size,
                "payout_medio_posterior_pct": media_posterior,
                "payout_mediano_posterior_pct": mediana_posterior,
                "payout_p10_posterior_pct": float(payout_p10),
                "payout_p50_posterior_pct": float(payout_p50),
                "payout_p90_posterior_pct": float(payout_p90),
                "payout_p05_posterior_pct": float(payout_p05),
                "payout_p95_posterior_pct": float(payout_p95),
                "payout_previsto_p10_pct": float(previsto_p10),
                "payout_previsto_p50_pct": float(previsto_p50),
                "payout_previsto_p90_pct": float(previsto_p90),
            }
        )
        densidades_posteriores.append(
            pd.DataFrame(
                {
                    "ticker": empresa,
                    "payout_pct": grade_payout,
                    "densidade_posterior": densidade_payout,
                }
            )
        )

    figura_payout.suptitle(
        "Densidade posterior do payout mediano, 2016-2025",
        fontsize=14,
    )
    figura_payout.tight_layout(rect=(0, 0, 1, 0.96))
    arquivo_densidade_payout = FIG_DIR / "payout_densidade_posterior.png"
    figura_payout.savefig(
        arquivo_densidade_payout,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figura_payout)

    payout_resumo_bayesiano = pd.DataFrame(resumos_posteriores).set_index("ticker")
    payout_resumo_bayesiano.to_csv(
        OUTPUT_DIR / "payout_resumo_bayesiano.csv",
        decimal=",",
        float_format="%.2f",
    )
    if densidades_posteriores:
        pd.concat(densidades_posteriores, ignore_index=True).to_csv(
            OUTPUT_DIR / "payout_densidade_posterior.csv",
            index=False,
            decimal=",",
            float_format="%.6f",
        )
    print(f"Densidade posterior do payout salva em: {arquivo_densidade_payout}")

payout_medio_itub3 = payout_medio_por_empresa.get("ITUB3.SA", np.nan)
payout_medio_fesa4 = payout_medio_por_empresa.get("FESA4.SA", np.nan)
payout_medio_egie3 = payout_medio_por_empresa.get("EGIE3.SA", np.nan)
payout_medio_vale3 = payout_medio_por_empresa.get("VALE3.SA", np.nan)

relatorio_payout = payout_anual_percentual.copy()
relatorio_payout.columns = [f"payout_{ano}_pct" for ano in anos_payout]
relatorio_payout["payout_central_2016_2025_pct"] = payout_medio_por_empresa
if payout_resumo_bayesiano is not None:
    relatorio_payout = relatorio_payout.join(
        payout_resumo_bayesiano.drop(columns="anos_observados")
    )
relatorio_payout.index.name = "ticker"

arquivo_payout = OUTPUT_DIR / "payout_estimado_2016_2025.csv"
relatorio_payout.to_csv(arquivo_payout, decimal=",", float_format="%.2f")

metodo_payout = (
    "mediana posterior Bayesiana"
    if USAR_PAYOUT_BAYESIANO
    else "média aritmética"
)
print(
    f"Payout estimado pela {metodo_payout}: dividendo total informado em "
    "proventos_totais.csv ou, onde vazio, proventos por ação x quantidade "
    "atual de ações, dividido pelo FCL nominal."
)
print(relatorio_payout)
print(f"Relatório de payout salvo em: {arquivo_payout}")
