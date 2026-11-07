import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytensor.tensor as pt
import pandas as pd
import pymc as pm
import yfinance as yf
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde, halfnorm, norm
from datetime import date
from urllib.parse import urlencode
from urllib.request import urlopen
import json
import time
from ler_carteira import (
    buscar_cotacoes_yahoo,
    gerar_grafico_legacy,
    ler_carteira_excel,
    plotar_pizza_carteira,
)

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "fig" / "FCL"
OUTPUT_DIR = BASE_DIR / "data_output"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def criar_anos_projecao(ano_inicial, ano_final):
    """Cria a sequência inclusiva de anos definida nas opções do programa."""
    return np.arange(ano_inicial, ano_final + 1)


ano_inicial = 2016
ano_final = 2030
ano_final_plot = 2030
TAXA = 0.06 #taxa de juros livre de riscos (real, acima da inflação)
INFLACAO = 0.04 #inflação anual esperada de longo prazo
PROB_ALVO = 0.8 #probabilidade desejada de o VP superar o limiar (define a cotação máxima)
FRACAO_COTACAO = 0.7 #limiar do VP: fração da cotação atual (1.0 = a própria cotação)
anos_projecao = criar_anos_projecao(ano_inicial, ano_final)
# Janela do gráfico volatilidade x lucro: [VarMin, VarMax, LucroMin, LucroMax].
# Eixo x = desvio padrão do VP (R$ mil); eixo y = lucro esperado, mediana do VP (R$).
varianciaXlucro = [300, 5000, 0, 1200000]
# Número de carteiras aleatórias sorteadas nos gráficos volatilidade/variância x lucro.
NUMERO_CARTEIRAS_ALEATORIAS = 50000
# Volatilidade (desvio padrão do VP, mesma unidade do eixo x de varianciaXlucro, R$ mil) em que se
# seleciona, sobre o envelope das carteiras, a carteira de maior lucro esperado.
VARIANCIA_ALVO_ENVELOPE = 1500
USAR_FCL_CORRIGIDO_capex_expandido = True
USAR_FCL_FUTURO_PROJETADO = True
# True: as projeções do arquivo estão em valores nominais; são deflacionadas
# pela INFLACAO até os preços do ano-base do FCL (False: já estão em reais de hoje)
DEFLACIONAR_FCL_FUTURO_PROJETADO = True
USAR_FCL_CORRIGIDO_INPC = True
USAR_EMPRESAS_MUTLIVARIADAS = True
USAR_PAYOUT_BAYESIANO = True
# True: payout independente do FCL; False: log(payout) depende do log(FCL).
USAR_PAYOUT_INDEPENDENTE_FCL = False
# Covariância usada no risco do portfólio:
# "FCL": covariância residual do FCL (covariancia_media), convertida para dividendos;
# "DIVIDENDOS": covariância dos dividendos simulados, ano a ano.
COR_FCL_ou_DIVIDENDOS = "DIVIDENDOS"
TICKERS_YAHOO_ACOES = {
    "ITUB3.SA": ("ITUB3.SA", "ITUB4.SA"),
    "FESA4.SA": ("FESA3.SA", "FESA4.SA"),
    "EGIE3.SA": ("EGIE3.SA",),
    "VALE3.SA": ("VALE3.SA",),
    "BPAC11.SA": ("BPAC11.SA",),
    "AXIA3.SA": ("AXIA3.SA",),
    "BRBI11.SA": ("BRBI11.SA",),
    "AURA33.SA": ("AURA33.SA",),
    "RANI3.SA": ("RANI3.SA",),
    "LEVE3.SA": ("LEVE3.SA",),
}
# Quantidades usadas quando o Yahoo Finance não informa sharesOutstanding.
QUANTIDADE_ACOES_MANUAL = {
    "BRBI11.SA": 314_987_000,
}
# Lista única de ativos: define as empresas de toda a análise.
empresas = list(TICKERS_YAHOO_ACOES)
empresas_multivariadas = empresas
def buscar_quantidade_acoes_yahoo(tickers):
    """Consulta no Yahoo Finance as quantidades de ações dos tickers."""
    quantidades = {}
    registros = []

    # Última consulta bem-sucedida, usada se o Yahoo falhar (instabilidade/limite de requisições).
    arquivo_cache = OUTPUT_DIR / "quantidade_acoes_yahoo.csv"
    cache = {}
    if arquivo_cache.exists():
        anterior = pd.read_csv(arquivo_cache).dropna(subset=["quantidade_acoes"])
        cache = dict(zip(anterior["ticker"], anterior["quantidade_acoes"].astype(int)))

    for ticker in tickers:
        tickers_yahoo = TICKERS_YAHOO_ACOES.get(ticker, (ticker,))

        try:
            quantidades_por_classe = []
            for ticker_yahoo in tickers_yahoo:
                quantidade = None
                for tentativa in range(3):
                    try:
                        quantidade = yf.Ticker(ticker_yahoo).get_info().get("sharesOutstanding")
                    except Exception:
                        quantidade = None
                    if quantidade is not None:
                        break
                    time.sleep(2 * (tentativa + 1))
                if quantidade is None:
                    raise ValueError(
                        f"Yahoo Finance não retornou sharesOutstanding para {ticker_yahoo}"
                    )
                quantidades_por_classe.append(int(quantidade))
            quantidades[ticker] = sum(quantidades_por_classe)
        except Exception as erro:
            quantidades[ticker] = QUANTIDADE_ACOES_MANUAL.get(ticker, cache.get(ticker))
            origem = (
                "usando a quantidade manual/da última consulta"
                if quantidades[ticker] is not None
                else "sem quantidade manual cadastrada"
            )
            print(
                f"Não foi possível obter a quantidade de ações de {ticker} "
                f"usando {', '.join(tickers_yahoo)}: {erro}; {origem}."
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
sem_quantidade = [t for t, q in quantidades_acoes.items() if q is None]
if sem_quantidade:
    raise RuntimeError(
        f"Sem quantidade de ações para {sem_quantidade}: o Yahoo falhou. Rode de novo "
        "ou cadastre em QUANTIDADE_ACOES_MANUAL."
    )
quantidade_acoes_itub3 = quantidades_acoes["ITUB3.SA"]
quantidade_acoes_fesa4 = quantidades_acoes["FESA4.SA"]
quantidade_acoes_egie3 = quantidades_acoes["EGIE3.SA"]
quantidade_acoes_vale3 = quantidades_acoes["VALE3.SA"]
quantidade_acoes_bpac11 = quantidades_acoes["BPAC11.SA"]
quantidade_acoes_axia3 = quantidades_acoes["AXIA3.SA"]
quantidade_acoes_brbi11 = quantidades_acoes["BRBI11.SA"]
quantidade_acoes_aura33 = quantidades_acoes["AURA33.SA"]
quantidade_acoes_rani3 = quantidades_acoes["RANI3.SA"]
quantidade_acoes_leve3 = quantidades_acoes["LEVE3.SA"]

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


def ler_dados_entrada(diretorio_dados):
    """Lê os CSVs usados na análise e padroniza os nomes de anos e tickers.

    Retorna FCL, capex de expansão, FCL projetado e proventos por ação,
    mantendo a ordem esperada pelo restante do programa.
    """
    opcoes_csv = {
        "sep": "\t",
        "index_col": 0,
    }
    def ler_tabela_numerica(nome_arquivo):
        # Aceita vírgula ou ponto como separador decimal, mesmo misturados.
        tabela = pd.read_csv(diretorio_dados / nome_arquivo, dtype=str, **opcoes_csv)
        return tabela.apply(
            lambda coluna: pd.to_numeric(
                coluna.str.strip().str.replace(",", ".", regex=False),
                errors="coerce",
            )
        )

    fcl = ler_tabela_numerica("FCL.csv")
    capex_expansao = ler_tabela_numerica("Capex_Expancao.csv")
    fcl_futuro_projetado = ler_tabela_numerica("FCL_futuro_projetado.csv")
    proventos_anuais = ler_tabela_numerica("proventos.csv")

    for tabela in (fcl, capex_expansao, proventos_anuais):
        tabela.index = tabela.index.str.strip()

    fcl.columns = fcl.columns.str.strip()
    capex_expansao.columns = capex_expansao.columns.str.strip()
    fcl_futuro_projetado.columns = fcl_futuro_projetado.columns.str.strip()
    fcl_futuro_projetado.index = fcl_futuro_projetado.index.str.strip()
    fcl_futuro_projetado.columns = (
        fcl_futuro_projetado.columns.str.extract(r"(\d{4})", expand=False).astype(int)
    )
    proventos_anuais.columns = proventos_anuais.columns.str.strip()

    return fcl, capex_expansao, fcl_futuro_projetado, proventos_anuais


def preparar_dados_fcl(fcl, capex_expansao, inpc_anual, usar_capex, diretorio_saida):
    """Aplica o ajuste de capex e calcula o FCL histórico corrigido pelo INPC.

    Retorna o FCL após o ajuste opcional de capex, os fatores de correção,
    o ano-base de preços e o FCL corrigido.
    """
    if usar_capex:
        fcl = fcl.add(capex_expansao, fill_value=0)

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
        diretorio_saida / "FCL_corrigido_INPC.csv",
        sep="\t",
        decimal=",",
    )
    return fcl, fatores_correcao_inpc, ano_base_fcl, fcl_corrigido


def mascarar_fcl_antes_do_primeiro_valor(fcl_por_ano):
    """Troca por NaN os valores anteriores ao primeiro FCL diferente de zero.

    Aceita Series ou DataFrame (anos nas linhas). Zeros posteriores ao primeiro
    valor não nulo continuam como dados. Séries inteiramente nulas ficam intactas.
    """
    nao_nulo = fcl_por_ano.ne(0) & fcl_por_ano.notna()
    ja_iniciou = nao_nulo.astype(int).cummax().astype(bool) | ~nao_nulo.any()
    return fcl_por_ano.where(ja_iniciou)


def ajustar_modelo_multivariado(dados_fcl, usar_correlacao):
    """Ajusta o modelo bayesiano conjunto aos FCLs das empresas.

    Cada empresa tem intercepto e tendência próprios. A opção
    `usar_correlacao` controla se os resíduos compartilham correlações.
    Retorna os resultados MCMC e os arrays usados nas previsões.
    """
    anos = dados_fcl.index.astype(float).to_numpy()
    fcl_observado = dados_fcl.to_numpy(dtype=float)
    mascara_observado = ~np.isnan(fcl_observado)
    anos_padronizados = (anos - anos.mean()) / anos.std()
    numero_empresas = dados_fcl.shape[1]

    with pm.Model() as modelo:
        alpha = pm.Normal(
            "alpha",
            mu=np.nanmean(fcl_observado, axis=0),
            sigma=10,
            shape=numero_empresas,
        )
        beta = pm.Normal(
            "beta",
            mu=0,
            sigma=10,
            shape=numero_empresas,
        )

        if usar_correlacao:
            chol, _, _ = pm.LKJCholeskyCov(
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
            chol = pt.diag(desvios)

        media_fcl = (
            alpha[None, :]
            + beta[None, :] * anos_padronizados[:, None]
        )
        covariancia = chol @ chol.T
        # Cada padrão de dados faltantes (anos antes do primeiro FCL não nulo)
        # usa a marginal da normal multivariada sobre as empresas observadas.
        padroes = {}
        for indice_ano, padrao in enumerate(mascara_observado):
            padroes.setdefault(tuple(padrao), []).append(indice_ano)
        for numero_padrao, (padrao, linhas) in enumerate(padroes.items()):
            colunas = np.flatnonzero(padrao)
            if colunas.size == 0:
                continue
            marginal = pm.MvNormal.dist(
                mu=media_fcl[linhas][:, colunas],
                cov=covariancia[colunas][:, colunas],
            )
            pm.Potential(
                f"FCL_observado_{numero_padrao}",
                pm.logp(marginal, fcl_observado[np.ix_(linhas, colunas)]),
            )
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=1,
            target_accept=0.90,
            random_seed=42,
            return_inferencedata=True,
        )

    return modelo, idata, anos, fcl_observado


def extrair_amostras_covariancia(idata, usar_correlacao, numero_empresas):
    """Extrai draws de parâmetros e monta covariância para cada draw posterior."""
    alpha_amostras = (
        idata.posterior["alpha"].stack(amostra=("chain", "draw")).values
    )
    beta_amostras = (
        idata.posterior["beta"].stack(amostra=("chain", "draw")).values
    )
    desvios_amostras = (
        idata.posterior["covariancia_stds"]
        .stack(amostra=("chain", "draw"))
        .values
    )

    if usar_correlacao:
        correlacao_amostras = (
            idata.posterior["covariancia_corr"]
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

    covariancia_amostras = np.empty(
        (desvios_amostras.shape[1], numero_empresas, numero_empresas)
    )
    for indice_amostra in range(desvios_amostras.shape[1]):
        matriz_desvios = np.diag(desvios_amostras[:, indice_amostra])
        correlacao = correlacao_amostras[:, :, indice_amostra]
        covariancia_amostras[indice_amostra] = (
            matriz_desvios @ correlacao @ matriz_desvios
        )

    return (
        alpha_amostras,
        beta_amostras,
        desvios_amostras,
        correlacao_amostras,
        covariancia_amostras,
    )


def simular_previsoes_multivariadas(
    anos_historicos,
    anos_projecao,
    alpha_amostras,
    beta_amostras,
    covariancia_amostras,
    random_seed=42,
):
    """Simula FCL conjunto por draw posterior e ano de projeção.

    A dimensão da saída é (draws, anos, empresas), preservando a correlação
    entre empresas em cada amostra.
    """
    anos_padronizados = (
        anos_projecao - anos_historicos.mean()
    ) / anos_historicos.std()
    numero_empresas = alpha_amostras.shape[0]
    previsoes = np.empty(
        (alpha_amostras.shape[1], len(anos_projecao), numero_empresas)
    )
    rng = np.random.default_rng(random_seed)

    for indice_amostra in range(alpha_amostras.shape[1]):
        media_projecao = (
            alpha_amostras[:, indice_amostra][None, :]
            + beta_amostras[:, indice_amostra][None, :]
            * anos_padronizados[:, None]
        )
        erros = rng.multivariate_normal(
            mean=np.zeros(numero_empresas),
            cov=covariancia_amostras[indice_amostra],
            size=len(anos_projecao),
        )
        previsoes[indice_amostra] = media_projecao + erros

    return previsoes


def ajustar_modelo_payout(observacoes, random_seed):
    """Ajusta a distribuição log-normal do payout anual positivo."""
    with pm.Model() as modelo:
        payout_mu_log = pm.Normal("payout_mu_log", mu=np.log(50.0), sigma=1.0)
        payout_sigma_log = pm.HalfNormal("payout_sigma_log", sigma=1.0)
        pm.LogNormal(
            "payout_anual_observado",
            mu=payout_mu_log,
            sigma=payout_sigma_log,
            observed=observacoes,
        )
        pm.Deterministic("payout_mediano", pt.exp(payout_mu_log))
        pm.Deterministic(
            "payout_medio",
            pt.exp(payout_mu_log + payout_sigma_log**2 / 2),
        )
        idata = pm.sample(
            draws=2000,
            tune=1000,
            chains=2,
            cores=1,
            target_accept=0.93,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False,
        )
    return idata


def ajustar_modelo_payout_dependente(tabela, indice_empresa, centro_log_fcl):
    """Ajusta payout log-normal cujo centro depende do log do FCL.

    O parâmetro gamma mede a associação: gamma negativo corresponde a payout
    relativamente maior quando o FCL observado é menor.
    """
    log_fcl = np.log(tabela["fcl"].to_numpy(dtype=float))
    with pm.Model() as modelo:
        dep_mu = pm.Normal("dep_mu", mu=np.log(50.0), sigma=1.0)
        dep_gamma = pm.Normal("dep_gamma", mu=0.0, sigma=0.5)
        dep_sigma = pm.HalfNormal("dep_sigma", sigma=1.0)
        pm.LogNormal(
            "payout_dep_observado",
            mu=dep_mu + dep_gamma * (log_fcl - centro_log_fcl),
            sigma=dep_sigma,
            observed=tabela["payout"].to_numpy(dtype=float),
        )
        idata = pm.sample(
            draws=2000,
            tune=1000,
            chains=2,
            cores=1,
            target_accept=0.93,
            random_seed=142 + indice_empresa,
            return_inferencedata=True,
            progressbar=False,
        )
    return idata


def simular_dividendos_futuros(
    anos_projecao,
    ano_final,
    previsoes_fcl,
    empresas,
    parametros_independentes,
    parametros_dependentes,
    payout_independente,
    random_seed=2026,
):
    """Combina draws futuros de FCL e payout para simular dividendos.

    FCL negativo é limitado a zero para não produzir dividendos negativos.
    Retorna os anos incluídos e uma matriz (draws, anos, empresas).
    """
    mascara_anos = anos_projecao <= ano_final
    anos_dividendos = anos_projecao[mascara_anos]
    fcl_amostras = previsoes_fcl[:, mascara_anos, :]
    rng = np.random.default_rng(random_seed)
    dividendos_amostras = np.full(fcl_amostras.shape, np.nan)

    for indice_empresa, empresa in enumerate(empresas):
        if payout_independente:
            if empresa not in parametros_independentes:
                continue
            mu_amostras, sigma_amostras = parametros_independentes[empresa]
        else:
            if empresa not in parametros_dependentes:
                continue
            mu_dep, gamma_dep, sigma_dep, centro_log_fcl = (
                parametros_dependentes[empresa]
            )

        fcl_empresa = np.clip(fcl_amostras[:, :, indice_empresa], 0, None)
        formato = fcl_empresa.shape
        numero_amostras = fcl_empresa.shape[0]
        if payout_independente:
            sorteio = rng.integers(0, mu_amostras.size, numero_amostras)
            mu_payout = np.broadcast_to(mu_amostras[sorteio][:, None], formato)
            sigma_payout = np.broadcast_to(
                sigma_amostras[sorteio][:, None], formato
            )
        else:
            sorteio = rng.integers(0, mu_dep.size, numero_amostras)
            log_fcl_futuro = np.log(np.where(fcl_empresa > 0, fcl_empresa, 1.0))
            mu_payout = (
                mu_dep[sorteio][:, None]
                + gamma_dep[sorteio][:, None]
                * (log_fcl_futuro - centro_log_fcl)
            )
            sigma_payout = np.broadcast_to(
                sigma_dep[sorteio][:, None], formato
            )

        payout_futuro = pm.draw(
            pm.LogNormal.dist(mu=mu_payout, sigma=sigma_payout),
            random_seed=int(rng.integers(1_000_000)),
        )
        dividendos_amostras[:, :, indice_empresa] = payout_futuro / 100 * fcl_empresa

    return anos_dividendos, dividendos_amostras


def resumir_dividendos(anos_dividendos, dividendos_amostras, empresas, quantidades):
    """Resume os draws de dividendos por empresa e ano em R$ bi e por ação."""
    linhas = []
    quantidades_por_empresa = pd.Series(quantidades, dtype=float)
    for indice_empresa, empresa in enumerate(empresas):
        for indice_ano, ano in enumerate(anos_dividendos):
            amostras = dividendos_amostras[:, indice_ano, indice_empresa]
            if np.isnan(amostras).all():
                continue
            p10, p50, p90 = np.percentile(amostras, [10, 50, 90])
            acoes_bilhoes = quantidades_por_empresa[empresa] / 1e9
            linhas.append(
                {
                    "ticker": empresa,
                    "ano": int(ano),
                    "dividendo_media_rs_bi": amostras.mean(),
                    "dividendo_p10_rs_bi": p10,
                    "dividendo_p50_rs_bi": p50,
                    "dividendo_p90_rs_bi": p90,
                    "dividendo_por_acao_p10": p10 / acoes_bilhoes,
                    "dividendo_por_acao_p50": p50 / acoes_bilhoes,
                    "dividendo_por_acao_p90": p90 / acoes_bilhoes,
                }
            )
    return pd.DataFrame(linhas)


def calcular_vp_amostras(anos_dividendos, dividendos_amostras, taxa, ano_base, inflacao):
    """Valor presente (R$ bi) de cada draw de dividendos, por empresa.

    Os dividendos estão a preços do ano_base (reais). Eles são inflacionados
    por (1 + inflacao)^(t - ano_base) e descontados pela taxa nominal
    (1 + taxa)(1 + inflacao) - 1, com `taxa` real. Retorna uma matriz
    (draws, empresas) e a taxa nominal.
    """
    anos = np.asarray(anos_dividendos)
    futuro = anos > ano_base
    taxa_nominal = (1 + taxa) * (1 + inflacao) - 1
    expoente = anos[futuro] - ano_base
    fatores = ((1 + inflacao) / (1 + taxa_nominal)) ** expoente
    vp_amostras = np.nansum(
        dividendos_amostras[:, futuro, :] * fatores[None, :, None], axis=1
    )
    return vp_amostras, taxa_nominal


def valor_presente_dividendos(
    anos_dividendos,
    dividendos_amostras,
    empresas,
    quantidades,
    taxa,
    ano_base,
    inflacao,
    cotacoes,
    fracao_cotacao,
    prob_alvo,
):
    """Desconta os dividendos futuros (anos > ano_base) a valor presente.

    Os dividendos estão a preços do ano_base (reais). Eles são inflacionados
    por (1 + inflacao)^(t - ano_base) e descontados pela taxa nominal
    (1 + taxa)(1 + inflacao) - 1, com `taxa` real. A inflação se cancela:
    o resultado é o mesmo que descontar o fluxo real pela taxa real.
    O VP é calculado em cada draw da posterior; assim tem média, P10, P50 e
    P90. Inclui a linha "CARTEIRA" com a soma das empresas (draws somados
    antes do resumo).
    """
    anos = np.asarray(anos_dividendos)
    futuro = anos > ano_base
    vp_amostras, taxa_nominal = calcular_vp_amostras(
        anos_dividendos, dividendos_amostras, taxa, ano_base, inflacao
    )
    com_dados = ~np.isnan(dividendos_amostras).all(axis=(0, 1))
    quantidades_por_empresa = pd.Series(quantidades, dtype=float)

    linhas = []
    series = [(e, vp_amostras[:, i], quantidades_por_empresa[e] / 1e9)
              for i, e in enumerate(empresas) if com_dados[i]]
    series.append(("CARTEIRA", vp_amostras[:, com_dados].sum(axis=1), np.nan))
    for nome, amostras, acoes_bilhoes in series:
        p10, p50, p90 = np.percentile(amostras, [10, 50, 90])
        cotacao = cotacoes.get(nome, np.nan)
        linhas.append(
            {
                "ticker": nome,
                "taxa_real": taxa,
                "inflacao": inflacao,
                "taxa_nominal": taxa_nominal,
                "ano_base": ano_base,
                "ano_final": int(anos[futuro].max()),
                "vp_media_rs_bi": amostras.mean(),
                "vp_p10_rs_bi": p10,
                "vp_p50_rs_bi": p50,
                "vp_p90_rs_bi": p90,
                "vp_por_acao_p10": p10 / acoes_bilhoes,
                "vp_por_acao_p50": p50 / acoes_bilhoes,
                "vp_por_acao_p90": p90 / acoes_bilhoes,
                "cotacao": cotacao,
                "vp_sobre_cotacao_p50": p50 / acoes_bilhoes / cotacao,
                "fracao_cotacao": fracao_cotacao,
                "prob_alvo": prob_alvo,
                "cotacao_maxima": (
                    np.percentile(amostras, 100 * (1 - prob_alvo))
                    / acoes_bilhoes
                    / fracao_cotacao
                ),
                "prob_vp_maior_fracao": (
                    np.mean(amostras / acoes_bilhoes > fracao_cotacao * cotacao)
                    if np.isfinite(cotacao)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(linhas)


def comparar_com_carteira(vp_amostras, empresas, quantidades, cotacoes, carteira):
    """Compara o VP por ação do modelo com a sua carteira real (planilha).

    Só entram os ativos que estão no modelo. Para cada um calcula: VP por
    ação (P50), VP/preço médio, probabilidade de o VP superar o preço médio
    pago, valor investido e VP da posição (VP por ação × suas ações, em R$).
    """
    posicao = carteira.set_index("ativo")
    linhas = []
    for i, empresa in enumerate(empresas):
        ativo = empresa.replace(".SA", "")
        if ativo not in posicao.index or not np.any(vp_amostras[:, i]):
            continue
        dados = posicao.loc[ativo]
        quantidade = float(dados["quantidade"])
        preco_medio = float(dados["preco_medio_compra"])
        vp_acao = vp_amostras[:, i] / (quantidades[empresa] / 1e9)
        p10, p50, p90 = np.percentile(vp_acao, [10, 50, 90])
        tem_posicao = quantidade > 0 and preco_medio > 0
        linhas.append(
            {
                "ativo": ativo,
                "quantidade": quantidade,
                "preco_medio": preco_medio,
                "cotacao": cotacoes.get(empresa, np.nan),
                "vp_acao_p10": p10,
                "vp_acao_p50": p50,
                "vp_acao_p90": p90,
                "vp_sobre_preco_medio": p50 / preco_medio if tem_posicao else np.nan,
                "prob_vp_maior_preco_medio": (
                    np.mean(vp_acao > preco_medio) if tem_posicao else np.nan
                ),
                "investido_rs": quantidade * preco_medio,
                "vp_posicao_p50_rs": quantidade * p50,
                "proventos_recebidos_rs": float(dados["proventos_recebidos"]),
            }
        )
    return pd.DataFrame(linhas)


def dividendos_anuais_carteira(
    anos_dividendos, dividendos_amostras, empresas, quantidades, carteira, ano_base
):
    """Tabela de dividendos esperados por ativo (linhas) e ano (colunas), em R$.

    Valor = mediana do dividendo por ação × quantidade de ações da sua carteira.
    Entram o `ano_base` (último ano observado) e os seguintes, e só ativos que estão no modelo. A última
    linha, TOTAL, soma os ativos.
    """
    posicao = carteira.set_index("ativo")["quantidade"]
    anos = np.asarray(anos_dividendos)
    indices_anos = np.where(anos >= ano_base)[0]
    linhas = {}
    for i, empresa in enumerate(empresas):
        ativo = empresa.replace(".SA", "")
        if ativo not in posicao.index:
            print(f"Aviso: {ativo} não está na planilha da carteira; ignorado na tabela.")
            continue
        if np.isnan(dividendos_amostras[:, indices_anos, i]).all():
            print(f"Aviso: {ativo} sem dados no modelo (falta histórico em data_input); ignorado.")
            continue
        por_acao = (
            np.nanmedian(dividendos_amostras[:, indices_anos, i], axis=0)
            / (quantidades[empresa] / 1e9)
        )
        linhas[ativo] = por_acao * float(posicao[ativo])
    tabela = pd.DataFrame(linhas, index=anos[indices_anos].astype(int)).T
    tabela.index.name = "ativo"
    tabela.loc["TOTAL"] = tabela.sum()
    return tabela


def lucro_medio_portfolio(quantidades, dividendos_medios):
    """Dividendo médio esperado do portfólio: soma de quantidade × dividendo médio.

    `quantidades` são as ações de cada ativo e `dividendos_medios` o dividendo
    médio por ação (mesma ordem e mesmo período). O resultado está na unidade
    do dividendo médio (por exemplo, R$ no ano).
    """
    quantidades = np.asarray(quantidades, dtype=float)
    dividendos_medios = np.asarray(dividendos_medios, dtype=float)
    if quantidades.shape != dividendos_medios.shape or quantidades.ndim != 1:
        raise ValueError("quantidades e dividendos_medios devem ser vetores do mesmo tamanho.")
    return float(quantidades @ dividendos_medios)


def variancia_portfolio(quantidades, covariancia):
    """Variância do dividendo do portfólio: q' Σ q.

    `covariancia` é a matriz de covariância dos dividendos por ação dos ativos
    (mesma ordem de `quantidades`), em unidade². O resultado está em unidade²;
    o desvio padrão do portfólio é a raiz quadrada dele.
    """
    quantidades = np.asarray(quantidades, dtype=float)
    covariancia = np.asarray(covariancia, dtype=float)
    numero_ativos = quantidades.size
    if quantidades.ndim != 1 or covariancia.shape != (numero_ativos, numero_ativos):
        raise ValueError("covariancia deve ser uma matriz quadrada com um lado por ativo.")
    return float(quantidades @ covariancia @ quantidades)


def resumo_portfolio_por_ano(
    anos_dividendos,
    dividendos_amostras,
    fcl_amostras,
    covariancia_fcl,
    empresas,
    quantidades,
    carteira,
    ano_base,
    fonte_covariancia,
):
    """Dividendo médio e risco da carteira por ano (R$), com a covariância escolhida.

    fonte_covariancia = "DIVIDENDOS": Σ é a covariância, entre os draws, dos
    dividendos por ação de cada ano. "FCL": Σ é `covariancia_fcl` (R$ bi²)
    convertida para dividendo por ação com o payout efetivo do ano
    (dividendo médio ÷ FCL médio positivo), isto é, Σ_div = D Σ_FCL D.
    A média sempre vem dos dividendos simulados.
    """
    fonte = str(fonte_covariancia).upper()
    if fonte not in ("FCL", "DIVIDENDOS"):
        raise ValueError('COR_FCL_ou_DIVIDENDOS deve ser "FCL" ou "DIVIDENDOS".')

    posicao = carteira.set_index("ativo")["quantidade"]
    anos = np.asarray(anos_dividendos)
    indices_ativos = []
    for i, empresa in enumerate(empresas):
        ativo = empresa.replace(".SA", "")
        if ativo in posicao.index and not np.isnan(dividendos_amostras[:, :, i]).all():
            indices_ativos.append(i)
    if not indices_ativos:
        raise ValueError("Nenhum ativo da carteira tem dividendos simulados.")

    q = np.array([float(posicao[empresas[i].replace(".SA", "")]) for i in indices_ativos])
    acoes_bi = np.array([quantidades[empresas[i]] / 1e9 for i in indices_ativos])
    por_acao = dividendos_amostras[:, :, indices_ativos] / acoes_bi
    sigma_fcl = np.asarray(covariancia_fcl, dtype=float)[np.ix_(indices_ativos, indices_ativos)]

    linhas = {}
    for indice_ano in np.where(anos >= ano_base)[0]:
        draws = por_acao[:, indice_ano, :]
        media = np.nanmean(draws, axis=0)
        total_draws = np.nansum(draws * q, axis=1)
        if fonte == "DIVIDENDOS":
            covariancia = np.atleast_2d(np.cov(draws, rowvar=False))
        else:
            fcl_medio = np.clip(fcl_amostras[:, indice_ano, indices_ativos], 0, None).mean(axis=0)
            dividendo_medio = np.nanmean(dividendos_amostras[:, indice_ano, indices_ativos], axis=0)
            payout = np.divide(
                dividendo_medio, fcl_medio, out=np.zeros_like(fcl_medio), where=fcl_medio > 0
            )
            escala = payout / acoes_bi
            covariancia = escala[:, None] * sigma_fcl * escala[None, :]
        variancia = variancia_portfolio(q, covariancia)
        linhas[int(anos[indice_ano])] = {
            "dividendo_medio_rs": lucro_medio_portfolio(q, media),
            "variancia_rs2": variancia,
            "desvio_padrao_rs": float(np.sqrt(variancia)),
            "dividendo_mediano_rs": float(np.median(total_draws)),
            "p10_rs": float(np.percentile(total_draws, 10)),
            "p90_rs": float(np.percentile(total_draws, 90)),
        }
    resumo = pd.DataFrame(linhas).T
    resumo.index.name = "ano"
    return resumo


def resumo_vp_portfolio(vp_amostras, empresas, quantidades, carteira):
    """Valor presente (R$) dos dividendos futuros da carteira: média, mediana, P10, P90, variância.

    `vp_amostras` (draws, empresas) vem de `calcular_vp_amostras`. Cada draw é
    somado entre os ativos antes dos percentis, preservando a correlação.
    A variância é q' Σ q, com Σ a covariância do VP por ação entre draws.
    """
    posicao = carteira.set_index("ativo")["quantidade"]
    indices = [
        i for i, e in enumerate(empresas)
        if e.replace(".SA", "") in posicao.index and not np.isnan(vp_amostras[:, i]).all()
    ]
    if not indices:
        raise ValueError("Nenhum ativo da carteira tem valor presente simulado.")
    q = np.array([float(posicao[empresas[i].replace(".SA", "")]) for i in indices])
    acoes_bi = np.array([quantidades[empresas[i]] / 1e9 for i in indices])
    por_acao = vp_amostras[:, indices] / acoes_bi
    total = por_acao @ q
    variancia = variancia_portfolio(q, np.atleast_2d(np.cov(por_acao, rowvar=False)))
    return pd.Series(
        {
            "vp_medio_rs": lucro_medio_portfolio(q, por_acao.mean(axis=0)),
            "vp_mediana_rs": float(np.median(total)),
            "vp_p10_rs": float(np.percentile(total, 10)),
            "vp_p90_rs": float(np.percentile(total, 90)),
            "variancia_rs2": variancia,
            "desvio_padrao_rs": float(np.sqrt(variancia)),
        }
    )


def plotar_risco_retorno_portfolio(
    vp_amostras, empresas, quantidades, cotacoes, carteira, caminho_saida,
    numero_carteiras=None, semente=0, eixo_x="desvio",
    limite_x=None, limite_lucro_mil=None, minimo_x=0, minimo_lucro_mil=0,
    fracao_outliers=0.05, variancia_alvo=None,
):
    """Volatilidade × lucro esperado (VP dos dividendos futuros, R$ mil) e sua carteira.

    Lucro esperado = mediana do VP da carteira; volatilidade = desvio padrão
    √(q'Σq). A nuvem são carteiras aleatórias com o mesmo capital investido
    (mesma soma quantidade × cotação) e pesos sorteados entre os ativos do
    modelo; a linha é a fronteira ótima (máxima mediana para cada risco, sem venda a descoberto).
    O envelope é o casco convexo da nuvem sem a fração `fracao_outliers` de pontos mais
    distantes (Mahalanobis).
    Carteiras sobre o envelope (retornadas em "carteiras_envelope", com pesos): a de menor
    volatilidade e a de maior lucro com volatilidade ≈ `variancia_alvo` (padrão
    VARIANCIA_ALVO_ENVELOPE, R$ mil). Ambas são carteiras sorteadas reais da nuvem sem outliers.
    `eixo_x` = "desvio" (desvio padrão, R$ mil) ou "variancia" (R$ milhões²).
    `limite_x` e `limite_lucro_mil` cortam os eixos (x na unidade do eixo, lucro
    em R$ mil); "dobro" em `limite_x` usa o dobro do valor da sua carteira.
    """
    if numero_carteiras is None:
        numero_carteiras = NUMERO_CARTEIRAS_ALEATORIAS
    if variancia_alvo is None:
        variancia_alvo = VARIANCIA_ALVO_ENVELOPE
    if eixo_x not in ("desvio", "variancia"):
        raise ValueError('eixo_x deve ser "desvio" ou "variancia".')
    posicao = carteira.set_index("ativo")["quantidade"]
    indices = [
        i for i, e in enumerate(empresas)
        if e.replace(".SA", "") in posicao.index
        and np.isfinite(cotacoes.get(e, np.nan))
        and np.isfinite(vp_amostras[:, i]).all()
    ]
    if len(indices) < 2:
        raise ValueError("São necessários ao menos 2 ativos com VP e cotação válidos.")
    nomes = [empresas[i].replace(".SA", "") for i in indices]
    acoes_bi = np.array([quantidades[empresas[i]] / 1e9 for i in indices])
    preco = np.array([cotacoes[empresas[i]] for i in indices], dtype=float)
    por_acao = vp_amostras[:, indices] / acoes_bi
    covariancia = np.cov(por_acao, rowvar=False)

    q_atual = np.array([float(posicao[n]) for n in nomes])
    capital = float(q_atual @ preco)

    def metricas(matriz_q):
        desvio = np.sqrt(np.einsum("ij,jk,ik->i", matriz_q, covariancia, matriz_q))
        mediana = np.concatenate([
            np.median(por_acao @ bloco.T, axis=0)
            for bloco in np.array_split(matriz_q, max(1, len(matriz_q) // 500))
        ])
        return desvio / 1000, mediana / 1000

    def para_eixo(desvio_mil):
        return desvio_mil**2 / 1e6 if eixo_x == "variancia" else desvio_mil

    aleatorio = np.random.default_rng(semente)
    pesos = aleatorio.dirichlet(np.ones(len(indices)), size=numero_carteiras)
    x_desvio, y = metricas(pesos * capital / preco)
    x = para_eixo(x_desvio)
    x_atual_desvio, y_atual = (v[0] for v in metricas(q_atual[None, :]))
    x_atual = para_eixo(x_atual_desvio)

    # Seleção sobre o envelope (feita sempre em desvio padrão, para ser a mesma nos dois gráficos)
    nuvem_desvio = np.column_stack([x_desvio, y])
    dif = nuvem_desvio - nuvem_desvio.mean(axis=0)
    dist = np.einsum("ij,jk,ik->i", dif, np.linalg.pinv(np.cov(nuvem_desvio, rowvar=False)), dif)
    dentro = np.where(dist <= np.quantile(dist, 1 - fracao_outliers))[0]
    vertices = dentro[ConvexHull(nuvem_desvio[dentro]).vertices]
    i_menor = int(vertices[np.argmin(x_desvio[vertices])])
    tolerancia = 0.05 * variancia_alvo
    faixa = dentro[np.abs(x_desvio[dentro] - variancia_alvo) <= tolerancia]
    while faixa.size == 0 and tolerancia < x_desvio.max():
        tolerancia *= 2
        faixa = dentro[np.abs(x_desvio[dentro] - variancia_alvo) <= tolerancia]
    if faixa.size == 0:
        raise ValueError("Nenhuma carteira sorteada perto da volatilidade alvo.")
    i_maior = int(faixa[np.argmax(y[faixa])])

    def tabela_carteira(i):
        q = pesos[i] * capital / preco
        return pd.DataFrame(
            {
                "peso_pct": pesos[i] * 100,
                "valor_rs": pesos[i] * capital,
                "quantidade_acoes": q,
                "quantidade_atual": q_atual,
            },
            index=pd.Index(nomes, name="ativo"),
        ).sort_values("peso_pct", ascending=False)

    carteiras_envelope = {
        "menor_volatilidade": {
            "tabela": tabela_carteira(i_menor),
            "volatilidade_mil": float(x_desvio[i_menor]),
            "lucro_mediano_mil": float(y[i_menor]),
        },
        "maior_lucro": {
            "tabela": tabela_carteira(i_maior),
            "volatilidade_mil": float(x_desvio[i_maior]),
            "lucro_mediano_mil": float(y[i_maior]),
        },
    }

    # Fronteira ótima: para cada nível de risco, maximiza a mediana do VP com
    # orçamento fixo e sem venda a descoberto. Parte da solução média-variância
    # (problema suave) e refina para a mediana.
    quantidade_por_peso = capital / preco
    media_por_acao = por_acao.mean(axis=0)
    orcamento = {"type": "eq", "fun": lambda w: w.sum() - 1}
    limites = [(0, 1)] * len(indices)

    def variancia_pesos(w):
        q = w * quantidade_por_peso
        return q @ covariancia @ q

    def mediana_pesos(w):
        return np.median(por_acao @ (w * quantidade_por_peso))

    w_min = minimize(variancia_pesos, np.full(len(indices), 1 / len(indices)),
                     bounds=limites, constraints=[orcamento], method="SLSQP").x
    w_max = np.zeros(len(indices))
    w_max[np.argmax(media_por_acao * quantidade_por_peso)] = 1.0
    var_min, var_max = variancia_pesos(w_min), variancia_pesos(w_max)

    pontos = [(var_min, mediana_pesos(w_min))]
    w_inicial = w_min
    for nivel_variancia in np.linspace(var_min, var_max, 30)[1:]:
        limite_risco = {"type": "ineq", "fun": lambda w, v=nivel_variancia: v - variancia_pesos(w)}
        restricoes = [orcamento, limite_risco]
        w_media = minimize(lambda w: -(w * quantidade_por_peso) @ media_por_acao, w_inicial,
                           bounds=limites, constraints=restricoes, method="SLSQP").x
        w_otimo = minimize(lambda w: -mediana_pesos(w), w_media,
                           bounds=limites, constraints=restricoes, method="SLSQP").x
        candidatos = [w_media, w_otimo]
        w_inicial = max(candidatos, key=mediana_pesos)
        pontos.append((variancia_pesos(w_inicial), mediana_pesos(w_inicial)))
    pontos = np.array(pontos)
    pontos = pontos[np.argsort(pontos[:, 0])]
    pontos[:, 1] = np.maximum.accumulate(pontos[:, 1])
    # Corta a cauda plana: mais risco sem mais lucro não pertence à fronteira
    ultimo = int(np.argmax(pontos[:, 1] >= pontos[:, 1].max() * 0.9995))
    pontos = pontos[: ultimo + 1]
    fronteira_x = np.sqrt(pontos[:, 0]) / 1000
    if eixo_x == "variancia":
        fronteira_x = fronteira_x**2 / 1e6
    fronteira_y = pontos[:, 1] / 1000

    figura, eixo = plt.subplots(figsize=(10, 7))
    eixo.scatter(x, y, s=8, alpha=0.3, color="tab:gray", label="Carteiras aleatórias (mesmo capital)")
    # Envelope: casco convexo da nuvem sem os outliers (maior distância de Mahalanobis)
    nuvem = np.column_stack([x, y])
    centro = nuvem.mean(axis=0)
    diferenca = nuvem - centro
    distancia = np.einsum(
        "ij,jk,ik->i", diferenca, np.linalg.pinv(np.cov(nuvem, rowvar=False)), diferenca
    )
    interior = nuvem[distancia <= np.quantile(distancia, 1 - fracao_outliers)]
    casco = interior[ConvexHull(interior).vertices]
    eixo.fill(casco[:, 0], casco[:, 1], facecolor="tab:orange", alpha=0.12,
              edgecolor="tab:orange", linewidth=1.5, label="Envelope das carteiras")
    eixo.plot(fronteira_x, fronteira_y, color="tab:blue", linewidth=2, label="Fronteira ótima")
    eixo.scatter(x_atual, y_atual, s=180, marker="*", color="tab:red", zorder=5,
                 edgecolor="black", label="Sua carteira")
    eixo.scatter(x[i_menor], y[i_menor], s=90, marker="D", color="tab:green", zorder=5,
                 edgecolor="black", label="Envelope: menor volatilidade")
    eixo.scatter(x[i_maior], y[i_maior], s=90, marker="s", color="tab:purple", zorder=5,
                 edgecolor="black", label="Envelope: maior lucro (volatilidade alvo)")
    eixo.axvline(para_eixo(variancia_alvo), color="tab:purple", linestyle=":", linewidth=1)
    eixo.set_xlabel(
        "Variância do VP (R$ milhões²)" if eixo_x == "variancia"
        else "Volatilidade: desvio padrão do VP (R$ mil)"
    )
    eixo.set_ylabel("Lucro esperado: mediana do VP dos dividendos (R$ mil)")
    eixo.set_title(
        f"{'Variância' if eixo_x == 'variancia' else 'Volatilidade'} × lucro esperado da carteira"
    )
    if limite_x == "dobro":
        limite_x = 2 * x_atual
    if limite_x is not None:
        eixo.set_xlim(minimo_x, limite_x)
    if limite_lucro_mil is not None:
        eixo.set_ylim(minimo_lucro_mil, limite_lucro_mil)
    eixo.grid(alpha=0.3)
    eixo.legend(loc="lower right")
    figura.tight_layout()
    figura.savefig(caminho_saida, dpi=200)
    plt.close(figura)
    return {
        "x_atual": float(x_atual),
        "lucro_mediano_mil": float(y_atual),
        "carteiras_envelope": carteiras_envelope,
    }


def plotar_pizza_carteira_selecionada(tabela, titulo, caminho_saida):
    """Pizza da composição (% do capital) de uma carteira selecionada."""
    pesos = tabela.loc[tabela["peso_pct"] >= 0.05, "peso_pct"]
    figura, eixo = plt.subplots(figsize=(9, 9))
    eixo.pie(
        pesos,
        labels=pesos.index,
        autopct="%1.1f%%",
        colors=plt.cm.tab20.colors[: len(pesos)],
        startangle=90,
        counterclock=False,
        pctdistance=0.8,
        wedgeprops={"edgecolor": "white"},
    )
    eixo.set_title(titulo)
    figura.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close(figura)


def dividendos_historicos_carteira(dividendos_historicos, empresas, quantidades, carteira):
    """Dividendos passados da carteira (R$) por ano: dividendo por ação × sua quantidade."""
    posicao = carteira.set_index("ativo")["quantidade"]
    total = None
    for empresa in empresas:
        ativo = empresa.replace(".SA", "")
        if ativo not in posicao.index or empresa not in dividendos_historicos.index:
            continue
        serie = dividendos_historicos.loc[empresa]
        if serie.isna().all():
            continue
        valor = serie.fillna(0) / (quantidades[empresa] / 1e9) * float(posicao[ativo])
        total = valor if total is None else total.add(valor, fill_value=0)
    return total


def plotar_risco_portfolio(resumo, historico, caminho_saida):
    """Dividendos da carteira (R$ mil): histórico, mediana futura e faixa P10–P90."""
    figura, eixo = plt.subplots(figsize=(12, 6))
    anos = resumo.index.astype(int)
    if historico is not None:
        eixo.plot(
            historico.index.astype(int), historico / 1000, color="black",
            marker="o", linewidth=2, label="Histórico",
        )
    eixo.plot(anos, resumo["dividendo_mediano_rs"] / 1000, color="tab:blue",
              marker="o", linewidth=2, label="Mediana prevista")
    eixo.plot(anos, resumo["p10_rs"] / 1000, color="tab:red", linestyle="--", label="P10")
    eixo.plot(anos, resumo["p90_rs"] / 1000, color="tab:green", linestyle="--", label="P90")
    eixo.fill_between(anos, resumo["p10_rs"] / 1000, resumo["p90_rs"] / 1000,
                      color="tab:blue", alpha=0.15)
    eixo.set_ylim(bottom=0)
    eixo.set_xlabel("Ano")
    eixo.set_ylabel("Dividendos (R$ mil, preços de 2025)")
    eixo.set_title("Dividendos da carteira: histórico, mediana prevista e P10–P90")
    eixo.grid(alpha=0.3)
    eixo.legend(loc="upper left")
    figura.tight_layout()
    figura.savefig(caminho_saida, dpi=200)
    plt.close(figura)


def plotar_tabela_dividendos_carteira(tabela, caminho_saida):
    """Salva uma figura com a tabela de dividendos por ativo e ano (R$ mil)."""
    texto = (tabela / 1000).map("{:,.1f}".format)
    texto.columns = [str(c) for c in texto.columns]
    figura, eixo = plt.subplots(figsize=(0.62 * len(texto.columns) + 2, 0.6 * len(texto) + 1.8))
    eixo.axis("off")
    desenho = eixo.table(
        cellText=texto.values,
        colLabels=texto.columns,
        rowLabels=texto.index,
        cellLoc="center",
        loc="center",
    )
    desenho.auto_set_font_size(False)
    desenho.set_fontsize(9)
    desenho.scale(1, 1.6)
    for (linha, coluna), celula in desenho.get_celld().items():
        if linha == 0 or coluna == -1:
            celula.set_facecolor("#d9e2f3")
            celula.set_text_props(weight="bold")
        if linha == len(texto):
            celula.set_text_props(weight="bold")
    figura.suptitle(
        "Dividendos esperados na sua carteira por ano (R$ mil, mediana × quantidade, preços de 2025)",
        fontsize=12,
    )
    figura.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close(figura)


def plotar_dividendos_carteira(tabela, caminho_saida):
    """Gráfico de dividendos esperados (R$ mil) ao longo do tempo, empilhado por ativo."""
    por_ativo = tabela.drop(index="TOTAL") / 1000
    anos = por_ativo.columns.astype(int)
    figura, eixo = plt.subplots(figsize=(12, 6))
    eixo.stackplot(anos, por_ativo.values, labels=por_ativo.index, alpha=0.85)
    eixo.plot(anos, por_ativo.sum(), color="black", linewidth=2, label="TOTAL")
    eixo.set_xlim(anos.min(), anos.max())
    eixo.set_ylim(bottom=0)
    eixo.set_xlabel("Ano")
    eixo.set_ylabel("Dividendos (R$ mil, preços de 2025)")
    eixo.set_title("Dividendos esperados na sua carteira (mediana × quantidade)")
    eixo.grid(alpha=0.3)
    eixo.legend(loc="lower left", ncol=5)
    figura.tight_layout()
    figura.savefig(caminho_saida, dpi=200)
    plt.close(figura)


def plotar_distribuicao_vp(
    vp_amostras,
    empresas,
    quantidades,
    cotacoes,
    fracao_cotacao,
    prob_alvo,
    caminho_saida,
):
    """Ilustra a distribuição do VP por ação de cada empresa contra a cotação.

    O eixo x é o VP por ação dividido pela cotação (1,0 = preço de mercado),
    o que permite comparar empresas de preços muito diferentes. A densidade é
    verde onde o VP supera o limiar (fracao_cotacao x cotação) e vermelha
    abaixo dele. A linha tracejada é a cotação; o losango marca a cotação
    alvo (onde a probabilidade de superar o limiar seria prob_alvo).
    """
    cores = {"ok": "#2a9d63", "ruim": "#d64550", "cotacao": "#1d3557", "alvo": "#f4a300"}
    quantidades_por_empresa = pd.Series(quantidades, dtype=float)
    linhas = []
    for indice, empresa in enumerate(empresas):
        cotacao = cotacoes.get(empresa, np.nan)
        amostras = vp_amostras[:, indice]
        if not np.isfinite(cotacao) or np.all(amostras == 0):
            continue
        razao = amostras / (quantidades_por_empresa[empresa] / 1e9) / cotacao
        razao = razao[np.isfinite(razao)]
        if razao.size < 2 or np.ptp(razao) == 0:
            print(f"Aviso: {empresa} sem VP válido (quantidade de ações ou dividendos ausentes); ignorado no gráfico.")
            continue
        linhas.append((empresa.replace(".SA", ""), razao))

    x_max = max(np.percentile(razao, 97) for _, razao in linhas) * 1.05
    grade = np.linspace(0, x_max, 600)
    figura, eixo = plt.subplots(figsize=(12, 1.9 * len(linhas) + 1.6))
    altura = 0.85

    for posicao, (nome, razao) in enumerate(reversed(linhas)):
        base = posicao
        densidade = gaussian_kde(razao)(grade)
        densidade = densidade / densidade.max() * altura
        acima = grade >= fracao_cotacao
        eixo.fill_between(grade, base, base + densidade, where=~acima,
                          color=cores["ruim"], alpha=0.75, linewidth=0)
        eixo.fill_between(grade, base, base + densidade, where=acima,
                          color=cores["ok"], alpha=0.75, linewidth=0)
        eixo.plot(grade, base + densidade, color="white", linewidth=1)
        eixo.hlines(base, 0, x_max, color="#bbbbbb", linewidth=0.8)

        p10, p50, p90 = np.percentile(razao, [10, 50, 90])
        alvo = np.percentile(razao, 100 * (1 - prob_alvo)) / fracao_cotacao
        probabilidade = np.mean(razao > fracao_cotacao)
        eixo.plot([p10, p90], [base - 0.12] * 2, color="black", linewidth=3,
                  solid_capstyle="round")
        eixo.plot(p50, base - 0.12, "o", color="white", markeredgecolor="black",
                  markersize=8, zorder=5)
        eixo.plot(alvo, base - 0.12, "D", color=cores["alvo"], markeredgecolor="black",
                  markersize=9, zorder=6)
        eixo.text(-0.01 * x_max, base + altura * 0.35, nome, ha="right",
                  va="center", fontsize=14, fontweight="bold")
        eixo.text(-0.01 * x_max, base + altura * 0.05,
                  f"cotação R$ {cotacoes[next(e for e in empresas if e.startswith(nome))]:.2f}",
                  ha="right", va="center", fontsize=9, color="#555555")
        eixo.text(x_max, base + altura * 0.8, f"{probabilidade:.0%}", ha="right",
                  va="center", fontsize=20, fontweight="bold",
                  color=cores["ok"] if probabilidade >= prob_alvo else cores["ruim"])
        eixo.text(x_max, base + altura * 0.5, "chance de VP > limiar", ha="right",
                  va="center", fontsize=8, color="#555555")

    topo = len(linhas) - 0.05 + altura
    eixo.vlines(1, -0.4, topo, color=cores["cotacao"], linestyle="--", linewidth=1.8)
    eixo.vlines(fracao_cotacao, -0.4, topo, color=cores["ruim"], linestyle=":",
                linewidth=1.5)
    eixo.text(1.02, topo, "cotação (1,0x)", ha="left", va="bottom",
              color=cores["cotacao"], fontsize=10, fontweight="bold")
    eixo.text(fracao_cotacao - 0.02, topo, f"limiar ({fracao_cotacao:.1f}x)".replace(".", ","),
              ha="right", va="bottom", color=cores["ruim"], fontsize=10,
              fontweight="bold")
    eixo.set_xlim(0, x_max)
    eixo.set_ylim(-0.4, topo + 0.3)
    eixo.set_yticks([])
    eixo.set_xlabel("VP dos dividendos por ação ÷ cotação atual")
    for borda in ("top", "right", "left"):
        eixo.spines[borda].set_visible(False)
    eixo.set_title(
        "Quanto valem os dividendos futuros frente ao preço de mercado?",
        fontsize=14, fontweight="bold", loc="left", pad=26,
    )
    eixo.text(
        0, 1.0,
        "Barra preta = P10 a P90 | círculo = mediana | ◆ laranja = cotação alvo "
        f"(chance de {prob_alvo:.0%} de superar o limiar)",
        transform=eixo.transAxes, ha="left", va="bottom", fontsize=9, color="#555555",
    )
    figura.tight_layout()
    figura.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.close(figura)


def plotar_tabela_valor_presente(valor_presente, caminho_saida):
    """Salva uma figura com a tabela do valor presente dos dividendos.

    Os parâmetros comuns (taxas, ano-base e ano final) vão no título, numa
    única linha, em vez de repetidos em colunas.
    """
    primeira = valor_presente.iloc[0]
    titulo = (
        "Valor presente dos dividendos por ação\n"
        f"Taxa real {primeira['taxa_real']:.1%} | Inflação {primeira['inflacao']:.1%} | "
        f"Taxa nominal {primeira['taxa_nominal']:.2%} | "
        f"Ano-base {int(primeira['ano_base'])} | Ano final {int(primeira['ano_final'])}\n"
        f"Limiar = {primeira['fracao_cotacao']:.0%} da cotação "
        "(Prob. > Limiar = chance de o VP por ação superar o limiar)\n"
        f"Cot. alvo = cotação em que essa probabilidade seria {primeira['prob_alvo']:.0%}"
    )
    por_acao = valor_presente[valor_presente["ticker"] != "CARTEIRA"]
    tabela = pd.DataFrame(
        {
            "Empresa": por_acao["ticker"].str.replace(".SA", "", regex=False),
            "Cotação (R$)": por_acao["cotacao"].map("{:,.2f}".format),
            "VP P10 (R$)": por_acao["vp_por_acao_p10"].map("{:,.2f}".format),
            "VP P50 (R$)": por_acao["vp_por_acao_p50"].map("{:,.2f}".format),
            "VP P90 (R$)": por_acao["vp_por_acao_p90"].map("{:,.2f}".format),
            "Prob. > Limiar": por_acao["prob_vp_maior_fracao"].map(
                "{:.0%}".format
            ),
            "Cot. alvo (R$)": por_acao["cotacao_maxima"].map(
                "{:,.2f}".format
            ),
        }
    )
    figura, eixo = plt.subplots(figsize=(12, 0.6 * len(tabela) + 1.8))
    eixo.axis("off")
    desenho = eixo.table(
        cellText=tabela.values,
        colLabels=tabela.columns,
        cellLoc="center",
        loc="center",
    )
    desenho.auto_set_font_size(False)
    desenho.set_fontsize(11)
    desenho.scale(1, 1.6)
    for (linha, _), celula in desenho.get_celld().items():
        if linha == 0:
            celula.set_facecolor("#d9e2f3")
            celula.set_text_props(weight="bold")
    figura.suptitle(titulo, fontsize=12)
    figura.tight_layout()
    figura.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.close(figura)


# def plotar_dividendos(
#     relatorio,
#     dividendos_historicos,
#     empresas,
#     ano_inicial,
#     ano_final,
#     ultimo_ano_observado,
#     ano_base_fcl,
#     modo,
#     caminho_saida,
# ):
#     """Desenha observações históricas e a preditiva de dividendos P10/P50/P90."""

# linhas_grade = int(np.ceil(len(empresas) / 2))
# if not USAR_EMPRESAS_MUTLIVARIADAS:

#     figura_final, eixos_finais = plt.subplots(
#             len(empresas), 1, figsize=(20, 5 * len(empresas)), squeeze=False
#     )

    
#     #figsize=(20, 5 * numero_empresas),
#      #   squeeze=False,

#     for indice_empresa, empresa in enumerate(empresas):
#         eixo = eixos.flat[indice_empresa]
#         tabela = relatorio[relatorio["ticker"] == empresa]
#         if tabela.empty:
#             eixo.set_axis_off()
#             continue

#         historico = dividendos_historicos.loc[empresa].dropna()
#         eixo.scatter(
#             historico.index, historico.to_numpy(), color="black",
#             label="Dividendo observado",
#         )
#         eixo.plot(
#             tabela["ano"], tabela["dividendo_p50_rs_bi"], color="blue",
#             label="Dividendo mediano",
#         )
#         eixo.plot(
#             tabela["ano"], tabela["dividendo_p90_rs_bi"], color="green",
#             linestyle="--", label="Dividendo P90",
#         )
#         eixo.plot(
#             tabela["ano"], tabela["dividendo_p10_rs_bi"], color="red",
#             linestyle="--", label="Dividendo P10",
#         )
#         eixo.axvline(ultimo_ano_observado + 0.5, color="gray", linestyle=":")
#         eixo.set_xlim(ano_inicial - 0.5, ano_final + 0.5)
#         eixo.set_xticks(range(ano_inicial, ano_final + 1, 2))
#         eixo.set_title(f"{empresa} - dividendos até {ano_final}")
#         eixo.set_xlabel("Ano")
#         eixo.set_ylabel(f"R$ bi (preços de {ano_base_fcl})")
#         eixo.grid(alpha=0.3)
#         eixo.legend(fontsize=8, framealpha=0.6)

#     for eixo_vazio in eixos.flat[len(empresas):]:
#         eixo_vazio.set_axis_off()
#     figura_final.suptitle(
#         f"Dividendos totais: observados e P10/mediana/P90 do modelo, "
#         f"{ano_inicial}-{ano_final} (payout {modo} do FCL)"
#     )
#     figura_final.tight_layout()
#     figura_final.savefig(caminho_saida, dpi=200)
#     plt.close(figura_final)

def plotar_dividendos(
    relatorio,
    dividendos_historicos,
    empresas,
    ano_inicial,
    ano_final,
    ultimo_ano_observado,
    ano_base_fcl,
    modo,
    caminho_saida,
):
    """Desenha observações históricas e a preditiva de dividendos P10/P50/P90."""
    numero_empresas = len(empresas)
    figura, eixos = plt.subplots(
        numero_empresas,
        1,
        figsize=(7, 4 * numero_empresas),
        squeeze=False,
    )
    for indice_empresa, empresa in enumerate(empresas):
        eixo = eixos.flat[indice_empresa]
        tabela = relatorio[relatorio["ticker"] == empresa]
        if tabela.empty:
            eixo.set_axis_off()
            continue

        historico = dividendos_historicos.loc[empresa].dropna()
        eixo.scatter(
            historico.index, historico.to_numpy(), color="black",
            label="Dividendo observado",
        )
        eixo.plot(
            tabela["ano"], tabela["dividendo_p50_rs_bi"], color="blue",
            label="Dividendo mediano",
        )
        eixo.plot(
            tabela["ano"], tabela["dividendo_p90_rs_bi"], color="green",
            linestyle="--", label="Dividendo P90",
        )
        eixo.plot(
            tabela["ano"], tabela["dividendo_p10_rs_bi"], color="red",
            linestyle="--", label="Dividendo P10",
        )
        eixo.axvline(ultimo_ano_observado + 0.5, color="gray", linestyle=":")
        eixo.set_xlim(ano_inicial, ano_final)
        eixo.set_title(f"{empresa} - dividendos até {ano_final}", fontsize=12)
        eixo.set_xlabel("Ano")
        eixo.set_ylabel(f"R$ bi (preços de {ano_base_fcl})")
        eixo.grid(alpha=0.3)
        eixo.legend(fontsize=8)

    for eixo_vazio in eixos.flat[len(empresas):]:
        eixo_vazio.set_axis_off()
    figura.suptitle(
        f"Dividendos totais: observados e P10/mediana/P90 do modelo, "
        f"{ano_inicial}-{ano_final} (payout {modo} do FCL)",
        fontsize=12,
    )
    figura.tight_layout(rect=(0, 0, 1, 0.96))
    figura.savefig(caminho_saida, dpi=200, bbox_inches="tight")
    plt.close(figura)


def calcular_payout_historico(
    proventos_anuais,
    fcl,
    quantidades_acoes,
    empresas,
    caminho_proventos_totais,
):
    """Calcula payout anual e combina proventos informados com estimados.

    Quando um dividendo total anual foi preenchido no CSV, ele prevalece;
    células vazias usam proventos por ação multiplicados pelas ações atuais.
    """
    anos_payout = pd.Index(range(2016, 2026), dtype=int)
    proventos_anuais = proventos_anuais.copy()
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
    quantidades = pd.Series(quantidades_acoes, dtype=float).reindex(empresas)
    dividendos_totais = proventos_por_acao.mul(quantidades, axis="index") / 1e9

    dividendos_informados = pd.read_csv(
        caminho_proventos_totais,
        sep=r"\s+",
        decimal=".",
        index_col=0,
    )
    dividendos_informados.index = dividendos_informados.index.str.strip()
    dividendos_informados.columns = dividendos_informados.columns.astype(int)
    dividendos_informados = (
        dividendos_informados.apply(pd.to_numeric, errors="coerce")
        .reindex(index=empresas, columns=anos_payout)
    )
    dividendos_totais = dividendos_informados.combine_first(dividendos_totais)
    payout_anual = dividendos_totais.div(fcl_payout.replace(0, np.nan)).mul(100)
    payout_medio = payout_anual.mean(axis="columns")
    return (
        anos_payout,
        payout_anual,
        payout_medio,
        dividendos_totais,
        dividendos_informados,
    )


inpc_anual = ler_inpc_anual(ano_inicial)
inpc_anual.to_csv(OUTPUT_DIR / "INPC_anual.csv", decimal=",")
print("INPC anual carregado:")
print(inpc_anual)

fcl, capex_expansao, fcl_futuro_projetado, proventos_anuais = ler_dados_entrada(
    BASE_DIR / "data_input"
)
(
    fcl,
    fatores_correcao_inpc,
    ano_base_fcl,
    fcl_corrigido,
) = preparar_dados_fcl(
    fcl=fcl,
    capex_expansao=capex_expansao,
    inpc_anual=inpc_anual,
    usar_capex=USAR_FCL_CORRIGIDO_capex_expandido,
    diretorio_saida=OUTPUT_DIR,
)
if DEFLACIONAR_FCL_FUTURO_PROJETADO:
    fatores_deflacao = (1 + INFLACAO) ** (fcl_futuro_projetado.columns - ano_base_fcl)
    fcl_futuro_projetado = fcl_futuro_projetado.div(fatores_deflacao, axis="columns")
    print(f"FCL futuro projetado deflacionado a {INFLACAO:.0%} a.a. para preços de {ano_base_fcl}.")
print(f"FCL corrigido para preços de {ano_base_fcl}:")
print(fcl_corrigido)

# Escolha os dados usados no modelo e nos gráficos:
# True usa o FCL corrigido pelo INPC; False usa os valores originais.
#USAR_FCL_CORRIGIDO_INPC = True

fcl_utilizado = fcl_corrigido if USAR_FCL_CORRIGIDO_INPC else fcl
nome_serie_fcl = "corrigido pelo INPC" if USAR_FCL_CORRIGIDO_INPC else "original"
print(f"FCL utilizado: {nome_serie_fcl}")


def construir_modelo_bayesiano(ticker, eixos):
    """Ajusta o modelo individual de FCL e preenche seus quatro gráficos.

    Retorna o modelo PyMC, os resultados posteriores e os resumos preditivos.
    """
    anos = np.asarray(fcl_utilizado.columns, dtype=float)
    fluxo_caixa = mascarar_fcl_antes_do_primeiro_valor(
        fcl_utilizado.loc[ticker].astype(float)
    ).to_numpy()
    valido = ~np.isnan(fluxo_caixa)
    anos_padronizados = ((anos - anos.mean()) / anos.std())[valido]
    fluxo_caixa = fluxo_caixa[valido]

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
    eixo_fcl.scatter(anos[valido], fluxo_caixa, color="black", label="FCL observado")
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

################################################################################

################################################################################


################################################################################33
#
# Modelo bayesiano multivariado para as três empresas


# Matriz: linhas = anos; colunas = empresas
dados_fcl = mascarar_fcl_antes_do_primeiro_valor(
    fcl_utilizado.loc[empresas_multivariadas].T
)
numero_empresas = len(empresas_multivariadas)
(
    modelo_multivariado,
    idata_multivariado,
    anos,
    fcl_observado,
) = ajustar_modelo_multivariado(
    dados_fcl=dados_fcl,
    usar_correlacao=USAR_EMPRESAS_MUTLIVARIADAS,
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


(
    alpha_amostras,
    beta_amostras,
    desvios_amostras,
    correlacao_amostras,
    covariancia_amostras,
) = extrair_amostras_covariancia(
    idata=idata_multivariado,
    usar_correlacao=USAR_EMPRESAS_MUTLIVARIADAS,
    numero_empresas=numero_empresas,
)


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
    figsize=(12, 4 * int(np.ceil(len(empresas) / 2)))
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
    "Correlação posterior média entre os FCLs",
    fontsize=12,
)

figura_correlacao.colorbar(
    imagem,
    ax=eixo_correlacao,
    label="Correlação",
)

figura_correlacao.tight_layout(rect=(0, 0, 1, 0.96))

figura_correlacao.savefig(
    FIG_DIR / "matriz_correlacao_posterior.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(figura_correlacao)


# Figura do modelo multivariado, no mesmo formato da figura individual.
#anos_projecao = np.arange(2016, 2037)
previsoes_multivariadas = simular_previsoes_multivariadas(
    anos_historicos=anos,
    anos_projecao=anos_projecao,
    alpha_amostras=alpha_amostras,
    beta_amostras=beta_amostras,
    covariancia_amostras=covariancia_amostras,
)

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
        eixo.set_title(f"{empresa} - Modelo {nome_modelo.lower()}", fontsize=12)
        eixo.set_xlabel("Ano")
        eixo.set_ylabel("Fluxo de caixa livre")
        eixo.legend(fontsize=8)
        eixo.grid(alpha=0.3)

figura_fcl.suptitle(
    "Projeções de FCL: "
    + ("modelo multivariado" if USAR_EMPRESAS_MUTLIVARIADAS else "modelos individual e independente"),
    fontsize=12,
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
                np.nanmean(fcl_observado[:, indice_empresa]),
                10 * np.nanstd(fcl_observado[:, indice_empresa]),
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

(
    anos_payout,
    payout_anual_percentual,
    payout_medio_por_empresa,
    proventos_totais_estimados,
    proventos_totais_informados,
) = calcular_payout_historico(
    proventos_anuais=proventos_anuais,
    fcl=fcl,
    quantidades_acoes=quantidades_acoes,
    empresas=empresas,
    caminho_proventos_totais=BASE_DIR / "data_input" / "proventos_totais.csv",
)
print("\nProventos totais informados (R$ bi; NaN = usa provento/ação x ações atuais):")
print(proventos_totais_informados)
payout_resumo_bayesiano = None

if USAR_PAYOUT_BAYESIANO:
    linhas_payout = int(np.ceil(len(empresas) / 2))
    figura_payout, eixos_payout = plt.subplots(
        linhas_payout,
        2,
        figsize=(12, 4 * linhas_payout),
        squeeze=False,
    )
    resumos_posteriores = []
    densidades_posteriores = []
    parametros_payout_posterior = {}
    parametros_payout_dependente = {}

    for indice_empresa, empresa in enumerate(empresas):
        eixo = eixos_payout.flat[indice_empresa]
        eixo.set_xlim(0, 120)
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
        idata_payout = ajustar_modelo_payout(
            observacoes=observacoes_payout,
            random_seed=42 + indice_empresa,
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
        parametros_payout_posterior[empresa] = (mu_log_amostras, sigma_log_amostras)

        if not USAR_PAYOUT_INDEPENDENTE_FCL:
            # Payout dependente do FCL: log(payout) = mu + gamma*(log FCL - centro).
            # gamma < 0 indica que o dividendo é "pegajoso": com FCL menor, o
            # payout sobe. O FCL usado como covariável é o mesmo (em preços
            # constantes) da preditiva futura.
            serie_fcl_dep = fcl_utilizado.loc[empresa].astype(float)
            serie_fcl_dep.index = serie_fcl_dep.index.astype(int)
            tabela_dep = pd.DataFrame(
                {
                    "payout": payout_anual_percentual.loc[empresa],
                    "fcl": serie_fcl_dep.reindex(
                        payout_anual_percentual.columns.astype(int)
                    ).to_numpy(),
                }
            ).replace([np.inf, -np.inf], np.nan).dropna()
            tabela_dep = tabela_dep[(tabela_dep["payout"] > 0) & (tabela_dep["fcl"] > 0)]
            print(f"{empresa}: {len(tabela_dep)} anos usados no modelo de payout dependente.")
            centro_log_fcl = float(np.log(tabela_dep["fcl"]).mean())
            idata_dep = ajustar_modelo_payout_dependente(
                tabela=tabela_dep,
                indice_empresa=indice_empresa,
                centro_log_fcl=centro_log_fcl,
            )
            parametros_payout_dependente[empresa] = (
                idata_dep.posterior["dep_mu"].values.flatten(),
                idata_dep.posterior["dep_gamma"].values.flatten(),
                idata_dep.posterior["dep_sigma"].values.flatten(),
                centro_log_fcl,
            )
            print(
                f"{empresa}: gamma (efeito de log FCL no log payout) = "
                f"{parametros_payout_dependente[empresa][1].mean():.2f} "
                f"[{np.percentile(parametros_payout_dependente[empresa][1], 5):.2f}, "
                f"{np.percentile(parametros_payout_dependente[empresa][1], 95):.2f}]"
            )
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
        eixo.set_title(f"{empresa} - payout mediano posterior", fontsize=12)
        eixo.set_xlabel("Payout (%)")
        eixo.set_ylabel("Densidade posterior")
        eixo.legend(fontsize=8)
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

    for eixo_vazio in eixos_payout.flat[len(empresas):]:
        eixo_vazio.set_axis_off()
    figura_payout.suptitle(
        "Densidade posterior do payout mediano, 2016-2025",
        fontsize=12,
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


#--------------------------------------------------------------
#inferencia dos dividendos futuros
#
# Dividendo(empresa, ano) = payout(empresa, ano) x max(FCL(empresa, ano), 0)
# Payout e FCL são tratados como independentes: cada amostra de dividendo
# multiplica uma amostra da preditiva do FCL (modelo multivariado, que mantém
# a correlação entre empresas) por uma amostra da preditiva do payout
# (log-normal), propagando a incerteza dos dois fatores.
# Unidades: R$ bilhões a preços de ano_base_fcl (o FCL do modelo é corrigido
# pelo INPC; o payout é uma razão e não depende da base de preços).

if USAR_PAYOUT_BAYESIANO and parametros_payout_posterior:
    ultimo_ano_observado = int(fcl_utilizado.columns.astype(int).max())
    anos_dividendos, dividendos_amostras = simular_dividendos_futuros(
        anos_projecao=anos_projecao,
        ano_final=ano_final_plot,
        previsoes_fcl=previsoes_multivariadas,
        empresas=empresas_multivariadas,
        parametros_independentes=parametros_payout_posterior,
        parametros_dependentes=parametros_payout_dependente,
        payout_independente=USAR_PAYOUT_INDEPENDENTE_FCL,
    )
    relatorio_dividendos = resumir_dividendos(
        anos_dividendos=anos_dividendos,
        dividendos_amostras=dividendos_amostras,
        empresas=empresas_multivariadas,
        quantidades=quantidades_acoes,
    )
    sufixo_modo = "independente" if USAR_PAYOUT_INDEPENDENTE_FCL else "dependente"
    arquivo_dividendos = OUTPUT_DIR / f"dividendos_futuros_posterior_{sufixo_modo}.csv"
    relatorio_dividendos.to_csv(arquivo_dividendos, index=False, decimal=",", float_format="%.3f")

    # Dividendos históricos corrigidos pelo INPC para a mesma base de preços
    # do FCL, para que passado e previsão fiquem comparáveis no mesmo gráfico.
    fatores_hist = pd.Series(
        fatores_correcao_inpc.to_numpy(),
        index=fatores_correcao_inpc.index.astype(int),
    )
    dividendos_historicos = proventos_totais_estimados.mul(
        fatores_hist.reindex(proventos_totais_estimados.columns.astype(int)).to_numpy(),
        axis="columns",
    )

    plotar_dividendos(
        relatorio=relatorio_dividendos,
        dividendos_historicos=dividendos_historicos,
        empresas=empresas_multivariadas,
        ano_inicial=ano_inicial,
        ano_final=ano_final_plot,
        ultimo_ano_observado=ultimo_ano_observado,
        ano_base_fcl=ano_base_fcl,
        modo=sufixo_modo,
        caminho_saida=FIG_DIR / f"dividendos_todas_{sufixo_modo}.png",
    )

    print("\nDividendos futuros (posterior preditiva, R$ bi):")
    print(relatorio_dividendos.round(2).to_string(index=False))
    print(f"Dividendos futuros salvos em: {arquivo_dividendos}")

    cotacoes = buscar_cotacoes_yahoo(empresas_multivariadas)
    valor_presente = valor_presente_dividendos(
        anos_dividendos=anos_dividendos,
        dividendos_amostras=dividendos_amostras,
        empresas=empresas_multivariadas,
        quantidades=quantidades_acoes,
        taxa=TAXA,
        ano_base=ultimo_ano_observado,
        inflacao=INFLACAO,
        cotacoes=cotacoes,
        fracao_cotacao=FRACAO_COTACAO,
        prob_alvo=PROB_ALVO,
    )
    arquivo_vp = OUTPUT_DIR / f"valor_presente_dividendos_{sufixo_modo}.csv"
    valor_presente.to_csv(arquivo_vp, index=False, decimal=",", float_format="%.3f")
    print(f"\nValor presente dos dividendos (taxa real {TAXA:.1%}, inflação {INFLACAO:.1%}, nominal {(1 + TAXA) * (1 + INFLACAO) - 1:.2%}, base {ultimo_ano_observado}, R$ bi):")
    print(valor_presente.round(2).to_string(index=False))
    print(f"Valor presente salvo em: {arquivo_vp}")
    figura_vp = FIG_DIR / f"valor_presente_dividendos_{sufixo_modo}.png"
    plotar_tabela_valor_presente(valor_presente, figura_vp)
    print(f"Tabela do valor presente salva em: {figura_vp}")

    vp_amostras, _ = calcular_vp_amostras(
        anos_dividendos, dividendos_amostras, TAXA, ultimo_ano_observado, INFLACAO
    )
    figura_distribuicao = FIG_DIR / f"valor_presente_distribuicao_{sufixo_modo}.png"
    plotar_distribuicao_vp(
        vp_amostras=vp_amostras,
        empresas=empresas_multivariadas,
        quantidades=quantidades_acoes,
        cotacoes=cotacoes,
        fracao_cotacao=FRACAO_COTACAO,
        prob_alvo=PROB_ALVO,
        caminho_saida=figura_distribuicao,
    )
    print(f"Ilustração do valor presente salva em: {figura_distribuicao}")

    carteira_planilha = ler_carteira_excel()
    comparacao = comparar_com_carteira(
        vp_amostras=vp_amostras,
        empresas=empresas_multivariadas,
        quantidades=quantidades_acoes,
        cotacoes=cotacoes,
        carteira=carteira_planilha,
    )
    arquivo_comparacao = OUTPUT_DIR / "comparacao_carteira.csv"
    comparacao.to_csv(arquivo_comparacao, index=False, decimal=",", sep=";", float_format="%.3f")
    print("\nVP do modelo x sua carteira (planilha):")
    print(comparacao.round(2).to_string(index=False))
    print(f"Comparação salva em: {arquivo_comparacao}")

    dividendos_carteira = dividendos_anuais_carteira(
        anos_dividendos=anos_dividendos,
        dividendos_amostras=dividendos_amostras,
        empresas=empresas_multivariadas,
        quantidades=quantidades_acoes,
        carteira=carteira_planilha,
        ano_base=ultimo_ano_observado,
    )
    arquivo_div_carteira = OUTPUT_DIR / "dividendos_carteira_por_ano.csv"
    dividendos_carteira.to_csv(arquivo_div_carteira, sep=";", decimal=",", float_format="%.2f")
    print("\nDividendos esperados na sua carteira (mediana × quantidade, R$):")
    print(dividendos_carteira.round(0).to_string())
    print(f"Tabela completa salva em: {arquivo_div_carteira}")
    figura_div_carteira = FIG_DIR / "dividendos_carteira_por_ano.png"
    plotar_tabela_dividendos_carteira(dividendos_carteira, figura_div_carteira)
    print(f"Figura da tabela salva em: {figura_div_carteira}")
    grafico_div_carteira = FIG_DIR / "dividendos_carteira_grafico.png"
    plotar_dividendos_carteira(dividendos_carteira, grafico_div_carteira)
    print(f"Gráfico dos dividendos da carteira salvo em: {grafico_div_carteira}")

    resumo_portfolio = resumo_portfolio_por_ano(
        anos_dividendos=anos_dividendos,
        dividendos_amostras=dividendos_amostras,
        fcl_amostras=previsoes_multivariadas[:, : len(anos_dividendos), :],
        covariancia_fcl=covariancia_media,
        empresas=empresas_multivariadas,
        quantidades=quantidades_acoes,
        carteira=carteira_planilha,
        ano_base=ultimo_ano_observado,
        fonte_covariancia=COR_FCL_ou_DIVIDENDOS,
    )
    arquivo_risco = OUTPUT_DIR / f"risco_portfolio_{COR_FCL_ou_DIVIDENDOS.lower()}.csv"
    resumo_portfolio.to_csv(arquivo_risco, sep=";", decimal=",", float_format="%.2f")
    print(f"\nDividendo médio e risco da carteira (covariância de {COR_FCL_ou_DIVIDENDOS}, R$):")
    print(resumo_portfolio.round(0).to_string())
    print(f"Risco da carteira salvo em: {arquivo_risco}")
    figura_risco = FIG_DIR / f"risco_portfolio_{COR_FCL_ou_DIVIDENDOS.lower()}.png"
    plotar_risco_portfolio(
        resumo_portfolio,
        dividendos_historicos_carteira(
            dividendos_historicos, empresas_multivariadas, quantidades_acoes, carteira_planilha
        ),
        figura_risco,
    )
    print(f"Gráfico de risco da carteira salvo em: {figura_risco}")

    resumo_vp = resumo_vp_portfolio(
        vp_amostras, empresas_multivariadas, quantidades_acoes, carteira_planilha
    )
    arquivo_vp = OUTPUT_DIR / "vp_portfolio.csv"
    resumo_vp.to_frame("valor").to_csv(arquivo_vp, sep=";", decimal=",", float_format="%.2f")
    print(f"\nValor presente dos dividendos futuros da carteira (taxa real {TAXA:.0%}, R$):")
    print(resumo_vp.round(0).to_string())
    print(f"Valor presente da carteira salvo em: {arquivo_vp}")
    figura_risco_retorno = FIG_DIR / "risco_retorno_portfolio.png"
    ponto = plotar_risco_retorno_portfolio(
        vp_amostras, empresas_multivariadas, quantidades_acoes, cotacoes,
        carteira_planilha, figura_risco_retorno,
        minimo_x=varianciaXlucro[0], limite_x=varianciaXlucro[1],
        minimo_lucro_mil=varianciaXlucro[2] / 1000, limite_lucro_mil=varianciaXlucro[3] / 1000,
    )
    print(f"Sua carteira: desvio padrão R$ {ponto['x_atual']:.1f} mil, "
          f"lucro mediano R$ {ponto['lucro_mediano_mil']:.1f} mil")
    print(f"Gráfico volatilidade x lucro salvo em: {figura_risco_retorno}")
    figura_variancia_retorno = FIG_DIR / "variancia_retorno_portfolio.png"
    plotar_risco_retorno_portfolio(
        vp_amostras, empresas_multivariadas, quantidades_acoes, cotacoes,
        carteira_planilha, figura_variancia_retorno, eixo_x="variancia",
        limite_x="dobro", limite_lucro_mil=1500,
    )
    print(f"Gráfico variância x lucro salvo em: {figura_variancia_retorno}")
    titulos_envelope = {
        "menor_volatilidade": "Carteira sobre o envelope de menor volatilidade",
        "maior_lucro": (
            "Carteira sobre o envelope de maior lucro "
            f"(volatilidade ≈ R$ {VARIANCIA_ALVO_ENVELOPE:,.0f} mil)"
        ),
    }
    for chave, carteira_sel in ponto["carteiras_envelope"].items():
        arquivo_sel = OUTPUT_DIR / f"carteira_envelope_{chave}.csv"
        carteira_sel["tabela"].to_csv(arquivo_sel, sep=";", decimal=",", float_format="%.2f")
        print(f"\n{titulos_envelope[chave]} (mesmo capital, sem venda a descoberto):")
        print(carteira_sel["tabela"].round(1).to_string())
        print(
            f"Volatilidade: R$ {carteira_sel['volatilidade_mil']:,.0f} mil; "
            f"lucro mediano (VP): R$ {carteira_sel['lucro_mediano_mil']:,.0f} mil "
            f"(sua carteira: R$ {ponto['x_atual']:,.0f} mil; R$ {ponto['lucro_mediano_mil']:,.0f} mil)"
        )
        figura_sel = FIG_DIR / f"carteira_envelope_{chave}_pizza.png"
        plotar_pizza_carteira_selecionada(
            carteira_sel["tabela"],
            f"{titulos_envelope[chave]}\n(volatilidade R$ {carteira_sel['volatilidade_mil']:,.0f} mil; "
            f"lucro mediano R$ {carteira_sel['lucro_mediano_mil']:,.0f} mil)",
            figura_sel,
        )
        print(f"Tabela salva em: {arquivo_sel}\nPizza salva em: {figura_sel}")
    gerar_grafico_legacy(ano_final=ano_final_plot)

    figura_pizza = FIG_DIR / "carteira_pizza_modelo.png"
    plotar_pizza_carteira(carteira_planilha, figura_pizza, ativos=list(TICKERS_YAHOO_ACOES))
    print(f"Pizza da carteira (ativos do modelo) salva em: {figura_pizza}")
