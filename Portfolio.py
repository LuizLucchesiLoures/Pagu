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


def criar_anos_projecao(ano_inicial, ano_final):
    """Cria a sequência inclusiva de anos definida nas opções do programa."""
    return np.arange(ano_inicial, ano_final + 1)


ano_inicial = 2016
ano_final = 2064
ano_final_plot = 2064
TAXA = 0.06 #taxa de juros livre de riscos (real, acima da inflação)
INFLACAO = 0.04 #inflação anual esperada de longo prazo
FRACAO_COTACAO = 0.7 #limiar do VP: fração da cotação atual (1.0 = a própria cotação)
anos_projecao = criar_anos_projecao(ano_inicial, ano_final)
empresas = ["ITUB3.SA", "FESA4.SA", "EGIE3.SA", "VALE3.SA"]
empresas_multivariadas = empresas
USAR_FCL_CORRIGIDO_capex_expandido = True
USAR_FCL_FUTURO_PROJETADO = True
USAR_FCL_CORRIGIDO_INPC = True
USAR_EMPRESAS_MUTLIVARIADAS = True
USAR_PAYOUT_BAYESIANO = True
# True: payout independente do FCL; False: log(payout) depende do log(FCL).
USAR_PAYOUT_INDEPENDENTE_FCL = False
TICKERS_YAHOO_ACOES = {
    "ITUB3.SA": ("ITUB3.SA", "ITUB4.SA"),
    "FESA4.SA": ("FESA3.SA", "FESA4.SA"),
    "EGIE3.SA": ("EGIE3.SA",),
    "VALE3.SA": ("VALE3.SA",),
}
def buscar_quantidade_acoes_yahoo(tickers):
    """Consulta no Yahoo Finance as quantidades de ações dos tickers."""
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


def ler_dados_entrada(diretorio_dados):
    """Lê os CSVs usados na análise e padroniza os nomes de anos e tickers.

    Retorna FCL, capex de expansão, FCL projetado e proventos por ação,
    mantendo a ordem esperada pelo restante do programa.
    """
    opcoes_csv = {
        "sep": "\t",
        "index_col": 0,
    }
    fcl = pd.read_csv(diretorio_dados / "FCL.csv", decimal=",", **opcoes_csv)
    capex_expansao = pd.read_csv(
        diretorio_dados / "Capex_Expancao.csv",
        decimal=",",
        **opcoes_csv,
    )
    fcl_futuro_projetado = pd.read_csv(
        diretorio_dados / "FCL_futuro_projetado.csv",
        decimal=",",
        **opcoes_csv,
    )
    proventos_anuais = pd.read_csv(
        diretorio_dados / "proventos.csv",
        decimal=",",
        **opcoes_csv,
    )

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


def ajustar_modelo_multivariado(dados_fcl, usar_correlacao):
    """Ajusta o modelo bayesiano conjunto aos FCLs das empresas.

    Cada empresa tem intercepto e tendência próprios. A opção
    `usar_correlacao` controla se os resíduos compartilham correlações.
    Retorna os resultados MCMC e os arrays usados nas previsões.
    """
    anos = dados_fcl.index.astype(float).to_numpy()
    fcl_observado = dados_fcl.to_numpy(dtype=float)
    anos_padronizados = (anos - anos.mean()) / anos.std()
    numero_empresas = dados_fcl.shape[1]

    with pm.Model() as modelo:
        alpha = pm.Normal(
            "alpha",
            mu=fcl_observado.mean(axis=0),
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
        pm.MvNormal(
            "FCL_observado",
            mu=media_fcl,
            chol=chol,
            observed=fcl_observado,
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


def buscar_cotacoes_yahoo(tickers):
    """Busca no Yahoo o último fechamento disponível (R$) de cada ticker."""
    cotacoes = {}
    for ticker in tickers:
        try:
            fechamentos = yf.Ticker(ticker).history(period="5d")["Close"].dropna()
            cotacoes[ticker] = float(fechamentos.iloc[-1])
        except Exception as erro:
            print(f"Não foi possível obter a cotação de {ticker}: {erro}")
            cotacoes[ticker] = np.nan
    return pd.Series(cotacoes, dtype=float)


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
    taxa_nominal = (1 + taxa) * (1 + inflacao) - 1
    expoente = anos[futuro] - ano_base
    fatores = ((1 + inflacao) / (1 + taxa_nominal)) ** expoente
    vp_amostras = np.nansum(
        dividendos_amostras[:, futuro, :] * fatores[None, :, None], axis=1
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
                "prob_vp_maior_fracao": (
                    np.mean(amostras / acoes_bilhoes > fracao_cotacao * cotacao)
                    if np.isfinite(cotacao)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(linhas)


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
        "(Prob. > Limiar = chance de o VP por ação superar o limiar)"
    )
    por_acao = valor_presente[valor_presente["ticker"] != "CARTEIRA"]
    tabela = pd.DataFrame(
        {
            "Empresa": por_acao["ticker"].str.replace(".SA", "", regex=False),
            "Cotação (R$)": por_acao["cotacao"].map("{:,.2f}".format),
            "VP P10 (R$)": por_acao["vp_por_acao_p10"].map("{:,.2f}".format),
            "VP P50 (R$)": por_acao["vp_por_acao_p50"].map("{:,.2f}".format),
            "VP P90 (R$)": por_acao["vp_por_acao_p90"].map("{:,.2f}".format),
            "VP/Cotação": por_acao["vp_sobre_cotacao_p50"].map(
                "{:.2f}x".format
            ),
            "Prob. > Limiar": por_acao["prob_vp_maior_fracao"].map(
                "{:.0%}".format
            ),
        }
    )
    figura, eixo = plt.subplots(figsize=(10, 0.6 * len(tabela) + 1.6))
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
    figura, eixos = plt.subplots(2, 2, figsize=(14, 8), squeeze=False)
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
        eixo.set_xlim(ano_inicial - 0.5, ano_final + 0.5)
        eixo.set_xticks(range(ano_inicial, ano_final + 1, 2))
        eixo.set_title(f"{empresa} - dividendos até {ano_final}")
        eixo.set_xlabel("Ano")
        eixo.set_ylabel(f"R$ bi (preços de {ano_base_fcl})")
        eixo.grid(alpha=0.3)
        eixo.legend(fontsize=8, framealpha=0.6)

    figura.suptitle(
        f"Dividendos totais: observados e P10/mediana/P90 do modelo, "
        f"{ano_inicial}-{ano_final} (payout {modo} do FCL)"
    )
    figura.tight_layout()
    figura.savefig(caminho_saida, dpi=200)
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
    figura_payout, eixos_payout = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        squeeze=False,
    )
    resumos_posteriores = []
    densidades_posteriores = []
    parametros_payout_posterior = {}
    parametros_payout_dependente = {}

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
    )
    arquivo_vp = OUTPUT_DIR / f"valor_presente_dividendos_{sufixo_modo}.csv"
    valor_presente.to_csv(arquivo_vp, index=False, decimal=",", float_format="%.3f")
    print(f"\nValor presente dos dividendos (taxa real {TAXA:.1%}, inflação {INFLACAO:.1%}, nominal {(1 + TAXA) * (1 + INFLACAO) - 1:.2%}, base {ultimo_ano_observado}, R$ bi):")
    print(valor_presente.round(2).to_string(index=False))
    print(f"Valor presente salvo em: {arquivo_vp}")
    figura_vp = FIG_DIR / f"valor_presente_dividendos_{sufixo_modo}.png"
    plotar_tabela_valor_presente(valor_presente, figura_vp)
    print(f"Tabela do valor presente salva em: {figura_vp}")
