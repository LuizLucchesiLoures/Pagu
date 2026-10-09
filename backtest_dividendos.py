"""Backtest dos dividendos: ajusta os modelos só com dados até 2022 e compara a
previsão de 2023-2025 com os dividendos que realmente ocorreram.

Compara payout independente do FCL x dependente do FCL, usando as mesmas
fórmulas do Portfolio.py. Tudo em R$ bilhões a preços de 2025 (INPC).
Não depende de internet: lê apenas data_input/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data_output"
FIG_DIR = BASE_DIR / "fig" / "FCL"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EMPRESAS = ["ITUB3.SA", "FESA4.SA", "EGIE3.SA", "VALE3.SA"]
ANOS_TREINO = list(range(2016, 2023))
ANOS_TESTE = [2023, 2024, 2025]
SEMENTE = 7


def ler_tabela(nome, decimal):
    tabela = pd.read_csv(BASE_DIR / "data_input" / nome, sep=r"\s+|\t", decimal=decimal,
                         index_col=0, engine="python")
    tabela.columns = [int(str(c).strip()[:4]) for c in tabela.columns]
    tabela.index = tabela.index.str.strip()
    return tabela.reindex(EMPRESAS).apply(pd.to_numeric, errors="coerce")


# Dados: mesmos critérios do Portfolio.py (FCL + Capex de expansão).
fcl_nominal = ler_tabela("FCL.csv", ",").add(ler_tabela("Capex_Expancao.csv", ","), fill_value=0)
dividendos_nominais = ler_tabela("proventos_totais.csv", ".")
if dividendos_nominais.isna().any().any():
    raise SystemExit("Preencha todo o proventos_totais.csv para rodar o backtest.")

inpc = pd.read_csv(BASE_DIR / "data_input" / "INPC_anual.csv", decimal=",").set_index("ano")["INPC"]
ano_base = int(inpc.index.max())
fator = pd.Series(
    {a: np.prod(1 + inpc.loc[a + 1 : ano_base].to_numpy() / 100) for a in range(2016, ano_base + 1)}
)
fcl_real = fcl_nominal.mul(fator.reindex(fcl_nominal.columns).to_numpy(), axis="columns")
dividendos_reais = dividendos_nominais.mul(
    fator.reindex(dividendos_nominais.columns).to_numpy(), axis="columns"
)
payout_pct = dividendos_reais.div(fcl_real.where(fcl_real > 0)) * 100

# --- FCL: modelo multivariado ajustado só com o treino ------------------------
fcl_treino = fcl_real.loc[EMPRESAS, ANOS_TREINO].T.to_numpy(dtype=float)
anos_treino = np.array(ANOS_TREINO, dtype=float)
media_ano, desvio_ano = anos_treino.mean(), anos_treino.std()
n = len(EMPRESAS)

with pm.Model():
    alpha = pm.Normal("alpha", mu=fcl_treino.mean(axis=0), sigma=10, shape=n)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=n)
    chol, _, _ = pm.LKJCholeskyCov(
        "cov", n=n, eta=2, sd_dist=pm.Exponential.dist(1), compute_corr=True
    )
    pm.MvNormal(
        "fcl_obs",
        mu=alpha[None, :] + beta[None, :] * ((anos_treino - media_ano) / desvio_ano)[:, None],
        chol=chol,
        observed=fcl_treino,
    )
    idata_fcl = pm.sample(draws=1000, tune=1000, chains=2, cores=1, target_accept=0.9,
                          random_seed=SEMENTE, progressbar=False)

alpha_s = idata_fcl.posterior["alpha"].stack(s=("chain", "draw")).values  # (n, S)
beta_s = idata_fcl.posterior["beta"].stack(s=("chain", "draw")).values
cov_s = idata_fcl.posterior["cov"].stack(s=("chain", "draw")).values
n_amostras = alpha_s.shape[1]
rng = np.random.default_rng(SEMENTE)

anos_teste_pad = (np.array(ANOS_TESTE, dtype=float) - media_ano) / desvio_ano
fcl_prev = np.empty((n_amostras, len(ANOS_TESTE), n))
indices_tri = np.tril_indices(n)
for s in range(n_amostras):
    L = np.zeros((n, n))
    L[indices_tri] = cov_s[:, s]
    media = alpha_s[:, s][None, :] + beta_s[:, s][None, :] * anos_teste_pad[:, None]
    fcl_prev[s] = media + rng.standard_normal((len(ANOS_TESTE), n)) @ L.T
fcl_prev = np.clip(fcl_prev, 0, None)

# --- Payout: modelos independente e dependente, só com o treino -----------------
dividendos_prev = {"independente": np.full_like(fcl_prev, np.nan),
                   "dependente": np.full_like(fcl_prev, np.nan)}
gammas = {}

for i, empresa in enumerate(EMPRESAS):
    dados = pd.DataFrame({"payout": payout_pct.loc[empresa, ANOS_TREINO],
                          "fcl": fcl_real.loc[empresa, ANOS_TREINO]}).dropna()
    dados = dados[(dados.payout > 0) & (dados.fcl > 0)]
    y = dados["payout"].to_numpy(dtype=float)
    log_fcl = np.log(dados["fcl"].to_numpy(dtype=float))
    centro = log_fcl.mean()

    with pm.Model():
        mu = pm.Normal("mu", mu=np.log(50.0), sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.LogNormal("y", mu=mu, sigma=sigma, observed=y)
        idata_ind = pm.sample(draws=1000, tune=1000, chains=2, cores=1, target_accept=0.93,
                              random_seed=SEMENTE + i, progressbar=False)
    with pm.Model():
        mu = pm.Normal("mu", mu=np.log(50.0), sigma=1.0)
        gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.LogNormal("y", mu=mu + gamma * (log_fcl - centro), sigma=sigma, observed=y)
        idata_dep = pm.sample(draws=1000, tune=1000, chains=2, cores=1, target_accept=0.93,
                              random_seed=SEMENTE + 10 + i, progressbar=False)

    fcl_emp = fcl_prev[:, :, i]
    forma = fcl_emp.shape

    mu_i = idata_ind.posterior["mu"].values.flatten()
    sg_i = idata_ind.posterior["sigma"].values.flatten()
    sorteio = rng.integers(0, mu_i.size, n_amostras)
    payout_ind = pm.draw(
        pm.LogNormal.dist(mu=np.broadcast_to(mu_i[sorteio][:, None], forma),
                          sigma=np.broadcast_to(sg_i[sorteio][:, None], forma)),
        random_seed=int(rng.integers(1_000_000)),
    )
    dividendos_prev["independente"][:, :, i] = payout_ind / 100 * fcl_emp

    mu_d = idata_dep.posterior["mu"].values.flatten()
    ga_d = idata_dep.posterior["gamma"].values.flatten()
    sg_d = idata_dep.posterior["sigma"].values.flatten()
    gammas[empresa] = (ga_d.mean(), *np.percentile(ga_d, [5, 95]))
    sorteio = rng.integers(0, mu_d.size, n_amostras)
    log_fcl_prev = np.log(np.where(fcl_emp > 0, fcl_emp, 1.0))
    payout_dep = pm.draw(
        pm.LogNormal.dist(
            mu=mu_d[sorteio][:, None] + ga_d[sorteio][:, None] * (log_fcl_prev - centro),
            sigma=np.broadcast_to(sg_d[sorteio][:, None], forma),
        ),
        random_seed=int(rng.integers(1_000_000)),
    )
    dividendos_prev["dependente"][:, :, i] = payout_dep / 100 * fcl_emp


def crps(amostras, observado):
    """CRPS pelas amostras: E|X-y| - 0.5 E|X-X'| (menor é melhor)."""
    x = amostras[rng.permutation(amostras.size)[:2000]]
    return np.mean(np.abs(x - observado)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))


linhas = []
for modo, prev in dividendos_prev.items():
    for i, empresa in enumerate(EMPRESAS):
        for j, ano in enumerate(ANOS_TESTE):
            a = prev[:, j, i]
            obs = float(dividendos_reais.loc[empresa, ano])
            p10, p50, p90 = np.percentile(a, [10, 50, 90])
            linhas.append({
                "modo": modo, "ticker": empresa, "ano": ano, "observado": obs,
                "p10": p10, "p50": p50, "p90": p90,
                "dentro_p10_p90": p10 <= obs <= p90,
                "erro_abs_mediana": abs(p50 - obs),
                "crps": crps(a, obs),
            })
resultado = pd.DataFrame(linhas)
resultado.to_csv(OUTPUT_DIR / "backtest_dividendos_2023_2025.csv", index=False,
                 decimal=",", float_format="%.3f")

resumo = resultado.groupby(["modo", "ticker"]).agg(
    cobertura_p10_p90=("dentro_p10_p90", "mean"),
    erro_abs_mediana=("erro_abs_mediana", "mean"),
    crps=("crps", "mean"),
)
geral = resultado.groupby("modo").agg(
    cobertura_p10_p90=("dentro_p10_p90", "mean"),
    erro_abs_mediana=("erro_abs_mediana", "mean"),
    crps=("crps", "mean"),
)
pd.set_option("display.width", 200)
print("\nGamma estimado só com 2016-2022 (média, P5, P95):")
for empresa, g in gammas.items():
    print(f"  {empresa}: {g[0]:.2f} [{g[1]:.2f}, {g[2]:.2f}]")
print("\nBacktest 2023-2025 (R$ bi a preços de 2025). Cobertura ideal do P10-P90 = 80%.")
print(resumo.round(2).to_string())
print("\nGeral:")
print(geral.round(2).to_string())
print("\nDetalhe:")
print(resultado.round(2).to_string(index=False))

# --- Figura: observado x previsão em cada modo ----------------------------------
figura, eixos = plt.subplots(2, 2, figsize=(14, 8), squeeze=False)
for i, empresa in enumerate(EMPRESAS):
    eixo = eixos.flat[i]
    eixo.bar(ANOS_TREINO + ANOS_TESTE, dividendos_reais.loc[empresa, ANOS_TREINO + ANOS_TESTE],
             color="lightgray", width=0.7, label="Ocorrido")
    for deslocamento, (modo, cor) in zip((-0.15, 0.15), (("independente", "#0072b2"),
                                                       ("dependente", "#d55e00"))):
        t = resultado[(resultado.modo == modo) & (resultado.ticker == empresa)]
        x = t["ano"] + deslocamento
        eixo.errorbar(x, t["p50"], yerr=[t["p50"] - t["p10"], t["p90"] - t["p50"]],
                      fmt="o", color=cor, capsize=4, label=f"{modo}: mediana e P10-P90")
    eixo.axvline(2022.5, color="gray", linestyle=":")
    eixo.set_title(empresa)
    eixo.set_xticks(ANOS_TREINO + ANOS_TESTE)
    eixo.set_ylabel(f"R$ bi (preços de {ano_base})")
    eixo.grid(alpha=0.3, axis="y")
    eixo.legend(loc="upper left", fontsize=8)
figura.suptitle("Backtest: modelos ajustados até 2022 prevendo 2023-2025")
figura.tight_layout()
figura.savefig(FIG_DIR / "backtest_dividendos_2023_2025.png", dpi=200)
plt.close(figura)
print(f"\nFigura: {FIG_DIR / 'backtest_dividendos_2023_2025.png'}")
