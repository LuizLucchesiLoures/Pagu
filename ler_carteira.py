"""Lê células específicas da planilha Carteira.xlsx e salva um CSV."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook

BASE = Path(__file__).parent
ARQUIVO = BASE / "data_input" / "Carteira.xlsx"
ABA = "ACOES III"
LINHA_INICIAL, LINHA_FINAL = 5, 20
COLUNAS = {
    "B": "ativo",
    "E": "quantidade",
    "C": "preco_medio_compra",
    "J": "proventos_recebidos",
}
# Nomes da planilha que não são o ticker completo
APELIDOS = {"VALE": "VALE3"}
SAIDA = BASE / "data_output" / "carteira_lida.csv"
FIGURA_PIZZA = BASE / "fig" / "carteira_pizza.png"


def ler_carteira_excel(caminho=ARQUIVO, aba=ABA):
    """Lê só as colunas/linhas definidas e devolve um DataFrame sem linhas vazias."""
    ws = load_workbook(caminho, data_only=True, read_only=True)[aba]
    dados = {nome: [] for nome in COLUNAS.values()}
    for linha in range(LINHA_INICIAL, LINHA_FINAL + 1):
        for col, nome in COLUNAS.items():
            dados[nome].append(ws[f"{col}{linha}"].value)
    df = pd.DataFrame(dados).dropna(subset=["ativo"])
    df["ativo"] = df["ativo"].astype(str).str.strip().str.upper()
    df["ativo"] = df["ativo"].replace(APELIDOS)
    return df.reset_index(drop=True)


def plotar_pizza_carteira(carteira, caminho_saida=FIGURA_PIZZA, ativos=None):
    """Pizza com a composição da carteira (quantidade × preço médio por ativo).

    Se `ativos` for informado (tickers com ou sem ".SA"), só eles entram.
    """
    if ativos is not None:
        ativos = [a.replace(".SA", "") for a in ativos]
        carteira = carteira[carteira["ativo"].isin(ativos)]
    investido = (carteira["quantidade"] * carteira["preco_medio_compra"]).set_axis(carteira["ativo"])
    investido = investido[investido > 0].sort_values(ascending=False)
    figura, eixo = plt.subplots(figsize=(9, 9))
    eixo.pie(
        investido,
        labels=investido.index,
        autopct="%1.1f%%",
        colors=plt.cm.tab20.colors[: len(investido)],
        startangle=90,
        counterclock=False,
        pctdistance=0.8,
        wedgeprops={"edgecolor": "white"},
    )
    eixo.set_title(f"Composição da carteira (valor investido: R$ {investido.sum():,.0f})")
    caminho_saida.parent.mkdir(exist_ok=True)
    figura.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close(figura)


if __name__ == "__main__":
    carteira = ler_carteira_excel()
    SAIDA.parent.mkdir(exist_ok=True)
    carteira.to_csv(SAIDA, index=False, decimal=",", sep=";")
    print(carteira.to_string())
    print(f"\nSalvo em {SAIDA}")
    plotar_pizza_carteira(carteira)
    print(f"Pizza salva em {FIGURA_PIZZA}")
