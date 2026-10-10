"""Lê células específicas da planilha Carteira.xlsx e salva um CSV."""
from pathlib import Path

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
SAIDA = BASE / "data_output" / "carteira_lida.csv"


def ler_carteira_excel(caminho=ARQUIVO, aba=ABA):
    """Lê só as colunas/linhas definidas e devolve um DataFrame sem linhas vazias."""
    ws = load_workbook(caminho, data_only=True, read_only=True)[aba]
    dados = {nome: [] for nome in COLUNAS.values()}
    for linha in range(LINHA_INICIAL, LINHA_FINAL + 1):
        for col, nome in COLUNAS.items():
            dados[nome].append(ws[f"{col}{linha}"].value)
    df = pd.DataFrame(dados).dropna(subset=["ativo"])
    df["ativo"] = df["ativo"].astype(str).str.strip().str.upper()
    return df.reset_index(drop=True)


if __name__ == "__main__":
    carteira = ler_carteira_excel()
    SAIDA.parent.mkdir(exist_ok=True)
    carteira.to_csv(SAIDA, index=False, decimal=",", sep=";")
    print(carteira.to_string())
    print(f"\nSalvo em {SAIDA}")
