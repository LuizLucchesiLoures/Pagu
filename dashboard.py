"""Gera data_output/dashboard.html a partir das saídas já calculadas.

Não recalcula nada: lê os CSVs de data_output/ e as figuras de fig/. Rode
Portfolio.py e ler_carteira.py antes para atualizar os dados.
"""
import base64
import io
from datetime import date
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).parent
SAIDAS = BASE / "data_output"
FIG = BASE / "fig"
ARQUIVO_HTML = SAIDAS / "dashboard.html"
ANO_META = 2064
VERDE, VERMELHO, AZUL = "#1a7f37", "#c0392b", "#2f5597"


def ler(nome, **kw):
    return pd.read_csv(SAIDAS / nome, sep=";", decimal=",", **kw)


def reais(x, casas=0):
    return f"R$ {x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x):
    return f"{x:+.1%}".replace(".", ",")


def imagem_png(caminho, legenda):
    dados = base64.b64encode(Path(caminho).read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{dados}" alt="{escape(legenda)}">'
            f"<figcaption>{escape(legenda)}</figcaption></figure>")


def figura_svg(figura):
    buffer = io.StringIO()
    figura.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(figura)
    svg = buffer.getvalue()
    return svg[svg.index("<svg"):]


def grafico_lucro(resumo):
    ativos = resumo.drop(index="TOTAL").sort_values("lucro_com_dividendos_pct")
    cores = [VERDE if v >= 0 else VERMELHO for v in ativos["lucro_com_dividendos_pct"]]
    figura, eixo = plt.subplots(figsize=(7, 0.45 * len(ativos) + 1))
    eixo.barh(ativos.index, ativos["lucro_com_dividendos_pct"] * 100, color=cores)
    eixo.axvline(0, color="black", linewidth=0.8)
    eixo.set_xlabel("Lucro com dividendos (% do investido)")
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    return figura_svg(figura)


def grafico_pizza_valor_atual(resumo):
    atual = resumo.drop(index="TOTAL")["valor_atual"].sort_values(ascending=False)
    figura, eixo = plt.subplots(figsize=(7, 7))
    eixo.pie(atual, labels=atual.index, autopct="%1.1f%%", startangle=90, counterclock=False,
             colors=plt.cm.tab20.colors[: len(atual)], pctdistance=0.8,
             wedgeprops={"edgecolor": "white"})
    eixo.set_title(f"Valor atual: {reais(atual.sum())}")
    return figura_svg(figura)


def grafico_acumulado(dividendos):
    por_ativo = dividendos.drop(index="TOTAL") / 1000
    acumulado = por_ativo.cumsum(axis=1)
    anos = acumulado.columns.astype(int)
    figura, eixo = plt.subplots(figsize=(9, 4.5))
    eixo.stackplot(anos, acumulado.values, labels=acumulado.index, alpha=0.9)
    eixo.set_xlim(anos.min(), anos.max())
    eixo.set_ylabel("Dividendos acumulados (R$ mil, preços de 2025)")
    eixo.set_xlabel("Ano")
    eixo.grid(alpha=0.3)
    eixo.legend(loc="upper left", ncol=2)
    eixo.annotate(f"{ANO_META}: {reais(acumulado.iloc[:, -1].sum() * 1000)}",
                  xy=(anos.max(), acumulado.iloc[:, -1].sum()), xytext=(-150, -35),
                  textcoords="offset points", arrowprops={"arrowstyle": "->"})
    return figura_svg(figura)


def tabela_html(df, formatos, classe_sinal=(), destacar_ultima=False):
    """Tabela HTML; clicar no cabeçalho ordena a coluna."""
    cabecalho = "".join(f"<th onclick='ordenar(this)'>{escape(c)}</th>" for c in df.columns)
    linhas = []
    for i, (_, linha) in enumerate(df.iterrows()):
        celulas = []
        for coluna, valor in linha.items():
            texto = formatos.get(coluna, str)(valor) if pd.notna(valor) else ""
            cor = ""
            if coluna in classe_sinal and pd.notna(valor):
                cor = " class='pos'" if valor >= 0 else " class='neg'"
            celulas.append(f"<td{cor}>{escape(texto)}</td>")
        total = " class='total'" if destacar_ultima and i == len(df) - 1 else ""
        linhas.append(f"<tr{total}>{''.join(celulas)}</tr>")
    return f"<table><thead><tr>{cabecalho}</tr></thead><tbody>{''.join(linhas)}</tbody></table>"


def cartao(titulo, valor, sub="", cor=AZUL):
    return (f"<div class='card' style='border-top-color:{cor}'><span>{escape(titulo)}</span>"
            f"<b>{escape(valor)}</b><small>{escape(sub)}</small></div>")


CSS = """
body{font-family:system-ui,sans-serif;margin:0;background:#f4f6fb;color:#1f2937}
header{background:linear-gradient(120deg,#1e3a8a,#2f5597 60%,#4f86c6);color:#fff;padding:24px 32px}
header h1{margin:0 0 4px;font-size:28px}header p{margin:0;opacity:.9}
.meta{margin-top:14px;max-width:520px}.barra{background:#ffffff40;border-radius:8px;height:10px}
.barra div{background:#fbbf24;height:10px;border-radius:8px}
main{max-width:1200px;margin:auto;padding:20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin-bottom:20px}
.card{background:#fff;border-radius:10px;padding:14px 16px;box-shadow:0 1px 4px #0001;border-top:4px solid}
.card span{display:block;font-size:13px;color:#6b7280}.card b{font-size:22px;display:block;margin:4px 0}
.card small{color:#6b7280}
nav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
nav button{border:0;background:#e5e9f4;padding:10px 18px;border-radius:20px;cursor:pointer;font-size:15px}
nav button.ativo{background:#2f5597;color:#fff}
section{display:none;background:#fff;border-radius:10px;padding:20px;box-shadow:0 1px 4px #0001}
section.ativo{display:block}
h2{margin-top:0}.grade{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}
figure{margin:0}img,svg{max-width:100%;height:auto}figcaption{font-size:13px;color:#6b7280}
.rolar{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th{background:#d9e2f3;cursor:pointer;padding:8px;white-space:nowrap}
td{padding:7px 8px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}
td:first-child,th:first-child{text-align:left;position:sticky;left:0;background:inherit}
tbody td:first-child{background:#fff}
.pos{color:#1a7f37}.neg{color:#c0392b}tr.total td{font-weight:bold;background:#f2f2f2}
.aviso{font-size:13px;color:#6b7280;margin-top:16px}
"""

JS = """
function aba(id,btn){document.querySelectorAll('section,nav button').forEach(e=>e.classList.remove('ativo'));
document.getElementById(id).classList.add('ativo');btn.classList.add('ativo');}
function ordenar(th){const t=th.closest('table'),i=[...th.parentNode.children].indexOf(th);
const corpo=t.tBodies[0],linhas=[...corpo.rows],total=linhas.filter(r=>r.classList.contains('total'));
const dados=linhas.filter(r=>!r.classList.contains('total')),asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
const num=s=>parseFloat(s.replace(/[^0-9,+\\-]/g,'').replace(',','.'));
dados.sort((a,b)=>{const x=a.cells[i].innerText,y=b.cells[i].innerText,nx=num(x),ny=num(y);
const c=(isNaN(nx)||isNaN(ny))?x.localeCompare(y):nx-ny;return asc?c:-c;});
[...dados,...total].forEach(r=>corpo.appendChild(r));}
"""


def montar():
    resumo = ler("resumo_carteira.csv", index_col=0)
    comparacao = ler("comparacao_carteira.csv")
    dividendos = ler("dividendos_carteira_por_ano.csv", index_col=0)
    vp = pd.read_csv(SAIDAS / "valor_presente_dividendos_dependente.csv", decimal=",")
    total = resumo.loc["TOTAL"]
    anos_div = dividendos.columns.astype(int)
    primeiro = dividendos.loc["TOTAL"].iloc[0]
    acumulado = dividendos.loc["TOTAL"].sum()
    carteira_vp = vp[vp["ticker"] == "CARTEIRA"].iloc[0]
    hoje = date.today()
    anos_restantes = ANO_META - hoje.year
    progresso = 100 * (hoje.year - 1964) / (ANO_META - 1964)

    cartoes = "".join([
        cartao("Investido", reais(total["investido"])),
        cartao("Valor atual", reais(total["valor_atual"])),
        cartao("Lucro", reais(total["lucro"]), pct(total["lucro_pct"]),
               VERDE if total["lucro"] >= 0 else VERMELHO),
        cartao("Dividendos recebidos", reais(total["dividendos"])),
        cartao("Lucro com dividendos", reais(total["lucro_com_dividendos"]),
               pct(total["lucro_com_dividendos_pct"]), VERDE),
        cartao(f"Dividendos esperados em {anos_div.min()}", reais(primeiro),
               "mediana, 4 ativos do modelo, preços de 2025", "#b8860b"),
        cartao(f"Acumulado até {anos_div.max()}", reais(acumulado),
               "preços de 2025, sem desconto", "#b8860b"),
        cartao("VP dos dividendos (4 ativos)", f"R$ {carteira_vp['vp_p50_rs_bi']:,.0f} bi".replace(",", "."),
               "mediana do modelo, empresas inteiras", "#7c3aed"),
    ])

    tab_resumo = tabela_html(
        resumo.reset_index().rename(columns={
            "ativo": "Ativo", "quantidade": "Qtd.", "preco_medio_compra": "Preço médio",
            "cotacao": "Cotação", "investido": "Investido", "valor_atual": "Valor atual",
            "dividendos": "Dividendos", "lucro": "Lucro", "lucro_pct": "Lucro %",
            "lucro_com_dividendos": "Lucro+div.", "lucro_com_dividendos_pct": "Lucro+div. %"}),
        {"Qtd.": "{:,.0f}".format, "Preço médio": "{:,.2f}".format, "Cotação": "{:,.2f}".format,
         "Investido": reais, "Valor atual": reais, "Dividendos": reais, "Lucro": reais,
         "Lucro %": pct, "Lucro+div.": reais, "Lucro+div. %": pct},
        classe_sinal=("Lucro", "Lucro %", "Lucro+div.", "Lucro+div. %"), destacar_ultima=True)

    tab_div = tabela_html(
        (dividendos / 1000).round(1).reset_index().rename(columns={"ativo": "Ativo"}),
        {str(a): "{:,.1f}".format for a in anos_div}, destacar_ultima=True)

    tab_comp = tabela_html(
        comparacao[["ativo", "preco_medio", "cotacao", "vp_acao_p10", "vp_acao_p50", "vp_acao_p90",
                    "vp_sobre_preco_medio", "prob_vp_maior_preco_medio"]].rename(columns={
            "ativo": "Ativo", "preco_medio": "Preço médio", "cotacao": "Cotação",
            "vp_acao_p10": "VP P10", "vp_acao_p50": "VP P50", "vp_acao_p90": "VP P90",
            "vp_sobre_preco_medio": "VP ÷ preço médio",
            "prob_vp_maior_preco_medio": "Prob. VP > preço médio"}),
        {c: "{:,.2f}".format for c in ["Preço médio", "Cotação", "VP P10", "VP P50", "VP P90", "VP ÷ preço médio"]}
        | {"Prob. VP > preço médio": "{:.0%}".format})

    figuras_modelo = "".join(
        imagem_png(FIG / "FCL" / arquivo, legenda)
        for arquivo, legenda in [
            ("FCL_graficos_multivariado.png", "FCL: histórico e previsão"),
            ("matriz_correlacao_posterior.png", "Correlação entre empresas"),
            ("payout_densidade_posterior.png", "Payout: densidade posterior"),
            ("dividendos_todas_dependente.png", "Dividendos totais por empresa (P10/P50/P90)"),
        ] if (FIG / "FCL" / arquivo).exists())

    abas = [
        ("carteira", "Carteira", f"""
<h2>Resumo da carteira</h2><div class='rolar'>{tab_resumo}</div>
<div class='grade' style='margin-top:20px'>
<div><h3>Lucro por ativo</h3>{grafico_lucro(resumo)}</div>
<div><h3>Composição (valor atual)</h3>{grafico_pizza_valor_atual(resumo)}</div></div>"""),
        ("dividendos", "Dividendos esperados", f"""
<h2>Dividendos esperados por ativo e ano (R$ mil)</h2>
<p>Mediana do dividendo por ação × sua quantidade, a preços de 2025.</p>
{imagem_png(FIG / 'FCL' / 'dividendos_carteira_grafico.png', 'Dividendos por ano, empilhado')}
<h3>Legacy × modelo × realizado</h3>
<p>Legacy é a estimativa antiga da planilha; realizado = dividendos + juros (aba resumo, 2026 parcial).</p>
{imagem_png(FIG / 'legacy_dividendos.png', 'Dividendos por ano: Legacy, modelo e realizado')}
<h3>Acumulado até {anos_div.max()}</h3>{grafico_acumulado(dividendos)}
<h3>Tabela completa</h3><div class='rolar'>{tab_div}</div>"""),
        ("vp", "Valor presente", f"""
<h2>Quanto vale o fluxo de dividendos hoje?</h2>
<p>VP por ação (R$) contra o seu preço médio. Clique nos cabeçalhos para ordenar.</p>
<div class='rolar'>{tab_comp}</div>
<div class='grade' style='margin-top:20px'>
{imagem_png(FIG / 'FCL' / 'valor_presente_dividendos_dependente.png', 'Tabela de valor presente')}
{imagem_png(FIG / 'FCL' / 'valor_presente_distribuicao_dependente.png', 'Distribuição do valor presente')}</div>"""),
        ("modelo", "Modelo", f"<h2>Por dentro do modelo bayesiano</h2><div class='grade'>{figuras_modelo}</div>"),
    ]
    botoes = "".join(
        f"<button class='{'ativo' if i == 0 else ''}' onclick=\"aba('{id}',this)\">{nome}</button>"
        for i, (id, nome, _) in enumerate(abas))
    secoes = "".join(
        f"<section id='{id}' class='{'ativo' if i == 0 else ''}'>{conteudo}</section>"
        for i, (id, _, conteudo) in enumerate(abas))

    return f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pagu — Dashboard</title>
<style>{CSS}</style></head><body>
<header><h1>Pagu · Dashboard da carteira</h1>
<p>Atualizado em {hoje:%d/%m/%Y} · análise bayesiana de dividendos até {anos_div.max()}</p>
<div class="meta"><small>Rumo a {ANO_META}: faltam {anos_restantes} anos</small>
<div class="barra"><div style="width:{progresso:.0f}%"></div></div></div></header>
<main><div class="cards">{cartoes}</div><nav>{botoes}</nav>{secoes}
<p class="aviso">Estimativas de um modelo, não recomendação de investimento. Dividendos a preços de 2025.
Dados pessoais: este arquivo não é versionado no Git.</p></main>
<script>{JS}</script></body></html>"""


if __name__ == "__main__":
    ARQUIVO_HTML.write_text(montar(), encoding="utf-8")
    print(f"Dashboard salvo em: {ARQUIVO_HTML}")
