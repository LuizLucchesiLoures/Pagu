# Pagu
Inferenica Bayesiana na otimizacao e controle de portfolio

## Análise bayesiana de carteira

Este repositório contém uma análise de Fluxo de Caixa Livre (FCL) de ações
brasileiras. O programa principal é `Portfolio.py`. Ele usa dados tabulares
guardados em `data_input/`, salva tabelas calculadas em `data_output/` e
salva gráficos em `fig/FCL/`.

## Estrutura do projeto

```text
PAGU/
├── Portfolio.py       # programa principal
├── ler_carteira.py   # lê a planilha da carteira, resume lucro e faz a pizza
├── backtest_dividendos.py  # teste retrospectivo: ajusta até 2022, prevê 2023-2025
├── requirements.txt   # bibliotecas Python necessárias
├── data_input/        # dados de entrada; não são alterados pelo programa
├── data_output/       # tabelas produzidas ao executar o programa
└── fig/
    └── FCL/           # gráficos produzidos ao executar o programa
```

Outros scripts de estudo (exemplos do PyMC, dashboard etc.) não fazem parte
deste projeto e ficam fora desta pasta.

## Preparar o ambiente

É recomendado usar Python 3.11 ou mais recente. No terminal, dentro da pasta
do projeto, crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows, a ativação é:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências e execute:

```bash
python -m pip install -r requirements.txt
python Portfolio.py
```

A execução consulta serviços externos para obter a quantidade de ações no
Yahoo Finance e, quando disponível, a série mensal do INPC no Banco Central.
Por isso, é necessária uma conexão com a internet. O arquivo local
`data_input/INPC_anual.csv` serve como alternativa quando a API do Banco
Central falha.

## Arquivos de entrada

O programa espera estes arquivos dentro de `data_input/`:

| Arquivo | Conteúdo |
| --- | --- |
| `FCL.csv` | FCL histórico por empresa e ano |
| `Capex_Expancao.csv` | Capex de expansão usado para ajustar o FCL |
| `FCL_futuro_projetado.csv` | projeções futuras opcionais por empresa |
| `proventos.csv` | proventos anuais por ação |
| `proventos_totais.csv` | dividendo total pago por ano, em R$ bilhões (opcional; colunas separadas por espaço ou Tab; `NaN` = usa proventos por ação × ações atuais) |
| `INPC_anual.csv` | INPC anual local, usado como alternativa à API |
| `Carteira.xlsx` | sua planilha pessoal (aba `ACOES III`); só as células B5:B20 (ativo), C5:C20 (preço médio), E5:E20 (quantidade) e J5:J20 (proventos recebidos) são lidas. Não vai para o GitHub (está no `.gitignore`) |

Os quatro primeiros arquivos usam colunas separadas por tabulação. `FCL.csv` e
`Capex_Expancao.csv` usam vírgula como separador decimal; os valores de
`FCL_futuro_projetado.csv` e `proventos.csv` usam ponto decimal. A primeira
coluna identifica o ticker da empresa e as demais colunas identificam anos.
Os tickers e anos devem corresponder entre os arquivos.
`INPC_anual.csv` tem uma coluna `ano` e uma coluna `INPC`.

Os arquivos de entrada originais são preservados. Resultados recalculados,
como o FCL corrigido pelo INPC e os relatórios de payout, ficam em
`data_output/`.

## O que o programa faz

1. Consulta a quantidade de ações em circulação de cada ticker no Yahoo
   Finance.
2. Obtém o INPC mensal do Banco Central, calcula a variação anual e usa o CSV
   local como alternativa se a consulta falhar.
3. Lê o FCL, o Capex, as projeções futuras e os proventos. Se a opção
   correspondente estiver ligada, soma o Capex ao FCL e corrige os valores
   históricos pela inflação medida pelo INPC.
4. Ajusta modelos bayesianos ao FCL histórico e simula valores futuros. O
   modelo multivariado também estima correlações entre os erros das empresas.
5. Calcula o payout histórico usando proventos, quantidade atual de ações e
   FCL; opcionalmente ajusta um modelo bayesiano para estimar o payout médio.
6. Infere os dividendos futuros: multiplica, amostra a amostra,
   o FCL futuro (preditiva do modelo multivariado) pelo payout de um ano novo
   (preditiva log-normal). A chave `USAR_PAYOUT_INDEPENDENTE_FCL` (início do
   programa) escolhe: `True` = payout independente do FCL; `False` = payout
   depende do FCL, com `log(payout) = mu + gamma·(log FCL − centro)` e
   `gamma` estimado por empresa (gamma < 0: FCL menor, payout maior).
   Resultado em `dividendos_futuros_posterior_independente.csv` ou
   `..._dependente.csv` (R$ bi e por ação, P10/P50/P90) e em uma figura com as 4 ações
   (`fig/FCL/dividendos_todas_<modo>.png`: pontos observados e curvas P10,
   mediana e P90 ao longo dos anos configurados); valores a preços do último ano do INPC.
7. Desconta os dividendos futuros a valor presente (VP) com a taxa real
   `TAXA` e a inflação `INFLACAO`, busca as cotações no Yahoo e calcula, por
   ação, o VP (P10/P50/P90), a probabilidade de o VP superar
   `FRACAO_COTACAO` × cotação e a cotação-alvo para `PROB_ALVO`. Gera a tabela
   `valor_presente_dividendos_<modo>.png` e a ilustração
   `valor_presente_distribuicao_<modo>.png`.
8. Cruza o modelo com a sua carteira (`Carteira.xlsx`): compara o VP por ação
   com o seu preço médio (`comparacao_carteira.csv`) e calcula os dividendos
   esperados por ativo e ano, mediana × quantidade
   (`dividendos_carteira_por_ano.csv`, tabela e gráfico empilhado
   `dividendos_carteira_grafico.png`), além da pizza dos ativos do modelo.
9. Salva tabelas em `data_output/` e gráficos em `fig/FCL/`.

### Resumo da carteira (`ler_carteira.py`)

Roda sozinho, em segundos e sem o modelo bayesiano:

```bash
.venv/bin/python ler_carteira.py
```

Gera `data_output/carteira_lida.csv`, `data_output/resumo_carteira.csv`,
`fig/carteira_pizza.png` e `fig/resumo_carteira.png` (por ativo: quantidade,
preço médio, cotação, investido, valor atual, lucro, dividendos e lucro com
dividendos, em R$ e em %). O dicionário `APELIDOS` converte nomes da planilha
em tickers do Yahoo (por exemplo `GRO` → `GROP31`).

### Como o código está organizado

- `criar_anos_projecao` monta a sequência de anos a partir das opções do
  início do programa. O ano final é incluído na sequência.
- `ler_dados_entrada` lê e padroniza os arquivos CSV de FCL, capex, projeções
  futuras e proventos por ação.
- `preparar_dados_fcl` aplica o ajuste opcional de capex e calcula a série de
  FCL corrigida pelo INPC.
- `buscar_quantidade_acoes_yahoo` e `ler_inpc_anual` cuidam, respectivamente,
  da consulta de ações e da leitura do INPC.
- `ajustar_modelo_multivariado` ajusta os parâmetros de FCL e
  `extrair_amostras_covariancia` recupera as amostras usadas nas previsões.
- `simular_previsoes_multivariadas` propaga essas amostras para os anos
  projetados, mantendo a correlação entre empresas.
- `construir_modelo_bayesiano` ajusta o modelo individual e prepara seus
  gráficos quando essa opção está selecionada.
- `calcular_payout_historico`, `ajustar_modelo_payout` e
  `ajustar_modelo_payout_dependente` preparam os payouts e ajustam os modelos
  de payout.
- `simular_dividendos_futuros`, `resumir_dividendos` e `plotar_dividendos`
  transformam as previsões de FCL e payout em distribuições de dividendos,
  resumos e gráficos.
- `calcular_vp_amostras`, `valor_presente_dividendos`,
  `plotar_tabela_valor_presente` e `plotar_distribuicao_vp` calculam e
  ilustram o valor presente e as probabilidades.
- `comparar_com_carteira`, `dividendos_anuais_carteira`,
  `plotar_tabela_dividendos_carteira` e `plotar_dividendos_carteira` ligam o
  modelo à sua carteira. `ler_carteira_excel`, `buscar_cotacoes_yahoo`,
  `resumir_carteira` e `plotar_pizza_carteira` ficam em `ler_carteira.py`.

### Vocabulário bayesiano em poucas palavras

- **Prior (priori):** o que o modelo considera plausível para um parâmetro
  antes de observar os dados.
- **Likelihood (verossimilhança):** o quanto os dados observados combinam com
  um valor proposto para os parâmetros.
- **Posterior (posteriori):** a distribuição atualizada dos parâmetros depois
  de combinar a priori com os dados.
- **Alpha (`alpha`):** nível médio estimado do FCL.
- **Beta (`beta`):** tendência estimada do FCL ao longo do tempo; anos são
  padronizados para facilitar o ajuste.
- **Sigma (`sigma`):** tamanho típico das diferenças entre os dados e a
  tendência estimada.
- **P10, médio e P90:** percentil 10, média e percentil 90 das simulações.
  Não são garantias de resultado; representam a distribuição produzida pelo
  modelo e dependem dos dados e das hipóteses escolhidas.

## Opções principais

No início de `Portfolio.py` existem opções que controlam a análise:

- `USAR_FCL_CORRIGIDO_capex_expandido`: soma o Capex de expansão ao FCL.
- `USAR_FCL_CORRIGIDO_INPC`: usa o FCL corrigido pela inflação.
- `USAR_FCL_FUTURO_PROJETADO`: inclui os valores do arquivo de projeções
  futuras.
- `USAR_EMPRESAS_MUTLIVARIADAS`: ajusta o modelo multivariado com correlação
  entre empresas.
- `USAR_PAYOUT_BAYESIANO`: estima o payout médio com um modelo bayesiano.

`True` liga a opção e `False` desliga. Também no início do arquivo:

- `ano_inicial` e `ano_final`: período da análise (o padrão vai até 2064).
- `TAXA` (taxa **real** de desconto) e `INFLACAO`: só a `TAXA` altera o VP.
- `FRACAO_COTACAO` e `PROB_ALVO`: limiar do VP (fração da cotação) e
  probabilidade desejada; são valores de exemplo, ajuste ao seu critério.
- `TICKERS_YAHOO_ACOES`: **lista única de ativos** da análise. As empresas de
  todo o programa saem das suas chaves. Para incluir um ativo, acrescente uma
  linha e preencha os dados dele em `data_input/` (FCL, capex, projeções e
  proventos). Se faltar histórico, o programa avisa e deixa o ativo de fora
  das tabelas da carteira.

Os resultados são estimativas de um modelo, não recomendação de investimento.

## Git e repositório remoto

Esta pasta já possui um repositório Git local. Para conferir o que será
incluído:

```bash
git status --short
```

Adicione e registre os arquivos do projeto:

```bash
git add .gitignore README.md requirements.txt Portfolio.py data_input
git add data_output/.gitkeep fig/FCL/.gitkeep
git commit -m "Organiza projeto de análise de carteira"
```

Para publicar no GitHub, crie um repositório vazio e configure a URL fornecida
pelo GitHub como `origin` antes de enviar os commits. Não envie `.venv/`, resultados gerados nem dados privados.

## Dashboard

`dashboard.py` junta tudo em uma página HTML (`data_output/dashboard.html`, não versionada, pois tem dados pessoais): cartões-resumo, tabelas ordenáveis, gráficos e abas Carteira, Dividendos, Valor presente e Modelo. Não recalcula nada; rode antes `Portfolio.py` e `ler_carteira.py`.

```bash
.venv/bin/python dashboard.py   # depois abra data_output/dashboard.html no navegador
```
