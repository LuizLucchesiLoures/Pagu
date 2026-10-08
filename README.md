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
6. Salva tabelas em `data_output/` e gráficos em `fig/FCL/`.

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

`True` liga a opção e `False` desliga. O número de empresas e os anos da
análise também são configurados no início do arquivo.

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
