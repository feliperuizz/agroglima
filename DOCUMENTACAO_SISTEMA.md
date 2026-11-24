# AgroClima — Documentação Técnica Completa (função por função)

> **Para a IA que está lendo este arquivo:** este documento descreve integralmente o sistema **AgroClima**, um TCC (Mackenzie) de recomendação de período de plantio usando dados meteorológicos históricos do INMET e Machine Learning (Random Forest). Ele foi escrito a partir da leitura direta de todo o código-fonte. Tudo que está aqui reflete o comportamento **real** do código, inclusive limitações e inconsistências (seção 13). Use-o como fonte de verdade para responder perguntas, sugerir melhorias, escrever o texto do TCC ou modificar o código. Nomes de funções, variáveis, arquivos e números estão exatamente como no código.

---

## ⚠ 0. Versão 1.1 (27/10/2026): o que mudou e PREVALECE sobre o restante

As seções 1 a 13 descrevem a **versão 1.0**, a que foi auditada no TCC. Depois do parecer da banca, o sistema foi corrigido. Quando houver conflito, vale esta seção.

**Pré-processamento (`api/preprocess.py`)**
- Cada CSV é lido uma única vez (`list_csv_files`). Antes, `*.CSV` + `*.csv` no Windows liam tudo em dobro e geravam 17.640 linhas duplicadas.
- O parser reconhece os dois formatos do INMET: até 2019 (`DATA (YYYY-MM-DD)`, `00:00`, ausente = -9999) e a partir de 2020 (`Data` em AAAA/MM/DD, `Hora UTC` como `0000 UTC`, metadados sem acento `ESTACAO`/`REGIAO`, ausente = campo vazio). Funções novas: `normalize`, `map_columns` e `to_numeric` (vetorizada).
- Faixa física (`LIMITES_FISICOS`): chuva fora de 0–150 mm/h, temperaturas fora de −15–50 °C e umidade fora de 0–100% viram ausentes. A contagem fica em `n_fora_faixa`.
- Completude **conjunta** (`apply_completeness`): o registro estação-ano-mês inteiro só entra se temperatura, chuva e umidade tiverem, cada uma, **≥ 80% das horas do mês**, contadas com os dias exatos do mês (`horas_esperadas`). Por isso nenhum indicador da base 1.1 tem variável ausente, a nota 0,5 nunca é usada e `n_anos` é igual para as três variáveis.
- Chuva escalada: `precip_total = precip_soma × horas_esperadas / n_precip`. A soma bruta fica em `precip_soma`. Chuva sem nenhuma hora válida fica `NaN`, não 0 (`sum(min_count=1)`).
- Lacunas seguidas: além dos 80%, nenhuma das três variáveis pode ter mais de **72 h seguidas** sem leitura no mês (`GAP_MAX_HORAS`, função `max_gaps`). A série é completada hora a hora a partir de `data` + `hora`.
- Estação vizinha (só em `main.py`, na hora da consulta, decidida mês a mês): candidatas com indicador no mês, **`n_anos` ≥ 3** e **desnível ≤ 500 m**. Vence a mais próxima por Haversine (raio de 6.371 km), até 150 km; empate fica com quem tem mais anos. As temperaturas são **corrigidas em 6,5 °C/km** pela diferença de altitude.
- Média geral de vários meses: os meses sem dado ficam fora, e os **inviáveis entram com 0**. O cartão de cada mês continua mostrando o próprio S.
- Saídas novas: `data/monthly_raw.pkl` (meses antes do filtro, com contagens `n_horas`, `n_precip` e `n_umid`) e `data/preprocess_summary.json`. Os dados da 1.0 foram preservados em `data/v1_0/`.
- Base atual: **5.755/5.755 arquivos lidos**, 12 anos (2005, 2007-2009, 2013, 2020-2026), **640 estações**, 40.476 meses completos (de 61.640) e **7.118 indicadores**. Cada indicador tem de 1 a 12 anos (mediana 6).

**Regra e modelo (`api/train_model.py`)**
- A regra continua a mesma (pesos 35/40/25, bônus de 1,1 e penalidades). A nova `score_breakdown(row, crop)` devolve `s_temp`, `s_precip`, `s_umid`, `base`, `bonus`, `pen_precip`, `pen_temp`, `score` e `veto` (`veto = True` quando alguma nota parcial é 0).
- O treino usa `GroupKFold` por estação (5 partes, com um RF novo em cada parte) mais uma linha de base (média de S por cultura × mês) e grava `api/models/metrics.json`. `RF_PARAMS` explicita `max_features=1.0`.
- O RF **não é mais usado pela API**. É mantido só como objeto de avaliação.

**Avaliação (`api/avaliacao.py` → `api/models/avaliacao.json`)**
- Compara três bases: D0 (1.0 duplicada), D1 (1.0 sem duplicatas) e D2 (1.1).
- Duplicação: o desvio caía pelo fator √(2(n−1)/(2n−1)) e as 565 células de um ano ficavam com desvio 0. A duplicação inflava S em 2,81 pontos em média (15,48 nas células de um ano) e mudava o veredito de 13,1% dos exemplos.
- RF em D2: R² 0,923 / MAE 4,45 por estação; 0,913 / 4,71 por UF; 0,925 / 4,39 com partes aleatórias. A linha de base fica em R² 0,29 / MAE 15,22.
- Compensação: 252 combinações com uma nota 0 ainda passam de 65 pontos ("Recomendado").
- Faixa física: 19.507 leituras descartadas. Escala da chuva: 37,8% dos meses ajustados, fator médio de 1,013 (máximo de 1,25).

**API (`api/main.py`)**
- Serve a **regra exata**. `MonthResult.details` traz a decomposição e uma `formula` em texto ("100 × (0,35×0,68 + …) × 1,1 × 0,79 × 0,96 = 74,2").
- Mês com `veto` recebe `status = "inviavel"` e sai dos melhores meses.
- Validação: `months` é uma lista de 1 a 12 com ao menos um item (422 caso contrário). A busca de estações é literal (`regex=False`).
- Sem indicador local, usa a **estação vizinha mais próxima até 150 km** e informa isso em `fonte`. Sem vizinha, o mês volta com `score = None` e `status = "sem_dados"`. Zeros nunca são exibidos como medição.
- `alerts` lista todos os meses e variáveis fora dos limites, inclusive temperatura abaixo do mínimo. `alert` continua existindo e traz o primeiro.
- `var_analyses` usa o **pior mês** selecionado, não a média, e traz `mes_referencia` e `faixa_observada`.
- Os textos falam do histórico e usam números no padrão brasileiro. Campos novos: `metodo` e `periodo_dados`.
- `/api/stats` calcula os anos a partir da base.
- O SSE não tem mais pausas simuladas. A lista de meses é validada antes do fluxo, o cálculo é feito uma vez e as etapas são `init`, `station`, `records`, `crop`, `rule_month` (uma por mês), `complete` e `result`.
- `startup` foi trocado por `lifespan`.

**Frontend (`templates/index.html`)**
- O rótulo passou a ser "pontos de 0 a 100" com uma casa decimal, em vez de %.
- A tabela de detalhes tem a coluna "Cálculo da regra".
- Os cards mostram "Inviável", "sem dados" e a estação vizinha usada, e os alertas aparecem em lista.
- Os links do modal foram corrigidos (Embrapa Soja SP15 e registro da FAO 33).

---

## Sumário

1. [Visão geral](#1-visão-geral)
2. [Stack e estrutura de pastas](#2-stack-e-estrutura-de-pastas)
3. [Fluxo ponta a ponta](#3-fluxo-ponta-a-ponta)
4. [Dados de entrada: CSVs do INMET](#4-dados-de-entrada-csvs-do-inmet)
5. [`api/preprocess.py` — ETL dos dados brutos](#5-apipreprocesspy--etl-dos-dados-brutos)
6. [`api/train_model.py` — rótulos e treinamento do modelo](#6-apitrain_modelpy--rótulos-e-treinamento-do-modelo)
7. [`api/main.py` — API FastAPI](#7-apimainpy--api-fastapi)
8. [`templates/index.html` — Frontend](#8-templatesindexhtml--frontend)
9. [Scripts e configuração](#9-scripts-e-configuração)
10. [Artefatos gerados (esquemas e números reais)](#10-artefatos-gerados-esquemas-e-números-reais)
11. [Exemplo real de execução](#11-exemplo-real-de-execução)
12. [Tabelas de regras de negócio](#12-tabelas-de-regras-de-negócio)
13. [Limitações, inconsistências e bugs conhecidos](#13-limitações-inconsistências-e-bugs-conhecidos)
14. [Como executar](#14-como-executar)
15. [Glossário](#15-glossário)

---

## 1. Visão geral

**O que o sistema faz:** o usuário escolhe uma **estação meteorológica do INMET** (por UF), uma **cultura agrícola** (10 opções) e **um ou mais meses**. O sistema devolve:

- um **score de viabilidade de 0 a 100%** para cada mês selecionado e um **score geral** (média);
- um **veredito** textual com cor ("Altamente Recomendado" … "Não Recomendado");
- os **melhores meses do ano** para aquela cultura naquela estação;
- um **alerta** de risco climático (calor, déficit ou excesso hídrico), se houver;
- **gráficos** anuais de temperatura, precipitação, umidade e score;
- uma **análise textual** por variável (temperatura, precipitação, umidade) comparando o clima com as faixas agronômicas da cultura;
- um **diagnóstico consolidado** em texto corrido;
- o **perfil agronômico** da cultura (usos, regiões, ciclo, fase crítica, sensibilidades).

**Ideia central do "ML":**
1. Os CSVs horários do INMET são agregados em **médias mensais** por estação/ano e depois em **normais climatológicas** (média histórica por estação × mês).
2. Para cada combinação *(estação, mês, cultura)* é calculado um **score de aptidão por fórmula determinística** baseada em faixas agronômicas (Embrapa/literatura).
3. Um **RandomForestRegressor** é treinado para reproduzir esse score a partir de features climáticas e geográficas.
4. Na API, o score exibido ao usuário é a **previsão do Random Forest** (não a fórmula).

> Em termos metodológicos, o Random Forest é um **modelo substituto (surrogate)** de uma função especialista: o "rótulo" não vem de produtividade real observada, e sim de uma regra construída a partir da literatura. Isso é importante para descrever o trabalho corretamente (ver seção 13).

---

## 2. Stack e estrutura de pastas

| Camada | Tecnologia |
|---|---|
| Backend | Python 3.11, **FastAPI**, Uvicorn, Pydantic v2 |
| Dados | pandas, numpy (arquivos `.pkl` do pandas) |
| ML | scikit-learn (`RandomForestRegressor`, `LabelEncoder`, `cross_val_score`), joblib |
| Frontend | HTML único, **Tailwind CSS (CDN)**, **Chart.js 4.4.0**, JavaScript puro, fontes Google (Inter, Playfair Display, JetBrains Mono) |
| Streaming | **Server-Sent Events (SSE)** via `StreamingResponse` + `EventSource` no navegador |
| Exposição pública | **ngrok** (túnel para a porta 8000) |

```
tcc/
├── api/
│   ├── main.py                 # API FastAPI (endpoints, lógica de recomendação, textos)
│   ├── preprocess.py           # ETL: CSVs INMET → .pkl (mensal, normais, estações)
│   ├── train_model.py          # Gera rótulos por fórmula e treina o Random Forest
│   └── models/
│       └── recommendation_model.joblib   # modelo + encoders + requisitos das culturas (~47 MB)
├── data/
│   ├── monthly_climate.pkl     # agregados mensais por estação/ano
│   ├── climate_normals.pkl     # normais climatológicas por estação/mês
│   └── stations.pkl            # cadastro de estações
├── templates/
│   └── index.html              # SPA inteira (HTML + CSS + JS)
├── static/
│   └── agroclimalogo.png       # logo
├── base/   (ou BASE/)          # CSVs brutos do INMET (~5.755 arquivos, 3,5 GB) — NÃO versionado
├── BASE_SAMPLE/                # amostra de 5% dos CSVs brutos (287 arquivos) — versionada
├── template_mackenzie_tcc/     # template LaTeX/Word do artigo de TCC (não faz parte do sistema)
├── iniciar.bat                 # inicializador Windows (preprocessa, treina, sobe API e ngrok)
├── teste.bat                   # script de teste trivial ("funcionou")
├── requirements.txt
└── .gitignore
```

---

## 3. Fluxo ponta a ponta

```
┌──────────────────────────┐
│ CSVs horários INMET      │  base/*.CSV  (1 arquivo = 1 estação × 1 ano)
└────────────┬─────────────┘
             │ python api/preprocess.py   → run()
             ▼
┌──────────────────────────┐   parse_header() → metadados da estação
│ process_file() por CSV   │   leitura, mapeamento de colunas, parse_float()
│ agrega hora → mês        │   groupby(year, month)
└────────────┬─────────────┘
             ▼
  monthly_climate.pkl  (filtro n_horas ≥ 100)
             │ groupby(codigo, month) → média/desvio entre anos
             ▼
  climate_normals.pkl  +  stations.pkl
             │ python api/train_model.py  → train()
             ▼
┌──────────────────────────┐   build_training_data(): 1 linha por (estação, mês, cultura)
│ rótulo = compute_crop_   │   aptidao_score() para temp, precip, umidade
│ score() (fórmula 0–100)  │   pesos 35/40/25, bônus de mês, penalidades de variabilidade
└────────────┬─────────────┘
             │ RandomForestRegressor(200 árvores, max_depth=12, min_samples_leaf=5)
             ▼
  recommendation_model.joblib  {model, encoders, feature_cols, crop_requirements}
             │ python api/main.py  (Uvicorn :8000)  → startup() → load_artifacts()
             ▼
┌──────────────────────────┐
│ FastAPI                  │  GET /  → index.html
│                          │  GET /api/stats, /api/stations, /api/crops, /api/climate/{code}
│                          │  POST /api/recommend
│                          │  GET /api/recommend/stream  (SSE)
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ index.html (navegador)   │  init() → carrega stats/estações/culturas
│                          │  analyze() → EventSource → "terminal" animado
│                          │  renderResult() → gauge, cards, tabelas, gráficos, textos
└──────────────────────────┘
```

---

## 4. Dados de entrada: CSVs do INMET

Arquivos das **estações meteorológicas automáticas** do INMET. Nome no padrão:
`INMET_<REGIAO>_<UF>_<CODIGO>_<NOME>_<DATA_INI>_A_<DATA_FIM>.CSV`
Ex.: `INMET_CO_DF_A045_AGUAS EMENDADAS_01-01-2009_A_31-12-2009.CSV`

- **Encoding:** `latin-1`. **Separador:** `;`. **Decimal:** vírgula. **Valor ausente:** `-9999`.
- **8 primeiras linhas = metadados** no formato `CHAVE:;VALOR`:
  ```
  REGIÃO:;CO
  UF:;DF
  ESTAÇÃO:;AGUAS EMENDADAS
  CODIGO (WMO):;A045
  LATITUDE:;-15,59638888
  LONGITUDE:;-47,62583332
  ALTITUDE:;1200
  DATA DE FUNDAÇÃO (YYYY-MM-DD):;2008-10-03
  ```
- **Linha 9 = cabeçalho** das colunas horárias; depois, **uma linha por hora (UTC)**.
- Colunas usadas pelo sistema (as demais — pressão, radiação, ponto de orvalho, vento etc. — são ignoradas):

| Coluna original INMET | Nome interno |
|---|---|
| `DATA (YYYY-MM-DD)` | `data` |
| `PRECIPITAÇÃO TOTAL, HORÁRIO (mm)` | `precip` |
| `TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)` | `temp` |
| `TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)` | `temp_max` |
| `TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)` | `temp_min` |
| `UMIDADE RELATIVA DO AR, HORARIA (%)` | `umidade` |

---

## 5. `api/preprocess.py` — ETL dos dados brutos

**Responsabilidade:** ler todos os CSVs de `base/`, agregar dados horários em mensais, calcular normais climatológicas e salvar três `.pkl` em `data/`.

### Constantes

| Nome | Valor / função |
|---|---|
| `BASE_DIR` | `<raiz>/BASE` — pasta com os CSVs brutos (no Windows, `base/` funciona por ser case-insensitive) |
| `DATA_DIR` | `<raiz>/data` — criada automaticamente (`mkdir(exist_ok=True)`) |
| `COLS_USE` | Dicionário coluna INMET → nome interno. **Não é usado** no código; o mapeamento real é feito por palavras-chave em `process_file()` |

### `parse_header(filepath: str) -> dict`
Lê as **8 primeiras linhas** do CSV (encoding latin-1). Para cada linha com `;`, divide no primeiro `;`: a chave é a parte esquerda sem o `:` final e sem espaços; o valor é a parte direita. Retorna um dicionário como `{"REGIÃO": "CO", "UF": "DF", "ESTAÇÃO": "AGUAS EMENDADAS", "CODIGO (WMO)": "A045", "LATITUDE": "-15,59638888", ...}`.

### `parse_float(val) -> float`
Converte uma string com vírgula decimal em `float` (`"21,3"` → `21.3`). Retorna `np.nan` se o valor for `-9999` (código de ausente do INMET) ou se a conversão falhar (`ValueError`/`TypeError`).

### `process_file(filepath: str) -> tuple[dict | None, pd.DataFrame | None]`
Processa **um** CSV. Passos:

1. Chama `parse_header()` e extrai `CODIGO (WMO)`, `ESTAÇÃO`, `UF`, `REGIÃO`, `LATITUDE`, `LONGITUDE`, `ALTITUDE` (troca vírgula por ponto; se falhar, usa `0.0` para lat/lon/alt).
2. Lê o CSV com `pd.read_csv(sep=";", encoding="latin-1", skiprows=8, decimal=",", na_values=["-9999","-9999.0",""], dtype=str)`.
3. Remove colunas `Unnamed*` (o INMET deixa um `;` sobrando no fim de cada linha) e faz `strip()` nos nomes.
4. **Mapeamento flexível de colunas** por palavras-chave em maiúsculas (tolera variações de acentuação/encoding entre anos):
   - `DATA` + `YYYY` → `data`
   - `PRECIPITA` + `TOTAL` → `precip`
   - `TEMPERATURA DO AR` + `BULBO` → `temp`
   - `TEMPERATURA M` + `XIMA` + `ANT` → `temp_max`
   - `TEMPERATURA M` + `NIMA` + `ANT` → `temp_min`
   - `UMIDADE RELATIVA DO AR` → `umidade`
5. Se faltar alguma das obrigatórias (`data`, `precip`, `temp`, `umidade`) → retorna `(None, None)` (arquivo descartado).
6. Converte `data` para datetime (`%Y-%m-%d`, erros viram `NaT`) e aplica `parse_float` em `precip`, `temp`, `temp_max`, `temp_min`, `umidade`. Descarta linhas sem data. Cria `year` e `month`.
7. **Agrega por (year, month):**

   | Coluna gerada | Cálculo |
   |---|---|
   | `precip_total` | **soma** da precipitação horária (mm no mês) |
   | `temp_media` | **média** da temperatura horária |
   | `temp_max` | **máximo** de `temp_max` no mês (se a coluna não existir, máximo de `temp`) |
   | `temp_min` | **mínimo** de `temp_min` no mês (se não existir, mínimo de `temp`) |
   | `umidade_media` | **média** da umidade horária |
   | `n_horas` | **contagem** de horas com temperatura válida |

8. Adiciona `codigo`, `nome`, `uf`, `regiao` ao DataFrame mensal e monta `station_info = {codigo, nome, uf, regiao, lat, lon, alt}`.
9. Qualquer exceção é capturada, impressa (`ERRO em <arquivo>`) e retorna `(None, None)`.

### `run()`
Função principal do ETL:

1. Lista `*.CSV` e `*.csv` em `BASE_DIR` e imprime o progresso a cada 200 arquivos.
2. Para cada arquivo, chama `process_file()`. Guarda a estação em `all_stations` (**a primeira ocorrência de cada código vence**) e acumula os DataFrames mensais.
3. Concatena tudo em `df_monthly` e cria `df_stations`.
4. **Filtro de qualidade:** mantém apenas meses com `n_horas >= 100`.
5. **Normais climatológicas:** `groupby(["codigo","month"])`:

   | Coluna | Cálculo (entre os anos disponíveis) |
   |---|---|
   | `precip_norm` / `precip_std` | média / desvio-padrão de `precip_total` |
   | `temp_norm` / `temp_std` | média / desvio-padrão de `temp_media` |
   | `umidade_norm` / `umidade_std` | média / desvio-padrão de `umidade_media` |
   | `temp_max_norm` | média de `temp_max` (média das máximas absolutas mensais) |
   | `temp_min_norm` | média de `temp_min` (média das mínimas absolutas mensais) |
   | `n_anos` | número de anos distintos |

6. Faz merge das normais com `nome, uf, regiao, lat, lon` da estação.
7. Salva `monthly_climate.pkl`, `climate_normals.pkl` e `stations.pkl` e imprime os totais.

> Observação: a docstring do arquivo fala em `.parquet`, mas o código salva `.pkl`.

---

## 6. `api/train_model.py` — rótulos e treinamento do modelo

**Responsabilidade:** definir os requisitos climáticos das culturas, gerar o score-alvo por fórmula e treinar o Random Forest.

### `CROP_REQUIREMENTS` (dicionário central de regras agronômicas)

Cada cultura tem `nome`, `emoji`, `desc`, `meses_favoraveis` e três tuplas no formato **`(mínimo absoluto, ótimo mínimo, ótimo máximo, máximo absoluto)`**:

| Chave | Nome | Temperatura °C | Precipitação mm/mês | Umidade % | Meses favoráveis (bônus) |
|---|---|---|---|---|---|
| `soja` | Soja | 20 / 25 / 30 / 40 | 60 / 100 / 200 / 300 | 50 / 60 / 85 / 95 | 9, 10, 11, 12, 1 |
| `milho` | Milho | 10 / 24 / 30 / 40 | 50 / 80 / 180 / 280 | 45 / 55 / 80 / 90 | 9, 10, 11, 12, 1, 2 |
| `cafe` | Café | 18 / 19 / 22 / 28 | 80 / 100 / 180 / 250 | 50 / 60 / 80 / 90 | 9, 10, 11 |
| `arroz` | Arroz | 20 / 22 / 33 / 38 | 100 / 125 / 200 / 300 | 60 / 70 / 90 / 98 | 10, 11, 12, 1 |
| `feijao` | Feijão | 15 / 18 / 26 / 33 | 40 / 70 / 130 / 200 | 50 / 55 / 75 / 85 | 1, 2, 6, 7, 8 |
| `trigo` | Trigo | 10 / 15 / 20 / 32 | 30 / 50 / 100 / 150 | 50 / 55 / 75 / 85 | 4, 5, 6, 7 |
| `algodao` | Algodão | 18 / 22 / 30 / 38 | 60 / 90 / 160 / 250 | 45 / 50 / 70 / 80 | 10, 11, 12, 1 |
| `cana` | Cana-de-açúcar | 18 / 22 / 30 / 38 | 100 / 150 / 250 / 350 | 55 / 65 / 85 / 95 | 8, 9, 10, 11 |
| `mandioca` | Mandioca | 18 / 22 / 28 / 35 | 50 / 80 / 150 / 250 | 50 / 60 / 80 / 90 | 9, 10, 11 |
| `sorgo` | Sorgo | 12 / 20 / 32 / 40 | 40 / 60 / 130 / 200 | 40 / 50 / 75 / 85 | 10, 11, 12, 1, 2 |

Fontes citadas nos comentários do código: Embrapa Soja (Sistema de Produção 15, 2011), Embrapa Milho e Sorgo, Camargo & Camargo (2001, Bragantia v.60 n.1), Embrapa Café, Embrapa Arroz e Feijão, Embrapa Trigo. Esse dicionário é salvo dentro do `.joblib` e é a **única fonte** das faixas usadas pela API.

### `aptidao_score(val, t_min, t_ot_min, t_ot_max, t_max) -> float`
Função de pertinência **trapezoidal** que devolve um score de 0 a 1:

```
score
 1.0 |        ┌────────────┐
     |       /              \
     |      /                \
 0.0 |─────┘                  └─────
         t_min  t_ot_min  t_ot_max  t_max
```

- `val` é `NaN` → **0.5** (incerteza).
- `val < t_min` ou `val > t_max` → **0.0**.
- `t_ot_min ≤ val ≤ t_ot_max` → **1.0**.
- Entre `t_min` e `t_ot_min` → rampa linear `(val − t_min) / (t_ot_min − t_min)`.
- Entre `t_ot_max` e `t_max` → rampa linear `(t_max − val) / (t_max − t_ot_max)`.
- Protege divisão por zero (se os limites coincidirem, retorna 1.0).

### `compute_crop_score(row, crop_key) -> float`
Calcula o **rótulo de treino** (0–100) para uma linha de normais e uma cultura:

```
s_temp   = aptidao_score(temp_norm,    *req["temp"])
s_precip = aptidao_score(precip_norm,  *req["precip"])
s_umid   = aptidao_score(umidade_norm, *req["umidade"])

bonus      = 1.1 se month ∈ meses_favoraveis, senão 1.0
pen_precip = max(0, 1 − (precip_std / (precip_norm + 1)) × 0.3)   # penaliza coeficiente de variação da chuva
pen_temp   = max(0, 1 − (temp_std / 5) × 0.2)                     # penaliza variabilidade da temperatura

score = (0.35·s_temp + 0.40·s_precip + 0.25·s_umid) × bonus × pen_precip × pen_temp
retorno = min(100, round(score × 100, 1))
```

**Pesos:** temperatura 35%, precipitação 40%, umidade 25%. O bônus de 10% para meses favoráveis pode estourar 100, por isso o `min(100, …)`.

**Exemplo real (Brasília A001, outubro, soja):**
`temp_norm=23.24`, `precip_norm=88.12`, `precip_std=100.07`, `temp_std=1.21`, `umidade_norm=58.67`
→ `s_temp=0.648`, `s_precip=0.703`, `s_umid=0.867`
→ base = 0.35·0.648 + 0.40·0.703 + 0.25·0.867 = 0.725
→ × 1.1 (outubro é favorável) × 0.663 (pen_precip) × 0.951 (pen_temp) = **50.3** (fórmula).
O Random Forest prevê **61.9** para o mesmo caso: o modelo suaviza a fórmula e não a reproduz exatamente.

### `build_training_data(normals: pd.DataFrame) -> pd.DataFrame`
Faz o produto cartesiano **(cada linha de normais) × (cada uma das 10 culturas)**. Para cada par, calcula `score = compute_crop_score(row, crop_key)` e monta uma linha com: `codigo, uf, regiao, lat, lon, month, crop, temp_norm, temp_std, temp_max_norm, temp_min_norm, precip_norm, precip_std, umidade_norm, umidade_std, n_anos, score`. Valores ausentes usam padrões (`temp_max_norm = temp_norm + 3`, `temp_min_norm = temp_norm − 3`, `n_anos = 1`, desvios = 0). Com os dados atuais: 5.621 normais × 10 = **56.210 exemplos**.

### `train()`
1. Carrega `climate_normals.pkl` (erro se não existir) e preenche desvios ausentes: `precip_std=30`, `temp_std=2`, `umidade_std=8` (acontece quando a estação só tem 1 ano daquele mês).
2. Gera o dataset com `build_training_data()`.
3. Cria os `LabelEncoder`s: `le_uf`, `le_regiao`, `le_crop`, `le_codigo` (este último é salvo, mas **não é usado** como feature).
4. **Features (15, nesta ordem exata — a API depende dela):**
   `lat, lon, month, crop_enc, uf_enc, regiao_enc, temp_norm, temp_std, temp_max_norm, temp_min_norm, precip_norm, precip_std, umidade_norm, umidade_std, n_anos`
   Alvo: `score`. NaNs em X viram 0.
5. Treina `RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)`.
6. Avalia com **validação cruzada 5-fold (R²)** e **MAE no treino** (só imprime; não salva as métricas).
7. Salva via `joblib.dump` o dicionário:
   ```python
   {"model", "le_uf", "le_regiao", "le_crop", "le_codigo", "feature_cols", "crop_requirements"}
   ```
   em `api/models/recommendation_model.joblib`.

**Importância das features no modelo atual** (`feature_importances_`):

| Feature | Importância |
|---|---|
| `precip_norm` | 0.442 |
| `crop_enc` | 0.253 |
| `temp_norm` | 0.142 |
| `precip_std` | 0.084 |
| `umidade_norm` | 0.063 |
| `temp_min_norm` | 0.005 |
| `temp_max_norm` | 0.004 |
| demais (`lat`, `lon`, `month`, `uf_enc`, `regiao_enc`, `temp_std`, `umidade_std`, `n_anos`) | < 0.002 cada |

Isso é coerente com a fórmula: precipitação tem o maior peso e a cultura define as faixas.

> Imports não usados: `GradientBoostingRegressor`, `StandardScaler`, `Pipeline`, `r2_score`.

---

## 7. `api/main.py` — API FastAPI

### 7.1 Configuração e estado global

- **Caminhos:** `BASE_ROOT` (raiz do projeto), `DATA_DIR`, `MODEL_DIR` (`api/models`), `TEMPLATES_DIR`, `STATIC_DIR`.
- `app = FastAPI(title="AgroClima API", version="1.0.0")`.
- **CORS** totalmente aberto (`allow_origins=["*"]`, todos os métodos e headers).
- Monta `/static` → `static/` se a pasta existir.
- **Estado global em memória** (carregado uma vez na inicialização):
  - `_artifacts` — dicionário do `.joblib`
  - `_normals` — DataFrame de normais
  - `_stations` — DataFrame de estações
  - `_monthly` — DataFrame mensal

### `load_artifacts()`
Carrega o `.joblib` e os três `.pkl` nas variáveis globais. Se o modelo não existir, lança `RuntimeError` instruindo a rodar `preprocess.py` e `train_model.py`. Imprime "Modelo e dados carregados com sucesso."

### `startup()`
Handler `@app.on_event("startup")` que chama `load_artifacts()` quando o Uvicorn sobe.

### 7.2 Modelos Pydantic

**`RecommendRequest`** (entrada do POST)
| Campo | Tipo | Descrição |
|---|---|---|
| `station_code` | `str` | código WMO da estação (ex.: `"A001"`) |
| `crop` | `str` | chave da cultura (ex.: `"soja"`) |
| `months` | `list[int]` | meses 1–12 |

**`MonthResult`** (um mês analisado)
`month`, `month_name`, `score`, `temp_media`, `precip_total`, `umidade_media`, `status` (`excelente|bom|moderado|ruim|critico`), `details` (`{temp_max, temp_min, precip_std, n_anos}`).

**`VarAnalysis`** (análise de uma variável climática)
`variavel` (`Temperatura|Precipitação|Umidade`), `valor`, `unidade`, `faixa_otima` (texto "a–b un"), `faixa_tolerada`, `status` (`otimo|atencao|critico`; o front também aceita `bom`, mas o backend nunca o gera), `icone` (emoji), `titulo`, `explicacao`.

**`RecommendResponse`** (saída completa)
`station_code`, `station_name`, `uf`, `crop`, `crop_name`, `overall_score`, `verdict`, `verdict_color`, `months: list[MonthResult]`, `best_months: list[str]`, `alert: Optional[str]`, `climate_summary: dict`, `crop_profile: dict`, `var_analyses: list[VarAnalysis]`, `diagnostico: str`.

### 7.3 Constantes e helpers

- `MONTH_NAMES` — `["", "Janeiro", …, "Dezembro"]` (índice 1 = janeiro).
- `MONTH_ABBR` — `["", "Jan", …, "Dez"]` (definido, mas não usado no backend).

#### `score_to_verdict(score: float) -> tuple[str, str]`
| Score | Veredito | Cor |
|---|---|---|
| ≥ 80 | Altamente Recomendado | `#22c55e` (verde) |
| ≥ 65 | Recomendado | `#84cc16` (verde-limão) |
| ≥ 50 | Moderado | `#eab308` (amarelo) |
| ≥ 35 | Risco Elevado | `#f97316` (laranja) |
| < 35 | Não Recomendado | `#ef4444` (vermelho) |

#### `score_to_status(score: float) -> str`
Mesmos limiares, retornando `excelente` / `bom` / `moderado` / `ruim` / `critico` (usado como classe CSS dos cards de mês).

### 7.4 `CROP_PROFILES`
Dicionário com conteúdo **textual** por cultura (usado só para explicação, não entra no modelo):
- `usos` — principais usos econômicos;
- `regioes` — principais regiões produtoras no Brasil;
- `ciclo` — duração do ciclo;
- `fase_critica` — fase fenológica mais sensível (inserida nos textos de `_explain_var`);
- `sensibilidades` — lista de riscos (vira "chips" vermelhos no front);
- `perfil` — parágrafo descritivo.

Ex. soja: fase crítica "florescimento e enchimento de grãos"; sensibilidades "déficit hídrico no florescimento", "temperaturas abaixo de 20°C", "geadas tardias".

### 7.5 `_explain_var(varname, value, unit, req_min, req_otmin, req_otmax, req_max, crop_nome, crop_key, fase_critica) -> VarAnalysis`
Gera a **análise textual** de uma variável comparando `value` (média dos meses selecionados) com as faixas da cultura. Classifica em **5 zonas** e, para cada variável, monta ícone, título e explicação específicos (com os números calculados, ex.: "3.0°C abaixo da zona ótima"):

| Zona | Status | Temperatura | Precipitação | Umidade |
|---|---|---|---|---|
| `value < req_min` | `critico` | 🥶 "Temperatura crítica — abaixo do limite" (risco de geada, germinação inviável) | 🏜️ "Precipitação insuficiente — risco de seca" (exige irrigação) | 🌵 "Umidade crítica — ar muito seco" |
| `req_min ≤ value < req_otmin` | `atencao` | 🌡️ "Temperatura abaixo do ideal" | 💧 "Precipitação abaixo do ideal" (risco de veranico) | 💨 "Umidade abaixo do ideal" |
| `req_otmin ≤ value ≤ req_otmax` | `otimo` | ✅ "Temperatura ideal" | 🌧️ "Precipitação ideal" | ✅ "Umidade ideal" |
| `req_otmax < value ≤ req_max` | `atencao` | 🌡️ "Temperatura acima do ideal" (estresse térmico) | 🌦️ "Precipitação elevada" (doenças fúngicas, drenagem) | 🌫️ "Umidade elevada" (fungos e pragas) |
| `value > req_max` | `critico` | 🔥 "Temperatura crítica — calor excessivo" (esterilidade do pólen) | 🌊 "Excesso hídrico — risco de encharcamento" | 🌫️ "Umidade excessiva" |

`faixa_otima` = `"{otmin}–{otmax} {unit}"` e `faixa_tolerada` = `"{min}–{max} {unit}"`. O ramo de umidade é o "else" final: qualquer `varname` que não seja "Temperatura" nem "Precipitação" cai nele.

### 7.6 `_diagnostico(crop_nome, crop_key, overall, verdict, station_name, uf, months_sel, analyses, best_months) -> str`
Monta o **diagnóstico consolidado** em até 4 partes, unidas por espaço (usa `**negrito**` em markdown, que o front converte em `<strong>`):

1. **Diagnóstico geral** conforme `overall`: ≥80 "altamente favoráveis", ≥65 "favoráveis", ≥50 "moderadas", ≥35 "risco elevado", senão "não recomendadas". Cita estação/UF e meses.
2. **Fatores limitantes:** se houver variáveis `critico`, cita-as e reaproveita a 1ª frase da explicação da primeira; senão, se houver `atencao`, diz que exigem atenção.
3. **Comparação com regiões produtoras** (`CROP_PROFILES[...]["regioes"]`): se `overall ≥ 65`, "climaticamente próxima dessas regiões"; senão, "difere significativamente".
4. **Melhores meses:** lista até 3 de `best_months`.

> A variável `otimos` é calculada, mas não é usada.

### 7.7 `predict_score(station_row: pd.Series, month: int, crop: str) -> float`
**Coração da inferência.** Usa o Random Forest para prever o score de um (estação, mês, cultura):

1. Codifica `uf`, `regiao` e `crop` com os `LabelEncoder`s. UF/região desconhecidas → `0`; cultura desconhecida → `HTTPException(400)`.
2. Lê `lat` e `lon` da estação.
3. Busca a linha de normais `(codigo, month)`.
   - **Fallback:** se a estação não tiver normal para esse mês, usa a **média de todas as estações da mesma UF naquele mês**; se nem isso existir, retorna **50.0** fixo.
4. Monta o vetor de **15 features na mesma ordem de `feature_cols`**, com defaults caso faltem (`temp_norm=25`, `temp_std=2`, `temp_max_norm=30`, `temp_min_norm=18`, `precip_norm=100`, `precip_std=30`, `umidade_norm=70`, `umidade_std=8`, `n_anos=5`).
5. `model.predict(features)` e limita o resultado a **[0, 100]**.

### 7.8 `get_climate_for_month(station_code: str, month: int) -> dict`
Retorna as normais de um mês da estação, arredondadas: `temp_media`, `temp_max`, `temp_min`, `precip_total`, `precip_std`, `umidade_media` (1 casa) e `n_anos` (int). Retorna `{}` se não houver dados (**sem** o fallback por UF de `predict_score`).

### 7.9 `get_annual_series(station_code: str) -> dict`
Retorna a série anual (normais ordenadas por mês) para os gráficos:
`{"months": [1..12], "temp": [...], "precip": [...], "umidade": [...]}` (NaN → 0, 1 casa decimal). `{}` se a estação não tiver normais.

### 7.10 Endpoints

#### `GET /` → `root()`
Devolve `templates/index.html` como `HTMLResponse` (ou um HTML mínimo "Template não encontrado").

#### `GET /api/stations?uf=&search=` → `get_stations()`
Lista as estações. Filtros opcionais: `uf` (case-insensitive) e `search` (substring no nome). Ordena por `uf, nome`, troca NaN por `""` e retorna uma lista de `{codigo, nome, uf, regiao, lat, lon, alt}`. O front não usa `search`.

#### `GET /api/crops` → `get_crops()`
Retorna `[{key, nome, emoji, desc}]` a partir de `_artifacts["crop_requirements"]`.

#### `GET /api/climate/{station_code}` → `get_climate()`
Retorna `{codigo, nome, uf, lat, lon, series}` com `series = get_annual_series()`. 404 se a estação não existir. **O front atual não chama este endpoint** (usa `climate_summary` do resultado).

#### `POST /api/recommend` → `recommend(req: RecommendRequest)`
**Lógica principal de recomendação:**

1. Valida se o modelo está carregado (500), se a estação existe (404) e se a cultura é suportada (400).
2. Para cada mês único e válido (1–12) de `req.months`, em ordem:
   - `score = predict_score(...)`;
   - `climate = get_climate_for_month(...)`;
   - monta um `MonthResult` (`status = score_to_status(score)`).
3. Nenhum mês válido → 400.
4. `overall = média simples dos scores` → `score_to_verdict()`.
5. **Melhores meses:** prevê o score dos **12 meses**, pega os 4 maiores e mantém apenas os que têm score ≥ **60**.
6. **Alerta** (só o **primeiro** encontrado, percorrendo os meses em ordem, para no `break`):
   - `temp_media > temp_max absoluto` → "Temperatura média (X°C) acima do limite…";
   - `precip_total < precip_min absoluto` → "Precipitação baixa… risco de déficit hídrico";
   - `precip_total > precip_max absoluto` → "Precipitação elevada… risco de excesso hídrico".
7. `climate_summary = get_annual_series()`.
8. Calcula as **médias** de temperatura, precipitação e umidade dos meses selecionados e chama `_explain_var()` para cada uma → `var_analyses`.
9. `diagnostico = _diagnostico(...)`.
10. Retorna `RecommendResponse` com `crop_profile = CROP_PROFILES[crop]`.

#### `GET /api/stats` → `get_stats()`
Retorna `{estacoes, registros_mensais, normais_climaticas, culturas, estados (lista de UFs), anos}`. **`anos` é fixo no código:** `["2005", "2007-2009", "2013", "2020-2026"]`.

#### `GET /api/recommend/stream?station_code=&crop=&months=1,2,3` → `recommend_stream()`
Endpoint **SSE** usado pelo frontend. Transmite etapas de progresso para o "terminal" animado e, no final, o resultado completo.

- Função interna **`evt(step, msg, detail="", data=None)`**: serializa `{"step", "msg", "detail", "data"?}` como `data: <json>\n\n`.
- Gerador assíncrono **`generate()`**, que emite em sequência:

| # | `step` | Conteúdo | Pausa |
|---|---|---|---|
| 1 | `init` | "Iniciando motor de análise climática…" / "AgroClima ML Engine v1.0 · Random Forest" | 0.5 s |
| 2 | `station` | Nome e código da estação; UF, altitude, lat/lon | 0.5 s |
| 3 | `loading` | "Carregando séries históricas do INMET…" | 0.7 s |
| 4 | `records` | Nº de registros mensais e período/anos (ou "normais regionais" se não houver) | 0.6 s |
| 5 | `normals_calc` | "Calculando normais climáticas mensais…" | 0.8 s |
| 6 | `normals_done` | Temperatura média anual, amplitude, precipitação anual e umidade média | 0.5 s |
| 7 | `crop` | Nome + emoji + descrição da cultura | 0.5 s |
| 8 | `crop_req` | Faixas ótimas de temperatura, precipitação e umidade | 0.5 s |
| 9 | `ml_init` | "Inicializando Random Forest (200 árvores)…" | 0.7 s |
| 10 | `ml_features` | Lista das features | 0.6 s |
| 11 | `ml_month` (1 por mês) | "Março: viabilidade = 72.3% [███████░░░]" + clima do mês; `data={month, score}` | 0.65 s cada |
| 12 | `scoring` | "Ponderando temperatura (35%), precipitação (40%) e umidade (25%)" | 0.5 s |
| 13 | `complete` | Tempo total, score e veredito; `data={overall_score, verdict, color}` | 0.4 s |
| 14 | `result` | `{"step":"result","result": <RecommendResponse completo>}` (chama `recommend()` internamente) | — |
| — | `error` | Em qualquer falha (estação/cultura inexistente, exceção) | — |

- Os `asyncio.sleep` são **pausas cosméticas** para a animação (~6,8 s + 0,65 s por mês). O cálculo real leva milissegundos.
- Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no`, `Connection: keep-alive`.

#### `if __name__ == "__main__"`
Roda `uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)`.

---

## 8. `templates/index.html` — Frontend

Página única (SPA) servida em `/`. Tema escuro verde, Tailwind via CDN, Chart.js para os gráficos.

### 8.1 Seções visuais (de cima para baixo)

| Seção | ID | Conteúdo |
|---|---|---|
| Hero | — | Logo, navegação (Sistema/Análise/Sobre), badge de stats, título "Plante com Inteligência", botões, 4 cards de estatísticas (`s-estacoes`, `s-registros`, `s-culturas`, `s-estados`) |
| Como funciona | `#sobre` | 3 cards (Dados INMET, Machine Learning, Recomendação) + botão "Literatura Agronômica" |
| Formulário | `#analise` | Select de UF (`sel-uf`), select de estação (`sel-station`), grade de culturas (`crop-grid`), 12 toggles de mês (`month-grid`), atalhos (todos / limpar / safra verão Out–Mar), botão `btn-analyze` |
| Resultados | `#resultado` (começa oculta) | Cabeçalho com **gauge SVG circular**, veredito, alerta, melhores meses; cards por mês; 4 gráficos (temperatura, precipitação, umidade, score); tabela detalhada; perfil da cultura; 3 cards de variáveis; diagnóstico; botão "Nova Análise" |
| Terminal | `#terminal-overlay` | Overlay fullscreen estilo terminal (barra de título macOS, cronômetro, barra de status estilo VS Code com barra de progresso) |
| Log | `#log-section` | Log recolhível com a cópia do terminal da última análise |
| Toast | `#toast` | Mensagem de erro temporária (4 s) |
| Modal Literatura | `#lit-overlay` | Abas por cultura com barras de faixas, tabela de valores e referências bibliográficas com links |
| Footer | — | Links para INMET, Embrapa e MAPA |

### 8.2 Estado global JS

| Variável | Conteúdo |
|---|---|
| `API` | `''` (mesma origem) |
| `MONTHS` / `MONTHS_FULL` | abreviações e nomes dos meses (índice 1 = janeiro) |
| `allStations` | lista vinda de `/api/stations` |
| `allCrops` | lista vinda de `/api/crops` |
| `selectedCrop` | chave da cultura selecionada |
| `LITERATURA` | dicionário com faixas e referências de cada cultura (**cópia manual** de `CROP_REQUIREMENTS` + bibliografia) |
| `_litCrop` | aba ativa do modal (padrão `soja`) |
| `STEP_PCT` | % de progresso de cada `step` SSE (init 12 … ml_features 73, scoring 92, complete 97, result/error 100) |
| `_termTimer`, `_termStart`, `_termMonths`, `_termMDone`, `_logLines` | estado do terminal |

### 8.3 Funções JS — inicialização e formulário

- **`init()`** — executada no carregamento. Roda em paralelo `loadStats()`, `loadStations()` e `loadCrops()` e depois `buildMonthGrid()`.
- **`loadStats()`** — `GET /api/stats`; preenche os 4 cards (registros em "k"), o texto do badge e faz o fade-in. Erros são ignorados silenciosamente.
- **`loadStations()`** — `GET /api/stations`; guarda em `allStations`, monta o select de UFs (únicas, ordenadas) e chama `filterStations()`.
- **`filterStations()`** — repopula `sel-station` com as estações da UF escolhida (ou todas), com o texto `"NOME (CODIGO)"`.
- **`loadCrops()`** — `GET /api/crops`; cria um botão por cultura (emoji + nome, tooltip = `desc`) que chama `selectCrop()`.
- **`selectCrop(key)`** — define `selectedCrop` e aplica o estilo "ativo" ao botão escolhido.
- **`buildMonthGrid()`** — cria 12 checkboxes ocultos `m1..m12` com labels estilizados.
- **`selectMonths(mode)`** — `'all'` marca todos, `'none'` limpa, `'safra'` marca Out–Mar (10, 11, 12, 1, 2, 3).

### 8.4 Funções JS — modal de Literatura

- **`openLiteratura()`** — monta as abas, renderiza a cultura atual, mostra o modal e trava o scroll do body.
- **`closeLiteratura()`** — esconde o modal e libera o scroll.
- **`closeLitOnBg(e)`** — fecha ao clicar no fundo escuro.
- **`buildLitTabs()`** — cria uma aba por cultura de `LITERATURA`; clicar troca `_litCrop` e re-renderiza.
- **`rangeBar(label, vMin, vOtMin, vOtMax, vMax, unit, color)`** — gera o HTML de uma barra horizontal: fundo amarelo = faixa tolerada inteira; sobreposição verde = faixa ótima, posicionada proporcionalmente.
- **`renderLitCrop(key)`** — renderiza emoji, descrição, 3 `rangeBar`s, a tabela "Valores Exatos (usados no modelo)" e a lista de referências (autor, título, local, ano, link).

### 8.5 Funções JS — terminal SSE

- **`termOpen(totalMonths, stationCode, cropKey, monthsList)`** — reseta o estado, mostra o overlay, inicia o cronômetro (atualiza a cada 200 ms) e escreve o "prompt" fictício `user@agroclima:~$ python analyze.py --estacao … --cultura … --meses …`.
- **`termClose()`** — esconde o overlay e para o cronômetro.
- **`termProg(pct, stepLabel)`** — atualiza a barra de progresso, a % e o rótulo na status bar.
- **`termRaw(cls, text)`** — adiciona uma linha com classe de cor, move o cursor piscante para ela, rola até o fim e registra em `_logLines`.
- **`termBlank()`** — adiciona uma linha em branco.
- **`termPrint(step, msg, detail)`** — traduz cada `step` SSE em linhas formatadas com timestamp e símbolos (`✔` ok, `›` mês do ML, `★` concluído, `✖` erro), usando as cores `tl-run`, `tl-ok`, `tl-ml`, `tl-done`, `tl-err`, `tl-sub`.
- **`termTimestamp()`** — retorna `[mm:ss.s]` desde o início.
- **`buildLog()`** — copia `_logLines` para o log persistente abaixo dos resultados.
- **`toggleLog()`** — expande/recolhe o log (`[expandir]`/`[recolher]`).

### 8.6 Funções JS — análise e renderização

- **`analyze()`** — fluxo principal:
  1. Valida estação, cultura e pelo menos um mês (senão `showToast`).
  2. Desabilita o botão e chama `termOpen()`.
  3. Abre `new EventSource('/api/recommend/stream?station_code=…&crop=…&months=1,2,3')`.
  4. Em cada mensagem: atualiza o progresso (para `ml_month`: `73 + (n/total)·18`%; senão, `STEP_PCT`) e chama `termPrint()`.
  5. `error` → fecha a conexão, reabilita o botão, fecha o terminal após 4 s e mostra um toast.
  6. `result` → fecha a conexão, `buildLog()` e, após 700 ms, `termClose()`, `renderResult(result)`, mostra o log e os resultados e rola até eles.
  7. `onerror` → toast "Conexão interrompida".
- **`renderResult(r)`** — preenche:
  - título `"<Cultura> · <Estação> / <UF>"`;
  - **gauge**: circunferência `2π·56`, `strokeDashoffset = circ·(1 − score/100)`, cor = `verdict_color`;
  - alerta (se houver) e "pills" dos melhores meses;
  - **cards por mês** (classe = `status`, barra, temperatura/precipitação/umidade);
  - **tabela detalhada** (temperatura média/máx/mín, precipitação, umidade, viabilidade);
  - **gráficos** anuais via `renderChart` (temperatura em linha laranja, precipitação em barras azuis, umidade em linha roxa) e `renderScoreChart`;
  - chama `renderDetailedAnalysis(r)`.
- **`renderDetailedAnalysis(r)`** — renderiza:
  - **Perfil da cultura** (emoji, descrição, parágrafo, cards de usos/regiões/ciclo/fase crítica, chips de sensibilidades);
  - **3 cards de variáveis** coloridos por `status` (mapa `SC`: otimo/bom/atencao/critico → cores e rótulo Ideal/Bom/Atenção/Crítico);
  - **Diagnóstico** (converte `**texto**` em `<strong>`).
- **`scoreColor(s)`** — mesma escala de cores de `score_to_verdict` no backend.
- **`renderChart(id, type, labels, data, unit, color, label)`** — cria/recria um gráfico Chart.js (destrói a instância anterior guardada em `window.__chart_<id>`); linha com preenchimento ou barras arredondadas, tooltip temático.
- **`renderScoreChart(labels, data, color)`** — gráfico de barras dos scores dos meses selecionados, cada barra colorida por `scoreColor`, eixo Y de 0 a 100%.
- **`resetAnalysis()`** — oculta resultados e log e rola de volta ao formulário.

### 8.7 Utilitários JS
- **`fetchJSON(url, opts)`** — `fetch` que lança erro com `body.detail` ou `HTTP <status>`.
- **`showLoading(_on)`** — vazio (legado, substituído pelo terminal).
- **`showToast(msg)`** — mostra o toast de erro por 4 s.

---

## 9. Scripts e configuração

### `iniciar.bat` (inicializador para Windows)
1. Verifica se `ngrok.exe` existe na pasta (senão, erro e sai).
2. Se `data\climate_normals.pkl` **não** existir: roda `preprocess.py` e `train_model.py`. Caso contrário, pula.
3. Mata instâncias antigas do ngrok.
4. Sobe a API minimizada (`python api\main.py`, janela "AgroClima-API") e espera 5 s.
5. Sobe `ngrok http 8000` minimizado e espera 6 s.
6. Via PowerShell, consulta `http://localhost:4040/api/tunnels`, mostra a URL pública, **copia para a área de transferência** e abre `http://localhost:8000` no navegador.
7. Ao pressionar uma tecla, encerra a API, o ngrok e **todos os processos `python.exe`** da máquina.

> Se só o modelo precisar ser re-treinado (e `climate_normals.pkl` já existir), o `.bat` não re-treina: rode `python api\train_model.py` manualmente.

### `teste.bat`
Imprime "funcionou" e pausa (teste trivial de execução de `.bat`).

### `requirements.txt`
`fastapi>=0.100.0`, `uvicorn>=0.20.0`, `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.2.0`, `joblib>=1.2.0`, `python-multipart>=0.0.6` (este último não é necessário pelo código atual).

### `.gitignore`
Ignora `BASE/` (3,5 GB de CSVs), `ngrok.exe`, `__pycache__/`, `*.pyc`, `.env`, `.vscode/`, `.idea/`.

---

## 10. Artefatos gerados (esquemas e números reais)

Valores medidos nos arquivos atuais do repositório:

| Artefato | Linhas | Colunas |
|---|---|---|
| `data/stations.pkl` | **473 estações** | `codigo, nome, uf, regiao, lat, lon, alt` |
| `data/monthly_climate.pkl` | **35.280** registros mensais | `year, month, precip_total, temp_media, temp_max, temp_min, umidade_media, n_horas, codigo, nome, uf, regiao` |
| `data/climate_normals.pkl` | **5.621** normais (estação × mês) | `codigo, month, precip_norm, precip_std, temp_norm, temp_std, umidade_norm, umidade_std, temp_max_norm, temp_min_norm, n_anos, nome, uf, regiao, lat, lon` |
| Dataset de treino (em memória) | **56.210** exemplos | 15 features + `score` |
| `api/models/recommendation_model.joblib` | ~47 MB | `model, le_uf, le_regiao, le_crop, le_codigo, feature_cols, crop_requirements` |

- **UFs cobertas:** 27 (todas): AC, AL, AM, AP, BA, CE, DF, ES, GO, MA, MG, MS, MT, PA, PB, PE, PI, PR, RJ, RN, RO, RR, RS, SC, SE, SP, TO.
- **Regiões:** CO, N, NE, S, SE.
- **Anos presentes nos `.pkl` atuais:** 2005, 2007, 2008, 2009, 2013.
- **`n_anos` por normal:** mínimo 1, mediana 3, média 3,1, máximo 5.
- **Classes do `le_crop`** (ordem alfabética = código numérico): algodao=0, arroz=1, cafe=2, cana=3, feijao=4, mandioca=5, milho=6, soja=7, sorgo=8, trigo=9.

---

## 11. Exemplo real de execução

`POST /api/recommend` com `{"station_code": "A001", "crop": "soja", "months": [10, 11, 12]}` (Brasília/DF). Resposta real (textos longos e listas encurtados com "…"):

```json
{
  "station_code": "A001",
  "station_name": "BRASILIA",
  "uf": "DF",
  "crop": "soja",
  "crop_name": "Soja",
  "overall_score": 59.5,
  "verdict": "Moderado",
  "verdict_color": "#eab308",
  "months": [
    {
      "month": 10, "month_name": "Outubro", "score": 61.9,
      "temp_media": 23.2, "precip_total": 88.1, "umidade_media": 58.7,
      "status": "moderado",
      "details": { "temp_max": 33.2, "temp_min": 15.7, "precip_std": 100.1, "n_anos": 5 }
    }
    /* … Novembro, Dezembro … */
  ],
  "best_months": ["Março", "Fevereiro", "Abril", "Novembro"],
  "alert": "Precipitação elevada (302.3mm) em Dezembro — risco de excesso hídrico.",
  "climate_summary": { "months": [1, 2, 3, "…"], "temp": [21.5, 21.9, 21.8, "…"], "precip": [269.8, 210.5, 172.8, "…"], "umidade": [79.1, 76.2, 76.4, "…"] },
  "crop_profile": { "usos": "…", "regioes": "…", "ciclo": "…", "fase_critica": "florescimento e enchimento de grãos", "sensibilidades": ["…"], "perfil": "…" },
  "var_analyses": [
    { "variavel": "Temperatura", "valor": 22.0, "unidade": "°C", "faixa_otima": "25–30 °C", "faixa_tolerada": "20–40 °C",
      "status": "atencao", "icone": "🌡️", "titulo": "Temperatura abaixo do ideal", "explicacao": "A temperatura média de 22.0°C está dentro da faixa tolerada, mas 3.0°C abaixo…" },
    { "variavel": "Precipitação", "valor": 211.3, "unidade": "mm", "faixa_otima": "100–200 mm", "faixa_tolerada": "60–300 mm",
      "status": "atencao", "icone": "🌦️", "titulo": "Precipitação elevada", "explicacao": "…" }
    /* … Umidade … */
  ],
  "diagnostico": "As condições climáticas de BRASILIA/DF no período de Outubro, Novembro, Dezembro são **moderadas** para o cultivo de Soja. …"
}
```

Note que "melhores meses" mede a **adequação climática de cada mês isolado** (março, fevereiro…), não a melhor data de semeadura considerando o ciclo da cultura.

---

## 12. Tabelas de regras de negócio

| Regra | Onde | Valor |
|---|---|---|
| Pesos da fórmula de aptidão | `compute_crop_score` | Temperatura 35%, precipitação 40%, umidade 25% |
| Bônus de mês favorável | `compute_crop_score` | ×1.1 |
| Penalidade de variabilidade da chuva | `compute_crop_score` | `1 − CV·0.3`, com `CV = precip_std/(precip_norm+1)` |
| Penalidade de variabilidade da temperatura | `compute_crop_score` | `1 − (temp_std/5)·0.2` |
| Valor quando a variável é NaN | `aptidao_score` | 0.5 |
| Filtro mínimo de qualidade mensal | `preprocess.run` | `n_horas ≥ 100` |
| Defaults de desvio ausente | `train` | precip 30, temperatura 2, umidade 8 |
| Score sem dados nem fallback por UF | `predict_score` | 50.0 |
| Limite do score | `predict_score` | [0, 100] |
| Score geral | `recommend` | média simples dos meses selecionados |
| Melhores meses | `recommend` | top 4 dos 12 meses com score ≥ 60 |
| Alerta | `recommend` | primeiro mês com temperatura > máx., precipitação < mín. ou precipitação > máx. absoluto |
| Faixas de veredito | `score_to_verdict` / `scoreColor` | 80 / 65 / 50 / 35 |
| Hiperparâmetros do RF | `train` | 200 árvores, `max_depth=12`, `min_samples_leaf=5`, `random_state=42` |

---

## 13. Limitações, inconsistências e bugs conhecidos

Pontos identificados na leitura do código. Úteis para a seção de "limitações/trabalhos futuros" do TCC ou para correções.

### Metodologia
1. **O alvo do modelo é sintético.** O Random Forest aprende a reproduzir `compute_crop_score()`, uma fórmula especialista, e não dados reais de produtividade. Por isso o R² de validação cruzada tende a ser alto, mas mede "o quão bem o RF imita a fórmula", não acurácia agronômica. Uma validação real usaria, por exemplo, produtividade municipal do IBGE/PAM ou o ZARC do MAPA.
2. **O modelo e a fórmula divergem.** No exemplo de Brasília (outubro, soja), a fórmula dá 50.3 e o RF prevê 61.9. O usuário vê o valor do RF.
3. **"Melhores meses" ignora o ciclo da cultura.** Cada mês é avaliado isoladamente; não há análise da janela de semeadura + ciclo (ex.: plantar em outubro implica que o florescimento cai em dezembro/janeiro).
4. **Variáveis não usadas:** radiação, vento, ponto de orvalho, solo, altitude (está em `stations.pkl`, mas não é feature), risco de geada (só indiretamente pela temperatura mínima).
5. **`lat`, `lon`, `uf_enc`, `regiao_enc`, `month` quase não têm importância** (< 0,2% cada), já que a fórmula não depende deles.

### Dados
6. **Período curto:** os `.pkl` atuais têm só os anos **2005, 2007–2009 e 2013** (mediana de 3 anos por normal). Normais climatológicas oficiais usam 30 anos. A pasta `base/` tem 5.755 CSVs (incluindo 2020–2026), então o preprocess parece ter sido rodado sobre um conjunto menor. Rodar `preprocess.py` + `train_model.py` de novo deve incorporar os anos recentes.
7. **`/api/stats` mostra `anos` fixo no código** (`"2020-2026"`), o que não corresponde aos dados carregados atualmente.
8. **Precipitação subestimada em meses com lacunas:** `precip_total` é a soma das horas válidas, e o filtro mínimo é de apenas 100 horas (~14% de um mês de 720 h). Um mês com muitas falhas aparece "mais seco" do que foi.
9. **`temp_max`/`temp_min` são extremos absolutos do mês** (máx./mín. horário), e as normais são médias desses extremos. A tabela do front os rotula como "Temp Máx/Mín", o que pode ser lido como média das máximas diárias.
10. Em `run()`, se a mesma estação aparecer com metadados diferentes em anos distintos, **vale o primeiro arquivo lido**.

### Código / API
11. **Meses sem normal na estação:** `predict_score` usa o fallback por UF, mas `get_climate_for_month` retorna `{}`, então o `MonthResult` fica com temperatura/precipitação/umidade = 0. Isso contamina as médias de `var_analyses` (podendo gerar "crítico" falso) e pode disparar o alerta "Precipitação baixa (0mm)".
12. **Indexação booleana encadeada** em `predict_score` (`_normals[_normals["uf"] == uf][_normals["month"] == month]`) gera `UserWarning` do pandas. Funciona, mas o certo seria uma única máscara `(…) & (…)`.
13. **UF/região desconhecidas viram 0**, que coincide com uma classe real do encoder (ex.: `AC`).
14. **A mensagem SSE `scoring`** diz "Ponderando temperatura (35%), precipitação (40%) e umidade (25%)", mas o score geral é a média das previsões do RF. Os pesos só existem na fórmula usada para treinar.
15. **As pausas do SSE são artificiais** (~7 s ou mais). O texto "Lendo arquivos de dados brutos" é cosmético: os dados já estão em memória.
16. **`recommend_stream` recalcula tudo:** chama `predict_score` nos meses e depois `recommend()`, que recalcula os mesmos meses e mais os 12 do ano.
17. **Duplicação de regras:** as faixas das culturas existem em `CROP_REQUIREMENTS` (Python) e em `LITERATURA` (JS). Hoje estão iguais, mas alterar uma sem a outra deixa o modal de literatura desatualizado.
18. **`@app.on_event("startup")` está obsoleto** em versões novas do FastAPI (o recomendado é `lifespan`).
19. **Código morto / imports não usados:** `COLS_USE`, `math`, `FileResponse`, `MONTH_ABBR`, `le_codigo`, a variável `otimos`, `showLoading`, o endpoint `/api/climate/{code}` (não chamado pelo front), o status `bom` em `SC` e `GradientBoostingRegressor`, `StandardScaler`, `Pipeline`, `r2_score`.
20. **Segurança/operação:** CORS `*` + ngrok deixam a API pública sem autenticação (aceitável para demonstração). O front monta HTML com `innerHTML` usando dados do servidor (risco baixo, já que os dados são locais). O `iniciar.bat` mata **todos** os `python.exe` da máquina ao encerrar.
21. **Emojis inconsistentes:** feijão usa 🥔 (batata), algodão usa 👕, e soja e sorgo compartilham 🌿.
22. **As métricas de treino (R², MAE) só são impressas**, nunca persistidas. Para o TCC, convém salvá-las (e, idealmente, avaliar em um conjunto separado por estação, evitando vazamento entre meses da mesma estação nos folds).

---

## 14. Como executar

```bash
pip install -r requirements.txt

# 1) ETL (precisa dos CSVs em base/ ou BASE/)
python api/preprocess.py

# 2) Treino do modelo
python api/train_model.py

# 3) API + frontend em http://localhost:8000
python api/main.py
# (ou, de dentro de api/:  uvicorn main:app --host 0.0.0.0 --port 8000)
```

No Windows, `iniciar.bat` faz tudo e ainda expõe a aplicação via ngrok.

**Teste rápido da API:**
```bash
curl http://localhost:8000/api/stats
curl "http://localhost:8000/api/stations?uf=DF"
curl -X POST http://localhost:8000/api/recommend -H "Content-Type: application/json" \
     -d '{"station_code":"A001","crop":"soja","months":[10,11,12]}'
curl -N "http://localhost:8000/api/recommend/stream?station_code=A001&crop=soja&months=10,11,12"
```

---

## 15. Glossário

| Termo | Significado |
|---|---|
| **INMET** | Instituto Nacional de Meteorologia, fonte dos dados horários das estações automáticas |
| **Código WMO** | Identificador da estação (ex.: `A001` = Brasília) |
| **Normal climatológica** | Média histórica de uma variável para um mês em um local (aqui, média entre os poucos anos disponíveis) |
| **Faixa ótima / tolerada** | Intervalo ideal / intervalo suportável de uma variável para a cultura |
| **Score de aptidão / viabilidade** | Nota de 0 a 100 da adequação climática de um mês para uma cultura |
| **Função trapezoidal** | Função de pertinência: 1 na faixa ótima, cai linearmente até 0 nos limites absolutos |
| **Random Forest** | Conjunto de árvores de decisão cuja previsão é a média das árvores |
| **Modelo substituto (surrogate)** | Modelo de ML treinado para imitar uma função conhecida (aqui, a fórmula especialista) |
| **SSE (Server-Sent Events)** | Protocolo HTTP de streaming do servidor para o navegador, usado no "terminal" de progresso |
| **Veranico** | Período de estiagem (mais de ~15 dias sem chuva) dentro da estação chuvosa |
| **Safra verão** | Janela de outubro a março, atalho de seleção de meses no front |
| **ZARC** | Zoneamento Agrícola de Risco Climático (MAPA/Embrapa), referência oficial para épocas de plantio |
