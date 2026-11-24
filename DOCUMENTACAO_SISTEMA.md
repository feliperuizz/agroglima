# AgroClima — Documentação Técnica Completa

> **Para quem lê (pessoa ou IA):** este documento descreve o sistema **AgroClima** como ele está **hoje no repositório**, função por função. Foi escrito a partir da leitura direta do código e conferido com a execução dos scripts em 24/11/2026. Nomes de arquivos, funções, variáveis e números estão exatamente como no código e nos arquivos de resultado. O AgroClima é um TCC da Faculdade de Computação e Informática do Mackenzie (2026): um sistema web que calcula uma **pontuação de adequação climática histórica, de 0 a 100 pontos**, para 10 culturas em estações automáticas do INMET.
>
> A seção 14 lista onde o código atual difere do texto do TCC entregue. A seção 16 resume a versão 1.0, que foi auditada no TCC.

---

## Sumário

1. [Visão geral](#1-visão-geral)
2. [Versões do sistema](#2-versões-do-sistema)
3. [Stack e estrutura de pastas](#3-stack-e-estrutura-de-pastas)
4. [Fluxo ponta a ponta](#4-fluxo-ponta-a-ponta)
5. [Dados de entrada: CSVs do INMET](#5-dados-de-entrada-csvs-do-inmet)
6. [`api/preprocess.py` — preparação dos dados](#6-apipreprocesspy--preparação-dos-dados)
7. [`api/train_model.py` — regra de pontuação e modelo](#7-apitrain_modelpy--regra-de-pontuação-e-modelo)
8. [`api/avaliacao.py` — experimentos do TCC](#8-apiavaliacaopy--experimentos-do-tcc)
9. [`api/main.py` — API FastAPI](#9-apimainpy--api-fastapi)
10. [`templates/index.html` — interface](#10-templatesindexhtml--interface)
11. [Arquivos estáticos e dados geográficos](#11-arquivos-estáticos-e-dados-geográficos)
12. [Artefatos gerados e números atuais](#12-artefatos-gerados-e-números-atuais)
13. [Exemplo real: estação A001, soja](#13-exemplo-real-estação-a001-soja)
14. [Diferenças entre o código atual e o TCC entregue](#14-diferenças-entre-o-código-atual-e-o-tcc-entregue)
15. [Limitações conhecidas](#15-limitações-conhecidas)
16. [Histórico: a versão 1.0 auditada](#16-histórico-a-versão-10-auditada)
17. [Como executar e publicar](#17-como-executar-e-publicar)
18. [Glossário](#18-glossário)

---

## 1. Visão geral

O usuário escolhe uma **estação meteorológica do INMET**, uma **cultura** (soja, milho, café, arroz, feijão, trigo, algodão, cana-de-açúcar, mandioca ou sorgo) e **um ou mais meses**. O sistema devolve:

- a **pontuação de cada mês** (0 a 100 pontos) e a **pontuação geral** da consulta;
- o **veredito** por faixas ("Altamente Recomendado" a "Não Recomendado");
- a **decomposição da nota** de cada mês: notas parciais de temperatura, chuva e umidade, bônus e penalidades;
- os **melhores meses do ano** para a cultura naquela estação;
- **alertas** para variáveis fora dos limites absolutos da cultura;
- um **resumo em uma frase**, uma análise textual por variável e um diagnóstico consolidado;
- os **anos de dados** usados em cada mês;
- **gráficos** anuais de temperatura, chuva, umidade e pontuação.

Há também um **mapa do Brasil** com a pontuação de todas as estações para uma cultura e um mês, que pode ser colorido pela pontuação ou pelo fator que mais limita a nota.

**Ideia central.** Os dados horários do INMET são agregados em médias mensais por estação e ano, e depois em **indicadores mensais históricos** (média e desvio entre os anos, por estação e mês do ano). Cada indicador é pontuado por uma **regra trapezoidal explícita**, com faixas agronômicas por cultura. Um **Random Forest** foi treinado para imitar essa regra e avaliado no TCC, mas **a API serve a regra exata**, não o modelo (ver seção 7.6).

> A pontuação é um índice adimensional de adequação climática histórica. **Não é probabilidade de sucesso, não é data de semeadura e não substitui o Zoneamento Agrícola de Risco Climático (ZARC)**, que simula balanço hídrico diário, solo e ciclo da cultivar.

---

## 2. Versões do sistema

| Versão | O que é | Onde está descrita |
|---|---|---|
| **1.0** | Protótipo entregue antes da auditoria. Servia a previsão do Random Forest e tinha falhas de leitura de dados. | TCC, seções 4.1 a 4.5; resumo na seção 16 deste documento |
| **1.1** | Versão corrigida descrita no TCC entregue: parser novo, sem duplicação, completude de 80%, chuva escalada, regra servida diretamente, testes de borda corrigidos. | TCC, seções 3 e 4 |
| **Código atual** | A 1.1 mais refinamentos feitos depois da entrega (regra de lacunas de 72 h, vizinha com altitude, média com inviáveis valendo 0) e melhorias de interface (mapa, explicação da nota, transparência dos dados). | Este documento; diferenças na seção 14 |

O rótulo exibido pela API continua "AgroClima 1.1" (`metodo` na resposta).

---

## 3. Stack e estrutura de pastas

| Camada | Tecnologia |
|---|---|
| Backend | Python 3.11, **FastAPI**, Uvicorn, Pydantic v2 |
| Dados | pandas 2.2.3, numpy 2.2.4 (tabelas em `.pkl`) |
| Modelo (avaliação) | scikit-learn 1.7.2 (`RandomForestRegressor`, `GroupKFold`, `KFold`, `LabelEncoder`), joblib |
| Interface | Uma página HTML, **Tailwind CSS** (CDN), **Chart.js 4.4.0**, **Leaflet 1.9.4**, JavaScript puro |
| Publicação | VM Ubuntu na Oracle Cloud (Always Free), nginx, systemd, certbot (HTTPS) |

```
tcc/
├── api/
│   ├── main.py              # API FastAPI: rotas, pontuação servida, textos, estação vizinha, mapa
│   ├── preprocess.py        # CSVs do INMET → tabelas .pkl (mensal, indicadores, estações)
│   ├── train_model.py       # Regra de pontuação (CROP_REQUIREMENTS, score_breakdown) e treino do RF
│   ├── avaliacao.py         # Experimentos do TCC (bases D0, D1, D2; validação; compensação; caso A001)
│   └── models/
│       ├── recommendation_model.joblib   # RF retreinado na base atual (~16 MB, compactado)
│       ├── metrics.json                  # Métricas do treino (GroupKFold por estação)
│       └── avaliacao.json                # Resultados completos de api/avaliacao.py
├── data/
│   ├── monthly_raw.pkl          # Meses antes do filtro de completude, com contagens de horas
│   ├── monthly_climate.pkl      # Meses que passaram no filtro (base usada)
│   ├── climate_normals.pkl      # Indicadores mensais históricos por estação × mês
│   ├── stations.pkl             # Cadastro das estações com indicador
│   ├── preprocess_summary.json  # Resumo do último pré-processamento
│   └── v1_0/                    # Tabelas da versão 1.0, preservadas para a auditoria
├── templates/index.html     # Interface completa (HTML + CSS + JS)
├── static/
│   ├── agroclimalogo.png
│   ├── br_estados.geojson   # Contorno dos 27 estados (simplificado)
│   └── municipios/XX.geojson  # Divisas municipais do IBGE, um arquivo por UF
├── deploy/
│   ├── setup.sh             # Instalação automática na VM Ubuntu
│   ├── agroclima.service    # Serviço systemd (Uvicorn em 127.0.0.1:8000)
│   └── nginx-agroclima.conf # Proxy reverso, pronto para SSE
├── BASE/ (ou base/)         # 5.755 CSVs brutos do INMET (~3,7 GB) — NÃO versionado
├── BASE_SAMPLE/             # Amostra de 287 CSVs, versionada
├── iniciar.bat              # Inicializador local no Windows (com ngrok)
├── requirements.txt
├── .gitignore / .gitattributes
└── DOCUMENTACAO_SISTEMA.md  # Este documento
```

Os CSVs brutos só são necessários para reprocessar. A API em funcionamento usa apenas os `.pkl` de `data/`.

---

## 4. Fluxo ponta a ponta

```
CSVs horários do INMET (BASE/, 1 arquivo por estação e ano, 2 formatos)
        │  python api/preprocess.py
        ▼
process_file(): metadados, colunas, faixa física, lacunas, agregação hora → mês
        ▼
monthly_raw.pkl ──► apply_completeness(): 80% das horas + lacuna ≤ 72 h nas 3 variáveis; chuva escalada
        ▼
monthly_climate.pkl ──► compute_normals(): média e desvio entre anos por estação × mês
        ▼
climate_normals.pkl + stations.pkl
        │                                   │  python api/train_model.py (opcional, avaliação)
        │                                   ▼
        │                       RF treinado para imitar a regra + metrics.json
        │
        │  python api/main.py (Uvicorn)
        ▼
FastAPI: carrega as tabelas na memória → aplica score_breakdown() a cada consulta
        ▼
index.html: formulário, mapa, resultado, explicação da nota
```

---

## 5. Dados de entrada: CSVs do INMET

Arquivos das estações meteorológicas automáticas, um por estação e ano, por exemplo `INMET_CO_DF_A001_BRASILIA_01-01-2021_A_31-12-2021.CSV`. Codificação latin-1, separador `;`, vírgula decimal e 8 linhas de metadados antes do cabeçalho.

| Item | Até 2019 | A partir de 2020 |
|---|---|---|
| Coluna de data | `DATA (YYYY-MM-DD)`, valores `2009-01-01` | `Data`, valores `2021/01/01` |
| Coluna de hora | `HORA (UTC)`, valores `00:00` | `Hora UTC`, valores `0000 UTC` |
| Chaves de metadados | `ESTAÇÃO`, `REGIÃO` (com acento) | `ESTACAO`, `REGIAO` (sem acento) |
| Valor ausente | `-9999` | campo vazio |

Acervo local: **5.755 arquivos**, anos 2005, 2007–2009, 2013 e 2020–2026.

Colunas usadas: precipitação total horária, temperatura do ar (bulbo seco), temperaturas máxima e mínima da hora anterior e umidade relativa do ar. As demais (pressão, radiação, vento, ponto de orvalho) são ignoradas.

---

## 6. `api/preprocess.py` — preparação dos dados

### Constantes

| Nome | Valor | Papel |
|---|---|---|
| `BASE_DIR` | `<raiz>/BASE` | Pasta dos CSVs brutos (no Windows, `base/` também funciona) |
| `DATA_DIR` | `<raiz>/data` | Saída das tabelas |
| `COMPLETUDE_MIN` | `0.80` | Fração mínima de horas válidas por variável |
| `GAP_MAX_HORAS` | `72` | Maior lacuna seguida admitida em cada variável no mês |
| `VARIAVEIS` | precip, temp, temp_max, temp_min, umidade | Colunas numéricas lidas |
| `LIMITES_FISICOS` | chuva 0–150 mm/h; temperaturas −15–50 °C; umidade 0–100% | Fora da faixa, o valor vira ausente |

### Funções

- **`normalize(text)`**: remove acentos e espaços extras e passa para maiúsculas (`ESTAÇÃO` → `ESTACAO`). É o que permite ler os dois formatos.
- **`parse_header(filepath)`**: lê as 8 linhas de metadados e devolve um dicionário com chaves normalizadas (`CODIGO (WMO)`, `ESTACAO`, `UF`, `REGIAO`, `LATITUDE`, `LONGITUDE`, `ALTITUDE`).
- **`parse_float(val)`**: converte um valor isolado com vírgula decimal; `-9999` vira `NaN`. Mantida por compatibilidade.
- **`to_numeric(series)`**: versão vetorizada de `parse_float` para colunas inteiras.
- **`map_columns(columns)`**: mapeia os nomes das colunas do INMET para `data`, `hora`, `precip`, `temp`, `temp_max`, `temp_min` e `umidade`, por palavras-chave normalizadas.
- **`max_gaps(df)`**: completa a série hora a hora (horas sem linha no arquivo contam como lacuna) e devolve, por ano e mês, a maior sequência de horas seguidas sem leitura válida de chuva, temperatura e umidade (`gap_precip`, `gap_temp`, `gap_umid`).
- **`process_file(filepath)`**: processa um CSV:
  1. lê metadados e dados (`-9999` e campo vazio como ausentes);
  2. converte a data dos dois formatos e o horário (`00:00` ou `0000 UTC`) para o instante `ts`;
  3. aplica `LIMITES_FISICOS` e conta as leituras descartadas em `n_fora_faixa`;
  4. agrega por ano e mês: `precip_total` (soma com `min_count=1`, de modo que um mês sem chuva válida fica `NaN`, e não 0), `temp_media`, `temp_max` (máximo do mês), `temp_min` (mínimo do mês), `umidade_media`, as contagens `n_horas` (temperatura), `n_precip` e `n_umid`, e `horas_esperadas` (dias exatos do mês × 24, inclusive fevereiro bissexto);
  5. sem as colunas de máxima e mínima da hora anterior, usa o extremo da temperatura horária;
  6. junta as lacunas de `max_gaps` e devolve `(station_info, monthly)`. Em erro, imprime o arquivo e devolve `(None, None)`.
- **`apply_completeness(df_monthly, limiar=0.80, gap_max=72)`**: filtro **conjunto**. O registro estação-ano-mês inteiro só entra se temperatura, chuva e umidade tiverem, cada uma, pelo menos 80% de `horas_esperadas` **e** nenhuma lacuna seguida maior que 72 h. Nos meses mantidos, guarda a soma bruta em `precip_soma` e **escala a chuva**: `precip_total = precip_soma × horas_esperadas / n_precip`, supondo que as horas faltantes choveram, em média, como as observadas.
- **`compute_normals(df_monthly, df_stations)`**: agrupa por estação e mês e calcula `precip_norm`/`precip_std`, `temp_norm`/`temp_std`, `umidade_norm`/`umidade_std` (média e desvio amostral entre anos; o desvio é indefinido com um único ano), `temp_max_norm` e `temp_min_norm` (média, entre os anos, dos extremos absolutos de cada mês) e `n_anos` (anos distintos). Junta nome, UF, região e coordenadas da estação.
- **`list_csv_files(base_dir)`**: lista cada CSV **uma única vez**, qualquer que seja a caixa da extensão. A versão 1.0 buscava `*.CSV` e `*.csv` separadamente e, no Windows, lia tudo em dobro.
- **`run()`**: executa tudo. Se um mesmo estação-ano-mês aparece em dois arquivos, fica o registro com mais horas. Estações sem nenhum mês completo saem do cadastro. Grava os quatro `.pkl` e o `preprocess_summary.json`.

Como o filtro é conjunto, **nenhum indicador da base atual tem variável ausente**, e `n_anos` é o mesmo para as três variáveis.

---

## 7. `api/train_model.py` — regra de pontuação e modelo

### 7.1 `CROP_REQUIREMENTS`

Cada cultura tem quatro limites por variável no formato **(mínimo absoluto *a*, ótimo mínimo *b*, ótimo máximo *c*, máximo absoluto *d*)**, mais os meses favoráveis (que dão o bônus):

| Cultura | Temperatura (°C) | Chuva (mm/mês) | Umidade (%) | Meses favoráveis |
|---|---|---|---|---|
| Soja | 20/25/30/40 | 60/100/200/300 | 50/60/85/95 | 9, 10, 11, 12, 1 |
| Milho | 10/24/30/40 | 50/80/180/280 | 45/55/80/90 | 9, 10, 11, 12, 1, 2 |
| Café | 18/19/22/28 | 80/100/180/250 | 50/60/80/90 | 9, 10, 11 |
| Arroz | 20/22/33/38 | 100/125/200/300 | 60/70/90/98 | 10, 11, 12, 1 |
| Feijão | 15/18/26/33 | 40/70/130/200 | 50/55/75/85 | 1, 2, 6, 7, 8 |
| Trigo | 10/15/20/32 | 30/50/100/150 | 50/55/75/85 | 4, 5, 6, 7 |
| Algodão | 18/22/30/38 | 60/90/160/250 | 45/50/70/80 | 10, 11, 12, 1 |
| Cana-de-açúcar | 18/22/30/38 | 100/150/250/350 | 55/65/85/95 | 8, 9, 10, 11 |
| Mandioca | 18/22/28/35 | 50/80/150/250 | 50/60/80/90 | 9, 10, 11 |
| Sorgo | 12/20/32/40 | 40/60/130/200 | 40/50/75/85 | 10, 11, 12, 1, 2 |

**Proveniência (TCC, seção 4.2):** só a soja foi conferida por completo contra fontes oficiais. Coincidem o máximo de 40 °C e a ordem de grandeza da chuva, mas o mínimo de 20 °C é mais rígido que a literatura (crescimento nulo só abaixo de 10 °C), e não há fonte oficial para a faixa de umidade. Nas outras nove culturas, o código cita no máximo a unidade da Embrapa, e os valores são tratados como **parâmetros empíricos do protótipo**. As mesmas faixas aparecem no modal "Literatura Agronômica" da interface (objeto `LITERATURA`); a conferência de 24/11/2026 confirmou que são idênticas às do código.

### 7.2 `aptidao_score(val, a, b, c, d)` — nota parcial trapezoidal

```
s = 0                  se x ≤ a ou x ≥ d    (fora dos limites absolutos: zona crítica)
s = (x − a)/(b − a)    se a < x < b         (zona de atenção, subindo)
s = 1                  se b ≤ x ≤ c         (faixa ótima)
s = (d − x)/(d − c)    se c < x < d         (zona de atenção, descendo)
s = 0,5                se x é ausente       (não ocorre na base atual)
```

### 7.3 `score_breakdown(row, crop)` — pontuação S

```
base = 0,35·s_temp + 0,40·s_precip + 0,25·s_umid      (PESOS)
B    = 1,1 em mês favorável; 1,0 nos demais
P_p  = max(0; 1 − 0,3 × precip_std / (precip_norm + 1))   penaliza chuva irregular entre anos
P_t  = max(0; 1 − 0,2 × temp_std / 5)                    penaliza temperatura irregular
S    = min(100; 100 × base × B × P_p × P_t)              arredondado a 1 casa
veto = alguma nota parcial igual a 0
```

Devolve `s_temp`, `s_precip`, `s_umid`, `base`, `bonus`, `pen_precip`, `pen_temp`, `score` e `veto`, todos em tipos nativos do Python. **`compute_crop_score(row, crop)`** devolve só o `score`.

Pesos, bônus e penalidades são escolhas de projeto. O peso maior da chuva acompanha a evidência de que o déficit hídrico domina as perdas da soja (Sentelhas et al., 2015).

**Propriedades documentadas no TCC:**
- **A soma é compensatória.** Com a chuva fora dos limites e as outras variáveis ótimas num mês favorável, S chega a 66 pontos. Por isso existe o `veto`, que marca o mês como inviável na API.
- **Desvios indefinidos** (células de um único ano) recebem os valores padrão `STD_DEFAULTS = {precip_std: 30, temp_std: 2, umidade_std: 8}`. Os 30 mm são absolutos e entram numa fórmula relativa: com `n_anos = 1`, *P_p* = 0 sempre que `precip_norm` ≤ 8 mm.
- **Só 7 das 15 variáveis do modelo entram em S:** `temp_norm`, `precip_norm`, `umidade_norm`, `temp_std`, `precip_std`, a cultura (que define os limites e os meses favoráveis) e o mês (que define *B*).

### 7.4 Dados de treino e modelo

- **`build_training_data(normals)`**: uma linha por indicador × cultura, com as 15 variáveis e o alvo `score`. Na base atual: 7.103 × 10 = **71.030 exemplos**.
- **`FEATURE_COLS`** (15, nesta ordem): `lat, lon, month, crop_enc, uf_enc, regiao_enc, temp_norm, temp_std, temp_max_norm, temp_min_norm, precip_norm, precip_std, umidade_norm, umidade_std, n_anos`.
- **`RF_PARAMS`**: `n_estimators=200, max_depth=12, min_samples_leaf=5, max_features=1.0, random_state=42` (`n_jobs=-1`). Com `max_features=1.0`, padrão do scikit-learn para regressão, cada divisão avalia as 15 variáveis.
- **`load_normals()`**: carrega os indicadores e aplica `STD_DEFAULTS`.
- **`encode(df)`**: `LabelEncoder` para UF, região, cultura e estação. Os códigos inteiros impõem uma ordem arbitrária.
- **`group_cv(df, groups, n_splits=5, feature_cols, splitter=None)`**: validação cruzada em que **um RF novo é treinado em cada parte** e a linha de base (média de S por cultura e mês) é calculada só nas partes de treino. Devolve R² e MAE médios, por parte e por cultura. Aceita `GroupKFold` (padrão) ou outro divisor, como `KFold`.
- **`train()`**: valida por estação, treina o modelo final com todas as estações e grava o `.joblib` (compactado) e o `metrics.json`.

### 7.5 Por que o modelo não é usado na API

O alvo do RF é a própria regra. Mesmo validado por estação, ele erra em média 4,5 pontos em relação a uma fórmula que pode ser calculada exatamente, com as mesmas entradas e sem custo. Um R² alto mede **imitação da regra**, não acerto de época de plantio. Por isso a API serve `score_breakdown()` diretamente, e o RF é mantido só como objeto de avaliação.

---

## 8. `api/avaliacao.py` — experimentos do TCC

Reproduz todos os números do TCC e grava `api/models/avaliacao.json`. Compara três bases:

| Base | Origem |
|---|---|
| **D0** | Tabelas da versão 1.0 (`data/v1_0/`), com as linhas duplicadas |
| **D1** | D0 sem duplicatas, para isolar o efeito do erro de leitura dupla |
| **D2** | Base atual (`data/`) |

| Função | O que faz |
|---|---|
| `normals_v10(monthly, stations)` | Recalcula os indicadores exatamente como a versão 1.0 |
| `verdict(score)` | Faixas de veredito (80/65/50/35) |
| `rule_table(normals)` | Decomposição da regra para cada estação × mês × cultura |
| `auditoria_duplicacao()` | Linhas duplicadas, fator do desvio, efeito em S e nos vereditos (D0 × D1) |
| `chuva_toda_ausente()` | Meses da 1.0 sem nenhuma hora de chuva válida (registrados como 0 mm) |
| `resumo_base(normals, monthly)` | Estações, indicadores, anos, `n_anos` |
| `compensacao(normals)` | Combinações com nota 0 e S alto; comparação com o operador de mínimo (Lei de Liebig) |
| `avaliar_modelo(normals, nome, extra)` | `GroupKFold` por estação e por UF, `KFold` aleatório e, em D2, a variante com as 7 variáveis da regra (`RULE_FEATURES`) |
| `caso_a001(n0, n1, n2)` | Caso A001, soja, outubro a dezembro, nas três bases |
| `main()` | Executa tudo e grava o JSON, com data e versões das bibliotecas |

---

## 9. `api/main.py` — API FastAPI

### 9.1 Inicialização

- Constantes de caminho: `BASE_ROOT`, `DATA_DIR`, `TEMPLATES_DIR`, `STATIC_DIR`.
- **Estação vizinha:** `VIZINHA_MAX_KM = 150`, `VIZINHA_MIN_ANOS = 3`, `VIZINHA_MAX_DALT_M = 500`, `GRADIENTE_TERMICO = 0,0065` °C/m.
- **`load_artifacts()`**: carrega `climate_normals.pkl` (com `STD_DEFAULTS` aplicados e a altitude da estação), `stations.pkl` e `monthly_climate.pkl` em variáveis globais. Não carrega o modelo.
- **`lifespan`** chama `load_artifacts()` ao subir o servidor.
- Middlewares: **GZip** (respostas acima de 1 KB, o que reduz bastante os GeoJSON e o mapa) e **CORS** aberto. `/static` serve a pasta `static/`.

### 9.2 Modelos Pydantic

- **`RecommendRequest`**: `station_code`, `crop`, `months` (lista com pelo menos um mês, cada um de 1 a 12; fora disso, erro 422).
- **`MonthResult`**: `month`, `month_name`, `score` (pode ser `None`), `temp_media`, `precip_total`, `umidade_media`, `status` (`excelente`, `bom`, `moderado`, `ruim`, `critico`, `inviavel` ou `sem_dados`), `details` e `fonte`.
  - `details`: `temp_max`, `temp_min`, `precip_std`, `temp_std`, `n_anos`, `s_temp`, `s_precip`, `s_umid`, `base`, `bonus`, `pen_precip`, `pen_temp`, `veto` e `formula` (a conta em texto).
  - `fonte`: `{tipo: "estacao"}`, ou `{tipo: "vizinha", codigo, nome, distancia_km, dif_altitude_m, n_anos}`, ou `{tipo: "indisponivel"}`.
- **`VarAnalysis`**: `variavel`, `valor`, `unidade`, `faixa_otima`, `faixa_tolerada`, `status` (`otimo`, `atencao` ou `critico`), `icone`, `titulo`, `explicacao`, `mes_referencia` e `faixa_observada`.
- **`RecommendResponse`**: dados da estação e da cultura, `overall_score`, `verdict`, `verdict_color`, `months`, `best_months`, `alert` (o primeiro), `alerts` (todos), `climate_summary`, `crop_profile`, `var_analyses`, `diagnostico`, `metodo`, `periodo_dados`, `anos_usados` e `n_anos_por_mes`.

### 9.3 Funções auxiliares

| Função | O que faz |
|---|---|
| `fmt(value, nd)` | Número no padrão brasileiro (`88.12` → `88,1`) |
| `score_to_verdict(score)` | ≥ 80 Altamente Recomendado (verde); ≥ 65 Recomendado; ≥ 50 Moderado; ≥ 35 Risco Elevado; abaixo, Não Recomendado (vermelho) |
| `score_to_status(score)` | As mesmas faixas como classes CSS dos cartões |
| `haversine_km(...)` | Distância de grande círculo (raio de 6.371 km) |
| `station_period(code)` | Texto como "2005-2026, 12 anos com ao menos um mês completo" |
| `CROP_PROFILES` | Textos por cultura: usos, regiões produtoras, ciclo, fase crítica, sensibilidades e perfil |
| `TEXTOS`, `SEVERIDADE`, `_zona()` | Frases da análise por variável, nas zonas crítica, atenção e ótima, coerentes com a nota trapezoidal |
| `_explain_var(...)` | Analisa uma variável mês a mês e relata o **pior mês** selecionado, não a média (que esconderia riscos opostos) |
| `_diagnostico(...)` | Parágrafo consolidado em linguagem histórica: nível, fatores fora dos limites, meses inviáveis, regiões produtoras e aviso de que melhor mês não é data de semeadura |
| `fator_limitante(b)` | Variável com a menor nota parcial, ou `nenhum` quando todas valem 1 (usado no mapa) |

### 9.4 Pontuação de uma consulta

- **`lookup_normal(station, month)`**: devolve o indicador local. Sem indicador local, procura uma **estação vizinha**, decidindo mês a mês. São candidatas as estações com indicador naquele mês, `n_anos ≥ 3` e diferença de altitude de até 500 m. Vence a mais próxima por Haversine, até 150 km; em empate de distância, a de mais anos. As temperaturas da vizinha são corrigidas pela altitude (6,5 °C/km); chuva e umidade não são corrigidas. Sem candidata, devolve `indisponivel`.
- **`month_result(station, month, crop)`**: aplica `score_breakdown()` e monta o `MonthResult`. Com `veto`, o status vira `inviavel`; sem dado, `score = None` e status `sem_dados` (zeros nunca são exibidos como medição).
- **`get_annual_series(code)`**: série dos 12 meses para os gráficos.
- **`build_alerts(months, crop)`**: alertas de todos os meses com temperatura, chuva ou umidade fora dos limites absolutos, além dos avisos de mês sem dado e de estação vizinha usada.
- **`build_recommendation(req)`**: valida estação e cultura (404 ou 400) e pontua os meses.
  - A **pontuação geral** é a média dos meses pontuados, com os **meses inviáveis entrando com 0**; meses sem dado ficam fora (422 se nenhum mês tiver dado).
  - Os **melhores meses** são os até 4 meses do ano com maior S, sem veto e com S ≥ 60.
  - Monta também os alertas, as análises por variável, o diagnóstico, os anos usados e `n_anos_por_mes`.

### 9.5 Rotas

| Método e rota | Entrada | Saída |
|---|---|---|
| `GET /` | — | `templates/index.html` |
| `GET /api/stations` | `uf`, `search` (opcionais; a busca é literal, `regex=False`) | Lista de estações (`codigo, nome, uf, regiao, lat, lon, alt`) |
| `GET /api/crops` | — | Culturas com `key, nome, emoji, desc, temp, precip, umidade, meses_favoraveis` |
| `GET /api/map` | `crop`, `month` (1–12) | Para cada estação com indicador local no mês: `score`, `inviavel`, `n_anos`, `s_temp`, `s_precip`, `s_umid`, `limitante`, `temp`, `precip`, `umidade` e coordenadas |
| `GET /api/climate/{codigo}` | — | Série anual e período de dados da estação (404 se não existir) |
| `POST /api/recommend` | `RecommendRequest` | `RecommendResponse` |
| `GET /api/stats` | — | Estações, registros mensais, indicadores, culturas, UFs e **anos calculados da base** |
| `GET /api/recommend/stream` | `station_code`, `crop`, `months` ("10,11,12") | Server-Sent Events com as etapas `init`, `station`, `records`, `crop`, `rule_month` (uma por mês), `complete` e `result`. Valida a entrada antes do fluxo e não tem pausas simuladas. **A interface atual não usa esta rota** (usa `POST /api/recommend`); ela foi mantida porque o TCC a descreve. |

---

## 10. `templates/index.html` — interface

Página única, tema escuro, servida em `/`.

### 10.1 Seções, de cima para baixo

| Seção | Conteúdo |
|---|---|
| Hero | Logo, menu (Mapa, Análise, Sobre), selo e cartões com estatísticas reais de `/api/stats` |
| Como funciona (`#sobre`) | Três cartões (dados INMET, regra explícita, recomendação) e o botão do modal **Literatura Agronômica**, com faixas e referências por cultura |
| **Mapa** (`#mapa`) | Cultura, mês e modo de cor ("pontuação" ou "fator que mais limita"); mapa Leaflet com legenda e contagens |
| Análise (`#analise`) | UF, estação, cultura, meses (atalhos: todos, limpar, safra de verão) e botão de análise |
| Resultado (`#resultado`) | Cabeçalho com medidor circular, veredito, **resumo em uma frase**, alertas, aviso de escopo (com link para o ZARC) e melhores meses; faixa explicando os fatores e pesos; **cartões por mês** clicáveis com barras das três notas e selo do fator limitante; **dados usados nesta estação**; gráficos; tabela com a conta de cada mês; perfil da cultura; análise por variável; diagnóstico |
| Modal **Como a nota foi calculada** | Três trapézios desenhados em SVG, com o valor do mês marcado, e a conta passo a passo; avisos de mês inviável ou com poucos anos |

### 10.2 Mapa

- Contorno e **siglas dos 27 estados** sempre visíveis (`static/br_estados.geojson`).
- A partir do zoom 5,5, **divisas municipais** do IBGE, carregadas só para os estados visíveis (`static/municipios/XX.geojson`), numa camada abaixo das estações.
- A partir do zoom 6, **nomes das cidades** das estações.
- Zoom pela rodinha do mouse; os pontos melhores são desenhados por cima.
- Ao passar o mouse, aparecem nome e nota. Ao clicar, abre um popup com as três barras e o botão **"Analisar esta estação"**, que preenche o formulário e roda a análise.
- No modo "fator que mais limita", a cor indica a variável de menor nota (azul = chuva, laranja = temperatura, roxo = umidade, verde = nenhuma), e a legenda conta as estações de cada grupo.
- O fundo (CARTO, OpenStreetMap) é externo. Se não carregar, os contornos locais mantêm o mapa legível.

### 10.3 Funções JavaScript

| Grupo | Funções |
|---|---|
| Inicialização e formulário | `init`, `loadStats`, `loadStations`, `filterStations`, `loadCrops`, `selectCrop`, `buildMonthGrid`, `selectMonths` |
| Literatura | `openLiteratura`, `closeLiteratura`, `closeLitOnBg`, `buildLitTabs`, `rangeBar`, `renderLitCrop` (dados em `LITERATURA`) |
| Análise | `analyze` (chama `POST /api/recommend`, com carregamento no botão), `renderResult`, `renderDetailedAnalysis`, `resetAnalysis` |
| Leitura rápida dos fatores | `FATORES`, `zonaCor`, `limitanteMes`, `fatorBars`, `chipLimitante`, `renderResumo` |
| Mapa | `initMap`, `loadMap`, `mapLegend`, `barraPopup`, `analyzeFromMap`, `atualizarRotulos`, `atualizarMunicipios`, `nomeCidade` (constantes `ZOOM_NOMES`, `ZOOM_MUNICIPIOS`, `COR_FATOR`) |
| Explicação da nota | `trapSVG`, `openExplain`, `closeExplain` |
| Transparência | `renderDataCard` |
| Gráficos | `scoreColor`, `renderChart`, `renderScoreChart` |
| Utilitários | `fetchJSON` (mostra a mensagem de erro da API, inclusive as de validação 422), `fmtBR`, `showToast` |

---

## 11. Arquivos estáticos e dados geográficos

| Arquivo | Origem | Observação |
|---|---|---|
| `static/br_estados.geojson` | Code for America, *click_that_hood* (brazil-states) | Coordenadas arredondadas para 2 casas (~890 KB) |
| `static/municipios/XX.geojson` | IBGE, API de malhas territoriais (`servicodados.ibge.gov.br/api/v3/malhas`), qualidade mínima | 27 arquivos, 5.570 municípios, ~2,8 MB no total, coordenadas com 3 casas |
| `static/agroclimalogo.png` | Projeto | Logo |

---

## 12. Artefatos gerados e números atuais

Valores da execução de 24/11/2026 (`preprocess_summary.json`, `metrics.json` e `avaliacao.json`).

### 12.1 Base de dados (D2)

| Item | Valor |
|---|---|
| Arquivos lidos | 5.755 de 5.755 |
| Anos efetivos | 2005, 2007, 2008, 2009, 2013 e 2020–2026 (12) |
| Leituras fora da faixa física | 19.507 |
| Meses brutos → completos | 61.640 → **39.882** (594 excluídos só pela regra de lacuna de 72 h) |
| Chuva escalada | 14.725 meses (36,9%); fator médio de 1,011; máximo de 1,25 |
| Estações com indicador | **637** em 27 UFs |
| Indicadores estação-mês | **7.103** (541 das 7.644 combinações possíveis estão ausentes) |
| Anos por indicador | 1 a 12; mediana de **6**; média de 5,6 |
| Células de um único ano | 431 |
| Indicadores com 10 anos ou mais | 448 |

### 12.2 Modelo × regra (D2, R² e MAE médios de 5 partes)

| Validação | R² | MAE (pontos) |
|---|---|---|
| Floresta, `GroupKFold` por estação | **0,922** | **4,47** |
| Floresta, `GroupKFold` por UF | 0,913 | 4,71 |
| Floresta, `KFold` aleatório | 0,925 | 4,39 |
| Floresta só com as 7 variáveis da regra, por estação | 0,924 | 4,39 |
| Linha de base (média por cultura e mês), por estação | 0,289 | 15,22 |
| Ajuste interno do modelo final (71.030 exemplos) | 0,935 | 3,97 |

O MAE por cultura vai de 3,17 (trigo) a 5,76 (café). A regra, aplicada diretamente, tem erro zero por construção.

### 12.3 Efeito compensatório da regra (D2)

De 71.030 combinações estação-mês-cultura, **37.824** têm ao menos uma nota parcial 0. Destas, **249** passam de 65 pontos ("Recomendado"), e a maior chega a 78,6. Com o operador de mínimo (EcoCrop), o veredito mudaria em 41.213 combinações, e os casos "Não Recomendado" passariam de 19.246 para 49.407.

### 12.4 Estação vizinha

Das 541 combinações sem indicador local, **502** são cobertas por uma vizinha (mediana de 53,8 km) e **39** ficam sem pontuação.

### 12.5 Auditoria da versão 1.0 (D0 × D1, não muda com reprocessamentos)

17.640 das 35.280 linhas eram duplicatas. O desvio-padrão caía pelo fator √(2(n−1)/(2n−1)), confirmado com diferença máxima de 10⁻¹⁴. As 565 células de um ano ficavam com desvio 0. A duplicação inflava S em **2,81 pontos** em média (15,48 nas células de um ano, 1,39 nas demais) e mudava o veredito de 7.339 dos 56.210 exemplos (**13,1%**). Havia ainda 207 meses sem nenhuma hora de chuva válida registrados como 0 mm.

---

## 13. Exemplo real: estação A001, soja

**Versão 1.0, outubro (base D0, 5 anos)**, a mesma conta da Tabela 4 do TCC:

| Etapa | Conta | Resultado |
|---|---|---|
| Nota da temperatura | (23,24 − 20)/(25 − 20) | 0,648 |
| Nota da chuva | (88,12 − 60)/(100 − 60) | 0,703 |
| Nota da umidade | (58,67 − 50)/(60 − 50) | 0,867 |
| Base ponderada | 0,35×0,648 + 0,40×0,703 + 0,25×0,867 | 0,725 |
| Bônus | outubro é favorável à soja | 1,1 |
| *P_p* | 1 − 0,3 × 100,07/89,12 | 0,663 |
| *P_t* | 1 − 0,2 × 1,215/5 | 0,951 |
| **S** | 100 × 0,725 × 1,1 × 0,663 × 0,951 | **50,3** |

A floresta da 1.0 previa 61,9 (11,6 acima). Em dezembro, 52,3 contra 33,5 pela regra.

**Código atual, `POST /api/recommend` com A001, soja e meses 10, 11 e 12:** pontuação geral de **57,8** ("Moderado"); outubro com 74,0 (10 anos, chuva média de 129,2 mm, dentro da faixa ótima), novembro com 51,9 e dezembro com 47,5. Melhores meses do ano: outubro, março e fevereiro. Os três meses perdem pontos principalmente pela temperatura abaixo do ideal (ótimo de 25–30 °C).

---

## 14. Diferenças entre o código atual e o TCC entregue

O TCC entregue descreve a versão 1.1. Depois da entrega, o código recebeu os ajustes abaixo, alguns em resposta ao terceiro parecer. Se a banca comparar o sistema com o texto, estas são as diferenças:

| Tema | TCC entregue | Código atual |
|---|---|---|
| Filtro dos dados | 4 regras (faixa física, 80% conjunto, chuva escalada, `n_anos`) | 5 regras: acrescenta **lacuna seguida máxima de 72 h** |
| Estação vizinha | Mais próxima por Haversine, até 150 km | Também exige **`n_anos` ≥ 3 e desnível ≤ 500 m** e **corrige a temperatura pela altitude** |
| Média de vários meses | Meses inviáveis entram com a própria nota | **Meses inviáveis entram com 0** |
| Interface | Formulário, resultado com a conta, terminal SSE | Também tem **mapa**, **explicação da nota**, **barras dos fatores**, **resumo em uma frase** e **dados usados**; o terminal SSE foi retirado |
| Números | 640 estações, 7.118 indicadores, 40.476 meses, R² 0,923 / MAE 4,45, 252 combinações, A001 = 57,6, vizinha 540/562 | 637, 7.103, 39.882, R² 0,922 / MAE 4,47, 249, A001 = 57,8, vizinha 502/541 |

As conclusões do TCC não mudam: a regra é reproduzível e explicável, o modelo só imita a regra, a base é curta e o sistema não substitui o ZARC. Os números da auditoria da versão 1.0 (seção 12.5) são idênticos.

Para mostrar exatamente a versão do TCC, é possível marcar no git o commit correspondente (por exemplo, `git tag v1.1-tcc <commit>`).

---

## 15. Limitações conhecidas

**De método (discutidas no TCC):**
- Cada mês é avaliado isoladamente, sem janela do ciclo da cultura; "melhor mês" não é data de semeadura.
- A soma ponderada é compensatória. O veto marca o mês e zera sua contribuição na média, mas a fórmula não foi trocada pelo operador de mínimo.
- As séries são curtas: mediana de 6 anos, máximo de 12, contra os 30 de uma normal climatológica.
- Os parâmetros só têm proveniência conferida para a soja; o mínimo térmico da soja (20 °C) é mais rígido que a literatura.
- O desvio padrão de 30 mm para células de um ano é absoluto numa fórmula relativa.
- A escala da chuva supõe que as falhas ocorrem ao acaso; falhas em temporais subestimam a chuva.
- A estação vizinha corrige só a temperatura; a chuva e a umidade da vizinha são usadas como estão.
- Sem solo, balanço hídrico nem validação agronômica com usuários ou especialistas.

**De software:**
- Sem testes automatizados; os casos de borda da Tabela 5 do TCC foram verificados manualmente e por scripts.
- API pública sem autenticação e com CORS aberto (adequado a uma consulta exploratória).
- Tailwind CSS, Chart.js, Leaflet e o fundo do mapa vêm de CDNs externas.
- `requirements.txt` inclui `python-multipart`, que o código não usa.
- O `iniciar.bat` exige o `ngrok.exe` na pasta e, ao encerrar, finaliza **todos** os processos `python.exe` da máquina.
- O CSS do antigo terminal (`.terminal-overlay`, `.log-section`) ainda está no HTML, sem uso.

---

## 16. Histórico: a versão 1.0 auditada

Problemas encontrados e corrigidos na 1.1 (TCC, seções 4.1 e 4.5):

1. **Leitura dupla:** `*.CSV` + `*.csv` no Windows liam cada arquivo duas vezes (17.640 linhas duplicadas), encolhendo os desvios e inflando S.
2. **Parser:** exigia "YYYY" no nome da coluna de data e descartava os 4.103 arquivos de 2020–2026. Só sobravam 2005, 2007–2009 e 2013.
3. **Completude:** 100 horas de temperatura bastavam (14% do mês), sem critério para chuva e umidade; chuva sem horas válidas virava 0 mm.
4. **Modelo servido:** a interface exibia a previsão do Random Forest, e não a regra declarada (A001, outubro: 61,9 contra 50,3).
5. **API e interface:** mês 13 gerava erro 500; a busca por "[" quebrava; combinação sem indicador usava a média da UF e exibia 0 °C, 0 mm e 0%; `/api/stats` anunciava anos fixos no código; o fluxo SSE tinha pausas cosméticas; o rótulo dizia "viabilidade %"; dois links da bibliografia abriam obras erradas; frases projetavam médias passadas como risco futuro.

---

## 17. Como executar e publicar

### Local

```bash
pip install -r requirements.txt
python api/preprocess.py     # só se for reprocessar (exige os CSVs em BASE/)
python api/train_model.py    # opcional: retreina o RF e grava metrics.json
python api/avaliacao.py      # opcional: refaz os experimentos do TCC
python api/main.py           # http://localhost:8000
```

No Windows, `iniciar.bat` faz o mesmo e expõe o sistema pelo ngrok.

Testes rápidos:

```bash
curl http://localhost:8000/api/stats
curl "http://localhost:8000/api/map?crop=soja&month=10"
curl -X POST http://localhost:8000/api/recommend -H "Content-Type: application/json" \
     -d '{"station_code":"A001","crop":"soja","months":[10,11,12]}'
```

### Publicação (Oracle Cloud)

- **Máquina:** VM **VM.Standard.E2.1.Micro** (Always Free), Ubuntu 22.04, com 2 GB de swap criados pelo `setup.sh`.
- **Endereço:** IP público `136.248.114.242`, domínio **https://agroclima.rumo.tec.br** (registro A no DNS e certificado Let's Encrypt via certbot, com renovação automática).
- **Instalação:** `deploy/setup.sh` clona o repositório, cria o ambiente Python, instala o serviço `agroclima` (systemd), configura o nginx e libera as portas 80 e 443 no firewall interno, na posição 1 da cadeia `INPUT`, antes da regra REJECT da imagem Oracle. No painel da Oracle, a *security list* precisa de regras de entrada TCP 80 e 443.
- **Atualizar a versão publicada** (o servidor roda o que estiver no último `git push`):

```bash
# no PC
git add -A && git commit -m "..." && git push
# na VM
cd ~/agroclima && git pull && sudo systemctl restart agroclima
```

- **Diagnóstico:** `sudo journalctl -u agroclima -n 50` e `sudo iptables -L INPUT -n --line-numbers`.

---

## 18. Glossário

| Termo | Significado |
|---|---|
| **INMET** | Instituto Nacional de Meteorologia, fonte dos dados horários das estações automáticas |
| **Código WMO** | Identificador da estação (ex.: `A001`, Brasília) |
| **Indicador mensal histórico** | Média e desvio entre anos de uma variável, por estação e mês do ano; não é normal climatológica (que exige 30 anos) |
| **Faixa ótima / tolerada** | Intervalo ideal (*b* a *c*) e intervalo suportável (*a* a *d*) de uma variável para a cultura |
| **Nota parcial *s*** | Nota de 0 a 1 de uma variável pela função trapezoidal |
| **Pontuação *S*** | Adequação climática do mês, de 0 a 100 pontos |
| **Mês inviável (veto)** | Mês com alguma variável fora dos limites absolutos (nota 0) |
| **Fator limitante** | Variável com a menor nota parcial no mês |
| **Random Forest / modelo substituto** | Conjunto de árvores treinado para imitar a regra; avaliado, mas não usado na API |
| **GroupKFold** | Validação cruzada que mantém todas as linhas de um grupo (estação ou UF) do mesmo lado da divisão |
| **ZARC** | Zoneamento Agrícola de Risco Climático (MAPA/Embrapa): janelas de semeadura por município, solo e ciclo, com frequência de risco |
| **EcoCrop** | Modelo da FAO que avalia limites cardinais com combinação não compensatória e janelas do ciclo |
| **SSE** | Server-Sent Events, fluxo de eventos do servidor para o navegador |
