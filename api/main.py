"""
AgroClima - API de Recomendação de Plantio
FastAPI backend para auxílio a decisões agrícolas

Versão 1.1: a pontuação servida é a regra trapezoidal aplicada diretamente
(exata e decomponível). O Random Forest de api/train_model.py é mantido apenas
como objeto de avaliação (ver api/models/metrics.json e api/avaliacao.py).
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from train_model import CROP_REQUIREMENTS, STD_DEFAULTS, score_breakdown

BASE_ROOT = Path(__file__).parent.parent
DATA_DIR = BASE_ROOT / "data"
TEMPLATES_DIR = BASE_ROOT / "templates"
STATIC_DIR = BASE_ROOT / "static"

# Estação vizinha quando falta o indicador local: até 150 km, com pelo menos 3 anos
# naquele mês e até 500 m de diferença de altitude; as temperaturas da vizinha são
# corrigidas pelo gradiente térmico vertical médio (6,5 °C por km).
VIZINHA_MAX_KM = 150.0
VIZINHA_MIN_ANOS = 3
VIZINHA_MAX_DALT_M = 500.0
GRADIENTE_TERMICO = 0.0065  # °C por metro

# ─── Estado global (carregado uma vez) ────────────────────────────────────────
_normals: pd.DataFrame | None = None
_stations: pd.DataFrame | None = None
_monthly: pd.DataFrame | None = None


def load_artifacts():
    global _normals, _stations, _monthly

    normals_path = DATA_DIR / "climate_normals.pkl"
    if not normals_path.exists():
        raise RuntimeError("Dados não encontrados. Execute: python api/preprocess.py")

    # Desvios indefinidos (células de um único ano) recebem os mesmos padrões do treino
    _normals = pd.read_pickle(normals_path).fillna(STD_DEFAULTS)
    _stations = pd.read_pickle(DATA_DIR / "stations.pkl")
    _normals = _normals.merge(_stations[["codigo", "alt"]], on="codigo", how="left")
    _monthly = pd.read_pickle(DATA_DIR / "monthly_climate.pkl")
    print("Dados carregados com sucesso.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="AgroClima API", version="1.1.0", lifespan=lifespan)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Modelos Pydantic ──────────────────────────────────────────────────────────

Mes = Annotated[int, Field(ge=1, le=12)]


class RecommendRequest(BaseModel):
    station_code: str
    crop: str
    months: list[Mes] = Field(min_length=1)   # meses de interesse [1..12]


class MonthResult(BaseModel):
    month: int
    month_name: str
    score: Optional[float]
    temp_media: Optional[float]
    precip_total: Optional[float]
    umidade_media: Optional[float]
    status: str
    details: dict
    fonte: dict


class VarAnalysis(BaseModel):
    variavel: str
    valor: float
    unidade: str
    faixa_otima: str
    faixa_tolerada: str
    status: str          # "otimo" | "atencao" | "critico"
    icone: str
    titulo: str
    explicacao: str
    mes_referencia: str
    faixa_observada: str


class RecommendResponse(BaseModel):
    station_code: str
    station_name: str
    uf: str
    crop: str
    crop_name: str
    overall_score: float
    verdict: str
    verdict_color: str
    months: list[MonthResult]
    best_months: list[str]
    alert: Optional[str]
    alerts: list[str]
    climate_summary: dict
    crop_profile: dict
    var_analyses: list[VarAnalysis]
    diagnostico: str
    metodo: str
    periodo_dados: str
    anos_usados: list[int]
    n_anos_por_mes: dict[int, int]


# ─── Helpers ──────────────────────────────────────────────────────────────────

MONTH_NAMES = [
    "", "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
]


def fmt(value: float, nd: int = 1) -> str:
    """Formata número no padrão brasileiro (88.12 -> '88,1')."""
    return f"{value:.{nd}f}".replace(".", ",")


def score_to_verdict(score: float) -> tuple[str, str]:
    if score >= 80:
        return "Altamente Recomendado", "#22c55e"
    if score >= 65:
        return "Recomendado", "#84cc16"
    if score >= 50:
        return "Moderado", "#eab308"
    if score >= 35:
        return "Risco Elevado", "#f97316"
    return "Não Recomendado", "#ef4444"


def score_to_status(score: float) -> str:
    if score >= 80:
        return "excelente"
    if score >= 65:
        return "bom"
    if score >= 50:
        return "moderado"
    if score >= 35:
        return "ruim"
    return "critico"


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em km entre pontos (aceita arrays em lat2/lon2)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def station_period(station_code: str) -> str:
    """Anos de dados efetivamente usados para a estação (ex.: '2005-2025, 14 anos')."""
    if _monthly is None:
        return ""
    anos = sorted(_monthly.loc[_monthly["codigo"] == station_code, "year"].unique())
    if not anos:
        return "sem dados locais"
    return f"{anos[0]}-{anos[-1]}, {len(anos)} anos com ao menos um mês completo"


# ─── Perfis detalhados das culturas (para explicações) ────────────────────────
CROP_PROFILES = {
    "soja": {
        "usos": "Óleo de cozinha, farelo proteico para ração animal, biodiesel e alimentos processados",
        "regioes": "Centro-Oeste (MT, GO, MS), Sul (PR, RS) e MATOPIBA",
        "ciclo": "90 a 150 dias dependendo da cultivar",
        "fase_critica": "florescimento e enchimento de grãos",
        "sensibilidades": ["déficit hídrico no florescimento", "temperaturas abaixo de 20°C", "geadas tardias"],
        "perfil": (
            "A soja é a principal cultura do agronegócio brasileiro e uma das mais exigentes em "
            "termos climáticos. Ela precisa de temperaturas entre 20 e 30°C para germinar e "
            "se desenvolver adequadamente. A fase mais crítica é o florescimento e o enchimento "
            "de grãos: um déficit hídrico nesse período pode reduzir a produtividade em até 80%. "
            "Geadas são letais em qualquer fase do ciclo."
        ),
    },
    "milho": {
        "usos": "Alimentação humana, ração animal, etanol, amido e produtos industriais",
        "regioes": "Mato Grosso, Paraná, Goiás, Minas Gerais e Rio Grande do Sul",
        "ciclo": "90 a 150 dias (precoce a tardio)",
        "fase_critica": "pendoamento e espigamento",
        "sensibilidades": ["calor acima de 35°C no florescimento", "seca no período crítico", "veranico"],
        "perfil": (
            "O milho é o cereal mais versátil cultivado no Brasil, presente da Amazônia ao Pampa. "
            "Sua temperatura ideal de desenvolvimento fica entre 24 e 30°C. O período mais delicado "
            "é o pendoamento e espigamento: temperaturas noturnas acima de 30°C durante a polinização "
            "causam esterilidade do pólen e queda drástica na produção de grãos. O milho consome "
            "cerca de 600mm de água por ciclo, sendo fundamental que a chuva coincida com essa fase."
        ),
    },
    "cafe": {
        "usos": "Bebida, produtos alimentícios, cosméticos e biocombustível",
        "regioes": "Sul de Minas, Cerrado Mineiro, Mogiana (SP), Conilon no Espírito Santo",
        "ciclo": "Perene — produção a partir do 3º ano, vida útil de 20–30 anos",
        "fase_critica": "floração e granação dos frutos",
        "sensibilidades": ["geadas (letais)", "temperatura acima de 23°C", "chuva excessiva na colheita"],
        "perfil": (
            "O café arábica é uma das culturas mais exigentes climaticamente no Brasil. Ele prospera "
            "em temperaturas médias anuais entre 19 e 22°C — acima de 23°C, o florescimento se torna "
            "irregular e os grãos amadurecem precocemente, comprometendo a qualidade da bebida. "
            "Uma única geada pode destruir anos de investimento, pois a planta é perene. A precipitação "
            "anual ideal fica entre 1.200 e 1.800mm, bem distribuída, com período seco para favorecer "
            "a maturação dos frutos antes da colheita."
        ),
    },
    "arroz": {
        "usos": "Alimentação humana direta, farinha, amido e ração",
        "regioes": "Rio Grande do Sul (irrigado), Mato Grosso e Pará (terras altas)",
        "ciclo": "90 a 130 dias",
        "fase_critica": "florescimento (polinização)",
        "sensibilidades": ["temperatura abaixo de 20°C na germinação", "calor acima de 35°C na floração", "falta d'água"],
        "perfil": (
            "O arroz é a gramínea mais aquática entre as cultivadas no Brasil. Precisa de "
            "temperaturas mínimas de 20°C para germinar e floresce melhor entre 25 e 33°C. "
            "Temperaturas acima de 35°C durante a polinização causam esterilidade das espiguetas, "
            "reduzindo drasticamente o número de grãos por panícula. O arroz irrigado do Sul "
            "é altamente produtivo justamente pelo controle da temperatura via lâmina d'água."
        ),
    },
    "feijao": {
        "usos": "Alimentação humana (feijão-com-arroz), proteína vegetal, ração",
        "regioes": "Paraná, Minas Gerais, Goiás, Bahia (três safras por ano em diferentes regiões)",
        "ciclo": "65 a 100 dias",
        "fase_critica": "florescimento e formação de vagens",
        "sensibilidades": ["encharcamento do solo", "granizo", "alta umidade favorece doenças fúngicas"],
        "perfil": (
            "O feijão é uma das culturas mais adaptáveis do Brasil, plantado em três safras "
            "por ano em regiões distintas. Sua faixa de temperatura ideal é relativamente estreita "
            "(18–26°C): calor excessivo causa aborto de flores e vagens. O excesso de umidade "
            "favorece doenças como a antracnose e o mofo-branco, que podem destruir lavouras inteiras "
            "em poucos dias. É sensível a encharcamento, especialmente na fase de florescimento."
        ),
    },
    "trigo": {
        "usos": "Farinha de trigo, massas, pães, biscoitos e ração animal",
        "regioes": "Rio Grande do Sul, Paraná e Santa Catarina",
        "ciclo": "90 a 150 dias (cultura de inverno)",
        "fase_critica": "espigamento e floração",
        "sensibilidades": ["chuva excessiva na maturação (brotamento)", "geadas no espigamento", "calor acima de 30°C"],
        "perfil": (
            "O trigo é o único cereal de inverno com expressão comercial no Brasil, cultivado "
            "principalmente no Sul do país. Precisa de temperaturas frias (10–20°C) para o "
            "afilhamento e desenvolver bem. Geadas moderadas são toleradas na fase vegetativa, "
            "mas são letais no espigamento. O principal risco no Sul é a chuva excessiva na "
            "maturação, que provoca brotamento prematuro dos grãos na espiga — fenômeno que "
            "desvaloriza completamente a produção."
        ),
    },
    "algodao": {
        "usos": "Fibra têxtil, óleo vegetal, ração (caroço) e cosméticos",
        "regioes": "Mato Grosso, Bahia (Oeste) e Goiás — Cerrado brasileiro",
        "ciclo": "150 a 200 dias",
        "fase_critica": "florescimento e formação de capulhos",
        "sensibilidades": ["chuva na abertura dos capulhos", "alta umidade favorece doenças", "temperaturas abaixo de 18°C"],
        "perfil": (
            "O algodão brasileiro é produzido principalmente no Cerrado, onde o regime de chuvas "
            "bem definido (chuvas no verão, seca no inverno) favorece a abertura natural dos capulhos. "
            "A umidade excessiva durante a colheita deprecia a qualidade da fibra e facilita doenças "
            "como a ramularia e o mofo do capulho. Temperatura abaixo de 18°C retarda o desenvolvimento "
            "e afeta a fecundação das flores. O Brasil é hoje um dos principais exportadores mundiais."
        ),
    },
    "cana": {
        "usos": "Açúcar, etanol, cogeração de energia elétrica e cachaça",
        "regioes": "São Paulo (Centro-Sul), Nordeste (Zona da Mata) e expansão no Cerrado",
        "ciclo": "12 a 18 meses (cana-planta); 12 meses (socas subsequentes)",
        "fase_critica": "maturação (estresse hídrico favorece acúmulo de sacarose)",
        "sensibilidades": ["geadas (destroem a parte aérea)", "seca prolongada no crescimento", "excesso de chuva na colheita"],
        "perfil": (
            "A cana-de-açúcar é a maior cultura em área colhida no Brasil e a base do setor "
            "sucroenergético. Para o máximo acúmulo de sacarose (a maturação), ela precisa de "
            "um período de estresse hídrico leve associado a temperaturas mais baixas — por isso "
            "a colheita no Sudeste ocorre no inverno seco. Nas fases de brotação e crescimento, "
            "porém, ela exige alta disponibilidade hídrica (150–250mm/mês). Geadas destroem a "
            "parte aérea e comprometem a rebrota."
        ),
    },
    "mandioca": {
        "usos": "Farinha, fécula, tapioca, alimentação animal e bioetanol",
        "regioes": "Todo o Brasil — de Roraima ao Rio Grande do Sul",
        "ciclo": "10 a 24 meses dependendo da finalidade",
        "fase_critica": "primeiros 3 meses (estabelecimento das raízes)",
        "sensibilidades": ["geadas (matam as estacas)", "encharcamento prolongado", "seca extrema nos primeiros meses"],
        "perfil": (
            "A mandioca é a cultura tropical mais resiliente cultivada no Brasil, presente "
            "em todos os estados. Seu metabolismo fotossintético (C3) permite tolerar secas "
            "prolongadas que destruiriam outras culturas. A temperatura ideal fica entre 22 e 28°C "
            "para o desenvolvimento das raízes tuberosas. O período crítico são os primeiros "
            "três meses, quando a estaca precisa enraizar: encharcamento nessa fase causa podridão "
            "das raízes. Geadas matam a parte aérea mas raramente as raízes já formadas."
        ),
    },
    "sorgo": {
        "usos": "Ração animal, bioetanol, silagem e alimentação humana no Nordeste",
        "regioes": "Mato Grosso, Goiás, Minas Gerais e Nordeste semiárido",
        "ciclo": "90 a 120 dias",
        "fase_critica": "florescimento (paniculamento)",
        "sensibilidades": ["encharcamento (menos tolerante que o milho)", "frio abaixo de 12°C retarda germinação"],
        "perfil": (
            "O sorgo é o cereal mais eficiente no uso de água cultivado no Brasil, adaptado "
            "a regiões semiáridas onde o milho fracassaria. Seu metabolismo C4 e raízes profundas "
            "permitem suportar déficits hídricos que eliminariam outras gramíneas. A temperatura "
            "ótima de desenvolvimento fica entre 20 e 32°C. Diferente do milho, o sorgo consegue "
            "entrar em dormência durante a seca e retomar o crescimento quando a chuva volta — "
            "característica que o torna estratégico no Nordeste e no cultivo de safrinha."
        ),
    },
}


# ─── Análise textual por variável ─────────────────────────────────────────────
# Zonas, coerentes com a nota trapezoidal s:
#   crítica  : x <= a ou x >= d  (s = 0)
#   atenção  : a < x < b ou c < x < d  (0 < s < 1)
#   ótima    : b <= x <= c  (s = 1)

TEXTOS = {
    "Temperatura": {
        "abaixo_limite": ("critico", "🥶", "Temperatura crítica — abaixo do limite",
            "No histórico, a temperatura média de {v} °C fica {dif} °C abaixo do mínimo absoluto "
            "adotado para {cultura} ({a} °C). Nesse patamar, a germinação e o crescimento ficam "
            "muito prejudicados, e há risco de geada."),
        "abaixo_otimo": ("atencao", "🌡️", "Temperatura abaixo do ideal",
            "No histórico, a temperatura média de {v} °C está na faixa tolerada, mas {dif} °C abaixo "
            "da faixa ótima ({otima}). Médias assim tendem a alongar o ciclo e reduzir o potencial "
            "produtivo; na fase de {fase}, o frio pode causar aborto de flores."),
        "otimo": ("otimo", "✅", "Temperatura ideal",
            "No histórico, a temperatura média de {v} °C está dentro da faixa ótima ({otima}) para "
            "{cultura}, patamar que favorece crescimento e florescimento regulares."),
        "acima_otimo": ("atencao", "🌡️", "Temperatura acima do ideal",
            "No histórico, a temperatura média de {v} °C está {dif} °C acima da faixa ótima. A cultura "
            "tolera esse patamar, mas sob estresse térmico, com mais transpiração e consumo de água; "
            "na fase de {fase}, o calor pode causar aborto de flores."),
        "acima_limite": ("critico", "🔥", "Temperatura crítica — calor excessivo",
            "No histórico, a temperatura média de {v} °C supera o máximo absoluto adotado ({d} °C). "
            "Nesse patamar há risco de danos fisiológicos graves, como esterilidade do pólen na fase "
            "de {fase}."),
    },
    "Precipitação": {
        "abaixo_limite": ("critico", "🏜️", "Precipitação insuficiente — risco de seca",
            "No histórico, a precipitação média de {v} mm fica {dif} mm abaixo do mínimo adotado para "
            "{cultura} ({a} mm). Com esse déficit, o cultivo de sequeiro dependeria de irrigação "
            "suplementar."),
        "abaixo_otimo": ("atencao", "💧", "Precipitação abaixo do ideal",
            "No histórico, a precipitação média de {v} mm está na faixa tolerada, mas {dif} mm abaixo "
            "da faixa ótima ({otima}). Veranicos nesse período seriam críticos, sobretudo na fase de "
            "{fase}."),
        "otimo": ("otimo", "🌧️", "Precipitação ideal",
            "No histórico, a precipitação média de {v} mm está dentro da faixa ótima ({otima}) para "
            "{cultura}, com boa disponibilidade de água no período."),
        "acima_otimo": ("atencao", "🌦️", "Precipitação elevada",
            "No histórico, a precipitação média de {v} mm está {dif} mm acima da faixa ótima, ainda "
            "dentro do tolerável. O excesso de umidade favorece doenças fúngicas e pode dificultar "
            "plantio e colheita."),
        "acima_limite": ("critico", "🌊", "Excesso hídrico — risco de encharcamento",
            "No histórico, a precipitação média de {v} mm ultrapassa o máximo adotado ({d} mm) para "
            "{cultura}. Esse volume favorece encharcamento, podridões e dificulta o uso de máquinas."),
    },
    "Umidade": {
        "abaixo_limite": ("critico", "🌵", "Umidade crítica — ar muito seco",
            "No histórico, a umidade relativa média de {v}% fica abaixo do mínimo adotado ({a}%) para "
            "{cultura}. Ar muito seco aumenta a transpiração e reduz a fotossíntese, sobretudo na fase "
            "de {fase}."),
        "abaixo_otimo": ("atencao", "💨", "Umidade abaixo do ideal",
            "No histórico, a umidade relativa média de {v}% está {dif} pontos percentuais abaixo da "
            "faixa ótima ({otima}), o que reduz a eficiência fotossintética."),
        "otimo": ("otimo", "✅", "Umidade ideal",
            "No histórico, a umidade relativa média de {v}% está dentro da faixa ótima ({otima}) para "
            "{cultura}."),
        "acima_otimo": ("atencao", "🌫️", "Umidade elevada",
            "No histórico, a umidade relativa média de {v}% está {dif} pontos percentuais acima da "
            "faixa ótima, condição que favorece fungos e pragas."),
        "acima_limite": ("critico", "🌫️", "Umidade excessiva",
            "No histórico, a umidade relativa média de {v}% supera o máximo adotado ({d}%) para "
            "{cultura}, condição muito favorável a doenças fúngicas."),
    },
}

SEVERIDADE = {"critico": 2, "atencao": 1, "otimo": 0}


def _zona(x: float, a: float, b: float, c: float, d: float) -> tuple[str, float]:
    """Retorna a zona textual e a distância até a faixa ótima."""
    if x <= a:
        return "abaixo_limite", a - x
    if x < b:
        return "abaixo_otimo", b - x
    if x <= c:
        return "otimo", 0.0
    if x < d:
        return "acima_otimo", x - c
    return "acima_limite", x - d


def _explain_var(varname: str, unit: str, req: tuple, valores: dict[int, float],
                 crop_nome: str, fase_critica: str) -> VarAnalysis | None:
    """
    Analisa uma variável mês a mês e reporta o pior mês selecionado,
    em vez da média dos meses (que esconderia riscos opostos).
    """
    a, b, c, d = req
    if not valores:
        return None

    avaliados = []
    for m, x in valores.items():
        zona, dif = _zona(x, a, b, c, d)
        status = TEXTOS[varname][zona][0]
        avaliados.append((SEVERIDADE[status], dif, m, x, zona))
    _, dif, mes, valor, zona = max(avaliados)

    status, icone, titulo, texto = TEXTOS[varname][zona]
    sep = "" if unit == "%" else " "
    explicacao = texto.format(
        v=fmt(valor), dif=fmt(dif), a=a, d=d, cultura=crop_nome, fase=fase_critica,
        otima=f"{b}–{c}{sep}{unit}",
    )
    if len(valores) > 1:
        explicacao = f"Pior mês selecionado: {MONTH_NAMES[mes]}. " + explicacao

    lo, hi = min(valores.values()), max(valores.values())
    return VarAnalysis(
        variavel=varname, valor=round(valor, 1), unidade=unit,
        faixa_otima=f"{b}–{c}{sep}{unit}", faixa_tolerada=f"{a}–{d}{sep}{unit}",
        status=status, icone=icone, titulo=titulo, explicacao=explicacao,
        mes_referencia=MONTH_NAMES[mes],
        faixa_observada=f"{fmt(lo)}–{fmt(hi)}{sep}{unit}" if len(valores) > 1 else f"{fmt(lo)}{sep}{unit}",
    )


def _diagnostico(crop_nome: str, crop_key: str, overall: float, station_name: str, uf: str,
                 months_sel: list[int], analyses: list[VarAnalysis], best_months: list[str],
                 inviaveis: list[str], periodo: str) -> str:
    """Gera parágrafo de diagnóstico climático consolidado, em linguagem histórica."""
    meses_str = ", ".join(MONTH_NAMES[m] for m in months_sel)
    regioes = CROP_PROFILES.get(crop_key, {}).get("regioes", "")

    if overall >= 80:
        nivel = "**altamente favorável**"
    elif overall >= 65:
        nivel = "**favorável**"
    elif overall >= 50:
        nivel = "**moderada**"
    elif overall >= 35:
        nivel = "**de risco elevado**"
    else:
        nivel = "**não recomendada**"

    partes = [
        f"No histórico de {station_name}/{uf} ({periodo}), a adequação climática média de "
        f"{meses_str} para {crop_nome} foi {nivel}, com {fmt(overall)} pontos de 0 a 100."
    ]

    criticos = [a for a in analyses if a.status == "critico"]
    atencoes = [a for a in analyses if a.status == "atencao"]
    if criticos:
        itens = "; ".join(f"{a.variavel.lower()} em {a.mes_referencia} ({fmt(a.valor)} {a.unidade}, "
                          f"faixa tolerada {a.faixa_tolerada})" for a in criticos)
        partes.append(f"Fatores fora dos limites absolutos: {itens}.")
    elif atencoes:
        nomes = " e ".join(a.variavel.lower() for a in atencoes)
        verbo = "ficaram" if len(atencoes) > 1 else "ficou"
        partes.append(f"{nomes.capitalize()} {verbo} fora da faixa ótima, mas dentro do tolerável.")

    if inviaveis:
        partes.append(
            f"Os meses {', '.join(inviaveis)} têm ao menos uma variável fora dos limites absolutos "
            f"e por isso foram marcados como inviáveis e contam como 0 na pontuação geral, mesmo que a "
            f"soma ponderada dê nota positiva."
        )

    if regioes:
        partes.append(f"Como referência, as principais regiões produtoras de {crop_nome} no Brasil são: {regioes}.")

    if best_months:
        partes.append(
            f"No ano inteiro, os meses de clima mensal mais adequado nessa estação foram: "
            f"{', '.join(best_months[:3])}. Isso não equivale a data de semeadura, que depende "
            f"do ciclo da cultura e do ZARC."
        )

    return " ".join(partes)


# ─── Dados e pontuação ────────────────────────────────────────────────────────

def lookup_normal(station_row: pd.Series, month: int) -> tuple[pd.Series | None, dict]:
    """
    Indicador do mês para a estação. Sem indicador local, escolhe, para este mês,
    a estação mais próxima (Haversine) com pelo menos VIZINHA_MIN_ANOS anos e diferença
    de altitude até VIZINHA_MAX_DALT_M, dentro de VIZINHA_MAX_KM; empate de distância
    fica com quem tem mais anos. As temperaturas são corrigidas pela altitude.
    """
    code = station_row["codigo"]
    rows = _normals[(_normals["codigo"] == code) & (_normals["month"] == month)]
    if len(rows) > 0:
        return rows.iloc[0], {"tipo": "estacao", "codigo": code}

    alt_alvo = float(station_row.get("alt", 0) or 0)
    cand = _normals[(_normals["month"] == month) & (_normals["codigo"] != code)
                    & (_normals["n_anos"] >= VIZINHA_MIN_ANOS)
                    & ((_normals["alt"] - alt_alvo).abs() <= VIZINHA_MAX_DALT_M)].copy()
    if len(cand) > 0:
        cand["dist"] = haversine_km(float(station_row["lat"]), float(station_row["lon"]),
                                    cand["lat"].astype(float).values, cand["lon"].astype(float).values)
        cand = cand[cand["dist"] <= VIZINHA_MAX_KM].sort_values(["dist", "n_anos", "codigo"],
                                                                ascending=[True, False, True])
        if len(cand) > 0:
            r = cand.iloc[0].copy()
            dalt = alt_alvo - float(r["alt"])
            for c in ("temp_norm", "temp_max_norm", "temp_min_norm"):
                r[c] = float(r[c]) - GRADIENTE_TERMICO * dalt
            return r, {"tipo": "vizinha", "codigo": r["codigo"], "nome": r["nome"],
                       "distancia_km": round(float(r["dist"]), 1),
                       "dif_altitude_m": round(dalt), "n_anos": int(r["n_anos"])}

    return None, {"tipo": "indisponivel"}


def month_result(station_row: pd.Series, month: int, crop: str) -> MonthResult:
    """Pontua um mês pela regra e devolve a decomposição completa."""
    r, fonte = lookup_normal(station_row, month)
    if r is None:
        return MonthResult(month=month, month_name=MONTH_NAMES[month], score=None,
                           temp_media=None, precip_total=None, umidade_media=None,
                           status="sem_dados", details={}, fonte=fonte)

    b = score_breakdown(r, crop)
    formula = (
        f"100 × (0,35×{fmt(b['s_temp'], 2)} + 0,40×{fmt(b['s_precip'], 2)} + 0,25×{fmt(b['s_umid'], 2)})"
        f" × {fmt(b['bonus'])} × {fmt(b['pen_precip'], 2)} × {fmt(b['pen_temp'], 2)} = {fmt(b['score'])}"
    )
    return MonthResult(
        month=month,
        month_name=MONTH_NAMES[month],
        score=b["score"],
        temp_media=round(float(r["temp_norm"]), 1),
        precip_total=round(float(r["precip_norm"]), 1),
        umidade_media=round(float(r["umidade_norm"]), 1),
        status="inviavel" if b["veto"] else score_to_status(b["score"]),
        details={
            "temp_max": round(float(r["temp_max_norm"]), 1),
            "temp_min": round(float(r["temp_min_norm"]), 1),
            "precip_std": round(float(r["precip_std"]), 1),
            "temp_std": round(float(r["temp_std"]), 2),
            "n_anos": int(r["n_anos"]),
            "s_temp": round(b["s_temp"], 3),
            "s_precip": round(b["s_precip"], 3),
            "s_umid": round(b["s_umid"], 3),
            "base": round(b["base"], 3),
            "bonus": b["bonus"],
            "pen_precip": round(b["pen_precip"], 3),
            "pen_temp": round(b["pen_temp"], 3),
            "veto": b["veto"],
            "formula": formula,
        },
        fonte=fonte,
    )


def get_annual_series(station_code: str) -> dict:
    """Retorna série mensal anual (indicadores) para a estação."""
    if _normals is None:
        return {}
    rows = _normals[_normals["codigo"] == station_code].sort_values("month")
    if len(rows) == 0:
        return {}
    return {
        "months": rows["month"].tolist(),
        "temp": [round(v, 1) for v in rows["temp_norm"].tolist()],
        "precip": [round(v, 1) for v in rows["precip_norm"].tolist()],
        "umidade": [round(v, 1) for v in rows["umidade_norm"].tolist()],
    }


def build_alerts(months_data: list[MonthResult], crop_info: dict) -> list[str]:
    """Alertas de todos os meses com variável fora dos limites absolutos."""
    alerts = []
    nome = crop_info["nome"]
    for md in months_data:
        if md.score is None:
            alerts.append(f"{md.month_name}: sem indicador da estação nem de vizinha a até "
                          f"{VIZINHA_MAX_KM:.0f} km — mês não pontuado.")
            continue
        t_min, _, _, t_max = crop_info["temp"]
        p_min, _, _, p_max = crop_info["precip"]
        u_min, _, _, u_max = crop_info["umidade"]
        if md.temp_media >= t_max:
            alerts.append(f"{md.month_name}: temperatura média de {fmt(md.temp_media)} °C, acima do máximo de {t_max} °C para {nome}.")
        if md.temp_media <= t_min:
            alerts.append(f"{md.month_name}: temperatura média de {fmt(md.temp_media)} °C, abaixo do mínimo de {t_min} °C para {nome}.")
        if md.precip_total <= p_min:
            alerts.append(f"{md.month_name}: precipitação de {fmt(md.precip_total)} mm, abaixo do mínimo de {p_min} mm — risco de déficit hídrico.")
        if md.precip_total >= p_max:
            alerts.append(f"{md.month_name}: precipitação de {fmt(md.precip_total)} mm, acima do máximo de {p_max} mm — risco de excesso hídrico.")
        if md.umidade_media <= u_min or md.umidade_media >= u_max:
            alerts.append(f"{md.month_name}: umidade de {fmt(md.umidade_media)}%, fora da faixa tolerada de {u_min}–{u_max}%.")
        if md.fonte.get("tipo") == "vizinha":
            f = md.fonte
            alerts.append(f"{md.month_name}: sem indicador local; usados os dados da estação vizinha "
                          f"{f['nome']} ({f['codigo']}), a {fmt(f['distancia_km'])} km, com {f['n_anos']} anos "
                          f"e temperaturas corrigidas por {f['dif_altitude_m']:+d} m de altitude.")
    return alerts


def build_recommendation(req: RecommendRequest) -> RecommendResponse:
    """Gera recomendação de plantio para uma estação, cultura e período."""
    if _normals is None or _stations is None:
        raise HTTPException(500, "Dados não carregados.")

    station = _stations[_stations["codigo"] == req.station_code]
    if len(station) == 0:
        raise HTTPException(404, f"Estação '{req.station_code}' não encontrada.")
    if req.crop not in CROP_REQUIREMENTS:
        raise HTTPException(400, f"Cultura '{req.crop}' não suportada.")

    st = station.iloc[0]
    crop_info = CROP_REQUIREMENTS[req.crop]
    meses = sorted(set(req.months))

    months_data = [month_result(st, m, req.crop) for m in meses]
    # A trava de inviabilidade também vale para o resumo: mês inviável entra com 0 na média
    scores = [0.0 if md.status == "inviavel" else md.score for md in months_data if md.score is not None]
    if not scores:
        raise HTTPException(422, "Não há dados para nenhum dos meses selecionados nesta estação.")

    overall = round(float(np.mean(scores)), 1)
    verdict, color = score_to_verdict(overall)

    # Melhores meses do ano: maiores pontuações, sem veto, com pelo menos 60 pontos
    ano = [month_result(st, m, req.crop) for m in range(1, 13)]
    candidatos = [r for r in ano if r.score is not None and not r.details["veto"] and r.score >= 60]
    best_months = [r.month_name for r in sorted(candidatos, key=lambda r: r.score, reverse=True)[:4]]

    alerts = build_alerts(months_data, crop_info)
    inviaveis = [md.month_name for md in months_data if md.status == "inviavel"]

    profile = CROP_PROFILES.get(req.crop, {})
    fase = profile.get("fase_critica", "florescimento")
    validos = [md for md in months_data if md.score is not None]
    var_analyses = [
        _explain_var(nome, unid, crop_info[chave], {md.month: getattr(md, campo) for md in validos},
                     crop_info["nome"], fase)
        for nome, unid, chave, campo in [
            ("Temperatura", "°C", "temp", "temp_media"),
            ("Precipitação", "mm", "precip", "precip_total"),
            ("Umidade", "%", "umidade", "umidade_media"),
        ]
    ]

    periodo = station_period(req.station_code)
    diagnostico = _diagnostico(
        crop_info["nome"], req.crop, overall, str(st.get("nome", "")), str(st.get("uf", "")),
        meses, var_analyses, best_months, inviaveis, periodo,
    )

    return RecommendResponse(
        station_code=req.station_code,
        station_name=str(st.get("nome", "")),
        uf=str(st.get("uf", "")),
        crop=req.crop,
        crop_name=crop_info["nome"],
        overall_score=overall,
        verdict=verdict,
        verdict_color=color,
        months=months_data,
        best_months=best_months,
        alert=alerts[0] if alerts else None,
        alerts=alerts,
        climate_summary=get_annual_series(req.station_code),
        crop_profile=profile,
        var_analyses=var_analyses,
        diagnostico=diagnostico,
        metodo="Regra trapezoidal aplicada diretamente (AgroClima 1.1)",
        periodo_dados=periodo,
        anos_usados=sorted(int(a) for a in _monthly.loc[_monthly["codigo"] == req.station_code, "year"].unique()),
        n_anos_por_mes={int(r.month): int(r.n_anos)
                        for r in _normals[_normals["codigo"] == req.station_code].itertuples()},
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>AgroClima</h1><p>Template não encontrado.</p>")


@app.get("/api/stations")
async def get_stations(uf: Optional[str] = None, search: Optional[str] = None):
    """Lista todas as estações disponíveis."""
    if _stations is None:
        raise HTTPException(500, "Dados não carregados.")

    df = _stations.copy()
    if uf:
        df = df[df["uf"].str.upper() == uf.upper()]
    if search:
        # Busca literal: caracteres como "[" não são interpretados como expressão regular
        df = df[df["nome"].str.contains(search, case=False, na=False, regex=False)]

    df = df.sort_values(["uf", "nome"])
    return df.fillna("").to_dict(orient="records")


@app.get("/api/crops")
async def get_crops():
    """Lista as culturas disponíveis."""
    return [
        {"key": k, "nome": v["nome"], "emoji": v["emoji"], "desc": v["desc"],
         "temp": v["temp"], "precip": v["precip"], "umidade": v["umidade"],
         "meses_favoraveis": v["meses_favoraveis"]}
        for k, v in CROP_REQUIREMENTS.items()
    ]


def fator_limitante(b: dict) -> str:
    """Variável com a menor nota parcial; 'nenhum' quando todas estão na faixa ótima."""
    notas = {"temperatura": b["s_temp"], "chuva": b["s_precip"], "umidade": b["s_umid"]}
    nome, nota = min(notas.items(), key=lambda kv: kv[1])
    return "nenhum" if nota >= 1.0 else nome


@app.get("/api/map")
async def get_map(crop: str, month: Annotated[int, Query(ge=1, le=12)]):
    """Pontuação de todas as estações com indicador local para uma cultura e um mês (mapa)."""
    if crop not in CROP_REQUIREMENTS:
        raise HTTPException(400, f"Cultura '{crop}' não suportada.")
    rows = _normals[_normals["month"] == month]
    out = []
    for r in rows.itertuples(index=False):
        b = score_breakdown(r._asdict(), crop)
        out.append({
            "codigo": r.codigo, "nome": r.nome, "uf": r.uf,
            "lat": round(float(r.lat), 4), "lon": round(float(r.lon), 4),
            "score": b["score"], "inviavel": b["veto"], "n_anos": int(r.n_anos),
            "s_temp": round(b["s_temp"], 3), "s_precip": round(b["s_precip"], 3), "s_umid": round(b["s_umid"], 3),
            "limitante": fator_limitante(b),
            "temp": round(float(r.temp_norm), 1), "precip": round(float(r.precip_norm), 1),
            "umidade": round(float(r.umidade_norm), 1),
        })
    return out


@app.get("/api/climate/{station_code}")
async def get_climate(station_code: str):
    """Retorna série climática anual da estação."""
    if _stations is None:
        raise HTTPException(500, "Dados não carregados.")

    station = _stations[_stations["codigo"] == station_code]
    if len(station) == 0:
        raise HTTPException(404, f"Estação {station_code} não encontrada.")

    st = station.iloc[0]
    return {
        "codigo": station_code,
        "nome": st.get("nome", ""),
        "uf": st.get("uf", ""),
        "lat": float(st.get("lat", 0)),
        "lon": float(st.get("lon", 0)),
        "periodo": station_period(station_code),
        "series": get_annual_series(station_code),
    }


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """Gera recomendação de plantio para uma estação, cultura e período."""
    return build_recommendation(req)


@app.get("/api/stats")
async def get_stats():
    """Retorna estatísticas do sistema, calculadas a partir da base carregada."""
    n_stations = len(_stations) if _stations is not None else 0
    n_records = len(_monthly) if _monthly is not None else 0
    n_normals = len(_normals) if _normals is not None else 0
    ufs = sorted(_stations["uf"].dropna().unique().tolist()) if _stations is not None else []
    anos = sorted(int(a) for a in _monthly["year"].unique()) if _monthly is not None else []
    return {
        "estacoes": n_stations,
        "registros_mensais": n_records,
        "normais_climaticas": n_normals,
        "culturas": len(CROP_REQUIREMENTS),
        "estados": ufs,
        "anos": anos,
    }


@app.get("/api/recommend/stream")
async def recommend_stream(station_code: str, crop: str, months: str):
    """Endpoint SSE que transmite as etapas reais da análise, sem pausas simuladas."""

    def evt(step: str, msg: str, detail: str = "", data: dict | None = None) -> str:
        obj = {"step": step, "msg": msg, "detail": detail}
        if data:
            obj["data"] = data
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def generate():
        t0 = time.time()
        try:
            # Valida a entrada antes de qualquer cálculo
            try:
                months_list = [int(m.strip()) for m in months.split(",") if m.strip()]
                req = RecommendRequest(station_code=station_code, crop=crop, months=months_list)
            except (ValueError, ValidationError):
                yield evt("error", "Meses inválidos: informe ao menos um mês entre 1 e 12.")
                return

            try:
                result = build_recommendation(req)
            except HTTPException as exc:
                yield evt("error", str(exc.detail))
                return

            st = _stations[_stations["codigo"] == station_code].iloc[0]
            crop_info = CROP_REQUIREMENTS[crop]

            yield evt("init", "Motor de análise climática AgroClima 1.1",
                      "Regra trapezoidal aplicada diretamente sobre indicadores pré-processados")
            yield evt("station", f"Estação encontrada: {st['nome']} ({station_code})",
                      f"{st['uf']} · {float(st.get('alt', 0)):.0f} m · "
                      f"Lat {float(st['lat']):.4f}° · Lon {float(st['lon']):.4f}°")
            yield evt("records", "Indicadores mensais históricos carregados (em memória)",
                      f"Período da estação: {result.periodo_dados}")

            t_req, p_req, u_req = crop_info["temp"], crop_info["precip"], crop_info["umidade"]
            yield evt("crop", f"Requisitos agronômicos: {crop_info['nome']} {crop_info['emoji']}",
                      f"Temp ótima {t_req[1]}-{t_req[2]} °C · Precip ótima {p_req[1]}-{p_req[2]} mm/mês · "
                      f"Umidade ótima {u_req[1]}-{u_req[2]}%")

            for md in result.months:
                if md.score is None:
                    yield evt("rule_month", f"{md.month_name}: sem dados", "Mês não pontuado",
                              {"month": md.month, "score": None})
                    continue
                bar_filled = round(md.score / 10)
                bar = "█" * bar_filled + "░" * (10 - bar_filled)
                marca = "  ⚠ inviável" if md.status == "inviavel" else ""
                yield evt("rule_month", f"{md.month_name}: {fmt(md.score)} pontos [{bar}]{marca}",
                          md.details["formula"], {"month": md.month, "score": md.score})
                await asyncio.sleep(0)

            elapsed = time.time() - t0
            yield evt("complete",
                      f"Análise concluída em {elapsed * 1000:.0f} ms — {fmt(result.overall_score)} pontos (0-100)",
                      f"Classificação: {result.verdict} · média dos meses pontuados (inviável conta 0)",
                      {"overall_score": result.overall_score, "verdict": result.verdict,
                       "color": result.verdict_color})

            yield f"data: {json.dumps({'step': 'result', 'result': result.model_dump()}, ensure_ascii=False)}\n\n"

        except Exception as exc:
            yield evt("error", f"Erro interno: {str(exc)}", "Verifique o console do servidor")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
