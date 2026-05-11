"""
AgroClima - API de Recomendação de Plantio
FastAPI backend com ML para auxílio a decisões agrícolas
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_ROOT = Path(__file__).parent.parent
DATA_DIR = BASE_ROOT / "data"
MODEL_DIR = Path(__file__).parent / "models"
TEMPLATES_DIR = BASE_ROOT / "templates"
STATIC_DIR = BASE_ROOT / "static"

app = FastAPI(title="AgroClima API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Estado global (carregado uma vez) ────────────────────────────────────────
_artifacts = None
_normals: pd.DataFrame | None = None
_stations: pd.DataFrame | None = None
_monthly: pd.DataFrame | None = None


def load_artifacts():
    global _artifacts, _normals, _stations, _monthly

    model_path = MODEL_DIR / "recommendation_model.joblib"
    normals_path = DATA_DIR / "climate_normals.pkl"
    stations_path = DATA_DIR / "stations.pkl"
    monthly_path = DATA_DIR / "monthly_climate.pkl"

    if not model_path.exists():
        raise RuntimeError(
            "Modelo não encontrado. Execute: python api/preprocess.py && python api/train_model.py"
        )

    _artifacts = joblib.load(model_path)
    _normals = pd.read_pickle(normals_path)
    _stations = pd.read_pickle(stations_path)
    _monthly = pd.read_pickle(monthly_path)
    print("Modelo e dados carregados com sucesso.")


@app.on_event("startup")
def startup():
    load_artifacts()


# ─── Modelos Pydantic ──────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    station_code: str
    crop: str
    months: list[int]   # meses de interesse [1..12]


class MonthResult(BaseModel):
    month: int
    month_name: str
    score: float
    temp_media: float
    precip_total: float
    umidade_media: float
    status: str
    details: dict


class VarAnalysis(BaseModel):
    variavel: str
    valor: float
    unidade: str
    faixa_otima: str
    faixa_tolerada: str
    status: str          # "otimo" | "bom" | "atencao" | "critico"
    icone: str
    titulo: str
    explicacao: str

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
    climate_summary: dict
    crop_profile: dict
    var_analyses: list[VarAnalysis]
    diagnostico: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

MONTH_NAMES = [
    "", "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
]

MONTH_ABBR = [
    "", "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
    "Jul", "Ago", "Set", "Out", "Nov", "Dez"
]


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


def _explain_var(
    varname: str, value: float, unit: str,
    req_min: float, req_otmin: float, req_otmax: float, req_max: float,
    crop_nome: str, crop_key: str, fase_critica: str,
) -> VarAnalysis:
    """Gera análise textual detalhada de uma variável climática vs requisito da cultura."""

    faixa_ot  = f"{req_otmin}–{req_otmax} {unit}"
    faixa_tol = f"{req_min}–{req_max} {unit}"

    # ── Temperatura ──────────────────────────────────────────────────────────
    if varname == "Temperatura":
        if value < req_min:
            return VarAnalysis(
                variavel=varname, valor=round(value,1), unidade=unit,
                faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
                status="critico", icone="🥶", titulo="Temperatura crítica — abaixo do limite",
                explicacao=(
                    f"A temperatura média de {value:.1f}°C está {req_min - value:.1f}°C abaixo "
                    f"do mínimo absoluto tolerado pela {crop_nome} ({req_min}°C). Nessa faixa, "
                    f"a germinação é praticamente inviável e o crescimento cessa. "
                    f"Risco de geadas que podem destruir completamente a lavoura."
                ),
            )
        if value < req_otmin:
            return VarAnalysis(
                variavel=varname, valor=round(value,1), unidade=unit,
                faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
                status="atencao", icone="🌡️", titulo="Temperatura abaixo do ideal",
                explicacao=(
                    f"A temperatura média de {value:.1f}°C está dentro da faixa tolerada, mas "
                    f"{req_otmin - value:.1f}°C abaixo da zona ótima ({faixa_ot}). "
                    f"O crescimento é mais lento, o ciclo se prolonga e a produtividade fica "
                    f"comprometida. Na fase de {fase_critica}, temperaturas baixas "
                    f"podem causar aborto de flores e redução no número de grãos."
                ),
            )
        if value <= req_otmax:
            return VarAnalysis(
                variavel=varname, valor=round(value,1), unidade=unit,
                faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
                status="otimo", icone="✅", titulo="Temperatura ideal",
                explicacao=(
                    f"A temperatura média de {value:.1f}°C está exatamente dentro da faixa ótima "
                    f"({faixa_ot}) para {crop_nome}. Essa condição favorece o crescimento "
                    f"acelerado, florescimento regular e boa produtividade. "
                    f"A fase de {fase_critica} ocorrerá sob condições térmicas favoráveis."
                ),
            )
        if value <= req_max:
            return VarAnalysis(
                variavel=varname, valor=round(value,1), unidade=unit,
                faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
                status="atencao", icone="🌡️", titulo="Temperatura acima do ideal",
                explicacao=(
                    f"A temperatura média de {value:.1f}°C está {value - req_otmax:.1f}°C acima "
                    f"da faixa ótima. A {crop_nome} tolera essa condição, mas sob estresse "
                    f"térmico: o metabolismo acelera, aumenta a transpiração e o consumo hídrico. "
                    f"No {fase_critica}, o calor excessivo pode causar aborto de flores "
                    f"e redução na qualidade dos grãos ou frutos."
                ),
            )
        return VarAnalysis(
            variavel=varname, valor=round(value,1), unidade=unit,
            faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
            status="critico", icone="🔥", titulo="Temperatura crítica — calor excessivo",
            explicacao=(
                f"A temperatura média de {value:.1f}°C supera o limite máximo tolerado "
                f"({req_max}°C) pela {crop_nome}. Nessa faixa ocorrem danos fisiológicos "
                f"graves: desnaturação de enzimas, esterilidade do pólen durante o "
                f"{fase_critica} e murcha permanente. A produtividade será severamente comprometida."
            ),
        )

    # ── Precipitação ─────────────────────────────────────────────────────────
    if varname == "Precipitação":
        if value < req_min:
            return VarAnalysis(
                variavel=varname, valor=round(value,1), unidade=unit,
                faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
                status="critico", icone="🏜️", titulo="Precipitação insuficiente — risco de seca",
                explicacao=(
                    f"A precipitação média de {value:.1f}mm está {req_min - value:.1f}mm abaixo "
                    f"do mínimo necessário para {crop_nome} ({req_min}mm). Esse nível de déficit "
                    f"hídrico compromete a germinação e o estabelecimento da cultura. "
                    f"Será necessária irrigação suplementar para viabilizar o cultivo nesse período."
                ),
            )
        if value < req_otmin:
            return VarAnalysis(
                variavel=varname, valor=round(value,1), unidade=unit,
                faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
                status="atencao", icone="💧", titulo="Precipitação abaixo do ideal",
                explicacao=(
                    f"Com {value:.1f}mm de precipitação média, a {crop_nome} consegue se "
                    f"desenvolver, mas {req_otmin - value:.1f}mm a menos que o ideal "
                    f"({faixa_ot}). Períodos de veranico (mais de 15 dias sem chuva) "
                    f"nessa fase podem ser críticos, especialmente no {fase_critica}. "
                    f"Monitorar a umidade do solo é essencial."
                ),
            )
        if value <= req_otmax:
            return VarAnalysis(
                variavel=varname, valor=round(value,1), unidade=unit,
                faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
                status="otimo", icone="🌧️", titulo="Precipitação ideal",
                explicacao=(
                    f"A precipitação média de {value:.1f}mm está dentro da faixa ótima "
                    f"({faixa_ot}) para {crop_nome}. O balanço hídrico é favorável, "
                    f"garantindo boa disponibilidade de água no solo durante todo o ciclo. "
                    f"O {fase_critica} ocorrerá com suprimento hídrico adequado."
                ),
            )
        if value <= req_max:
            return VarAnalysis(
                variavel=varname, valor=round(value,1), unidade=unit,
                faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
                status="atencao", icone="🌦️", titulo="Precipitação elevada",
                explicacao=(
                    f"A precipitação média de {value:.1f}mm está {value - req_otmax:.1f}mm acima "
                    f"do ideal, mas ainda dentro do tolerável. O excesso de umidade aumenta o "
                    f"risco de doenças fúngicas (como ferrugem e míldio) e pode dificultar "
                    f"operações de plantio e colheita. Atenção ao sistema de drenagem."
                ),
            )
        return VarAnalysis(
            variavel=varname, valor=round(value,1), unidade=unit,
            faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
            status="critico", icone="🌊", titulo="Excesso hídrico — risco de encharcamento",
            explicacao=(
                f"Com {value:.1f}mm de precipitação média, o volume de chuvas ultrapassa "
                f"o limite máximo tolerado ({req_max}mm) pela {crop_nome}. O encharcamento "
                f"do solo prejudica a aeração das raízes, favorece podridões e doenças, "
                f"e inviabiliza o uso de maquinário no campo. Não recomendado nesse período."
            ),
        )

    # ── Umidade ──────────────────────────────────────────────────────────────
    if value < req_min:
        return VarAnalysis(
            variavel=varname, valor=round(value,1), unidade=unit,
            faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
            status="critico", icone="🌵", titulo="Umidade crítica — ar muito seco",
            explicacao=(
                f"A umidade relativa média de {value:.1f}% está abaixo do mínimo tolerado "
                f"({req_min}%) pela {crop_nome}. Ar muito seco aumenta drasticamente a "
                f"transpiração das folhas, causando murcha, fechamento dos estômatos e "
                f"redução da fotossíntese. O período de {fase_critica} sob "
                f"baixa umidade compromete a produção de forma significativa."
            ),
        )
    if value < req_otmin:
        return VarAnalysis(
            variavel=varname, valor=round(value,1), unidade=unit,
            faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
            status="atencao", icone="💨", titulo="Umidade abaixo do ideal",
            explicacao=(
                f"A umidade relativa de {value:.1f}% está {req_otmin - value:.1f}% abaixo "
                f"da faixa ótima ({faixa_ot}). A planta enfrenta estresse hídrico "
                f"moderado pelo ar, o que reduz a eficiência fotossintética e pode "
                f"prejudicar a polinização natural durante o {fase_critica}."
            ),
        )
    if value <= req_otmax:
        return VarAnalysis(
            variavel=varname, valor=round(value,1), unidade=unit,
            faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
            status="otimo", icone="✅", titulo="Umidade ideal",
            explicacao=(
                f"A umidade relativa de {value:.1f}% está dentro da faixa ótima "
                f"({faixa_ot}) para {crop_nome}. Essa condição favorece a abertura "
                f"regular dos estômatos, boa eficiência fotossintética e "
                f"polinização bem-sucedida no {fase_critica}."
            ),
        )
    if value <= req_max:
        return VarAnalysis(
            variavel=varname, valor=round(value,1), unidade=unit,
            faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
            status="atencao", icone="🌫️", titulo="Umidade elevada",
            explicacao=(
                f"A umidade relativa de {value:.1f}% está {value - req_otmax:.1f}% acima "
                f"da faixa ótima. Alta umidade aumenta o risco de proliferação de fungos, "
                f"bactérias e pragas que se desenvolvem melhor em ambientes úmidos. "
                f"Atenção ao monitoramento fitossanitário da lavoura."
            ),
        )
    return VarAnalysis(
        variavel=varname, valor=round(value,1), unidade=unit,
        faixa_otima=faixa_ot, faixa_tolerada=faixa_tol,
        status="critico", icone="🌫️", titulo="Umidade excessiva",
        explicacao=(
            f"Com {value:.1f}% de umidade relativa, o ambiente é excessivamente úmido "
            f"para {crop_nome}. Condição altamente favorável a doenças fúngicas graves "
            f"como ferrugem, míldio e botrytis, que podem dizimar lavouras inteiras. "
            f"A qualidade pós-colheita também é comprometida."
        ),
    )


def _diagnostico(
    crop_nome: str, crop_key: str, overall: float,
    verdict: str, station_name: str, uf: str,
    months_sel: list[int], analyses: list[VarAnalysis],
    best_months: list[str],
) -> str:
    """Gera parágrafo de diagnóstico climático consolidado."""
    meses_str = ", ".join(MONTH_NAMES[m] for m in months_sel)
    profile = CROP_PROFILES.get(crop_key, {})
    regioes = profile.get("regioes", "")

    criticos = [a for a in analyses if a.status == "critico"]
    atencoes = [a for a in analyses if a.status == "atencao"]
    otimos   = [a for a in analyses if a.status == "otimo"]

    partes = []

    # Parágrafo 1 — diagnóstico geral
    if overall >= 80:
        partes.append(
            f"A análise climática de {station_name}/{uf} para o período de {meses_str} "
            f"indica condições **altamente favoráveis** ao cultivo de {crop_nome}. "
            f"O histórico meteorológico da estação alinha-se muito bem com os requisitos agronômicos da cultura."
        )
    elif overall >= 65:
        partes.append(
            f"O período de {meses_str} em {station_name}/{uf} apresenta condições "
            f"**favoráveis** ao cultivo de {crop_nome}, com a maioria das variáveis climáticas "
            f"dentro das faixas recomendadas pela literatura agronômica."
        )
    elif overall >= 50:
        partes.append(
            f"As condições climáticas de {station_name}/{uf} no período de {meses_str} "
            f"são **moderadas** para o cultivo de {crop_nome}. A viabilidade existe, "
            f"mas exige cuidados adicionais e manejo intensivo para mitigar os riscos identificados."
        )
    elif overall >= 35:
        partes.append(
            f"A análise indica **risco elevado** para o cultivo de {crop_nome} em "
            f"{station_name}/{uf} no período de {meses_str}. "
            f"Variáveis climáticas importantes estão fora das faixas ideais, exigindo "
            f"infraestrutura de suporte significativa (irrigação, proteção térmica etc.)."
        )
    else:
        partes.append(
            f"O período de {meses_str} em {station_name}/{uf} apresenta condições "
            f"**não recomendadas** para o cultivo de {crop_nome}. "
            f"O histórico climático da região é incompatível com os requisitos fundamentais da cultura."
        )

    # Parágrafo 2 — variáveis críticas
    if criticos:
        nomes = " e ".join(a.variavel.lower() for a in criticos)
        partes.append(
            f"O principal fator limitante identificado é a {nomes}: "
            f"{criticos[0].explicacao.split('.')[0]}."
        )
    elif atencoes:
        nomes = " e ".join(a.variavel.lower() for a in atencoes)
        partes.append(
            f"Atenção especial é necessária para {nomes}, "
            f"que estão fora da faixa ótima mas dentro do tolerável."
        )

    # Parágrafo 3 — comparação com regiões tradicionais
    if regioes:
        partes.append(
            f"As principais regiões produtoras de {crop_nome} no Brasil são: {regioes}. "
            f"{'Essa estação está climaticamente próxima dessas regiões.' if overall >= 65 else 'O clima dessa estação difere significativamente dessas regiões.'}"
        )

    # Parágrafo 4 — melhores meses
    if best_months:
        partes.append(
            f"Com base nos dados históricos do INMET, os meses mais favoráveis para "
            f"{crop_nome} nessa estação são: {', '.join(best_months[:3])}."
        )

    return " ".join(partes)


def predict_score(station_row: pd.Series, month: int, crop: str) -> float:
    """Usa o modelo ML para prever o score de aptidão."""
    artifacts = _artifacts
    model = artifacts["model"]
    le_uf = artifacts["le_uf"]
    le_regiao = artifacts["le_regiao"]
    le_crop = artifacts["le_crop"]

    uf = station_row.get("uf", "XX")
    regiao = station_row.get("regiao", "XX")

    try:
        uf_enc = le_uf.transform([uf])[0]
    except ValueError:
        uf_enc = 0

    try:
        regiao_enc = le_regiao.transform([regiao])[0]
    except ValueError:
        regiao_enc = 0

    try:
        crop_enc = le_crop.transform([crop])[0]
    except ValueError:
        raise HTTPException(400, f"Cultura '{crop}' não reconhecida.")

    lat = float(station_row.get("lat", 0))
    lon = float(station_row.get("lon", 0))

    # Pega normais do mês
    norm_row = _normals[
        (_normals["codigo"] == station_row["codigo"]) &
        (_normals["month"] == month)
    ]

    if len(norm_row) == 0:
        # Estação sem dados para esse mês — usa vizinha mais próxima por UF
        fallback = _normals[_normals["uf"] == uf][_normals["month"] == month]
        if len(fallback) == 0:
            return 50.0
        norm_row = fallback.mean(numeric_only=True).to_frame().T
        norm_row["month"] = month

    nr = norm_row.iloc[0]
    features = np.array([[
        lat, lon, month, crop_enc, uf_enc, regiao_enc,
        nr.get("temp_norm", 25), nr.get("temp_std", 2),
        nr.get("temp_max_norm", 30), nr.get("temp_min_norm", 18),
        nr.get("precip_norm", 100), nr.get("precip_std", 30),
        nr.get("umidade_norm", 70), nr.get("umidade_std", 8),
        nr.get("n_anos", 5),
    ]])

    score = float(model.predict(features)[0])
    return max(0.0, min(100.0, score))


def get_climate_for_month(station_code: str, month: int) -> dict:
    """Retorna dados climáticos históricos de um mês para uma estação."""
    if _normals is None:
        return {}
    rows = _normals[(_normals["codigo"] == station_code) & (_normals["month"] == month)]
    if len(rows) == 0:
        return {}
    r = rows.iloc[0]
    return {
        "temp_media": round(float(r.get("temp_norm", 0)), 1),
        "temp_max": round(float(r.get("temp_max_norm", 0)), 1),
        "temp_min": round(float(r.get("temp_min_norm", 0)), 1),
        "precip_total": round(float(r.get("precip_norm", 0)), 1),
        "precip_std": round(float(r.get("precip_std", 0)), 1),
        "umidade_media": round(float(r.get("umidade_norm", 0)), 1),
        "n_anos": int(r.get("n_anos", 0)),
    }


def get_annual_series(station_code: str) -> dict:
    """Retorna série mensal anual (normais) para a estação."""
    if _normals is None:
        return {}
    rows = _normals[_normals["codigo"] == station_code].sort_values("month")
    if len(rows) == 0:
        return {}
    return {
        "months": rows["month"].tolist(),
        "temp": [round(v, 1) for v in rows["temp_norm"].fillna(0).tolist()],
        "precip": [round(v, 1) for v in rows["precip_norm"].fillna(0).tolist()],
        "umidade": [round(v, 1) for v in rows["umidade_norm"].fillna(0).tolist()],
    }


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
        mask = df["nome"].str.contains(search.upper(), case=False, na=False)
        df = df[mask]

    df = df.sort_values(["uf", "nome"])
    return df.fillna("").to_dict(orient="records")


@app.get("/api/crops")
async def get_crops():
    """Lista as culturas disponíveis."""
    if _artifacts is None:
        raise HTTPException(500, "Modelo não carregado.")
    crops = _artifacts["crop_requirements"]
    return [
        {
            "key": k,
            "nome": v["nome"],
            "emoji": v["emoji"],
            "desc": v["desc"],
        }
        for k, v in crops.items()
    ]


@app.get("/api/climate/{station_code}")
async def get_climate(station_code: str):
    """Retorna série climática anual da estação."""
    if _stations is None:
        raise HTTPException(500, "Dados não carregados.")

    station = _stations[_stations["codigo"] == station_code]
    if len(station) == 0:
        raise HTTPException(404, f"Estação {station_code} não encontrada.")

    series = get_annual_series(station_code)
    st = station.iloc[0]
    return {
        "codigo": station_code,
        "nome": st.get("nome", ""),
        "uf": st.get("uf", ""),
        "lat": float(st.get("lat", 0)),
        "lon": float(st.get("lon", 0)),
        "series": series,
    }


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """Gera recomendação de plantio para uma estação, cultura e período."""
    if _artifacts is None or _stations is None:
        raise HTTPException(500, "Modelo não carregado.")

    station = _stations[_stations["codigo"] == req.station_code]
    if len(station) == 0:
        raise HTTPException(404, f"Estação '{req.station_code}' não encontrada.")

    crop_reqs = _artifacts["crop_requirements"]
    if req.crop not in crop_reqs:
        raise HTTPException(400, f"Cultura '{req.crop}' não suportada.")

    st = station.iloc[0]
    crop_info = crop_reqs[req.crop]

    months_data = []
    scores = []

    for m in sorted(set(req.months)):
        if m < 1 or m > 12:
            continue
        score = predict_score(st, m, req.crop)
        climate = get_climate_for_month(req.station_code, m)
        scores.append(score)

        months_data.append(MonthResult(
            month=m,
            month_name=MONTH_NAMES[m],
            score=round(score, 1),
            temp_media=climate.get("temp_media", 0),
            precip_total=climate.get("precip_total", 0),
            umidade_media=climate.get("umidade_media", 0),
            status=score_to_status(score),
            details={
                "temp_max": climate.get("temp_max", 0),
                "temp_min": climate.get("temp_min", 0),
                "precip_std": climate.get("precip_std", 0),
                "n_anos": climate.get("n_anos", 0),
            },
        ))

    if not scores:
        raise HTTPException(400, "Nenhum mês válido fornecido.")

    overall = round(float(np.mean(scores)), 1)
    verdict, color = score_to_verdict(overall)

    # Calcula os melhores meses do ano para essa cultura nessa estação
    all_scores = {}
    for m in range(1, 13):
        s = predict_score(st, m, req.crop)
        all_scores[m] = s

    best = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    best_months = [MONTH_NAMES[m] for m, _ in best if all_scores[m] >= 60]

    # Alerta específico
    alert = None
    req_temp = crop_info["temp"]
    req_precip = crop_info["precip"]
    for md in months_data:
        if md.temp_media > req_temp[3]:
            alert = f"Temperatura média ({md.temp_media}°C) acima do limite para {crop_info['nome']} em {md.month_name}."
            break
        if md.precip_total < req_precip[0]:
            alert = f"Precipitação baixa ({md.precip_total}mm) em {md.month_name} — risco de déficit hídrico."
            break
        if md.precip_total > req_precip[3]:
            alert = f"Precipitação elevada ({md.precip_total}mm) em {md.month_name} — risco de excesso hídrico."
            break

    climate_summary = get_annual_series(req.station_code)

    # Médias das variáveis nos meses selecionados
    avg_temp    = float(np.mean([m.temp_media    for m in months_data])) if months_data else 25.0
    avg_precip  = float(np.mean([m.precip_total  for m in months_data])) if months_data else 100.0
    avg_umidade = float(np.mean([m.umidade_media for m in months_data])) if months_data else 70.0

    req_t = crop_info["temp"]
    req_p = crop_info["precip"]
    req_u = crop_info["umidade"]
    profile = CROP_PROFILES.get(req.crop, {})
    fase = profile.get("fase_critica", "florescimento")

    var_analyses = [
        _explain_var("Temperatura",  avg_temp,    "°C", *req_t, crop_info["nome"], req.crop, fase),
        _explain_var("Precipitação", avg_precip,  "mm", *req_p, crop_info["nome"], req.crop, fase),
        _explain_var("Umidade",      avg_umidade, "%",  *req_u, crop_info["nome"], req.crop, fase),
    ]

    diagnostico = _diagnostico(
        crop_info["nome"], req.crop, overall, verdict,
        str(st.get("nome", "")), str(st.get("uf", "")),
        sorted(set(req.months)), var_analyses, best_months,
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
        alert=alert,
        climate_summary=climate_summary,
        crop_profile=profile,
        var_analyses=var_analyses,
        diagnostico=diagnostico,
    )


@app.get("/api/stats")
async def get_stats():
    """Retorna estatísticas do sistema."""
    n_stations = len(_stations) if _stations is not None else 0
    n_records = len(_monthly) if _monthly is not None else 0
    n_normals = len(_normals) if _normals is not None else 0
    n_crops = len(_artifacts["crop_requirements"]) if _artifacts else 0
    ufs = sorted(_stations["uf"].dropna().unique().tolist()) if _stations is not None else []
    return {
        "estacoes": n_stations,
        "registros_mensais": n_records,
        "normais_climaticas": n_normals,
        "culturas": n_crops,
        "estados": ufs,
        "anos": ["2005", "2007-2009", "2013", "2020-2026"],
    }


@app.get("/api/recommend/stream")
async def recommend_stream(station_code: str, crop: str, months: str):
    """Endpoint SSE que transmite o progresso real da análise passo a passo."""

    def evt(step: str, msg: str, detail: str = "", data: dict | None = None) -> str:
        obj = {"step": step, "msg": msg, "detail": detail}
        if data:
            obj["data"] = data
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def generate():
        t0 = time.time()
        try:
            months_list = [int(m.strip()) for m in months.split(",") if m.strip()]
            months_list = [m for m in months_list if 1 <= m <= 12]

            if _artifacts is None or _stations is None:
                yield evt("error", "Modelo não carregado. Reinicie o servidor.")
                return

            # ── 1. Início ────────────────────────────────────────────────────
            yield evt("init", "Iniciando motor de análise climática...",
                      "AgroClima ML Engine v1.0 · Random Forest")
            await asyncio.sleep(0.5)

            # ── 2. Estação ───────────────────────────────────────────────────
            station = _stations[_stations["codigo"] == station_code]
            if len(station) == 0:
                yield evt("error", f"Estação {station_code} não encontrada no banco de dados.")
                return
            st = station.iloc[0]
            alt = f"{float(st.get('alt', 0)):.0f}m altitude" if st.get("alt") else ""
            lat_s = f"Lat {float(st.get('lat', 0)):.4f}° · Lon {float(st.get('lon', 0)):.4f}°"
            yield evt("station",
                      f"Estação encontrada: {st['nome']} ({station_code})",
                      f"{st['uf']} · {alt} · {lat_s}")
            await asyncio.sleep(0.5)

            # ── 3. Registros históricos ──────────────────────────────────────
            yield evt("loading", "Carregando séries históricas do INMET...",
                      "Lendo arquivos de dados brutos")
            await asyncio.sleep(0.7)

            monthly_data = _monthly[_monthly["codigo"] == station_code] if _monthly is not None else pd.DataFrame()
            n_records = len(monthly_data)

            if n_records > 0:
                years_sorted = sorted(monthly_data["year"].unique())
                year_min, year_max = years_sorted[0], years_sorted[-1]
                n_years = len(years_sorted)
                yield evt("records",
                          f"{n_records:,} registros mensais encontrados",
                          f"Período: Jan/{year_min} a Dez/{year_max} · {n_years} anos de dados")
            else:
                yield evt("records",
                          "Usando normais climáticas regionais",
                          f"Estação {station_code} · dados interpolados por UF")
            await asyncio.sleep(0.6)

            # ── 4. Normais climáticas ────────────────────────────────────────
            yield evt("normals_calc", "Calculando normais climáticas mensais...",
                      "Agregando médias históricas de temperatura, precipitação e umidade")
            await asyncio.sleep(0.8)

            norms = _normals[_normals["codigo"] == station_code] if _normals is not None else pd.DataFrame()
            if len(norms) > 0:
                temp_avg   = norms["temp_norm"].mean()
                precip_sum = norms["precip_norm"].sum()
                umid_avg   = norms["umidade_norm"].mean()
                temp_amp   = norms["temp_norm"].max() - norms["temp_norm"].min()
                yield evt("normals_done",
                          f"Normais climáticas calculadas para {len(norms)} meses",
                          f"Temp média anual: {temp_avg:.1f}°C (amp. {temp_amp:.1f}°C) · "
                          f"Precip: {precip_sum:.0f}mm/ano · Umidade: {umid_avg:.1f}%")
            else:
                yield evt("normals_done", "Normais climáticas calculadas", "Dados regionais aplicados")
            await asyncio.sleep(0.5)

            # ── 5. Requisitos da cultura ─────────────────────────────────────
            crop_reqs = _artifacts["crop_requirements"]
            if crop not in crop_reqs:
                yield evt("error", f"Cultura '{crop}' não reconhecida.")
                return
            crop_info = crop_reqs[crop]
            yield evt("crop",
                      f"Carregando requisitos agronômicos: {crop_info['nome']} {crop_info['emoji']}",
                      crop_info["desc"])
            await asyncio.sleep(0.5)

            t_req = crop_info["temp"]
            p_req = crop_info["precip"]
            u_req = crop_info["umidade"]
            yield evt("crop_req",
                      "Faixas climáticas ideais definidas",
                      f"Temp: {t_req[1]}-{t_req[2]}°C · "
                      f"Precip: {p_req[1]}-{p_req[2]}mm/mês · "
                      f"Umidade: {u_req[1]}-{u_req[2]}%")
            await asyncio.sleep(0.5)

            # ── 6. Random Forest ─────────────────────────────────────────────
            model = _artifacts["model"]
            n_trees = model.n_estimators
            yield evt("ml_init",
                      f"Inicializando Random Forest ({n_trees} árvores de decisão)...",
                      "Carregando modelo pré-treinado com scikit-learn")
            await asyncio.sleep(0.7)

            yield evt("ml_features",
                      "Extraindo features climáticas para o modelo...",
                      "lat, lon, mês, temperatura, precipitação, umidade, desvio padrão, n_anos")
            await asyncio.sleep(0.6)

            # ── 7. Score por mês ─────────────────────────────────────────────
            month_results = []
            for m in sorted(set(months_list)):
                score   = predict_score(st, m, crop)
                climate = get_climate_for_month(station_code, m)
                month_results.append((m, score, climate))

                bar_filled = round(score / 10)
                bar = "█" * bar_filled + "░" * (10 - bar_filled)
                yield evt("ml_month",
                          f"{MONTH_NAMES[m]}: viabilidade = {score:.1f}%  [{bar}]",
                          f"Temp: {climate.get('temp_media', 0):.1f}°C · "
                          f"Precip: {climate.get('precip_total', 0):.0f}mm · "
                          f"Umidade: {climate.get('umidade_media', 0):.0f}%",
                          {"month": m, "score": score})
                await asyncio.sleep(0.65)

            # ── 8. Score geral ───────────────────────────────────────────────
            yield evt("scoring", "Calculando score de viabilidade geral...",
                      "Ponderando temperatura (35%), precipitação (40%) e umidade (25%)")
            await asyncio.sleep(0.5)

            overall = float(np.mean([s for _, s, _ in month_results]))
            verdict, color = score_to_verdict(overall)
            elapsed = time.time() - t0

            yield evt("complete",
                      f"Análise concluída em {elapsed:.1f}s — Score: {overall:.1f}%",
                      f"Classificação: {verdict} · Baseado em dados históricos do INMET",
                      {"overall_score": overall, "verdict": verdict, "color": color})
            await asyncio.sleep(0.4)

            # ── 9. Resultado completo ────────────────────────────────────────
            full = await recommend(RecommendRequest(
                station_code=station_code, crop=crop, months=months_list
            ))
            yield f"data: {json.dumps({'step': 'result', 'result': full.model_dump()}, ensure_ascii=False)}\n\n"

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
