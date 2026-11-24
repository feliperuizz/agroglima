"""
Experimentos de avaliação do artigo (OE1 e OE3).

Bases comparadas:
- D0: base entregue na versão 1.0 (data/v1_0/), com as linhas duplicadas;
- D1: D0 sem duplicatas (isola o efeito do erro de leitura dupla);
- D2: base da versão 1.1 (data/), com o parser corrigido para 2020-2026
      e o critério de completude de 80% por variável.

Em cada base, o alvo S é recalculado pela regra antes da validação. Em cada
partição, um Random Forest novo é treinado do zero e a linha de base (média do
alvo por cultura e mês) usa só as partições de treino.

Uso: python api/avaliacao.py   ->  grava api/models/avaliacao.json
"""

import json
import math
import time
import numpy as np
import pandas as pd
import sklearn
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold

from train_model import (
    CROP_REQUIREMENTS, FEATURE_COLS, RF_PARAMS, STD_DEFAULTS,
    build_training_data, encode, group_cv, score_breakdown,
)

ROOT = Path(__file__).parent.parent
V10_DIR = ROOT / "data" / "v1_0"
V11_DIR = ROOT / "data"
OUT = Path(__file__).parent / "models" / "avaliacao.json"

# Variáveis que de fato entram na fórmula de S
RULE_FEATURES = ["month", "crop_enc", "temp_norm", "temp_std", "precip_norm", "precip_std", "umidade_norm"]


def normals_v10(monthly: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Recalcula os indicadores exatamente como o preprocess.py da versão 1.0."""
    normals = monthly.groupby(["codigo", "month"]).agg(
        precip_norm=("precip_total", "mean"),
        precip_std=("precip_total", "std"),
        temp_norm=("temp_media", "mean"),
        temp_std=("temp_media", "std"),
        umidade_norm=("umidade_media", "mean"),
        umidade_std=("umidade_media", "std"),
        temp_max_norm=("temp_max", "mean"),
        temp_min_norm=("temp_min", "mean"),
        n_anos=("year", "nunique"),
    ).reset_index()
    return normals.merge(stations[["codigo", "nome", "uf", "regiao", "lat", "lon"]], on="codigo", how="left")


def verdict(score: float) -> str:
    if score >= 80:
        return "Altamente Recomendado"
    if score >= 65:
        return "Recomendado"
    if score >= 50:
        return "Moderado"
    if score >= 35:
        return "Risco Elevado"
    return "Não Recomendado"


def rule_table(normals: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (estação, mês, cultura) com a decomposição completa da regra."""
    filled = normals.fillna(STD_DEFAULTS)
    rows = []
    for _, r in filled.iterrows():
        for crop in CROP_REQUIREMENTS:
            b = score_breakdown(r, crop)
            rows.append({"codigo": r["codigo"], "month": int(r["month"]), "crop": crop,
                         "n_anos": int(r["n_anos"]), **b})
    return pd.DataFrame(rows)


def auditoria_duplicacao() -> dict:
    """OE1: efeito exato da leitura dupla sobre desvios e pontuação (D0 x D1)."""
    m0 = pd.read_pickle(V10_DIR / "monthly_climate.pkl")
    st = pd.read_pickle(V10_DIR / "stations.pkl")
    m1 = m0.drop_duplicates()
    n0 = pd.read_pickle(V10_DIR / "climate_normals.pkl")
    n1 = normals_v10(m1, st)

    j = n0.merge(n1, on=["codigo", "month"], suffixes=("_dup", "_real"))
    multi = j[j["n_anos_real"] >= 2]
    fator_obs = (multi["precip_std_dup"] / multi["precip_std_real"]).replace([np.inf], np.nan)
    fator_teo = np.sqrt(2 * (multi["n_anos_real"] - 1) / (2 * multi["n_anos_real"] - 1))
    desvio_fator = float(np.nanmax(np.abs(fator_obs - fator_teo)))

    muda_desvio = int(((j["precip_std_dup"] - j["precip_std_real"]).abs() > 1e-9).sum()
                      + (j["precip_std_dup"].notna() & j["precip_std_real"].isna()).sum())

    s0 = rule_table(n0)
    s1 = rule_table(n1)
    d = s1.merge(s0, on=["codigo", "month", "crop"], suffixes=("_real", "_dup"))
    d["delta"] = d["score_real"] - d["score_dup"]
    d["muda_veredito"] = [verdict(a) != verdict(b) for a, b in zip(d["score_real"], d["score_dup"])]
    um_ano = d["n_anos_real"] == 1

    fatores = {n: math.sqrt(2 * (n - 1) / (2 * n - 1)) for n in range(2, 6)}

    return {
        "linhas_monthly": int(len(m0)),
        "linhas_unicas": int(len(m1)),
        "duplicatas": int(len(m0) - len(m1)),
        "indicadores": int(len(n0)),
        "celulas_um_ano": int((n1["n_anos"] == 1).sum()),
        "indicadores_com_desvio_alterado": muda_desvio,
        "fator_teorico_por_n": fatores,
        "maior_diferenca_fator_observado_vs_teorico": desvio_fator,
        "exemplos_regra": int(len(d)),
        "delta_medio": float(d["delta"].mean()),
        "delta_abs_medio": float(d["delta"].abs().mean()),
        "delta_min": float(d["delta"].min()),
        "delta_abs_medio_um_ano": float(d.loc[um_ano, "delta"].abs().mean()),
        "delta_abs_medio_multi_ano": float(d.loc[~um_ano, "delta"].abs().mean()),
        "exemplos_com_veredito_alterado": int(d["muda_veredito"].sum()),
    }, n1


def chuva_toda_ausente() -> dict:
    """OE1: meses que passavam no filtro de 100 h da v1.0 sem nenhuma hora de chuva válida."""
    raw = pd.read_pickle(V11_DIR / "monthly_raw.pkl")
    antigos = raw[raw["year"] <= 2013]
    passou = antigos[antigos["n_horas"] >= 100]
    sem_chuva = passou[passou["n_precip"] == 0]
    return {
        "meses_que_passavam_no_filtro_v10": int(len(passou)),
        "meses_sem_nenhuma_hora_de_chuva": int(len(sem_chuva)),
        "meses_com_menos_de_80pct_chuva": int((passou["n_precip"] < 0.8 * passou["horas_esperadas"]).sum()),
    }


def resumo_base(normals: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    n = normals["n_anos"]
    return {
        "linhas_monthly": int(len(monthly)),
        "anos": sorted(int(a) for a in monthly["year"].unique()),
        "estacoes": int(normals["codigo"].nunique()),
        "ufs": int(normals["uf"].nunique()),
        "indicadores": int(len(normals)),
        "n_anos_min": int(n.min()), "n_anos_mediana": float(n.median()),
        "n_anos_media": float(n.mean()), "n_anos_max": int(n.max()),
        "celulas_um_ano": int((n == 1).sum()),
        "celulas_10_anos_ou_mais": int((n >= 10).sum()),
    }


def compensacao(normals: pd.DataFrame) -> dict:
    """
    Efeito compensatório da soma ponderada: células com alguma nota parcial 0
    (variável fora dos limites absolutos) que ainda recebem nota alta, e contraste
    com o operador de mínimo (Lei de Liebig), usado pelo EcoCrop.
    """
    t = rule_table(normals)
    veto = t[t["veto"]]
    s_min = (np.minimum(np.minimum(t["s_temp"], t["s_precip"]), t["s_umid"])
             * t["bonus"] * t["pen_precip"] * t["pen_temp"] * 100).clip(upper=100)
    v_soma = t["score"].map(verdict)
    v_min = s_min.map(verdict)
    return {
        "exemplos": int(len(t)),
        "com_alguma_nota_zero": int(len(veto)),
        "nota_zero_e_S_maior_igual_50": int((veto["score"] >= 50).sum()),
        "nota_zero_e_S_maior_igual_60": int((veto["score"] >= 60).sum()),
        "nota_zero_e_S_maior_igual_65": int((veto["score"] >= 65).sum()),
        "maior_S_com_nota_zero": float(veto["score"].max()) if len(veto) else None,
        "veredito_muda_com_minimo": int((v_soma != v_min).sum()),
        "distribuicao_soma": v_soma.value_counts().to_dict(),
        "distribuicao_minimo": v_min.value_counts().to_dict(),
    }


def avaliar_modelo(normals: pd.DataFrame, nome: str, extra: bool = False) -> dict:
    df, _ = encode(build_training_data(normals.fillna(STD_DEFAULTS)))
    out = {"exemplos": int(len(df))}
    t0 = time.time()
    out["groupkfold_estacao"] = group_cv(df, df["codigo"])
    out["groupkfold_uf"] = group_cv(df, df["uf"])
    out["kfold_aleatorio"] = group_cv(df, None, splitter=KFold(5, shuffle=True, random_state=42))
    if extra:
        out["groupkfold_estacao_7_variaveis"] = group_cv(df, df["codigo"], feature_cols=RULE_FEATURES)
    print(f"  {nome}: {time.time() - t0:.0f}s")
    return out


def caso_a001(n0: pd.DataFrame, n1: pd.DataFrame, n2: pd.DataFrame) -> dict:
    out = {}
    for nome, n in [("D0_duplicada", n0), ("D1_sem_duplicatas", n1), ("D2_v11", n2)]:
        meses = {}
        for m in (10, 11, 12):
            r = n[(n["codigo"] == "A001") & (n["month"] == m)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            b = score_breakdown(r.fillna(STD_DEFAULTS), "soja")
            meses[m] = {
                "n_anos": int(r["n_anos"]),
                "temp_norm": float(r["temp_norm"]), "temp_std": None if pd.isna(r["temp_std"]) else float(r["temp_std"]),
                "precip_norm": float(r["precip_norm"]), "precip_std": None if pd.isna(r["precip_std"]) else float(r["precip_std"]),
                "umidade_norm": float(r["umidade_norm"]),
                **{k: (bool(v) if k == "veto" else float(v)) for k, v in b.items()},
            }
        out[nome] = meses
    return out


def main():
    t0 = time.time()
    res = {"data": pd.Timestamp.now().isoformat(timespec="seconds"),
           "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
           "rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
           "cv": "5 partes; RF novo por parte; linha de base = média de S por cultura e mês no treino"}

    print("OE1 - duplicação (D0 x D1)...")
    res["duplicacao"], n1 = auditoria_duplicacao()
    res["chuva_ausente_v10"] = chuva_toda_ausente()

    n0 = pd.read_pickle(V10_DIR / "climate_normals.pkl")
    n2 = pd.read_pickle(V11_DIR / "climate_normals.pkl")
    m1 = pd.read_pickle(V10_DIR / "monthly_climate.pkl").drop_duplicates()
    m2 = pd.read_pickle(V11_DIR / "monthly_climate.pkl")
    res["base_D1"] = resumo_base(n1, m1)
    res["base_D2"] = resumo_base(n2, m2)
    res["preprocess_v11"] = json.loads((V11_DIR / "preprocess_summary.json").read_text(encoding="utf-8"))
    res["preprocess_v11"].pop("arquivos_com_falha", None)

    res["caso_A001"] = caso_a001(n0, n1, n2)
    res["compensacao_D2"] = compensacao(n2)

    print("OE3 - validação do modelo...")
    res["modelo_D1"] = avaliar_modelo(n1, "D1")
    res["modelo_D2"] = avaliar_modelo(n2, "D2", extra=True)

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Resultados em {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
