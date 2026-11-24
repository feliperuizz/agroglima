"""
Treinamento do modelo de Machine Learning para recomendação de plantio.

Estratégia:
- Para cada estação + mês + cultura: calcula um "score de aptidão"
  baseado em quão bem as condições históricas atendem os requisitos da cultura
- Treina um Random Forest Regressor para aprender esses padrões
- Avalia com GroupKFold por estação contra uma linha de base (média por cultura e mês)
- Salva o modelo, os encoders e as métricas (api/models/metrics.json)

Desde a versão 1.1 a API serve a regra diretamente (exata e explicável);
o modelo é mantido como objeto de avaliação do trabalho.
"""

import json
import numpy as np
import pandas as pd
import joblib
import sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ─── Requisitos climáticos por cultura ────────────────────────────────────────
# Fonte: literatura agronômica e EMBRAPA
# (temp_min, temp_otima_min, temp_otima_max, temp_max): temperaturas em °C
# (precip_min, precip_otima_min, precip_otima_max, precip_max): precipitação mensal em mm
# (umid_min, umid_otima_min, umid_otima_max, umid_max): umidade relativa em %
# meses_plantio: meses preferenciais de plantio (1=jan...12=dez)

CROP_REQUIREMENTS = {
    "soja": {
        "nome": "Soja",
        "emoji": "🌿",
        # Fonte: Embrapa Soja — Sistema de Produção 15 (2011)
        # Solo para semeadura: 20-30°C; dano acima de 40°C; mín absoluto operacional: 20°C
        "temp": (20, 25, 30, 40),
        "precip": (60, 100, 200, 300),
        "umidade": (50, 60, 85, 95),
        "meses_favoraveis": [9, 10, 11, 12, 1],
        "desc": "Oleaginosa de ciclo curto, sensível a geadas e déficit hídrico",
    },
    "milho": {
        "nome": "Milho",
        "emoji": "🌽",
        # Fonte: Embrapa Milho e Sorgo — Agência de Informação Tecnológica
        # Germinação solo: 25-30°C; desenvolvimento: 24-30°C; dano abaixo de 10°C ou acima de 40°C
        "temp": (10, 24, 30, 40),
        "precip": (50, 80, 180, 280),
        "umidade": (45, 55, 80, 90),
        "meses_favoraveis": [9, 10, 11, 12, 1, 2],
        "desc": "Cereal versátil, requer boa disponibilidade hídrica no florescimento",
    },
    "cafe": {
        "nome": "Café",
        "emoji": "☕",
        # Fonte: Camargo & Camargo (2001) Bragantia v.60 n.1; Embrapa Café
        # Ótimo: 18-22°C (ideal 19-21°C); acima de 23°C: frutificação precoce e queda de qualidade
        # Mín absoluto: 18°C (geada causa danos severos); máx: 28°C sem prejuízo grave
        "temp": (18, 19, 22, 28),
        # Precipitação anual ideal: 1.200-1.800mm → 100-150mm/mês em média
        "precip": (80, 100, 180, 250),
        "umidade": (50, 60, 80, 90),
        "meses_favoraveis": [9, 10, 11],
        "desc": "Cultura perene, sensível a geadas e excesso de chuvas no período de colheita",
    },
    "arroz": {
        "nome": "Arroz",
        "emoji": "🍚",
        # Fonte: Embrapa Arroz e Feijão — Agência de Informação Tecnológica
        # Germinação: ≥20°C; florescimento: 30-33°C ideal; esterilidade acima de 35°C
        # Precipitação: ~1.400-1.500mm/ciclo para terras altas
        "temp": (20, 22, 33, 38),
        "precip": (100, 125, 200, 300),
        "umidade": (60, 70, 90, 98),
        "meses_favoraveis": [10, 11, 12, 1],
        "desc": "Gramínea semiaquática, requer alta disponibilidade hídrica",
    },
    "feijao": {
        "nome": "Feijão",
        "emoji": "🥔",
        "temp": (15, 18, 26, 33),
        "precip": (40, 70, 130, 200),
        "umidade": (50, 55, 75, 85),
        "meses_favoraveis": [1, 2, 6, 7, 8],
        "desc": "Leguminosa de ciclo curto com alta adaptabilidade climática",
    },
    "trigo": {
        "nome": "Trigo",
        "emoji": "🌾",
        # Fonte: Embrapa Trigo — Agência de Informação Tecnológica
        # Emergência: ~15°C; afilhamento: 15-20°C; espigamento: 18-24°C (mín 10°C, máx 32°C)
        "temp": (10, 15, 20, 32),
        "precip": (30, 50, 100, 150),
        "umidade": (50, 55, 75, 85),
        "meses_favoraveis": [4, 5, 6, 7],
        "desc": "Cereal de inverno, sensível ao calor excessivo e chuvas na maturação",
    },
    "algodao": {
        "nome": "Algodão",
        "emoji": "👕",
        "temp": (18, 22, 30, 38),
        "precip": (60, 90, 160, 250),
        "umidade": (45, 50, 70, 80),
        "meses_favoraveis": [10, 11, 12, 1],
        "desc": "Cultura de fibra, sensível à umidade excessiva durante a colheita",
    },
    "cana": {
        "nome": "Cana-de-açúcar",
        "emoji": "🎋",
        "temp": (18, 22, 30, 38),
        "precip": (100, 150, 250, 350),
        "umidade": (55, 65, 85, 95),
        "meses_favoraveis": [8, 9, 10, 11],
        "desc": "Gramínea de ciclo longo, alta produtividade em regiões tropicais úmidas",
    },
    "mandioca": {
        "nome": "Mandioca",
        "emoji": "🌱",
        "temp": (18, 22, 28, 35),
        "precip": (50, 80, 150, 250),
        "umidade": (50, 60, 80, 90),
        "meses_favoraveis": [9, 10, 11],
        "desc": "Raiz tuberosa resistente à seca, base alimentar brasileira",
    },
    "sorgo": {
        "nome": "Sorgo",
        "emoji": "🌿",
        "temp": (12, 20, 32, 40),
        "precip": (40, 60, 130, 200),
        "umidade": (40, 50, 75, 85),
        "meses_favoraveis": [10, 11, 12, 1, 2],
        "desc": "Cereal tolerante à seca, alternativa ao milho em regiões semiáridas",
    },
}


def aptidao_score(val, t_min, t_ot_min, t_ot_max, t_max):
    """
    Calcula score 0-1 de aptidão dado um valor e os thresholds da cultura.
    Score = 1.0 dentro da faixa ótima, decaimento linear fora, 0 fora dos limites.
    """
    if np.isnan(val):
        return 0.5  # incerteza
    if val < t_min or val > t_max:
        return 0.0
    if t_ot_min <= val <= t_ot_max:
        return 1.0
    if val < t_ot_min:
        return (val - t_min) / (t_ot_min - t_min) if t_ot_min != t_min else 1.0
    return (t_max - val) / (t_max - t_ot_max) if t_max != t_ot_max else 1.0


PESOS = {"temp": 0.35, "precip": 0.40, "umid": 0.25}


def score_breakdown(row, crop_key) -> dict:
    """
    Decompõe a pontuação de um mês: notas parciais, base ponderada, bônus,
    penalidades e pontuação final. `veto` indica variável fora dos limites absolutos.
    """
    req = CROP_REQUIREMENTS[crop_key]

    s_temp = aptidao_score(row["temp_norm"], *req["temp"])
    s_precip = aptidao_score(row["precip_norm"], *req["precip"])
    s_umid = aptidao_score(row["umidade_norm"], *req["umidade"])
    base = s_temp * PESOS["temp"] + s_precip * PESOS["precip"] + s_umid * PESOS["umid"]

    # Bonus de mês favorável
    bonus = 1.1 if row["month"] in req["meses_favoraveis"] else 1.0

    # Penalidade por alta variabilidade
    pen_precip = max(0.0, 1.0 - (row.get("precip_std", 0) / (row["precip_norm"] + 1)) * 0.3)
    pen_temp = max(0.0, 1.0 - (row.get("temp_std", 0) / 5.0) * 0.2)

    score = min(100.0, round(base * bonus * pen_precip * pen_temp * 100, 1))
    return {
        "s_temp": float(s_temp), "s_precip": float(s_precip), "s_umid": float(s_umid),
        "base": float(base), "bonus": bonus,
        "pen_precip": float(pen_precip), "pen_temp": float(pen_temp),
        "score": float(score),
        "veto": bool(min(s_temp, s_precip, s_umid) == 0.0),
    }


def compute_crop_score(row, crop_key):
    """Calcula score total (0-100) de aptidão climática de um mês para uma cultura."""
    return score_breakdown(row, crop_key)["score"]


def build_training_data(normals: pd.DataFrame) -> pd.DataFrame:
    """Cria dataset de treino: uma linha por (estação, mês, cultura)."""
    rows = []
    for _, row in normals.iterrows():
        for crop_key in CROP_REQUIREMENTS:
            score = compute_crop_score(row, crop_key)
            rows.append({
                "codigo": row["codigo"],
                "uf": row["uf"],
                "regiao": row["regiao"],
                "lat": row.get("lat", 0),
                "lon": row.get("lon", 0),
                "month": row["month"],
                "crop": crop_key,
                "temp_norm": row["temp_norm"],
                "temp_std": row.get("temp_std", 0),
                "temp_max_norm": row.get("temp_max_norm", row["temp_norm"] + 3),
                "temp_min_norm": row.get("temp_min_norm", row["temp_norm"] - 3),
                "precip_norm": row["precip_norm"],
                "precip_std": row.get("precip_std", 0),
                "umidade_norm": row["umidade_norm"],
                "umidade_std": row.get("umidade_std", 0),
                "n_anos": row.get("n_anos", 1),
                "score": score,
            })
    return pd.DataFrame(rows)


FEATURE_COLS = [
    "lat", "lon", "month", "crop_enc", "uf_enc", "regiao_enc",
    "temp_norm", "temp_std", "temp_max_norm", "temp_min_norm",
    "precip_norm", "precip_std",
    "umidade_norm", "umidade_std",
    "n_anos",
]

# Hiperparâmetros do modelo; max_features=1.0 é o padrão do scikit-learn >= 1.1
# (todas as variáveis avaliadas em cada divisão) e fica explícito para reprodução.
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    max_features=1.0,
    n_jobs=-1,
    random_state=42,
)

STD_DEFAULTS = {"precip_std": 30, "temp_std": 2, "umidade_std": 8}


def load_normals(path: Path = DATA_DIR / "climate_normals.pkl") -> pd.DataFrame:
    """Carrega os indicadores e preenche desvios indefinidos (células de um único ano)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo {path} não encontrado. Execute preprocess.py primeiro."
        )
    normals = pd.read_pickle(path)
    return normals.fillna(STD_DEFAULTS)


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica os LabelEncoders de UF, região, cultura e estação."""
    encoders = {
        "le_uf": LabelEncoder(), "le_regiao": LabelEncoder(),
        "le_crop": LabelEncoder(), "le_codigo": LabelEncoder(),
    }
    df = df.copy()
    df["uf_enc"] = encoders["le_uf"].fit_transform(df["uf"].fillna("XX"))
    df["regiao_enc"] = encoders["le_regiao"].fit_transform(df["regiao"].fillna("XX"))
    df["crop_enc"] = encoders["le_crop"].fit_transform(df["crop"])
    df["codigo_enc"] = encoders["le_codigo"].fit_transform(df["codigo"].fillna("XX"))
    return df, encoders


def group_cv(df: pd.DataFrame, groups: pd.Series, n_splits: int = 5,
             feature_cols: list[str] = FEATURE_COLS, splitter=None) -> dict:
    """
    Validação cruzada agrupada: um Random Forest novo é treinado em cada partição
    de treino; a linha de base (média do alvo por cultura e mês) também é calculada
    só com as partições de treino. Retorna métricas médias, por parte e por cultura.
    """
    X = df[feature_cols].fillna(0).values
    y = df["score"].values
    pred_rf = np.zeros(len(df))
    pred_base = np.zeros(len(df))
    folds = []

    splitter = splitter or GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
        model = RandomForestRegressor(**RF_PARAMS).fit(X[tr], y[tr])
        pred_rf[te] = model.predict(X[te])

        base = df.iloc[tr].groupby(["crop", "month"])["score"].mean()
        chave = pd.MultiIndex.from_frame(df.iloc[te][["crop", "month"]])
        pred_base[te] = base.reindex(chave).fillna(df["score"].iloc[tr].mean()).values

        folds.append({
            "parte": k + 1,
            "grupos_teste": int(pd.Series(groups).iloc[te].nunique()) if groups is not None else None,
            "r2_rf": float(r2_score(y[te], pred_rf[te])),
            "mae_rf": float(mean_absolute_error(y[te], pred_rf[te])),
            "r2_base": float(r2_score(y[te], pred_base[te])),
            "mae_base": float(mean_absolute_error(y[te], pred_base[te])),
        })

    por_cultura = {
        crop: {
            "mae_rf": float(mean_absolute_error(y[m], pred_rf[m])),
            "mae_base": float(mean_absolute_error(y[m], pred_base[m])),
        }
        for crop in sorted(df["crop"].unique())
        for m in [(df["crop"] == crop).values]
    }

    return {
        "n_splits": n_splits,
        "r2_rf": float(np.mean([f["r2_rf"] for f in folds])),
        "mae_rf": float(np.mean([f["mae_rf"] for f in folds])),
        "r2_base": float(np.mean([f["r2_base"] for f in folds])),
        "mae_base": float(np.mean([f["mae_base"] for f in folds])),
        "partes": folds,
        "por_cultura": por_cultura,
    }


def train():
    print("Carregando indicadores mensais...")
    normals = load_normals()
    print(f"  {len(normals)} indicadores estação-mês carregados")

    print("Construindo dataset de treino...")
    df = build_training_data(normals)
    print(f"  {len(df)} exemplos de treino gerados")

    df, encoders = encode(df)

    print("Validação cruzada agrupada por estação (GroupKFold, 5 partes)...")
    cv = group_cv(df, df["codigo"])
    print(f"  Random Forest:  R² {cv['r2_rf']:.3f} | MAE {cv['mae_rf']:.2f}")
    print(f"  Linha de base:  R² {cv['r2_base']:.3f} | MAE {cv['mae_base']:.2f}")

    print("Treinando Random Forest final (todas as estações)...")
    X = df[FEATURE_COLS].fillna(0).values
    y = df["score"].values
    model = RandomForestRegressor(**RF_PARAMS).fit(X, y)
    pred_train = model.predict(X)
    print(f"  Ajuste interno: R² {r2_score(y, pred_train):.3f} | MAE {mean_absolute_error(y, pred_train):.2f}")

    # Salva tudo
    artifacts = {
        "model": model,
        **encoders,
        "feature_cols": FEATURE_COLS,
        "crop_requirements": CROP_REQUIREMENTS,
    }
    joblib.dump(artifacts, MODEL_DIR / "recommendation_model.joblib", compress=3)

    metrics = {
        "data": pd.Timestamp.now().isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__,
        "rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
        "n_exemplos": int(len(df)),
        "ajuste_interno": {
            "r2": float(r2_score(y, pred_train)),
            "mae": float(mean_absolute_error(y, pred_train)),
        },
        "groupkfold_estacao": cv,
    }
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nModelo salvo em {MODEL_DIR / 'recommendation_model.joblib'}")
    print(f"Métricas salvas em {MODEL_DIR / 'metrics.json'}")
    print("Treinamento concluído!")


if __name__ == "__main__":
    train()
