"""
Preprocessamento dos dados INMET: lê todos os CSVs da pasta BASE/
e gera agregados mensais por estação (temp, precipitação, umidade).
Salva em data/monthly_climate.pkl, data/climate_normals.pkl e data/stations.pkl

Versão 1.1 (correções da auditoria):
- cada arquivo é lido uma única vez (a busca por *.CSV e *.csv duplicava tudo no Windows);
- reconhece os dois formatos do INMET: até 2019 ("DATA (YYYY-MM-DD)", "HORA (UTC)", -9999)
  e a partir de 2020 ("Data" AAAA/MM/DD, "Hora UTC" "0000 UTC", campo vazio como ausente);
- leituras fisicamente impossíveis viram ausentes (LIMITES_FISICOS);
- completude conjunta: um estação-ano-mês só entra se temperatura, chuva e umidade tiverem,
  cada uma, >= 80% das horas do mês (dias do mês × 24) e nenhuma lacuna seguida > 72 h;
- a chuva mensal é a soma das horas válidas escalada pelas horas esperadas/horas válidas,
  para não tratar as horas faltantes como chuva zero;
- chuva de um mês sem nenhuma hora válida fica ausente (NaN), e não 0 mm.
"""

import json
import os
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "BASE"
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Fração mínima de horas válidas para um mês entrar na base (por variável).
# 80% de um mês de 30 dias = 576 horas.
COMPLETUDE_MIN = 0.80

# Maior lacuna seguida (em horas) admitida em cada variável dentro do mês: 3 dias.
GAP_MAX_HORAS = 72

VARIAVEIS = ["precip", "temp", "temp_max", "temp_min", "umidade"]

# Faixas físicas plausíveis para leituras horárias; fora delas o valor é tratado como ausente.
# Chuva: 0 a 150 mm/h; temperatura: -15 a 50 °C; umidade relativa: 0 a 100%.
LIMITES_FISICOS = {
    "precip": (0.0, 150.0),
    "temp": (-15.0, 50.0),
    "temp_max": (-15.0, 50.0),
    "temp_min": (-15.0, 50.0),
    "umidade": (0.0, 100.0),
}


def normalize(text: str) -> str:
    """Remove acentos, espaços extras e passa para maiúsculas (ESTAÇÃO -> ESTACAO)."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.upper().split())


def parse_header(filepath: str) -> dict:
    """Extrai metadados da estação das 8 primeiras linhas do CSV (chaves normalizadas)."""
    meta = {}
    with open(filepath, encoding="latin-1") as f:
        for i, line in enumerate(f):
            if i >= 8:
                break
            line = line.strip()
            if ";" in line:
                parts = line.split(";", 1)
                key = normalize(parts[0].rstrip(":").strip())
                val = parts[1].strip().rstrip(";") if len(parts) > 1 else ""
                meta[key] = val
    return meta


def parse_float(val):
    """Converte string com vírgula decimal para float, trata -9999 como NaN."""
    try:
        v = float(str(val).replace(",", "."))
        return np.nan if v == -9999 else v
    except (ValueError, TypeError):
        return np.nan


def to_numeric(series: pd.Series) -> pd.Series:
    """Versão vetorizada de parse_float para uma coluna inteira."""
    s = pd.to_numeric(series.str.replace(",", ".", regex=False), errors="coerce")
    return s.mask(s == -9999)


def map_columns(columns) -> dict:
    """Mapeia as colunas do INMET (formatos antigo e novo) para nomes internos."""
    col_map = {}
    for col in columns:
        col_u = normalize(col)
        if col_u == "DATA" or col_u.startswith("DATA ("):
            col_map[col] = "data"
        elif col_u.startswith("HORA"):
            col_map[col] = "hora"
        elif "PRECIPITA" in col_u and "TOTAL" in col_u:
            col_map[col] = "precip"
        elif "TEMPERATURA DO AR" in col_u and "BULBO" in col_u:
            col_map[col] = "temp"
        elif col_u.startswith("TEMPERATURA MAXIMA") and "ANT" in col_u:
            col_map[col] = "temp_max"
        elif col_u.startswith("TEMPERATURA MINIMA") and "ANT" in col_u:
            col_map[col] = "temp_min"
        elif "UMIDADE RELATIVA DO AR" in col_u:
            col_map[col] = "umidade"
    return col_map


def max_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maior sequência de horas seguidas sem leitura válida, por mês, para chuva, temperatura
    e umidade. Horas ausentes do arquivo contam como lacuna (a série é completada hora a hora).
    """
    s = (df.dropna(subset=["ts"]).drop_duplicates("ts").set_index("ts")
           [["precip", "temp", "umidade"]].sort_index())
    ini = s.index.min().to_period("M").to_timestamp()
    fim = (s.index.max().to_period("M") + 1).to_timestamp() - pd.Timedelta(hours=1)
    full = s.reindex(pd.date_range(ini, fim, freq="h"))
    out = {}
    for c, nome in [("precip", "gap_precip"), ("temp", "gap_temp"), ("umidade", "gap_umid")]:
        na = full[c].isna()
        run = na.astype(int).groupby((~na).cumsum()).cumsum()
        out[nome] = run.groupby([run.index.year, run.index.month]).max()
    gaps = pd.DataFrame(out)
    gaps.index.names = ["year", "month"]
    return gaps.reset_index()


def process_file(filepath: str) -> tuple[dict | None, pd.DataFrame | None]:
    """Processa um arquivo CSV do INMET. Retorna (metadados, df_mensal)."""
    try:
        meta = parse_header(filepath)
        station_code = meta.get("CODIGO (WMO)", "")
        station_name = meta.get("ESTACAO", "")
        uf = meta.get("UF", "")
        regiao = meta.get("REGIAO", "")

        try:
            lat = float(meta.get("LATITUDE", "0").replace(",", "."))
            lon = float(meta.get("LONGITUDE", "0").replace(",", "."))
            alt = float(meta.get("ALTITUDE", "0").replace(",", "."))
        except ValueError:
            lat, lon, alt = 0.0, 0.0, 0.0

        # Lê o CSV pulando as 8 linhas de metadados; campo vazio e -9999 viram NaN
        df = pd.read_csv(
            filepath,
            sep=";",
            encoding="latin-1",
            skiprows=8,
            dtype=str,
            keep_default_na=False,
            na_values=["", "-9999", "-9999,0", "-9999.0"],
        )

        # Remove coluna vazia final
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()
        df.rename(columns=map_columns(df.columns), inplace=True)

        required = ["data", "precip", "temp", "umidade"]
        if not all(c in df.columns for c in required):
            return None, None

        # Datas: "2009-01-01" (até 2019) ou "2020/01/01" (a partir de 2020)
        df["data"] = pd.to_datetime(
            df["data"].str.strip().str.replace("/", "-", regex=False),
            format="%Y-%m-%d", errors="coerce",
        )
        for c in VARIAVEIS:
            if c in df.columns:
                df[c] = to_numeric(df[c])
            else:
                df[c] = np.nan

        # Leituras fora da faixa física viram ausentes (e são contadas)
        df["n_fora_faixa"] = 0
        for c, (lo, hi) in LIMITES_FISICOS.items():
            fora = df[c].notna() & ~df[c].between(lo, hi)
            df.loc[fora, c] = np.nan
            df["n_fora_faixa"] += fora.astype(int)

        df.dropna(subset=["data"], inplace=True)
        df["year"] = df["data"].dt.year
        df["month"] = df["data"].dt.month

        # Instante de cada leitura: "00:00" (até 2019) ou "0000 UTC" (a partir de 2020)
        hora = (df["hora"].str.extract(r"(\d{2})")[0].astype(float)
                if "hora" in df.columns else pd.Series(0.0, index=df.index))
        df["ts"] = df["data"] + pd.to_timedelta(hora, unit="h")

        # Agrega por mês, contando as horas válidas de cada variável
        g = df.groupby(["year", "month"])
        monthly = pd.DataFrame({
            "precip_total": g["precip"].sum(min_count=1),
            "temp_media": g["temp"].mean(),
            "temp_max": g["temp_max"].max(),
            "temp_min": g["temp_min"].min(),
            "umidade_media": g["umidade"].mean(),
            "n_horas": g["temp"].count(),
            "n_precip": g["precip"].count(),
            "n_umid": g["umidade"].count(),
            "n_fora_faixa": g["n_fora_faixa"].sum(),
        }).reset_index()

        # Sem máx./mín. da hora anterior, usa o extremo da temperatura horária
        sem_max = monthly["temp_max"].isna()
        monthly.loc[sem_max, "temp_max"] = g["temp"].max().values[sem_max.values]
        sem_min = monthly["temp_min"].isna()
        monthly.loc[sem_min, "temp_min"] = g["temp"].min().values[sem_min.values]

        # Dias exatos do mês (fevereiro bissexto = 29 dias = 696 h)
        dias = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1)).dt.days_in_month
        monthly["horas_esperadas"] = dias * 24
        monthly = monthly.merge(max_gaps(df), on=["year", "month"], how="left")

        monthly["codigo"] = station_code
        monthly["nome"] = station_name
        monthly["uf"] = uf
        monthly["regiao"] = regiao
        monthly["arquivo"] = os.path.basename(filepath)

        station_info = {
            "codigo": station_code,
            "nome": station_name,
            "uf": uf,
            "regiao": regiao,
            "lat": lat,
            "lon": lon,
            "alt": alt,
        }

        return station_info, monthly

    except Exception as e:
        print(f"  ERRO em {os.path.basename(filepath)}: {e}")
        return None, None


def apply_completeness(df_monthly: pd.DataFrame, limiar: float = COMPLETUDE_MIN,
                       gap_max: int = GAP_MAX_HORAS) -> pd.DataFrame:
    """
    Filtro conjunto: o estação-ano-mês inteiro é descartado se temperatura, chuva ou umidade
    tiver menos de `limiar` das horas do mês ou uma lacuna seguida maior que `gap_max` horas. Nos meses mantidos, a chuva é escalada pelas
    horas esperadas/horas válidas (supõe que as horas faltantes chovem, em média, como as válidas).
    """
    esperadas = df_monthly["horas_esperadas"]
    ok = ((df_monthly["n_horas"] >= limiar * esperadas)
          & (df_monthly["n_precip"] >= limiar * esperadas)
          & (df_monthly["n_umid"] >= limiar * esperadas)
          & (df_monthly[["gap_precip", "gap_temp", "gap_umid"]].max(axis=1) <= gap_max))
    df = df_monthly[ok].reset_index(drop=True)
    df["precip_soma"] = df["precip_total"]
    df["precip_total"] = df["precip_soma"] * df["horas_esperadas"] / df["n_precip"]
    return df


def compute_normals(df_monthly: pd.DataFrame, df_stations: pd.DataFrame) -> pd.DataFrame:
    """Indicadores mensais históricos por estação e mês do ano (média e desvio entre anos)."""
    normals = df_monthly.groupby(["codigo", "month"]).agg(
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

    return normals.merge(
        df_stations[["codigo", "nome", "uf", "regiao", "lat", "lon"]],
        on="codigo", how="left"
    )


def list_csv_files(base_dir: Path) -> list[Path]:
    """Lista cada CSV uma única vez, sem depender de maiúsculas/minúsculas na extensão."""
    files = {p.resolve() for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"}
    return sorted(files)


def run():
    csv_files = list_csv_files(BASE_DIR)
    total = len(csv_files)
    print(f"Encontrados {total} arquivos CSV em {BASE_DIR}")

    all_monthly = []
    all_stations = {}
    falhas = []

    for i, fp in enumerate(csv_files):
        if i % 200 == 0:
            print(f"  Processando {i}/{total}...")
        station_info, monthly_df = process_file(str(fp))
        if station_info and monthly_df is not None and len(monthly_df) > 0:
            # Metadados do arquivo mais recente prevalecem (lista ordenada por nome)
            all_stations[station_info["codigo"]] = station_info
            all_monthly.append(monthly_df)
        else:
            falhas.append(fp.name)

    if not all_monthly:
        print("Nenhum dado processado!")
        return

    df_raw = pd.concat(all_monthly, ignore_index=True)
    df_stations = pd.DataFrame(list(all_stations.values()))

    # Um mesmo estação-ano-mês em dois arquivos: fica o registro com mais horas válidas
    df_raw = (df_raw.sort_values("n_horas", ascending=False)
                    .drop_duplicates(subset=["codigo", "year", "month"])
                    .sort_values(["codigo", "year", "month"])
                    .reset_index(drop=True))

    df_monthly = apply_completeness(df_raw)
    normals = compute_normals(df_monthly, df_stations)
    # Estações que ficaram sem nenhum mês completo saem do cadastro
    df_stations = df_stations[df_stations["codigo"].isin(normals["codigo"])].reset_index(drop=True)

    df_monthly.to_pickle(DATA_DIR / "monthly_climate.pkl")
    normals.to_pickle(DATA_DIR / "climate_normals.pkl")
    df_stations.to_pickle(DATA_DIR / "stations.pkl")
    df_raw.to_pickle(DATA_DIR / "monthly_raw.pkl")

    resumo = {
        "arquivos_encontrados": total,
        "arquivos_lidos": total - len(falhas),
        "arquivos_com_falha": falhas,
        "meses_brutos": int(len(df_raw)),
        "meses_completos": int(len(df_monthly)),
        "limiar_completude": COMPLETUDE_MIN,
        "lacuna_max_horas": GAP_MAX_HORAS,
        "meses_excluidos_so_pela_lacuna": int(((df_raw["n_horas"] >= COMPLETUDE_MIN * df_raw["horas_esperadas"])
            & (df_raw["n_precip"] >= COMPLETUDE_MIN * df_raw["horas_esperadas"])
            & (df_raw["n_umid"] >= COMPLETUDE_MIN * df_raw["horas_esperadas"])
            & (df_raw[["gap_precip", "gap_temp", "gap_umid"]].max(axis=1) > GAP_MAX_HORAS)).sum()),
        "leituras_fora_da_faixa_fisica": int(df_raw["n_fora_faixa"].sum()),
        "fator_escala_chuva_medio": float((df_monthly["horas_esperadas"] / df_monthly["n_precip"]).mean()),
        "fator_escala_chuva_max": float((df_monthly["horas_esperadas"] / df_monthly["n_precip"]).max()),
        "meses_com_chuva_escalada": int((df_monthly["n_precip"] < df_monthly["horas_esperadas"]).sum()),
        "anos": sorted(int(a) for a in df_monthly["year"].unique()),
        "estacoes": int(len(df_stations)),
        "indicadores": int(len(normals)),
    }
    (DATA_DIR / "preprocess_summary.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nProcessamento concluído!")
    print(f"  Arquivos lidos: {resumo['arquivos_lidos']}/{total}")
    print(f"  Estações únicas: {len(df_stations)}")
    print(f"  Meses brutos: {len(df_raw)} | completos (>= {COMPLETUDE_MIN:.0%}): {len(df_monthly)}")
    print(f"  Indicadores estação-mês: {len(normals)}")
    print(f"  Arquivos salvos em: {DATA_DIR}")


if __name__ == "__main__":
    run()
