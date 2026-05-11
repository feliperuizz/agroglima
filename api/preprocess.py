"""
Preprocessamento dos dados INMET: lê todos os CSVs da pasta BASE/
e gera agregados mensais por estação (temp, precipitação, umidade).
Salva em data/monthly_climate.parquet e data/stations.parquet
"""

import os
import glob
import re
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "BASE"
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Colunas que vamos usar (posição após o cabeçalho de 8 linhas)
COLS_USE = {
    "DATA (YYYY-MM-DD)": "data",
    "HORA (UTC)": "hora",
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": "precip",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "temp",
    "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)": "temp_max",
    "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)": "temp_min",
    "UMIDADE RELATIVA DO AR, HORARIA (%)": "umidade",
}


def parse_header(filepath: str) -> dict:
    """Extrai metadados da estação das 8 primeiras linhas do CSV."""
    meta = {}
    with open(filepath, encoding="latin-1") as f:
        for i, line in enumerate(f):
            if i >= 8:
                break
            line = line.strip()
            if ";" in line:
                parts = line.split(";", 1)
                key = parts[0].rstrip(":").strip()
                val = parts[1].strip() if len(parts) > 1 else ""
                meta[key] = val
    return meta


def parse_float(val):
    """Converte string com vírgula decimal para float, trata -9999 como NaN."""
    try:
        v = float(str(val).replace(",", "."))
        return np.nan if v == -9999 else v
    except (ValueError, TypeError):
        return np.nan


def process_file(filepath: str) -> tuple[dict | None, pd.DataFrame | None]:
    """Processa um arquivo CSV do INMET. Retorna (metadados, df_mensal)."""
    try:
        meta = parse_header(filepath)
        station_code = meta.get("CODIGO (WMO)", "")
        station_name = meta.get("ESTAÇÃO", "")
        uf = meta.get("UF", "")
        regiao = meta.get("REGIÃO", "")

        lat_str = meta.get("LATITUDE", "0").replace(",", ".")
        lon_str = meta.get("LONGITUDE", "0").replace(",", ".")
        alt_str = meta.get("ALTITUDE", "0").replace(",", ".")

        try:
            lat = float(lat_str)
            lon = float(lon_str)
            alt = float(alt_str)
        except ValueError:
            lat, lon, alt = 0.0, 0.0, 0.0

        # Lê o CSV pulando as 8 linhas de metadados
        df = pd.read_csv(
            filepath,
            sep=";",
            encoding="latin-1",
            skiprows=8,
            decimal=",",
            na_values=["-9999", "-9999.0", ""],
            dtype=str,
        )

        # Remove coluna vazia final
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()

        # Mapeamento flexível de colunas por palavras-chave
        col_map = {}
        for col in df.columns:
            col_u = col.upper()
            if "DATA" in col_u and "YYYY" in col_u:
                col_map[col] = "data"
            elif "PRECIPITA" in col_u and "TOTAL" in col_u:
                col_map[col] = "precip"
            elif "TEMPERATURA DO AR" in col_u and "BULBO" in col_u:
                col_map[col] = "temp"
            elif "TEMPERATURA M" in col_u and "XIMA" in col_u and "ANT" in col_u:
                col_map[col] = "temp_max"
            elif "TEMPERATURA M" in col_u and "NIMA" in col_u and "ANT" in col_u:
                col_map[col] = "temp_min"
            elif "UMIDADE RELATIVA DO AR" in col_u:
                col_map[col] = "umidade"

        df.rename(columns=col_map, inplace=True)

        required = ["data", "precip", "temp", "umidade"]
        if not all(c in df.columns for c in required):
            return None, None

        # Conversão de tipos
        df["data"] = pd.to_datetime(df["data"], format="%Y-%m-%d", errors="coerce")
        for c in ["precip", "temp", "temp_max", "temp_min", "umidade"]:
            if c in df.columns:
                df[c] = df[c].apply(parse_float)

        df.dropna(subset=["data"], inplace=True)
        df["year"] = df["data"].dt.year
        df["month"] = df["data"].dt.month

        # Agrega por mês
        monthly = df.groupby(["year", "month"]).agg(
            precip_total=("precip", "sum"),
            temp_media=("temp", "mean"),
            temp_max=("temp_max", "max") if "temp_max" in df.columns else ("temp", "max"),
            temp_min=("temp_min", "min") if "temp_min" in df.columns else ("temp", "min"),
            umidade_media=("umidade", "mean"),
            n_horas=("temp", "count"),
        ).reset_index()

        monthly["codigo"] = station_code
        monthly["nome"] = station_name
        monthly["uf"] = uf
        monthly["regiao"] = regiao

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


def run():
    csv_files = glob.glob(str(BASE_DIR / "*.CSV")) + glob.glob(str(BASE_DIR / "*.csv"))
    total = len(csv_files)
    print(f"Encontrados {total} arquivos CSV em {BASE_DIR}")

    all_monthly = []
    all_stations = {}

    for i, fp in enumerate(csv_files):
        if i % 200 == 0:
            print(f"  Processando {i}/{total}...")
        station_info, monthly_df = process_file(fp)
        if station_info and monthly_df is not None and len(monthly_df) > 0:
            code = station_info["codigo"]
            if code not in all_stations:
                all_stations[code] = station_info
            all_monthly.append(monthly_df)

    if not all_monthly:
        print("Nenhum dado processado!")
        return

    df_monthly = pd.concat(all_monthly, ignore_index=True)
    df_stations = pd.DataFrame(list(all_stations.values()))

    # Filtra meses com dados insuficientes (menos de 100 horas)
    df_monthly = df_monthly[df_monthly["n_horas"] >= 100]

    # Calcula normais climáticas (média histórica por estação + mês)
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

    # Merge com info da estação
    normals = normals.merge(
        df_stations[["codigo", "nome", "uf", "regiao", "lat", "lon"]],
        on="codigo", how="left"
    )

    out_monthly = DATA_DIR / "monthly_climate.pkl"
    out_normals = DATA_DIR / "climate_normals.pkl"
    out_stations = DATA_DIR / "stations.pkl"

    df_monthly.to_pickle(out_monthly)
    normals.to_pickle(out_normals)
    df_stations.to_pickle(out_stations)

    print(f"\nProcessamento concluído!")
    print(f"  Estações únicas: {len(df_stations)}")
    print(f"  Registros mensais: {len(df_monthly)}")
    print(f"  Normais climáticas: {len(normals)}")
    print(f"  Arquivos salvos em: {DATA_DIR}")


if __name__ == "__main__":
    run()
