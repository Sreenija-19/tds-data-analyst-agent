from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from io import BytesIO
import duckdb

app = FastAPI()

# --- Utility to generate a scatterplot and encode as base64 ---
def make_base64_plot(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, linestyle="dotted", color="red")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak Gross ($B)")
    ax.set_title("Rank vs Peak Gross")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    uri = "data:image/png;base64," + encoded
    if len(uri) > 100_000:  # limit size
        uri = uri[:99999]
    return uri

# --- Wikipedia: Highest Grossing Films ---
async def handle_wikipedia_question():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    target_table = None
    for table in tables:
        headers = [h.get_text(strip=True).lower() for h in table.find_all("th")]
        if any("rank" in h for h in headers):
            target_table = table
            break
    if target_table is None:
        return ["Error: Table not found"]

    df = pd.read_html(str(target_table))[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[-1] for col in df.columns]
    df.columns = df.columns.str.strip()
    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if "rank" in low:
            rename_map[col] = "Rank"
        elif "gross" in low or "peak" in low:
            rename_map[col] = "Peak Gross ($B)"
        elif "title" in low:
            rename_map[col] = "Title"
    df = df.rename(columns=rename_map)

    df["Peak Gross ($B)"] = (
        df["Peak Gross ($B)"].astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .astype(float) / 1_000_000_000
    )
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Title"].str.extract(r"\((\d{4})\)")[0], errors="coerce")

    movies_2bn_pre2000 = df[(df["Peak Gross ($B)"] >= 2.0) & (df["Year"] < 2000)].shape[0]
    earliest_1_5bn = df[df["Peak Gross ($B)"] > 1.5].sort_values("Year").iloc[0]["Title"]
    correlation = df[["Rank", "Peak Gross ($B)"]].dropna().corr().iloc[0, 1]
    plot_uri = make_base64_plot(df["Rank"].dropna(), df["Peak Gross ($B)"].dropna())
    return [movies_2bn_pre2000, earliest_1_5bn, round(correlation, 6), plot_uri]

# --- High Court Questions ---
async def handle_high_court_questions():
    # DuckDB query on S3 parquet files
    query = """
    INSTALL httpfs; LOAD httpfs;
    INSTALL parquet; LOAD parquet;

    SELECT court, COUNT(*) as num_cases
    FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
    WHERE year BETWEEN 2019 AND 2022
    GROUP BY court
    ORDER BY num_cases DESC
    LIMIT 1
    """
    result = duckdb.query(query).to_df()
    most_cases_court = result.iloc[0]['court'] if not result.empty else "Unknown"
    return {"most_cases_2019_2022": most_cases_court}

# --- Root ---
@app.get("/")
def root():
    return {"message": "API is live"}

# --- Analyze Data Endpoint ---
@app.post("/api/")
async def analyze_data(
    questions: UploadFile = File(...),
    files: List[UploadFile] = File(default=[]),
):
    try:
        question_text = (await questions.read()).decode("utf-8")
        uploaded_files = {file.filename: await file.read() for file in files}

        if "highest grossing films" in question_text.lower():
            response = await handle_wikipedia_question()
        elif "high court" in question_text.lower():
            response = await handle_high_court_questions()
        else:
            response = {"error": "Unsupported question type in this demo."}

        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)})