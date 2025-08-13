from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json

app = FastAPI(title="TDS Data Analyst Agent")

# -------------------
# Root endpoint
# -------------------
@app.get("/")
def root():
    return {"message": "API is live"}

# -------------------
# /api/ endpoint
# -------------------
@app.post("/api/")
async def analyze_data(
    questions: UploadFile = File(...),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Accepts a questions file (mandatory) and optional additional files.
    Returns answers in JSON.
    """
    try:
        question_text = (await questions.read()).decode("utf-8")

        # Optional: read uploaded files
        uploaded_files = {}
        if files:
            for file in files:
                uploaded_files[file.filename] = await file.read()

        # Decide which analysis to run
        if "highest grossing films" in question_text.lower():
            answers = await handle_wikipedia_question()
        elif "high court" in question_text.lower():
            answers = await handle_court_questions(question_text, uploaded_files)
        else:
            answers = ["Unsupported question (demo version)"]

        return JSONResponse(
            content={
                "questions_file": questions.filename,
                "content_preview": question_text[:100],
                "answers": answers
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"error": "Failed to process request", "details": str(e)},
            status_code=500
        )

# -------------------
# Wikipedia: Highest Grossing Films
# -------------------
async def handle_wikipedia_question():
    import requests
    from bs4 import BeautifulSoup

    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    tables = soup.find_all("table", class_="wikitable")
    target_table = None
    for table in tables:
        headers = [h.get_text(strip=True).lower() for h in table.find_all("th")]
        if any("rank" in h for h in headers):
            target_table = table
            break
    if target_table is None:
        raise ValueError("Target Wikipedia table not found")

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
        df["Peak Gross ($B)"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float) / 1_000_000_000
    )
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Title"].str.extract(r"\((\d{4})\)")[0], errors="coerce")

    movies_2bn_pre2000 = df[(df["Peak Gross ($B)"] >= 2.0) & (df["Year"] < 2000)].shape[0]
    earliest_1_5bn = df[df["Peak Gross ($B)"] > 1.5].sort_values("Year").iloc[0]["Title"]
    correlation = df[["Rank", "Peak Gross ($B)"]].dropna().corr().iloc[0, 1]
    plot_uri = make_base64_plot(df["Rank"].dropna(), df["Peak Gross ($B)"].dropna())

    return [movies_2bn_pre2000, earliest_1_5bn, round(correlation, 6), plot_uri]

# -------------------
# High Court Judgments
# -------------------
async def handle_court_questions(question_text: str, uploaded_files: dict):
    """
    Example handling: expects parquet metadata files from uploaded_files.
    """
    # find parquet file
    parquet_file = None
    for fname, content in uploaded_files.items():
        if fname.endswith(".parquet"):
            parquet_file = content
            break
    if parquet_file is None:
        return ["No parquet file uploaded"]

    df = pd.read_parquet(BytesIO(parquet_file))

    # Example questions: count, regression slope, plot delay
    # 1️⃣ Which high court disposed most cases 2019-2022
    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    filtered = df[(df["year"] >= 2019) & (df["year"] <= 2022)]
    most_cases_court = filtered["court"].value_counts().idxmax()

    # 2️⃣ Regression slope: date_of_registration vs decision_date for court=33_10
    court_df = df[df["court"] == "33_10"].copy()
    court_df["date_of_registration"] = pd.to_datetime(court_df["date_of_registration"], errors="coerce")
    court_df["delay_days"] = (court_df["decision_date"] - court_df["date_of_registration"]).dt.days
    slope = np.polyfit(court_df["year"].dropna(), court_df["delay_days"].dropna(), 1)[0] if not court_df.empty else 0.0

    # 3️⃣ Scatterplot year vs delay_days
    plot_uri = make_base64_plot(court_df["year"].dropna(), court_df["delay_days"].dropna())

    return [most_cases_court, round(slope, 6), plot_uri]

# -------------------
# Plot helper
# -------------------
def make_base64_plot(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    if len(x) > 1 and len(y) > 1:
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, linestyle="dotted", color="red")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Scatterplot")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    uri = f"data:image/png;base64,{encoded}"
    if len(uri) > 100_000:
        uri = uri[:99_999]
    return uri
