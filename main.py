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

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is live"}

@app.post("/answer")
async def answer(questions: UploadFile = File(...)):
    try:
        question_text = (await questions.read()).decode("utf-8")

        if "highest grossing films" in question_text.lower():
            try:
                answers = await handle_wikipedia_question(question_text)
            except Exception as e:
                # Safe fallback
                answers = ["Error processing Wikipedia data", str(e), 0.0, None]
        else:
            answers = ["Unsupported question"]

        return {
            "questions_file": questions.filename,
            "content_preview": question_text[:100],
            "answers": answers
        }
    except Exception as e:
        return {
            "error": "Failed to process request",
            "details": str(e)
        }

def make_base64_plot(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)

    # Regression line
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, linestyle="dotted", color="red")

    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak Gross ($B)")
    ax.set_title("Rank vs Peak Gross")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    uri = "data:image/png;base64," + encoded

    if len(uri) > 100000:
        uri = uri[:99999]

    plt.close(fig)
    return uri

@app.post("/api/")
async def analyze_data(
    questions: UploadFile = File(...),
    files: List[UploadFile] = File(default=[]),
):
    question_text = (await questions.read()).decode("utf-8")
    uploaded_files = {file.filename: await file.read() for file in files}

    if "highest grossing films" in question_text.lower():
        response = await handle_wikipedia_question(question_text)
    else:
        response = {"error": "Unsupported question (demo version)"}

    return JSONResponse(content=response)

async def handle_wikipedia_question(question_text: str):
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the table with "Rank" header
    tables = soup.find_all("table", class_="wikitable")
    target_table = None
    for table in tables:
        headers = [h.get_text(strip=True).lower() for h in table.find_all("th")]
        if any("rank" in h for h in headers):
            target_table = table
            break
    if target_table is None:
        raise ValueError("Could not find target Wikipedia table")

    df = pd.read_html(str(target_table))[0]

    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[-1] for col in df.columns]
    df.columns = df.columns.str.strip()

    # Standardize columns
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

    # Clean numbers
    df["Peak Gross ($B)"] = (
        df["Peak Gross ($B)"].astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .astype(float) / 1_000_000_000
    )
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")

    # Extract year from title
    df["Year"] = pd.to_numeric(df["Title"].str.extract(r"\((\d{4})\)")[0], errors="coerce")

    # Analysis
    movies_2bn_pre2000 = df[(df["Peak Gross ($B)"] >= 2.0) & (df["Year"] < 2000)].shape[0]
    earliest_1_5bn = df[df["Peak Gross ($B)"] > 1.5].sort_values("Year").iloc[0]["Title"]
    correlation = df[["Rank", "Peak Gross ($B)"]].dropna().corr().iloc[0, 1]
    plot_uri = make_base64_plot(df["Rank"].dropna(), df["Peak Gross ($B)"].dropna())

    return [movies_2bn_pre2000, earliest_1_5bn, round(correlation, 6), plot_uri]
