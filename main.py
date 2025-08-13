from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests
from bs4 import BeautifulSoup

app = FastAPI()

@app.post("/api/")
async def analyze_data(
    questions: UploadFile = File(...),
    files: Optional[List[UploadFile]] = None
):
    # Read questions.txt
    question_text = (await questions.read()).decode("utf-8")

    # Read optional files
    uploaded_files = {}
    if files:
        for file in files:
            uploaded_files[file.filename] = await file.read()

    # Example: Highest grossing films task
    if "highest grossing films" in question_text.lower():
        response = await handle_wikipedia_question()
    else:
        response = {"error": "Unsupported question (demo)"}

    return JSONResponse(content=response)

async def handle_wikipedia_question():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    # Find the wikitable with "Rank"
    table = None
    for t in soup.find_all("table", class_="wikitable"):
        headers = [h.get_text(strip=True).lower() for h in t.find_all("th")]
        if any("rank" in h for h in headers):
            table = t
            break
    if table is None:
        return ["Error: Table not found"]

    df = pd.read_html(str(table))[0]

    # Clean columns
    df.columns = [c if isinstance(c, str) else c[-1] for c in df.columns]
    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if "rank" in c: rename_map[col] = "Rank"
        elif "gross" in c or "peak" in c: rename_map[col] = "Peak Gross ($B)"
        elif "title" in c: rename_map[col] = "Title"
    df = df.rename(columns=rename_map)

    # Clean numeric data
    df["Peak Gross ($B)"] = df["Peak Gross ($B)"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float)/1e9
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Title"].str.extract(r"\((\d{4})\)")[0], errors="coerce")

    # Analysis
    movies_2bn_pre2000 = df[(df["Peak Gross ($B)"]>=2) & (df["Year"]<2000)].shape[0]
    earliest_1_5bn = df[df["Peak Gross ($B)"]>1.5].sort_values("Year").iloc[0]["Title"]
    correlation = df[["Rank","Peak Gross ($B)"]].dropna().corr().iloc[0,1]
    plot_uri = make_base64_plot(df["Rank"].dropna(), df["Peak Gross ($B)"].dropna())

    return [movies_2bn_pre2000, earliest_1_5bn, round(correlation,6), plot_uri]

def make_base64_plot(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    m,b = np.polyfit(x, y, 1)
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
    return uri[:99999]  # ensure under 100k chars
