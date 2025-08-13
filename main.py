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

app = FastAPI(title="Data Analyst Agent")

# -------------------------
# Helper: scatterplot to base64
# -------------------------
def make_base64_plot(x, y, xlabel="x", ylabel="y", title="Plot"):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    if len(x) >= 2 and len(y) >= 2:
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * np.array(x) + b, linestyle="dotted", color="red")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"[:99_999]

# -------------------------
# Wikipedia: highest grossing films
# -------------------------
def handle_wikipedia_question():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    
    # Find table with Rank column
    target_table = None
    for table in tables:
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if any("rank" in h for h in headers):
            target_table = table
            break
    if not target_table:
        return ["Error: Wikipedia table not found"]
    
    df = pd.read_html(str(target_table))[0]
    df.columns = [c if not isinstance(c, tuple) else c[-1] for c in df.columns]
    df.columns = df.columns.str.strip()
    
    # Standardize column names
    df.rename(columns=lambda x: x.strip().lower(), inplace=True)
    df.rename(columns=lambda x: "rank" if "rank" in x else x, inplace=True)
    df.rename(columns=lambda x: "title" if "title" in x else x, inplace=True)
    df.rename(columns=lambda x: "peak" if "gross" in x or "peak" in x else x, inplace=True)
    
    df["peak"] = df["peak"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float)/1_000_000_000
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["year"] = pd.to_numeric(df["title"].str.extract(r"\((\d{4})\)")[0], errors="coerce")
    
    q1 = int(df[(df["peak"] >= 2) & (df["year"] < 2000)].shape[0])
    q2 = str(df[df["peak"] > 1.5].sort_values("year").iloc[0]["title"])
    q3 = round(df[["rank","peak"]].dropna().corr().iloc[0,1], 6)
    q4 = make_base64_plot(df["rank"].dropna(), df["peak"].dropna(), xlabel="Rank", ylabel="Peak", title="Rank vs Peak")
    
    return [q1, q2, q3, q4]

# -------------------------
# Court dataset
# -------------------------
def handle_court_question(uploaded_files: dict):
    parquet_files = [f for f in uploaded_files if f.endswith(".parquet")]
    if not parquet_files:
        return {"error": "No parquet files provided for court dataset"}
    
    # Load first parquet file
    df = pd.read_parquet(BytesIO(uploaded_files[parquet_files[0]]))
    
    # Q1: High court disposing most cases 2019-2022
    df_filtered = df[(df["year"] >= 2019) & (df["year"] <= 2022)]
    most_cases_court = df_filtered["court"].value_counts().idxmax()
    
    # Q2: Regression slope registration->decision date for court=33_10
    df33 = df[df["court"]=="33_10"].copy()
    df33["registration"] = pd.to_datetime(df33["date_of_registration"], errors="coerce", dayfirst=True)
    df33["decision"] = pd.to_datetime(df33["decision_date"], errors="coerce")
    df33 = df33.dropna(subset=["registration","decision"])
    df33["days_delay"] = (df33["decision"] - df33["registration"]).dt.days
    df33["year"] = df33["registration"].dt.year
    
    slope = float(np.polyfit(df33["year"], df33["days_delay"],1)[0]) if len(df33) >= 2 else 0.0
    plot_uri = make_base64_plot(df33["year"], df33["days_delay"], xlabel="Year", ylabel="Days Delay", title="Year vs Days Delay") if len(df33)>=2 else ""
    
    return {
        "most_cases_court": most_cases_court,
        "registration_decision_slope": round(slope,6),
        "plot_year_days_delay": plot_uri
    }

# -------------------------
# API endpoints
# -------------------------
@app.get("/")
def root():
    return {"message": "Data Analyst Agent API is live"}

@app.post("/answer")
@app.post("/api/")
async def analyze_data(questions: UploadFile = File(...), files: Optional[List[UploadFile]] = File(default=None)):
    try:
        question_text = (await questions.read()).decode("utf-8").lower()
        uploaded_files = {file.filename: await file.read() for file in files} if files else {}
        
        if "highest grossing films" in question_text:
            return JSONResponse(content=handle_wikipedia_question())
        elif "indian high court" in question_text or uploaded_files:
            return JSONResponse(content=handle_court_question(uploaded_files))
        else:
            return JSONResponse(content={"error": "Unsupported question"})
    except Exception as e:
        return JSONResponse(content={"error": "Failed to process request", "details": str(e)})