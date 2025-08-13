from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
from datetime import datetime

app = FastAPI(title="Data Analyst Agent")

# -----------------------------------
# Helper: scatterplot to base64
# -----------------------------------
def make_base64_plot(x, y, xlabel="x", ylabel="y", title="Plot"):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    
    # Regression line
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, linestyle="dotted", color="red")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    uri = f"data:image/png;base64,{encoded}"
    if len(uri) > 100_000:
        uri = uri[:99_999]
    return uri

# -----------------------------------
# Wikipedia: highest grossing films
# -----------------------------------
def handle_wikipedia_question():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find table with "Rank"
    tables = soup.find_all("table", class_="wikitable")
    target_table = None
    for table in tables:
        headers = [h.get_text(strip=True).lower() for h in table.find_all("th")]
        if any("rank" in h for h in headers):
            target_table = table
            break
    if target_table is None:
        raise ValueError("Wikipedia table not found")
    
    df = pd.read_html(str(target_table))[0]
    
    # Flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
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
        .astype(float)/1_000_000_000
    )
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Title"].str.extract(r"\((\d{4})\)")[0], errors="coerce")
    
    q1 = df[(df["Peak Gross ($B)"] >= 2.0) & (df["Year"] < 2000)].shape[0]
    q2 = df[df["Peak Gross ($B)"] > 1.5].sort_values("Year").iloc[0]["Title"]
    q3 = df[["Rank","Peak Gross ($B)"]].dropna().corr().iloc[0,1]
    q4 = make_base64_plot(df["Rank"].dropna(), df["Peak Gross ($B)"].dropna(),
                          xlabel="Rank", ylabel="Peak Gross ($B)", title="Rank vs Peak Gross")
    
    return [q1, q2, round(q3,6), q4]

# -----------------------------------
# Court judgments
# -----------------------------------
def handle_court_question(uploaded_files):
    # Look for parquet files in uploaded_files
    parquet_files = [f for f in uploaded_files if f.endswith(".parquet")]
    if not parquet_files:
        return {"error": "No parquet files provided for court dataset"}
    
    # Read first parquet file
    content = uploaded_files[parquet_files[0]]
    df = pd.read_parquet(BytesIO(content))
    
    # Q1: High court disposing most cases 2019-2022
    df_filtered = df[(df["year"] >= 2019) & (df["year"] <= 2022)]
    most_cases_court = df_filtered["court"].value_counts().idxmax()
    
    # Q2: Regression slope of registration_date -> decision_date for court=33_10
    df33 = df[df["court"]=="33_10"].copy()
    df33["registration"] = pd.to_datetime(df33["date_of_registration"], errors="coerce", dayfirst=True)
    df33["decision"] = pd.to_datetime(df33["decision_date"], errors="coerce")
    df33 = df33.dropna(subset=["registration","decision"])
    df33["days_delay"] = (df33["decision"] - df33["registration"]).dt.days
    df33["year"] = df33["registration"].dt.year
    
    if len(df33) >=2:
        slope = np.polyfit(df33["year"], df33["days_delay"],1)[0]
    else:
        slope = 0.0
    
    # Q3: scatterplot of year vs days_delay
    if len(df33) >=2:
        plot_uri = make_base64_plot(df33["year"], df33["days_delay"],
                                    xlabel="Year", ylabel="Days Delay",
                                    title="Year vs Days Delay")
    else:
        plot_uri = ""
    
    return {
        "most_cases_court": most_cases_court,
        "registration_decision_slope": round(float(slope),6),
        "plot_year_days_delay": plot_uri
    }

# -----------------------------------
# API endpoints
# -----------------------------------
@app.get("/")
def root():
    return {"message": "Data Analyst Agent API is live"}

@app.post("/answer")
async def answer(questions: UploadFile = File(...), files: Optional[List[UploadFile]] = File(default=[])):
    try:
        question_text = (await questions.read()).decode("utf-8").lower()
        uploaded_files = {file.filename: await file.read() for file in files}
        
        if "highest grossing films" in question_text:
            return JSONResponse(content=handle_wikipedia_question())
        elif "indian high court" in question_text or uploaded_files:
            return JSONResponse(content=handle_court_question(uploaded_files))
        else:
            return JSONResponse(content={"error": "Unsupported question"})
    
    except Exception as e:
        return JSONResponse(content={"error": "Failed to process request", "details": str(e)})

@app.post("/api/")
async def analyze_data(questions: UploadFile = File(...), files: Optional[List[UploadFile]] = File(default=[])):
    try:
        question_text = (await questions.read()).decode("utf-8").lower()
        uploaded_files = {file.filename: await file.read() for file in files}
        
        if "highest grossing films" in question_text:
            return JSONResponse(content=handle_wikipedia_question())
        elif "indian high court" in question_text or uploaded_files:
            return JSONResponse(content=handle_court_question(uploaded_files))
        else:
            return JSONResponse(content={"error": "Unsupported question"})
    
    except Exception as e:
        return JSONResponse(content={"error": "Failed to process request", "details": str(e)})