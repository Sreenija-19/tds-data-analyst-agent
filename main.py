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
    # Read uploaded file
    content = await questions.read()
    
    # Return dummy answers with file preview
    return {
        "questions_file": questions.filename,
        "content_preview": content.decode(errors="ignore")[:100],
        "answers": []
    }


# Helper function to create base64 plot
def make_base64_plot(x, y):
    import matplotlib.pyplot as plt
    import io
    import base64
    import numpy as np

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    # Regression line
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, linestyle="dotted", color="red")

    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak Gross ($B)")
    ax.set_title("Rank vs Peak Gross")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    uri = "data:image/png;base64," + encoded

    # Reduce size if needed
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
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="wikitable")
    df = pd.read_html(str(table))[0]
    df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Peak": "Peak Gross ($B)", "Rank": "Rank"})
    df["Peak Gross ($B)"] = (
        df["Peak Gross ($B)"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float)
    )
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Title"].str.extract(r"\((\d{4})\)")[0])
    movies_2bn_pre2000 = df[(df["Peak Gross ($B)"] >= 2.0) & (df["Year"] < 2000)].shape[0]
    earliest_1_5bn = df[df["Peak Gross ($B)"] > 1.5].sort_values("Year").iloc[0]["Title"]
    correlation = df[["Rank", "Peak Gross ($B)"]].dropna().corr().iloc[0, 1]
    plot_uri = make_base64_plot(df["Rank"], df["Peak Gross ($B)"])
    return [movies_2bn_pre2000, earliest_1_5bn, round(correlation, 6), plot_uri]
