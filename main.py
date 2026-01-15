from fastapi import FastAPI, Query
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os  # Import os to access environment variables

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the CSV data
df = pd.read_csv("jobs_big.csv")
texts = (df["title"] + " " + df["description"]).tolist()

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000
)

vectors = vectorizer.fit_transform(texts).toarray().astype("float32")

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

@app.get("/")
def get_jobs():
    return { "message" : "Hello from search service" }

@app.get("/getjobs")
def get_jobs(query: str | None = Query(None), location: str | None = None, company: str | None = None, top_k: int = Query(20, le=50)):

    if not query:
        # Return first top_k jobs if query is missing
        results = []
        for _, job in df.head(top_k).iterrows():
            results.append({
                "id": int(job["id"]),
                "title": job["title"],
                "description": job["description"],
                "location": job["location"],
                "company": job["company"],
                "url": job["url"]
            })
        return {
            "query": None,
            "count": len(results),
            "results": results
        }

    # Original TF-IDF + FAISS search
    query_vec = vectorizer.transform([query]).toarray().astype("float32")
    D, I = index.search(query_vec, top_k * 3)  # extra for filtering

    results = []
    for idx in I[0]:
        job = df.iloc[idx]
        if location and job["location"] != location:
            continue
        if company and job["company"] != company:
            continue

        results.append({
            "id": int(job["id"]),
            "title": job["title"],
            "description": job["description"],
            "location": job["location"],
            "company": job["company"],
            "url": job["url"]
        })
        if len(results) == top_k:
            break

    return {
        "query": query,
        "filters": {
            "location": location,
            "company": company
        },
        "count": len(results),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable for the port
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT isn't set
    uvicorn.run(app, host="0.0.0.0", port=port)
