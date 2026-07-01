from fastapi import FastAPI

app = FastAPI(title="MLOps Final Project API")


@app.get("/health")
def health():
    """Health check endpoint — used to verify the API is up and running."""
    return {"status": "ok"}
