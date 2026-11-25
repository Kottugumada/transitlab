from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

_STATIC = Path(__file__).parent.parent / "static"

app = FastAPI(title="TransitLab", version="0.1.0")

app.include_router(router, prefix="/api/v1")


@app.get("/candidates", response_class=HTMLResponse, include_in_schema=False)
async def candidates_page():
    return (_STATIC / "candidates.html").read_text()


@app.get("/vetting", response_class=HTMLResponse, include_in_schema=False)
async def vetting_page():
    return (_STATIC / "vetting.html").read_text()


@app.get("/explorer", response_class=HTMLResponse, include_in_schema=False)
async def explorer_page():
    return (_STATIC / "explorer.html").read_text()


app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")
