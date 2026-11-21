import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.base import base_router
from src.api.routes.data import data_router
from src.api.routes.search import search_router
from src.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    settings = get_settings()

    yield
    # --- SHUTDOWN ---


app = FastAPI(lifespan=lifespan)

app.include_router(base_router)
app.include_router(data_router)
app.include_router(search_router)
