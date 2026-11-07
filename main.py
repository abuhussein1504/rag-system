from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.api.routes.base import base_router
from src.utils.config import get_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    settings = get_settings()

    yield
    # --- SHUTDOWN ---


app = FastAPI(lifespan=lifespan)

app.include_router(base_router)
