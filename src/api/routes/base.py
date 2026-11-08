from typing import Annotated

from fastapi import APIRouter, Depends

from src.config import Settings, get_settings

base_router = APIRouter(prefix="/api/info", tags=["base"])

SettingsDep = Annotated[Settings, Depends(get_settings)]

@base_router.get("/info")
def info(settings: SettingsDep):
    return {"name": settings.APP_NAME, "version": settings.APP_VERSION}

