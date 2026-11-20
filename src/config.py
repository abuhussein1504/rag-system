from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
 
    APP_NAME: str 
    APP_VERSION: str

    FILE_ALLOWED_TYPES: list[str]
    FILE_MAX_SIZE: int
    DATA_DIR: str


    FILE_DEFAULT_CHUNK_SIZE: int = Field(gt=0)
    FILE_DEFAULT_OVERLAP_SIZE: int = Field(ge=0)


@model_validator(mode="after")
def _check_overlap(self) -> "Settings":
    if self.FILE_DEFAULT_OVERLAP_SIZE >= self.FILE_DEFAULT_CHUNK_SIZE:
        raise ValueError("FILE_DEFAULT_OVERLAP_SIZE must be smaller than chunk size")
    return self

 
@lru_cache
def get_settings() -> Settings:
    return Settings() # type: ignore[call-arg]
