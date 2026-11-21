from functools import lru_cache

from pydantic import Field, model_validator
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.models.enums import DistanceMethod, EmbeddingBackend, VectorDBBackend


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


    # EMBEDDING
    EMBEDDING_BACKEND: EmbeddingBackend
    EMBEDDING_MODEL_ID: str
    EMBEDDING_MODEL_SIZE: int
    GOOGLE_API_KEY: SecretStr | None = None

    # VECTOR
    VECTOR_DB_BACKEND: VectorDBBackend
    VECTOR_DB_PATH: str
    VECTOR_DB_DISTANCE_METHOD: DistanceMethod = DistanceMethod.COSINE


@model_validator(mode="after")
def _check_overlap(self) -> "Settings":
    if self.FILE_DEFAULT_OVERLAP_SIZE >= self.FILE_DEFAULT_CHUNK_SIZE:
        raise ValueError("FILE_DEFAULT_OVERLAP_SIZE must be smaller than chunk size")
    return self

 
@lru_cache
def get_settings() -> Settings:
    return Settings() # type: ignore[call-arg]
