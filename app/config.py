import os


def _get_positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return default


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///data/elite_verity.db",
)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://", "postgresql+psycopg://", 1
    )
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )


REDIS_URL = os.getenv("REDIS_URL")
CACHE_TTL_SECONDS = _get_positive_int("CACHE_TTL_SECONDS", 300)
CHAT_RATE_LIMIT = _get_positive_int("CHAT_RATE_LIMIT", 30)
CHAT_RATE_LIMIT_WINDOW_SECONDS = _get_positive_int(
    "CHAT_RATE_LIMIT_WINDOW_SECONDS", 60
)
