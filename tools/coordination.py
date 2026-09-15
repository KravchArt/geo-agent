from enum import StrEnum


class ProviderExecutionStrategy(StrEnum):
    FIRST = "first"
    FALLBACK = "fallback"
    PARALLEL = "parallel"
