"""Sharing and physical-query layout constants used by the evaluated models."""

MODELS = ("AAAB", "AABC", "ABBC")
Q_HEADS = {"Q1": list(range(1, 16, 2)), "Q2": list(range(0, 16, 2))}
ROUTE_COUNTS = {"AAAB": 2, "AABC": 3, "ABBC": 3}
CONDITIONAL = ("ABAA", "AABA", "AABB", "ABBA")


def require(condition, message):
    if not condition:
        raise ValueError(message)
