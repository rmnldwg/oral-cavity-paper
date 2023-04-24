"""
Some shared params, functions, and classes.
"""
from pathlib import Path

DATAFILE = Path("./data/enhanced.csv")
FIGURES_DIR = Path("./figures")
TABLES_DIR = Path("./tables")

MPLSTYLE = Path("./scripts/.mplstyle")
COLORS = {
    "green": '#00afa5',
    "red": '#ae0060',
    "blue": '#005ea8',
    "orange": '#f17900',
    "gray": '#c5d5db',
}

ORAL_CAVITY_ICD_CODES = {
    "tongue": ["C02", "C02.0", "C02.1", "C02.2", "C02.3", "C02.4", "C02.8", "C02.9",],
    "gums and cheeks": [
        "C03", "C03.0", "C03.1", "C03.9", "C06", "C06.0", "C06.1", "C06.2", "C06.8",
        "C06.9",
    ],
    "floor of mouth": ["C04", "C04.0", "C04.1", "C04.8", "C04.9",],
    # "palate": ["C05", "C05.0", "C05.1", "C05.2", "C05.8", "C05.9",],
    # "salivary glands": ["C08", "C08.0", "C08.1", "C08.9",],
}
