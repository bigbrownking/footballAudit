from pathlib import Path

import pandas as pd


def load_players_csv(path: str | Path) -> pd.DataFrame:
    """Load a player-season CSV while preserving empty values as missing data."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {file_path}")

    return pd.read_csv(file_path, encoding="utf-8-sig")
