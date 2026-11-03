import pandas as pd

from src.validation import validate_players


def valid_players() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_name": "Player One",
                "team": "Club A",
                "league": "League A",
                "season": "2024",
                "position": "FW",
                "age": 24,
                "minutes": 1800,
                "goals": 10,
            }
        ]
    )


def test_valid_player_data_passes() -> None:
    assert validate_players(valid_players()).is_valid


def test_missing_column_is_reported() -> None:
    players = valid_players().drop(columns="position")

    report = validate_players(players)

    assert not report.is_valid
    assert report.errors[0].code == "missing_columns"


def test_duplicate_player_team_season_is_reported() -> None:
    players = pd.concat([valid_players(), valid_players()], ignore_index=True)

    report = validate_players(players)

    assert not report.is_valid
    assert any(issue.code == "duplicate_player_team_season" for issue in report.errors)


def test_missing_minutes_warns_without_discarding_record() -> None:
    players = valid_players()
    players.loc[0, "minutes"] = None

    report = validate_players(players)

    assert report.is_valid
    assert any(issue.code == "missing_required_measure" for issue in report.warnings)
