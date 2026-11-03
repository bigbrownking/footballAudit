from dataclasses import dataclass, field

import pandas as pd

REQUIRED_TEXT_COLUMNS = (
    "player_id",
    "player_name",
    "team",
    "league",
    "season",
    "position",
)
REQUIRED_NUMERIC_COLUMNS = ("age", "minutes")
REQUIRED_COLUMNS = REQUIRED_TEXT_COLUMNS + REQUIRED_NUMERIC_COLUMNS
NON_STAT_COLUMNS = {
    "age",
    "minutes",
    "market_value",
    "estimated_fee",
    "salary",
}


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_players(players: pd.DataFrame) -> ValidationReport:
    """Validate the canonical one-row-per-player/team/season table."""
    report = ValidationReport()
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in players]
    if missing_columns:
        report.issues.append(
            ValidationIssue(
                "error",
                "missing_columns",
                "Missing required columns: " + ", ".join(missing_columns),
            )
        )
        return report

    for column in REQUIRED_TEXT_COLUMNS:
        values = players[column]
        blank = values.isna() | values.astype("string").str.strip().eq("").fillna(False)
        if blank.any():
            report.issues.append(
                ValidationIssue(
                    "error",
                    "blank_identity",
                    f"{column} is empty in {int(blank.sum())} row(s).",
                )
            )

    numeric_columns = [
        column for column in players.columns if column not in REQUIRED_TEXT_COLUMNS
    ]
    numeric_values: dict[str, pd.Series] = {}
    for column in numeric_columns:
        converted = pd.to_numeric(players[column], errors="coerce")
        invalid = players[column].notna() & converted.isna()
        if invalid.any():
            report.issues.append(
                ValidationIssue(
                    "error",
                    "non_numeric_value",
                    f"{column} contains {int(invalid.sum())} non-numeric value(s).",
                )
            )
        numeric_values[column] = converted

    for column in REQUIRED_NUMERIC_COLUMNS:
        missing = numeric_values[column].isna()
        if missing.any():
            report.issues.append(
                ValidationIssue(
                    "warning",
                    "missing_required_measure",
                    f"{column} is missing in {int(missing.sum())} row(s).",
                )
            )

    for column in numeric_columns:
        values = numeric_values[column]
        if column in {"age", "minutes", "market_value", "estimated_fee", "salary"}:
            negative = values.lt(0)
            if negative.any():
                report.issues.append(
                    ValidationIssue(
                        "error",
                        "negative_value",
                        f"{column} is negative in {int(negative.sum())} row(s).",
                    )
                )

    if "age" in numeric_values:
        unusual_age = numeric_values["age"].lt(15) | numeric_values["age"].gt(50)
        if unusual_age.any():
            report.issues.append(
                ValidationIssue(
                    "warning",
                    "unusual_age",
                    f"age is outside 15–50 in {int(unusual_age.sum())} row(s); check the source data.",
                )
            )

    key_columns = ["player_id", "team", "season"]
    duplicate_rows = players.duplicated(subset=key_columns, keep=False)
    if duplicate_rows.any():
        report.issues.append(
            ValidationIssue(
                "error",
                "duplicate_player_team_season",
                f"Found {int(duplicate_rows.sum())} row(s) with duplicate "
                "player_id/team/season keys.",
            )
        )

    return report
