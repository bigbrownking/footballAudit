import argparse
from pathlib import Path

from src.data_loader import load_players_csv
from src.preprocessing import prepare_players
from src.validation import validate_players


def _format_issues(issues: list) -> str:
    return "\n".join(f"{issue.level.upper()} [{issue.code}]: {issue.message}" for issue in issues)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate player-season data and calculate per-90 statistics."
    )
    parser.add_argument("input", help="Path to a CSV using the project data contract.")
    parser.add_argument(
        "--output",
        default="data/processed/players.csv",
        help="Output CSV path (default: data/processed/players.csv).",
    )
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=450,
        help="Minimum minutes for per-90 eligibility (default: 450).",
    )
    args = parser.parse_args()

    players = load_players_csv(args.input)
    report = validate_players(players)
    if report.issues:
        print(_format_issues(report.issues))
    if not report.is_valid:
        return 1

    prepared = prepare_players(players, min_minutes=args.min_minutes)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    print(f"Prepared {len(prepared)} row(s): {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
