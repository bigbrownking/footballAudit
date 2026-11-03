import argparse
from pathlib import Path
from urllib.request import urlretrieve

from src.top5_source import (
    SOURCE_URL,
    load_source_csv,
    prepare_player_financials,
    prepare_player_stats,
    prepare_scouting_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the pinned top-five-leagues dataset and prepare project tables."
    )
    parser.add_argument(
        "--source",
        help="Path to an existing master CSV; if omitted, download the pinned GitHub revision.",
    )
    parser.add_argument(
        "--raw-output",
        default="data/raw/player_seasons.csv",
        help="Path for the source CSV when downloading.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for generated CSV tables.",
    )
    args = parser.parse_args()

    source_path = Path(args.source) if args.source else Path(args.raw_output)
    if not args.source:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(SOURCE_URL, source_path)
        print(f"Downloaded pinned source CSV: {source_path}")

    source = load_source_csv(source_path)
    stats = prepare_player_stats(source)
    financials = prepare_player_financials(source)
    scouting = prepare_scouting_dataset(stats)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "player_stats.csv": stats,
        "player_stats_2024_25.csv": stats.loc[stats["season"].eq("2024/25")],
        "player_stats_2025_26.csv": stats.loc[stats["season"].eq("2025/26")],
        "player_financials.csv": financials,
        "scouting_dataset.csv": scouting,
    }
    for filename, frame in outputs.items():
        output_path = output_dir / filename
        frame.to_csv(output_path, index=False)
        print(f"Wrote {len(frame)} row(s): {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
