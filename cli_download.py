import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from astroflow.gaia_download import download_join_chunked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Gaia DR3 rows by source_id using AIP TAP + SJS."
    )

    parser.add_argument(
        "--token",
        default=os.getenv("GAIA_AIP_TOKEN"),
        help="Authorization token. If omitted, uses GAIA_AIP_TOKEN env var.",
    )

    parser.add_argument(
        "--join-table",
        default="gaiadr3.xp_continuous_mean_spectrum",
        help="Join table to download (e.g. gaiadr3.xp_continuous_mean_spectrum).",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="How many IDs per request chunk.",
    )

    parser.add_argument(
        "--out-dir",
        default="out_gaia_downloads",
        help="Output directory for downloaded files.",
    )

    parser.add_argument(
        "--save-parquet",
        action="store_true",
        help="Save each chunk as a parquet file.",
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--ids",
        nargs="+",
        help="One or more Gaia source_id values (space-separated).",
    )

    group.add_argument(
        "--csv",
        help="Path to CSV file containing a 'source_id' column.",
    )

    parser.add_argument(
        "--csv-col",
        default="source_id",
        help="Column name in CSV that contains source IDs (default: source_id).",
    )

    parser.add_argument(
        "--csv-out",
        default=None,
        help="If set, save final combined result to this CSV path.",
    )

    return parser.parse_args()


def load_ids_from_csv(path: str, col: str) -> list[int]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"CSV is missing column '{col}'. Columns: {list(df.columns)}")
    return [int(x) for x in df[col].dropna().astype("int64").tolist()]


def main() -> None:
    args = parse_args()

    if not args.token:
        print("Missing token. Set GAIA_AIP_TOKEN or pass --token.", file=sys.stderr)
        sys.exit(1)

    if args.ids:
        source_ids = [int(x) for x in args.ids]
    else:
        source_ids = load_ids_from_csv(args.csv, args.csv_col)

    df = download_join_chunked(
        source_ids=source_ids,
        join_table=args.join_table,
        token=args.token,
        chunk_size=args.chunk_size,
        out_dir=Path(args.out_dir),
        save_chunks_parquet=args.save_parquet,
    )

    print(f"Rows: {len(df)}")
    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv_out, index=False)
        print(f"Saved: {args.csv_out}")


if __name__ == "__main__":
    main()