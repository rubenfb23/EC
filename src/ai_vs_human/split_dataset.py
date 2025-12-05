"""Utility to split the merged AI/Human dataset into reproducible train/test CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# Preferred defaults if no input file is provided explicitly.
DEFAULT_CANDIDATES = (
    "merged_ai_human_multisocial_features_cleaned.csv",
    "merged_ai_human_multisocial_features.csv",
    "ai_human_content_detection_dataset.csv",
)


def find_input_path(explicit_path: Optional[str]) -> Path:
    """Resolve the dataset path, preferring the merged file if present."""
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Input file not found: {candidate}")
        return candidate

    base_dir = Path(__file__).resolve().parent
    for filename in DEFAULT_CANDIDATES:
        candidate = base_dir / filename
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No dataset found. Place merged_ai_human_multisocial_features.csv or "
        "ai_human_content_detection_dataset.csv next to this script or pass --input."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the AI/Human dataset into train and test CSV files "
            "(90/10 by default, stratified by the label column)."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to the input CSV. Defaults to merged_ai_human_multisocial_features.csv if present.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to write the splits. Defaults to the input file directory.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of rows to allocate to the test split (0 < test_size < 1). Default: 0.1",
    )
    parser.add_argument(
        "--stratify-column",
        default="label",
        help="Column to use for stratified splitting. Default: label",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--dataset-test",
        help=(
            "Optional dataset flag suffix to use as test split. "
            "For example, 'multisocial' will use rows where ds_multisocial is True "
            "as the test set and the remaining rows as train."
        ),
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified splitting even if the column exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset_test is None and not 0 < args.test_size < 1:
        raise ValueError(
            f"test_size must be between 0 and 1. Received: {args.test_size}"
        )

    input_path = find_input_path(args.input)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if args.dataset_test:
        ds_col = f"ds_{args.dataset_test}"
        if ds_col not in df.columns:
            raise ValueError(
                f"--dataset-test was set to '{args.dataset_test}', "
                f"but column `{ds_col}` was not found in the input data."
            )
        test_mask = df[ds_col].astype(bool)
        test_df = df[test_mask]
        train_df = df[~test_mask]
        if test_df.empty or train_df.empty:
            raise ValueError(
                f"Dataset split using `{ds_col}` produced an empty "
                f"{'test' if test_df.empty else 'train'} set."
            )
        print(
            f"Using dataset-based split with `{ds_col}`: "
            f"train={len(train_df)}, test={len(test_df)}"
        )
    else:
        stratify_series = None
        if not args.no_stratify:
            if args.stratify_column in df.columns:
                unique_vals = df[args.stratify_column].dropna().unique()
                if len(unique_vals) > 1:
                    stratify_series = df[args.stratify_column]
                else:
                    print(
                        f"Stratify column `{args.stratify_column}` has a single class; "
                        "falling back to random split."
                    )
            else:
                print(
                    f"Stratify column `{args.stratify_column}` not found; "
                    "falling back to random split."
                )

        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify_series,
        )

    suffix = input_path.suffix or ".csv"
    train_path = output_dir / f"{input_path.stem}_train{suffix}"
    test_path = output_dir / f"{input_path.stem}_test{suffix}"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Input file: {input_path}")
    print(f"Rows total: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train split saved to: {train_path}")
    print(f"Test split saved to:  {test_path}")


if __name__ == "__main__":
    main()
