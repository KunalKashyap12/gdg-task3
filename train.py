"""
Training entrypoint for News Classification System.

Usage:
    python train.py                     # Full training on all 120,000 articles
    python train.py --sample_size 10000 # Fast iteration benchmark
    python train.py --advanced_clean    # Train with lemmatization & stopword removal
"""

import sys
import argparse
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train, benchmark, and serialize the News Classification model."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional number of samples for fast iteration (default: all data)",
    )
    parser.add_argument(
        "--advanced_clean",
        action="store_true",
        help="Enable advanced lemmatization and stopword removal",
    )
    args = parser.parse_args()

    run_pipeline(
        sample_size=args.sample_size,
        use_advanced_clean=args.advanced_clean
    )


if __name__ == "__main__":
    main()
