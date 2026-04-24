"""Train the apartment price model: python train.py --epochs 50"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raise NotImplementedError(f"Training not implemented yet: {args}")


if __name__ == "__main__":
    main()
