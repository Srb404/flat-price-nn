"""Train the apartment price model and write metrics and figures to reports/."""

import argparse
import json
from pathlib import Path

import numpy as np

from flat_price.baselines import CityPricePerM2, LinearRegression
from flat_price.data import load_listings, split
from flat_price.features import FeatureEncoder
from flat_price.metrics import regression_report
from flat_price.plots import make_figures
from flat_price.predictor import PricePredictor
from nn import MSE, Adam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("models/flat-price"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()

    train, val, test = split(load_listings(), seed=args.seed)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")

    encoder = FeatureEncoder().fit(train)
    x_train, x_val = encoder.transform(train), encoder.transform(val)

    predictor = PricePredictor(encoder, args.hidden, args.dropout, seed=args.seed)
    predictor.fit_target(train["price"].to_numpy())
    history = predictor.network.fit(
        x_train,
        predictor.encode_target(train["price"].to_numpy()),
        loss=MSE(),
        optimizer=Adam(lr=args.lr, weight_decay=args.weight_decay),
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation=(x_val, predictor.encode_target(val["price"].to_numpy())),
        patience=args.patience,
        rng=np.random.default_rng(args.seed),
    )
    predictor.save(args.out)

    y_test = test["price"].to_numpy()
    x_test = encoder.transform(test)
    predictions = {
        "city median price per m2": CityPricePerM2().fit(train).predict(test),
        "linear regression": LinearRegression()
        .fit(x_train, train["price"].to_numpy())
        .predict(x_test),
        "neural network": predictor.predict(test),
    }
    results = {name: regression_report(y_test, pred) for name, pred in predictions.items()}
    print(f"\n{'model':<26}" + "".join(f"{k:>13}" for k in results["neural network"]))
    for name, report in results.items():
        print(f"{name:<26}" + "".join(f"{v:>13,.2f}" for v in report.values()))

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "metrics.json").write_text(json.dumps(results, indent=2))
    (args.reports / "history.json").write_text(json.dumps(history))
    test[["city", "squareMeters", "price"]].assign(
        predicted=predictions["neural network"].round(),
        linear=predictions["linear regression"].round(),
    ).to_csv(args.reports / "test_predictions.csv", index=False)
    make_figures(args.reports)


if __name__ == "__main__":
    main()
