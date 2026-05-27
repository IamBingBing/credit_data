#!/usr/bin/env python3
"""Generate synthetic credit-card-like blockchain transaction data as Parquet.

The amount model follows the simulation appendix of:
"Derivation and utilization of probability distribution of credit card usage
behavior" by Lee and Roh.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


SECONDS_PER_DAY = 24 * 60 * 60


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Parquet transactions with columns: tx_id, from, to, "
            "amount, timestamp."
        )
    )
    parser.add_argument(
        "-w",
        "--wallets",
        type=positive_int,
        required=True,
        help="number of wallets to create; must be at least 2",
    )
    parser.add_argument(
        "-n",
        "--transactions",
        type=non_negative_int,
        required=True,
        help="number of transactions to create",
    )
    parser.add_argument(
        "-d",
        "--date",
        type=parse_date,
        default=dt.date.today(),
        help="single transaction date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("credit_card_transactions.parquet"),
        help="output Parquet path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducible data",
    )
    parser.add_argument(
        "--wallet-prefix",
        default="wallet",
        help="wallet id prefix",
    )
    parser.add_argument(
        "--tx-id-prefix",
        default="tx",
        help="transaction id prefix",
    )
    parser.add_argument(
        "--amount-step",
        type=positive_int,
        default=100,
        help="round amount to this KRW unit",
    )
    parser.add_argument(
        "--min-amount",
        type=positive_int,
        default=100,
        help="minimum transaction amount in KRW",
    )
    parser.add_argument(
        "--max-amount",
        type=non_negative_int,
        default=10_000_000,
        help="maximum transaction amount in KRW; set 0 to disable clipping",
    )
    parser.add_argument(
        "--lambda-shape",
        type=positive_float,
        default=1.5,
        help="Gamma shape for each wallet's usage rate, from the paper",
    )
    parser.add_argument(
        "--lambda-scale",
        type=positive_float,
        default=1.0,
        help="Gamma scale for each wallet's usage rate",
    )
    parser.add_argument(
        "--mu-mean",
        type=float,
        default=9.0,
        help="mean of wallet-level log amount parameter mu",
    )
    parser.add_argument(
        "--mu-sd",
        type=positive_float,
        default=0.6,
        help="standard deviation of wallet-level log amount parameter mu",
    )
    parser.add_argument(
        "--inv-delta-shape",
        type=positive_float,
        default=15.0,
        help="Gamma shape for the inverse variance correction",
    )
    parser.add_argument(
        "--inv-delta-scale",
        type=positive_float,
        default=1.0,
        help="Gamma scale for the inverse variance correction",
    )
    parser.add_argument(
        "--compression",
        choices=("snappy", "gzip", "brotli", "zstd", "none"),
        default="snappy",
        help="Parquet compression codec",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="generate and validate rows without writing Parquet",
    )
    parser.add_argument(
        "--preview",
        type=non_negative_int,
        default=0,
        help="print this many generated rows as JSON after generation",
    )
    return parser.parse_args()


def make_wallet_ids(count: int, prefix: str) -> list[str]:
    width = max(4, len(str(count - 1)))
    return [f"{prefix}_{idx:0{width}d}" for idx in range(count)]


def build_wallet_model(args: argparse.Namespace, rng: random.Random) -> list[dict[str, float]]:
    wallets = []
    for _ in range(args.wallets):
        usage_rate = rng.gammavariate(args.lambda_shape, args.lambda_scale)
        mu = rng.gauss(args.mu_mean, args.mu_sd)
        inv_delta = rng.gammavariate(args.inv_delta_shape, args.inv_delta_scale)
        sigma = math.sqrt(max(mu, 0.000001) / max(inv_delta, 0.000001))
        wallets.append({"usage_rate": max(usage_rate, 0.000001), "mu": mu, "sigma": sigma})
    return wallets


def round_amount(raw_amount: float, step: int, min_amount: int, max_amount: int) -> int:
    amount = int(round(raw_amount / step) * step)
    amount = max(amount, min_amount)
    if max_amount > 0:
        amount = min(amount, max_amount)
    return amount


def random_counterparty(wallet_count: int, payer_idx: int, rng: random.Random) -> int:
    target = rng.randrange(wallet_count - 1)
    if target >= payer_idx:
        target += 1
    return target


def generate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.wallets < 2:
        raise ValueError("--wallets must be at least 2")
    if args.max_amount and args.max_amount < args.min_amount:
        raise ValueError("--max-amount must be 0 or greater than/equal to --min-amount")

    rng = random.Random(args.seed)
    wallet_ids = make_wallet_ids(args.wallets, args.wallet_prefix)
    wallet_model = build_wallet_model(args, rng)
    weights = [wallet["usage_rate"] for wallet in wallet_model]
    base_datetime = dt.datetime.combine(args.date, dt.time())

    rows: list[dict[str, Any]] = []
    payer_indices = range(args.wallets)
    for tx_number in range(args.transactions):
        payer_idx = rng.choices(payer_indices, weights=weights, k=1)[0]
        payee_idx = random_counterparty(args.wallets, payer_idx, rng)
        payer_model = wallet_model[payer_idx]

        raw_amount = math.exp(rng.gauss(payer_model["mu"], payer_model["sigma"]))
        amount = round_amount(raw_amount, args.amount_step, args.min_amount, args.max_amount)
        timestamp = base_datetime + dt.timedelta(seconds=rng.randrange(SECONDS_PER_DAY))

        rows.append(
            {
                "tx_id": f"{args.tx_id_prefix}_{tx_number:012d}",
                "from": wallet_ids[payer_idx],
                "to": wallet_ids[payee_idx],
                "amount": amount,
                "timestamp": timestamp,
            }
        )

    rows.sort(key=lambda row: (row["timestamp"], row["tx_id"]))
    return rows


def write_parquet(rows: list[dict[str, Any]], output: Path, compression: str) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow. Install it with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    schema = pa.schema(
        [
            ("tx_id", pa.string()),
            ("from", pa.string()),
            ("to", pa.string()),
            ("amount", pa.int64()),
            ("timestamp", pa.timestamp("s")),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    codec = None if compression == "none" else compression
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output, compression=codec)


def main() -> int:
    args = parse_args()
    try:
        rows = generate_rows(args)
        if not args.dry_run:
            write_parquet(rows, args.output, args.compression)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.preview:
        preview_rows = rows[: args.preview]
        print(
            json.dumps(
                preview_rows,
                ensure_ascii=False,
                default=lambda value: value.isoformat(sep=" "),
                indent=2,
            )
        )

    action = "validated" if args.dry_run else "wrote"
    print(
        f"{action} {len(rows)} transactions for {args.wallets} wallets "
        f"on {args.date.isoformat()}"
        + ("" if args.dry_run else f" -> {args.output}")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
