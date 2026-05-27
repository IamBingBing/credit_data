# credit_data

Credit-card-like Korean won transaction data generator.

## Generate Parquet

Install the Parquet writer dependency:

```bash
python3 -m pip install -r requirements.txt
```

Create a dataset:

```bash
./generate_credit_card_parquet.py \
  --wallets 100 \
  --transactions 10000 \
  --date 2026-05-26 \
  --output data/transactions.parquet \
  --seed 42
```

The output columns are:

```text
tx_id, from, to, amount, timestamp
```

The transaction amount model follows the simulation appendix of Lee and Roh's
credit card usage behavior paper: wallet-level usage rate is sampled from a
Gamma distribution, and transaction amounts are sampled from a corrected
log-normal model.
