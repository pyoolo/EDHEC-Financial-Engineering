"""Step 1. Build the daily panel, targets and features; save descriptive statistics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fuelguard import config as C  # noqa: E402
from fuelguard.data import descriptive_table  # noqa: E402
from fuelguard.dataset import build_dataset  # noqa: E402
from fuelguard.features import GROUPS  # noqa: E402

if __name__ == "__main__":
    C.TABLES.mkdir(parents=True, exist_ok=True)
    panel, data, latest, feats = build_dataset()
    desc = descriptive_table(panel)
    desc.to_csv(C.TABLES / "descriptive_raw.csv")
    tgt = data[["_premium", "_spike", "_market_cost", "_lock_price"]].describe().T
    tgt.to_csv(C.TABLES / "descriptive_targets.csv")
    data["_spike"].groupby(data.index.year).agg(["mean", "size"]).to_csv(C.TABLES / "spike_rate_by_year.csv")
    print(f"Raw series: {desc.shape[0] - 2} (+ diesel in EUR)")
    print(f"Feature groups: {', '.join(f'{k} ({len(v)})' for k, v in GROUPS.items())}")
    print(f"Model dataset: {data.shape[0]} days x {len(feats)} features, "
          f"{data.index.min().date()} -> {data.index.max().date()}; spike rate {data['_spike'].mean():.3f}")
