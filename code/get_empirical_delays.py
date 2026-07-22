from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Iterable

import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

try:
    import pyreadr  # for .rds/.rda
except ImportError:
    pyreadr = None


# ============================================================
# Helpers: aggregation -> linelist, parsing, cleaning
# ============================================================

def cumulative_aggregated_to_linelist(
    df: pd.DataFrame,
    *,
    start_col: str,
    end_col: str,
    n_col: str = "n_samples",
) -> pd.DataFrame:
    """
    Interprets df as a cumulative distribution of delays for each start date:
      - grouped by start_col
      - sorted by end_col
      - n_col is cumulative counts up to that end_col
    Produces a linelist with one row per implied individual.
    """
    df = df.copy()
    df = df.sort_values([start_col, end_col])

    rows = []
    for start, g in df.groupby(start_col):
        g = g.sort_values(end_col)
        prev_n = 0
        for _, row in g.iterrows():
            curr_n = int(row[n_col])
            delta = curr_n - prev_n
            delay = (row[end_col] - start).days

            if delta > 0:
                for _ in range(delta):
                    rows.append({start_col: start, end_col: row[end_col], "delay": delay})
            elif delta < 0:
                remove = -delta
                for i in range(len(rows) - 1, -1, -1):
                    if rows[i][start_col] == start:
                        rows.pop(i)
                        remove -= 1
                        if remove == 0:
                            break
            prev_n = curr_n

    return pd.DataFrame(rows)


def _to_datetime_series(s: pd.Series) -> pd.Series:
    """Robust datetime conversion and normalize to date (drop time)."""
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    try:
        dt = dt.dt.tz_localize(None)
    except Exception:
        pass
    return dt.dt.normalize()


def compute_delays_days(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    *,
    max_delay_days: int = 120,
    drop_negative: bool = True,
    min_start_date: Optional[datetime] = None,
) -> np.ndarray:
    """
    Return integer delays (end - start) in days after basic cleaning.
    """
    if df is None or start_col not in df.columns or end_col not in df.columns:
        return np.array([], dtype=int)

    start = _to_datetime_series(df[start_col])
    end = _to_datetime_series(df[end_col])

    m = start.notna() & end.notna()
    if min_start_date is not None:
        m = m & (start >= pd.Timestamp(min_start_date))

    if not m.any():
        return np.array([], dtype=int)

    delays = (end[m] - start[m]).dt.days
    delays = pd.Series(delays).dropna().astype(int)

    if drop_negative:
        delays = delays[delays >= 0]

    if max_delay_days is not None:
        if drop_negative:
            delays = delays[(delays >= 0) & (delays <= max_delay_days)]
        else:
            delays = delays[(delays >= -max_delay_days) & (delays <= max_delay_days)]

    return delays.to_numpy(dtype=int)


# ============================================================
# Column standardization and safe guards.
# ============================================================

def norm_col_key(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_")
    c = re.sub(r"__+", "_", c)
    return c


RISKY_RAW_NAMES = {
    "report_date", "reported_date", "notification_date",
    "confirm_date", "confirmation_date",
    "date", "event_date"
}


def apply_column_overrides(
    df: pd.DataFrame,
    overrides: Dict[str, str],
    standard_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Overrides are raw column names -> standardized names.
    Matching is done case-insensitively by normalizing.
    """
    raw_cols = list(df.columns)
    raw_norm_map = {norm_col_key(c): c for c in raw_cols}

    rename_map = {}
    for raw_name, std_name in (overrides or {}).items():
        k = norm_col_key(raw_name)
        if k in raw_norm_map:
            rename_map[raw_norm_map[k]] = std_name

    out = df.rename(columns=rename_map).copy()

    for c in standard_columns:
        if c not in out.columns:
            out[c] = pd.NA

    return out, rename_map


def risky_column_guard(df_raw: pd.DataFrame, overrides: Dict[str, str], name: str) -> None:
    raw_norms = {norm_col_key(c) for c in df_raw.columns}
    risky_present = raw_norms.intersection({norm_col_key(x) for x in RISKY_RAW_NAMES})
    if not risky_present:
        return

    override_norms = {norm_col_key(k) for k in (overrides or {}).keys()}
    missing = [r for r in risky_present if r not in override_norms]
    if missing:
        raise ValueError(
            f"[{name}] Risky columns present but not explicitly mapped in column_overrides: {missing}. "
            f"Add them to avoid semantic mis-mapping."
        )


def parse_dates_inplace(
    df: pd.DataFrame,
    date_columns: list[str] | None = None,
    dayfirst: bool = True,
) -> pd.DataFrame:
    if date_columns is None:
        return df
    for c in date_columns:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst)
    return df


def normalize_sex(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sex" in out.columns:
        s = out["sex"].astype("string").str.strip().str.lower()
        out["sex"] = s.replace({"m": "male", "f": "female"})
    return out


def sanity_checks(df: pd.DataFrame, name: str) -> List[str]:
    warns = []

    def frac_violate(a: str, b: str) -> float:
        ok = df[a].notna() & df[b].notna()
        if ok.sum() == 0:
            return 0.0
        return float((df.loc[ok, a] > df.loc[ok, b]).mean())

    if "date_symptom_onset" in df.columns and "date_death" in df.columns:
        f = frac_violate("date_symptom_onset", "date_death")
        if f > 0.05:
            warns.append(f"[{name}] symptom onset > death for {f:.1%} of paired rows.")
    if "date_symptom_onset" in df.columns and "date_report" in df.columns:
        f = frac_violate("date_symptom_onset", "date_report")
        if f > 0.20:
            warns.append(f"[{name}] onset > report for {f:.1%} of paired rows (verify semantics).")

    return warns


# ============================================================
# Source reading utilities.
# ============================================================

def read_from_source(source: Dict[str, Any], fmt: str, read_options: Dict[str, Any]) -> pd.DataFrame:
    src_type = source["type"]
    value = source["value"]

    if fmt == "csv":
        return pd.read_csv(value, **read_options)

    if fmt in ("tsv", "txt"):
        if "sep" not in read_options:
            read_options = {**read_options, "sep": "\t"}
        return pd.read_csv(value, **read_options)

    if fmt in ("xlsx", "xls"):
        return pd.read_excel(value, **read_options)

    if fmt in ("rda", "rdata"):
        if pyreadr is None:
            raise ImportError("pyreadr not installed. Run: pip install pyreadr")
        if src_type != "path":
            raise ValueError("Reading .rda/.RData from URL is not supported directly; download first.")
        res = pyreadr.read_r(value)
        obj = next(iter(res.values()))
        if not isinstance(obj, pd.DataFrame):
            obj = pd.DataFrame(obj)
        return obj

    raise ValueError(f"Unsupported format: {fmt}")


# ============================================================
# Catalog loading
# ============================================================

def load_catalog(catalog_path: str | Path) -> Dict[str, Any]:
    p = Path(catalog_path)
    return json.loads(p.read_text())


def load_all_from_catalog(catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Loads all datasets specified in the catalog config.
    Applies standard column names, date parsing, and metadata warnings.
    """
    standard_columns: List[str] = catalog["standard_columns"]
    datasets_cfg: List[Dict[str, Any]] = catalog["datasets"]

    datasets: Dict[str, Dict[str, Any]] = {}

    for cfg in datasets_cfg:
        name = cfg["name"]
        print(name)

        source = cfg["source"]
        fmt = cfg["format"]

        read_options = cfg.get("read_options", {}) or {}
        preprocess = cfg.get("preprocess", {}) or {}
        date_parse_cfg = (preprocess.get("date_parse", {}) or {}) if isinstance(preprocess, dict) else {}

        df_raw = read_from_source(source, fmt, read_options)

        overrides = cfg.get("column_overrides", {}) or {}
        risky_column_guard(df_raw, overrides, name)
        df_std, rename_map = apply_column_overrides(df_raw, overrides, standard_columns)

        # Auto detect standardized date columns from overrides
        auto_date_cols = []
        for std_col in overrides.values():
            if "date" in std_col.lower():
                auto_date_cols.append(std_col)

        dayfirst = date_parse_cfg.get("dayfirst", True)
        df_std = parse_dates_inplace(df_std, date_columns=auto_date_cols, dayfirst=dayfirst)
        df_std = normalize_sex(df_std)

        meta = {
            "name": name,
            "disease": cfg.get("disease"),
            "citation_short": cfg.get("citation_short"),
            "data_availability": cfg.get("meta", {}).get("data_availability"),
            "location": cfg.get("meta", {}).get("location"),
            "format": fmt,
            "linelist": cfg.get("linelist", True),
            "read_options": read_options,
            "preprocess": preprocess,
            "date_parse": date_parse_cfg,
            "delay_definitions": cfg.get("delay_definitions", []),
            "n_rows": int(len(df_std)),
            "raw_columns": [str(c) for c in df_std.columns],
            "applied_rename_map": rename_map,
            "warnings": sanity_checks(df_std, name),
        }

        datasets[name] = {"df": df_std, "meta": meta}

    return datasets


# ============================================================
# Summary table (keep + save CSV)
# ============================================================

def summarize_datasets_to_table(
    datasets: Dict[str, Dict[str, Any]],
    *,
    max_delay_days: int = 120,
    out_csv: str = "./figs/dataset_delay_summary.csv",
    min_start_date: datetime = datetime(2000, 1, 1),
) -> pd.DataFrame:
    rows = []

    for ds_name, obj in datasets.items():
        df = obj.get("df")
        meta = obj.get("meta", {})
        delay_defs = meta.get("delay_definitions", [])
        if df is None or not delay_defs:
            continue

        linelist = bool(meta.get("linelist", True))

        for d in delay_defs:
            delay_name = d.get("name")
            start_col = d.get("start_col")
            end_col = d.get("end_col")

            if start_col not in df.columns or end_col not in df.columns:
                continue

            df_work = df
            if not linelist:
                # convert cumulative-aggregated to linelist
                df_work = cumulative_aggregated_to_linelist(
                    df_work,
                    start_col=start_col,
                    end_col=end_col,
                    n_col="n_samples",
                )

            delays = compute_delays_days(
                df_work,
                start_col=start_col,
                end_col=end_col,
                max_delay_days=max_delay_days,
                drop_negative=True,
                min_start_date=min_start_date,
            )
            if delays.size == 0:
                continue

            n_samples = int(len(delays))
            median = float(np.median(delays))
            if n_samples > 1:
                q25, q75 = np.percentile(delays, [25, 75])
                iqr = float(q75 - q25)
            else:
                iqr = 0.0

            rows.append({
                "name": ds_name,
                "disease": meta.get("disease"),
                "delay_name": delay_name,
                "from": start_col,
                "to": end_col,
                "start_date": df_work[start_col].min(),
                "end_date": df_work[start_col].max(),
                "n_days": (df_work[start_col].max() - df_work[start_col].min()).days
                         if pd.notna(df_work[start_col].min()) and pd.notna(df_work[start_col].max()) else None,
                "n_samples": n_samples,
                "location": meta.get("location"),
                "median": median,
                "IQR": iqr,
                "source": meta.get("data_availability"),
            })

    if not rows:
        raise ValueError(
            "No dataset-delay combinations produced valid summaries. "
            "Check delay_definitions, column names, and filtering."
        )

    out = pd.DataFrame(rows)[
        [
            "name", "disease", "delay_name", "from", "to",
            "start_date", "end_date", "n_days",
            "n_samples", "location", "median", "IQR", "source"
        ]
    ].sort_values(["disease", "name", "delay_name", "from", "to"])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


# ============================================================
# Continuous distribution support objects (Lessler-like)
# ============================================================

@dataclass(frozen=True)
class EmpiricalDelaySupport:
    """
    A flexible support package for downstream scripts.

    Key:
      - key: unique identifier (dataset + delay_name + from/to)
      - dist: scipy frozen RV for continuous distribution
      - params: fitted params (family-specific)
      - delays: optionally keep raw delays array (can be None to save memory)
      - meta: minimal provenance
    """
    key: str
    dataset: str
    disease: Optional[str]
    location: Optional[str]
    delay_name: Optional[str]
    start_col: str
    end_col: str
    family: str                 # "gamma" or "lognorm"
    dist: Any                   # scipy.stats frozen dist
    params: Dict[str, float]    # fitted params
    n: int
    median: float
    iqr: float
    source: Optional[str]


def _fit_continuous_distribution(
    delays: np.ndarray,
    family: str,
    *,
    zero_handling: str = "jitter",   # "jitter" | "drop" | "shift"
    eps: float = 0.5,                # days; 0.5 is a nice “same-day ≈ half-day”
) -> Tuple[Any, Dict[str, float]]:
    """
    Fit a continuous distribution to delay samples.

    zero_handling:
      - "jitter": replace x<=0 with eps
      - "drop":   drop x<=0
      - "shift":  x := x + eps (shifts everything to be >0)
    """
    x = np.asarray(delays, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 5:
        raise ValueError("Need at least 5 delay samples to fit.")

    if zero_handling == "drop":
        x = x[x > 0]
    elif zero_handling == "jitter":
        x = np.where(x <= 0, eps, x)
    elif zero_handling == "shift":
        x = x + eps
    else:
        raise ValueError("zero_handling must be one of {'jitter','drop','shift'}")

    if x.size < 5:
        raise ValueError("Not enough positive delays after zero_handling.")

    if family == "gamma":
        # gamma support: x>0 with loc fixed at 0
        a, loc, scale = stats.gamma.fit(x, floc=0.0)
        dist = stats.gamma(a=a, loc=0.0, scale=scale)
        return dist, {"a": float(a), "loc": 0.0, "scale": float(scale), "eps": float(eps), "zero_handling": zero_handling}

    if family in ("lognorm", "lognormal"):
        # lognormal support: x>0 with loc fixed at 0
        s, loc, scale = stats.lognorm.fit(x, floc=0.0)
        dist = stats.lognorm(s=s, loc=0.0, scale=scale)
        return dist, {"s": float(s), "loc": 0.0, "scale": float(scale), "eps": float(eps), "zero_handling": zero_handling}

    raise ValueError("family must be one of {'gamma','lognorm'}")


def build_empirical_delay_support(
    datasets: Dict[str, Dict[str, Any]],
    *,
    dist_family: str = "gamma",
    max_delay_days: int = 120,
    min_start_date: datetime = datetime(2000, 1, 1),
    keep_raw_delays: bool = False,
) -> Tuple[Dict[str, EmpiricalDelaySupport], pd.DataFrame]:
    """
    Lessler-like entry point for empirical delay data.

    Returns:
      support_dict: key -> EmpiricalDelaySupport (continuous dist + fitted params)
      summary_df: summary table for downstream inspection or export

    Notes:
      - Supports linelist or aggregated (converted using meta['linelist'])
      - dist_family controls whether support.dist is gamma or lognorm
      - key is stable and usable for plotting selections in other scripts
    """
    summary_df = summarize_datasets_to_table(
        datasets,
        max_delay_days=max_delay_days,
        out_csv="/tmp/_dummy.csv",  # not used; caller can save separately if desired
        min_start_date=min_start_date,
    )

    support: Dict[str, EmpiricalDelaySupport] = {}

    for ds_name, obj in datasets.items():
        df = obj.get("df")
        meta = obj.get("meta", {})
        delay_defs = meta.get("delay_definitions", [])
        if df is None or not delay_defs:
            continue

        linelist = bool(meta.get("linelist", True))

        for d in delay_defs:
            delay_name = d.get("name")
            start_col = d.get("start_col")
            end_col = d.get("end_col")

            if start_col not in df.columns or end_col not in df.columns:
                continue

            df_work = df
            if not linelist:
                df_work = cumulative_aggregated_to_linelist(
                    df_work, start_col=start_col, end_col=end_col, n_col="n_samples"
                )

            delays = compute_delays_days(
                df_work,
                start_col=start_col,
                end_col=end_col,
                max_delay_days=max_delay_days,
                drop_negative=True,
                min_start_date=min_start_date,
            )
            if delays.size < 5:
                continue

            n = int(len(delays))
            med = float(np.median(delays))
            q25, q75 = np.percentile(delays, [25, 75])
            iqr = float(q75 - q25)

            try:
                dist, params = _fit_continuous_distribution(delays, dist_family)
            except Exception as e:
                print(f"[WARN] fit failed for {ds_name} / {delay_name} ({start_col}->{end_col}): {e}")
                continue

            key = f"{ds_name}::{delay_name or ''}::{start_col}->{end_col}::{dist_family}"

            support[key] = EmpiricalDelaySupport(
                key=key,
                dataset=ds_name,
                disease=meta.get("disease"),
                location=meta.get("location"),
                delay_name=delay_name,
                start_col=start_col,
                end_col=end_col,
                family=dist_family,
                dist=dist,
                params=params,
                n=n,
                median=med,
                iqr=iqr,
                source=meta.get("data_availability"),
            )

    # Return the *real* summary table without forcing a write side-effect
    # (call summarize_datasets_to_table separately to save)
    summary_df = summarize_datasets_to_table(
        datasets,
        max_delay_days=max_delay_days,
        out_csv="/tmp/_ignore_empirical_summary.csv",
        min_start_date=min_start_date,
    )
    try:
        Path("/tmp/_ignore_empirical_summary.csv").unlink(missing_ok=True)
    except Exception:
        pass

    return support, summary_df


def select_empirical_support(
    support_dict: Dict[str, EmpiricalDelaySupport],
    *,
    datasets: Optional[Iterable[str]] = None,
    delay_name_contains: Optional[str] = None,
    cols_match: Optional[Tuple[str, str]] = None,  # (start_col, end_col)
) -> Dict[str, EmpiricalDelaySupport]:
    """
    Flexible filtering helper for downstream scripts.
    """
    out: Dict[str, EmpiricalDelaySupport] = {}
    for k, s in support_dict.items():
        if datasets is not None and s.dataset not in set(datasets):
            continue
        if delay_name_contains is not None:
            dn = (s.delay_name or "")
            if delay_name_contains.lower() not in dn.lower():
                continue
        if cols_match is not None:
            if (s.start_col, s.end_col) != cols_match:
                continue
        out[k] = s
    return out

def load_schema(schema_path):
    with open(schema_path) as f:
        return json.load(f)

def load_datasets_cfg(datasets_dir):
    datasets_cfg = []
    for path in Path(datasets_dir).glob("*.json"):
        with open(path) as f:
            datasets_cfg.append(json.load(f))
    return datasets_cfg

    #
    #
    # # 1) Keep summary table writing
    # summarize_datasets_to_table(
    #     datasets,
    #     max_delay_days=120,
    #     out_csv="./data/output/dataset_delay_summary.csv",
    # )
    #
    # # 2) Build Lessler-like support dict (continuous)
    # support_gamma, summary_df = build_empirical_delay_support(
    #     datasets,
    #     dist_family="gamma",
    #     max_delay_days=120,
    # )
    # print(f"Built {len(support_gamma)} gamma supports.")
    #
    # support_lognorm, _ = build_empirical_delay_support(
    #     datasets,
    #     dist_family="lognorm",
    #     max_delay_days=120,
    # )
    # print(f"Built {len(support_lognorm)} lognorm supports.")
