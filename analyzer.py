"""Portfolio performance analysis: data loading, metrics, stretch analyses and charts.

All prices and values are in CAD. USD-listed stocks are converted with the daily
USD/CAD rate (unhedged), which is what a Canadian investor actually experiences.
"""
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize

TRADING_DAYS = 252
FX_TICKER = "CAD=X"  # Yahoo's USD/CAD rate: CAD per 1 USD
CAD_SUFFIXES = (".TO", ".V")
BOC_SERIES = "TB.CDN.90D.MID"  # Government of Canada 3-month T-bill yield, daily
BOC_URL = "https://www.bankofcanada.ca/valet/observations/{series}/json?start_date={start}&end_date={end}"


# ---------------------------------------------------------------- data

def validate_weights(portfolio):
    total = sum(portfolio.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Weights sum to {total:.4f}, not 1.0")
    if any(w < 0 for w in portfolio.values()):
        raise ValueError("Negative weights (short positions) are not supported")


def is_cad(ticker):
    return ticker.upper().endswith(CAD_SUFFIXES)


def download_prices(tickers, years):
    """Download adjusted closes and return (prices in CAD, USD/CAD rate)."""
    raw = yf.download(
        tickers + [FX_TICKER],
        period=f"{years}y",
        auto_adjust=True,  # adjust for dividends and splits -> total return
        progress=False,
        threads=False,     # threaded downloads can fail with 'database is locked'
    )
    close = raw["Close"]

    # yfinance returns an empty column instead of raising, so check ourselves
    missing = [t for t in tickers + [FX_TICKER] if t not in close or close[t].isna().all()]
    if missing:
        raise RuntimeError(f"No data for {missing}. Check the symbol or try again.")

    # Drop days when none of our stocks traded, carry prices over single-exchange
    # holidays, then drop leading gaps.
    prices = close.dropna(how="all", subset=tickers).ffill().dropna()

    fx = prices[FX_TICKER]
    prices_cad = prices[tickers].copy()
    usd = [t for t in tickers if not is_cad(t)]
    prices_cad[usd] = prices_cad[usd].mul(fx, axis=0)
    return prices_cad, fx


def download_risk_free(start, end):
    """Daily 3-month Government of Canada T-bill yield from the Bank of Canada, as a decimal."""
    url = BOC_URL.format(series=BOC_SERIES, start=f"{start:%Y-%m-%d}", end=f"{end:%Y-%m-%d}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    rows = {
        obs["d"]: float(obs[BOC_SERIES]["v"]) / 100
        for obs in data["observations"]
        if obs.get(BOC_SERIES, {}).get("v")
    }
    series = pd.Series(rows, name="rf_annual")
    series.index = pd.to_datetime(series.index)
    return series.rename_axis("Date")


def load_data(portfolio, benchmark, years, data_dir="data", refresh=False, rf_fallback=0.03):
    """Return (prices_cad, rf_annual), using the saved snapshot in data_dir when it matches.

    The snapshot keeps every reported number reproducible. Pass refresh=True to re-download.
    """
    validate_weights(portfolio)
    tickers = list(dict.fromkeys(list(portfolio) + [benchmark]))
    data_dir = Path(data_dir)
    prices_path = data_dir / "prices_cad.csv"
    rf_path = data_dir / "risk_free.csv"

    prices = None
    if not refresh and prices_path.exists():
        cached = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        span = (cached.index[-1] - cached.index[0]).days / 365.25
        if set(tickers) <= set(cached.columns) and abs(span - years) < 0.1:
            prices = cached[tickers]

    if prices is None:
        prices, fx = download_prices(tickers, years)
        data_dir.mkdir(exist_ok=True)
        prices.to_csv(prices_path)
        fx.to_csv(data_dir / "usdcad.csv")
        rf_path.unlink(missing_ok=True)  # new date range -> refresh risk-free too

    if rf_path.exists():
        rf = pd.read_csv(rf_path, index_col=0, parse_dates=True)["rf_annual"]
    else:
        try:
            rf = download_risk_free(prices.index[0], prices.index[-1])
            rf.to_csv(rf_path)
        except Exception as exc:  # network down, API changed, ...
            print(f"Warning: risk-free download failed ({exc}); using {rf_fallback:.2%}")
            rf = pd.Series(rf_fallback, index=prices.index, name="rf_annual")
    return prices, rf


def daily_risk_free(rf_annual, index):
    """Convert annual T-bill yields to a daily return on each trading day in index."""
    aligned = rf_annual.reindex(rf_annual.index.union(index)).ffill().bfill().reindex(index)
    return (1 + aligned) ** (1 / TRADING_DAYS) - 1


# ---------------------------------------------------------------- portfolio values

def buy_and_hold_value(prices, portfolio, initial=10_000):
    """Buy at the target weights on day one, then never trade. Weights drift with prices."""
    weights = pd.Series(portfolio)
    shares = initial * weights / prices[weights.index].iloc[0]
    return (prices[weights.index] * shares).sum(axis=1).rename("Portfolio")


def rebalanced_value(prices, portfolio, freq="M", initial=10_000):
    """Reset holdings to the target weights on the last trading day of each period."""
    weights = pd.Series(portfolio)
    p = prices[weights.index]
    rebalance_days = set(p.groupby(p.index.to_period(freq)).tail(1).index)

    shares = initial * weights / p.iloc[0]
    values = []
    for date, row in p.iterrows():
        value = (shares * row).sum()
        values.append(value)
        if date in rebalance_days:
            shares = value * weights / row
    return pd.Series(values, index=p.index, name="Rebalanced")


# ---------------------------------------------------------------- metrics

def drawdown(value):
    """Percent below the running peak on each day (0 at a new high, negative otherwise)."""
    return value / value.cummax() - 1


def compute_metrics(value, benchmark_value, rf_daily):
    """Core performance metrics for a value series, all annualized where relevant."""
    returns = value.pct_change().dropna()
    bench_returns = benchmark_value.pct_change().dropna()
    years = (value.index[-1] - value.index[0]).days / 365.25

    total_return = value.iloc[-1] / value.iloc[0] - 1
    annual_return = (1 + total_return) ** (1 / years) - 1
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS)

    excess = returns - rf_daily.reindex(returns.index)
    sharpe = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)

    dd = drawdown(value)
    trough = dd.idxmin()
    peak = value.loc[:trough].idxmax()

    beta = returns.cov(bench_returns) / bench_returns.var()

    return {
        "Total return": total_return,
        "Annualized return": annual_return,
        "Annualized volatility": annual_vol,
        "Sharpe ratio": sharpe,
        "Max drawdown": dd.min(),
        "Max drawdown peak": peak.date().isoformat(),
        "Max drawdown trough": trough.date().isoformat(),
        "Beta": beta,
        "Ending value": value.iloc[-1],
    }


# ---------------------------------------------------------------- efficient frontier

def annualized_stats(asset_returns):
    """Annualized mean return vector and covariance matrix from daily returns."""
    return asset_returns.mean() * TRADING_DAYS, asset_returns.cov() * TRADING_DAYS


def portfolio_stats(weights, mu, cov):
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    return ret, vol


def random_portfolios(mu, cov, n, rf, seed=42):
    """n random long-only portfolios. Returns (weights, returns, vols, sharpes)."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(mu)), size=n)  # each row is >= 0 and sums to 1
    rets = weights @ mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov.values, weights))
    return weights, rets, vols, (rets - rf) / vols


def _optimize(objective, n_assets, extra_constraints=()):
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra_constraints]
    result = minimize(
        objective,
        x0=np.full(n_assets, 1 / n_assets),
        bounds=[(0, 1)] * n_assets,  # long-only, no leverage
        constraints=constraints,
        method="SLSQP",
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x


def max_sharpe_weights(mu, cov, rf):
    def neg_sharpe(w):
        ret, vol = portfolio_stats(w, mu.values, cov.values)
        return -(ret - rf) / vol
    return pd.Series(_optimize(neg_sharpe, len(mu)), index=mu.index)


def min_volatility_weights(mu, cov):
    return pd.Series(
        _optimize(lambda w: portfolio_stats(w, mu.values, cov.values)[1], len(mu)), index=mu.index
    )


def efficient_frontier(mu, cov, n_points=40):
    """Lowest achievable volatility for a range of target returns. Returns (vols, rets)."""
    w_min = min_volatility_weights(mu, cov).values
    targets = np.linspace(portfolio_stats(w_min, mu.values, cov.values)[0], mu.max(), n_points)
    vols = []
    for target in targets:
        w = _optimize(
            lambda w: portfolio_stats(w, mu.values, cov.values)[1],
            len(mu),
            [{"type": "eq", "fun": lambda w, t=target: w @ mu.values - t}],
        )
        vols.append(portfolio_stats(w, mu.values, cov.values)[1])
    return np.array(vols), targets


# ---------------------------------------------------------------- Monte Carlo

def monte_carlo(asset_returns, portfolio, n_sims=10_000, days=252, initial=10_000, seed=42):
    """Bootstrap simulation: build each 1-year path from randomly drawn historical days.

    Drawing whole days keeps the real co-movement between holdings and the real
    fat-tailed shape of returns. Returns an (n_sims, days) array of portfolio values.
    """
    weights = pd.Series(portfolio)
    daily = (asset_returns[weights.index] @ weights).values  # fixed-weight portfolio return
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(daily), size=(n_sims, days))
    return initial * np.cumprod(1 + daily[picks], axis=1)


def summarize_simulation(paths, initial=10_000):
    ending = paths[:, -1]
    pct = np.percentile(ending, [5, 25, 50, 75, 95])
    return {
        "5th percentile": pct[0],
        "25th percentile": pct[1],
        "Median": pct[2],
        "75th percentile": pct[3],
        "95th percentile": pct[4],
        "Probability of loss": float((ending < initial).mean()),
        "95% 1-year VaR": initial - pct[0],
    }


# ---------------------------------------------------------------- charts

COLORS = {
    "portfolio": "#2a78d6",  # blue
    "benchmark": "#eb6834",  # orange
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e4e3df",
    "surface": "#fcfcfb",
}
CORR_CMAP = LinearSegmentedColormap.from_list("corr", ["#e34948", "#f0efec", "#2a78d6"])
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#0d366b"])
LABEL_BOX = dict(boxstyle="round,pad=0.25", fc=COLORS["surface"], ec="none", alpha=0.9)


def _style(ax, title, ylabel=None):
    ax.set_facecolor(COLORS["surface"])
    ax.figure.set_facecolor(COLORS["surface"])
    ax.set_title(title, loc="left", fontsize=13, color=COLORS["text"], pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=COLORS["muted"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.tick_params(colors=COLORS["muted"], length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["grid"])


def _end_label(ax, series, text, color):
    ax.annotate(text, (series.index[-1], series.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=COLORS["text"],
                bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["surface"], ec=color, lw=1))


def plot_growth(port_value, bench_value, bench_name="Benchmark"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(port_value, color=COLORS["portfolio"], lw=2, label="Portfolio")
    ax.plot(bench_value, color=COLORS["benchmark"], lw=2, label=bench_name)
    _end_label(ax, port_value, f"${port_value.iloc[-1]:,.0f}", COLORS["portfolio"])
    _end_label(ax, bench_value, f"${bench_value.iloc[-1]:,.0f}", COLORS["benchmark"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    _style(ax, f"Growth of ${port_value.iloc[0]:,.0f} (CAD)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_drawdown(port_value, bench_value, bench_name="Benchmark"):
    fig, ax = plt.subplots(figsize=(10, 4))
    for value, name, color in [(bench_value, bench_name, COLORS["benchmark"]),
                               (port_value, "Portfolio", COLORS["portfolio"])]:
        dd = drawdown(value) * 100
        if name == "Portfolio":  # shade only one series so the fills don't blend
            ax.fill_between(dd.index, dd, 0, color=color, alpha=0.15, lw=0)
        ax.plot(dd, color=color, lw=1.5, label=f"{name} (max {dd.min():.1f}%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _style(ax, "Drawdown from previous peak")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_correlation(corr):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1))
    im = ax.imshow(corr.values, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.index)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.8 else COLORS["text"])
    _style(ax, "Correlation of daily returns (CAD)")
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_rebalancing(buy_hold, rebalanced):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    top.plot(buy_hold, color=COLORS["portfolio"], lw=2, label="Buy-and-hold")
    top.plot(rebalanced, color=COLORS["benchmark"], lw=2, label="Monthly rebalanced")
    _end_label(top, buy_hold, f"${buy_hold.iloc[-1]:,.0f}", COLORS["portfolio"])
    _end_label(top, rebalanced, f"${rebalanced.iloc[-1]:,.0f}", COLORS["benchmark"])
    top.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    _style(top, "Buy-and-hold vs monthly rebalancing")
    top.legend(frameon=False, loc="upper left")

    gap = rebalanced - buy_hold
    bottom.fill_between(gap.index, gap, 0, color=COLORS["muted"], alpha=0.25, lw=0)
    bottom.plot(gap, color=COLORS["muted"], lw=1.2)
    bottom.axhline(0, color=COLORS["grid"], lw=1)
    bottom.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    _style(bottom, "Rebalanced minus buy-and-hold")
    fig.tight_layout()
    return fig


def plot_frontier(rand_vols, rand_rets, rand_sharpes, frontier_vols, frontier_rets, points):
    """points: {label: (vol, ret)} for portfolios to mark, e.g. current and max-Sharpe."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(rand_vols * 100, rand_rets * 100, c=rand_sharpes, cmap=SEQ_CMAP,
                    s=4, alpha=0.8, lw=0)
    ax.plot(frontier_vols * 100, frontier_rets * 100, color=COLORS["text"], lw=2,
            label="Efficient frontier")
    markers = {"Max Sharpe": ("*", COLORS["benchmark"], 320, (12, 10), "left"),
               "Min volatility": ("D", COLORS["muted"], 90, (12, -4), "left"),
               "Your portfolio": ("o", COLORS["portfolio"], 120, (12, -16), "left")}
    for label, (vol, ret) in points.items():
        marker, color, size, offset, ha = markers.get(label, ("o", COLORS["muted"], 90, (12, -4), "left"))
        ax.scatter(vol * 100, ret * 100, marker=marker, s=size, color=color,
                   edgecolor=COLORS["surface"], linewidth=1.5, zorder=5, label=label)
        ax.annotate(label, (vol * 100, ret * 100), xytext=offset, textcoords="offset points",
                    ha=ha, fontsize=9, color=COLORS["text"], bbox=LABEL_BOX, zorder=6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Annualized volatility", color=COLORS["muted"])
    _style(ax, "Efficient frontier (long-only, historical returns)", "Annualized return")
    fig.colorbar(sc, ax=ax, label="Sharpe ratio").outline.set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_monte_carlo(paths, initial=10_000):
    ending = paths[:, -1]
    p5, p50, p95 = np.percentile(ending, [5, 50, 95])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ending, bins=80, color=COLORS["portfolio"], alpha=0.85, edgecolor=COLORS["surface"],
            linewidth=0.5)
    ymax = ax.get_ylim()[1]
    for value, label in [(p5, "5th pct"), (p50, "Median"), (p95, "95th pct")]:
        ax.axvline(value, color=COLORS["text"], lw=1, ls="--")
        ax.text(value, ymax * 0.97, f"{label}\n${value:,.0f}", va="top", ha="center", fontsize=9,
                color=COLORS["text"], bbox=LABEL_BOX)
    ax.axvline(initial, color=COLORS["benchmark"], lw=1.5)
    ax.text(initial, ymax * 0.80, f"Start ${initial:,.0f}", ha="center", fontsize=9,
            color=COLORS["text"], bbox=LABEL_BOX)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlabel("Portfolio value after 1 year", color=COLORS["muted"])
    _style(ax, f"Monte Carlo: {len(ending):,} simulated 1-year outcomes", "Simulations")
    fig.tight_layout()
    return fig
