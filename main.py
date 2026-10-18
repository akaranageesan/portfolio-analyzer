"""Run the full analysis: print metrics, save charts to charts/ and numbers to results/.

Usage:
    python main.py            # use the saved data snapshot in data/ if it matches config.py
    python main.py --refresh  # re-download prices and risk-free rates
"""
import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
matplotlib.use("Agg")  # save charts to files without opening windows

import analyzer as az
import config


def fmt(name, value):
    if isinstance(value, str):
        return value
    if name in ("Sharpe ratio", "Beta"):
        return f"{value:.2f}"
    if "value" in name.lower() or "percentile" in name.lower() or name in ("Median", "95% 1-year VaR"):
        return f"${value:,.0f}"
    return f"{value:.2%}"


def print_table(title, columns):
    """columns: {column name: {metric: value}}"""
    names = list(next(iter(columns.values())))
    print(f"\n{title}\n" + "-" * len(title))
    print(f"{'':26}" + "".join(f"{c:>18}" for c in columns))
    for name in names:
        print(f"{name:26}" + "".join(f"{fmt(name, col[name]):>18}" for col in columns.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download all data")
    args = parser.parse_args()

    prices, rf_annual = az.load_data(config.PORTFOLIO, config.BENCHMARK, config.YEARS,
                                     refresh=args.refresh, rf_fallback=config.RISK_FREE_FALLBACK)
    rf_daily = az.daily_risk_free(rf_annual, prices.index)
    holdings = list(config.PORTFOLIO)
    asset_returns = prices[holdings].pct_change().dropna()
    start = config.INITIAL_INVESTMENT

    print(f"Data: {len(prices)} trading days, {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Risk-free rate: 3-month GoC T-bill, average {rf_annual.mean():.2%} over the period")

    # Core metrics
    port = az.buy_and_hold_value(prices, config.PORTFOLIO, start)
    bench = az.buy_and_hold_value(prices, {config.BENCHMARK: 1.0}, start).rename(config.BENCHMARK)
    core = {
        "Portfolio": az.compute_metrics(port, bench, rf_daily),
        config.BENCHMARK: az.compute_metrics(bench, bench, rf_daily),
    }
    print_table("Core metrics (buy-and-hold, CAD)", core)
    corr = asset_returns.corr()
    print("\nCorrelation matrix of daily returns")
    print(corr.round(2).to_string())

    # Stretch 1: rebalancing
    rebal = az.rebalanced_value(prices, config.PORTFOLIO, config.REBALANCE_FREQ, start)
    rebalancing = {
        "Buy-and-hold": core["Portfolio"],
        "Rebalanced": az.compute_metrics(rebal, bench, rf_daily),
    }
    print_table(f"Buy-and-hold vs rebalanced ({config.REBALANCE_FREQ})", rebalancing)

    # Stretch 2: efficient frontier
    mu, cov = az.annualized_stats(asset_returns)
    rf_avg = float(rf_annual.mean())
    _, r_rets, r_vols, r_sharpes = az.random_portfolios(mu, cov, config.N_RANDOM_PORTFOLIOS,
                                                        rf_avg, config.RANDOM_SEED)
    w_sharpe = az.max_sharpe_weights(mu, cov, rf_avg)
    w_minvol = az.min_volatility_weights(mu, cov)
    f_vols, f_rets = az.efficient_frontier(mu, cov)
    w_current = pd.Series(config.PORTFOLIO)[mu.index]
    points = {}
    for label, w in [("Your portfolio", w_current), ("Max Sharpe", w_sharpe),
                     ("Min volatility", w_minvol)]:
        ret, vol = az.portfolio_stats(w.values, mu.values, cov.values)
        points[label] = (vol, ret)
    print("\nEfficient frontier (annualized, from daily mean returns)")
    print(f"{'':16}{'Return':>10}{'Volatility':>12}{'Sharpe':>9}")
    for label, (vol, ret) in points.items():
        print(f"{label:16}{ret:>10.2%}{vol:>12.2%}{(ret - rf_avg) / vol:>9.2f}")
    print("\nMax-Sharpe weights:")
    print(w_sharpe[w_sharpe > 0.005].sort_values(ascending=False).map("{:.1%}".format).to_string())

    # Stretch 3: Monte Carlo
    paths = az.monte_carlo(asset_returns, config.PORTFOLIO, config.N_SIMULATIONS,
                           config.SIM_DAYS, start, config.RANDOM_SEED)
    sim = az.summarize_simulation(paths, start)
    print_table(f"Monte Carlo: {config.N_SIMULATIONS:,} simulated years", {"1-year outcome": sim})

    # Save charts
    charts = Path("charts")
    charts.mkdir(exist_ok=True)
    figures = {
        "growth.png": az.plot_growth(port, bench, config.BENCHMARK),
        "drawdown.png": az.plot_drawdown(port, bench, config.BENCHMARK),
        "correlation.png": az.plot_correlation(corr),
        "rebalancing.png": az.plot_rebalancing(port, rebal),
        "frontier.png": az.plot_frontier(r_vols, r_rets, r_sharpes, f_vols, f_rets, points),
        "monte_carlo.png": az.plot_monte_carlo(paths, start),
    }
    for name, fig in figures.items():
        fig.savefig(charts / name, dpi=150)

    # Save numbers
    Path("results").mkdir(exist_ok=True)
    summary = {
        "data": {"start": str(prices.index[0].date()), "end": str(prices.index[-1].date()),
                 "trading_days": len(prices), "risk_free_avg": rf_avg},
        "portfolio": config.PORTFOLIO,
        "core": core,
        "correlation": corr.round(4).to_dict(),
        "rebalancing": rebalancing,
        "frontier": {
            "points": {k: {"return": r, "volatility": v, "sharpe": (r - rf_avg) / v}
                       for k, (v, r) in points.items()},
            "max_sharpe_weights": w_sharpe.round(4).to_dict(),
            "min_volatility_weights": w_minvol.round(4).to_dict(),
        },
        "monte_carlo": sim,
    }
    Path("results/summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved {len(figures)} charts to charts/ and numbers to results/summary.json")


if __name__ == "__main__":
    main()
