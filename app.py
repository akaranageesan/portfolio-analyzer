"""Streamlit dashboard. Run with:  streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import analyzer as az
import config

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("Portfolio Performance Analyzer")
st.caption("All values in CAD. USD stocks are converted at the daily USD/CAD rate (unhedged). "
           "Risk-free rate: 3-month Government of Canada T-bill (Bank of Canada).")

# ---------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Portfolio")
    default = pd.DataFrame({"Ticker": list(config.PORTFOLIO),
                            "Weight %": [round(w * 100, 2) for w in config.PORTFOLIO.values()]})
    edited = st.data_editor(default, num_rows="dynamic", hide_index=True, width="stretch")
    benchmark = st.text_input("Benchmark", config.BENCHMARK).strip().upper()
    years = st.slider("Years of history", 1, 10, config.YEARS)
    initial = st.number_input("Starting amount (CAD)", 1_000, 10_000_000,
                              config.INITIAL_INVESTMENT, step=1_000)

edited = edited.dropna()
edited = edited[edited["Ticker"].astype(str).str.strip() != ""]
portfolio = {str(t).strip().upper(): float(w) / 100 for t, w in zip(edited["Ticker"], edited["Weight %"])}
total = sum(portfolio.values())
if not portfolio or abs(total - 1) > 1e-4:
    st.error(f"Weights add up to {total:.2%}. Adjust them to 100% in the sidebar.")
    st.stop()


@st.cache_data(show_spinner="Loading prices...")
def get_data(portfolio_items, benchmark, years):
    portfolio = dict(portfolio_items)
    # The saved snapshot in data/ backs the README numbers, so only reuse it, never overwrite it.
    is_default = portfolio == config.PORTFOLIO and benchmark == config.BENCHMARK and years == config.YEARS
    data_dir = "data" if is_default else "data/dashboard"
    return az.load_data(portfolio, benchmark, years, data_dir=data_dir, refresh=not is_default,
                        rf_fallback=config.RISK_FREE_FALLBACK)


try:
    prices, rf_annual = get_data(tuple(portfolio.items()), benchmark, years)
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

rf_daily = az.daily_risk_free(rf_annual, prices.index)
holdings = list(portfolio)
asset_returns = prices[holdings].pct_change().dropna()
port = az.buy_and_hold_value(prices, portfolio, initial)
bench = az.buy_and_hold_value(prices, {benchmark: 1.0}, initial).rename(benchmark)
m_port = az.compute_metrics(port, bench, rf_daily)
m_bench = az.compute_metrics(bench, bench, rf_daily)

st.write(f"{len(prices):,} trading days, {prices.index[0]:%b %d, %Y} to {prices.index[-1]:%b %d, %Y}. "
         f"Average risk-free rate {rf_annual.mean():.2%}.")


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def metrics_table(columns):
    df = pd.DataFrame(columns).drop(["Max drawdown peak", "Max drawdown trough"])
    fmt = lambda name, v: (f"{v:.2f}" if name in ("Sharpe ratio", "Beta")
                           else f"${v:,.0f}" if name == "Ending value" else f"{v:.2%}")
    return df.apply(lambda col: [fmt(name, v) for name, v in col.items()])


perf, risk, rebal, frontier, mc = st.tabs(
    ["Performance", "Risk", "Rebalancing", "Efficient frontier", "Monte Carlo"])

with perf:
    cols = st.columns(5)
    for col, (name, kind) in zip(cols, [("Total return", "pct"), ("Annualized return", "pct"),
                                        ("Sharpe ratio", "num"), ("Max drawdown", "pct"),
                                        ("Beta", "num")]):
        p, b = m_port[name], m_bench[name]
        value = f"{p:.2%}" if kind == "pct" else f"{p:.2f}"
        delta = (f"{(p - b) * 100:+.1f} pts vs benchmark" if kind == "pct"
                 else f"{p - b:+.2f} vs benchmark")
        col.metric(name, value, delta, delta_color="off" if name == "Beta" else "normal")
    show(az.plot_growth(port, bench, benchmark))
    st.dataframe(metrics_table({"Portfolio": m_port, benchmark: m_bench}), width="stretch")

with risk:
    show(az.plot_drawdown(port, bench, benchmark))
    st.caption(f"Portfolio max drawdown {m_port['Max drawdown']:.2%}: "
               f"peak {m_port['Max drawdown peak']} to trough {m_port['Max drawdown trough']}.")
    left, _ = st.columns([3, 1])
    with left:
        show(az.plot_correlation(asset_returns.corr()))

with rebal:
    freq = st.radio("Rebalance", ["M", "Q"], horizontal=True,
                    format_func={"M": "Monthly", "Q": "Quarterly"}.get)
    rebal_value = az.rebalanced_value(prices, portfolio, freq, initial)
    st.dataframe(metrics_table({"Buy-and-hold": m_port,
                                "Rebalanced": az.compute_metrics(rebal_value, bench, rf_daily)}),
                 width="stretch")
    show(az.plot_rebalancing(port, rebal_value))
    st.caption("Ignores trading costs and taxes.")


@st.cache_data(show_spinner="Optimizing...")
def frontier_data(asset_returns, rf):
    mu, cov = az.annualized_stats(asset_returns)
    _, r_rets, r_vols, r_sharpes = az.random_portfolios(mu, cov, config.N_RANDOM_PORTFOLIOS, rf,
                                                        config.RANDOM_SEED)
    f_vols, f_rets = az.efficient_frontier(mu, cov)
    return (mu, cov, r_rets, r_vols, r_sharpes, f_vols, f_rets,
            az.max_sharpe_weights(mu, cov, rf), az.min_volatility_weights(mu, cov))


with frontier:
    if len(holdings) < 2:
        st.info("Add at least two holdings to see the efficient frontier.")
    else:
        rf_avg = float(rf_annual.mean())
        mu, cov, r_rets, r_vols, r_sharpes, f_vols, f_rets, w_sharpe, w_minvol = \
            frontier_data(asset_returns, rf_avg)
        weights = pd.DataFrame({"Current": pd.Series(portfolio)[mu.index],
                                "Max Sharpe": w_sharpe, "Min volatility": w_minvol})
        points, rows = {}, {}
        for label in ["Current", "Max Sharpe", "Min volatility"]:
            ret, vol = az.portfolio_stats(weights[label].values, mu.values, cov.values)
            points["Your portfolio" if label == "Current" else label] = (vol, ret)
            rows[label] = {"Return": f"{ret:.2%}", "Volatility": f"{vol:.2%}",
                           "Sharpe": f"{(ret - rf_avg) / vol:.2f}"}
        show(az.plot_frontier(r_vols, r_rets, r_sharpes, f_vols, f_rets, points))
        left, right = st.columns(2)
        left.dataframe(pd.DataFrame(rows).T, width="stretch")
        right.dataframe(weights.map("{:.1%}".format), width="stretch")
        st.caption("Returns are annualized arithmetic means of daily returns, so they run higher than "
                   "CAGR. Optimal weights are fitted to past data and are not a forecast.")

with mc:
    n_sims = st.select_slider("Simulations", [1_000, 5_000, 10_000, 20_000], config.N_SIMULATIONS)
    paths = az.monte_carlo(asset_returns, portfolio, n_sims, config.SIM_DAYS, initial,
                           config.RANDOM_SEED)
    summary = az.summarize_simulation(paths, initial)
    cols = st.columns(4)
    cols[0].metric("Median 1-year value", f"${summary['Median']:,.0f}")
    cols[1].metric("5th percentile", f"${summary['5th percentile']:,.0f}")
    cols[2].metric("Probability of loss", f"{summary['Probability of loss']:.1%}")
    cols[3].metric("95% 1-year VaR", f"${summary['95% 1-year VaR']:,.0f}")
    show(az.plot_monte_carlo(paths, initial))
    st.caption("Bootstrap: each simulated year is 252 randomly drawn historical days. "
               "Assumes the future resembles the sample period.")
