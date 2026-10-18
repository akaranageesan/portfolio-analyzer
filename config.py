"""Portfolio settings. Edit this file to analyze a different portfolio."""

# Ticker -> weight. Weights must sum to 1.0.
# Tickers ending in .TO (Toronto) or .V (TSX Venture) are treated as CAD; everything else as USD.
PORTFOLIO = {
    "RY.TO":   0.20,   # Royal Bank of Canada
    "TD.TO":   0.15,   # Toronto-Dominion Bank
    "SHOP.TO": 0.15,   # Shopify
    "ENB.TO":  0.15,   # Enbridge
    "AAPL":    0.15,   # Apple (USD)
    "MSFT":    0.10,   # Microsoft (USD)
    "XIU.TO":  0.10,   # iShares S&P/TSX 60 ETF
}

BENCHMARK = "XIU.TO"        # S&P/TSX 60
YEARS = 5                   # length of price history
INITIAL_INVESTMENT = 10_000 # CAD

# Used only if the Bank of Canada T-bill download fails.
RISK_FREE_FALLBACK = 0.03

# Stretch goals
REBALANCE_FREQ = "M"        # "M" = monthly, "Q" = quarterly
N_RANDOM_PORTFOLIOS = 20_000
N_SIMULATIONS = 10_000
SIM_DAYS = 252              # one trading year
RANDOM_SEED = 42            # fixed so results are reproducible
