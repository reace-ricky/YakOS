"""Monte Carlo simulation stub for YakOS DFS optimizer."""
import pandas as pd


def run_monte_carlo_for_lineups(
    lineups_df: pd.DataFrame,
    n_sims: int = 500,
    volatility_mode: str = "standard",
) -> pd.DataFrame:
    """Run Monte Carlo simulations on lineup projections.

    Parameters
    ----------
    lineups_df : pd.DataFrame
        DataFrame of lineups with at least a ``proj`` column.
    n_sims : int, optional
        Number of simulation iterations (default 500).
    volatility_mode : str, optional
        Volatility profile to use (default "standard").
        Reserved for future use.

    Returns
    -------
    pd.DataFrame
        Copy of *lineups_df* with added ``sim_mean`` and ``sim_std`` columns.
    """
    result = lineups_df.copy()
    result["sim_mean"] = result["proj"]
    result["sim_std"] = 0.0
    return result
