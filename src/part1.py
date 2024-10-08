import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# Constants
FILE_PATHS: List[str] = ['../data/stock1.csv', '../data/stock2.csv']
STOCK_NAMES: List[str] = ['Stock 1', 'Stock 2']
DISTRIBUTIONS: Dict[str, Any] = {
    'norm': ss.norm,
    'lognorm': ss.lognorm,
    'beta': ss.beta
}
COLORS: Dict[str, str] = {
    'norm': 'red',
    'lognorm': 'blue',
    'beta': 'green'
}

def main() -> None:
    for file_path, stock_name in zip(FILE_PATHS, STOCK_NAMES):
        data: Optional[pd.Series] = load_data(file_path)

        if data is None or data.empty:
            continue

        # Normalize data and filter out exact 0 or 1
        normalized_data: pd.Series = normalize_data(data)
        normalized_data = normalized_data[(normalized_data > 0) & (normalized_data < 1)]

        # Determine the number of bins based on the normalized data
        number_of_bins: int = determine_number_of_bins(normalized_data)

        # Plot the histogram of normalized data with dynamic bins
        plot_prices(normalized_data, stock_name, number_of_bins)

        # Fit distributions to normalized data
        fits: Dict[str, Tuple] = fit_distributions(normalized_data)
        
        # fit beta distribution to denormalized data
        denormalized_fits = fit_denormalized_distributions(data)

        # Plot fitted distributions
        plot_fitted_distributions(normalized_data, fits)
        plt.show()

        # Perform goodness-of-fit tests
        ks_results: Dict[str, Any] = goodness_of_fit(normalized_data, fits)

        # Identify the best fitting distribution based on the highest p-value
        best_fit: Any = max(ks_results.values(), key=lambda x: x.pvalue)
        best_distribution: str = [dist for dist, result in ks_results.items() if result == best_fit][0]

        # Print results
        print(f'Stock Name: {stock_name}')
        for dist, result in ks_results.items():
            stat: float = result.statistic
            p_val: float = result.pvalue
            print(f'\t{dist.capitalize()} Fit: Statistic={stat:.4f}, p-value={p_val:.4f}')

            if dist != best_distribution:
                diff_stat: float = stat - best_fit.statistic
                diff_p_val: float = p_val - best_fit.pvalue
                print(f'\t\tDifference from Best Fit: Statistic Diff={diff_stat:.4f}, p-value Diff={diff_p_val:.4f}')
            else:
                print('\t\tBest Fit')
        print()

def load_data(file_path: str) -> Optional[pd.Series]:
    """Load stock data from a CSV file and return the 'value' column."""
    try:
        df: pd.DataFrame = pd.read_csv(file_path)
        return df['value']
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except KeyError:
        print(f"'value' column not found in {file_path}.")
        return None

def normalize_data(data: pd.Series) -> pd.Series:
    """Normalize the data to a range of (0, 1)."""
    data_min: float = np.min(data)
    data_max: float = np.max(data)
    return (data - data_min + 1e-9) / (data_max - data_min + 1e-9)

def determine_number_of_bins(data: pd.Series) -> int:
    """Determine the number of bins based on the length of the data."""
    n: int = len(data)
    if n < 10:
        return 5  # Minimum number of bins
    elif n < 50:
        return int(np.sqrt(n))  # Square root choice
    else:
        return int(np.log2(n) + 1)  # Sturges' formula

def plot_prices(data: pd.Series, stock_name: str, number_of_bins: int) -> None:
    """Plot a histogram of normalized prices for the given stock."""
    plt.hist(data, density=True, bins=number_of_bins, alpha=0.5, color='g')
    plt.title(f'Histogram of {stock_name}')
    plt.xlabel('Normalized Price')
    plt.ylabel('Density')

def fit_distributions(data: pd.Series) -> Dict[str, Tuple]:
    """Fit various distributions to the normalized price data."""
    fits: Dict[str, Tuple] = {}
    for name, distribution in DISTRIBUTIONS.items():
        try:
            fits[name] = distribution.fit(data)
        except Exception as e:
            print(f"Fitting {name} distribution failed: {e}")
    return fits

def fit_denormalized_distributions(data: pd.Series) -> Dict[str, Tuple]:
    """Fit various distributions, including beta, to the denormalized price data."""
    fits: Dict[str, Tuple] = {}
    for name, distribution in DISTRIBUTIONS.items():
        try:
            # Fit directly to denormalized data
            fits[name] = distribution.fit(data)
        except Exception as e:
            print(f"Fitting {name} distribution failed: {e}")
    return fits

def plot_fitted_distributions(data: pd.Series, fits: Dict[str, Tuple]) -> None:
    """Overlay fitted distributions on the histogram of normalized prices."""
    x: np.ndarray = np.linspace(min(data), max(data), 100)

    # Loop through the fitted distributions and plot them
    for dist_name, params in fits.items():
        if dist_name in COLORS:
            plt.plot(x, DISTRIBUTIONS[dist_name].pdf(x, *params), color=COLORS[dist_name], label=f'{dist_name.capitalize()} Fit')

    plt.legend()

def goodness_of_fit(data: pd.Series, fits: Dict[str, Tuple]) -> Dict[str, Any]:
    """Perform goodness-of-fit tests for the fitted distributions."""
    results: Dict[str, Any] = {}
    for dist, params in fits.items():
        results[dist] = ss.kstest(data, dist, args=params)
    return results

if __name__ == '__main__':
    main()
