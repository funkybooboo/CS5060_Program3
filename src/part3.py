import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Callable
import warnings

# Constants
FILE_PATHS: List[str] = ['../data/stock1.csv', '../data/stock2.csv']
STOCK_NAMES: List[str] = ['Stock1', 'Stock2']
DISTRIBUTIONS: Dict[str, Any] = {
    'norm': ss.norm,
    'lognorm': ss.lognorm,
    'beta': ss.beta,
}
COLORS: Dict[str, str] = {
    'norm': 'red',
    'lognorm': 'blue',
    'beta': 'green',
}
GENERATOR: Dict[str, Callable] = {
    'norm': lambda params: lambda: np.random.normal(params[0], abs(params[1])),
    'lognorm': lambda params: lambda: np.random.lognormal(mean=params[0], sigma=abs(params[1])),
    'beta': lambda params: lambda: np.random.beta(params[0], params[1]),
}

NUM_PATHS: int = 5000
INITIAL_STOCK_PRICE: float = 100.0
DRIFT_RATE: float = 0.03
VOLATILITY_RATE: float = 17.04
TIME_INCREMENT: float = 1 / 365  # Daily increments
TOTAL_TIME: float = 1.0  # 1 year
STRIKE_PRICE: float = 100.0
RISK_FREE_RATE: float = 0.01
BETA_A: int = 9
BETA_B: int = 10
BETA_SHIFT: float = 0.35

def main() -> None:
    """Main function to execute the distribution fitting and plotting."""

    stock_distributions: Dict[str, Dict[str, Any]] = {}

    # Part 1
    for file_path, stock_name in zip(FILE_PATHS, STOCK_NAMES):
        best_fitted_distribution = process_stock(file_path, stock_name)
        if not best_fitted_distribution:
            print(f"{stock_name}: could not get best_fitted_distribution")
            continue  # Process all stocks even if one fails
        stock_distributions[stock_name] = best_fitted_distribution

    # Part 2
    simulate_stock_and_plot({"Stock0": {'name': "norm", "generator":
        lambda: np.random.normal(0, np.sqrt(TIME_INCREMENT))}})

    print()

    simulate_stock_and_plot({"Stock0": {"name": "beta", "generator":
        lambda: np.random.beta(BETA_A, BETA_B) - BETA_SHIFT}})

    print()

    ## Part 3
    simulate_stock_and_plot(stock_distributions)

    print()

def process_stock(file_path: str, stock_name: str) -> Optional[Dict[str, Callable]]:
    """
    Process a single stock: load data, normalize, fit distributions, and plot results.

    Parameters:
        file_path (str): Path to the stock data file.
        stock_name (str): Name of the stock.

    Returns:
        Optional[Dict[str, Any]]: Dictionary with the best distribution name and parameters, or None if unsuccessful.
    """
    non_normalized_data = load_data(file_path)

    if non_normalized_data is None or non_normalized_data.empty:
        return None

    normalized_data = normalize_data_and_filter(non_normalized_data)
    number_of_bins = determine_number_of_bins(normalized_data)

    plot_prices(normalized_data, stock_name, number_of_bins)

    fits = fit_distributions(normalized_data, non_normalized_data)
    plot_fitted_distributions(normalized_data, fits)
    plt.show()

    return evaluate_fit_results(non_normalized_data, fits, stock_name)

def normalize_data_and_filter(data: pd.Series) -> pd.Series:
    """
    Normalize data and filter out exact 0 or 1.

    Parameters:
        data (pd.Series): The data to normalize.

    Returns:
        pd.Series: Normalized data, filtered to exclude 0 and 1.
    """
    normalized_data = normalize_data(data)
    return normalized_data[(normalized_data > 0) & (normalized_data < 1)]

def evaluate_fit_results(non_normalized_data: pd.Series, fits: Dict[str, Dict[str, Tuple]], stock_name: str) -> Optional[Dict[str, Any]]:
    """
    Evaluate the fit results and identify the best fitting distribution.

    Parameters:
        non_normalized_data (pd.Series): Non-normalized stock data.
        fits (Dict[str, Dict[str, Tuple]]): Fitted parameters for each distribution.
        stock_name (str): Name of the stock.

    Returns:
        Optional[Dict[str, Any]]: Best fitting distribution and parameters, or None if no fits.
    """
    non_normalized_ks_results = goodness_of_fit(non_normalized_data, fits)

    non_normalized_best_fit = max(non_normalized_ks_results.values(), key=lambda x: x.pvalue)
    non_normalized_best_distribution = [dist for dist, result in non_normalized_ks_results.items() if result == non_normalized_best_fit][0]

    print(f'Stock Name: {stock_name}')
    for dist, result in non_normalized_ks_results.items():
        stat = result.statistic
        p_val = result.pvalue
        print(f'\t{dist.capitalize()} Fit: Statistic={stat:.4f}, p-value={p_val:.4f}')

        if dist != non_normalized_best_distribution:
            diff_stat = stat - non_normalized_best_fit.statistic
            diff_p_val = p_val - non_normalized_best_fit.pvalue
            print(f'\t\tDifference from Best Fit: Statistic Diff={diff_stat:.4f}, p-value Diff={diff_p_val:.4f}')
        else:
            print('\t\tBest Fit')

    return {
        'name': non_normalized_best_distribution,
        'generator': GENERATOR[non_normalized_best_distribution](fits[non_normalized_best_distribution]['non_normalized'])
    }

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

def plot_prices(normalized_data: pd.Series, stock_name: str, number_of_bins: int) -> None:
    """Plot a histogram of normalized prices for the given stock."""
    plt.hist(normalized_data, density=True, bins=number_of_bins, alpha=0.5, color='g')
    plt.title(f'Histogram of {stock_name}')
    plt.xlabel('Normalized Price')
    plt.ylabel('Density')

def fit_distributions(normalized_data: pd.Series, non_normalized_data: pd.Series) -> Dict[str, Dict[str, Tuple]]:
    """Fit various distributions to the normalized and non-normalized price data."""
    fits: Dict[str, Dict[str, Tuple]] = {}

    for name, distribution in DISTRIBUTIONS.items():
        fits[name] = {}

        try:
            # Fit to normalized data
            fits[name]['normalized'] = distribution.fit(normalized_data)
        except Exception as e:
            print(f"Fitting {name} distribution to normalized data failed: {e}")
            raise  # Stop execution on exception

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                # Fit to non-normalized data
                fits[name]['non_normalized'] = distribution.fit(non_normalized_data)
            except Exception:
                fits[name]['non_normalized'] = distribution.fit(normalized_data)

    return fits

def plot_fitted_distributions(normalized_data: pd.Series, fits: Dict[str, Dict[str, Tuple]]) -> None:
    """Overlay fitted distributions on the histogram of normalized prices."""
    x: np.ndarray = np.linspace(min(normalized_data), max(normalized_data), 100)

    # Loop through the fitted distributions and plot them
    for dist_name, params in fits.items():
        if dist_name in COLORS:
            plt.plot(x, DISTRIBUTIONS[dist_name].pdf(x, *params['normalized']), color=COLORS[dist_name], label=f'{dist_name.capitalize()} Fit (Normalized)')

    plt.legend()

def goodness_of_fit(data: pd.Series, fits: Dict[str, Dict[str, Tuple]]) -> Dict[str, Any]:
    """Perform goodness-of-fit tests for the fitted distributions."""
    results: Dict[str, Any] = {}
    for dist, params in fits.items():
        results[dist] = ss.kstest(data, dist, args=params['non_normalized'])
    return results

def simulate_stock_and_plot(stock_distributions: Dict[str, Dict[str, Any]]) -> None:
    stocks_price_paths: List[np.ndarray] = []
    stock_names: List[str] = list(stock_distributions.keys())
    simulation_name: str = ""

    for i in range(len(stock_names)):
        stock_name: str = stock_names[i]
        distribution_name = stock_distributions[stock_name]["name"]
        simulation_name += f"{stock_name} {distribution_name}\n"

        random_generator: Callable[[], float] = stock_distributions[stock_name]['generator']
        stock_price_paths: np.ndarray = generate_stock_price_paths(random_generator)
        stocks_price_paths.append(stock_price_paths)

    pricings: List[Tuple[np.ndarray, np.ndarray]] = calculate_basket_option_pricing(stocks_price_paths)

    plot_stock_predictions(stock_distributions, stock_names, stocks_price_paths)

    # TODO Bug fix somewhere in here
    average_final_price = get_average_final_price(pricings, stock_names)

    max_final_price = get_max_final_price(pricings, stock_names)

    outperforms_average = get_outperforms(average_final_price, pricings, stock_names)

    outperforms_max: bool = get_outperforms(max_final_price, pricings, stock_names) # TODO may have to fix

    basket_option_payoffs = get_basket_option_payoffs(pricings, stock_names)

    print(f" ---\n {simulation_name} \n--- ")
    print("Scenario 1")
    if outperforms_average:
        print(f"Average stock price after {int(1 / TIME_INCREMENT) * TOTAL_TIME} days: ${average_final_price:.2f}")
        print(f"Average payoff for a block of 100 options: ${basket_option_payoffs * 100:.2f}")
        print(f"Estimated cost of the option: ${basket_option_payoffs:.2f}")
    else:
        print("Did not outperform average, no payoff.")
    print()
    print("Scenario 2:")
    if outperforms_max:
        print(f"Max stock price after {int(1 / TIME_INCREMENT) * TOTAL_TIME} days: ${max_final_price:.2f}")
        print(f"Max payoff for a block of 100 options: ${basket_option_payoffs * 100:.2f}")
        print(f"Estimated cost of the option: ${basket_option_payoffs:.2f}")
    else:
        print("Did not outperform max, no payoff.")
    print()

def get_basket_option_payoffs(pricings, stock_names):
    basket_option_payoffs: float = 0
    for i in range(len(stock_names)):
        option_payoffs = pricings[i][0]
        average_option_payoffs = np.average(option_payoffs)
        basket_option_payoffs += average_option_payoffs
    basket_option_payoffs /= len(stock_names)
    return basket_option_payoffs

def get_outperforms(average_final_price, pricings, stock_names):
    outperforms_average: bool = True
    for i in range(len(stock_names)):
        option_payoffs = pricings[i][0]
        average_option_payoffs = np.average(option_payoffs)
        if average_final_price > average_option_payoffs:
            outperforms_average = False
            break
    return outperforms_average

def get_max_final_price(pricings, stock_names):
    max_final_price: float = 0
    for i in range(len(stock_names)):
        final_price = max(pricings[i][1])
        if max_final_price < final_price:
            max_final_price = final_price
    return max_final_price

def get_average_final_price(pricings, stock_names):
    average_final_price: float = 0
    for i in range(len(stock_names)):
        final_prices = pricings[i][1]
        average_final_price += final_prices
    average_final_price /= (NUM_PATHS * len(stock_names))
    return average_final_price

def plot_stock_predictions(stock_distributions, stock_names, stocks_price_paths):
    for i in range(len(stock_names)):
        distribution_name: str = stock_distributions[stock_names[i]]['name']
        # Plotting stock price paths
        plt.figure(figsize=(12, 6))
        for j in range(min(NUM_PATHS, 10)):
            plt.plot(stocks_price_paths[i][j])
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title(f"Simulated {stock_names[i]} Price Paths Using {distribution_name} Distribution")
        plt.show()

def generate_stock_price_paths(
        random_generator: Callable[[], float]
) -> np.ndarray:
    price_paths: np.ndarray = np.zeros((NUM_PATHS, int(TOTAL_TIME / TIME_INCREMENT)))
    for i in range(NUM_PATHS):
        current_price: float = INITIAL_STOCK_PRICE
        for t in range(int(TOTAL_TIME / TIME_INCREMENT)):
            price_change: float = DRIFT_RATE * TIME_INCREMENT + VOLATILITY_RATE * random_generator()
            current_price += price_change
            price_paths[i, t] = current_price
    return price_paths

def calculate_european_call_option_payoff(strike_price: float, final_stock_price: float) -> float:
    return max(final_stock_price - strike_price, 0)

def calculate_basket_option_pricing(
        stocks_price_paths: List[np.ndarray]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    pricings: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(stocks_price_paths)):
        price_paths = stocks_price_paths[i]
        option_payoffs: np.ndarray = np.zeros(price_paths.shape[0])
        final_prices: np.ndarray = np.zeros(price_paths.shape[0])

        for j in range(price_paths.shape[0]):
            final_price: float = float(price_paths[j, -1])
            final_prices[j] = final_price
            option_payoffs[j] = calculate_european_call_option_payoff(STRIKE_PRICE, final_price) / (1 + RISK_FREE_RATE)
        pricings.append((option_payoffs, final_prices))
    return pricings

if __name__ == '__main__':
    main()
