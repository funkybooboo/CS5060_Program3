import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Callable

from scipy.linalg import solve_toeplitz

FILE_PATHS: List[str] = ['../data/stock1.csv', '../data/stock2.csv']
STOCK_NAMES: List[str] = ['Stock1', 'Stock2']

DISTRIBUTIONS: Dict[str, Dict[str, Any]] = {
    'norm': {
        'name': 'Normal',
        'distribution_object': ss.norm,
        'color': 'red',
        'generator': lambda params: lambda: np.random.normal(params[0], params[1]) - params[0],
    },
    'lognorm': {
        'name': 'Log-Normal',
        'distribution_object': ss.lognorm,
        'color': 'blue',
        'generator': lambda params: lambda: (np.random.lognormal(mean=np.log(params[2]), sigma=params[0]) - params[2]),
    },
    'beta': {
        'name': 'Beta',
        'distribution_object': ss.beta,
        'color': 'green',
        'generator': lambda params: lambda: params[2] + (np.random.beta(params[0], params[1])) * params[3],
    },
}

NUM_SIMULATION_PATHS: int = 5000
TIME_INCREMENT: float = 1 / 365  # Daily increments
TOTAL_TIME_YEARS: float = 1.0  # 1 year
OPTION_STRIKE_PRICE: float = 100.0
RISK_FREE_RATE: float = 0.01

def main() -> None:
    stock_fitted_distributions: Dict[str, Dict[str, Any]] = {
        "simulation_title": "",
    }

    # Part 1
    for file_path, stock_name in zip(FILE_PATHS, STOCK_NAMES):
        fitted_distribution = get_fitted_distribution_from_stock_data(file_path, stock_name)
        if not fitted_distribution:
            print(f"{stock_name}: could not get best_fitted_distribution")
            continue  # Process all stocks even if one fails
        stock_fitted_distributions[stock_name] = fitted_distribution

    # Part 2
    simulate_and_plot_stock({
        "simulation_title": "",
        "Stock0": {
            'name': "Normal",
            "generator": lambda: np.random.normal(0, np.sqrt(TIME_INCREMENT)),
            "initial_stock_price": 100.0,
            'drift_rate': 0.03,
            'volatility_rate': 17.04
        }
    })

    simulate_and_plot_stock({
        "simulation_title": "",
        "Stock0": {
            "name": "Beta",
            "generator": lambda: np.random.beta(9, 10) - 0.35,
            "initial_stock_price": 100.0,
            'drift_rate': 0.03,
            'volatility_rate': 17.04
        }
    })

    # Part 3
    simulate_and_plot_stock(stock_fitted_distributions)

def get_fitted_distribution_from_stock_data(file_path: str, stock_name: str) -> Optional[Dict[str, Callable]]:
    raw_stock_data: Optional[pd.Series] = load_stock_data(file_path)

    if raw_stock_data is None or raw_stock_data.empty:
        return None

    normalized_stock_data: pd.Series = normalize_stock_data(raw_stock_data)
    num_bins: int = calculate_number_of_bins(normalized_stock_data)

    plot_stock_prices(normalized_stock_data, stock_name, num_bins)

    fitted_distributions: Dict[str, Dict[str, Tuple]] = fit_distribution_models(normalized_stock_data, raw_stock_data)
    plot_fitted_distribution_models(normalized_stock_data, fitted_distributions)
    plt.show()

    return select_best_fitted_distribution(normalized_stock_data, raw_stock_data, fitted_distributions, stock_name)

def select_best_fitted_distribution(normalized_data: pd.Series, raw_data: pd.Series, fitted_distributions: Dict[str, Dict[str, Tuple]], stock_name: str) -> Optional[Dict[str, Any]]:
    goodness_of_fit_results: Dict[str, Any] = calculate_goodness_of_fit(raw_data, fitted_distributions)

    best_fit_result = max(goodness_of_fit_results.values(), key=lambda x: x.pvalue)
    best_fit_distribution_name: str = [dist for dist, result in goodness_of_fit_results.items() if result == best_fit_result][0]

    print(f'Stock Name: {stock_name}')
    for dist, result in goodness_of_fit_results.items():
        statistic: float = result.statistic
        p_value: float = result.pvalue
        print(f'\t{DISTRIBUTIONS[dist]["name"]} Fit: Statistic={statistic:.4f}, p-value={p_value:.4f}')

        if dist != best_fit_distribution_name:
            diff_statistic: float = statistic - best_fit_result.statistic
            diff_p_value: float = p_value - best_fit_result.pvalue
            print(f'\t\tDifference from Best Fit: Statistic Diff={diff_statistic:.4f}, p-value Diff={diff_p_value:.4f}')
        else:
            print('\t\tBest Fit')

    if best_fit_distribution_name == "beta":
        log_returns: pd.Series = np.log(normalized_data / normalized_data.shift(1)).dropna()
    else:
        log_returns: pd.Series = np.log(raw_data / raw_data.shift(1)).dropna()

    drift: float = log_returns.mean() * 365  # Annualized drift based on all days
    volatility: float = log_returns.std() * np.sqrt(365)  # Annualized volatility based on all days

    fitted_distribution_info: Dict[str, Any] = {
        'name': best_fit_distribution_name,
        'generator': DISTRIBUTIONS[best_fit_distribution_name]['generator'](fitted_distributions[best_fit_distribution_name]['non_normalized']),
        'initial_stock_price': raw_data.iloc[-1],
        'drift_rate': drift,
        'volatility_rate': volatility,
    }

    return fitted_distribution_info

def load_stock_data(file_path: str) -> Optional[pd.Series]:
    try:
        df: pd.DataFrame = pd.read_csv(file_path)
        return df['value']
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except KeyError:
        print(f"'value' column not found in {file_path}.")
        return None

def normalize_stock_data(raw_data: pd.Series) -> pd.Series:
    data_min: float = np.min(raw_data)
    data_max: float = np.max(raw_data)
    normalized_data: pd.Series = (raw_data - data_min + 1e-9) / (data_max - data_min + 1e-9)
    normalized_data = normalized_data[(normalized_data != 0) & (normalized_data != 1)]
    return normalized_data

def calculate_number_of_bins(normalized_data: pd.Series) -> int:
    n: int = len(normalized_data)
    if n < 10:
        return 5  # Minimum number of bins
    elif n < 50:
        return int(np.sqrt(n))  # Square root choice
    else:
        return int(np.log2(n) + 1)  # Sturges' formula

def plot_stock_prices(normalized_data: pd.Series, stock_name: str, num_bins: int) -> None:
    plt.hist(normalized_data, density=True, bins=num_bins, alpha=0.5, color='g')
    plt.title(f'Histogram of {stock_name}')
    plt.xlabel('Normalized Price')
    plt.ylabel('Density')

def fit_distribution_models(normalized_data: pd.Series, raw_data: pd.Series) -> Dict[str, Dict[str, Tuple]]:
    fitted_models: Dict[str, Dict[str, Tuple]] = {}

    for distribution_name, distribution in DISTRIBUTIONS.items():
        fitted_models[distribution_name] = {}

        try:
            # Fit to normalized data
            fitted_models[distribution_name]['normalized'] = distribution['distribution_object'].fit(normalized_data)
        except Exception as e:
            print(f"Fitting {distribution_name} distribution to normalized data failed: {e}")
            raise  # Stop execution on exception

        try:
            # Fit to raw data
            if distribution_name == "beta":
                fitted_models[distribution_name]['non_normalized'] = distribution['distribution_object'].fit(normalized_data)
            else:
                fitted_models[distribution_name]['non_normalized'] = distribution['distribution_object'].fit(raw_data)
        except Exception as e:
            print(f"Fitting {distribution_name} distribution to raw data failed: {e}")
            raise  # Stop execution on exception

    return fitted_models

def plot_fitted_distribution_models(normalized_data: pd.Series, fitted_models: Dict[str, Dict[str, Tuple]]) -> None:
    x: np.ndarray = np.linspace(min(normalized_data), max(normalized_data), 100)

    # Loop through the fitted distributions and plot them
    for distribution_name, params in fitted_models.items():
        if distribution_name in DISTRIBUTIONS:
            plt.plot(x, DISTRIBUTIONS[distribution_name]['distribution_object'].pdf(x, *params['normalized']),
                     color=DISTRIBUTIONS[distribution_name]['color'], label=f'{DISTRIBUTIONS[distribution_name]["name"]} Fit (Normalized)')

    plt.legend()

def calculate_goodness_of_fit(raw_data: pd.Series, fitted_models: Dict[str, Dict[str, Tuple]]) -> Dict[str, Any]:
    goodness_of_fit_results: Dict[str, Any] = {}
    for distribution_name, params in fitted_models.items():
        goodness_of_fit_results[distribution_name] = ss.kstest(raw_data, distribution_name, args=params['non_normalized'])
    return goodness_of_fit_results

def simulate_and_plot_stock(stock_distributions: Dict[str, str | Dict[str, Any]]) -> None:
    simulation_title: str = stock_distributions['simulation_title']
    stock_names: List[str] = [name for name in stock_distributions.keys() if name != 'simulation_title']

    stocks_price_paths: List[np.ndarray] = generate_stock_price_paths(stock_distributions, stock_names)

    option_pricing_data: List[Tuple[np.ndarray, np.ndarray]] = calculate_basket_option_pricing(stocks_price_paths)

    plot_simulated_stock_predictions(stock_distributions, stock_names, stocks_price_paths)

    avg_final_price: float = calculate_average_final_price(option_pricing_data, stock_names)

    max_final_prices: List[float] = calculate_max_final_prices(option_pricing_data, stock_names)

    average_outperform: bool = check_if_average_outperformed(avg_final_price, option_pricing_data, stock_names)

    max_outperform: bool = check_if_max_outperformed(max_final_prices, option_pricing_data, stock_names)

    basket_option_payoffs: float = calculate_basket_option_payoffs(option_pricing_data, stock_names)

    print(f" ---\n {simulation_title} \n--- ")
    print("Scenario 1")
    if average_outperform:
        print(f"Average stock price after {int(1 / TIME_INCREMENT) * TOTAL_TIME_YEARS} days: ${avg_final_price:.2f}")
        print(f"Average payoff for a block of 100 options: ${basket_option_payoffs * 100:.2f}")
        print(f"Estimated cost of the option: ${basket_option_payoffs:.2f}")
    else:
        print("Did not outperform average, no payoff.")
    print()
    print("Scenario 2:")
    if max_outperform:
        print(f"Max stock price after {int(1 / TIME_INCREMENT) * TOTAL_TIME_YEARS} days: ${np.average(max_final_prices):.2f}")
        print(f"Max payoff for a block of 100 options: ${basket_option_payoffs * 100:.2f}")
        print(f"Estimated cost of the option: ${basket_option_payoffs:.2f}")
    else:
        print("Did not outperform max, no payoff.")
    print()

def generate_stock_price_paths(stock_distributions: Dict[str, Dict[str, Any]], stock_names: List[str]) -> List[np.ndarray]:
    stocks_price_paths: List[np.ndarray] = []
    for i in range(len(stock_names)):
        stock_name: str = stock_names[i]
        random_generator: Callable[[], float] = stock_distributions[stock_name]['generator']
        initial_stock_price: float = stock_distributions[stock_name]['initial_stock_price']
        drift_rate: float = stock_distributions[stock_name]['drift_rate']
        volatility_rate: float = stock_distributions[stock_name]['volatility_rate']
        stock_price_paths: np.ndarray = create_stock_price_paths(random_generator, initial_stock_price, drift_rate, volatility_rate)
        stocks_price_paths.append(stock_price_paths)
    return stocks_price_paths

def calculate_basket_option_payoffs(option_pricing_data: List[Tuple[np.ndarray, np.ndarray]], stock_names: List[str]) -> float:
    total_basket_option_payoffs: float = 0
    for i in range(len(stock_names)):
        option_payoffs: np.ndarray = option_pricing_data[i][0]
        avg_option_payoffs: float = float(np.average(option_payoffs))
        total_basket_option_payoffs += avg_option_payoffs
    total_basket_option_payoffs /= len(stock_names)
    return total_basket_option_payoffs

def check_if_average_outperformed(avg_final_price: float, option_pricing_data: List[Tuple[np.ndarray, np.ndarray]], stock_names: List[str]) -> bool:
    average_outperform: bool = True
    for i in range(len(stock_names)):
        option_payoffs: np.ndarray = option_pricing_data[i][0]
        avg_option_payoffs: float = float(np.average(option_payoffs))
        if avg_final_price > avg_option_payoffs:
            average_outperform = False
            break
    return average_outperform

def check_if_max_outperformed(max_final_prices: List[float], option_pricing_data: List[Tuple[np.ndarray, np.ndarray]], stock_names: List[str]) -> bool:
    max_outperform: bool = True
    for i in range(len(stock_names)):
        option_payoffs: np.ndarray = option_pricing_data[i][0]
        avg_option_payoffs: float = float(np.average(option_payoffs))
        for max_final_price in max_final_prices:
            if max_final_price >= avg_option_payoffs:
                max_outperform = False
                break
    return max_outperform

def calculate_max_final_prices(option_pricing_data: List[Tuple[np.ndarray, np.ndarray]], stock_names: List[str]) -> List[float]:
    max_final_prices: List[float] = []
    for i in range(len(stock_names)):
        final_price: float = max(option_pricing_data[i][1])
        max_final_prices.append(final_price)
    return max_final_prices

def calculate_average_final_price(option_pricing_data: List[Tuple[np.ndarray, np.ndarray]], stock_names: List[str]) -> float:
    total_avg_final_price: float = 0
    for i in range(len(stock_names)):
        final_prices: np.ndarray = option_pricing_data[i][1]
        total_avg_final_price += np.sum(final_prices)
    total_avg_final_price /= (NUM_SIMULATION_PATHS * len(stock_names))
    return total_avg_final_price

def plot_simulated_stock_predictions(stock_distributions: Dict[str, Dict[str, Any]], stock_names: List[str], stocks_price_paths: List[np.ndarray]) -> None:
    for i in range(len(stock_names)):
        distribution_name: str = stock_distributions[stock_names[i]]['name']
        # Plotting stock price paths
        plt.figure(figsize=(12, 6))
        for j in range(min(NUM_SIMULATION_PATHS, 10)):
            plt.plot(stocks_price_paths[i][j])
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title(f"Simulated {stock_names[i]} Price Paths Using {distribution_name} Distribution")
        plt.show()

def create_stock_price_paths(
        random_generator: Callable[[], float], initial_stock_price: float, drift_rate: float, volatility_rate: float
) -> np.ndarray:
    price_paths: np.ndarray = np.zeros((NUM_SIMULATION_PATHS, int(TOTAL_TIME_YEARS / TIME_INCREMENT)))
    for i in range(NUM_SIMULATION_PATHS):
        current_price: float = initial_stock_price
        for t in range(int(TOTAL_TIME_YEARS / TIME_INCREMENT)):
            price_change: float = drift_rate * TIME_INCREMENT + volatility_rate * random_generator()
            current_price += price_change
            price_paths[i, t] = current_price
    return price_paths

def calculate_european_call_option_payoff(strike_price: float, final_stock_price: float) -> float:
    return max(final_stock_price - strike_price, 0)

def calculate_basket_option_pricing(
        stocks_price_paths: List[np.ndarray]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    pricing_data: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(stocks_price_paths)):
        price_paths: np.ndarray = stocks_price_paths[i]
        option_payoffs: np.ndarray = np.zeros(price_paths.shape[0])
        final_prices: np.ndarray = np.zeros(price_paths.shape[0])

        for j in range(price_paths.shape[0]):
            final_price: float = float(price_paths[j, -1])
            final_prices[j] = final_price
            option_payoffs[j] = calculate_european_call_option_payoff(OPTION_STRIKE_PRICE, final_price) / (1 + RISK_FREE_RATE)
        pricing_data.append((option_payoffs, final_prices))
    return pricing_data

if __name__ == '__main__':
    main()
