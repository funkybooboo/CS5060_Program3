import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Callable

# What datasets to predict
FILE_PATHS: List[str] = ['../data/stock1.csv', '../data/stock2.csv']
STOCK_NAMES: List[str] = ['Stock1', 'Stock2']

# What distributions to use in the simulation
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
    'norm': lambda params: lambda: np.random.normal(params[0], params[1]) - params[0],
    'lognorm': lambda params: lambda: (np.random.lognormal(mean=np.log(params[2]), sigma=params[0]) - params[2]),
    'beta': lambda params: lambda: params[2] + (np.random.beta(params[0], params[1])) * params[3],
}

# Simulation accuracy
NUM_PATHS: int = 5000

# Option contract info
TIME_INCREMENT: float = 1 / 365  # Daily increments
TOTAL_TIME: float = 1.0  # 1 year
STRIKE_PRICE: float = 100.0
RISK_FREE_RATE: float = 0.01

# TODO
#   1. Combine multiple graphs into one graph for part 3
#   2. Fix Scenario logic

def main() -> None:
    stock_distributions: Dict[str, Dict[str, Any]] = {}

    # Part 1
    for file_path, stock_name in zip(FILE_PATHS, STOCK_NAMES):
        stock_distribution = get_stock_distribution(file_path, stock_name)
        if not stock_distribution:
            print(f"{stock_name}: could not get best_fitted_distribution")
            continue  # Process all stocks even if one fails
        stock_distributions[stock_name] = stock_distribution

    # Part 2
    simulate_stock_and_plot({
        "Stock0": {
            'name': "norm", 
            "generator": lambda: np.random.normal(0, np.sqrt(TIME_INCREMENT)),
            "initial_stock_price": 100.0,
            'drift_rate': 0.03,
            'volatility_rate': 17.04
        }
    })

    simulate_stock_and_plot({
            "Stock0": {
                "name": "beta", 
                "generator": lambda: np.random.beta(9, 10) - 0.35,
                "initial_stock_price": 100.0,
                'drift_rate': 0.03,
                'volatility_rate': 17.04
        }
    })

    ## Part 3
    simulate_stock_and_plot(stock_distributions)

def get_stock_distribution(file_path: str, stock_name: str) -> Optional[Dict[str, Callable]]:
    non_normalized_data = load_data(file_path)

    if non_normalized_data is None or non_normalized_data.empty:
        return None

    normalized_data = normalize_data(non_normalized_data)
    number_of_bins = determine_number_of_bins(normalized_data)

    plot_prices(normalized_data, stock_name, number_of_bins)

    fits = fit_distributions(normalized_data, non_normalized_data)
    plot_fitted_distributions(normalized_data, fits)
    plt.show()

    return get_best_fitted_stock_distribution(normalized_data, non_normalized_data, fits, stock_name)

def get_best_fitted_stock_distribution(normalized_data, non_normalized_data: pd.Series, fits: Dict[str, Dict[str, Tuple]], stock_name: str) -> Optional[Dict[str, Any]]:
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

    if non_normalized_best_distribution == "beta":
        log_returns = np.log(normalized_data / normalized_data.shift(1)).dropna()
    else:
        log_returns = np.log(non_normalized_data / non_normalized_data.shift(1)).dropna()
    
    drift = log_returns.mean() * 365  # Annualized drift based on all days
    volatility = log_returns.std() * np.sqrt(365)  # Annualized volatility based on all days
    
    stock_distribution = {
        'name': non_normalized_best_distribution,
        'generator': GENERATOR[non_normalized_best_distribution](fits[non_normalized_best_distribution]['non_normalized']),
        'initial_stock_price': non_normalized_data[non_normalized_data.size - 1],
        'drift_rate': drift,
        'volatility_rate': volatility,
    }
    
    return stock_distribution

def load_data(file_path: str) -> Optional[pd.Series]:
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
    data_min: float = np.min(data)
    data_max: float = np.max(data)
    normalized_data = (data - data_min + 1e-9) / (data_max - data_min + 1e-9)
    normalized_data = normalized_data[(normalized_data != 0) & (normalized_data != 1)]
    return normalized_data

def determine_number_of_bins(data: pd.Series) -> int:
    n: int = len(data)
    if n < 10:
        return 5  # Minimum number of bins
    elif n < 50:
        return int(np.sqrt(n))  # Square root choice
    else:
        return int(np.log2(n) + 1)  # Sturges' formula

def plot_prices(normalized_data: pd.Series, stock_name: str, number_of_bins: int) -> None:
    plt.hist(normalized_data, density=True, bins=number_of_bins, alpha=0.5, color='g')
    plt.title(f'Histogram of {stock_name}')
    plt.xlabel('Normalized Price')
    plt.ylabel('Density')

def fit_distributions(normalized_data: pd.Series, non_normalized_data: pd.Series) -> Dict[str, Dict[str, Tuple]]:
    fits: Dict[str, Dict[str, Tuple]] = {}

    for name, distribution in DISTRIBUTIONS.items():
        fits[name] = {}

        try:
            # Fit to normalized data
            fits[name]['normalized'] = distribution.fit(normalized_data)
        except Exception as e:
            print(f"Fitting {name} distribution to normalized data failed: {e}")
            raise  # Stop execution on exception

        try:
            # Fit to non-normalized data
            if name == "beta":
                fits[name]['non_normalized'] = distribution.fit(normalized_data)
            else:
                fits[name]['non_normalized'] = distribution.fit(non_normalized_data)
        except Exception as e:
            print(f"Fitting {name} distribution to non-normalized data failed: {e}")
            raise  # Stop execution on exception

    return fits

def plot_fitted_distributions(normalized_data: pd.Series, fits: Dict[str, Dict[str, Tuple]]) -> None:
    x: np.ndarray = np.linspace(min(normalized_data), max(normalized_data), 100)

    # Loop through the fitted distributions and plot them
    for dist_name, params in fits.items():
        if dist_name in COLORS:
            plt.plot(x, DISTRIBUTIONS[dist_name].pdf(x, *params['normalized']), color=COLORS[dist_name], label=f'{dist_name.capitalize()} Fit (Normalized)')

    plt.legend()

def goodness_of_fit(data: pd.Series, fits: Dict[str, Dict[str, Tuple]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for dist, params in fits.items():
        results[dist] = ss.kstest(data, dist, args=params['non_normalized'])
    return results

def simulate_stock_and_plot(stock_distributions: Dict[str, Dict[str, Any]]) -> None:
    stock_names: List[str] = list(stock_distributions.keys())

    simulation_name, stocks_price_paths = get_stocks_price_paths(stock_distributions, stock_names)

    pricings: List[Tuple[np.ndarray, np.ndarray]] = calculate_basket_option_pricing(stocks_price_paths)

    plot_stock_predictions(stock_distributions, stock_names, stocks_price_paths)

    average_final_price = get_average_final_price(pricings, stock_names)

    max_final_prices = get_max_final_prices(pricings, stock_names)

    outperforms_average: bool = get_outperforms_average(average_final_price, pricings, stock_names)

    outperforms_max: bool = get_outperforms_max(max_final_prices, pricings, stock_names) 

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
        print(f"Max stock price after {int(1 / TIME_INCREMENT) * TOTAL_TIME} days: ${np.average(max_final_prices):.2f}")
        print(f"Max payoff for a block of 100 options: ${basket_option_payoffs * 100:.2f}")
        print(f"Estimated cost of the option: ${basket_option_payoffs:.2f}")
    else:
        print("Did not outperform max, no payoff.")
    print()

def get_stocks_price_paths(stock_distributions, stock_names):
    simulation_name: str = ""
    stocks_price_paths: List[np.ndarray] = []
    for i in range(len(stock_names)):
        stock_name: str = stock_names[i]
        distribution_name = stock_distributions[stock_name]["name"]
        simulation_name += f"{stock_name} {distribution_name}\n"

        random_generator: Callable[[], float] = stock_distributions[stock_name]['generator']
        initial_stock_price = stock_distributions[stock_name]['initial_stock_price']
        drift_rate = stock_distributions[stock_name]['drift_rate']
        volatility_rate = stock_distributions[stock_name]['volatility_rate']
        stock_price_paths: np.ndarray = generate_stock_price_paths(random_generator, initial_stock_price, drift_rate, volatility_rate)
        stocks_price_paths.append(stock_price_paths)
    return simulation_name, stocks_price_paths

def get_basket_option_payoffs(pricings, stock_names):
    basket_option_payoffs: float = 0
    for i in range(len(stock_names)):
        option_payoffs = pricings[i][0]
        average_option_payoffs = np.average(option_payoffs)
        basket_option_payoffs += average_option_payoffs
    basket_option_payoffs /= len(stock_names)
    return basket_option_payoffs

def get_outperforms_average(average_final_price, pricings, stock_names):
    outperforms_average: bool = True
    for i in range(len(stock_names)):
        option_payoffs = pricings[i][0]
        average_option_payoffs = np.average(option_payoffs)
        if average_final_price > average_option_payoffs:
            outperforms_average = False
            break
    return outperforms_average

def get_outperforms_max(max_final_prices, pricings, stock_names):
    outperforms_max: bool = True
    for i in range(len(stock_names)):
        option_payoffs = pricings[i][0]
        average_option_payoffs = np.average(option_payoffs)
        for max_final_price in max_final_prices:
            if max_final_price >= average_option_payoffs:
                outperforms_max = False
                break
    return outperforms_max

def get_max_final_prices(pricings, stock_names):
    max_final_prices: List[float] = []
    for i in range(len(stock_names)):
        final_price = max(pricings[i][1])
        max_final_prices.append(final_price)
    return max_final_prices

def get_average_final_price(pricings, stock_names):
    average_final_price: float = 0
    for i in range(len(stock_names)):
        final_prices = pricings[i][1]
        average_final_price += np.sum(final_prices)
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
        random_generator, initial_stock_price, drift_rate, volatility_rate
) -> np.ndarray:
    price_paths: np.ndarray = np.zeros((NUM_PATHS, int(TOTAL_TIME / TIME_INCREMENT)))
    for i in range(NUM_PATHS):
        current_price: float = initial_stock_price
        for t in range(int(TOTAL_TIME / TIME_INCREMENT)):
            price_change: float = drift_rate * TIME_INCREMENT + volatility_rate * random_generator()
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
