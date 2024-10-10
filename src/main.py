import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Callable

# File paths for stock data
FILE_PATHS: List[str] = ['../data/stock1.csv', '../data/stock2.csv']
# Names corresponding to the stock data
STOCK_NAMES: List[str] = ['Stock1', 'Stock2']

# Dictionary defining distributions and their properties
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

# Simulation parameters
NUM_SIMULATION_PATHS: int = 5000
TIME_INCREMENT: float = 1 / 365  # Daily increments
TOTAL_TIME_YEARS: float = 1.0  # Total duration of simulation in years
OPTION_STRIKE_PRICE: float = 100.0  # Strike price for the option
RISK_FREE_RATE: float = 0.01  # Risk-free interest rate

def main() -> None:
    # Dictionary to store fitted distributions for each stock
    stock_fitted_distributions: Dict[str, str | Dict[str, Any]] = {
        "simulation_title": "Predicting the future with past data",
    }

    # Part 1: Fit distributions to stock data
    for file_path, stock_name in zip(FILE_PATHS, STOCK_NAMES):
        fitted_distribution = get_fitted_distribution_from_stock_data(file_path, stock_name)
        if not fitted_distribution:
            print(f"{stock_name}: could not get best_fitted_distribution")
            continue  # Continue processing all stocks even if one fails
        stock_fitted_distributions[stock_name] = fitted_distribution

    # Part 2: Simulate and plot stocks with defined distributions
    simulate_and_plot_stock({
        "simulation_title": "Brownian Motion",
        "Stock0": {
            'name': "Normal",
            "generator": lambda: np.random.normal(0, np.sqrt(TIME_INCREMENT)),
            "initial_stock_price": 100.0,
            'drift_rate': 0.03,
            'volatility_rate': 17.04
        }
    })

    simulate_and_plot_stock({
        "simulation_title": "Weird Beta",
        "Stock0": {
            "name": "Beta",
            "generator": lambda: np.random.beta(9, 10) - 0.35,
            "initial_stock_price": 100.0,
            'drift_rate': 0.03,
            'volatility_rate': 17.04
        }
    })

    # Part 3: Simulate and plot stocks based on fitted distributions
    simulate_and_plot_stock(stock_fitted_distributions)

def get_fitted_distribution_from_stock_data(file_path: str, stock_name: str) -> Optional[Dict[str, Callable]]:
    # Load stock data from the provided file path
    raw_stock_data: Optional[pd.Series] = load_stock_data(file_path)

    if raw_stock_data is None or raw_stock_data.empty:
        return None  # Return None if data is not loaded properly

    # Normalize the stock data for analysis
    normalized_stock_data: pd.Series = normalize_stock_data(raw_stock_data)
    num_bins: int = calculate_number_of_bins(normalized_stock_data)  # Calculate bins for histogram

    # Plot normalized stock prices
    plot_stock_prices(normalized_stock_data, stock_name, num_bins)

    # Fit distribution models to the normalized and raw data
    fitted_distributions: Dict[str, Dict[str, Tuple]] = fit_distribution_models(normalized_stock_data, raw_stock_data)
    plot_fitted_distribution_models(normalized_stock_data, fitted_distributions)  # Plot fitted distributions
    plt.show()

    # Select the best-fitted distribution based on goodness of fit
    return select_best_fitted_distribution(normalized_stock_data, raw_stock_data, fitted_distributions, stock_name)

def select_best_fitted_distribution(normalized_data: pd.Series, raw_data: pd.Series, fitted_distributions: Dict[str, Dict[str, Tuple]], stock_name: str) -> Optional[Dict[str, Any]]:
    # Calculate goodness of fit statistics for fitted distributions
    goodness_of_fit_results: Dict[str, Any] = calculate_goodness_of_fit(normalized_data, raw_data, fitted_distributions)

    # Identify the best-fitting distribution based on p-value
    best_fit_result = max(goodness_of_fit_results.values(), key=lambda x: x.pvalue)
    best_fit_distribution_name: str = [dist for dist, result in goodness_of_fit_results.items() if result == best_fit_result][0]

    print(f'Stock Name: {stock_name}')
    for dist, result in goodness_of_fit_results.items():
        statistic: float = result.statistic  # Goodness of fit statistic
        p_value: float = result.pvalue  # p-value of the fit
        print(f'\t{DISTRIBUTIONS[dist]["name"]} Fit: Statistic={statistic:.4f}, p-value={p_value:.4f}')

        # Compare with the best fit distribution
        if dist != best_fit_distribution_name:
            diff_statistic: float = statistic - best_fit_result.statistic
            diff_p_value: float = p_value - best_fit_result.pvalue
            print(f'\t\tDifference from Best Fit: Statistic Diff={diff_statistic:.4f}, p-value Diff={diff_p_value:.4f}')
        else:
            print('\t\tBest Fit')

    # Calculate drift and volatility based on log returns
    if best_fit_distribution_name == "beta":
        log_returns: pd.Series = np.log(normalized_data / normalized_data.shift(1)).dropna()
    else:
        log_returns: pd.Series = np.log(raw_data / raw_data.shift(1)).dropna()

    drift: float = log_returns.mean() * 365  # Annualized drift based on daily returns
    volatility: float = log_returns.std() * np.sqrt(365)  # Annualized volatility

    # Prepare the fitted distribution info for simulation
    fitted_distribution_info: Dict[str, Any] = {
        'name': best_fit_distribution_name,
        'generator': DISTRIBUTIONS[best_fit_distribution_name]['generator'](fitted_distributions[best_fit_distribution_name]['non_normalized']),
        'initial_stock_price': raw_data.iloc[-1],
        'drift_rate': drift,
        'volatility_rate': volatility,
    }

    return fitted_distribution_info

def load_stock_data(file_path: str) -> Optional[pd.Series]:
    # Load the stock data from a CSV file
    try:
        df: pd.DataFrame = pd.read_csv(file_path)
        return df['value']  # Return the 'value' column
    except FileNotFoundError:
        print(f"File {file_path} not found.")  # Handle file not found error
        return None
    except KeyError:
        print(f"'value' column not found in {file_path}.")  # Handle missing column error
        return None

def normalize_stock_data(raw_data: pd.Series) -> pd.Series:
    # Normalize the stock data to a 0-1 range
    data_min: float = np.min(raw_data)
    data_max: float = np.max(raw_data)
    normalized_data: pd.Series = (raw_data - data_min + 1e-9) / (data_max - data_min + 1e-9)  # Normalize with a small epsilon
    normalized_data = normalized_data[(normalized_data != 0) & (normalized_data != 1)]  # Exclude boundaries
    return normalized_data

def calculate_number_of_bins(normalized_data: pd.Series) -> int:
    # Calculate the number of bins for the histogram
    n: int = len(normalized_data)
    if n < 10:
        return 5  # Minimum number of bins
    elif n < 50:
        return int(np.sqrt(n))  # Square root choice for bins
    else:
        return int(np.log2(n) + 1)  # Sturges' formula for bins

def plot_stock_prices(normalized_data: pd.Series, stock_name: str, num_bins: int) -> None:
    # Plot histogram of normalized stock prices
    plt.hist(normalized_data, density=True, bins=num_bins, alpha=0.5, color='g')
    plt.title(f'Histogram of {stock_name}')  # Title of the plot
    plt.xlabel('Normalized Price')  # X-axis label
    plt.ylabel('Density')  # Y-axis label

def fit_distribution_models(normalized_data: pd.Series, raw_data: pd.Series) -> Dict[str, Dict[str, Tuple]]:
    # Fit various distribution models to both normalized and raw stock data
    fitted_models: Dict[str, Dict[str, Tuple]] = {}

    for distribution_name, distribution in DISTRIBUTIONS.items():
        fitted_models[distribution_name] = {}

        try:
            # Fit distribution to normalized data
            fitted_models[distribution_name]['normalized'] = distribution['distribution_object'].fit(normalized_data)
        except Exception as e:
            print(f"Fitting {distribution_name} distribution to normalized data failed: {e}")
            raise  # Stop execution on exception

        try:
            # Fit distribution to raw data
            if distribution_name == "beta":
                fitted_models[distribution_name]['non_normalized'] = distribution['distribution_object'].fit(normalized_data)
            else:
                fitted_models[distribution_name]['non_normalized'] = distribution['distribution_object'].fit(raw_data)
        except Exception as e:
            print(f"Fitting {distribution_name} distribution to raw data failed: {e}")
            raise  # Stop execution on exception

    return fitted_models

def plot_fitted_distribution_models(normalized_data: pd.Series, fitted_models: Dict[str, Dict[str, Tuple]]) -> None:
    # Plot fitted distribution models over normalized data
    x: np.ndarray = np.linspace(min(normalized_data), max(normalized_data), 100)

    for distribution_name, params in fitted_models.items():
        if distribution_name in DISTRIBUTIONS:
            # Plot the PDF of the fitted distribution
            plt.plot(x, DISTRIBUTIONS[distribution_name]['distribution_object'].pdf(x, *params['normalized']),
                     color=DISTRIBUTIONS[distribution_name]['color'], label=f'{DISTRIBUTIONS[distribution_name]["name"]} Fit (Normalized)')

    plt.legend()  # Show legend for the plot

def calculate_goodness_of_fit(normalized_data: pd.Series, raw_data: pd.Series, fitted_models: Dict[str, Dict[str, Tuple]]) -> Dict[str, Any]:
    # Calculate goodness of fit statistics for each fitted model
    goodness_of_fit_results: Dict[str, Any] = {}
    for distribution_name, params in fitted_models.items():
        if distribution_name == "beta":
            goodness_of_fit_results[distribution_name] = ss.kstest(normalized_data, distribution_name, args=params['non_normalized'])
        else:
            goodness_of_fit_results[distribution_name] = ss.kstest(raw_data, distribution_name, args=params['non_normalized'])
    return goodness_of_fit_results

def simulate_and_plot_stock(stock_distributions: Dict[str, str | Dict[str, Any]]) -> None:
    # Simulate stock price paths and plot results
    simulation_title: str = stock_distributions['simulation_title']
    print(f"--- {simulation_title} ---")
    stock_names: List[str] = [name for name in stock_distributions.keys() if name != 'simulation_title']

    # Generate simulated stock price paths
    stocks_price_paths: List[np.ndarray] = generate_stocks_price_paths(stock_distributions, stock_names)
    
    # Plot the simulated stock predictions
    plot_simulated_stock_predictions(stock_distributions, stock_names, stocks_price_paths)

    # Calculate basket option pricing data based on simulated paths
    option_pricing_data: List[Tuple[np.ndarray, np.ndarray]] = calculate_basket_option_pricing(stocks_price_paths)

    print_stats(option_pricing_data, stock_names)
    print()

def print_stats(option_pricing_data, stock_names):
    print("Average Basket Option\n")

    # Initialize lists to store averages for stock prices and payoffs
    average_stock_prices = []
    average_payoffs = []
    min_stock_prices = []
    max_stock_prices = []
    std_dev_stock_prices = []
    paths_exceeding_strike = []

    for i, (option_payoffs, final_prices) in enumerate(option_pricing_data):
        average_stock_price = np.average(final_prices)
        average_payoff = np.average(option_payoffs)
        min_stock_price = np.min(final_prices)
        max_stock_price = np.max(final_prices)
        std_dev_stock_price = np.std(final_prices)
        count_exceeding_strike = np.sum(final_prices > OPTION_STRIKE_PRICE)

        average_stock_prices.append(average_stock_price)
        average_payoffs.append(average_payoff)
        min_stock_prices.append(min_stock_price)
        max_stock_prices.append(max_stock_price)
        std_dev_stock_prices.append(std_dev_stock_price)
        paths_exceeding_strike.append(count_exceeding_strike)

        print(f"Stock: {stock_names[i]}")
        print(f"\tAverage stock price after {int(1 / TIME_INCREMENT) * TOTAL_TIME_YEARS} days: ${average_stock_price:.2f}")
        print(f"\tMinimum stock price: ${min_stock_price:.2f}")
        print(f"\tMaximum stock price: ${max_stock_price:.2f}")
        print(f"\tStandard Deviation of stock prices: ${std_dev_stock_price:.2f}")
        print(f"\tCount of paths exceeding strike price: {count_exceeding_strike}")
        print(f"\tAverage payoff (option block of 100): ${average_payoff * 100:.2f}")
        print(f"\tCost of option: ${average_payoff:.2f}\n")

    overall_average_stock_price = np.mean(average_stock_prices)
    overall_average_payoff = np.mean(average_payoffs)

    print(f"Overall Average Stock Price: ${overall_average_stock_price:.2f}")
    print(f"Overall Average Basket Option Payoff: ${overall_average_payoff:.2f}\n")

    print("Max Basket Option\n")

    # Calculate maximum option price for each stock
    for i, (option_payoffs, _) in enumerate(option_pricing_data):
        max_payoff = np.max(option_payoffs)
        print(f"Stock: {stock_names[i]} - Max Payoff: ${max_payoff:.2f}")

    overall_max_payoff = np.max([np.max(option_payoffs) for option_payoffs, _ in option_pricing_data])
    print(f"Overall Max Basket Option Payoff: ${overall_max_payoff:.2f}")

def calculate_basket_option_pricing(
        stocks_price_paths: List[np.ndarray]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Calculate the basket option pricing based on the simulated stock paths
    option_pricing_data: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(stocks_price_paths)):
        price_paths: np.ndarray = stocks_price_paths[i]  # Get the price paths for the current stock
        option_payoffs: np.ndarray = np.zeros(price_paths.shape[0])  # Initialize option payoffs array
        final_prices: np.ndarray = np.zeros(price_paths.shape[0])  # Initialize final prices array

        for j in range(price_paths.shape[0]):
            final_price: float = float(price_paths[j, -1])  # Get the final price for the current path
            final_prices[j] = final_price  # Store the final price
            option_payoffs[j] = calculate_european_call_option_payoff(OPTION_STRIKE_PRICE, final_price) / (1 + RISK_FREE_RATE)

        option_pricing_data.append((option_payoffs, final_prices))  # Store payoffs and final prices
    return option_pricing_data

def calculate_european_call_option_payoff(strike_price: float, final_stock_price: float) -> float:
    # Calculate the payoff for a European call option
    return max(final_stock_price - strike_price, 0)  # Payoff formula

def generate_stocks_price_paths(stock_distributions: Dict[str, Dict[str, Any]], stock_names: List[str]) -> List[np.ndarray]:
    # Generate simulated stock price paths based on distributions
    stocks_price_paths: List[np.ndarray] = []
    for i in range(len(stock_names)):
        stock_name: str = stock_names[i]
        random_generator: Callable[[], float] = stock_distributions[stock_name]['generator']
        initial_stock_price: float = stock_distributions[stock_name]['initial_stock_price']
        drift_rate: float = stock_distributions[stock_name]['drift_rate']
        volatility_rate: float = stock_distributions[stock_name]['volatility_rate']

        # Create stock price paths using the generator
        stock_price_paths: np.ndarray = create_stock_price_paths(random_generator, initial_stock_price, drift_rate, volatility_rate)
        stocks_price_paths.append(stock_price_paths)  # Append generated paths

    return stocks_price_paths

def create_stock_price_paths(
        random_generator: Callable[[], float], initial_stock_price: float, drift_rate: float, volatility_rate: float
) -> np.ndarray:
    # Create simulated stock price paths using a random generator
    stock_price_paths: np.ndarray = np.zeros((NUM_SIMULATION_PATHS, int(TOTAL_TIME_YEARS / TIME_INCREMENT)))  # Initialize price paths array
    for i in range(NUM_SIMULATION_PATHS):
        current_price: float = initial_stock_price  # Start with the initial stock price
        for t in range(int(TOTAL_TIME_YEARS / TIME_INCREMENT)):
            price_change: float = drift_rate * TIME_INCREMENT + volatility_rate * random_generator()  # Calculate price change
            current_price += price_change  # Update current price
            stock_price_paths[i, t] = current_price  # Store the price
    return stock_price_paths

def plot_simulated_stock_predictions(stock_distributions: Dict[str, Dict[str, Any]], stock_names: List[str], stocks_price_paths: List[np.ndarray]) -> None:
    # Plot the simulated stock price paths
    for i in range(len(stock_names)):
        distribution_name: str = stock_distributions[stock_names[i]]['name']
        # Create a new figure for plotting
        plt.figure(figsize=(12, 6))
        for j in range(min(NUM_SIMULATION_PATHS, 10)):
            plt.plot(stocks_price_paths[i][j])  # Plot a limited number of paths
        plt.xlabel('Days')  # X-axis label
        plt.ylabel('Stock Price')  # Y-axis label
        plt.title(f"Simulated {stock_names[i]} Price Paths Using {distribution_name} Distribution")  # Title
        plt.show()  # Show the plot

if __name__ == '__main__':
    main()  # Run the main function to start the simulation
