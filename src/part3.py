import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from typing import Tuple, Callable

# Constants for Option Pricing
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
    # Run distribution fitting and stock data simulation
    stock1_data, stock2_data = load_stock_data()
    fit_distributions(stock1_data, stock2_data)

    # Run Monte Carlo simulations for Vanilla European Option Pricing
    simulate_vanilla_option_pricing()

    # Run the basket option pricing
    simulate_basket_option_pricing()

def load_stock_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load stock data from CSV files."""
    stock1_data = pd.read_csv('data/stock1.csv')['Close'].values
    stock2_data = pd.read_csv('data/stock2.csv')['Close'].values
    return stock1_data, stock2_data

def fit_distributions(stock1_data: np.ndarray, stock2_data: np.ndarray) -> None:
    """Fit distributions to stock data (this is a placeholder for your actual fitting code)."""
    # Here, implement the logic to fit distributions for stock1 and stock2 and save parameters.
    print("Fitting distributions...")  # Placeholder

def simulate_vanilla_option_pricing() -> None:
    """Simulate stock price paths and calculate option pricing."""
    stock_paths = generate_stock_price_paths(normal_price_change_generator)
    final_prices, option_payoffs = calculate_option_pricing(stock_paths)

    # Plotting stock price paths
    plt.figure(figsize=(12, 6))
    for i in range(min(NUM_PATHS, 10)):
        plt.plot(stock_paths[i])
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.title(f"Simulated Stock Price Paths Using Normal Distribution")
    plt.show()

    # Output average results
    print(f"Average stock price after {int(1 / TIME_INCREMENT) * TOTAL_TIME} days: ${np.average(final_prices):.2f}")
    print(f"Average payoff for a block of 100 options: ${np.average(option_payoffs) * 100:.2f}")
    print(f"Estimated cost of the option: ${np.average(option_payoffs):.2f}")

@njit
def normal_price_change_generator() -> float:
    """Generate price changes using a normal distribution."""
    random_shock: float = np.random.normal(0, np.sqrt(TIME_INCREMENT))
    price_change: float = DRIFT_RATE * TIME_INCREMENT + VOLATILITY_RATE * random_shock
    return price_change

@njit
def calculate_european_call_option_payoff(strike_price: float, final_stock_price: float) -> float:
    """Calculate the payoff for a European call option."""
    return max(final_stock_price - strike_price, 0)

@njit(parallel=True)
def generate_stock_price_paths(price_change_generator: Callable[[], float]) -> np.ndarray:
    """Generate stock price paths using the specified price change generator."""
    price_paths: np.ndarray = np.zeros((NUM_PATHS, int(TOTAL_TIME / TIME_INCREMENT)))
    for i in prange(NUM_PATHS):
        current_price: float = INITIAL_STOCK_PRICE
        for t in range(int(TOTAL_TIME / TIME_INCREMENT)):
            price_change: float = price_change_generator()
            current_price += price_change
            price_paths[i, t] = current_price
    return price_paths

@njit(parallel=True)
def calculate_option_pricing(price_paths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the final prices and option payoffs from stock price paths."""
    option_payoffs: np.ndarray = np.zeros(price_paths.shape[0])
    final_prices: np.ndarray = np.zeros(price_paths.shape[0])

    for i in prange(price_paths.shape[0]):
        final_price: float = float(price_paths[i, -1])
        final_prices[i] = final_price
        option_payoffs[i] = calculate_european_call_option_payoff(STRIKE_PRICE, final_price) / (1 + RISK_FREE_RATE)

    return final_prices, option_payoffs

def simulate_basket_option_pricing() -> None:
    """Simulate basket option pricing using two stocks."""
    stock1_paths, stock2_paths = simulate_stock_paths(stock1_price_change_generator, stock2_price_change_generator)

    # Plot stock price paths
    plot_stock_paths(stock1_paths, stock2_paths)

    # Calculate option pricing for both scenarios
    final_prices_avg, option_payoffs_avg = calculate_basket_option(stock1_paths, stock2_paths, average=True)
    final_prices_max, option_payoffs_max = calculate_basket_option(stock1_paths, stock2_paths, average=False)

    # Output the results for both scenarios
    print(f"Scenario 1 (Average):")
    print(f"Average stock price: ${np.average(final_prices_avg):.2f}")
    print(f"Average payoff for a block of 100 options: ${np.average(option_payoffs_avg) * 100:.2f}")
    print(f"Estimated cost of the option: ${np.average(option_payoffs_avg):.2f}")
    print()
    print(f"Scenario 2 (Maximum):")
    print(f"Average stock price: ${np.average(final_prices_max):.2f}")
    print(f"Average payoff for a block of 100 options: ${np.average(option_payoffs_max) * 100:.2f}")
    print(f"Estimated cost of the option: ${np.average(option_payoffs_max):.2f}")

@njit(parallel=True)
def simulate_stock_paths(price_change_generator1: Callable[[], float], 
                         price_change_generator2: Callable[[], float]) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate price paths for both stock1 and stock2."""
    stock1_price_paths = np.zeros((NUM_PATHS, int(TOTAL_TIME / TIME_INCREMENT)))
    stock2_price_paths = np.zeros((NUM_PATHS, int(TOTAL_TIME / TIME_INCREMENT)))
    
    for i in prange(NUM_PATHS):
        current_price1 = INITIAL_STOCK_PRICE
        current_price2 = INITIAL_STOCK_PRICE
        
        for t in range(int(TOTAL_TIME / TIME_INCREMENT)):
            price_change1 = price_change_generator1()
            price_change2 = price_change_generator2()
            
            current_price1 += price_change1
            current_price2 += price_change2
            
            stock1_price_paths[i, t] = current_price1
            stock2_price_paths[i, t] = current_price2

    return stock1_price_paths, stock2_price_paths

@njit
def stock1_price_change_generator() -> float:
    """Simulate price changes for stock1 using the best-fitting distribution (e.g., Beta)."""
    random_shock = np.random.beta(BETA_A, BETA_B) - BETA_SHIFT
    price_change = DRIFT_RATE * TIME_INCREMENT + VOLATILITY_RATE * random_shock
    return price_change

@njit
def stock2_price_change_generator() -> float:
    """Simulate price changes for stock2 using the best-fitting distribution (e.g., Normal)."""
    random_shock = np.random.normal(0, np.sqrt(TIME_INCREMENT))
    price_change = DRIFT_RATE * TIME_INCREMENT + VOLATILITY_RATE * random_shock
    return price_change

@njit(parallel=True)
def calculate_basket_option(stock1_paths: np.ndarray, stock2_paths: np.ndarray, average: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate option pricing for both stocks based on average or maximum of stock1 and stock2."""
    final_prices_1 = stock1_paths[:, -1]
    final_prices_2 = stock2_paths[:, -1]
    
    if average:
        final_prices = (final_prices_1 + final_prices_2) / 2
    else:
        final_prices = np.maximum(final_prices_1, final_prices_2)

    option_payoffs = np.maximum(final_prices - STRIKE_PRICE, 0) / (1 + RISK_FREE_RATE)
    
    return final_prices, option_payoffs

def plot_stock_paths(stock1_paths: np.ndarray, stock2_paths: np.ndarray) -> None:
    """Plot the stock price paths for stock1 and stock2."""
    plt.figure(figsize=(12, 6))
    
    # Plot a few paths for stock1
    for i in range(min(NUM_PATHS, 5)):  # Limit the number of paths to plot
        plt.plot(stock1_paths[i], label=f'Stock 1 - Path {i+1}', alpha=0.5)
    
    # Plot a few paths for stock2
    for i in range(min(NUM_PATHS, 5)):  # Limit the number of paths to plot
        plt.plot(stock2_paths[i], label=f'Stock 2 - Path {i+1}', alpha=0.5, linestyle='dashed')
    
    plt.title('Simulated Stock Price Paths for Stock1 and Stock2')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

