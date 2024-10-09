# CS5060 Program3

## Team
- Ann Marie Humble
- Brighton Ellis
- Madison Patch
- Nate Stott

## Discussion

### Part 1: Fitting Stock Data to Distributions

To begin, we used pandas to load the historical data from `stock1.csv` and `stock2.csv` into series. Since the beta distribution (by definition) ranges from 0 to 1, we normalized the data for both stocks so that we could plot each fitted distribution on the same graph. We plotted a histogram of the normalized daily returns for each stock using `matplotlib`, and then fitted the histogram with a normal, lognormal, and beta distribution (from `scipy.stats`). We then overlaid the histogram with these fitted distributions. It is worth noting here that our normal and log-normal distributions for Stock 1 are nearly identical. At the regular scale it looks like we only have two distributions plotted, but if you zoom in, you will see that the normal and lognormal distributions are just very close to each other. While the normal distribution is always symmetrical, the lognormal distribution is always right-skewed. The close overlap between Stock 1's normal and lognormal fits indicates that the historical data for Stock 1 is not significantly skewed. If it had been skewed, the lognormal distribution would have reflected it.

After this, we used `scipy.stats.kstest` to perform a Kolmogorov-Smirnov test for goodness-of-fit on each distribution. Our statistic and p-values are reported by in full in `part1.py`'s terminal output, but we will a summary of the results here. For Stock 1 and Stock 2, our smallest statistic value was 0.029, and our largest statistic value was 0.0335. The statistic value gives us the distance between each distribution, and our small statistic values indicate a small distance between the dataset and each distribution. The p-value gives us the probability that the two distributions came from the same sample. Our p-values ranged from 0.79 to 0.93, indicating that all of these distributions fit the historical data for Stock 1 and Stock 2 well. We evaluated the best fitting distribution based on which distribution had the highest p-value, and found that the best fitting distribution for both stocks is the lognormal distribution, with a p-value of 86% for Stock 1 and a p-value of 93% for Stock 2. The lognormal distribution is bound by 0 on the left side, and therefore perfect for modeling data that doesn't contain negative values. It is commonly used in finance to analyze stock prices, and thus it makes sense that this distribution has the best fit.

### Part 2: Monte Carlo Simulation for Vanilla European Option Pricing

For part 2, we simulated future stock prices using a normal distribution and a beta distribution. We did this with the generic function `simulate_stock_and_plot()`, which takes a distribution name and the stock's price path. The price path is generated in a lambda function called `generate_stock_price_paths()`, which calls various price change generators to give it a `price_change` value using the Monte Carlo method. Volatility and drift are applied using the values given to us in the assignment. To calculate the payoff of the option at maturity, we used `numpy.zeros()`.

For our model of price paths using the normal distribution, the highest density of price paths seem to end near the starting price. The highest price was around $131 and the lowest was around $62. Since the beta distribution will always return a value from 0 to 1, the paths simulated by the beta distribution only increased in stock price (and all of the price paths increased at about the same rate). After one year, the simulated stock prices ranged from around $800 to $945.

Consistantly, the average stock price when using the normal distribution hovers around $100, and the average stock price when using the beta distribution hovers around $868. The average payoff for a block of 100 options using the normal distribution is around $675 and the average payoff using the beta distribution is around $76127. The estimated cost of the option using the normal distribution hovers around $6.70, and the estimated option cost using the beta distribution hovers around $761. As you can see, the estimations for the beta distribution are unrealistically inflated.

This is a good illustration of why the normal distrubution is much better at modeling stock price paths than the beta distribution. Unless the output of the beta distribution is modified to return negative numbers enough of the time, you will get unrealistic price predictions. The shift that we were given (of 0.35) was not large enough to do this.

### Part 3: Stochastic Jumps and Basket Option Pricing
Deliverables: Python code, results of the option prices for both scenarios, and a comparison of the two.

