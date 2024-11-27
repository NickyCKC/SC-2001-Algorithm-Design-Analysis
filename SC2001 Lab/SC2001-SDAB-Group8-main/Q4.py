from collections import Counter

def unbounded_knapsack(capacity, weights, profits):
    """
    Args:
        capacity (int): The capacity of the knapsack
        weights (list): List of weights for each item type
        profits (list): List of profits for each item type
    
    Returns:
        tuple: (maximum profit, list of items chosen)
    """
    # Initialize dp array to store maximum profit for each capacity
    dp = [0] * (capacity + 1)
    # Array to keep track of which item was chosen for each capacity
    chosen_item = [0] * (capacity + 1)
    
    # For each possible capacity
    for c in range(1, capacity + 1):
        # Try each item type
        max_profit = dp[c-1]  # Initialize with previous value
        best_item = -1
        
        for i in range(len(weights)):
            if weights[i] <= c:
                current_profit = profits[i] + dp[c - weights[i]]
                if current_profit > max_profit:
                    max_profit = current_profit
                    best_item = i
        
        dp[c] = max_profit
        chosen_item[c] = best_item
    
    # Reconstruct solution
    remaining_capacity = capacity
    items_chosen = []
    
    while remaining_capacity > 0:
        item = chosen_item[remaining_capacity]
        if item == -1:
            break
        items_chosen.append(item)
        remaining_capacity -= weights[item]
    
    return dp[capacity], items_chosen


def optimized_unbounded_knapsack(capacity, weights, profits):
    """
    Optimized unbounded knapsack using profit per unit weight heuristic.
    
    Args:
        capacity (int): Knapsack capacity
        weights (list): List of weights for each item type
        profits (list): List of profits for each item type
    
    Returns:
        tuple: (maximum profit, list of items chosen)
    """
    n = len(weights)
    
    # Calculate profit per unit weight for each item
    profit_per_weight = [(profits[i] / weights[i], i) for i in range(n)]
    # Sort items by profit per unit weight in descending order
    profit_per_weight.sort(reverse=True)
    sorted_indices = [i for _, i in profit_per_weight]
    
    # Optimization 1: Pre-calculate best single item for each small capacity
    max_small_capacity = min(capacity, 1000)  # Limit for small capacities
    best_single_items = [0] * (max_small_capacity + 1)
    for c in range(1, max_small_capacity + 1):
        best_profit = 0
        best_item = -1
        for i in sorted_indices:  # Try items in order of profit density
            if weights[i] <= c:
                current_profit = (c // weights[i]) * profits[i]
                if current_profit > best_profit:
                    best_profit = current_profit
                    best_item = i
        best_single_items[c] = (best_profit, best_item)
    
    # Initialize dp array with optimal solutions for small capacities
    dp = [0] * (capacity + 1)
    chosen_item = [0] * (capacity + 1)
    for c in range(1, min(capacity + 1, max_small_capacity + 1)):
        dp[c] = best_single_items[c][0]
        chosen_item[c] = best_single_items[c][1]
    
    # Optimization 2: For larger capacities, use profit density ordering
    # and early stopping when no better solution is possible
    for c in range(max_small_capacity + 1, capacity + 1):
        max_profit = dp[c-1]
        best_item = -1
        
        # Try items in order of profit density
        for _, i in profit_per_weight:
            if weights[i] <= c:
                # Optimization 3: Upper bound calculation
                current_profit = profits[i] + dp[c - weights[i]]
                remaining_capacity = c - weights[i]
                
                # If this item can't possibly lead to a better solution, skip it
                if current_profit + (remaining_capacity * profit_per_weight[0][0]) <= max_profit:
                    continue
                
                if current_profit > max_profit:
                    max_profit = current_profit
                    best_item = i
            else:
                # Since items are sorted by profit density, if this item doesn't fit,
                # subsequent items with lower profit density won't give better results
                break
        
        dp[c] = max_profit
        chosen_item[c] = best_item
    
    # Reconstruct solution more efficiently
    remaining_capacity = capacity
    items_chosen = []
    while remaining_capacity > 0:
        item = chosen_item[remaining_capacity]
        if item == -1:
            break
        # Optimization 4: Take multiple items at once when possible
        count = remaining_capacity // weights[item]
        items_chosen.extend([item] * count)
        remaining_capacity -= weights[item] * count
    
    return dp[capacity], items_chosen

def benchmark_comparison(capacity, weights, profits):
    """Compare performance of original and optimized versions"""
    import time
    
    print("\nBenchmarking both versions:")
    
    # Test original version
    start_time = time.time()
    result1 = unbounded_knapsack(capacity, weights, profits)
    original_time = time.time() - start_time
    
    # Test optimized version
    start_time = time.time()
    result2 = optimized_unbounded_knapsack(capacity, weights, profits)
    optimized_time = time.time() - start_time
    
    print(f"Original version time: {original_time:.6f} seconds")
    print(f"Optimized version time: {optimized_time:.6f} seconds")
    print(f"Speedup: {original_time/optimized_time:.2f}x")
    print(f"Same result: {result1[0] == result2[0]}")

# Test with larger example
def test_larger_case():
    weights = [4, 6, 8, 2, 9, 5, 7, 3]
    profits = [7, 6, 9, 4, 11, 8, 10, 5]
    capacity = 100
    
    print("\nLarger Test Case:")
    print("Weights:", weights)
    print("Profits:", profits)
    print("Capacity:", capacity)
    
    benchmark_comparison(capacity, weights, profits)
    max_profit, items = optimized_unbounded_knapsack(capacity, weights, profits)
    print_solution(capacity, weights, profits, max_profit, items)

def print_solution(capacity, weights, profits, max_profit, items_chosen):
    """
    Prints the detailed solution including items chosen and verification.
    """
    print(f"\nKnapsack Capacity: {capacity}")
    print(f"Maximum Profit: {max_profit}")
    
    # Count frequency of each item
    item_counts = Counter(items_chosen)
    
    print("\nItems chosen:")
    total_weight = 0
    total_profit = 0
    
    for item, count in item_counts.items():
        print(f"Item {item} (weight={weights[item]}, profit={profits[item]}): {count} times")
        total_weight += weights[item] * count
        total_profit += profits[item] * count
    
    print(f"\nVerification:")
    print(f"Total weight: {total_weight}/{capacity}")
    print(f"Total profit: {total_profit}")

# Test cases from the problem
def test_case_1():
    weights = [4, 6, 8]
    profits = [7, 6, 9]
    capacity = 14
    
    print("\nTest Case 1:")
    max_profit, items = unbounded_knapsack(capacity, weights, profits)
    print_solution(capacity, weights, profits, max_profit, items)

def test_case_2():
    weights = [5, 6, 8]
    profits = [7, 6, 9]
    capacity = 14
    
    print("\nTest Case 2:")
    max_profit, items = unbounded_knapsack(capacity, weights, profits)
    print_solution(capacity, weights, profits, max_profit, items)


# Run the test
test_larger_case()

# Run the test cases
test_case_1()
test_case_2()