import random
import time
import matplotlib.pyplot as plt

def insertion_sort(arr, start, end):
    comparisons = 0
    for i in range(start + 1, end + 1):
        key = arr[i]
        j = i - 1
        while j >= start and arr[j] > key:
            comparisons += 1
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        comparisons += 1  # Count the last comparison that caused the while loop to exit
    return comparisons

def merge(arr, start, mid, end):
    left = arr[start:mid + 1]
    right = arr[mid + 1:end + 1]
    i = j = 0
    k = start
    comparisons = 0

    while i < len(left) and j < len(right):
        comparisons += 1
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1

    return comparisons

def hybrid_sort(arr, start, end, S):
    if end - start + 1 <= S:
        return insertion_sort(arr, start, end)
    else:
        mid = (start + end) // 2
        left_comparisons = hybrid_sort(arr, start, mid, S)
        right_comparisons = hybrid_sort(arr, mid + 1, end, S)
        merge_comparisons = merge(arr, start, mid, end)
        return left_comparisons + right_comparisons + merge_comparisons

def generate_data(size, max_value):
    return [random.randint(1, max_value) for _ in range(size)]

def test_hybrid_sort(arr, S):
    arr_copy = arr.copy()
    start_time = time.time()
    comparisons = hybrid_sort(arr_copy, 0, len(arr_copy) - 1, S)
    end_time = time.time()
    return end_time - start_time, comparisons

def find_optimal_S(sizes, max_value, S_values):
    results = {}
    for size in sizes:
        print(f"Testing array size: {size}")
        arr = generate_data(size, max_value)
        size_results = []
        for S in S_values:
            time_taken, comparisons = test_hybrid_sort(arr, S)
            size_results.append((S, time_taken, comparisons))
        results[size] = size_results
    return results

def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    for size, size_results in results.items():
        S_values = [r[0] for r in size_results]
        times = [r[1] for r in size_results]
        comparisons = [r[2] for r in size_results]

        ax1.plot(S_values, times, marker='o', label=f'Size {size}')
        ax2.plot(S_values, comparisons, marker='o', label=f'Size {size}')

    ax1.set_xlabel('S value')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Hybrid Sort Performance (Time) for Different S Values')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('S value')
    ax2.set_ylabel('Number of Comparisons')
    ax2.set_title('Hybrid Sort Performance (Comparisons) for Different S Values')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def original_mergesort(arr, start, end):
    if start < end:
        mid = (start + end) // 2
        left_comparisons = original_mergesort(arr, start, mid)
        right_comparisons = original_mergesort(arr, mid + 1, end)
        merge_comparisons = merge(arr, start, mid, end)
        return left_comparisons + right_comparisons + merge_comparisons
    return 0

def compare_with_original_mergesort(arr, optimal_S):
    arr_copy1 = arr.copy()
    arr_copy2 = arr.copy()

    # Hybrid Sort
    hybrid_start_time = time.time()
    hybrid_comparisons = hybrid_sort(arr_copy1, 0, len(arr_copy1) - 1, optimal_S)
    hybrid_end_time = time.time()
    hybrid_time = hybrid_end_time - hybrid_start_time

    # Original Mergesort
    mergesort_start_time = time.time()
    mergesort_comparisons = original_mergesort(arr_copy2, 0, len(arr_copy2) - 1)
    mergesort_end_time = time.time()
    mergesort_time = mergesort_end_time - mergesort_start_time

    return {
        'hybrid': {'time': hybrid_time, 'comparisons': hybrid_comparisons},
        'mergesort': {'time': mergesort_time, 'comparisons': mergesort_comparisons}
    }

# Main execution
sizes = [1000, 10000, 100000, 1000000, 10000000]
max_value = 1000000
S_values = [4, 8, 16, 32, 64, 128]

print("Finding optimal S value...")
results = find_optimal_S(sizes, max_value, S_values)
plot_results(results)

# Determine optimal S
optimal_S = min(results[max(sizes)], key=lambda x: x[1])[0]
print(f"Optimal S value: {optimal_S}")

# Compare with original Mergesort
print("\nComparing hybrid sort with original Mergesort for 10 million integers...")
large_arr = generate_data(10000000, max_value)
comparison_results = compare_with_original_mergesort(large_arr, optimal_S)

print("\nResults for 10 million integers:")
print(f"Hybrid Sort (S={optimal_S}):")
print(f"  Time: {comparison_results['hybrid']['time']:.4f} seconds")
print(f"  Comparisons: {comparison_results['hybrid']['comparisons']}")
print("Original Mergesort:")
print(f"  Time: {comparison_results['mergesort']['time']:.4f} seconds")
print(f"  Comparisons: {comparison_results['mergesort']['comparisons']}")