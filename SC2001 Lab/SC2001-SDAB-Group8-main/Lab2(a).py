import numpy as np
import time

# Dijkstra's algorithm with adjacency matrix and array-based priority queue
def dijkstra_matrix_array(graph, source):
    V = len(graph)
    dist = [float('inf')] * V
    dist[source] = 0
    visited = [False] * V

    for _ in range(V):
        # Find the vertex with the minimum distance
        min_dist = float('inf')
        u = -1
        for v in range(V):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                u = v
        
        if u == -1:
            break  # All remaining vertices are inaccessible from the source

        # Mark the picked vertex as visited
        visited[u] = True

        # Update distances for the neighbors of u
        for v in range(V):
            if graph[u][v] > 0 and not visited[v]:  # edge exists and v is unvisited
                new_dist = dist[u] + graph[u][v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
    
    return dist

# Function to generate a random weighted graph as an adjacency matrix
def generate_graph(V, density=0.5, max_weight=10):
    graph = np.zeros((V, V), dtype=int)
    for i in range(V):
        for j in range(i+1, V):
            if np.random.rand() < density:
                weight = np.random.randint(1, max_weight)
                graph[i][j] = weight
                graph[j][i] = weight  # undirected graph
    return graph

# Testing Dijkstra's algorithm and measuring performance
def test_dijkstra(V, density=0.5):
    graph = generate_graph(V, density)
    start_time = time.time()
    dijkstra_matrix_array(graph, 0)
    end_time = time.time()
    return end_time - start_time

# Run empirical tests for different graph sizes
for V in [100, 200, 400, 800]:
    duration = test_dijkstra(V, density=0.1)
    print(f"Graph size: {V}, Time taken: {duration:.4f} seconds")