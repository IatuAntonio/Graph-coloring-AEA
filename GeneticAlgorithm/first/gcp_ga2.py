import random
from collections import Counter
import psutil
import os
import time
import networkx as nx
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


# ===================== DISTRIBUTIONS FOR TABLE 2 =====================
synthetic_distributions = {
    "Power Law Tree": lambda: nx.random_powerlaw_tree(n=50, tries=1000, seed=42),
    "Small-world": lambda: nx.watts_strogatz_graph(n=50, k=4, p=0.1, seed=42),
    "Holme and Kim": lambda: nx.powerlaw_cluster_graph(n=50, m=2, p=0.3, seed=42),
}


def accuracy(predictions, graph):
    # Verificăm numărul de conflicte (muchiile care leagă noduri cu aceeași culoare)
    conflicts = 0
    for node in graph:
        for neighbor in graph[node]:
            if node < neighbor:  # Pentru a evita să verificăm de două ori aceleași muchii
                if predictions[node] == predictions[neighbor]:  # Dacă două noduri adiacente au aceeași culoare
                    conflicts += 1

    # Acuratețea poate fi calculată ca inversul procentajului de conflicte
    total_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    accuracy_score = 1 - (conflicts / total_edges if total_edges > 0 else 0)

    return accuracy_score



def convert_nx_to_custom_graph(nx, graph):
    graph = {}
    for node in nx.nodes:
        graph[node] = list(nx.neighbors(node))
    return graph, nx.number_of_nodes(), nx.number_of_edges()


def read_file(filename):
    graph = {}
    with open(filename, 'a') as file:
        line = file.readline().strip().split()
        nodes = int(line[2])
        edges = int(line[3])

        for i in range(0, nodes):
            graph[i] = []

        for _ in range(edges):
            line = file.readline().strip().split()
            u = int(line[1]) - 1
            v = int(line[2]) - 1
            graph[u].append(v)
            graph[v].append(u)
    
    return graph, nodes, edges


def load_graph_from_file(file_path):
    edge_index = []
    num_nodes = 0
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == 'p':
                num_nodes = int(parts[2])
            elif parts[0] == 'e':
                edge_index.append([int(parts[1]) - 1, int(parts[2]) - 1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return num_nodes, edge_index


def population_initialization(population_size, nodes):
    population = []
    for _ in range(population_size):
        individual = [random.randint(0, nodes - 1) for _ in range(nodes)]
        population.append(individual)

    return population


def fitness(solution, graph):
    conflicts = 0

    for node in graph:
        for neighbor in graph[node]:
            if node < neighbor:
                if solution[node] == solution[neighbor]:
                    conflicts += 1

    if conflicts == 0:
        return len(set(solution))
    else:
        return 1000 * conflicts + len(set(solution)) * 20


def selection(population, graph):
    scores = [(solution, fitness(solution, graph)) for solution in population]
    scores.sort(key=lambda x: x[1])
    return scores[0][0], scores[1][0]


def crossover(parent1, parent2):
    point = len(parent1) // 2
    child = parent1[:point] + parent2[point:]
    return child


def smart_crossover(parent1, parent2):
    child = []

    freq1 = Counter(parent1)
    freq2 = Counter(parent2)

    for i in range(len(parent1)):
        color1 = parent1[i]
        color2 = parent2[i]

        if color1 == color2:
            child.append(color1)
        else:
            if freq1[color1] + freq2[color1] > freq1[color2] + freq2[color2]:
                chosen = color1
            elif freq1[color1] + freq2[color1] < freq1[color2] + freq2[color2]:
                chosen = color2
            else:
                chosen = min(color1, color2)

            child.append(chosen)

    return child


def mutation(solution, nodes):
    index = random.randint(0, len(solution) - 1)
    value = random.randint(0, nodes - 1)
    solution[index] = value
    return solution


def smart_mutation(solution):
    index = random.randint(0, len(solution) - 1)
    available_colors = sorted(set(solution))
    solution[index] = random.choice(available_colors)
    return solution


def smarter_mutation(solution, nodes, graph):
    new_solution = solution[:]

    for node in graph:
        conflict = any(new_solution[node] == new_solution[neighbor] for neighbor in graph[node])
        if conflict:
            neighbor_colors = set(new_solution[neighbor] for neighbor in graph[node])
            available_colors = list(set(range(nodes)) - neighbor_colors)
            if available_colors:
                new_solution[node] = random.choice(available_colors)

    return new_solution


def is_valid(solution, graph):
    for node in graph:
        for neighbor in graph[node]:
            if node < neighbor:
                if solution[node] == solution[neighbor]:
                    return False
    return True


def genetic_algorithm(graph, nodes, population_size=100, generations=1000):
    population = population_initialization(population_size, nodes)
    valid_solutions = []

    for _ in range(generations):

        parent1, parent2 = selection(population, graph)
        population[0] = parent1
        population[1] = parent2

        # child = crossover(parent1, parent2)
        child = smart_crossover(parent1, parent2)
        population[2] = child

        child = smart_mutation(child)
        # child = smarter_mutation(child, nodes, graph)

        worst_individual = max(population, key=lambda x: fitness(x, graph))
        population[population.index(worst_individual)] = child

        population[random.randint(0, population_size - 1)] = color_reduction(child)

        for individual in population:
            if is_valid(individual, graph):
                valid_solutions.append((individual, len(set(individual))))
        
    # return min(population, key=lambda x: fitness(x, graph))
    if valid_solutions:
        best = min(valid_solutions, key=lambda x: x[1])
        return best[0]
    else:
        best = min(population, key=lambda x: fitness(x, graph))
        return best


def genetic_algorithm_with_info(graph, nodes, population_size=100, generations=1000):
    population = population_initialization(population_size, nodes)
    valid_solutions = []

    start_time = time.time()
    ram_usage = []
    generation_times = []

    for _ in range(generations):
        start_time = time.time()

        parent1, parent2 = selection(population, graph)
        population[0] = parent1
        population[1] = parent2

        # child = crossover(parent1, parent2)
        child = smart_crossover(parent1, parent2)
        population[2] = child

        child = smart_mutation(child)
        # child = smarter_mutation(child, nodes, graph)

        worst_individual = max(population, key=lambda x: fitness(x, graph))
        population[population.index(worst_individual)] = child

        population[random.randint(0, population_size - 1)] = color_reduction(child)

        for individual in population:
            if is_valid(individual, graph):
                valid_solutions.append((individual, len(set(individual))))
        
        ram_usage.append(track_ram_usage())
        generation_times.append(time.time() - start_time)
        
    # return min(population, key=lambda x: fitness(x, graph))
    if valid_solutions:
        best = min(valid_solutions, key=lambda x: x[1])
        total_time = time.time() - start_time
        return best[0], generation_times, ram_usage, total_time
    else:
        best = min(population, key=lambda x: fitness(x, graph))
        total_time = time.time() - start_time
        return best, generation_times, ram_usage, total_time


def normalize(solution):
    color_map = {}
    color_index = 1
    normalized_solution = []

    for color in solution:
        if color not in color_map:
            color_map[color] = color_index
            color_index += 1
        normalized_solution.append(color_map[color])

    return normalized_solution


def color_reduction(solution):
    new_solution = solution[:]
    colors_used = sorted(set(new_solution))

    for current_color in colors_used:
        for new_color in range(current_color):
            can_replace = True

            for node in range(len(new_solution)):
                if new_solution[node] == current_color:
                    for neighbor in graph[node]:
                        if new_solution[neighbor] == new_color:
                            can_replace = False
                            break
                if not can_replace:
                    break

            if can_replace:
                for node in range(len(new_solution)):
                    if new_solution[node] == current_color:
                        new_solution[node] = new_color
                break

    return new_solution


def track_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def data_plot(solution, num_classes, total_time, generation_times, ram_usage, filename):

    # RAM usage plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(1000), ram_usage, label="RAM Usage (MB)")
    plt.xlabel("Generations")
    plt.ylabel("RAM Usage (MB)")
    plt.title(f"RAM Usage for {filename}")
    plt.savefig(f'table2_results/{filename}_ram_usage.png')
    plt.close()

    cumulative_times = [sum(generation_times[:i+1]) for i in range(len(generation_times))]
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 1000 + 1), cumulative_times, marker='o', linewidth=2, color='teal')
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Execution Time (s)")
    plt.title(f"Cumulative Execution Time - {filename}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'table2_results/{filename}_cumulative_execution_time.png')
    plt.close()


def plot_colored_graph(graph, colors, instance_name):
    num_nodes = len(graph.nodes)
    if len(colors) != num_nodes:
        print(f"Warning: Color array size {len(colors)} does not match the number of nodes {num_nodes}. Adjusting colors array.")
        colors = colors[:num_nodes]
    plt.figure(figsize=(5, 5))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color=colors, cmap=plt.cm.rainbow, edge_color='gray', node_size=700)
    plt.title(f"Colored Graph - {instance_name}")
    plt.savefig(f'table2_results/{instance_name}_colored_graph.png')
    plt.close()


def visualize_embedding(h, colors, instance_name):
    h = h.detach().cpu().numpy()
    if h.ndim == 1:
        h = h.reshape(-1, 1)
    h = (h - h.mean()) / (h.std() + 1e-9)
    if h.shape[1] > 1:
        h = TSNE(n_components=2).fit_transform(h)

    plt.figure(figsize=(6, 6))
    plt.scatter(h[:, 0], h[:, 1], c=colors, cmap=plt.cm.rainbow, s=80)
    plt.colorbar()
    plt.title(f"Node Embeddings - {instance_name}")
    plt.savefig(f'table2_results/{instance_name}_embeddings.png')
    plt.close()


def multiple_main_with_info(synthetic_distributions, output_dir="table2_results", population_size=100, generations=1000):

    for name, distribution in synthetic_distributions.items():
        synthetic_graph = distribution()  # Generează graficul sintetic
        nodes = synthetic_graph.number_of_nodes()
        edges = synthetic_graph.number_of_edges()
        
        # Conversia la formatul customizat
        graph, nodes, edges = convert_nx_to_custom_graph(synthetic_graph, synthetic_graph)

        solution, generation_times, ram_usage, total_time = genetic_algorithm_with_info(graph, nodes, population_size, generations)

        # Calculăm acuratețea soluției
        accuracy_score = accuracy(solution, graph)
        print(f"Acuratețea pentru {name}: {accuracy_score:.4f}")

        instance_name = f"{name}_synthetic"
        data_plot(solution, len(set(solution)), total_time, generation_times, ram_usage, instance_name)
        plot_colored_graph(synthetic_graph, solution, instance_name)
        visualize_embedding(solution, solution, instance_name)

        # Poți salva acuratețea într-un fișier sau într-un raport dacă vrei
        with open(f'{output_dir}/{instance_name}_accuracy.txt', 'w') as f:
            f.write(f"Acuratețea pentru {name}: {accuracy_score:.4f}\n")



# Update the graph for testing
graph_filename = 'your_graph_file.gr'
multiple_main_with_info(graph_filename, synthetic_distributions)
