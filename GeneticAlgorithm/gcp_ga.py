import random
from collections import Counter
import psutil
import os
import time
import networkx as nx
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

def read_file(filename):
    graph = {}
    with open(filename, 'r') as file:
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


# def smarter_mutation(solution, nodes, graph):
#     new_solution = solution[:]

#     for node in graph:
#         conflict = any(new_solution[node] == new_solution[neighbor] for neighbor in graph[node])
#         if conflict:
#             neighbor_colors = set(new_solution[neighbor] for neighbor in graph[node])
#             available_colors = list(set(range(nodes)) - neighbor_colors)
#             if available_colors:
#                 new_solution[node] = random.choice(available_colors)

#     return new_solution


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
    plt.savefig(f'table1_results/{filename}_ram_usage.png')
    plt.close()

    cumulative_times = [sum(generation_times[:i+1]) for i in range(len(generation_times))]
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 1000 + 1), cumulative_times, marker='o', linewidth=2, color='teal')
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Execution Time (s)")
    plt.title(f"Cumulative Execution Time - {filename}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'table1_results/{filename}_cumulative_execution_time.png')
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
    plt.savefig(f'table1_results/{instance_name}_colored_graph.png')
    plt.close()


def visualize_embedding(h, colors, instance_name):
    h = h.detach().cpu().numpy()
    if h.ndim == 1:
        h = h.reshape(-1, 1)
    h = (h - h.mean()) / (h.std() + 1e-9)
    if h.shape[1] > 1:
        h = TSNE(n_components=2, perplexity=min(2, h.shape[0] - 1)).fit_transform(h)
    else:
        h = torch.cat((torch.linspace(-1, 1, h.shape[0]).unsqueeze(1), torch.zeros(h.shape[0], 1)), dim=1).numpy()
    plt.figure(figsize=(7, 7))
    scatter = plt.scatter(h[:, 0], h[:, 1], s=140, c=colors, cmap="Set2")
    plt.title(f"Embedding - {instance_name}")
    plt.legend(*scatter.legend_elements(), title="Node Colors")
    plt.savefig(f'table1_results/{instance_name}_embedding.png')
    plt.close()


def multiple_main(graph, nodes, runs, filename):

    for i in range(runs):

        # solution = genetic_algorithm(graph, nodes)
        solution= genetic_algorithm(graph, nodes)

        if is_valid(solution, graph) and solution is not None:
            reduced_solution = color_reduction(solution)
            for _ in range(3):
                reduced_solution = color_reduction(reduced_solution)

            solution = reduced_solution

            print(f'{i+1}. Valid solution found!')
            print(f'{i+1}. Predicted χ: ', len(set(solution)))
            print(f'{i+1}. Predicted Colors: ', normalize(solution))

            with open(f'results/{filename}_results.txt', "a") as f:
                f.write(f"{i+1}. Valid solution found!\n")
                f.write(f"{i+1}. Predicted X: {len(set(solution))}\n")
                f.write(f"{i+1}. Predicted Colors: {normalize(solution)}\n\n")
        else:
            print(f'{i+1}. No solution found.')


def multiple_main_with_info(graph, nodes, runs, filename, num_classes):

    table_1_results = []

    better_solution = [i for i in range(nodes)]

    m_generation_times = []
    m_ram_usage = []
    m_total_time = []

    mem_ram = []

    ttl_tm = time.time()

    for i in range(runs):

        # solution = genetic_algorithm(graph, nodes)
        solution, generation_times, ram_usage, total_time = genetic_algorithm_with_info(graph, nodes)
        mem_ram.append(track_ram_usage())

        if is_valid(solution, graph) and solution is not None:
            reduced_solution = color_reduction(solution)
            for _ in range(3):
                reduced_solution = color_reduction(reduced_solution)

            solution = reduced_solution

            if len(set(solution)) < len(set(better_solution)):
                better_solution = solution
                m_generation_times = generation_times
                m_ram_usage = ram_usage
                m_total_time = total_time

            print(f'{i+1}. Valid solution found!')
            print(f'{i+1}. True χ: ', len(set(solution)))
            # print(f'{i+1}. Predicted Colors: ', normalize(solution))

            with open(f'results/{filename}_results.txt', "a") as f:
                f.write(f"{i+1}. Valid solution found!\n")
                f.write(f"{i+1}. True X: {len(set(solution))}\n")
                f.write(f"{i+1}. Predicted Colors: {normalize(solution)}\n\n")

        else:
            print(f'{i+1}. No solution found.')
    
    if better_solution:
        table_1_results.append({
            'Instance': filename,
            'Size': nodes,
            'True X': num_classes,
            'Predicted X (GA)': len(set(better_solution)),
            'Predicted Colors': better_solution,
            'Execution Time (s)': m_total_time,
            'Overall Time (s)': time.time() - ttl_tm,
        })

        data_plot(better_solution, num_classes, m_total_time, generation_times, m_ram_usage, filename)

        G = nx.Graph()
        _, edge_index = load_graph_from_file(f"instances/{filename}.col")
        edge_list = edge_index.t().cpu().numpy()
        G.add_edges_from(edge_list)
        plot_colored_graph(G, better_solution, filename)
        visualize_embedding(torch.tensor(better_solution), better_solution, filename)
        # plot_colored_graph(graph, nodes, solution, filename)

    return table_1_results


if __name__ == '__main__':

    instances = [
        # ('queen5_5', 5),
        # ('queen6_6', 7),
        # ('myciel5', 6),
        # ('queen7_7', 7),
        # ('queen8_8', 9),
        # ('1-Insertions_4', 4),
        # ('huck', 11),
        # ('jean', 10),
        # ('queen9_9', 10),
        # ('david', 11),
        # ('mug88_1', 4),
        # ('myciel6', 7),
        # ('queen8_12', 12),
        # ('games120', 9),
        # ('queen11_11', 11),
        # ('anna', 11),
        # ('2-Insertions_4', 4),
        # ('queen13_13', 13),
        # ('myciel7', 8),
        ('homer', 13),
    ]
    
    table_1_results = []

    for instance in instances:
        filename = instance[0]
        num_classes = instance[1]
        print(f'Processing instance: {filename}')

        graph, nodes, edges = read_file(f'instances/{filename}.col')

        # result = multiple_main_with_info(graph, nodes, 1, filename, num_classes)

        tab1 = multiple_main_with_info(graph, nodes, 10, filename, num_classes)

        table_1_results += tab1

        with open('table1_results/table_1_results.txt', 'a', encoding='utf-8') as f:
            f.write("Table 1: Results for Color02/03/04 instances:\n")
            f.write(f"Instance: {tab1[0]['Instance']}\n")
            f.write(f"Size: {tab1[0]['Size']}\n")
            f.write(f"True X: {tab1[0]['True X']}\n")
            f.write(f"Predicted X (GA): {tab1[0]['Predicted X (GA)']}\n")
            f.write(f"Predicted Colors: {tab1[0]['Predicted Colors']}\n")
            f.write(f"Execution Time (s): {tab1[0]['Execution Time (s)']}\n")
            f.write(f"Overall Time (s): {tab1[0]['Overall Time (s)']}\n\n\n")

    # with open('table1_results/table_1_results.txt', 'a', encoding='utf-8') as f:
    #     f.write("Table 1: Results for Color02/03/04 instances:\n")
    #     for result in table_1_results:
    #         f.write(f"Instance: {result['Instance']}\n")
    #         f.write(f"Size: {result['Size']}\n")
    #         f.write(f"True X: {result['True X']}\n")
    #         f.write(f"Predicted X (GA): {result['Predicted X (GA)']}\n")
    #         f.write(f"Predicted Colors: {result['Predicted Colors']}\n")
    #         f.write(f"Execution Time (s): {result['Execution Time (s)']}\n")
    #         f.write(f"Overall Time (s): {result['Overall Time (s)']}\n\n\n")

