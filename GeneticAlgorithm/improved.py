import random
from collections import Counter
import psutil
import os
import time
import networkx as nx
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from pyvis.network import Network
import matplotlib.patches as mpatches


def read_file(filename):
    graph = {}
    with open(filename, 'r') as file:
        line = file.readline().strip().split()
        nodes = int(line[2])
        edges = int(line[3])

        for i in range(nodes):
            graph[i] = []

        for _ in range(edges):
            line = file.readline().strip().split()
            u, v = int(line[1]) - 1, int(line[2]) - 1
            graph[u].append(v)
            graph[v].append(u)

    return graph, nodes, edges


def population_initialization(population_size, nodes):
    return [[random.randint(0, nodes - 1) for _ in range(nodes)] for _ in range(population_size)]


def fitness(solution, graph):
    conflicts = sum(1 for node in graph for neighbor in graph[node] if node < neighbor and solution[node] == solution[neighbor])
    if conflicts == 0:
        return 10 * len(set(solution))
    return 1000 * conflicts + 100 * len(set(solution))


def selection(population, graph, elite_size=10):
    sorted_population = sorted(population, key=lambda sol: fitness(sol, graph))
    
    elite = sorted_population[:elite_size]
    
    parent1 = random.choice(elite)
    parent2 = random.choice(elite)
    
    return parent1, parent2


def tournament_selection_two(population, graph, k=8):
    tournament = random.sample(population, k)
    sorted_tournament = sorted(tournament, key=lambda sol: fitness(sol, graph))
    return sorted_tournament[0], sorted_tournament[1]


def smart_crossover(parent1, parent2):
    child = []

    total_freq = Counter(parent1) + Counter(parent2)

    for i in range(len(parent1)):
        color1 = parent1[i]
        color2 = parent2[i]

        if color1 == color2:
            child.append(color1)
        else:
            freq1 = total_freq[color1]
            freq2 = total_freq[color2]

            if freq1 > freq2:
                chosen = color1
            elif freq2 > freq1:
                chosen = color2
            else:
                chosen = random.choice([color1, color2])

            child.append(chosen)

    return child


# def smart_mutation(solution):
#     index = random.randint(0, len(solution) - 1)
#     available_colors = sorted(set(solution))
#     solution[index] = random.choice(available_colors)
#     return solution


def smart_mutation(solution, graph):
    mutated = solution[:]
    index = random.randint(0, len(solution) - 1)

    neighbor_colors = set(solution[neighbor] for neighbor in graph[index])

    current_color = solution[index]
    all_colors = set(solution)

    available = list(all_colors - neighbor_colors)
    if available:
        mutated[index] = random.choice(available)
    else:
        mutated[index] = max(solution) + 1 if current_color in neighbor_colors else current_color

    return mutated


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


def color_reduction(solution, graph):
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


def is_valid(solution, graph):
    for node in graph:
        for neighbor in graph[node]:
            if node < neighbor:
                if solution[node] == solution[neighbor]:
                    return False
    return True


def save_ram_plot(ram_usage, instance_name, output_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(ram_usage)), ram_usage)
    plt.xlabel("Epoch")
    plt.ylabel("RAM Usage (MB)")
    plt.title(f"RAM Usage - {instance_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{instance_name}_ram.png"))
    plt.close()


def save_time_plot(gen_times, instance_name, output_dir):
    cumulative = [sum(gen_times[:i+1]) for i in range(len(gen_times))]
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(gen_times) + 1), cumulative, color='teal')
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Execution Time (s)")
    plt.title(f"Cumulative Execution Time - {instance_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{instance_name}_time.png"))
    plt.close()


def save_static_graph_plot(graph, solution_colors, instance_name, output_dir):
    G = nx.Graph(graph)
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(7, 6))
    nx.draw(G, pos, with_labels=False, node_color=solution_colors, cmap=plt.cm.rainbow,
            node_size=100, edge_color='lightgray')
    unique_colors = sorted(set(solution_colors))
    patches = [mpatches.Patch(color=plt.cm.rainbow(i / max(unique_colors)), label=f'Color {i}') for i in unique_colors]
    plt.legend(handles=patches, title="Colors", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Colored Graph - {instance_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{instance_name}_graph.png"))
    plt.savefig(os.path.join(output_dir, f"{instance_name}_graph.svg"), format='svg')
    plt.close()


def save_interactive_graph(graph, solution_colors, instance_name, output_dir):
    G = nx.Graph(graph)
    pos = nx.kamada_kawai_layout(G)
    net = Network(height='700px', width='100%', notebook=False)
    cmap = plt.cm.rainbow
    max_color = max(solution_colors)
    
    for i, node in enumerate(G.nodes()):
        rgb = cmap(solution_colors[i] / max_color)[:3]
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        x, y = pos[node]
        net.add_node(str(node), label=str(node), color=hex_color, x=float(x*1000), y=float(y*1000),
                     title=f"Node {node}, Color {solution_colors[i]}")
    for u, v in G.edges():
        net.add_edge(str(u), str(v))
    net.set_options("""var options = {"physics": {"enabled": false}}""")
    net.save_graph(os.path.join(output_dir, f"{instance_name}_interactive.html"))


def genetic_algorithm(graph, nodes, instance_name, population_size=100, generations=1000, output_dir='./output'):
    population = population_initialization(population_size, nodes)
    # best_valid = None
    valid_solutions = []
    ram_usage = []
    gen_times = []

    for _ in range(generations):
        start_gen = time.time()
        ram_usage.append(track_ram_usage())

        if random.random() < 0.8:
            parent1, parent2 = tournament_selection_two(population, graph)
        else:
            parent1, parent2 = selection(population, graph)

        population[0] = color_reduction(parent1, graph)
        population[1] = color_reduction(parent2, graph)

        child = smart_crossover(parent1, parent2)
        child = smart_mutation(child, graph)
        child = color_reduction(child, graph)

        worst = max(population, key=lambda sol: fitness(sol, graph))
        population[population.index(worst)] = child
        population[random.randint(0, population_size - 1)] = child


        for individual in population:
            if is_valid(individual, graph):
                valid_solutions.append((individual, len(set(individual))))

        gen_times.append(time.time() - start_gen)

    if valid_solutions:
        best = min(valid_solutions, key=lambda x: x[1])[0]
    else:
        best = min(population, key=lambda x: fitness(x, graph))

    normalized = normalize(best)
    
    save_ram_plot(ram_usage, instance_name, output_dir)
    save_time_plot(gen_times, instance_name, output_dir)
    save_static_graph_plot(graph, normalized, instance_name, output_dir)
    save_interactive_graph(graph, normalized, instance_name, output_dir)

    return best


def track_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


if __name__ == '__main__':
    # filename = './instances/queen5_5.col'

    instances = [
        # ('queen5_5', 5),
        # ('queen6_6', 7),
        # ('myciel5', 6),
        ('queen7_7', 7),
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
        # ('homer', 13),
    ]
     
    runs = 10

    for instance, _ in instances:
        filename = f'./instances/{instance}.col'
        print(f"\nProcessing {instance}...")

        graph, nodes, _ = read_file(filename)

        best_solution = None
        best_chromatic = float('inf')
        total_time_runs = 0

        log_file_path = f'logs/{instance}.log'

        with open(log_file_path, 'a') as f:
            f.write(f'Logs for instance {instance}\n\n')

        for run in range(runs):
            start_time = time.time()
            solution = genetic_algorithm(graph, nodes, instance_name=instance)
            run_time = time.time() - start_time

            chromatic_number = len(set(solution))
            total_time_runs += run_time

            if is_valid(solution, graph):
                print(f"Run {run + 1}: Chromatic number = {chromatic_number}, Time taken = {run_time:.2f} seconds")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f'--- Run {run + 1} ---\n')
                    log_file.write(f'Chromatic number: {chromatic_number}\n')
                    log_file.write(f'Time taken: {run_time:.2f} seconds\n\n')
                    log_file.flush()

                if chromatic_number < best_chromatic:
                    best_chromatic = chromatic_number
                    best_solution = solution
            else:
                print(f"Run {run + 1}: Invalid solution found.")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f'--- Run {run + 1} ---\n')
                    log_file.write('Invalid solution found. Time taken: {run_time:.2f} seconds\n\n')
                    log_file.flush()

            # if is_valid(solution, graph) and chromatic_number < best_chromatic:
            #     best_chromatic = chromatic_number
            #     best_solution = solution

        result_file_path = f'results/{instance}_best.txt'
        with open(result_file_path, 'a') as result_file:
            if best_solution is not None:
                result_file.write(f'Best chromatic number: {best_chromatic}\n')
                result_file.write(f'Best normalized colors: {normalize(best_solution)}\n')
                result_file.write(f'Total time for {runs} runs: {total_time_runs:.2f} seconds\n')
                result_file.write(f'Average time per run: {total_time_runs / runs:.2f} seconds\n')
            else:
                result_file.write('No valid solution found in any run.\n')

        print(f"Best chromatic number for {instance}: {best_chromatic}")
        print(f"Total time for {runs} runs: {total_time_runs:.2f} seconds\n\n")