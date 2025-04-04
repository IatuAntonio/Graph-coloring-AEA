import random
import networkx as nx
import matplotlib.pyplot as plt


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


def fitness(solution, graph):
    conflicts = 0
    for node in graph:
        for neighbor in graph[node]:
            if node < neighbor:
                if solution[node] == solution[neighbor]:
                    conflicts += 1

    # penalty = len(set(solution)) * 5
    # de la 50+ culori am ajuns la valori de 40 (chiar 34)
    # aparent asta scade super mult din culori

    return conflicts + len(set(solution)) * 5


def population_initialization(population_size, nodes):
    population = []
    for _ in range(population_size):
        individual = [random.randint(1, nodes) for _ in range(nodes)]
        population.append(individual)

    return population


def selection(population, graph):
    scores = [(solution, fitness(solution, graph)) for solution in population]
    scores.sort(key=lambda x: x[1])
    return scores[0][0], scores[1][0]


def selection_enhanced(population, graph):
    scores = [(solution, fitness(solution, graph)) for solution in population]
    total_fitness = sum(score for solution, score in scores)
    probabilities = [score / total_fitness for solution, score in scores]
    selected_parents = random.choices([solution for solution, score in scores], weights=probabilities, k=2)
    return selected_parents


def tournament_selection(population, graph, k=5):
    selected = random.sample(population, k)  # alegem aleatoriu 5 
    return min(selected, key=lambda x: fitness(x, graph))



def crossover(parent1, parent2):
    child = []
    
    parent1_colors = set(parent1)
    parent2_colors = set(parent2)

    common_colors = parent1_colors.intersection(parent2_colors)

    for i in range(len(parent1)):
        if common_colors:
            child.append(random.choice(list(common_colors)))
            # am ales culorile comune si deja am scazut la 12/13 culori
        else:
            if parent1[i] == parent2[i]: 
                child.append(parent1[i]) 
            else:
                child.append(min(parent1[i], parent2[i]))


    return child


# def crossover(parent1, parent2):
#     mid = len(parent1) // 2
#     child = parent1[:mid] + parent2[mid:]
#     return child



# def mutation(solution, nodes):
#     index = random.randint(0, len(solution) - 1)
#     # solution[index] = random.randint(1, nodes)
#     available_colors = list(set(solution))  # Doar culorile existente
#     solution[index] = random.choice(available_colors)

#     return solution


def mutation(solution, nodes):
    index = random.randint(0, len(solution) - 1)

    neighbor_colors = set()
    for neighbor in graph[index]:
        neighbor_colors.add(solution[neighbor])

    available_colors = set(range(0, nodes)) - neighbor_colors
    if available_colors:
        new_color = random.choice(list(available_colors))
        solution[index] = new_color
    else:
        solution[index] = random.randint(0, nodes - 1)

    return solution


def genetic_algorithm(graph, nodes, population_size = 75, generations = 200):
    population = population_initialization(population_size, nodes)

    for _ in range(generations):
        # selection
        parent1, parent2 = selection(population, graph)
        # parent1, parent2 = selection_enhanced(population, graph)
        # parent1 = tournament_selection(population, graph)
        # parent2 = tournament_selection(population, graph)


        # crossover
        child = crossover(parent1, parent2)

        # mutation
        child = mutation(child, nodes)

        # replacement
        population[random.randint(0, population_size - 1)] = child

        best_solution = min(population, key=lambda x: fitness(x, graph))
        # print (fitness(best_solution, graph))

        if fitness(best_solution, graph) == 0:
            print('Solution found!!!')
            return best_solution
        
    return min(population, key=lambda x: fitness(x, graph))


def visualize_graph(graph, coloring):
    G = nx.Graph()

    # Adaugam nodurile si muchiile in NetworkX
    for node in graph:
        G.add_node(node)
        for neighbor in graph[node]:
            G.add_edge(node, neighbor)

    # Folosim un mapator explicit pentru culori, pentru o vizualizare mai clara
    node_colors = [coloring[node] for node in graph]

    # Setari pentru afisare
    pos = nx.spring_layout(G)  # Algoritm pentru pozitionarea nodurilor
    cmap = plt.colormaps['Set1']  # Inlocuirea cu noul mod de acces la colormap

    nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, cmap=cmap, vmin=1, vmax=max(node_colors))

    # Afisare grafic
    plt.title("Graph with Coloring")
    plt.show()


def verify_coloring(graph, coloring):
    for node in graph:
        for neighbor in graph[node]:
            if coloring[node] == coloring[neighbor]:
                print (f"Conflict between node {node} and neighbor {neighbor}")
                # return False
    # return True


def multiple_runs(graph, nodes, runs):
    best_solutions = []
    for _ in range(runs):
        best_solution = genetic_algorithm(graph, nodes)
        best_solutions.append(best_solution)
    return best_solutions


def genetic_second_layer():
    population = multiple_runs(graph, nodes, 10)
    parent1, parent2 = selection(population, graph)
    child = crossover(parent1, parent2)
    child = mutation(child, nodes)

    population[random.randint(0, len(population) - 1)] = child
    best_solution = min(population, key=lambda x: fitness(x, graph))
        # print (fitness(best_solution, graph))

    if fitness(best_solution, graph) == 0:
        print('Solution found!!!')
        return best_solution
        
    return min(population, key=lambda x: fitness(x, graph))



if __name__ == '__main__':
    graph, nodes, edges = read_file('mug88_1.col')

    # solution = genetic_algorithm(graph, nodes)

    solution = genetic_second_layer()

    # print('Best solution found:\n', solution)
    # print('Number of conflicts:', fitness(solution, graph))
    # print('Number of colors used:', len(set(solution)))

    
    # for i in range(len(solution)):
    #     print(f"Node {i}: Color {solution[i]}")

    print('Size: ', nodes)
    print('True χ: ', len(set(solution)))
    print('Predicted Colors: ', solution)

    verify_coloring(graph, solution)

    visualize_graph(graph, solution)
