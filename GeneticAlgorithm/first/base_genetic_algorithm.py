import random


def read_file(filename):
    graph = {}
    with open(filename, 'r') as file:
        line = file.readline().strip().split()
        nodes = int(line[2])
        edges = int(line[3])

        for i in range(1, nodes + 1):
            graph[i] = []

        for _ in range(edges):
            line = file.readline().strip().split()
            u = int(line[1])
            v = int(line[2])
            graph[u].append(v)
            graph[v].append(u)
    
    return graph, nodes, edges


def fitness(solution, graph):
    conflicts = 0
    for node in graph:
        for neighbor in graph[node]:
            if node < neighbor:
                if solution[node - 1] == solution[neighbor - 1]:
                    conflicts += 1

    # penalty = len(set(solution)) * 10
    # de la 50+ culori am ajuns la valori de 40 (chiar 34)
    # aparent asta scade super mult din culori

    # return conflicts + penalty
    return conflicts


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


def crossover(parent1, parent2):
    # point = random.randint(1, len(parent1) - 1)
    # child = parent1[:point] + parent2[point:]
    # # pot sa mai adaug un child ca sa diversific (gen sa le inversez) parent2[:point] + parent1[point:]
    # return child
    child = []
    
    parent1_colors = set(parent1)
    parent2_colors = set(parent2)

    common_colors = parent1_colors.intersection(parent2_colors)

    for i in range(len(parent1)):
        if common_colors:
            child.append(random.choice(list(common_colors)))
            # am ales culorile comune si deja am scazut la 12/13 culori
        else:
            # child.append(random.choice([parent1[i], parent2[i]]))
            color1 = parent1[i]
            color2 = parent2[i]

            parent1_frq = parent1.count(color1)
            parent2_frq = parent2.count(color2)

            if parent1_frq > parent2_frq:
                child.append(color1)
            else:
                child.append(color2)

    return child


def mutation(solution, nodes):
    index = random.randint(0, len(solution) - 1)
    solution[index] = random.randint(1, nodes)
    return solution


def genetic_algorithm(graph, nodes, population_size = 20, generations = 200):
    population = population_initialization(population_size, nodes)

    for _ in range(generations):
        # selection
        parent1, parent2 = selection(population, graph)

        # crossover
        child = crossover(parent1, parent2)

        # mutation
        child = mutation(child, nodes)

        # replacement
        population[random.randint(0, population_size - 1)] = child

        best_solution = min(population, key=lambda x: fitness(x, graph))

        if fitness(best_solution, graph) == 0:
            print('Solution found!!!')
            return best_solution
        
    return min(population, key=lambda x: fitness(x, graph))


if __name__ == '__main__':
    graph, nodes, edges = read_file('mug88_1.col')

    solution = genetic_algorithm(graph, nodes)

    # print('Best solution found:\n', solution)
    # print('Number of conflicts:', fitness(solution, graph))
    # print('Number of colors used:', len(set(solution)))

    print('Size: ', nodes)
    print('True χ: ', len(set(solution)))
    print('Predicted Colors: ', solution)
