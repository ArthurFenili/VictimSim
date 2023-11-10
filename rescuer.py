##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim

import os
import random
from abstract_agent import AbstractAgent
from physical_agent import PhysAgent
from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans
from deap import base, creator, tools, algorithms
import heapq
import csv

## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstractAgent):
    def __init__(self, env, config_file, number_of_explorers, preferencia):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.my_plan = []              # a list of planned actions
        self.my_victims = []        # a list of victims to rescue
        self.rtime = self.TLIM      # for controlling the remaining time
        self.full_map = {}          # the full map of the environment
        self.counter = 0
        self.x = 0      # initial relative x position
        self.y = 0      # initial relative y position
        self.valid_path = [(self.x,self.y)]
        self.movements = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1), (1, 1), (-1,-1)]
        self.preferencia = preferencia
        self.my_cluster = []
        self.number_of_explorers = number_of_explorers
        self.savedvictims = []
        
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.body.set_state(PhysAgent.IDLE)
    
    def go_save_victims(self, info_coord):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""
        
        #print("INFOCOORD: ", info_coord)

        #cada item no mapa contem as informações de uma coordenada
        #se o dicionario não está na lista de mapas coloca ele
        for key, value in info_coord.items():
            if key not in self.full_map.keys():
                self.full_map[key] = value

        self.counter += 1

        if self.counter == self.number_of_explorers:
            self.body.set_state(PhysAgent.ACTIVE)
            print("FULL MAP RECEIVED")
            print("=====================================")
            self.clusters = self.weighted_kmeans_clustering()
            if len(self.clusters) < self.preferencia + 1:
                self.my_cluster = []
            else:
                self.my_cluster = self.clusters[self.preferencia] 

            # planning
            self.my_plan = self.__planner()         
        
            # print(self.full_map)
        # for item in mapa:
        #     self.list_of_maps.append(item)  
        
        
    def weighted_kmeans_clustering(self):
        # Extrair pesos e coordenadas dos elementos 
        weights = []
        coordinates = []
        number_of_victims = 0
        for key, value in self.full_map.items():
            if value[0] == 'victim':
                number_of_victims += 1
                weights.append(value[1])
                coordinates.append(key)

        for key, value in self.full_map.items():
            if value[0] != 'obstacle':
                self.valid_path.append(key)

        print(f"Total de caminhos explorados: {len(self.full_map)}")
        # print(weights)
        # print(coordinates)
        # Normalize weights to sum up to 1 (optional)
        total_weight = sum(weights)
        weights = [weight / total_weight for weight in weights]

        # Combine coordinates and weights into a single array
        data = np.column_stack((coordinates, weights))
        print(data)
        if number_of_victims < self.number_of_explorers:
            self.number_of_explorers = number_of_victims

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=self.number_of_explorers, n_init='auto', random_state=0)
        kmeans.fit(data)

        # Assign each point to a cluster
        cluster_assignments = kmeans.labels_

        # Organize points into clusters
        clusters = [[] for _ in range(self.number_of_explorers)]
        for i, coord in enumerate(coordinates):
            cluster_id = cluster_assignments[i]
            clusters[cluster_id].append((coord, weights[i]))

        return clusters
    
    def shortest_path_with_costs(self, start, goal):
        if goal != (0,0):
            print("start: ", start) 
            print("goal: ", goal)
        def heuristic(node):
            x1, y1 = node
            x2, y2 = goal
            return abs(x1 - x2) + abs(y1 - y2)  # Distância de Manhattan
        
        def neighbors(node):
            x, y = node
            valid_neighbors = []

            for dx, dy in self.movements:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if neighbor in self.valid_path:
                    if dx == 0 or dy == 0:
                        cost = self.COST_LINE  # Movimento em linha
                    else:
                        cost = self.COST_DIAG  # Movimento em diagonal

                    valid_neighbors.append((neighbor, cost))

            return valid_neighbors
        
        open_list = [(0, start)]  # Lista de prioridade (f, nó)
        came_from = {}  # Dicionário para rastrear o caminho
        g_score = {node: float('inf') for node in self.valid_path}
        g_score[start] = 0

        while open_list:
            current_f, current_node = heapq.heappop(open_list)
            if current_node == goal:
                path2 = []
                while current_node in came_from:
                    path2.append(current_node)
                    current_node = came_from[current_node]
                path2.append(start)
                path2.reverse()
                return path2, g_score[goal]  # Retorna o caminho e o custo total

            for neighbor, cost in neighbors(current_node):
                tentative_g = g_score[current_node] + cost
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_list, (f_score, neighbor))

        return path2, g_score[goal]  # Não foi possível encontrar um caminho
    
    def find_best_route(self):
        # Define your objectives as a list of coordinates (e.g., (x, y))
        objectives = self.my_victims

        # Define the number of individuals in the population
        population_size = 500

        # Define the maximum number of generations
        max_generations = 1000

        # Define the mutation rate
        mutation_rate = 0.1

        # Define the fitness function (you can customize this)
        def fitness(chromosome):
            # Calculate the total distance traveled for a given chromosome
            total_distance = 0
            for i in range(len(chromosome) - 1):
                x1, y1 = chromosome[i]
                x2, y2 = chromosome[i + 1]
                distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                total_distance += distance
            return 1 / total_distance  # Inverse of distance as a fitness

        # Generate an initial population
        population = [random.sample(objectives, len(objectives)) for _ in range(population_size)]

        # Main genetic algorithm loop
        for generation in range(max_generations):
            # Evaluate the fitness of each chromosome
            fitness_values = [fitness(chromosome) for chromosome in population]

            # Select parents based on fitness
            parents = random.choices(population, weights=fitness_values, k=population_size)

            # Create a new population through crossover and mutation
            new_population = []
            for _ in range(population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = parent1[:]
                
                if random.random() < mutation_rate:
                    # Apply mutation by swapping two objectives
                    idx1, idx2 = random.sample(range(len(child)), 2)
                    child[idx1], child[idx2] = child[idx2], child[idx1]

                new_population.append(child)

            # Replace the old population with the new population
            population = new_population

        # Get the best chromosome from the final population
        best_chromosome = max(population, key=fitness)

        print("Best sequence of objectives:", best_chromosome)
        print("Total distance for the best sequence:", sum(fitness_values))

        # You can visualize the best sequence of objectives on the map using the coordinates in `best_chromosome`.

        return best_chromosome


    def __planner(self):
        """ A private method that calculates the walk actions to rescue the
        victims. Further actions may be necessary and should be added in the
        deliberata method"""

        # This is a off-line trajectory plan, each element of the list is
        # a pair dx, dy that do the agent walk in the x-axis and/or y-axis
        self.my_cluster.sort(key=lambda x: x[1])
        for point, weight in self.my_cluster:
            self.my_victims.append(point)
        self.my_victims.reverse()
        print(f"SO THE VICTIMS I (CLUSTER {self.preferencia}) HAVE TO RESCUE ARE:")
        print(self.my_victims)

        clustertxt = "cluster" + str(self.preferencia) + ".csv"
        with open(clustertxt, "w", newline="") as arquivo_csv:
            # Cria um objeto escritor CSV
            escritor_csv = csv.writer(arquivo_csv)
            
            for coord in self.my_victims:
                for key, value in self.full_map.items():
                    if coord == key:
                        new_linha = list(coord)
                        new_linha.insert(0, value[2][0])
                        new_linha.append(0)
                        if value[2][7] == 1:
                            new_linha.append("critical")
                        if value[2][7] == 2:
                            new_linha.append("unstable")
                        if value[2][7] == 3:
                            new_linha.append("potentially stable")
                        if value[2][7] == 4:
                            new_linha.append("stable")
                        escritor_csv.writerow(new_linha)
                

        # self.my_victims.insert(0, (0,0))

        melhor_rota = self.find_best_route()
        print(f"THE BEST ROUTE IS: {melhor_rota}")
        # map_coordinates = []
        # for key, value in self.full_map.items():
        #     if value[0] == 'victim':
        #         map_coordinates.append(key)

        # # Função para calcular a distância entre duas coordenadas
        # def distancia(coord1, coord2):
        #     return np.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2) #ver o melhor calculo de distancia

        #def aptidao(individual):
        #    dist = distancia((0,0), self.my_victims[individual[0]])  # Distância da base à primeira coordenada
        #    for i in range(len(individual)-1):
        #        dist += distancia(self.my_victims[individual[i]], self.my_victims[individual[i+1]])
        #    dist += distancia(self.my_victims[individual[-1]], (0,0))  # Distância da última coordenada à base
        #    return dist,

        # # Definindo o problema como um problema de minimização
        # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # creator.create("Individual", list, fitness=creator.FitnessMin)

        # toolbox = base.Toolbox()

        # # Inicialização
        # toolbox.register("indices", random.sample, range(len(self.my_victims)), len(self.my_victims))
        # toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        # toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # # Operadores
        # toolbox.register("evaluate", aptidao)
        # toolbox.register("mate", tools.cxOrdered)
        # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        # toolbox.register("select", tools.selTournament, tournsize=3)

        # # Parâmetros do algoritmo genético
        # pop = toolbox.population(n=100)
        # hof = tools.HallOfFame(1)
        # stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register("Avg", np.mean)
        # stats.register("Std", np.std)
        # stats.register("Min", np.min)
        # stats.register("Max", np.max)

        # # Executando o algoritmo genético
        # pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, 
        #                                 stats=stats, halloffame=hof, verbose=True)

        # # Imprimindo a melhor rota
        # print("Melhor rota:", [self.my_victims[i] for i in hof[0]])
        # melhor_rota =  [self.my_victims[i] for i in hof[0]]
        # print(f"THE BEST ROUTE IS: {melhor_rota}")
        x_aux, y_aux = self.x, self.y
        time_aux = self.rtime
        
        plan_aux = []
        # #THE PLAN HAS TO HAVE ONLY DX,DY MOVEMENTS

        for victim in melhor_rota:
            print(f"CURRENT VICTIM: {victim}")
            self.savedvictims.append((victim[0], victim[1]))
            victim_path, victim_cost = self.shortest_path_with_costs((x_aux, y_aux), (victim[0], victim[1]))
            return_path_now, return_cost_now = self.shortest_path_with_costs((x_aux,y_aux), (self.x, self.y))
            return_path_later, return_cost_later = self.shortest_path_with_costs((victim[0], victim[1]), (self.x, self.y))
            if return_cost_later + self.COST_FIRST_AID >= time_aux: 
                #if the returning path from the next victim takes more time than I have, I have to go back to the base
                plan_aux.extend(return_path_now[1:])
                time_aux -= return_cost_now
                break
            elif victim == melhor_rota[-1]:
                #if it is the last victim, I have 
                # to go back to the base
                plan_aux.extend(victim_path[1:])
                plan_aux.extend(return_path_later[1:])
                break
            else:
                plan_aux.extend(victim_path[1:])
                print("path" ,victim_path)
                time_aux -= victim_cost
                x_aux = victim[0]
                y_aux = victim[1]

        print(x_aux, y_aux)
        print(f"MY PLAN: {plan_aux}")

        plan_aux.insert(0, (0,0))
        plan_aux.reverse()
        return plan_aux       

    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """
        # print(self.full_map)
        #clusters = self.weighted_kmeans_clustering()
        # for cluster in clusters:
        #     print(cluster)
        # self.next_victim = self.my_victims.pop()
        #print(f"MY PLAN: {self.my_plan}")
        # No more actions to do
        if self.my_plan == []:  # empty list, no more actions to do
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        (x,y) = self.my_plan.pop()
        dx = x - self.x
        dy = y - self.y
        # # Walk - just one step per deliberation
        result = self.body.walk(dx,dy)
        self.x += dx
        self.y += dy

        # Rescue the victim at the current position
        if result == PhysAgent.EXECUTED:
            # check if there is a victim at the current position
            seq = self.body.check_for_victim()
            if seq >= 0:
                res = self.body.first_aid(seq) # True when rescued
                if(res == True): 
                    savedtxt = "salvas" + str(self.preferencia) + ".csv"
                    with open(savedtxt, "w", newline="") as arquivo_csv:
                        # Cria um objeto escritor CSV
                        escritor_csv = csv.writer(arquivo_csv)
                        
                        for coord in self.savedvictims:
                            for key, value in self.full_map.items():
                                if coord == key:
                                    new_linha = list(coord)
                                    new_linha.insert(0, value[2][0])
                                    new_linha.append(value[2][7])
                                    if value[2][7] == 1:
                                        new_linha.append("critical")
                                    if value[2][7] == 2:
                                        new_linha.append("unstable")
                                    if value[2][7] == 3:
                                        new_linha.append("potentially stable")
                                    if value[2][7] == 4:
                                        new_linha.append("stable")
                                    escritor_csv.writerow(new_linha)         

        return True

