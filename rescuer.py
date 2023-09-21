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


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstractAgent):
    def __init__(self, env, config_file, number_of_explorers, preferencia):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.plan = []              # a list of planned actions
        self.rtime = self.TLIM      # for controlling the remaining time
        self.full_map = {}          # the full map of the environment
        self.counter = 0
        self.valid_path = []
        self.preferencia = preferencia
        self.my_cluster = []
        self.my_victims = []
        self.next_victim = None    # the next victim to rescue
        self.number_of_explorers = number_of_explorers
        
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.body.set_state(PhysAgent.IDLE)
    
    def go_save_victims(self, info_coord):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""
        self.counter += 1
        # print(info_coord)
        #cada item no mapa contem as informações de uma coordenada
        #se o dicionario não está na lista de mapas coloca ele
        for key, value in info_coord.items():
            if key not in self.full_map.keys():
                self.full_map[key] = value

        
        if self.counter == self.number_of_explorers:
            self.body.set_state(PhysAgent.ACTIVE)
            print("FULL MAP RECEIVED")
            print("=====================================")
            self.clusters = self.weighted_kmeans_clustering()
            self.my_cluster = self.clusters[self.preferencia] 
            # planning
            self.__planner()         
                
            # print(self.full_map)
        # for item in mapa:
        #     self.list_of_maps.append(item)  
        
        
    def weighted_kmeans_clustering(self):
        # Extrair pesos e coordenadas dos elementos 
        weights = []
        coordinates = []
        for key, value in self.full_map.items():
            if value[0] == 'victim':
                weights.append(value[1])
                coordinates.append(key)

        for key, value in self.full_map.items():
            if value[0] == 'path':
                self.valid_path.append(key)

        # print(weights)
        # print(coordinates)
        # Normalize weights to sum up to 1 (optional)
        total_weight = sum(weights)
        weights = [weight / total_weight for weight in weights]

        # Combine coordinates and weights into a single array
        data = np.column_stack((coordinates, weights))

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
    
    def __planner(self):
        """ A private method that calculates the walk actions to rescue the
        victims. Further actions may be necessary and should be added in the
        deliberata method"""

        # This is a off-line trajectory plan, each element of the list is
        # a pair dx, dy that do the agent walk in the x-axis and/or y-axis
        self.my_cluster.sort(key=lambda x: x[1])
        for point, weight in self.my_cluster:
            self.my_victims.append(point)

        print(f"SO THE VICTIMS I (CLUSTER {self.preferencia}) HAVE TO RESCUE ARE:")
        print(self.my_victims)

    def shortest_path_with_costs(self, start, goal):
        def heuristic(node):
            x1, y1 = node
            x2, y2 = goal
            return abs(x1 - x2) + abs(y1 - y2)  # Distância de Manhattan
        
        def neighbors(node):
            x, y = node
            possible_moves = self.movements  # Movimentos em todas as direções
            valid_neighbors = []

            for dx, dy in possible_moves:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)

                if neighbor in self.visited_cells:
                    if dx == 0 or dy == 0:
                        cost = self.COST_LINE  # Movimento em linha
                    else:
                        cost = self.COST_DIAG  # Movimento em diagonal

                    valid_neighbors.append((neighbor, cost))

            return valid_neighbors
        
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
        
        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)

        # Walk - just one step per deliberation
        result = self.body.walk(dx, dy)

        # Rescue the victim at the current position
        if result == PhysAgent.EXECUTED:
            # check if there is a victim at the current position
            seq = self.body.check_for_victim()
            if seq >= 0:
                res = self.body.first_aid(seq) # True when rescued             

        return True

