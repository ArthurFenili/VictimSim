## EXPLORER AGENT
### @Author: Tacla, UTFPR
### It walks randomly in the environment looking for victims.

import sys
import os
import random
import time
import heapq
from abstract_agent import AbstractAgent
from physical_agent import PhysAgent
from abc import ABC, abstractmethod
from collections import deque


class Explorer(AbstractAgent):
    def __init__(self, env, config_file, rescs, preferencia):
        """ Construtor do agente random on-line
        @param env referencia o ambiente
        @config_file: the absolute path to the explorer's config file
        @param resc referencia o rescuer para poder acorda-lo
        """

        super().__init__(env, config_file)
        
        # Specific initialization for the rescuer
        self.rescs = rescs           # reference to the rescuer agent
        self.rtime = self.TLIM     # remaining time to explore     
        self.path = []             # path executed
        self.returning_path = []   # path to return to base
        self.visited_cells = []    # cells already visited
        self.movements_cost = {}   # cost of each movement
        self.mapa = []             # map of the environment
        self.coordinates_info = {}      # coordinates information
        self.x = 0  # initial relative x position
        self.y = 0  # initial relative y position
        self.preferencia = preferencia
        self.returning = 0 
        if preferencia == 0:
            self.movements = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1), (1, 1), (-1,-1)]
        elif preferencia == 1:
            self.movements = [(-1, 0), (0, 1), (1, 0), (0, -1), (1, -1), (-1, 1), (1, 1), (-1,-1)]
        elif preferencia == 2:
            self.movements = [(0, -1), (-1, 0), (0, 1),(1, 0), (1, -1), (-1, 1), (1, 1), (-1,-1)]
        elif preferencia == 3:
            self.movements = [(1, 0), (0, -1), (-1, 0),(0, 1),(1, -1), (-1, 1), (1, 1), (-1,-1)]

   
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

                if neighbor in self.visited_cells: #Só considera vizinhos em células já visitadas
                    if dx == 0 or dy == 0:
                        cost = self.COST_LINE  # Movimento em linha
                    else:
                        cost = self.COST_DIAG  # Movimento em diagonal

                    valid_neighbors.append((neighbor, cost))

            return valid_neighbors
        
        open_list = [(0, start)]  # Lista de prioridade (f_score, nó)
        came_from = {}  # Dicionário para rastrear o caminho
        g_score = {node: float('inf') for node in self.visited_cells}
        g_score[start] = 0

        while open_list:
            current_f, current_node = heapq.heappop(open_list)

            if current_node == goal:
                path = []
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(start)
                # print("path: ", path)
                #print("cost: ", g_score[goal])
                return path, g_score[goal]  # Retorna o caminho e o custo total

            for neighbor, cost in neighbors(current_node):
                tentative_g = g_score[current_node] + cost

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_list, (f_score, neighbor))

        return None, float('inf')  # Não foi possível encontrar um caminho

    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""
        def update_time(dx, dy):
            # Update remaining time
            if dx != 0 and dy != 0:
                self.rtime -= self.COST_DIAG
            else:
                self.rtime -= self.COST_LINE

        # map(self.base, (0,0))
        new_path_find = False
        x,y = self.x, self.y
 
        # No more actions, time almost ended
        if self.returning:
            if self.rtime <= 5.0 and self.body.at_base(): 
                # time to wake up the rescuer
                # pass the walls and the victims (here, they're empty)
                #print(self.mapa)
                print(f"{self.NAME} I believe I've remaining time of {self.rtime:.1f}")
                
                # Call go_save_victims for each rescuer in the list
                for resc in self.rescs:
                    resc.go_save_victims(self.coordinates_info)

                return False
            else:
                x2, y2 = self.returning_path.pop()
                dx2 = x2 - x
                dy2 = y2 - y
                self.body.walk(dx2, dy2)
                self.x = x + dx2
                self.y = y + dy2
                update_time(dx2, dy2)
                return True

        
        
        current_neighbors = []
        self.visited_cells.append((x,y))

        for mov in self.movements:
            (nx, ny) = (x + mov[0], y + mov[1])
            obstacles = self.body.check_obstacles()
            obst = {(0,-1):obstacles[0] ,(1,-1):obstacles[1] ,(1,0):obstacles[2] ,(1,1):obstacles[3] ,(0,1):obstacles[4] ,(-1,1):obstacles[5] ,(-1,0):obstacles[6] ,(-1,-1):obstacles[7] }
            if obst[(mov[0], mov[1])] == 0:
                current_neighbors.append((nx, ny))
            else:
                self.coordinates_info[(nx,ny)] = ['obstacle' , 0]

        for neighbor in current_neighbors:
            if neighbor not in self.visited_cells:
                dx = neighbor[0] - x
                dy = neighbor[1] - y
                new_path_find = True
                break
        
        current_pos = (x,y)
        self.returning_path, cost = self.shortest_path_with_costs(current_pos, (0,0))
        #print(self.returning_path)
        #print("remaining time ", self.rtime)
        if self.returning_path:
            if self.rtime - cost <= 5.0 and self.rtime - cost > 0.0:
                self.returning = 1
                print("returning")
                return True

        if not new_path_find:
            (dx,dy) = self.path.pop()  # Remove o nó atual do caminho ao retroceder
        else:
            self.path.append((-dx,-dy))

        # Moves the body to another position
        result = self.body.walk(dx, dy)
        self.x = x+dx
        self.y = y+dy
        update_time(dx, dy)

        # Test the result of the walk action
        if result == PhysAgent.BUMPED:
            walls = 1 
            self.coordinates_info[(self.x,self.y)] = ['obstacle' , 0]
            # print(self.name() + ": wall or grid limit reached")

        if result == PhysAgent.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.coordinates_info[(self.x,self.y)] = ['path' , 0]
            seq = self.body.check_for_victim()
            if seq >= 0:
                vs = self.body.read_vital_signals(seq)
                self.rtime -= self.COST_READ
                if vs[7] == 1:
                    peso = 6
                elif vs[7] == 2:
                    peso = 3
                elif vs[7] == 3:
                    peso = 2
                elif vs[7] == 4:
                    peso = 1
                self.coordinates_info[(self.x,self.y)] = ['victim' , peso, vs]
                # print("exp: read vital signals of " + str(seq))
                # print(vs)

        self.visited_cells.append((self.x,self.y))
        
        return True
