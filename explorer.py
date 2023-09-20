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
    def __init__(self, env, config_file, resc, preferencia):
        """ Construtor do agente random on-line
        @param env referencia o ambiente
        @config_file: the absolute path to the explorer's config file
        @param resc referencia o rescuer para poder acorda-lo
        """

        super().__init__(env, config_file)
        
        # Specific initialization for the rescuer
        self.resc = resc           # reference to the rescuer agent
        self.rtime = self.TLIM     # remaining time to explore     
        self.path = []             # path executed
        self.returning_path = []   # path to return to base
        self.came_from = {}        # path to return to base in dictionary format
        self.visited_cells = []    # cells already visited
        self.current_neighbors = [] # neighbors of the current cell
        self.movements_cost = {}   # cost of each movement
        self.base = (self.body.x, self.body.y) # base position
        self.preferencia = preferencia
        if preferencia == 0:
            self.movements = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1), (1, 1), (-1,-1)]
        elif preferencia == 1:
            self.movements = [(-1, 0), (0, 1), (1, 0), (0, -1), (1, -1), (-1, 1), (1, 1), (-1,-1)]
        elif preferencia == 2:
            self.movements = [(0, -1), (-1, 0), (0, 1),(1, 0), (1, -1), (-1, 1), (1, 1), (-1,-1)]
        elif preferencia == 3:
            self.movements = [(1, 0), (0, -1), (-1, 0),(0, 1),(1, -1), (-1, 1), (1, 1), (-1,-1)]

   
    def shortest_path_with_costs(self, start, goal):
        print("start: ", start)
        print("goal: ", goal)
        def heuristic(node):
            x1, y1 = node
            x2, y2 = goal
            return abs(x1 - x2) + abs(y1 - y2)  # Distância de Manhattan
        valid_neighbors = []
        for neighbor in self.current_neighbors:
            if neighbor in self.visited_cells:
                valid_neighbors.append(neighbor)

        open_list = [(0, start)]  # Lista de prioridade (f, nó)
        came_from = {}  # Dicionário para rastrear o caminho
        custo = {node: float('inf') for node in self.visited_cells}
        custo[start] = 0

        while open_list:
            current_f, current_node = heapq.heappop(open_list)
            #print(came_from)
            if current_node == goal:
                path = []
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                print("path: ", path)
                print("custo: ", custo[goal])
                return path, custo[goal]  # Retorna o caminho e o custo total

            for neighbor in valid_neighbors:
                dx = neighbor[0] - current_node[0]
                dy = neighbor[1] - current_node[1]
                if dx != 0 and dy != 0:
                    novo_custo =  custo[current_node] + self.COST_DIAG
                else:
                    novo_custo =  custo[current_node] + self.COST_LINE

                if novo_custo <  custo[neighbor]:
                    self.came_from[current_node] = neighbor
                    custo[neighbor] = novo_custo
                    f_score = novo_custo + heuristic(neighbor)
                    heapq.heappush(open_list, (f_score, neighbor))


        return None, float('inf')  # Não foi possível encontrar um caminho

    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""
        # def create_grid(height, width, default_value=0):
        #     grid = []
        #     for _ in range(height):
        #         row = [default_value] * width
        #         grid.append(row)
        #     return grid
        new_path_find = False
        # No more actions, time almost ended
        if self.rtime < 5.0 and self.body.at_base(): 
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            print(f"{self.NAME} I believe I've remaining time of {self.rtime:.1f}")
            self.resc.go_save_victims([],[])
            return False

        x,y = self.body.x, self.body.y
        self.current_neighbors = []
        self.visited_cells.append(self.base)


        for mov in self.movements:
            (nx, ny) = (x + mov[0], y + mov[1])
            if nx >= 0 and nx < self.env.dic["GRID_WIDTH"] and ny >= 0 and ny < self.env.dic["GRID_HEIGHT"] and self.env.walls[nx][ny] == 0:
                self.current_neighbors.append((nx, ny))
        
        for neighbor in self.current_neighbors:
            #print(neighbor)
            if neighbor not in self.visited_cells:
                dx = neighbor[0] - x
                dy = neighbor[1] - y
                new_path_find = True
                break
        
        current_pos = (x,y)
        self.returning_path, cost = self.shortest_path_with_costs(current_pos, self.base)
        #print(self.returning_path)
        if self.returning_path:
            if self.rtime - cost < 5.0:
                (dx, dy) = self.returning_path.pop()
        else:
            if not new_path_find:
                (dx,dy) = self.path.pop()  # Remove o nó atual do caminho ao retroceder
            else:
                self.path.append((-dx,-dy))

        # Check the neighborhood obstacles
        obstacles = self.body.check_obstacles()

        # Moves the body to another position
        result = self.body.walk(dx, dy)
        ex_pos = current_pos
        current_pos = self.body.x, self.body.y

        # Update remaining time
        if dx != 0 and dy != 0:
            self.rtime -= self.COST_DIAG
        else:
            self.rtime -= self.COST_LINE

        # Test the result of the walk action
        if result == PhysAgent.BUMPED:
            walls = 1  # build the map- to do
            # print(self.name() + ": wall or grid limit reached")

        if result == PhysAgent.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            seq = self.body.check_for_victim()
            if seq >= 0:
                vs = self.body.read_vital_signals(seq)
                self.rtime -= self.COST_READ
                # print("exp: read vital signals of " + str(seq))
                # print(vs)
        
        self.visited_cells.append(current_pos)
        return True

