## EXPLORER AGENT
### @Author: Tacla, UTFPR
### It walks randomly in the environment looking for victims.

import sys
import os
import random
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
        self.path = []             # path to the base
        self.visited_cells = []    # cells already visited
        self.current_neighbors = [] # neighbors of the current cell
        if preferencia == 0:
            self.movements = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1), (1, 1), (-1,-1)]
        elif preferencia == 1:
            self.movements = [(-1, 0), (0, 1), (1, 0), (0, -1), (1, -1), (-1, 1), (1, 1), (-1,-1)]
        elif preferencia == 2:
            self.movements = [(0, -1), (-1, 0), (0, 1),(1, 0), (1, -1), (-1, 1), (1, 1), (-1,-1)]
        elif preferencia == 3:
            self.movements = [(1, 0), (0, -1), (-1, 0),(0, 1),(1, -1), (-1, 1), (1, 1), (-1,-1)]

   
    
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        new_path_find = False
        # No more actions, time almost ended
        if self.rtime < 10.0: 
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            print(f"{self.NAME} I believe I've remaining time of {self.rtime:.1f}")
            self.resc.go_save_victims([],[])
            return False
        


        if self.body.at_base():
            x,y = self.body.x_base, self.body.y_base
        else:
            x,y = self.body.x, self.body.y

        
        self.current_neighbors = []
        self.visited_cells.append((x,y))
  
        for mov in self.movements:
            (nx, ny) = (x + mov[0], y + mov[1])
            self.current_neighbors.append((nx, ny))
        
        for neighbor in self.current_neighbors:
            #print(neighbor)
            if neighbor not in self.visited_cells:
                dx = neighbor[0] - x
                dy = neighbor[1] - y
                self.visited_cells.append(neighbor)
                new_path_find = True
                break

        if not new_path_find:
            (dx,dy) = self.path.pop()  # Remove o nó atual do caminho ao retroceder
        else:
            self.path.append((-dx,-dy))

        # Check the neighborhood obstacles
        obstacles = self.body.check_obstacles()

        # Moves the body to another position
        result = self.body.walk(dx, dy)

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
                
        return True

