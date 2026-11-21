import numpy as np
from map_data import MapData
import math

class ACOPlanner:
    def __init__(self, grid:np.ndarray, start: tuple, goal: tuple,
                  m_ants=30, tau_min=0.05, tau_max=5.0, max_iter=50,
                    alfa=1.0, beta=2.0, rho=0.15, Q_const=100.0, q_0 = 0.7,
                    max_steps = 500, tau_0 = 1.0):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.m_ants = m_ants
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.max_iter = max_iter
        self.alfa = alfa
        self.beta = beta
        self.rho = rho
        self.Q_const = Q_const
        self.q_0 = q_0
        self.max_steps = max_steps
        self.height, self.width = self.grid.shape
        self.tau_0 = tau_0  
        self.tau = np.full(self.grid.shape, tau_0)
        self.best_path = None
        self.best_cost = np.inf
        self.rng = np.random.default_rng(seed=None)
        self._validate_params()

    def _validate_params(self):
        if not (0 < self.rho < 1):
            raise ValueError(f"rho musi być w przedziale (0, 1), podano {self.rho}")
        if not (0 <= self.q_0 <= 1):
            raise ValueError(f"q_0 musi być w przedziale [0, 1], podano {self.q_0}")
        if not (self.alfa >= 0 and self.beta >= 0):
            raise ValueError(f"alfa i beta muszą być >= 0, podano alfa={self.alfa}, beta={self.beta}")
        if not (0 < self.tau_min < self.tau_max):
            raise ValueError(f"Musi być 0 < tau_min < tau_max, podano {self.tau_min}, {self.tau_max}")
        if not (self.tau_min <= self.tau_0 <= self.tau_max):
            raise ValueError(f"tau_0 musi leżeć w [tau_min, tau_max], podano {self.tau_0}")
        if not (self.Q_const > 0):
            raise ValueError(f"Q_const musi być > 0, podano {self.Q_const}")
        if not (self.m_ants >= 1):
            raise ValueError(f"Ilość mrówek musi być >= 1, podano {self.m_ants}")
        if not (self.max_iter >= 1):
            raise ValueError(f"Ilość iteracji musi być >= 1, podano {self.max_iter}")
        if not (self.max_steps >= 1):
            raise ValueError(f"max_steps musi być >= 1, podano {self.max_steps}")

        self._check_cell(self.start, "start")
        self._check_cell(self.goal, "goal")

    def _check_cell(self, cell, name):
        r, c = cell
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise ValueError(f"{name} {cell} jest poza mapą {self.height}x{self.width}")
        if self.grid[r, c] != 0:
            raise ValueError(f"{name} {cell} leży na przeszkodzie (po dylatacji)")
        

    def get_neighbors(self, cell, visited):
        MOVES = [(-1,0),(1,0),(0,-1), (0,1), 
                 (-1,1),(1,1),(-1,-1), (1,-1)] #przesunięcia punktów sąsiednich w okół aktualnego punktu

        r, c = cell
        neighbors = []
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc

            # sprawdzenie 3 warunków
            if not (0 <= nr < self.height and 0 <= nc < self.width):
                continue

            if self.grid[nr,nc] != 0:
                continue

            if (nr, nc) in visited:
                continue

            neighbors.append((nr,nc))

        return neighbors

    def heuristic(self,cell,neighbor) -> float:

        r,c = cell
        nr, nc = neighbor
        gr, gc = self.goal

        #jeżeli oba przesunięcia niezerowe -> ruch po skosie, jakieś zerowe -> ruch prosty 
        dr,dc = nr-r, nc-c
        if (dr != 0 and dc != 0):
            step_cost = math.sqrt(2)
        else:
            step_cost = 1.0

        dist_to_goal = math.hypot(nr - gr, nc - gc) #wzór na pierwiastek z sumy kwadratów

        return 1.0/ (step_cost + dist_to_goal)

    


            






            

        



