import numpy as np
import math

class ACOPlanner:
    def __init__(self, grid:np.ndarray, start: tuple, goal: tuple,
                  m_ants=30, tau_min=0.05, tau_max=5.0, max_iter=50,
                    alfa=1.0, beta=2.0, rho=0.15, Q_const=100.0, q_0 = 0.7,
                    max_steps = 500, tau_0 = 1.0, lam = 0.0, seed=None):
        self.grid = grid
        self.start = tuple(start)
        self.goal = tuple(goal)
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
        self.rng = np.random.default_rng(seed)
        self.lam = lam
        self._validate_params()
        self._precompute()

    def _precompute(self):
        """wolne komórki i heurystyka^beta dla całej mapy - liczone raz """
        self._free = (self.grid == 0).tolist()

        gr, gc = self.goal
        rows, cols = np.indices(self.grid.shape)
        dist = np.hypot(rows - gr, cols - gc)                       # odległość każdej komórki do celu
        self._eta_straight = ((1.0 / (1.0 + dist)) ** self.beta).tolist()            # ruch prosty
        self._eta_diag = ((1.0 / (math.sqrt(2) + dist)) ** self.beta).tolist()      # ruch po skosie
        self._refresh_tau()

    def _refresh_tau(self):
        """tau^alfa dla całej mapy - liczone raz na iterację, po aktualizacji feromonu."""
        self._tau_a = (self.tau ** self.alfa).tolist()

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

    def _check_cell(self, cell:tuple, name):
        r, c = cell
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise ValueError(f"{name} {cell} jest poza mapą {self.height}x{self.width}")
        if self.grid[r, c] != 0:
            raise ValueError(f"{name} {cell} leży na przeszkodzie (po dylatacji)")
        
    # zebranie wszystkich sąsiadów
    def get_neighbors(self, cell:tuple, visited):
        MOVES = [(-1,0),(1,0),(0,-1), (0,1), 
                 (-1,1),(1,1),(-1,-1), (1,-1)] #przesunięcia punktów sąsiednich w okół aktualnego punktu

        r, c = cell
        neighbors = []
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc

            # sprawdzenie 3 warunków
            if not (0 <= nr < self.height and 0 <= nc < self.width):
                continue

            if not self._free[nr][nc]:
                continue

            if (nr, nc) in visited:
                continue

            neighbors.append((nr,nc))

        return neighbors

    # informacja heurestyczna - im bliżejk celu tym większa wartość zwracana przez funkcję
    def heuristic(self,cell: tuple[int, int],neighbor: tuple[int, int]) -> float:

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

    # Wybór następnej komórki do której idzie mrówka
    def choose_next_cell(self, cell: tuple[int, int], neighbors: list[tuple[int, int]]) -> tuple:
        r, c = cell
        tau_a = self._tau_a
        a = []
        for nr, nc in neighbors:
            eta = self._eta_diag if (nr != r and nc != c) else self._eta_straight
            a.append(tau_a[nr][nc] * eta[nr][nc])     # tau^alfa * eta^beta - już policzone

        if self.rng.random() <= self.q_0:              # eksploatacja
            return neighbors[a.index(max(a))]

        # eksploracja
        threshold = self.rng.random() * sum(a)
        cumulative = 0.0
        for cell_n, value in zip(neighbors, a):
            cumulative += value
            if cumulative >= threshold:
                return cell_n
        return neighbors[-1]                           # zabezpieczenie przed błędem zaokrągleń

    def construct_path(self):  # ścieżka jednej mrówki albo None
        path = [self.start]
        visited = {self.start}
        current = self.start

        for _ in range(self.max_steps):
            if current == self.goal:
                return path
            
            neighbors = self.get_neighbors(current,visited)

            if not neighbors:
                path.pop() # usuwa ostatnią komórkę ze scieżki
                if not path:
                    return None

                current = path[-1]
                continue

            next_cell = self.choose_next_cell(current,neighbors)
            path.append(next_cell)
            visited.add(next_cell)

            current = next_cell

        if current == self.goal:
            return path
        else:
            return None


    def path_cost(self, path: list[tuple[int, int]]) -> float: # Ocena jakości trasy funkcją kosztu J_k
        
        L = 0
        delta_theta = 0

        for a, b in zip(path, path[1:]):
            r_1, c_1 = a
            r_2, c_2 = b

            dr = r_2 - r_1
            dc = c_2 - c_1

            
            if (dr != 0 and dc != 0):
                step_cost = math.sqrt(2)
            else:
                step_cost = 1

            L += step_cost # suma długości ścieżki danej mrówki

        for (r0, c0), (r1, c1), (r2, c2) in zip (path, path[1:], path[2:]):
            dr1 = r1 - r0
            dc1 = c1 - c0
            dr2 = r2 - r1
            dc2 = c2 - c1

            theta1 = math.atan2(dr1,dc1)
            theta2 = math.atan2(dr2,dc2)

            d = theta2 - theta1 
            d = (d + math.pi) % (2 * math.pi) - math.pi # przejście do zakresu [0,2pi] --> modulo z dwóch pi --> powrót do zakresu [-pi,pi]
            delta_theta += abs(d)


        return L + self.lam * delta_theta

            
    def update_pheromones(self, best_path, best_cost):  # Krok 5 -> parowanie, depozyt, przycięcie
        self.tau *= (1 - self.rho) # parowanie - zmniejszenie wszystkich wartości feromonu
        if best_path is not None: #aktualizacja feromonu
            delta_tau = self.Q_const/best_cost
            rows, cols = zip(*best_path)
            self.tau[rows,cols] += delta_tau

        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)
        self._refresh_tau()

    def run(self):
        self.history = []

        for i in range(self.max_iter):
            paths = []
            for _ in range(self.m_ants):
                paths.append(self.construct_path())

            successful = []
            for p in paths: # Wybór ścieżek mrówek tylko tych, które dosżły do celu
                if p is not None:
                    successful.append(p)

            paths = successful

            if paths:
                costs = []
                for p in paths:
                    costs.append(self.path_cost(p))

                i_best = int(np.argmin(costs))

                if costs[i_best] < self.best_cost:
                    self.best_cost = costs[i_best]
                    self.best_path = paths[i_best]

            #wariant MMAS, global best
            self.update_pheromones(self.best_path, self.best_cost)

            self.history.append(self.best_cost)
            print(f"iter {i + 1}/{self.max_iter}: doszło {len(paths)}/{self.m_ants}, "
                f"best = {self.best_cost:.2f}")

        return self.best_path

