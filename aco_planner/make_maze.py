#!/usr/bin/env python3
"""
Generator labiryntu dla projektu aco_planner.

Tworzy w jednym kroku:
  maze.pgm + maze.yaml   - mapa dla MapData / aco_node (ten sam format co mapa areny)
  maze.world             - świat dla Gazebo (niskie ściany)
  maze_world.launch.py   - launch: Gazebo + TurtleBot3 postawiony na starcie labiryntu
  maze_preview.png       - podgląd: mapa, mapa po dylatacji, najkrótsza trasa
  maze_info.txt          - współrzędne startu/celu, komendy, scenariusze do tuningu

Mapa i świat powstają z TEJ SAMEJ listy ścian, więc zawsze się zgadzają.
Układ 'map' = układ świata Gazebo (origin mapy jest podany w metrach świata).

Użycie:
  python3 make_maze.py                         # domyślnie 5x5, wynik w ~/ros2_ws/src/aco_planner/maps/maze
  python3 make_maze.py --rows 6 --cols 6 --seed 3
  python3 make_maze.py --out ./maze_test --cell 0.9
"""

import argparse
import heapq
import math
import os
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TB3_SHARE = Path("/opt/ros/jazzy/share/turtlebot3_gazebo")
TB3_EMPTY_WORLD = TB3_SHARE / "worlds" / "empty_world.world"
TB3_WORLD_LAUNCH = TB3_SHARE / "launch" / "turtlebot3_world.launch.py"

FREE, OCCUPIED, UNKNOWN = 254, 0, 205          # wartości pikseli jak w map_saver


# ----------------------------------------------------------------------------
# 1. Labirynt jako graf komórek (perfect maze: dokładnie jedna droga między komórkami)
# ----------------------------------------------------------------------------
def generate_maze(rows, cols, seed):
    """Recursive backtracker (DFS). Zwraca zbiory ścian:
    h_walls[(k, c)] - pozioma ściana na linii y = k (k = 0..rows), nad kolumną c
    v_walls[(r, k)] - pionowa ściana na linii x = k (k = 0..cols), obok wiersza r
    Wiersz 0 = dół labiryntu, kolumna 0 = lewa strona."""
    rng = random.Random(seed)
    h_walls = {(k, c) for k in range(rows + 1) for c in range(cols)}
    v_walls = {(r, k) for r in range(rows) for k in range(cols + 1)}

    visited = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        r, c = stack[-1]
        options = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                options.append((nr, nc, dr, dc))
        if not options:
            stack.pop()
            continue
        nr, nc, dr, dc = rng.choice(options)
        if dr == 1:
            h_walls.discard((r + 1, c))
        elif dr == -1:
            h_walls.discard((r, c))
        elif dc == 1:
            v_walls.discard((r, c + 1))
        else:
            v_walls.discard((r, c))
        visited.add((nr, nc))
        stack.append((nr, nc))
    return h_walls, v_walls


# ----------------------------------------------------------------------------
# 2. Ściany jako prostokąty w metrach (sąsiednie odcinki łączone w jeden blok)
# ----------------------------------------------------------------------------
def walls_to_boxes(h_walls, v_walls, rows, cols, cell, thick, x0, y0):
    """Zwraca listę prostokątów (cx, cy, sx, sy) w metrach świata."""
    boxes = []
    half = thick / 2
    # poziome: dla każdej linii k łączymy kolejne kolumny w jeden odcinek
    for k in range(rows + 1):
        c = 0
        while c < cols:
            if (k, c) in h_walls:
                start = c
                while c < cols and (k, c) in h_walls:
                    c += 1
                xa, xb = x0 + start * cell - half, x0 + c * cell + half
                y = y0 + k * cell
                boxes.append(((xa + xb) / 2, y, xb - xa, thick))
            else:
                c += 1
    # pionowe
    for k in range(cols + 1):
        r = 0
        while r < rows:
            if (r, k) in v_walls:
                start = r
                while r < rows and (r, k) in v_walls:
                    r += 1
                ya, yb = y0 + start * cell - half, y0 + r * cell + half
                x = x0 + k * cell
                boxes.append((x, (ya + yb) / 2, thick, yb - ya))
            else:
                r += 1
    return boxes


# ----------------------------------------------------------------------------
# 3. Mapa zajętości (rasteryzacja prostokątów)
# ----------------------------------------------------------------------------
def rasterize(boxes, bounds, res, margin):
    """bounds = (xmin, ymin, xmax, ymax) labiryntu razem ze ścianami.
    Zwraca (obraz uint8 z wierszem 0 = GÓRA, origin_x, origin_y)."""
    xmin, ymin, xmax, ymax = bounds
    ox, oy = xmin - margin, ymin - margin
    width = int(math.ceil((xmax - xmin + 2 * margin) / res))
    height = int(math.ceil((ymax - ymin + 2 * margin) / res))

    # środki komórek; wiersz 0 tablicy = dół mapy (jak po flipud w MapData)
    xs = ox + (np.arange(width) + 0.5) * res
    ys = oy + (np.arange(height) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)

    img = np.full((height, width), UNKNOWN, dtype=np.uint8)
    inside = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
    img[inside] = FREE
    for cx, cy, sx, sy in boxes:
        hit = (np.abs(X - cx) <= sx / 2) & (np.abs(Y - cy) <= sy / 2)
        img[hit] = OCCUPIED

    return np.flipud(img), ox, oy        # PGM zapisuje się od góry


def write_pgm(path, img):
    h, w = img.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(img.tobytes())


def write_yaml(path, image_name, res, ox, oy):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"image: {image_name}\n")
        f.write("mode: trinary\n")
        f.write(f"resolution: {res}\n")
        f.write(f"origin: [{ox:.4f}, {oy:.4f}, 0.0]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")


# ----------------------------------------------------------------------------
# 4. Świat Gazebo (SDF)
# ----------------------------------------------------------------------------
def walls_model_sdf(boxes, height):
    links = []
    for i, (cx, cy, sx, sy) in enumerate(boxes):
        geom = f"<box><size>{sx:.4f} {sy:.4f} {height:.4f}</size></box>"
        links.append(f"""
      <link name="wall_{i}">
        <pose>{cx:.4f} {cy:.4f} {height / 2:.4f} 0 0 0</pose>
        <collision name="collision"><geometry>{geom}</geometry></collision>
        <visual name="visual">
          <geometry>{geom}</geometry>
          <material>
            <ambient>0.55 0.55 0.6 1</ambient>
            <diffuse>0.55 0.55 0.6 1</diffuse>
          </material>
        </visual>
      </link>""")
    return f"""
    <model name="aco_maze_walls">
      <static>true</static>{''.join(links)}
    </model>
"""


FALLBACK_WORLD = """<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="aco_maze">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
"""


def build_world(walls_sdf):
    """Wstawia ściany do pustego świata TurtleBota (ma wszystkie potrzebne pluginy).
    Jeśli go nie ma, używa wbudowanego minimalnego świata."""
    if TB3_EMPTY_WORLD.exists():
        template = TB3_EMPTY_WORLD.read_text(encoding="utf-8")
        source = str(TB3_EMPTY_WORLD)
    else:
        template = FALLBACK_WORLD
        source = "wbudowany szablon (nie znaleziono empty_world.world)"
    idx = template.rfind("</world>")
    if idx == -1:
        raise RuntimeError("Szablon świata nie zawiera </world>")
    world = template[:idx] + walls_sdf + template[idx:]
    ET.fromstring(world.encode("utf-8"))          # sprawdzenie, czy XML jest poprawny
    return world, source


# ----------------------------------------------------------------------------
# 5. Launch: kopia turtlebot3_world.launch.py z podmienionym światem i pozycją
# ----------------------------------------------------------------------------
def build_launch(world_path, x, y):
    if not TB3_WORLD_LAUNCH.exists():
        return None, f"nie znaleziono {TB3_WORLD_LAUNCH}"
    text = TB3_WORLD_LAUNCH.read_text(encoding="utf-8")
    # os.path.join(..., 'worlds', '/abs/maze.world') zwraca samą ścieżkę absolutną
    text, n_world = re.subn(r"""['"]turtlebot3_world\.world['"]""", repr(str(world_path)), text)
    text, n_x = re.subn(r"""(['"]x_pose['"]\s*,\s*default\s*=\s*)['"][-0-9.]+['"]""",
                        rf"\g<1>'{x:.3f}'", text)
    text, n_y = re.subn(r"""(['"]y_pose['"]\s*,\s*default\s*=\s*)['"][-0-9.]+['"]""",
                        rf"\g<1>'{y:.3f}'", text)
    if n_world != 1 or n_x != 1 or n_y != 1:
        return None, (f"nie udało się podmienić (świat: {n_world}, x: {n_x}, y: {n_y}) - "
                      f"zrób to ręcznie, patrz maze_info.txt")
    return text, None


# ----------------------------------------------------------------------------
# 6. Sprawdzenie przejezdności po dylatacji (Dijkstra na siatce 8-sąsiedztwa)
# ----------------------------------------------------------------------------
def dilate(occ, radius_px):
    co = np.arange(-radius_px, radius_px + 1)
    X, Y = np.meshgrid(co, co)
    disk = X ** 2 + Y ** 2 <= radius_px ** 2
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(occ, structure=disk)
    except ImportError:                                  # wersja bez scipy
        out = occ.copy()
        h, w = occ.shape
        rs, cs = np.nonzero(occ)
        for dr, dc in zip(*np.nonzero(disk)):
            dr -= radius_px
            dc -= radius_px
            r2, c2 = rs + dr, cs + dc
            ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
            out[r2[ok], c2[ok]] = True
        return out


def shortest_path(free, start, goal):
    h, w = free.shape
    dist = np.full(free.shape, np.inf)
    dist[start] = 0.0
    prev = {}
    heap = [(0.0, start)]
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
             (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if (r, c) == goal:
            path = [goal]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return path[::-1], d
        if d > dist[r, c]:
            continue
        for dr, dc, cost in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and d + cost < dist[nr, nc]:
                dist[nr, nc] = d + cost
                prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (d + cost, (nr, nc)))
    return None, np.inf


def octile(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Generator labiryntu: mapa + świat Gazebo + launch")
    p.add_argument("--rows", type=int, default=5, help="liczba komórek w pionie")
    p.add_argument("--cols", type=int, default=5, help="liczba komórek w poziomie")
    p.add_argument("--cell", type=float, default=0.8, help="rozstaw ścian [m] (szerokość komórki)")
    p.add_argument("--thick", type=float, default=0.15, help="grubość ścian [m]")
    p.add_argument("--height", type=float, default=0.3, help="wysokość ścian [m]")
    p.add_argument("--res", type=float, default=0.05, help="rozdzielczość mapy [m/piksel]")
    p.add_argument("--margin", type=float, default=0.5, help="margines wokół labiryntu na mapie [m]")
    p.add_argument("--robot-radius", type=float, default=0.18, help="promień do dylatacji [m]")
    p.add_argument("--seed", type=int, default=3, help="ziarno losowania labiryntu")
    p.add_argument("--out", default=str(Path.home() / "ros2_ws/src/aco_planner/maps/maze"),
                   help="folder wynikowy")
    a = p.parse_args()

    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # labirynt wyśrodkowany w (0, 0) świata
    W, H = a.cols * a.cell, a.rows * a.cell
    x0, y0 = -W / 2, -H / 2

    h_walls, v_walls = generate_maze(a.rows, a.cols, a.seed)
    # wyjście: prawa ściana górnej prawej komórki; start: dolna lewa komórka
    v_walls.discard((a.rows - 1, a.cols))
    start_cell = (0, 0)
    exit_cell = (a.rows - 1, a.cols - 1)

    boxes = walls_to_boxes(h_walls, v_walls, a.rows, a.cols, a.cell, a.thick, x0, y0)
    half = a.thick / 2
    bounds = (x0 - half, y0 - half, x0 + W + half, y0 + H + half)

    # --- mapa ---
    img_top, ox, oy = rasterize(boxes, bounds, a.res, a.margin)
    write_pgm(out / "maze.pgm", img_top)
    write_yaml(out / "maze.yaml", "maze.pgm", a.res, ox, oy)

    # --- pozycje w metrach (środki komórek labiryntu) ---
    def cell_center(rc):
        return x0 + (rc[1] + 0.5) * a.cell, y0 + (rc[0] + 0.5) * a.cell

    sx, sy = cell_center(start_cell)
    gx, gy = cell_center(exit_cell)

    def to_grid(x, y):
        return int(math.floor((y - oy) / a.res)), int(math.floor((x - ox) / a.res))

    start_px, goal_px = to_grid(sx, sy), to_grid(gx, gy)

    # --- świat i launch ---
    world_text, world_src = build_world(walls_model_sdf(boxes, a.height))
    (out / "maze.world").write_text(world_text, encoding="utf-8")
    launch_text, launch_err = build_launch(out / "maze.world", sx, sy)
    if launch_text is not None:
        (out / "maze_world.launch.py").write_text(launch_text, encoding="utf-8")

    # --- przejezdność po dylatacji ---
    grid = np.flipud(img_top)                          # wiersz 0 = dół, jak w MapData
    occ = grid != FREE                                 # ściany i nieznane = przeszkoda
    radius_px = math.ceil(round(a.robot_radius / a.res, 6))
    dil = dilate(occ, radius_px)
    free = ~dil
    path, length = shortest_path(free, start_px, goal_px)
    l_min = octile(start_px, goal_px)

    # --- scenariusze do tuningu (komórki labiryntu → piksele) ---
    scen_cells = [
        ("start -> wyjście", start_cell, exit_cell),
        ("start -> środek", start_cell, (a.rows // 2, a.cols // 2)),
        ("lewy górny -> prawy dolny", (a.rows - 1, 0), (0, a.cols - 1)),
        ("środek -> wyjście", (a.rows // 2, a.cols // 2), exit_cell),
    ]
    scen_lines, scen_stats = [], []
    for name, c1, c2 in scen_cells:
        p1, p2 = to_grid(*cell_center(c1)), to_grid(*cell_center(c2))
        pth, ln = shortest_path(free, p1, p2)
        scen_lines.append(f'    ("{name}", {p1}, {p2}),')
        scen_stats.append((name, p1, p2, ln, octile(p1, p2)))

    # --- podgląd ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
        ax[0].imshow(grid, cmap="gray", origin="lower", vmin=0, vmax=255)
        ax[0].set_title("maze.pgm (biały = wolne, szary = nieznane)")
        ax[1].imshow(dil, cmap="gray_r", origin="lower")
        if path:
            pr, pc = zip(*path)
            ax[1].plot(pc, pr, "r-", lw=2, label=f"najkrótsza: {length:.0f} px")
        for axx in ax:
            axx.plot(start_px[1], start_px[0], "go", ms=8, label="start")
            axx.plot(goal_px[1], goal_px[0], "bo", ms=8, label="cel (przy wyjściu)")
        ax[1].set_title(f"po dylatacji r = {a.robot_radius} m")
        ax[1].legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "maze_preview.png", dpi=90)
        plt.close(fig)
        preview = "maze_preview.png"
    except Exception as e:                              # brak matplotlib to nie problem
        preview = f"(pominięty: {e})"

    # --- raport ---
    yaml_path = out / "maze.yaml"
    launch_path = out / "maze_world.launch.py"
    lines = [
        f"Labirynt {a.rows}x{a.cols}, komórka {a.cell} m, ściany {a.thick} m x {a.height} m, seed {a.seed}",
        f"Mapa: {grid.shape[1]}x{grid.shape[0]} px, res {a.res} m, origin ({ox:.3f}, {oy:.3f})",
        f"Świat zbudowany z: {world_src}",
        "",
        f"START robota (map = świat): x = {sx:.3f}, y = {sy:.3f}   -> komórka {start_px}",
        f"CEL przy wyjściu:          x = {gx:.3f}, y = {gy:.3f}   -> komórka {goal_px}",
        "",
        f"Przejezdność po dylatacji ({a.robot_radius} m): " +
        ("TAK" if path else "NIE - zwiększ --cell albo zmniejsz --thick"),
        f"Najkrótsza trasa: {length:.1f} px = {length * a.res:.2f} m; "
        f"L_min (bez ścian): {l_min:.1f} px; stosunek {length / l_min:.2f}",
        f"Uwaga: max_steps w tuning.py = steps_factor * L_min. Przy stosunku "
        f"{length / l_min:.1f} steps_factor = 4 może być za mało - ustaw co najmniej "
        f"{math.ceil(2 * length / l_min)}.",
        "",
        "=== Uruchomienie ===",
        "Terminal 1 (Gazebo):",
        "  export TURTLEBOT3_MODEL=burger",
        (f"  ros2 launch {launch_path}" if launch_text is not None else
         f"  !!! launch nie został utworzony: {launch_err}\n"
         f"  Skopiuj {TB3_WORLD_LAUNCH} i zmień w nim: plik świata na {out / 'maze.world'},\n"
         f"  domyślne x_pose na {sx:.3f}, y_pose na {sy:.3f}."),
        "Terminal 2 (map -> odom):",
        f"  ros2 run tf2_ros static_transform_publisher --x {sx:.3f} --y {sy:.3f} "
        f"--frame-id map --child-frame-id odom",
        "Terminal 3 (planer):",
        f"  ros2 run aco_planner aco_node --ros-args -p map_yaml:={yaml_path}",
        "Terminal 4 (sterownik):",
        f"  ros2 run aco_planner pure_pursuit --ros-args "
        f"-p odom_offset_x:={sx:.3f} -p odom_offset_y:={sy:.3f}",
        "Terminal 5 (RViz2):",
        "  rviz2 -d ~/ros2_ws/src/aco_planner/aco.rviz --ros-args -p use_sim_time:=true",
        "",
        "=== Scenariusze do tuning.run_tuning(..., scenarios=MAZE_SCENARIOS) ===",
        "MAZE_SCENARIOS = [",
        *scen_lines,
        "]",
        "",
        "Najkrótsze trasy scenariuszy (px) / L_min (px):",
        *[f"  {n:<28} {ln:7.1f} / {lm:6.1f}  (x{ln / lm:.2f})" for n, _, _, ln, lm in scen_stats],
        "",
        f"Podgląd: {preview}",
    ]
    report = "\n".join(lines)
    (out / "maze_info.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nPliki zapisane w: {out}")


if __name__ == "__main__":
    main()
