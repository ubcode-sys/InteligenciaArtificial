import tkinter as tk
from tkinter import messagebox, simpledialog
import random
import time

class LaberintoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador de Laberintos - Algoritmos de Búsqueda César Verdugo, Fernando Olivarria")
        self.f = simpledialog.askinteger("Configuración", "Número de filas (m):", initialvalue=20)
        self.c = simpledialog.askinteger("Configuración", "Número de columnas (n):", initialvalue=20)
        self.p = simpledialog.askfloat("Configuración", "% de obstáculos (0-100):", initialvalue=25)
        
        self.inicio = (0, 0)
        self.meta = (self.f - 1, self.c - 1)
        self.tamano_celda = 25
        self.grid = self.generar_laberinto()
        self.visitados = []
        self.pos_actual = list(self.inicio)
        self.main_frame = tk.Frame(root, bg="#2c3e50")
        self.main_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.main_frame, width=self.c * self.tamano_celda, 
                                height=self.f * self.tamano_celda, bg="white")
        self.canvas.pack(side="left", padx=10, pady=10)
        self.menu_frame = tk.Frame(self.main_frame, width=200, bg="#ecf0f1", padx=10, pady=10)
        self.menu_frame.pack(side="right", fill="y")
        
        self.crear_menu_grafico()
        self.dibujar_tablero()
        self.root.bind("<KeyPress>", self.mover_manual)

    def crear_menu_grafico(self):
        tk.Label(self.menu_frame, text="ALGORITMOS", font=("Arial", 12, "bold"), bg="#ecf0f1").pack(pady=10)
        btn_estilo = {"font": ("Arial", 10), "width": 15, "pady": 5}
        
        tk.Button(self.menu_frame, text="BFS (Anchura)", command=self.ejecutar_bfs, **btn_estilo).pack(pady=5)
        tk.Button(self.menu_frame, text="DFS (Profundidad)", command=self.ejecutar_dfs, **btn_estilo).pack(pady=5)
        tk.Button(self.menu_frame, text="LDFS / ILDFS", command=self.ejecutar_ldfs, **btn_estilo).pack(pady=5)
        tk.Button(self.menu_frame, text="Búsqueda Voraz", command=self.ejecutar_voraz, **btn_estilo).pack(pady=5)
        tk.Button(self.menu_frame, text="Algoritmo A*", command=self.ejecutar_astar, **btn_estilo).pack(pady=5)
        
        tk.Label(self.menu_frame, text="  ", bg="#ecf0f1").pack(pady=10)
        tk.Button(self.menu_frame, text="Nuevo Laberinto", command=self.reset_laberinto, bg="#e74c3c", fg="white", **btn_estilo).pack(pady=5)

    def generar_laberinto(self):
        grid = [[0 for _ in range(self.c)] for _ in range(self.f)]
        for r in range(self.f):
            for c in range(self.c):
                if (r, c) == self.inicio or (r, c) == self.meta: continue
                if random.random() < (self.p / 100): grid[r][c] = 1
        return grid

    def dibujar_tablero(self, actual=None, camino_final=None):
        self.canvas.delete("all")
        for r in range(self.f):
            for c in range(self.c):
                x1, y1 = c * self.tamano_celda, r * self.tamano_celda
                x2, y2 = x1 + self.tamano_celda, y1 + self.tamano_celda
                
                if (r, c) == self.inicio: color = "#e67e22"
                elif (r, c) == self.meta: color = "#27ae60"
                elif camino_final and (r, c) in camino_final: color = "#f1c40f"
                elif (r, c) == actual: color = "#3498db"
                elif (r, c) in self.visitados: color = "#bdc3c7"
                elif self.grid[r][c] == 1: color = "#2c3e50"
                else: color = "white"
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#ecf0f1")
        self.root.update()
        
    def heuristica(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def ejecutar_bfs(self):
        self.visitados = []
        nodo_inicial = self.inicio
        if nodo_inicial == self.meta: return
        frontera = [(nodo_inicial, [nodo_inicial])]
        explorados = []
        while True:
            if not frontera:
                messagebox.showinfo("BFS", "No se encontró solución.")
                break
            actual, camino = frontera.pop(0)
            if actual in explorados: continue
            explorados.append(actual)
            self.visitados.append(actual)
            self.dibujar_tablero(actual=actual)
            time.sleep(0.02)
            if actual == self.meta:
                self.dibujar_tablero(camino_final=camino)
                messagebox.showinfo("BFS", "¡Ruta encontrada!")
                break
            r, c = actual
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.f and 0 <= nc < self.c and self.grid[nr][nc] == 0:
                    if (nr, nc) not in explorados:
                        frontera.append(((nr, nc), camino + [(nr, nc)]))

    def ejecutar_dfs(self):
        self.visitados = []
        nodo_inicial = self.inicio
        if nodo_inicial == self.meta:
            self.dibujar_tablero(camino_final=[nodo_inicial])
            return
        frontera = [(nodo_inicial, [nodo_inicial])]
        explorados = []
        while True:
            if not frontera:
                messagebox.showinfo("DFS", "No se encontró solución.")
                break
            actual, camino = frontera.pop(0)
            if actual in explorados:
                continue
            explorados.append(actual)
            self.visitados.append(actual)
            self.dibujar_tablero(actual=actual)
            time.sleep(0.02) 
            if actual == self.meta:
                self.dibujar_tablero(camino_final=camino)
                messagebox.showinfo("DFS", f"¡Ruta encontrada! Pasos: {len(camino)}")
                break
            r, c = actual
            hijos = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]: 
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.f and 0 <= nc < self.c and self.grid[nr][nc] == 0:
                    if (nr, nc) not in explorados:
                        hijos.append(((nr, nc), camino + [(nr, nc)]))
            frontera = hijos + frontera
        
    def ejecutar_ldfs(self):
        limite = simpledialog.askinteger("LDFS", "Ingrese el límite de profundidad:", initialvalue=10)
        if limite is None: return
        self.visitados = []
        nodo_inicial = self.inicio
        if nodo_inicial == self.meta:
            self.dibujar_tablero(camino_final=[nodo_inicial])
            return
        frontera = [(nodo_inicial, [nodo_inicial])]
        explorados = []
        while True:
            if not frontera:
                messagebox.showinfo("LDFS", f"No se encontró solución con límite {limite}.")
                break
            actual, camino = frontera.pop(0)
            profundidad_actual = len(camino) - 1
            if actual in explorados:
                continue
            explorados.append(actual)
            self.visitados.append(actual)
            self.dibujar_tablero(actual=actual)
            time.sleep(0.02)
            if actual == self.meta:
                self.dibujar_tablero(camino_final=camino)
                messagebox.showinfo("LDFS", f"¡Meta encontrada! Profundidad: {profundidad_actual}")
                break
            if profundidad_actual < limite:
                r, c = actual
                hijos = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.f and 0 <= nc < self.c and self.grid[nr][nc] == 0:
                        if (nr, nc) not in explorados:
                            hijos.append(((nr, nc), camino + [(nr, nc)]))
                frontera = hijos + frontera
            else:
                continue
        
    def ejecutar_voraz(self):
        self.visitados = []
        actual = self.inicio
        camino = [actual]
        if actual == self.meta:
            self.dibujar_tablero(camino_final=camino)
            return
        explorados = []
        while True:
            if actual is None:
                messagebox.showinfo("Voraz", "No se encontró solución (Camino bloqueado).")
                break
            if actual in explorados:
                actual = None
                continue
            explorados.append(actual)
            self.visitados.append(actual)
            self.dibujar_tablero(actual=actual)
            time.sleep(0.1)
            if actual == self.meta:
                self.dibujar_tablero(camino_final=camino)
                messagebox.showinfo("Voraz", "¡Meta encontrada!")
                break
            r, c = actual
            hijos_candidatos = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.f and 0 <= nc < self.c and self.grid[nr][nc] == 0:
                    if (nr, nc) not in explorados:
                        hijos_candidatos.append((nr, nc))
            if hijos_candidatos:
                hijos_candidatos.sort(key=lambda h: self.heuristica(h, self.meta))
                actual = hijos_candidatos[0]
                camino.append(actual)
            else:
                actual = None 

    def ejecutar_astar(self):
        self.visitados = []
        nodo_inicial = self.inicio
        if nodo_inicial == self.meta:
            self.dibujar_tablero(camino_final=[nodo_inicial])
            return
        f_ini = self.heuristica(nodo_inicial, self.meta)
        frontera = [(f_ini, nodo_inicial, [nodo_inicial])]
        explorados = []
        while True:
            if not frontera:
                messagebox.showinfo("A*", "No se encontró solución.")
                break
            frontera.sort(key=lambda x: x[0]) 
            f_actual, actual, camino = frontera.pop(0)
            if actual in explorados:
                continue
            explorados.append(actual)
            self.visitados.append(actual)
            self.dibujar_tablero(actual=actual)
            time.sleep(0.01)
            if actual == self.meta:
                self.dibujar_tablero(camino_final=camino)
                messagebox.showinfo("A*", f"¡Ruta hallada! Pasos: {len(camino)-1}")
                break
            r, c = actual
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.f and 0 <= nc < self.c and self.grid[nr][nc] == 0:
                    if (nr, nc) not in explorados:
                        nuevo_camino = camino + [(nr, nc)]
                        g = len(nuevo_camino) - 1
                        h = self.heuristica((nr, nc), self.meta)
                        f = g + h
                        frontera.append((f, (nr, nc), nuevo_camino))

    def mover_manual(self, event):
        tecla = event.keysym
        r, c = self.pos_actual
        if tecla == "Up": r -= 1
        elif tecla == "Down": r += 1
        elif tecla == "Left": c -= 1
        elif tecla == "Right": c += 1
        else: return
        if 0 <= r < self.f and 0 <= c < self.c and self.grid[r][c] == 0:
            self.pos_actual = [r, c]
            self.visitados.append(tuple(self.pos_actual))
            self.dibujar_tablero(actual=tuple(self.pos_actual))
            if tuple(self.pos_actual) == self.meta:
                messagebox.showinfo("Manual", "¡Llegaste!")

    def reset_laberinto(self):
        self.grid = self.generar_laberinto()
        self.visitados = []
        self.pos_actual = list(self.inicio)
        self.dibujar_tablero()

if __name__ == "__main__":
    root = tk.Tk()
    app = LaberintoApp(root)
    root.mainloop()