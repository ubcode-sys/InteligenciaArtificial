import tkinter as tk
from tkinter import simpledialog, messagebox
import time

class NReinasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador N-Reinas - Algoritmos de Búsqueda César Verdugo, Fernando Olivarria")
        
        self.N = simpledialog.askinteger("Configuración", "Ingrese el número de reinas (N):", initialvalue=8, minvalue=4, maxvalue=15)
        if not self.N:
            self.N = 8
            
        self.tamano_celda = 50
        self.velocidad = 0.05 
        
        self.main_frame = tk.Frame(root, bg="#2c3e50")
        self.main_frame.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(self.main_frame, width=self.N * self.tamano_celda, height=self.N * self.tamano_celda, bg="white")
        self.canvas.pack(side="left", padx=10, pady=10)
        
        self.menu_frame = tk.Frame(self.main_frame, width=200, bg="#ecf0f1", padx=10, pady=10)
        self.menu_frame.pack(side="right", fill="y")
        
        self.crear_menu()
        self.dibujar_tablero([]) 

    def crear_menu(self):
        tk.Label(self.menu_frame, text="ALGORITMOS", font=("Arial", 12, "bold"), bg="#ecf0f1").pack(pady=10)
        btn_estilo = {"font": ("Arial", 10), "width": 18, "pady": 5}
        
        tk.Button(self.menu_frame, text="BFS (Anchura)", command=self.ejecutar_bfs, **btn_estilo).pack(pady=5)
        tk.Button(self.menu_frame, text="DFS (Profundidad)", command=self.ejecutar_dfs, **btn_estilo).pack(pady=5)
        tk.Button(self.menu_frame, text="LDFS / ILDFS", command=self.ejecutar_ildfs, **btn_estilo).pack(pady=5)
        tk.Button(self.menu_frame, text="Voraz (Greedy)", command=self.ejecutar_voraz, **btn_estilo).pack(pady=5)
        tk.Button(self.menu_frame, text="A* (A Estrella)", command=self.ejecutar_astar, **btn_estilo).pack(pady=5)

    def dibujar_tablero(self, estado):
        """
        Dibuja el tablero. 'estado' es una tupla donde el índice es la columna y el valor es la fila.
        Ejemplo: (2, 0, 3, 1) significa reinas en (2,0), (0,1), (3,2), (1,3).
        """
        self.canvas.delete("all")
        for r in range(self.N):
            for c in range(self.N):
                color = "#ecf0f1" if (r + c) % 2 == 0 else "#bdc3c7"
                x1, y1 = c * self.tamano_celda, r * self.tamano_celda
                x2, y2 = x1 + self.tamano_celda, y1 + self.tamano_celda
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
                
        for c, r in enumerate(estado):
            x = c * self.tamano_celda + self.tamano_celda // 2
            y = r * self.tamano_celda + self.tamano_celda // 2
            self.canvas.create_text(x, y, text="♛", font=("Arial", self.tamano_celda//2), fill="#c0392b")
            
        self.root.update()

    def es_seguro(self, estado, nueva_fila):
        """Verifica si colocar una reina en 'nueva_fila' en la siguiente columna es seguro."""
        col_actual = len(estado)
        for c, r in enumerate(estado):
            if r == nueva_fila or abs(r - nueva_fila) == abs(c - col_actual):
                return False
        return True

    def generar_hijos(self, estado):
        """Genera los estados válidos agregando una reina en la siguiente columna."""
        hijos = []
        if len(estado) < self.N:
            for r in range(self.N):
                if self.es_seguro(estado, r):
                    hijos.append(estado + (r,))
        return hijos

    def heuristica(self, estado):
        """
        Para Voraz y A*. Retorna un valor estimado para ordenar los nodos.
        En este caso, estimamos el costo basado en cuántas reinas faltan por colocar.
        """
        return self.N - len(estado)


    def ejecutar_bfs(self):
        frontera = [()] 
        
        while frontera:
            estado = frontera.pop(0)
            self.dibujar_tablero(estado)
            time.sleep(self.velocidad)
            
            if len(estado) == self.N:
                messagebox.showinfo("BFS", "¡Solución encontrada!")
                return
                
            hijos = self.generar_hijos(estado)
            frontera.extend(hijos) 

        messagebox.showinfo("BFS", "No hay solución.")

    def ejecutar_dfs(self):
        frontera = [()]
        
        while frontera:
            estado = frontera.pop()
            self.dibujar_tablero(estado)
            time.sleep(self.velocidad)
            
            if len(estado) == self.N:
                messagebox.showinfo("DFS", "¡Solución encontrada!")
                return
                
            hijos = self.generar_hijos(estado)
            frontera.extend(hijos) 
    def ejecutar_ildfs(self):
        for limite in range(1, self.N + 1):
            if self.ldfs(limite):
                messagebox.showinfo("ILDFS", f"¡Solución encontrada en profundidad {limite}!")
                return
        messagebox.showinfo("ILDFS", "No hay solución.")

    def ldfs(self, limite):
        frontera = [()]
        while frontera:
            estado = frontera.pop()
            self.dibujar_tablero(estado)
            time.sleep(0.01) 
            
            if len(estado) == self.N:
                return True
                
            if len(estado) < limite:
                hijos = self.generar_hijos(estado)
                frontera.extend(hijos)
        return False

    def ejecutar_voraz(self):
        frontera = [()]
        
        while frontera:
            frontera.sort(key=lambda e: self.heuristica(e))
            estado = frontera.pop(0)
            
            self.dibujar_tablero(estado)
            time.sleep(self.velocidad)
            
            if len(estado) == self.N:
                messagebox.showinfo("Voraz", "¡Solución encontrada!")
                return
                
            hijos = self.generar_hijos(estado)
            frontera.extend(hijos)

    def ejecutar_astar(self):
        frontera = [()]
        
        while frontera:
            frontera.sort(key=lambda e: len(e) + self.heuristica(e))
            estado = frontera.pop(0)
            
            self.dibujar_tablero(estado)
            time.sleep(self.velocidad)
            
            if len(estado) == self.N:
                messagebox.showinfo("A*", "¡Solución óptima encontrada!")
                return
                
            hijos = self.generar_hijos(estado)
            frontera.extend(hijos)

if __name__ == "__main__":
    root = tk.Tk()
    app = NReinasApp(root)
    root.mainloop()