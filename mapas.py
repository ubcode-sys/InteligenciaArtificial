import math
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import simpledialog
import tkintermapview
import xml.etree.ElementTree as ET
import os

class AppRutasMapaReal:
    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title("Proyecto 1 - Maps César Verdugo, Fernando Olivarria")
        self.raiz.geometry("950x750")
        
        self.grafo = {}
        self.nodo_inicio = None
        self.nodo_meta = None
        self.marcador_inicio = None
        self.marcador_meta = None
        self.ruta_busqueda_final = None
        self.lineas_exploradas = []

        marco_control = tk.Frame(raiz)
        marco_control.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.boton_ejecutar = tk.Button(marco_control, text="Ejecutar Búsqueda", command=self.ejecutar, state=tk.DISABLED, bg="lightblue")
        self.boton_ejecutar.pack(side=tk.RIGHT, padx=5)

        self.combo_algoritmos = ttk.Combobox(marco_control, values=["BFS", "DFS", "LDFS", "Voraz", "A*"], state="readonly", width=8)
        self.combo_algoritmos.pack(side=tk.RIGHT, padx=5)
        self.combo_algoritmos.current(0)
        tk.Label(marco_control, text="Algoritmo:").pack(side=tk.RIGHT)

        boton_limpiar = tk.Button(marco_control, text="Limpiar Mapa", command=self.limpiar_mapa)
        boton_limpiar.pack(side=tk.RIGHT, padx=10)
        
        self.widget_mapa = tkintermapview.TkinterMapView(raiz, width=800, height=600, corner_radius=0)
        self.widget_mapa.pack(fill="both", expand=True)
        
        self.widget_mapa.set_position(24.8020, -107.3924)
        self.widget_mapa.set_zoom(15)
        self.widget_mapa.add_right_click_menu_command(label="Establecer Origen", command=self.establecer_origen, pass_coords=True)
        self.widget_mapa.add_right_click_menu_command(label="Establecer Destino", command=self.establecer_destino, pass_coords=True)

        self.cargar_datos_osm()

    def cargar_datos_osm(self):
        ruta_archivo = "map.osm"
        if not os.path.exists(ruta_archivo):
            messagebox.showerror("Error", "No se encontró el archivo 'map.osm' en el directorio actual. Por favor descárgalo de OpenStreetMap y colócalo aquí.")
            return

        try:
            arbol = ET.parse(ruta_archivo)
            raiz_xml = arbol.getroot()
            
            nodos_crudos = {}
            for nodo in raiz_xml.findall('node'):
                nodos_crudos[nodo.get('id')] = (float(nodo.get('lat')), float(nodo.get('lon')))
                
            self.grafo = {}
            for via in raiz_xml.findall('way'):
                es_calle = False
                for etiqueta in via.findall('tag'):
                    if etiqueta.get('k') == 'highway':
                        es_calle = True
                        break
                        
                if es_calle:
                    nodos_via = [nd.get('ref') for nd in via.findall('nd')]
                    for i in range(len(nodos_via)-1):
                        id1, id2 = nodos_via[i], nodos_via[i+1]
                        if id1 in nodos_crudos and id2 in nodos_crudos:
                            coord1 = nodos_crudos[id1]
                            coord2 = nodos_crudos[id2]
                            distancia = self.haversine(coord1, coord2)
                            
                            self.grafo.setdefault(coord1, {})[coord2] = distancia
                            self.grafo.setdefault(coord2, {})[coord1] = distancia
        except Exception as e:
            messagebox.showerror("Error de Lectura", f"Ocurrió un error leyendo map.osm:\n{e}")

    def haversine(self, nodo1, nodo2):
        lat1, lon1 = nodo1
        lat2, lon2 = nodo2
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return 6371 * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

    def obtener_nodo_mas_cercano(self, coordenadas):
        if not self.grafo: return None
        return min(self.grafo.keys(), key=lambda nodo: self.haversine(coordenadas, nodo))

    def establecer_origen(self, coordenadas):
        if not self.grafo:
            messagebox.showwarning("Advertencia", "Primero debes cargar el archivo map.osm.")
            return
        
        nodo_cercano = self.obtener_nodo_mas_cercano(coordenadas)
        self.nodo_inicio = nodo_cercano
        
        if self.marcador_inicio: self.marcador_inicio.delete()
        self.marcador_inicio = self.widget_mapa.set_marker(nodo_cercano[0], nodo_cercano[1], text="Origen", marker_color_circle="green", marker_color_outside="darkgreen")
        self.intentar_activar_ejecucion()

    def establecer_destino(self, coordenadas):
        if not self.grafo:
            messagebox.showwarning("Advertencia", "Primero debes cargar el archivo map.osm.")
            return
            
        nodo_cercano = self.obtener_nodo_mas_cercano(coordenadas)
        self.nodo_meta = nodo_cercano
        
        if self.marcador_meta: self.marcador_meta.delete()
        self.marcador_meta = self.widget_mapa.set_marker(nodo_cercano[0], nodo_cercano[1], text="Destino", marker_color_circle="yellow", marker_color_outside="orange")
        self.intentar_activar_ejecucion()

    def intentar_activar_ejecucion(self):
        if self.marcador_inicio and self.marcador_meta:
            self.boton_ejecutar.config(state=tk.NORMAL)

    def limpiar_mapa(self):
        if self.marcador_inicio: self.marcador_inicio.delete()
        if self.marcador_meta: self.marcador_meta.delete()
        if self.ruta_busqueda_final: self.ruta_busqueda_final.delete()
        for linea in self.lineas_exploradas: linea.delete()
        
        self.marcador_inicio = None
        self.marcador_meta = None
        self.nodo_inicio = None
        self.nodo_meta = None
        self.ruta_busqueda_final = None
        self.lineas_exploradas = []
        self.boton_ejecutar.config(state=tk.DISABLED)

    def ejecutar(self):
        algoritmo = self.combo_algoritmos.get()
        self.boton_ejecutar.config(state=tk.DISABLED)
        
        if self.ruta_busqueda_final: self.ruta_busqueda_final.delete()
        for linea in self.lineas_exploradas: linea.delete()
        self.lineas_exploradas = []
        visitados_interfaz = set() 

        def paso_callback(nodo_padre, nodo_actual):
            if nodo_padre is None: return
            
            arista = (nodo_padre, nodo_actual)
            arista_inversa = (nodo_actual, nodo_padre)
            
            if arista not in visitados_interfaz and arista_inversa not in visitados_interfaz:
                visitados_interfaz.add(arista)
                linea = self.widget_mapa.set_path([nodo_padre, nodo_actual], color="blue", width=2)
                self.lineas_exploradas.append(linea)
                self.raiz.update()
                time.sleep(0.005) 

        try:
            ruta = None
            if algoritmo == "BFS": ruta = self.bfs_gui(paso_callback)
            elif algoritmo == "DFS": ruta = self.dfs_gui(paso_callback)
            elif algoritmo == "LDFS": ruta = self.ldfs_gui(paso_callback)
            elif algoritmo == "Voraz": ruta = self.voraz_gui(paso_callback)
            elif algoritmo == "A*": ruta = self.a_estrella_gui(paso_callback)
            
            if ruta:
                self.ruta_busqueda_final = self.widget_mapa.set_path(ruta, color="yellow", width=4)
                messagebox.showinfo("Completado", f"Ruta trazada con {algoritmo}.\nNodos en ruta: {len(ruta)}")
            else:
                messagebox.showwarning("Sin ruta", "El algoritmo no encontró una ruta que conecte esos dos puntos.")
        finally:
            self.boton_ejecutar.config(state=tk.NORMAL)

    def bfs_gui(self, callback):
        if self.nodo_inicio == self.nodo_meta: return [self.nodo_inicio]
        
        frontera = [(self.nodo_inicio, [self.nodo_inicio])]
        explorados = []
        
        while True:
            if not frontera:
                return None
                
            actual, camino = frontera.pop(0)
            if actual in explorados: continue
            
            explorados.append(actual)
            padre = camino[-2] if len(camino) > 1 else None
            callback(padre, actual)
            
            if actual == self.nodo_meta:
                return camino
                
            for vecino in self.grafo.get(actual, {}):
                if vecino not in explorados:
                    frontera.append((vecino, camino + [vecino]))

    def dfs_gui(self, callback):
        if self.nodo_inicio == self.nodo_meta: return [self.nodo_inicio]
        
        frontera = [(self.nodo_inicio, [self.nodo_inicio])]
        explorados = []
        
        while True:
            if not frontera:
                messagebox.showinfo("DFS", "No se encontró solución.")
                return None
                
            actual, camino = frontera.pop(0)
            if actual in explorados: continue
            
            explorados.append(actual)
            padre = camino[-2] if len(camino) > 1 else None
            callback(padre, actual)
            
            if actual == self.nodo_meta:
                messagebox.showinfo("DFS", f"¡Ruta encontrada! Nodos: {len(camino)}")
                return camino
                
            hijos = []
            for vecino in self.grafo.get(actual, {}):
                if vecino not in explorados:
                    hijos.append((vecino, camino + [vecino]))
            
            frontera = hijos + frontera

    def ldfs_gui(self, callback): 
        limite = simpledialog.askinteger("LDFS", "Ingrese el límite de profundidad:", initialvalue=50)
        if limite is None: return None
        
        if self.nodo_inicio == self.nodo_meta:
            return [self.nodo_inicio]
            
        frontera = [(self.nodo_inicio, [self.nodo_inicio])]
        explorados = []
        
        while True:
            if not frontera:
                messagebox.showinfo("LDFS", f"No se encontró solución con límite {limite}.")
                return None
                
            actual, camino = frontera.pop(0)
            profundidad_actual = len(camino) - 1
            
            if actual in explorados:
                continue
                
            explorados.append(actual)
            
            padre = camino[-2] if len(camino) > 1 else None
            callback(padre, actual)
            
            if actual == self.nodo_meta:
                return camino
                
            if profundidad_actual < limite:
                hijos = []
                for vecino in self.grafo.get(actual, {}):
                    if vecino not in explorados:
                        hijos.append((vecino, camino + [vecino]))
                
                frontera = hijos + frontera
            else:
                continue

    def voraz_gui(self, callback):
        if self.nodo_inicio == self.nodo_meta: return [self.nodo_inicio]
        
        h_inicial = self.haversine(self.nodo_inicio, self.nodo_meta)
        frontera = [(h_inicial, self.nodo_inicio, [self.nodo_inicio])]
        explorados = []
        
        while True:
            if not frontera:
                messagebox.showinfo("Voraz", "No se encontró solución.")
                return None
                
            frontera.sort(key=lambda x: x[0])
            h_actual, actual, camino = frontera.pop(0)
            
            if actual in explorados:
                continue
            explorados.append(actual)
            
            padre = camino[-2] if len(camino) > 1 else None
            callback(padre, actual)
            
            if actual == self.nodo_meta:
                messagebox.showinfo("Voraz", f"¡Meta encontrada!\nNodos en la ruta: {len(camino)}")
                return camino
                
            for vecino in self.grafo.get(actual, {}):
                if vecino not in explorados:
                    h = self.haversine(vecino, self.nodo_meta)
                    frontera.append((h, vecino, camino + [vecino]))

    def a_estrella_gui(self, callback):
        if self.nodo_inicio == self.nodo_meta:
            return [self.nodo_inicio]
            
        f_inicial = self.haversine(self.nodo_inicio, self.nodo_meta)
        frontera = [(f_inicial, 0, self.nodo_inicio, [self.nodo_inicio])]
        explorados = []
        
        while True:
            if not frontera:
                return None
            frontera.sort(key=lambda x: x[0]) 
            f_actual, g_actual, actual, camino = frontera.pop(0)
            
            if actual in explorados:
                continue
                
            explorados.append(actual)
            padre = camino[-2] if len(camino) > 1 else None
            callback(padre, actual)
            
            if actual == self.nodo_meta:
                return camino
            
            for vecino, costo_arista in self.grafo.get(actual, {}).items():
                if vecino not in explorados:
                    nuevo_camino = camino + [vecino]
                    g = g_actual + costo_arista
                    h = self.haversine(vecino, self.nodo_meta)
                    f = g + h
                    frontera.append((f, g, vecino, nuevo_camino))

if __name__ == "__main__":
    raiz_principal = tk.Tk()
    aplicacion = AppRutasMapaReal(raiz_principal)
    raiz_principal.mainloop()