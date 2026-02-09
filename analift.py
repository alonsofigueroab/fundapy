import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
from scipy.spatial import ConvexHull
import matplotlib.colors as mcolors

def leer_desde_portapapeles(tipo_dato):
    """
    Solicita al usuario copiar datos y los lee del portapapeles.
    Maneja errores comunes de lectura.
    """
    print(f"\n>>> ACCIÓN REQUERIDA: Copie la tabla de {tipo_dato} al portapapeles (Ctrl+C).")
    input(f"    Presione [ENTER] una vez que los datos estén en el portapapeles...")
    
    try:
        # header=0 asume que la primera línea copiada son los encabezados
        df = pd.read_clipboard(sep='\t') # RISA suele usar tabulaciones
        print(f"    Lectura exitosa: {len(df)} filas detectadas.")
        return df
    except Exception as e:
        print(f"    Error leyendo el portapapeles: {e}")
        return None

def limpiar_columnas(df):
    """Normaliza nombres de columnas eliminando espacios y unidades entre corchetes."""
    df.columns = df.columns.str.strip().str.replace(r"\[.*\]", "", regex=True).str.strip()
    return df

def ejecutar_analisis_portapapeles():
    print("--- INICIANDO ANÁLISIS DE LEVANTAMIENTO (RISA FOUNDATION) ---\n")

    # 1. Obtener Tabla de Nodos (Geometría)
    # Espera columnas: Label, Z, X
    df_nodos = leer_desde_portapapeles("NODOS (Coordinates)")
    if df_nodos is None: return
    df_nodos = limpiar_columnas(df_nodos)
    
    # Validación de columnas críticas en Nodos
    cols_nodos_req = {'Label', 'X', 'Z'}
    if not cols_nodos_req.issubset(df_nodos.columns):
        print(f"    Error: Faltan columnas en tabla de nodos. Se requiere: {cols_nodos_req}")
        print(f"    Columnas detectadas: {df_nodos.columns.tolist()}")
        return

    # 2. Obtener Tabla de Presiones (Fuerzas)
    # Espera columnas: LC, Label, Soil Pressure
    df_presiones = leer_desde_portapapeles("PRESIONES DE SUELO (Soil Pressures)")
    if df_presiones is None: return
    df_presiones = limpiar_columnas(df_presiones)
    
    # Validación de columnas críticas en Presiones
    # Nota: A veces RISA exporta 'Soil Pressure' o 'Soil Pressure [kksf]' -> La limpieza ya quitó corchetes
    cols_presion_req = {'LC', 'Label', 'Soil Pressure'}
    if not cols_presion_req.issubset(df_presiones.columns):
        print(f"    Error: Faltan columnas en tabla de presiones. Se requiere: {cols_presion_req}")
        print(f"    Columnas detectadas: {df_presiones.columns.tolist()}")
        return

    # 3. Procesamiento y Cruce de Datos (Merge)
    print("\nProcesando datos...")
    
    # Convertir Labels a string para asegurar match perfecto
    df_nodos['Label'] = df_nodos['Label'].astype(str)
    df_presiones['Label'] = df_presiones['Label'].astype(str)
    
    # Merge: Unimos la geometría (X, Z) a los resultados de presión usando 'Label'
    df_merged = pd.merge(df_presiones, df_nodos[['Label', 'X', 'Z']], on='Label', how='left')
    
    # Verificar si hubo nodos sin coordenadas (mismatch)
    if df_merged['X'].isnull().any():
        print("    Advertencia: Algunos nodos en la tabla de presión no se encontraron en la tabla de coordenadas.")
        df_merged = df_merged.dropna(subset=['X', 'Z'])

    # 4. Análisis Iterativo por Combinación (LC)
    combinaciones = df_merged['LC'].unique()
    resultados = []
    
    # Definir límites de la zapata para la grilla
    x_min, x_max = df_merged['X'].min(), df_merged['X'].max()
    z_min, z_max = df_merged['Z'].min(), df_merged['Z'].max()
    
    # Configuración de Grilla (Resolución Alta)
    grid_x, grid_z = np.mgrid[x_min:x_max:500j, z_min:z_max:500j]
    
    for lc in combinaciones:
        subset = df_merged[df_merged['LC'] == lc]
        
        points = subset[['X', 'Z']].values
        values = subset['Soil Pressure'].values
        
        # Interpolación Lineal (conservadora y precisa para bordes)
        grid_p = griddata(points, values, (grid_x, grid_z), method='linear')
        
        # Cálculo de Uplift (Presión <= 0)
        # Filtramos NaNs (puntos fuera del casco convexo de los nodos)
        valid_mask = ~np.isnan(grid_p)
        total_area_pixels = np.count_nonzero(valid_mask)
        
        if total_area_pixels == 0:
            continue
            
        # Tolerancia numérica pequeña (1e-5) para capturar el "Cero" numérico
        uplift_pixels = np.count_nonzero(grid_p[valid_mask] <= 1e-5)
        pct_uplift = (uplift_pixels / total_area_pixels) * 100
        
        resultados.append((lc, pct_uplift, subset))

    # 5. Reporte y Selección de Crítico
    resultados.sort(key=lambda x: x[1], reverse=True) # Ordenar mayor a menor uplift
    
    print("\n--- RESUMEN DE LEVANTAMIENTO POR COMBINACIÓN ---")
    print(f"{'LC':<20} | {'% Levantamiento':<15}")
    print("-" * 40)
    for lc, pct, _ in resultados[:5]: # Mostrar top 5
        print(f"{lc:<20} | {pct:.2f}%")
        
    # 6. Graficar el Peor Caso
    lc_critico, pct_critico, df_critico = resultados[0]
    
    print(f"\nGraficando combinación crítica: {lc_critico}")
    
    points = df_critico[['X', 'Z']].values
    values = df_critico['Soil Pressure'].values
    
    # --- MEJORA 1: Aumentar Resolución de Grilla ---
    # Subimos a 1000x1000 para que las curvas sean muy suaves (HD)
    grid_x, grid_z = np.mgrid[x_min:x_max:1000j, z_min:z_max:1000j]
    
    # --- MEJORA 2: Interpolación Cúbica Controlada ---
    # 'cubic' genera curvas suaves reales, eliminando el efecto "sierra" de 'linear'.
    grid_p_crit = griddata(points, values, (grid_x, grid_z), method='cubic')
    
    # IMPORTANTE: La interpolación cúbica puede generar valores negativos falsos (-0.001) 
    # cerca del cero. Los limpiamos volviéndolos 0 o NaN para que no afecten el gráfico.
    grid_p_crit[grid_p_crit < 0] = 0 
    
    plt.figure(figsize=(12, 10))
    
    # --- ESTRATEGIA VISUAL: Definir niveles exactos ---
    v_max = np.nanmax(values)
    
    # Definimos niveles que parten estrictamente desde un epsilon positivo (ej. 0.01 ton/m2)
    # Esto asegura que el "Blanco" sea todo lo que está entre 0 y 0.01
    niveles = np.linspace(0.01, v_max, 100) 
    
    # Graficamos el relleno (Solo compresiones positivas)
    # cmap='YlOrRd' (Amarillo a Rojo) o 'jet' / 'viridis'
    contour_fill = plt.contourf(grid_x, grid_z, grid_p_crit, 
                                levels=niveles, 
                                cmap='YlOrRd', 
                                extend='max') 
    
    # --- DIBUJAR LA FRONTERA EXACTA (Línea de Levantamiento) ---
    # Dibujamos una línea sólida exactamente en el mismo umbral donde empieza el color (0.01)
    # Esto "cierra" visualmente el área blanca perfectamente.
    plt.contour(grid_x, grid_z, grid_p_crit, 
                levels=[0.01], 
                colors='black', 
                linewidths=2, 
                linestyles='-') # Línea sólida para definir claramente el borde
    
    cbar = plt.colorbar(contour_fill)
    cbar.set_label('Presión de Suelo [ton/m²]')

    # --- CONTORNO FÍSICO DE LA ZAPATA ---
    try:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=3) # Borde más grueso
    except:
        pass

    # Títulos y Ajustes
    plt.title(f"Distribución de Presiones Combinación LC{lc_critico}\nLevantamiento: {pct_critico:.2f}%, Compresión: {100 - pct_critico:.2f}%")
    plt.xlabel("Coordenada X [m]")
    plt.ylabel("Coordenada Z [m]")
    plt.axis('equal') 
    
    # Fondo gris muy claro para diferenciar el "blanco de levantamiento" del "fondo del plot"
    plt.gca().set_facecolor('#f0f0f0') 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ejecutar_analisis_portapapeles()