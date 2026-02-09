import polars as pl
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
from scipy.spatial import ConvexHull
import matplotlib.colors as mcolors

from scipy.ndimage import gaussian_filter

def leer_desde_portapapeles(tipo_dato):
    """
    Solicita al usuario copiar datos y los lee del portapapeles.
    Maneja errores comunes de lectura.
    """
    print(f"\n>>> ACCIÓN REQUERIDA: Copie la tabla de {tipo_dato} al portapapeles (Ctrl+C).")
    input(f"    Presione [ENTER] una vez que los datos estén en el portapapeles...")
    
    try:
        # header=0 asume que la primera línea copiada son los encabezados
        df = pl.read_clipboard(separator='\t') # RISA suele usar tabulaciones
        print(f"    Lectura exitosa: {len(df)} filas detectadas.")
        return df
    except Exception as e:
        print(f"    Error leyendo el portapapeles: {e}")
        return None

def limpiar_columnas(df):
    """Normaliza nombres de columnas eliminando espacios y unidades entre corchetes."""
    new_cols = {col: re.sub(r"\[.*\]", "", col).strip() for col in df.columns}
    df = df.rename(new_cols)
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
    df_nodos = df_nodos.with_columns(pl.col('Label').cast(pl.String))
    df_presiones = df_presiones.with_columns(pl.col('Label').cast(pl.String))
    
    # Merge: Unimos la geometría (X, Z) a los resultados de presión usando 'Label'
    df_merged = df_presiones.join(df_nodos.select(['Label', 'X', 'Z']), on='Label', how='left')
    
    # Verificar si hubo nodos sin coordenadas (mismatch)
    if df_merged['X'].is_null().any():
        print("    Advertencia: Algunos nodos en la tabla de presión no se encontraron en la tabla de coordenadas.")
        df_merged = df_merged.drop_nulls(subset=['X', 'Z'])

    # 4. Análisis Iterativo por Combinación (LC)
    combinaciones = df_merged['LC'].unique().to_list()
    resultados = []
    
    # Definir límites de la zapata para la grilla
    x_min, x_max = df_merged['X'].min(), df_merged['X'].max()
    z_min, z_max = df_merged['Z'].min(), df_merged['Z'].max()
    
    # Configuración de Grilla (Resolución Alta)
    grid_x, grid_z = np.mgrid[x_min:x_max:500j, z_min:z_max:500j]
    
    for lc in combinaciones:
        subset = df_merged.filter(pl.col('LC') == lc)
        
        points = subset.select(['X', 'Z']).to_numpy()
        values = subset['Soil Pressure'].to_numpy()
        
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
    
    # Calcular los offsets para ubicar la grilla correctamente en el origen del sistema de coordenadas.
    x_offset = df_critico['X'].min()
    z_offset = df_critico['Z'].min()
    
    # Generación de coordenadas relativas (Locales)
    x_relativo = df_critico['X'].to_numpy() - x_offset
    z_relativo = df_critico['Z'].to_numpy() - z_offset
    
    values = df_critico['Soil Pressure'].to_numpy()
    
    # Empaquetado de puntos trasladados
    points_rel = np.column_stack((z_relativo, x_relativo))
    
    # Definición de grilla basada en dimensiones
    ancho_x = x_relativo.max()
    alto_z = z_relativo.max()
    
    # Recalcular límites para la grilla basada en coordenadas relativas
    grid_x, grid_z = np.mgrid[0:ancho_x:1000j, 0:alto_z:1000j]
    
    # Interpolación de puntos relativos
    grid_p_crit = griddata(points_rel, values, (grid_z, grid_x), method='cubic')
    
    # IMPORTANTE: La interpolación cúbica puede generar valores negativos falsos (-0.001) 
    # cerca del cero. Los limpiamos volviéndolos 0 o NaN para que no afecten el gráfico.
    grid_p_crit[grid_p_crit < 0] = 0 
    
    # Relleno de NaNs exteriores con 0 para que el filtro no expanda bordes quebrados
    grid_p_crit = np.nan_to_num(grid_p_crit, nan=0.0)
    
    # Filtro gaussiano para suavizar la transición entre zonas de presión (opcional, ajustar sigma según necesidad)
    sigma_suavizado = 8
    grid_p_crit = gaussian_filter(grid_p_crit, sigma=sigma_suavizado)
    
    # En caso de que el filtro haya agregado valores negativos, los corregimos nuevamente
    grid_p_crit[grid_p_crit < 0] = 0
        
    plt.figure(figsize=(12, 10))
    
    # Configuración de niveles
    umbral_corte = 0.01
    v_max = np.nanmax(values)
    niveles = np.linspace(umbral_corte, v_max, 100)
    
    # Graficamos el relleno (Solo compresiones positivas)
    # cmap='YlOrRd' (Amarillo a Rojo) o 'jet' / 'viridis'
    contour_fill = plt.contourf(grid_z, grid_x, grid_p_crit, 
                                levels=niveles, 
                                cmap='jet', 
                                extend='max') 
    
    # Graficar frontera de levantamiento (Presión <= 0.01 ton/m²)
    plt.contour(grid_z, grid_x, grid_p_crit, 
                levels=[0.01], 
                colors='black', 
                linewidths=2, 
                linestyles='-') # Línea sólida para definir claramente el borde
    
    cbar = plt.colorbar(contour_fill)
    cbar.set_label('Presión de Suelo [ton/m²]')

    # Contorno de la Zapata (Convex Hull de los puntos)
    try:
        hull = ConvexHull(points_rel)
        for simplex in hull.simplices:
            plt.plot(points_rel[simplex, 0], points_rel[simplex, 1], 'k-', linewidth=3) # Borde más grueso
    except:
        pass

    # Títulos y Ajustes
    plt.title(f"Distribución de Presiones Combinación LC{lc_critico}\nCompresión: {100 - pct_critico:.2f}%, Levantamiento: {pct_critico:.2f}%")
    plt.xlabel("Coordenada Z [m]")
    plt.ylabel("Coordenada X [m]")
    plt.axis('equal') 
    
    # Fondo gris muy claro para diferenciar el "blanco de levantamiento" del "fondo del plot"
    plt.gca().set_facecolor('#f0f0f0') 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ejecutar_analisis_portapapeles()