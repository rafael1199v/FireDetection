import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def crear_imagen_mapa_calor(points, probabilidades, nombre_archivo, grupo_num):
    """
    Args:
        points: Lista de puntos (tuplas (lat, lon))
        probabilidades: Lista de probabilidades para cada punto
        nombre_archivo: Nombre del archivo a guardar
        grupo_num: Número del grupo
    """
    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])
    probs = np.array(probabilidades)
    
    print(f"Rango de probabilidades: {probs.min():.3f} - {probs.max():.3f}")
    print(f"Probabilidades: {probs}")
    
    if probs.max() > 1.0:
        print("Probabilidades > 1, normalizando...")
        probs = probs / 100.0
    
    probs = np.clip(probs, 0, 1)
    
    lat_center = np.mean(lats)
    lon_center = np.mean(lons)
    # lat_range = max(lats) - min(lats)
    # lon_range = max(lons) - min(lons)
    
    margin = 0.002
    lat_min, lat_max = min(lats) - margin, max(lats) + margin
    lon_min, lon_max = min(lons) - margin, max(lons) + margin
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    try:
        zoom = calcular_zoom_level(lat_min, lat_max, lon_min, lon_max, 1600, 1200)
        
        import contextily as cx
        from pyproj import Transformer
        
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        
        x_coords, y_coords = transformer.transform(lons, lats)
        x_min, y_min = transformer.transform(lon_min, lat_min)
        x_max, y_max = transformer.transform(lon_max, lat_max)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        cx.add_basemap(ax, 
                      crs="EPSG:3857",
                      source=cx.providers.Esri.WorldImagery,
                      zoom=zoom,
                      attribution=False,
                      alpha=0.8)

        try:
            cx.add_basemap(ax,
                          crs="EPSG:3857",
                          source=cx.providers.CartoDB.PositronOnlyLabels,
                          zoom=zoom,
                          attribution=False,
                          alpha=0.9)
        except:
            try:
                cx.add_basemap(ax,
                              crs="EPSG:3857",
                              source=cx.providers.OpenStreetMap.Mapnik,
                              zoom=zoom,
                              attribution=False,
                              alpha=0.3)
            except:
                pass
        
        usar_mercator = True
        
    except Exception as e:
        print(f"No se pudo cargar imagen satelital: {e}")
        print("Usando mapa simple como alternativa...")
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_facecolor('#d4e7d4')
        ax.grid(True, alpha=0.2, linestyle='--')
        x_coords, y_coords = lons, lats
        usar_mercator = False
    
    if usar_mercator:
        grid_x, grid_y = np.mgrid[x_min:x_max:300j, y_min:y_max:300j]
    else:
        grid_x, grid_y = np.mgrid[lon_min:lon_max:300j, lat_min:lat_max:300j]
    
    grid_prob = np.zeros_like(grid_x, dtype=float)
    
    for x, y, prob in zip(x_coords, y_coords, probs):
        if usar_mercator:
            dist = np.sqrt((grid_x - x)**2 + (grid_y - y)**2)
            radius = 400
        else:
            dist = np.sqrt((grid_x - x)**2 + (grid_y - y)**2)
            radius = 0.002
        
        sigma = radius / 3
        gaussian = np.exp(-(dist**2) / (2 * sigma**2))
        
        gaussian = np.where(dist <= radius, gaussian, 0)
        
        contribution = gaussian * prob
        
        grid_prob = np.maximum(grid_prob, contribution)
    
    print("Grid probabilidad después de procesamiento:")
    print(f"    Min: {grid_prob.min():.4f} ({grid_prob.min()*100:.2f}%)")
    print(f"    Max: {grid_prob.max():.4f} ({grid_prob.max()*100:.2f}%)")
    print(f"    Promedio: {grid_prob.mean():.4f} ({grid_prob.mean()*100:.2f}%)")
    
    colors_heat = [
        (0.0, '#00000000'),
        (0.05, '#0066ff30'),
        (0.20, '#0066ff60'),
        (0.40, '#00ff0080'),
        (0.60, '#ffff00aa'),
        (0.75, '#ff6600cc'),
        (1.0, '#ff0000ee')
    ]
    
    cmap_heat = mcolors.LinearSegmentedColormap.from_list(
        'fire_heat',
        [(val, color) for val, color in colors_heat]
    )
    
    print(f"Grid probabilidad: min={grid_prob.min():.3f}, max={grid_prob.max():.3f}")
    
    grid_prob_display = np.where(grid_prob < 0.02, np.nan, grid_prob)
    
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    
    heatmap = ax.imshow(grid_prob_display.T, 
                        origin='lower',
                        extent=extent,
                        cmap=cmap_heat,
                        vmin=0.0,
                        vmax=1.0,
                        alpha=0.5,
                        zorder=3,
                        interpolation='gaussian',
                        aspect='auto')
    
    cmap_points = plt.cm.RdYlBu_r
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    # scatter = ax.scatter(x_coords, y_coords, 
    #                     c=probs, 
    #                     cmap=cmap_points, 
    #                     s=800,
    #                     alpha=0.9, 
    #                     norm=norm, 
    #                     edgecolors='white', 
    #                     linewidth=4,
    #                     zorder=5)
    
    ax.scatter(x_coords, y_coords, 
              c=probs, 
              cmap=cmap_points, 
              s=800,
              alpha=0.9, 
              norm=norm, 
              edgecolors='black', 
              linewidth=2,
              zorder=4)
    
    for x, y, prob in zip(x_coords, y_coords, probs):
        text_color = 'white' if prob > 0.5 else 'black'
        outline_color = 'black' if prob > 0.5 else 'white'
        
        for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            ax.text(x + offset[0]*20, y + offset[1]*20, f'{prob:.0%}', 
                   ha='center', va='center', 
                   fontweight='bold', fontsize=12, 
                   color=outline_color,
                   zorder=6)
        
        ax.text(x, y, f'{prob:.0%}', 
               ha='center', va='center', 
               fontweight='bold', fontsize=12, 
               color=text_color,
               zorder=7)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    max_prob = max(probs)
    puntos_alerta = sum(1 for p in probs if p >= 0.6)
    
    titulo = 'DETECCIÓN DE INCENDIOS FORESTALES\n'
    titulo += f'Zona #{grupo_num} - Santa Cruz, Bolivia'
    
    ax.set_title(titulo, 
                fontsize=18, fontweight='bold', pad=25,
                bbox=dict(boxstyle='round,pad=1.2', 
                         facecolor='white', 
                         alpha=0.95,
                         edgecolor='black',
                         linewidth=2))
    
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Probabilidad de Incendio', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='#0066ff', markersize=14, 
                  label='Bajo (0-40%)', 
                  markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='#ffff00', markersize=14, 
                  label='Medio (40-60%)', 
                  markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='#ff0000', markersize=14, 
                  label='Alto (60-100%)', 
                  markeredgecolor='black', markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, 
             loc='upper left', 
             fontsize=12, 
             framealpha=0.95, 
             edgecolor='black',
             fancybox=True,
             shadow=True)
    
    # Cuadro de resumen
    resumen = f"ALERTAS: {puntos_alerta}/{len(probs)} puntos\n"
    resumen += f"Riesgo máximo: {max_prob:.0%}\n"
    
    if puntos_alerta > 0:
        resumen += "ACCIÓN REQUERIDA"
        bg_color = '#ff6b6b'
    else:
        resumen += "Todo bajo control"
        bg_color = '#51cf66'
    
    ax.text(0.02, 0.02, resumen,
           transform=ax.transAxes,
           fontsize=13,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round,pad=1', 
                    facecolor=bg_color,
                    alpha=0.95, 
                    edgecolor='black', 
                    linewidth=2),
           fontweight='bold',
           color='white')
    

    ax.text(0.98, 0.98, f'Lat: {lat_center:.4f}°\nLon: {lon_center:.4f}°',
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='white',
                    alpha=0.8),
           color='black')
    
    plt.tight_layout()
    
    plt.savefig(nombre_archivo, dpi=200, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Mapa guardado: {nombre_archivo}")
    return nombre_archivo


def calcular_zoom_level(lat_min, lat_max, lon_min, lon_max, width_px, height_px):
    import math
    
    lat_diff = lat_max - lat_min
    lon_diff = lon_max - lon_min
    
    zoom_lat = math.log2(360 / lat_diff * height_px / 256)
    zoom_lon = math.log2(360 / lon_diff * width_px / 256)
    
    zoom = int(min(zoom_lat, zoom_lon))
    return max(10, min(zoom, 18))