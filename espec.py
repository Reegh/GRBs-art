import matplotlib.pyplot as plt
from gdt.core import data_path
from gdt.missions.fermi.gbm.phaii import Cspec
from gdt.core.plot.spectrum import Spectrum
import json
import os

# Cargamos el json
with open('objects.json', 'r') as f:
    config = json.load(f)

# detectores
detector_map = {'n7':'cspec_nai1', 'n3':'cspec_nai2', 'b0':'cspec_bgo1',
                'n9':'cspec_nai1', 'na':'cspec_nai2', 'b1':'cspec_bgo1'}

# Creamos la carpeta
espectros_dir = os.path.join('results', 'espectros')
os.makedirs(espectros_dir, exist_ok=True)

# procesamos cada objeto
for obj in config['objects']:
    grb_name = obj['name']
    grb_dir = os.path.join(espectros_dir, grb_name.replace(' ', '_'))
    os.makedirs(grb_dir, exist_ok=True)
    
    # Parámetros
    detector = obj['spectrum_config']['detector_especifico']
    time_range = tuple(obj['spectrum_config']['time_range'])
    rangos = obj['spectrum_config']['rangos_energia']
    
    # Cargamos los datos
    file_key = detector_map.get(detector, 'cspec_nai1')
    file_name = obj['data_paths'][file_key]
    filepath = os.path.join(config.get('default_data_dir', 'data'), file_name)
    phaii = Cspec.open(filepath)
    
    # Espectro 
    spectrum_full = phaii.to_spectrum(time_range=time_range)
    
    # Colores
    colors = ['limegreen', 'dodgerblue', 'tomato']
    
    # 1. BANDA COMPLETA 
    specplot = Spectrum(data=spectrum_full, interactive=False)
    
    # Rango completo
    rango_min = rangos[0]['min']
    rango_max = rangos[-1]['max']
    spectrum_continuo = phaii.to_spectrum(time_range=time_range, 
                                         energy_range=(rango_min, rango_max))
    specplot.add_selection(spectrum_continuo)
    
    # toda la banda
    specplot.selections[0].color = 'mediumpurple'  
    specplot.selections[0].fill_alpha = 0.25
    
    specplot.spectrum.color = 'black'
    specplot.spectrum.linewidth = 2.5
    
    # Configuracion
    plt.title(f"{grb_name}\nDetector:{detector}|Tiempo:{time_range[0]}-{time_range[1]} s")
    plt.xlabel("Energía (keV)")
    plt.ylabel("Tasa (count/s/keV)")
    plt.xscale('log')
    plt.yscale('log')
    plt.figtext(0.02, 0.98, f"Archivo: {file_name}", fontsize=9, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(grb_dir, f"completa_continua_{detector}.png"), dpi=300)
    plt.show()
    plt.close()
    
    # BANDAS A DIFERENTES RANGOS DE ENERGÍA
    for i, rango in enumerate(rangos):
        specplot = Spectrum(data=spectrum_full, interactive=False)
        
        spec_banda = phaii.to_spectrum(time_range=time_range, 
                                      energy_range=(rango['min'], rango['max']))
        specplot.add_selection(spec_banda)
        specplot.selections[0].color = colors[i]
        specplot.selections[0].fill_alpha = 0.25
        
        specplot.spectrum.color = 'black'
        specplot.spectrum.linewidth = 2.5
        
        plt.title(f"{grb_name}\nDetector:{detector}|Tiempo:{time_range[0]}-{time_range[1]} s")
        plt.xlabel("Energía (keV)")
        plt.ylabel("Tasa (count/s/keV)")
        plt.xscale('log')
        plt.yscale('log')
        plt.figtext(0.02, 0.98, f"Archivo: {file_name}", fontsize=9, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(grb_dir, f"solo_{rango['nombre']}_{detector}.png"), dpi=300)
        plt.show()
        plt.close()

print(f"✅ 4 imágenes guardadas por destello")