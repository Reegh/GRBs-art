#intento_10 CORREGIDO

#importar lo necesario
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from gdt.core import data_path
from gdt.missions.fermi.gbm.phaii import Cspec
from gdt.core.plot.lightcurve import Lightcurve as GDT_Lightcurve  # Renombrar para evitar conflicto

print("=" * 60)
print("CURVAS DE LUZ - ORGANIZADO POR DESTELLOS")
print("=" * 60)


class LightcurveAutomator: #creamos la clase
    def __init__(self, json_config_path, yaml_config_path=None):
        print(f"\nInicializando LightcurveAutomator...")
        
        # Cargamos el JSON
        try:
            with open(json_config_path, 'r') as f:
                self.json_config = json.load(f)
            print(f"✓ JSON cargado: {json_config_path}")
        except Exception as e:
            print(f"✗ Error cargando JSON: {e}")
            exit(1)
        
        # Cargamos el YAML
        self.yaml_config = None
        if yaml_config_path and Path(yaml_config_path).exists():
            try:
                with open(yaml_config_path, 'r') as f:
                    self.yaml_config = yaml.safe_load(f)
                print(f"✓ YAML cargado: {yaml_config_path}")
            except Exception as e:
                print(f"⚠ Error cargando YAML: {e}")
        else:
            print(f"ℹ YAML no encontrado: {yaml_config_path}")
        
        # Configuramos directorios
        self.data_dir = Path(self.json_config.get('default_data_dir', 'data'))
        self.results_dir = Path(self.json_config.get('default_results_dir', 'results'))
        
        print(f"📁 Directorio de datos: {self.data_dir}")
        print(f"💾 Directorio de resultados: {self.results_dir}")
        
        # Crear directorio base de resultados
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
    
    def process_all_objects(self):
        """Procesa todos los objetos del JSON"""
        print(f"\n{'='*60}")
        print(f"PROCESANDO {len(self.json_config['objects'])} OBJETOS")
        print(f"{'='*60}")
        
        # carpeta principal de lightcurves
        self.plots_dir = self.results_dir / 'lightcurves'
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        print(f" Carpeta principal de gráficos: {self.plots_dir}")
        
        results = []
        
        for obj in self.json_config['objects']:
            obj_name = obj.get('name', 'Desconocido')
            print(f"\nObjeto: {obj_name}")
            print(f"{'-'*40}")
            
            # Crear carpeta específica para este destello
            destello_dir = self.plots_dir / obj_name.replace(' ', '_')
            destello_dir.mkdir(exist_ok=True, parents=True)
            print(f"   Carpeta del destello: {destello_dir.name}/")
            
            # Verificar si hay archivos de datos
            if 'data_paths' not in obj:
                print(f"⚠ No hay data_paths para {obj_name}")
                continue
            
            # Procesamos cada archivo cspec  (SE REALIZO SEGUN LOS TUTORIALES DE GDT)
            for key in ['cspec_nai1', 'cspec_nai2', 'cspec_bgo1']:
                if key in obj['data_paths']:
                    filename = obj['data_paths'][key]
                    detector = key.replace('cspec_', '')
                    file_path = self.data_dir / filename
                    
                    print(f"  • Detector: {detector}")
                    print(f"    Archivo: {filename}")
                    
                    if file_path.exists():
                        print(f"    ✓ Encontrado")
                        
                        try:
                            # 1. Cargar datos
                            print(f"     Cargando PHAII...")
                            phaii = Cspec.open(file_path)
                            
                            # 2. Obtener rango de tiempo si existe
                            time_range = None
                            if 'analysis_config' in obj:
                                cfg = obj['analysis_config']
                                start = cfg.get('analysis_start')
                                stop = cfg.get('analysis_stop')
                                if start is not None and stop is not None:
                                    time_range = (float(start), float(stop))
                                    print(f"    ⏱ Rango: {time_range[0]} - {time_range[1]} s")
                            
                            # 3. Crear curva de luz
                            print(f"     Creando curva de luz...")
                            if time_range:
                                lc_data = phaii.to_lightcurve(time_range=time_range)
                            else:
                                lc_data = phaii.to_lightcurve()
                            
                            # 4. Graficar usando GDT_Lightcurve 
                            print(f"     Generando gráfico...")
                            lcplot = GDT_Lightcurve(data=lc_data, interactive=False)
                            
                            # Personalizar
                            plt.title(f'{obj_name} - {detector}', fontsize=14)
                            plt.xlabel('Time (s)', fontsize=12)
                            plt.ylabel('Count Rate (count/s)', fontsize=12)
                            plt.grid(True, alpha=0.3)
                            
                            # Guardar EN LA CARPETA DEL DESTELLO
                            plot_filename = f"{obj_name.replace(' ', '_')}_{detector}_lightcurve.png"
                            plot_path = destello_dir / plot_filename  # ¡Aquí está el cambio!
                            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            
                            print(f"     Guardado: {destello_dir.name}/{plot_filename}")
                            results.append({
                                'object': obj_name,
                                'detector': detector,
                                'status': 'success',
                                'plot': str(plot_path)
                            })
                            
                        except Exception as e:
                            print(f"     Error: {e}")
                            results.append({
                                'object': obj_name,
                                'detector': detector,
                                'status': 'error',
                                'error': str(e)
                            })
                    else:
                        print(f"     No encontrado: {file_path}")
                        results.append({
                            'object': obj_name,
                            'detector': detector,
                            'status': 'file_not_found'
                        })
        
        # Resumen
        print(f"\n{'='*60}")
        print("RESUMEN")
        print(f"{'='*60}")
        
        success = sum(1 for r in results if r['status'] == 'success')
        total = len(results)
        
        print(f" Éxitos: {success}/{total}")
        print(f"Estructura creada:")
        print(f"   {self.plots_dir}/")
        
        # Mostrar estructura de carpetas
        for item in self.plots_dir.iterdir():
            if item.is_dir():
                print(f"   ├── {item.name}/")
                # Mostrar archivos dentro de cada carpeta
                for plot_file in item.glob("*.png"):
                    print(f"   │   ├── {plot_file.name}")
        
        # Guardar log
        log_file = self.results_dir / 'processing_log.json'
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n Log guardado: {log_file}")
        
        return results

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Configuración
    JSON_FILE = "objects.json"
    YAML_FILE = "config.yaml"
    
    print(f" Buscando archivos de configuración...")
    print(f"   JSON: {JSON_FILE} - {'✓' if Path(JSON_FILE).exists() else '✗'}")
    print(f"   YAML: {YAML_FILE} - {'✓' if Path(YAML_FILE).exists() else '✗'}")
    
    if not Path(JSON_FILE).exists():
        print(f"\n ERROR: No se encuentra {JSON_FILE}")
        print("   Asegúrate de que el archivo existe en el directorio actual.")
        exit(1)
    
    # automatizador
    try:
        automator = LightcurveAutomator(JSON_FILE, YAML_FILE)
        results = automator.process_all_objects()
        
        print(f"\n{'='*60}")
        print(" PROCESAMIENTO COMPLETADO")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹ Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n ERROR NO ESPERADO: {e}")
        import traceback
        traceback.print_exc()