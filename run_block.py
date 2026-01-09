# run_block.py
import sys
import os
import json
import copy
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gbm_analysis import GBMAnalysis
from config_manager import ConfigManager


class BlockRunner:
    """Ejecuta análisis por bloques de objetos/GRBs definidos en un archivo JSON"""
    
    def __init__(self, base_config_file="config.yaml", objects_file="objects.json"):
        self.base_config_file = base_config_file
        self.objects_file = objects_file
        self.base_config = ConfigManager(base_config_file)
        self.objects_data = self.load_objects()
    
    def load_objects(self):
        """Carga los objetos/GRBs desde el archivo JSON"""
        try:
            with open(self.objects_file, 'r') as f:
                data = json.load(f)
            
            if 'objects' not in data:
                raise ValueError("El archivo JSON debe contener una clave 'objects'")
            
            for i, obj in enumerate(data['objects']):
                if 'data_paths' not in obj:
                    raise ValueError(f"Objeto '{obj.get('name', f'#{i}')}' no tiene la clave 'data_paths'")
            
            print(f"Cargados {len(data['objects'])} objetos desde {self.objects_file}")
            return data
            
        except FileNotFoundError:
            print(f"Archivo no encontrado: {self.objects_file}")
            print("   Por favor crea un archivo objects.json con la estructura correcta.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error al parsear JSON: {e}")
            sys.exit(1)
    
    def check_data_files(self, data_paths):
        """Verifica que existan los archivos de datos"""
        missing = []
        for key, path in data_paths.items():
            if not os.path.exists(path):
                missing.append((key, path))
        
        return missing
    
    def merge_dicts(self, d1, d2):
        """Fusiona dos diccionarios recursivamente"""
        result = copy.deepcopy(d1)
        
        for key, value in d2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def run_object(self, obj):
        """Ejecuta análisis para un objeto/GRB individual"""
        print(f"\n{'='*70}")
        print(f"OBJETO: {obj['name']}")
        print(f"{'='*70}")
        print(f"Descripcion: {obj['description']}")
        
        # Obtener directorios base
        data_dir = self.objects_data.get('default_data_dir', 'data')
        results_dir = self.objects_data.get('default_results_dir', 'results')
        
        # Usar rutas explícitas del JSON
        data_paths = {}
        for key, path in obj['data_paths'].items():
            if not os.path.isabs(path):
                data_paths[key] = os.path.join(data_dir, path)
            else:
                data_paths[key] = path
        
        # Verificar archivos
        missing = self.check_data_files(data_paths)
        if missing:
            print(f"Archivos faltantes ({len(missing)}):")
            for key, path in missing[:3]:
                print(f"   - {key}: {path}")
            if len(missing) > 3:
                print(f"   ... y {len(missing)-3} mas")
            print("   Saltando este objeto...")
            return None
        
        # Crear configuración personalizada
        config = copy.deepcopy(self.base_config.config)
        
        # Actualizar rutas de datos
        config['data_paths'] = data_paths
        
        # Actualizar configuración del modelo si existe
        if 'model_config' in obj:
            config['spectral_fitting'] = self.merge_dicts(
                config.get('spectral_fitting', {}),
                obj['model_config']
            )
        
        # Actualizar configuración de análisis si existe
        if 'analysis_config' in obj:
            for key, value in obj['analysis_config'].items():
                if value is not None:
                    config['spectral_fitting'][key] = value
        
        # Actualizar configuración de gráficas si existe
        if 'plotting_config' in obj:
            config['plotting'] = self.merge_dicts(
                config.get('plotting', {}),
                obj['plotting_config']
            )
        
        # Actualizar nombres de salida
        obj_name = obj['name']
        config['output']['file_name'] = f"{results_dir}/{obj_name}_fits.csv"
        
        # Actualizar directorio de gráficas
        if 'plotting' in config:
            config['plotting']['plots_dir'] = f"{results_dir}/plots/{obj_name}"
        
        # Crear directorios necesarios
        os.makedirs(os.path.dirname(config['output']['file_name']), exist_ok=True)
        if 'plotting' in config:
            os.makedirs(config['plotting']['plots_dir'], exist_ok=True)
        
        # Guardar configuración temporal
        import yaml
        temp_config_file = f"temp_{obj_name}.yaml"
        with open(temp_config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        try:
            # Ejecutar análisis
            analysis = GBMAnalysis(temp_config_file)
            analysis.run_full_analysis()
            
            # Recopilar resultados
            result = {
                'objeto': obj_name,
                'descripcion': obj['description'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'modelo': config['spectral_fitting'].get('model_expression', ''),
                'intervalo_tiempo': config['spectral_fitting'].get('time_interval', 1.0),
                'rango_analisis': f"{config['spectral_fitting'].get('analysis_start', 'auto')}-{config['spectral_fitting'].get('analysis_stop', 'auto')}",
                'fits_validos': len(analysis.results) if analysis.results else 0,
                'intervalos_totales': len(analysis.time_ranges) if analysis.time_ranges else 0,
                'tasa_exito': len(analysis.results) / len(analysis.time_ranges) * 100 
                               if analysis.time_ranges and len(analysis.time_ranges) > 0 else 0,
                'archivo_salida': config['output']['file_name']
            }
            
            print(f"Completado exitosamente")
            print(f"   Fits validos: {result['fits_validos']}/{result['intervalos_totales']} ({result['tasa_exito']:.1f}%)")
            print(f"   Resultados: {result['archivo_salida']}")
            
            return result
            
        except Exception as e:
            print(f"Error durante el analisis: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
    
    def run_all_objects(self, filter_names=None):
        """Ejecuta análisis para todos los objetos o solo los filtrados"""
        objects = self.objects_data['objects']
        
        # Filtrar por nombres si se especifica
        if filter_names:
            objects = [obj for obj in objects if obj['name'] in filter_names]
            if not objects:
                print(f"No se encontraron objetos con nombres: {filter_names}")
                return []
        
        print(f"\n{'#'*80}")
        print(f"INICIANDO ANALISIS POR BLOQUES")
        print(f"Total de objetos a analizar: {len(objects)}")
        print(f"{'#'*80}")
        
        results = []
        exitosos = 0
        fallidos = 0
        
        for i, obj in enumerate(objects, 1):
            print(f"\n[{i}/{len(objects)}] ", end="")
            
            result = self.run_object(obj)
            
            if result is None:
                fallidos += 1
            else:
                results.append(result)
                exitosos += 1
        
        # Generar resumen
        self.generar_resumen(results, exitosos, fallidos)
        
        return results
    
    def generar_resumen(self, results, exitosos, fallidos):
        """Genera un resumen detallado de los analisis"""
        print(f"\n{'#'*80}")
        print(f"RESUMEN FINAL")
        print(f"{'#'*80}")
        
        if not results:
            print("No se completo ningun analisis exitosamente")
            return
        
        # Crear DataFrame con resultados
        df = pd.DataFrame(results)
        
        print(f"\nResultados por objeto:")
        print("-" * 100)
        for idx, row in df.iterrows():
            print(f"{row['objeto']:30} | {row['fits_validos']:3d}/{row['intervalos_totales']:3d} fits "
                  f"| {row['tasa_exito']:5.1f}% | {row['modelo']}")
        
        # Estadísticas generales
        print(f"\nEstadisticas globales:")
        print(f"   Objetos exitosos: {exitosos}")
        print(f"   Objetos fallidos: {fallidos}")
        
        if len(results) > 0:
            total_fits = df['fits_validos'].sum()
            total_intervalos = df['intervalos_totales'].sum()
            tasa_promedio = df['tasa_exito'].mean()
            
            print(f"\n   Total de fits validos: {total_fits}")
            print(f"   Total de intervalos procesados: {total_intervalos}")
            print(f"   Tasa de exito promedio: {tasa_promedio:.1f}%")
            
            # Guardar resumen en CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = f"{self.objects_data.get('default_results_dir', 'results')}/resumen_analisis_{timestamp}.csv"
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)
            df.to_csv(summary_file, index=False)
            print(f"\nResumen guardado en: {summary_file}")
            
            # Analisis por modelo
            print(f"\nAnalisis por modelo:")
            model_stats = df.groupby('modelo').agg({
                'tasa_exito': 'mean',
                'fits_validos': 'sum',
                'objeto': 'count'
            }).round(1)
            
            model_stats.columns = ['Tasa exito (%)', 'Fits validos', 'Objetos']
            print(model_stats.to_string())
            
            # Guardar tambien en JSON para facil lectura
            summary_json = summary_file.replace('.csv', '.json')
            with open(summary_json, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"Resumen JSON guardado en: {summary_json}")
    
    def listar_objetos(self):
        """Lista todos los objetos/GRBs disponibles"""
        objects = self.objects_data['objects']
        
        print(f"\nObjetos disponibles ({len(objects)}):")
        print("=" * 80)
        
        for i, obj in enumerate(objects, 1):
            print(f"\n{i:3d}. {obj['name']}")
            print(f"     Descripcion: {obj['description']}")
            print(f"     Modelo: {obj['model_config'].get('model_expression', 'N/A')}")
            
            if 'analysis_config' in obj:
                time_int = obj['analysis_config'].get('time_interval', 1.0)
                start = obj['analysis_config'].get('analysis_start', 'auto')
                stop = obj['analysis_config'].get('analysis_stop', 'auto')
                print(f"     Configuracion: bins={time_int}s, rango={start}-{stop}")
            
            if 'data_paths' in obj:
                archivos = list(obj['data_paths'].keys())
                print(f"     Archivos: {', '.join(archivos[:3])}")
                if len(archivos) > 3:
                    print(f"               ... y {len(archivos)-3} mas")
        
        print(f"\n{'='*80}")
        print("Para analizar objetos especificos:")
        print("  python run_block.py --objetos nombre1 nombre2")
        print("Para analizar todos:")
        print("  python run_block.py")


def main():
    """Funcion principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ejecuta analisis por bloques de objetos/GRBs definidos en JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s                            # Analizar todos los objetos
  %(prog)s --listar                   # Listar objetos disponibles
  %(prog)s --objetos obj1 obj2        # Analizar objetos especificos
  %(prog)s --config my_config.yaml    # Usar configuracion base diferente
  %(prog)s --json my_objects.json     # Usar archivo JSON diferente
        """
    )
    
    parser.add_argument("--objetos", nargs="+", 
                       help="Nombres de objetos especificos a analizar")
    parser.add_argument("--config", default="config.yaml", 
                       help="Archivo de configuracion base YAML")
    parser.add_argument("--json", default="objects.json", 
                       help="Archivo JSON con definicion de objetos")
    parser.add_argument("--listar", action="store_true", 
                       help="Listar todos los objetos disponibles")
    
    args = parser.parse_args()
    
    # Crear runner
    runner = BlockRunner(args.config, args.json)
    
    if args.listar:
        # Listar objetos disponibles
        runner.listar_objetos()
        return
    
    # Ejecutar analisis
    runner.run_all_objects(args.objetos)


if __name__ == "__main__":
    main()