# src/results_manager.py
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from config_manager import ConfigManager

class ResultsManager:
    """Manejador de resultados del análisis"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        
    def save_results(self, results: List[Dict], 
                    output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Guarda resultados en archivo CSV
        
        Args:
            results: Lista de resultados de ajustes
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            DataFrame con resultados
        """
        if not results:
            print("No hay resultados para guardar")
            return pd.DataFrame()
        
        # Crear DataFrame
        df = pd.DataFrame(results)
        
        # Ordenar por tiempo de inicio
        df = df.sort_values('t_start').reset_index(drop=True)
        
        # Determinar archivo de salida
        if output_file is None:
            output_file = self.config.get_output_params()['file_name']
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Guardar en CSV
        df.to_csv(output_file, index=False)
        print(f"   Resultados guardados en: {output_file}")
        print(f"   Total de ajustes válidos: {len(df)}")
        
        return df
    
    def load_results(self, input_file: str) -> pd.DataFrame:
        """
        Carga resultados previos desde archivo CSV
        
        Args:
            input_file: Nombre del archivo de entrada
            
        Returns:
            DataFrame con resultados
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Archivo no encontrado: {input_file}")
        
        df = pd.read_csv(input_file)
        print(f"Resultados cargados desde: {input_file}")
        print(f"   {len(df)} ajustes cargados")
        
        return df
    
    def plot_parameters(self, df: pd.DataFrame, 
                       parameters: List[str] = None,
                       output_dir: Optional[str] = None):
        """
        Grafica la evolución temporal de parámetros especificados
        
        Args:
            df: DataFrame con resultados
            parameters: Lista de nombres de parámetros a graficar.
                       Si es None, se obtienen del archivo de configuración.
            output_dir: Directorio donde guardar las figuras (opcional)
        """
        if df.empty:
            print("No hay datos para graficar")
            return
        
        # Obtener parámetros del archivo de configuración si no se especifican
        if parameters is None:
            try:
                parameters = self.config.config.get('plotting', {}).get('parameters', [])
            except:
                parameters = []
        
        # Si aún no hay parámetros, detectar automáticamente del CSV
        if not parameters:
            # Buscar columnas que no sean de error ni metadatos
            exclude_patterns = ['t_', 'Cstat', 'dof', '_err_', '_rel']
            all_columns = df.columns.tolist()
            
            for col in all_columns:
                if not any(pattern in col for pattern in exclude_patterns):
                    if '_err_' not in col and '_rel' not in col:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            parameters.append(col)
        
        # Calcular tiempo central de cada bin
        df["t_center"] = 0.5 * (df["t_start"] + df["t_stop"])
        
        print(f"Graficando {len(parameters)} parámetros: {parameters}")
        
        # Crear directorio de salida si se especifica
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Graficar cada parámetro
        for param in parameters:
            if param not in df.columns:
                print(f"{param} no está en los resultados, se omite.")
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Verificar si existen errores para este parámetro
            err_low_col = f"{param}_err_low"
            err_high_col = f"{param}_err_high"
            
            if err_low_col in df.columns and err_high_col in df.columns:
                err_low = df[err_low_col].abs()
                err_high = df[err_high_col].abs()
                
                plt.errorbar(
                    df["t_center"],
                    df[param],
                    yerr=[err_low, err_high],
                    fmt="o",
                    ecolor="gray",
                    elinewidth=1,
                    capsize=3,
                    ms=4,
                    color="tab:blue"
                )
            else:
                plt.plot(df["t_center"], df[param], "o", ms=4, color="tab:blue")
            
            plt.xlabel("Tiempo (s)")
            plt.ylabel(param)
            plt.title(f"Evolución temporal de {param}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Guardar si se especifica directorio
            if output_dir:
                plt.savefig(f"{output_dir}/{param}_evolution.png", dpi=150, bbox_inches='tight')
                print(f"Guardado: {output_dir}/{param}_evolution.png")
            
            plt.show()
        
        # Limpiar columna temporal
        df.drop("t_center", axis=1, inplace=True)