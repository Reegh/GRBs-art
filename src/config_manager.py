# src/config_manager.py
import yaml
import os

class ConfigManager:
    """Manejador de configuración desde archivo YAML"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> dict:
        """Carga configuración desde archivo YAML"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def get_data_path(self, key: str) -> str:
        """Obtiene ruta de archivo de datos"""
        return self.config['data_paths'][key]
    
    def get_energy_range(self, detector_type: str) -> tuple:
        """Obtiene rango de energía para tipo de detector"""
        return tuple(self.config['energy_ranges'][detector_type])
    
    def get_background_params(self) -> dict:
        """Obtiene parámetros para ajuste de background"""
        return self.config['background']
    
    def get_burst_params(self) -> dict:
        """Obtiene parámetros para detección de burst"""
        return self.config['burst_detection']
    
    def get_fitting_params(self) -> dict:
        """Obtiene parámetros para ajuste espectral"""
        return self.config['spectral_fitting']
    
    def get_output_params(self) -> dict:
        """Obtiene parámetros de salida"""
        return self.config['output']
    
    def get_model_params(self) -> dict:
        """Obtiene parámetros específicos del modelo"""
        return self.config['spectral_fitting']['model_params']
    
    def get_plotting_params(self) -> dict:
        """Obtiene parámetros para gráficas"""
        return self.config.get('plotting', {})
    
    def get_all_config(self) -> dict:
        """Obtiene toda la configuración"""
        return self.config