# src/gbm_analysis.py
import numpy as np
import pandas as pd
import os
from typing import Tuple, List

from config_manager import ConfigManager
from data_loader import DataLoader
from background_fitter import BackgroundFitterManager
from burst_detector import BurstDetector
from spectral_fitter import SpectralFitterManager
from results_manager import ResultsManager

class GBMAnalysis:
    """Clase principal para análisis de datos GBM"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.data_loader = DataLoader(self.config)
        self.bkg_fitter = BackgroundFitterManager(self.config)
        self.burst_detector = BurstDetector(self.config)
        self.spectral_fitter = SpectralFitterManager(self.config)
        self.results_manager = ResultsManager(self.config)
        
        # Datos cargados
        self.tte7 = None
        self.cspecs = None
        self.phaii7 = None
        self.rsps = None
        self.erange_nai = None
        self.erange_bgo = None
        
        # Resultados intermedios
        self.t0 = None
        self.T90 = None
        self.bkgds = None
        self.burst_start = None
        self.burst_end = None
        self.time_ranges = None
        self.results = []
    
    def load_data(self):
        """Carga todos los datos necesarios"""
        print("=" * 50)
        print("CARGANDO DATOS")
        print("=" * 50)
        
        # Cargar datos
        self.tte7 = self.data_loader.load_tte_data()
        self.cspecs = self.data_loader.load_cspec_data()
        self.phaii7 = self.data_loader.create_phaii_from_tte(self.tte7)
        self.rsps = self.data_loader.load_response_files()
        
        # Obtener rangos de energía
        self.erange_nai, self.erange_bgo = self.data_loader.get_energy_ranges()
        
        # Leer t0 y T90 desde config
        t90_params = self.config.get_t90_params()
        self.t0 = float(t90_params["t0"])
        self.T90 = float(t90_params["T90"])
        
        print(f"t0 = {self.t0:.2f} s")
        print(f"T90 = {self.T90:.2f} s")
        print("Datos cargados exitosamente")
    
    def fit_backgrounds(self):
        """Ajusta backgrounds usando t0 y T90"""
        print("\n" + "=" * 50)
        print("AJUSTE DE BACKGROUND")
        print("=" * 50)
        
        if self.t0 is None or self.T90 is None:
            raise ValueError("t0 y T90 deben estar definidos")
        
        burst_start = self.t0
        burst_end = self.t0 + self.T90
        
        self.bkgds = self.bkg_fitter.fit_background(
            self.cspecs,
            burst_start,
            burst_end
        )
        return self.bkgds
    
    def detect_burst(self):
        """Detecta intervalo del burst"""
        print("\n" + "=" * 50)
        print("DETECCIÓN DE BURST")
        print("=" * 50)
        
        self.burst_start, self.burst_end = self.burst_detector.detect(self.phaii7)
        return self.burst_start, self.burst_end
    
    def define_time_intervals(self) -> List[Tuple[float, float]]:
        """Define intervalos de tiempo para ajuste espectral"""
        print("\n" + "=" * 50)
        print("DEFINICIÓN DE INTERVALOS")
        print("=" * 50)
        
        if self.burst_start is None or self.burst_end is None:
            raise ValueError("Burst debe detectarse primero")
        
        # Obtener configuración
        fit_params = self.config.get_fitting_params()
        time_interval = fit_params.get('time_interval', 1.0)
        
        analysis_start = fit_params.get('analysis_start')
        analysis_stop = fit_params.get('analysis_stop')
        
        start = float(analysis_start) if analysis_start is not None else self.burst_start
        stop = float(analysis_stop) if analysis_stop is not None else self.burst_end
        
        if start >= stop:
            raise ValueError(f"Rango inválido: start={start:.2f} >= stop={stop:.2f}")
        
        time_edges = np.arange(start, stop, time_interval)
        
        self.time_ranges = [
            (t, t + time_interval)
            for t in time_edges
            if t + time_interval <= stop
        ]
        
        print(f"Intervalos definidos: {len(self.time_ranges)}")
        return self.time_ranges
    
    def run_spectral_analysis(self):
        """Ejecuta análisis espectral completo"""
        print("\n" + "=" * 50)
        print("ANÁLISIS ESPECTRAL")
        print("=" * 50)
        
        self.results = self.spectral_fitter.run_spectral_analysis(
            cspecs=self.cspecs,
            bkgds=self.bkgds,
            rsps=self.rsps,
            time_ranges=self.time_ranges,
            erange_nai=self.erange_nai,
            erange_bgo=self.erange_bgo
        )
        return self.results
    
    def save_results(self):
        """Guarda resultados en archivo CSV"""
        print("\n" + "=" * 50)
        print("GUARDANDO RESULTADOS")
        print("=" * 50)
        
        return self.results_manager.save_results(self.results)
    
    def generate_plots(self):
        """Genera gráficas de los resultados"""
        print("\n" + "=" * 50)
        print("GENERANDO GRÁFICAS")
        print("=" * 50)
        
        # Obtener configuración de plotting
        plotting_config = self.config.get_plotting_params()
        
        # Verificar si se deben guardar gráficas
        if not plotting_config.get('save_plots', False):
            print("   Gráficas desactivadas en configuración")
            return
        
        # Obtener parámetros a graficar
        parameters = plotting_config.get('parameters', [])
        if not parameters:
            print("   No se especificaron parámetros para graficar")
            return
        
        # Obtener directorio de salida
        plots_dir = plotting_config.get('plots_dir', 'results/plots')
        
        # Crear directorio si no existe
        os.makedirs(plots_dir, exist_ok=True)
        
        # Convertir resultados a DataFrame
        if not self.results:
            print("   No hay resultados para graficar")
            return
        
        df = pd.DataFrame(self.results)
        
        # Llamar al método de graficado del ResultsManager
        try:
            self.results_manager.plot_parameters(df, parameters, plots_dir)
            print(f"   Gráficas generadas en: {plots_dir}")
        except Exception as e:
            print(f"   Error al generar gráficas: {e}")
    
    def run_full_analysis(self):
        """Ejecuta el flujo completo de análisis"""
        try:
            self.load_data()
            self.fit_backgrounds()
            self.detect_burst()
            self.define_time_intervals()
            self.run_spectral_analysis()
            self.save_results()
            self.generate_plots()
        except Exception as e:
            print(f"\n Error durante el analisis: {e}")
            raise

    def get_summary(self) -> dict:
        """Obtiene resumen del análisis"""
        return {
            "T90": self.T90,
            "Burst_duration": self.burst_end - self.burst_start if self.burst_start and self.burst_end else None,
            "Num_intervals": len(self.time_ranges) if self.time_ranges else 0,
            "Valid_fits": len(self.results),
            "Success_rate": len(self.results) / len(self.time_ranges) if self.time_ranges and len(self.time_ranges) > 0 else 0
        }
