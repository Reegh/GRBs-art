# src/gbm_analysis.py
import numpy as np
from typing import Tuple, List

from config_manager import ConfigManager
from data_loader import DataLoader
from t90_calculator import T90Calculator
from background_fitter import BackgroundFitterManager
from burst_detector import BurstDetector
from spectral_fitter import SpectralFitterManager
from results_manager import ResultsManager

class GBMAnalysis:
    """Clase principal para análisis de datos GBM"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.data_loader = DataLoader(self.config)
        self.t90_calc = T90Calculator()
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
        self.T5 = None
        self.T95 = None
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
        
        print("Datos cargados exitosamente")
    
    def calculate_t90(self) -> Tuple[float, float, float]:
        """Calcula T90 del burst"""
        print("\n" + "=" * 50)
        print("CÁLCULO DE T90")
        print("=" * 50)
        
        self.T5, self.T95, self.T90 = self.t90_calc.calculate(self.phaii7)
        self.t90_calc.print_results(self.T5, self.T95, self.T90)
        
        return self.T5, self.T95, self.T90
    
    def fit_backgrounds(self):
        """Ajusta backgrounds usando T5 y T95"""
        print("\n" + "=" * 50)
        print("AJUSTE DE BACKGROUND")
        print("=" * 50)
        
        if self.T5 is None or self.T95 is None:
            raise ValueError("T5 y T95 deben calcularse primero")
        
        self.bkgds = self.bkg_fitter.fit_background(self.cspecs, self.T5, self.T95)
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
        
        # Obtener rango de análisis (manual o automático)
        analysis_start = fit_params.get('analysis_start')
        analysis_stop = fit_params.get('analysis_stop')
        
        # Determinar inicio del análisis
        if analysis_start is not None:
            start = float(analysis_start)
            print(f"   Usando inicio MANUAL: {start:.2f} s")
        else:
            start = self.burst_start
            print(f"   Usando inicio DETECTADO: {start:.2f} s")
        
        # Determinar fin del análisis
        if analysis_stop is not None:
            stop = float(analysis_stop)
            print(f"   Usando fin MANUAL: {stop:.2f} s")
        else:
            stop = self.burst_end
            print(f"   Usando fin DETECTADO: {stop:.2f} s")
        
        # Verificar que el rango sea válido
        if start >= stop:
            raise ValueError(f"Rango inválido: start={start:.2f} >= stop={stop:.2f}")
        
        # Verificar que esté dentro del burst detectado (solo advertencia)
        if start < self.burst_start:
            print(f"   Advertencia: Inicio del análisis ({start:.2f}) es ANTES del burst detectado ({self.burst_start:.2f})")
        
        if stop > self.burst_end:
            print(f"   Advertencia: Fin del análisis ({stop:.2f}) es DESPUÉS del burst detectado ({self.burst_end:.2f})")
        
        # Generar intervalos
        time_edges = np.arange(start, stop, time_interval)
        
        self.time_ranges = [
            (t, t + time_interval) 
            for t in time_edges 
            if t + time_interval <= stop
        ]
        
        print(f"   Intervalo de análisis: {start:.2f} - {stop:.2f} s")
        print(f"   Duración total: {stop - start:.2f} s")
        print(f"   Tamaño de bin: {time_interval} s")
        print(f"   Número de intervalos: {len(self.time_ranges)}")
        
        return self.time_ranges
    
    def run_spectral_analysis(self):
        """Ejecuta análisis espectral completo"""
        print("\n" + "=" * 50)
        print("ANÁLISIS ESPECTRAL")
        print("=" * 50)
        
        # Verificar que todos los datos estén cargados
        if None in [self.cspecs, self.bkgds, self.rsps, self.time_ranges]:
            raise ValueError("Todos los datos deben cargarse antes del análisis espectral")
        
        # Ejecutar análisis espectral
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
    
    def run_full_analysis(self):
        """Ejecuta el flujo completo de análisis"""
        try:
            # Paso 1: Cargar datos
            self.load_data()
            
            # Paso 2: Calcular T90
            self.calculate_t90()
            
            # Paso 3: Ajustar backgrounds
            self.fit_backgrounds()
            
            # Paso 4: Detectar burst
            self.detect_burst()
            
            # Paso 5: Definir intervalos
            self.define_time_intervals()
            
            # Paso 6: Análisis espectral
            self.run_spectral_analysis()
            
            # Paso 7: Guardar resultados
            self.save_results()
            
            print("\n" + "=" * 50)
            print("ANÁLISIS COMPLETADO EXITOSAMENTE")
            print("=" * 50)
            
        except Exception as e:
            print(f"\n Error durante el análisis: {e}")
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