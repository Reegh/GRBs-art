# src/burst_detector.py
import numpy as np
from gdt.missions.fermi.gbm.phaii import GbmPhaii

from config_manager import ConfigManager

class BurstDetector:
    """Detector automático de bursts"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        
    def detect(self, phaii: GbmPhaii) -> tuple:
        """
        Detecta intervalo de burst automáticamente
        
        Args:
            phaii: Objeto Phaii con light curve
            
        Returns:
            Tupla con (burst_start, burst_end)
        """
        print("Detectando burst...")
        
        detection_params = self.config.get_burst_params()
        
        # Obtener counts y tiempos
        counts = phaii.data.counts.sum(axis=1)
        times = phaii.time_range[0] + np.arange(len(counts)) * 1.0
        
        # Calcular umbral
        percentile = detection_params['percentile_for_bkg']
        sigma = detection_params['sigma_threshold']
        
        bkg_level = np.percentile(counts, percentile)
        threshold = bkg_level + sigma * np.std(
            counts[counts < np.percentile(counts, 50)]
        )
        
        # Encontrar burst
        mask = counts > threshold
        if not np.any(mask):
            raise RuntimeError("No se detectó ningún destello")
        
        burst_start = times[np.argmax(mask)]
        burst_end = times[len(mask) - np.argmax(mask[::-1]) - 1] + 1.0
        
        print(f"   Burst detectado: {burst_start:.2f} - {burst_end:.2f} s")
        print(f"   Umbral: {threshold:.1f} counts/bin")
        print(f"   Duración: {burst_end - burst_start:.2f} s")
        
        return burst_start, burst_end
    
    def plot_detection(self, phaii: GbmPhaii, burst_start: float, 
                      burst_end: float):
        """
        Genera plot de la detección del burst (opcional)
        
        Args:
            phaii: Objeto Phaii con light curve
            burst_start: Inicio del burst
            burst_end: Fin del burst
        """
        try:
            import matplotlib.pyplot as plt
            
            counts = phaii.data.counts.sum(axis=1)
            times = phaii.time_range[0] + np.arange(len(counts)) * 1.0
            
            plt.figure(figsize=(10, 5))
            plt.plot(times, counts, 'k-', linewidth=0.5)
            plt.axvspan(burst_start, burst_end, alpha=0.3, color='red', 
                       label='Burst detectado')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Counts')
            plt.title('Detección del Burst')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib no disponible para plotting")