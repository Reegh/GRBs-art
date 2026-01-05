# src/background_fitter.py
import numpy as np
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.missions.fermi.gbm.collection import GbmDetectorCollection

from config_manager import ConfigManager

class BackgroundFitterManager:
    """Manejador de ajuste de background"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        
    def fit_background(self, cspecs: GbmDetectorCollection, 
                      T5: float, T95: float) -> GbmDetectorCollection:
        """
        Ajusta background usando rangos pre/post burst
        
        Args:
            cspecs: Colección de detectores
            T5: Tiempo T5 del burst
            T95: Tiempo T95 del burst
            
        Returns:
            Colección de backgrounds ajustados
        """
        print("Ajustando backgrounds...")
        
        bg_params = self.config.get_background_params()
        pre_range = bg_params['pre_bkg_range']
        post_range = bg_params['post_bkg_range']
        order = bg_params['polynomial_order']
        
        # Definir rangos de background
        bkgd_range = [
            (T5 - pre_range[0], T5 - pre_range[1]),
            (T95 + post_range[0], T95 + post_range[1])
        ]
        
        print(f"   Rangos de background: {bkgd_range}")
        print(f"   Orden del polinomio: {order}")
        
        # Crear fitters de background
        backfitters = [
            BackgroundFitter.from_phaii(cspec, Polynomial, time_ranges=bkgd_range)
            for cspec in cspecs
        ]
        backfitters = GbmDetectorCollection.from_list(
            backfitters, dets=cspecs.detector()
        )
        
        # Ajustar backgrounds
        backfitters.fit(order=order)
        
        # Interpolar para todo el rango de tiempo
        bkgds = backfitters.interpolate_bins(
            cspecs.data()[0].tstart, cspecs.data()[0].tstop
        )
        bkgds = GbmDetectorCollection.from_list(bkgds, dets=cspecs.detector())
        
        print("Backgrounds ajustados correctamente")
        return bkgds
    
    def plot_background_fit(self, backfitters: GbmDetectorCollection, 
                          detector_index: int = 0):
        """
        Genera plot del ajuste de background (opcional)
        
        Args:
            backfitters: Colección de fitters
            detector_index: Índice del detector a plotear
        """
        try:
            import matplotlib.pyplot as plt
            
            backfitter = backfitters[detector_index]
            fig = backfitter.plot()
            plt.title(f"Background Fit - Detector {backfitters.detector()[detector_index]}")
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib no disponible para plotting")