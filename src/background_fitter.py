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

    def fit_background(
        self,
        cspecs: GbmDetectorCollection,
        t0: float,
        T90: float
    ) -> GbmDetectorCollection:
        """
        Ajusta background usando rangos pre/post burst definidos con T90

        Args:
            cspecs: Colección de detectores
            t0: Tiempo de referencia del burst
            T90: Duración T90 del burst

        Returns:
            Colección de backgrounds ajustados
        """
        print("Ajustando backgrounds...")

        bg_params = self.config.get_background_params()
        pre_range = bg_params['pre_bkg_range']
        post_range = bg_params['post_bkg_range']
        order = bg_params['polynomial_order']

        # Definir intervalo del burst usando T90
        t_burst_start = t0 - 0.5 * T90
        t_burst_end   = t0 + 0.5 * T90

        # Rangos de background
        bkgd_range = [
            (t_burst_start - pre_range[0], t_burst_start - pre_range[1]),
            (t_burst_end   + post_range[0], t_burst_end   + post_range[1])
        ]

        print(f"   Intervalo del burst: [{t_burst_start:.2f}, {t_burst_end:.2f}]")
        print(f"   Rangos de background: {bkgd_range}")
        print(f"   Orden del polinomio: {order}")

        # Crear fitters de background
        backfitters = [
            BackgroundFitter.from_phaii(
                cspec,
                Polynomial,
                time_ranges=bkgd_range
            )
            for cspec in cspecs
        ]

        backfitters = GbmDetectorCollection.from_list(
            backfitters, dets=cspecs.detector()
        )

        # Ajustar backgrounds
        backfitters.fit(order=order)

        # Interpolar para todo el rango de tiempo
        tstart = cspecs.data()[0].tstart
        tstop  = cspecs.data()[0].tstop

        bkgds = backfitters.interpolate_bins(tstart, tstop)
        bkgds = GbmDetectorCollection.from_list(
            bkgds, dets=cspecs.detector()
        )

        print("Backgrounds ajustados correctamente")
        return bkgds