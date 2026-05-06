import os
import re
import matplotlib.pyplot as plt
from gdt.core.plot.lightcurve import Lightcurve
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.phaii import GbmPhaii
from gdt.core.binning.unbinned import bin_by_time
from gdt.missions.fermi.gbm.collection import GbmDetectorCollection

from config_manager import ConfigManager
from background_fitter import BackgroundFitterManager

class LightCurveBackgroundGenerator:
    """
    Genera curvas de luz con background integrado
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.bkg_manager = BackgroundFitterManager(self.config)
        self.grb_nombre = None
        self.detectores_tte = {}
        self.detectores_cspec = {}

        print(f"Configuración cargada: {config_path}")
        self._mapear_archivos_desde_yaml()

    def _mapear_archivos_desde_yaml(self):
        """Mapea pares TTE y CSPEC definidos en el YAML"""
        data_paths = self.config.config.get("data_paths", {})
        
        for key, path in data_paths.items():
            if not os.path.exists(path):
                continue
            
            det_id = key.replace("tte_", "").replace("cspec_", "")
            
            if key.startswith("tte_"):
                self.detectores_tte[det_id] = path
                if self.grb_nombre is None:
                    self.grb_nombre = self._extraer_nombre_grb(os.path.basename(path))
            elif key.startswith("cspec_"):
                self.detectores_cspec[det_id] = path

        print(f"Detectores listos para Lightcurve + Bkg: {list(self.detectores_tte.keys())}")

    def _extraer_nombre_grb(self, filename: str) -> str:
        match = re.search(r"(bn\d+)", filename, re.IGNORECASE)
        return match.group(1).lower() if match else filename.split(".")[0].lower()

    def _obtener_rango_energia(self, det_id: str):
        energy_ranges = self.config.config.get("energy_ranges", {})
        if det_id.startswith("b"):
            return energy_ranges.get("bgo", [325.0, 25000.0])
        return energy_ranges.get("nai", [50.0, 900.0])

    def generar_con_background(self, det_id: str):
        """Genera la LC + Fit de Background para un detector específico con selección de rango"""
        try:
            # 1. Cargar Datos (TTE y CSPEC)
            tte_data = GbmTte.open(self.detectores_tte[det_id])
            cspec_data = GbmPhaii.open(self.detectores_cspec[det_id])
            
            # 2. Ajustar Background usando tu Manager
            t90_params = self.config.get_t90_params()
            cspec_coll = GbmDetectorCollection.from_list([cspec_data], dets=[det_id])
            
            bkgds_coll = self.bkg_manager.fit_background(
                cspec_coll, 
                float(t90_params["t0"]), 
                float(t90_params["T90"])
            )
            bkgd_obj = list(bkgds_coll)[0]

            # 3. Preparar Curva de Luz y Background
            e_min, e_max = self._obtener_rango_energia(det_id)
            phaii = tte_data.to_phaii(bin_by_time, 1.024, time_ref=0.0)
            lc_data = phaii.slice_energy((e_min, e_max)).to_lightcurve()
            lc_bkgd = bkgd_obj.slice_energy(e_min, e_max)

            # 4. Configuración de la Selección
            plot_sel = self.config.config.get("plot_selection", {})
            selection_enabled = plot_sel.get("enabled", False)
            # Extraer el directorio del YAML (con fallback por seguridad)
            out_dir = plot_sel.get("output_directory", "results/plots_with_background")
            
            # 5. Graficado
            plt.figure(figsize=(12, 7))
            lcplot = Lightcurve(data=lc_data, background=lc_bkgd)
            
            if selection_enabled:
                sel_start = float(plot_sel.get("start", 0))
                sel_end = float(plot_sel.get("end", 0))
                lc_selection = lc_data.slice(sel_start, sel_end)
                lcplot.add_selection(lc_selection)

            ax = plt.gca()
            grb_display = self.grb_nombre.upper().replace("BN", "GRB ")
            ax.set_title(f"{grb_display} | Detector {det_id.upper()} | Background Fit", fontsize=14, pad=15)
            ax.set_xlabel("Tiempo desde trigger (s)")
            ax.set_ylabel("Tasa de conteo (cts/s)")
            ax.set_xlim(-25.0, 150.0)
            ax.grid(True, alpha=0.3, linestyle="--")

            # 6. Guardar usando la ruta definida en plot_selection
            os.makedirs(out_dir, exist_ok=True)
            output = os.path.join(out_dir, f"bkg_lc_{self.grb_nombre}_{det_id}.png")
            
            plt.savefig(output, dpi=300, bbox_inches="tight")
            plt.close()
            print(f" -> [OK] Gráfica con selección generada: {output}")

        except Exception as e:
            print(f" -> [ERROR] Falló detector {det_id}: {e}")

    def generar_todos(self):
        for det_id in self.detectores_tte:
            if det_id in self.detectores_cspec:
                self.generar_con_background(det_id)
            else:
                print(f" -> [SKIP] Falta archivo CSPEC para {det_id}")