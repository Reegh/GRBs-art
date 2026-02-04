# src/light.py
import os
import re
import matplotlib.pyplot as plt

from gdt.core.plot.lightcurve import Lightcurve
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.binning.unbinned import bin_by_time

from config_manager import ConfigManager


class LightCurveGenerator:
    """
    Generador de curvas de luz a partir de archivos TTE de Fermi-GBM.
    Los archivos se toman directamente de data_paths en el archivo YAML.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.grb_nombre = None
        self.detectores = {}

        print(f"Configuración cargada desde: {config_path}")
        self._cargar_detectores_desde_yaml()

    def _cargar_detectores_desde_yaml(self):
        """
        Carga los detectores definidos explícitamente en data_paths.
        Solo se consideran claves que comienzan con 'tte_'.
        """
        data_paths = self.config.config.get("data_paths", {})

        for key in data_paths:
            if not key.startswith("tte_"):
                continue

            detector_id = key.replace("tte_", "")
            path = self.config.get_data_path(key)

            if not os.path.exists(path):
                raise FileNotFoundError(f"Archivo no encontrado: {path}")

            if self.grb_nombre is None:
                self.grb_nombre = self._extraer_nombre_grb(
                    os.path.basename(path)
                )

            self.detectores[detector_id] = path
            print(f"Detector cargado: {detector_id}")

        if not self.detectores:
            raise RuntimeError(
                "No se encontraron detectores TTE en data_paths"
            )

        if self.grb_nombre is None:
            self.grb_nombre = "grb_desconocido"

        print(f"GRB identificado: {self.grb_nombre}")
        print(f"Detectores disponibles: {list(self.detectores.keys())}")

    def _extraer_nombre_grb(self, filename: str) -> str:
        """
        Extrae el identificador del GRB a partir del nombre del archivo.
        """
        match = re.search(r"(bn\d+)", filename, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        return filename.split(".")[0].lower()

    def _cargar_datos_detector(self, detector_id: str):
        """
        Carga el archivo TTE correspondiente a un detector.
        """
        tte_path = self.detectores.get(detector_id)

        if tte_path is None:
            raise ValueError(
                f"No existe información para el detector {detector_id}"
            )

        return GbmTte.open(tte_path)

    def _obtener_rango_energia_detector(self, detector_id: str):
        """
        Devuelve el rango de energía apropiado según el tipo de detector.
        """
        energy_ranges = self.config.config.get("energy_ranges", {})

        if detector_id.startswith("b"):
            return energy_ranges.get("bgo", [8.0, 1000.0])

        return energy_ranges.get("nai", [8.0, 900.0])

    def generar_curvas_detector(
        self, detector_id: str, generar_bandas: bool = True
    ):
        """
        Genera curvas de luz para un detector específico.
        """
        rango_completo = self._obtener_rango_energia_detector(detector_id)

        self._curva_detector_rango(
            detector_id,
            f"{detector_id}_total",
            rango_completo[0],
            rango_completo[1],
        )

        if generar_bandas and not detector_id.startswith("b"):
            bandas = [
                ("soft", rango_completo[0], 50.0),
                ("medium", 50.0, 200.0),
                ("hard", 200.0, rango_completo[1]),
            ]

            for nombre, e_min, e_max in bandas:
                self._curva_detector_rango(
                    detector_id,
                    f"{detector_id}_{nombre}",
                    e_min,
                    e_max,
                )

    def _curva_detector_rango(
        self,
        detector_id: str,
        nombre: str,
        energia_min: float,
        energia_max: float,
        guardar: bool = True,
    ):
        """
        Genera una curva de luz para un detector en un rango de energía.
        """
        try:
            tte_data = self._cargar_datos_detector(detector_id)

            phaii = tte_data.to_phaii(
                bin_by_time,
                1.024,
                time_ref=0.0,
            )

            phaii = phaii.slice_energy((energia_min, energia_max))
            lc = phaii.to_lightcurve()
            lc = lc.slice(-25.0, 150.0)

            # === BLOQUE DE GRAFICADO ORIGINAL (NO MODIFICADO) ===
            plt.figure(figsize=(12, 7))
            lcplot = Lightcurve(data=lc)
            ax = plt.gca()
            # ===================================================

            grb_display = self.grb_nombre.upper().replace("BN", "GRB ")

            ax.set_title(
                f"{grb_display} | Detector {detector_id.upper()} | "
                f"{energia_min}-{energia_max} keV",
                fontsize=14,
                pad=15,
            )

            ax.set_xlabel("Tiempo desde trigger (s)")
            ax.set_ylabel("Tasa de conteo (cts/s)")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xlim(-25.0, 150.0)

            if guardar:
                os.makedirs("results", exist_ok=True)
                output = (
                    f"results/lightcurve_{self.grb_nombre}_{nombre}.png"
                )
                plt.savefig(output, dpi=300, bbox_inches="tight")

            plt.close()

        except Exception as exc:
            print(
                f"Error al generar {nombre} para {detector_id}: {exc}"
            )

    def generar_todos_los_detectores(self):
        """
        Genera curvas de luz para todos los detectores definidos.
        """
        for detector_id in self.detectores:
            generar_bandas = not detector_id.startswith("b")
            self.generar_curvas_detector(detector_id, generar_bandas)
