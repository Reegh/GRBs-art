# main.py
import sys
import os
import argparse

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gbm_analysis import GBMAnalysis
from light import LightCurveGenerator
from light_with_background import LightCurveBackgroundGenerator


def run_lightcurves(config_path: str):
    """Genera curvas de luz"""
    print("\nIniciando generación de curvas de luz...")
    lc_generator = LightCurveGenerator(config_path)
    lc_generator.generar_todos_los_detectores()
    print("Curvas de luz finalizadas.")


def run_analysis(config_path: str):
    """Ejecuta el análisis GBM completo"""
    print("\nIniciando análisis GBM completo...")
    analysis = GBMAnalysis(config_path)
    analysis.run_full_analysis()

    summary = analysis.get_summary()
    print("\nResumen del análisis:")
    for key, value in summary.items():
        print(f"   {key}: {value}")

def run_lightcurves_with_background(config_path: str):
    print("\nIniciando generación de curvas de luz con background...")
    generator = LightCurveBackgroundGenerator(config_path)
    generator.generar_todos()
    print("Proceso finalizado.")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de análisis Fermi-GBM"
    )

    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Ejecutar solo el análisis GBM",
    )

    parser.add_argument(
        "--lightcurves",
        action="store_true",
        help="Generar solo curvas de luz",
    )

    parser.add_argument(
        "--light_bkgd",
        action="store_true",
        help="Generar curvas de luz con background ajustado",
    )

    args = parser.parse_args()

    config_path = "config.yaml"

    any_flag = any([args.analysis, args.lightcurves, args.light_bkgd])

    # Caso 1: no se pasa ningún flag -> ejecutar todo
    if not any_flag:
        run_lightcurves(config_path)
        run_lightcurves_with_background(config_path)
        run_analysis(config_path)
        return

    # Caso 2: solo curvas de luz
    if args.lightcurves and not args.analysis:
        run_lightcurves(config_path)
        return

    # Caso 3: solo análisis GBM
    if args.analysis and not args.lightcurves:
        run_analysis(config_path)
        return
    
    if args.light_bkgd:
        run_lightcurves_with_background(config_path)
        return

    # Caso 4: ambos flags explícitos
    run_lightcurves(config_path)
    run_analysis(config_path)


if __name__ == "__main__":
    main()
