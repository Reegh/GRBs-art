# main.py - VERSIÓN CON ORDEN ORIGINAL
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gbm_analysis import GBMAnalysis
from light import LightCurveGenerator

def main():
    """Función principal - Mismo orden que antes (GBM primero)"""
    print("\n" + "="*60)
    print("       ANÁLISIS COMPLETO DE GRBs")
    print("="*60)
    
    # PRIMERO: Análisis espectral (ORDEN ORIGINAL)
    print("\n🔬 PASO 1: ANÁLISIS ESPECTRAL COMPLETO...")
    print("-" * 40)
    
    try:
        analysis = GBMAnalysis('config.yaml')
        analysis.run_full_analysis()
        
        summary = analysis.get_summary()
        print("\n📋 RESUMEN DEL ANÁLISIS ESPECTRAL:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        print("✅ ANÁLISIS ESPECTRAL COMPLETADO")
        
    except Exception as e:
        print(f"❌ Error en análisis espectral: {e}")
    
    # SEGUNDO: Curvas de luz
    print("\n\n🎯 PASO 2: GENERANDO CURVAS DE LUZ...")
    print("-" * 40)
    
    try:
        generador = LightCurveGenerator('config.yaml')
        print(f"\n🔭 GRB: {generador.grb_nombre.upper()}")
        print(f"📊 Detectores detectados: {list(generador.detectores.keys())}")
        
        print("\n📈 Generando todas las curvas automáticamente...")
        generador.generar_todos_los_detectores()
        print("✅ CURVAS DE LUZ COMPLETADAS")
        
    except Exception as e:
        print(f"❌ Error en curvas de luz: {e}")
    
    # RESUMEN FINAL
    print("\n" + "="*60)
    print("       PROCESO COMPLETADO")
    print("="*60)
    print("\n📁 Archivos generados en orden:")
    print("  1. results/valid_fits.csv (resultados espectrales)")
    print("  2. results/plots/ (gráficas de parámetros)")
    print("  3. results/lightcurve_*.png (curvas de luz)")
    print("\n🎉 ¡Análisis completo finalizado!")

# Funciones originales para compatibilidad
def run_step_by_step():
    """Función original - Análisis paso a paso"""
    analysis = GBMAnalysis('config.yaml')
    
    analysis.load_data()
    analysis.calculate_t90()
    analysis.fit_backgrounds()
    analysis.detect_burst()
    analysis.define_time_intervals()
    analysis.run_spectral_analysis()
    analysis.save_results()
    analysis.generate_plots()

def generate_lightcurves_only():
    """Función original - Solo curvas de luz"""
    generador = LightCurveGenerator('config.yaml')
    generador.generar_todos_los_detectores()

if __name__ == "__main__":
    # Ejecuta TODO en el orden original
    main()
    
    # Para usar las funciones originales desde otros scripts:
    # run_step_by_step()
    # generate_lightcurves_only()