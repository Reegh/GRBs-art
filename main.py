# main.py
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gbm_analysis import GBMAnalysis

def main():
    """Función principal para ejecutar el análisis"""
    
    # Crear instancia del análisis
    analysis = GBMAnalysis('config.yaml')
    
    # Ejecutar análisis completo
    print("Iniciando análisis GBM completo...")
    analysis.run_full_analysis()
    
    # Mostrar resumen
    summary = analysis.get_summary()
    print("\nResumen del análisis:")
    for key, value in summary.items():
        print(f"   {key}: {value}")

def run_step_by_step():
    """Ejecutar análisis paso a paso"""
    analysis = GBMAnalysis('config.yaml')
    
    # Paso 1: Cargar datos
    analysis.load_data()
    
    # Paso 2: Calcular T90
    analysis.calculate_t90()
    
    # Paso 3: Ajustar backgrounds
    analysis.fit_backgrounds()
    
    # Paso 4: Detectar burst
    analysis.detect_burst()
    
    # Paso 5: Definir intervalos
    analysis.define_time_intervals()
    
    # Paso 6: Análisis espectral
    analysis.run_spectral_analysis()
    
    # Paso 7: Guardar resultados
    analysis.save_results()
    
    # Paso 8: Generar gráficas
    analysis.generate_plots()
    

if __name__ == "__main__":
    # Ejecutar análisis completo
    main()
    
    # run_step_by_step()