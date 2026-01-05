# src/t90_calculator.py
import numpy as np
from gdt.missions.fermi.gbm.phaii import GbmPhaii

class T90Calculator:
    """Calculador de T90 para light curves de GBM"""
    
    @staticmethod
    def calculate(phaii: GbmPhaii) -> tuple:
        """
        Calcula T5, T95 y T90 para una curva de luz dada
        
        Args:
            phaii: Objeto Phaii con datos de la curva de luz
            
        Returns:
            Tupla con (T5, T95, T90)
        """
        times = phaii.data.time_centroids
        counts = phaii.data.counts.sum(axis=1)
        
        cumulative = np.cumsum(counts)
        cumulative = (cumulative - cumulative.min()) / (cumulative.max() - cumulative.min())
        
        T5 = np.interp(0.05, cumulative, times)
        T95 = np.interp(0.95, cumulative, times)
        T90 = T95 - T5
        
        return float(T5), float(T95), float(T90)
    
    @staticmethod
    def print_results(T5: float, T95: float, T90: float):
        """Imprime resultados formateados de T90"""
        print("Resultados T90:")
        print(f"   T5:  {T5:.2f} s")
        print(f"   T95: {T95:.2f} s")
        print(f"   T90: {T90:.2f} s")