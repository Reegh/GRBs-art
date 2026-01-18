# src/light.py - VERSIÓN CON TÍTULO SIMPLIFICADO
import os
import re
import matplotlib.pyplot as plt
from gdt.core.plot.lightcurve import Lightcurve
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.binning.unbinned import bin_by_time
from config_manager import ConfigManager

class LightCurveGenerator:
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.grb_nombre = None
        self.detectores = {}
        
        print(f"📁 Configuración cargada desde: {config_path}")
        self._detectar_detectores()
    
    def _detectar_detectores(self):
        """Detecta automáticamente todos los detectores en config.yaml"""
        data_paths = self.config.config.get('data_paths', {})
        
        for key, path in data_paths.items():
            if key.startswith('tte_'):
                detector_id = key.replace('tte_', '')
                
                if os.path.exists(path):
                    filename = os.path.basename(path)
                    
                    if not self.grb_nombre:
                        self.grb_nombre = self._extraer_nombre_grb(filename)
                    
                    # Extraer detector del filename
                    detector_match = re.search(r'_tte_([a-z0-9]+)_', filename)
                    if detector_match:
                        detector_id = detector_match.group(1)
                    else:
                        parts = filename.split('_')
                        for part in parts:
                            if part in ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb',
                                       'b0','b1','b2']:
                                detector_id = part
                                break
                    
                    self.detectores[detector_id] = path
                    print(f"🔍 Detectado: {detector_id} → {os.path.basename(path)}")
        
        if not self.grb_nombre:
            self.grb_nombre = "GRB_desconocido"
        
        print(f"🔭 GRB identificado: {self.grb_nombre}")
        print(f"📊 Detectores encontrados: {list(self.detectores.keys())}")
    
    def _extraer_nombre_grb(self, filename):
        """Extrae el nombre del GRB del filename"""
        match = re.search(r'(bn\d+)', filename, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        
        parts = filename.split('_')
        for part in parts:
            if re.match(r'^(grb|bn|gbm)\d+', part.lower()):
                return part.lower()
        
        return filename.split('.')[0]
    
    def _cargar_datos_detector(self, detector_id):
        """Carga datos de un detector específico"""
        tte_path = self.detectores.get(detector_id)
        
        if not tte_path:
            raise ValueError(f"No se encontró ruta para detector {detector_id}")
        
        print(f"\n🔵 Cargando detector {detector_id.upper()}: {os.path.basename(tte_path)}")
        
        if not os.path.exists(tte_path):
            raise FileNotFoundError(f"Archivo no encontrado: {tte_path}")
        
        tte_data = GbmTte.open(tte_path)
        print(f"✅ {detector_id.upper()} cargado")
        
        return tte_data
    
    def _obtener_rango_energia_detector(self, detector_id):
        """Obtiene rango de energía apropiado para el detector"""
        energy_ranges = self.config.config.get('energy_ranges', {})
        
        if detector_id.startswith('b'):
            return energy_ranges.get('bgo', [8, 1000.0])
        else:
            return energy_ranges.get('nai', [8.0, 900.0])
    
    def generar_curvas_detector(self, detector_id, generar_bandas=True):
        """Genera curvas para un detector específico"""
        print(f"\n" + "="*60)
        print(f"   GENERANDO CURVAS DETECTOR {detector_id.upper()}")
        print("="*60)
        
        rango_completo = self._obtener_rango_energia_detector(detector_id)
        
        # 1. Curva completa
        self._curva_detector_rango(
            detector_id=detector_id,
            nombre=f"{detector_id}_total",
            energia_min=rango_completo[0],
            energia_max=rango_completo[1]
        )
        
        # 2. Curvas por bandas (solo para NAI)
        if generar_bandas and not detector_id.startswith('b'):
            bandas = [
                ("soft", rango_completo[0], 50.0),
                ("medium", 50.0, 200.0),
                ("hard", 200.0, rango_completo[1])
            ]
            
            for nombre_banda, e_min, e_max in bandas:
                self._curva_detector_rango(
                    detector_id=detector_id,
                    nombre=f"{detector_id}_{nombre_banda}",
                    energia_min=e_min,
                    energia_max=e_max
                )
        
        print(f"\n✅ Curvas generadas para detector {detector_id.upper()}")
    
    def _curva_detector_rango(self, detector_id, nombre, energia_min, energia_max, guardar=True):
        """Genera curva para un detector en un rango específico - TÍTULO SIMPLIFICADO"""
        print(f"\n🎯 {detector_id.upper()}: {nombre} ({energia_min}-{energia_max} keV)...")
        
        try:
            # Cargar datos
            tte_data = self._cargar_datos_detector(detector_id)
            
            # Convertir TTE a PHAII
            phaii = tte_data.to_phaii(bin_by_time, 1.024, time_ref=0.0)
            
            # Filtrar por energía
            phaii_filtrado = phaii.slice_energy((energia_min, energia_max))
            
            # Crear curva de luz
            lc = phaii_filtrado.to_lightcurve()
            
            # Filtrar por tiempo (-25 a 150 segundos)
            time_range = (-25.0, 150.0)
            lc = lc.slice(time_range[0], time_range[1])
            print(f"   📊 Rango temporal: {time_range[0]} a {time_range[1]} s")
            
            # Crear figura
            plt.figure(figsize=(12, 7))
            
            # Usar Lightcurve
            lcplot = Lightcurve(data=lc)
            
            # TÍTULO SIMPLIFICADO - SOLO 3 PARTES
            ax = plt.gca()
            
            # Formatear el nombre del GRB para mejor presentación
            grb_display = self.grb_nombre.upper().replace('BN', 'GRB ')
            
            # Título simple: Destello | Detector | Energía
            ax.set_title(f"{grb_display} | Detector {detector_id.upper()} | {energia_min}-{energia_max} keV", 
                        fontsize=14, pad=15)
            
            ax.set_xlabel("Tiempo desde trigger (s)", fontsize=12)
            ax.set_ylabel("Tasa de conteo (cts/s)", fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(time_range)
            
            # Guardar
            if guardar:
                os.makedirs("results", exist_ok=True)
                output_path = f"results/lightcurve_{self.grb_nombre}_{nombre}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"💾 Guardada: {output_path}")
            
            plt.close()
            print(f"✅ {detector_id.upper()}: {nombre} generada")
            return True
            
        except Exception as e:
            print(f"❌ Error en {detector_id.upper()} {nombre}: {e}")
            return False
    
    def generar_todos_los_detectores(self):
        """Genera curvas para TODOS los detectores detectados"""
        print("\n" + "="*60)
        print("   GENERANDO CURVAS PARA TODOS LOS DETECTORES")
        print("="*60)
        
        if not self.detectores:
            print("⚠️  No se encontraron detectores en la configuración")
            return
        
        for detector_id in self.detectores.keys():
            generar_bandas = not detector_id.startswith('b')
            self.generar_curvas_detector(detector_id, generar_bandas)
        
        print(f"\n🎉 ¡Todas las curvas generadas para {self.grb_nombre.upper()}!")
        print(f"📁 {len(self.detectores)} detectores procesados")
        print("📄 Revisa la carpeta 'results/'")
    
    def generar_solo_detector(self, detector_id):
        """Genera curvas solo para un detector específico"""
        if detector_id not in self.detectores:
            print(f"❌ Detector {detector_id} no encontrado")
            print(f"   Detectores disponibles: {list(self.detectores.keys())}")
            return
        
        self.generar_curvas_detector(detector_id)

if __name__ == "__main__":
    print("=" * 60)
    print("   GENERADOR DE CURVAS DE LUZ")
    print("=" * 60)
    print("\nPara usar, ejecuta:")
    print("  python3 graf.py")