# src/data_loader.py
import os
from typing import Tuple

from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.phaii import GbmPhaii
from gdt.missions.fermi.gbm.collection import GbmDetectorCollection
from gdt.core.binning.unbinned import bin_by_time
from gdt.missions.fermi.gbm.response import GbmRsp2 as RSP

from config_manager import ConfigManager

class DataLoader:
    """Cargador de datos GBM"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        
    def load_tte_data(self) -> GbmTte:
        """Carga datos TTE para análisis de curvas de luz"""
        tte_path = self.config.get_data_path('tte_nai1')
        if not os.path.exists(tte_path):
            raise FileNotFoundError(f"Archivo TTE no encontrado: {tte_path}")
        
        print(f"Cargando datos TTE: {tte_path}")
        return GbmTte.open(tte_path)
    
    def load_cspec_data(self) -> GbmDetectorCollection:
        """Carga datos CSPEC de los detectores configurados"""
        candidates = ['cspec_nai1', 'cspec_nai2', 'cspec_bgo1']
        phaii_list = []
        
        for det in candidates:
            pha_path = self.config.get_data_path(det)
            if pha_path is None:          # el detector no está en config → omitir
                continue
            if not os.path.exists(pha_path):
                raise FileNotFoundError(f"Archivo PHA no encontrado: {pha_path}")
            
            print(f"Cargando datos CSPEC: {pha_path}")
            phaii_list.append(GbmPhaii.open(pha_path))
        
        if not phaii_list:
            raise ValueError("No se encontraron detectores CSPEC en la configuración")
        
        return GbmDetectorCollection.from_list(phaii_list)
    
    def load_response_files(self) -> GbmDetectorCollection:
        """Carga archivos de respuesta de los detectores configurados"""
        candidates = ['rsp_nai1', 'rsp_nai2', 'rsp_bgo1']
        rsp_list = []
        
        for key in candidates:
            rsp_path = self.config.get_data_path(key)
            if rsp_path is None:
                continue
            if not os.path.exists(rsp_path):
                raise FileNotFoundError(f"Archivo RSP no encontrado: {rsp_path}")
            
            print(f"Cargando respuesta: {rsp_path}")
            rsp_list.append(RSP.open(rsp_path))
        
        if not rsp_list:
            raise ValueError("No se encontraron archivos de respuesta en la configuración")
        
        return GbmDetectorCollection.from_list(rsp_list)
    
    def create_phaii_from_tte(self, tte: GbmTte) -> GbmPhaii:
        """Convierte datos TTE a PHAII para análisis de curvas de luz"""
        print("Convirtiendo TTE a PHAII...")
        return tte.to_phaii(bin_by_time, 1.024, time_ref=0.0)
    
    def get_energy_ranges(self) -> Tuple[tuple, tuple]:
        """Obtiene rangos de energía para NAI y BGO"""
        erange_nai = self.config.get_energy_range('nai')
        erange_bgo = self.config.get_energy_range('bgo')
        return erange_nai, erange_bgo