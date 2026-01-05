# src/spectral_fitter.py
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

from gdt.core.spectra.functions import Band
from gdt.core.spectra.fitting import SpectralFitterCstat
from gdt.missions.fermi.gbm.collection import GbmDetectorCollection

from config_manager import ConfigManager

class SpectralFitterManager:
    """Manejador de ajustes espectrales"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        
    def create_model(self) -> Band:
        """Crea y configura un nuevo modelo Band"""
        model_params = self.config.get_model_params()
        
        model = Band()
        
        # Convertir 'inf' string a np.inf si es necesario
        min_values = model_params['min_values']
        max_values = model_params['max_values']
        
        # Asegurarse de que 'inf' se convierta a np.inf
        min_values = [float(v) if v != 'inf' else np.inf for v in min_values]
        max_values = [float(v) if v != 'inf' else np.inf for v in max_values]
        
        model.min_values = min_values
        model.max_values = max_values
        model.default_values = model_params['default_values']
        model.fix = model_params['fixed']
        
        print(f"   Modelo Band creado:")
        print(f"   Parámetros fijos: {model.fix}")
        print(f"   Valores por defecto: {model.default_values}")
        
        return model
    
    def precompute_responses(self, rsps: GbmDetectorCollection, 
                            time_ranges: List[Tuple[float, float]]) -> List:
        """Precalcula respuestas interpoladas para cada intervalo"""
        print(f"Precalculando respuestas para {len(time_ranges)} intervalos...")
        
        rsps_interpoladas = []
        for i, (t0, t1) in enumerate(time_ranges):
            tcent = (t0 + t1) / 2
            rsps_interpoladas.append([rsp.interpolate(tcent) for rsp in rsps])
            
            if i % 10 == 0 and i > 0:
                print(f"   {i}/{len(time_ranges)} intervalos procesados")
        
        print("Respuestas precalculadas")
        return rsps_interpoladas
    
    def fit_interval(self, phas: GbmDetectorCollection, 
                    bkgds: GbmDetectorCollection,
                    rsps_interp: List, 
                    t0: float, t1: float) -> Optional[Dict]:
        """
        Realiza ajuste espectral para un intervalo específico
        """
        fit_params = self.config.get_fitting_params()
        fit_options = fit_params['fit_options']
        max_fit_time = fit_params['max_fit_time']
        error_threshold = fit_params['relative_error_threshold']
        
        try:
            # Crear un NUEVO modelo para cada ajuste
            model = self.create_model()
            
            # Crear fitter
            fitter = SpectralFitterCstat(
                phas, bkgds.to_list(), rsps_interp, method=fit_options['method']
            )
            
            start_time = time.time()
            
            # Intentar ajuste
            try:
                fitter.fit(model, options={'maxiter': fit_options['maxiter']})
            except Exception as e:
                print(f"Error en ajuste {t0:.3f}-{t1:.3f}: {e}")
                return None
            
            # Verificar timeout
            if time.time() - start_time > max_fit_time:
                print(f"Timeout en ajuste {t0:.3f}-{t1:.3f}")
                return None
            
            # Validar convergencia
            if not np.isfinite(fitter.statistic):
                print(f"Stat no finita en {t0:.3f}-{t1:.3f}")
                return None
            
            if fitter.covariance is None or np.isnan(fitter.covariance).any():
                print(f"Covarianza inválida en {t0:.3f}-{t1:.3f}")
                return None
            
            # Calcular errores
            errs = np.array(fitter.asymmetric_errors(cl=0.9), dtype=float)
            params = np.array(fitter.parameters)
            
            if len(errs) != len(params):
                print(f"Inconsistencia en errores/parámetros en {t0:.3f}-{t1:.3f}")
                return None
            
            eps = 1e-10
            rel_errs = np.mean(
                np.abs(errs) / np.maximum(np.abs(params[:, None]), eps),
                axis=1
            )
            
            # Evaluar errores de alpha y Epeak
            indices_eval = [1, 2]  # Epeak y alpha
            rel_errs_eval = rel_errs[indices_eval]
            max_rel = np.max(rel_errs_eval)
            
            # Guardar si errores aceptables
            if max_rel < error_threshold:
                res = self._create_result_dict(t0, t1, fitter, model, errs, rel_errs)
                print(f"{t0:.3f}–{t1:.3f} s: ✅ Fit guardado (C-stat={fitter.statistic:.1f})")
                return res
            else:
                print(f"{t0:.3f}–{t1:.3f} s: ❌ Errores grandes ({max_rel*100:.1f}%)")
                return None
                
        except Exception as e:
            print(f"Error crítico en {t0}-{t1}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_result_dict(self, t0: float, t1: float, 
                          fitter: SpectralFitterCstat, model: Band,
                          errs: np.ndarray, rel_errs: np.ndarray) -> Dict:
        """Crea diccionario estructurado con resultados"""
        res = {
            "t_start": t0,
            "t_stop": t1,
            "Cstat": fitter.statistic,
            "dof": fitter.dof,
            "Cstat_dof": fitter.statistic / fitter.dof,
        }
        
        # Parámetros y errores
        param_names = [name[0] for name in model.param_list]
        for name, val in zip(param_names, fitter.parameters):
            res[name] = float(val)
        
        for i, (name, val) in enumerate(zip(param_names, fitter.parameters)):
            err_low, err_high = errs[i]
            rel_err_mean = rel_errs[i] * 100
            res[f"{name}_err_low"] = float(err_low)
            res[f"{name}_err_high"] = float(err_high)
            res[f"{name}_err_rel(%)"] = float(rel_err_mean)
        
        return res
    
    def run_spectral_analysis(self, cspecs: GbmDetectorCollection,
                             bkgds: GbmDetectorCollection,
                             rsps: GbmDetectorCollection,
                             time_ranges: List[Tuple[float, float]],
                             erange_nai: Tuple[float, float],
                             erange_bgo: Tuple[float, float]) -> List[Dict]:
        """
        Ejecuta análisis espectral completo
        """
        print("⚡ Ejecutando análisis espectral...")
        
        # Precalcular respuestas
        rsps_interpoladas = self.precompute_responses(rsps, time_ranges)
        
        results = []
        valid_count = 0
        
        # Procesar cada intervalo
        for i, (t0, t1) in enumerate(time_ranges):
            print(f"   Intervalo {i+1}/{len(time_ranges)}: {t0:.3f}–{t1:.3f} s", end=" ")
            
            # Extraer datos PHA para el intervalo
            phas = cspecs.to_pha(
                time_ranges=[(t0, t1)],
                nai_kwargs={'energy_range': erange_nai},
                bgo_kwargs={'energy_range': erange_bgo}
            )
            
            # Obtener respuestas interpoladas
            rsps_interp = rsps_interpoladas[i]
            
            # Ajustar espectro (se crea un nuevo modelo dentro de fit_interval)
            result = self.fit_interval(phas, bkgds, rsps_interp, t0, t1)
            
            if result:
                results.append(result)
                valid_count += 1
            else:
                print("❌ Fit falló")
        
        print(f"✅ Análisis espectral completado: {valid_count}/{len(time_ranges)} ajustes válidos")
        return results