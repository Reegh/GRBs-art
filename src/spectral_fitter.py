# src/spectral_fitter.py
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

from gdt.core.spectra.functions import Band, PowerLaw, Comptonized, BlackBody
from gdt.core.spectra.fitting import SpectralFitterCstat
from gdt.missions.fermi.gbm.collection import GbmDetectorCollection

from config_manager import ConfigManager

class SpectralFitterManager:
    """Manejador de ajustes espectrales"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.model_template = None  # Plantilla del modelo
        self.model_expression = None
        self.eval_indices = None
        
    def initialize_model(self):
        """Inicializa el modelo una sola vez al inicio del análisis"""
        fit_params = self.config.get_fitting_params()
        self.model_expression = fit_params.get('model_expression')
        
        if not self.model_expression:
            raise ValueError("Debe especificar 'model_expression' en config.yaml")
        
        print(f"   Modelo: {self.model_expression}")
        
        # Crear entorno seguro con los modelos disponibles
        available_models = {
            'Band': Band,
            'PowerLaw': PowerLaw,
            'Comptonized': Comptonized,
            'BlackBody': BlackBody,
        }
        
        safe_dict = {model: available_models[model] for model in available_models}
        
        try:
            # Crear el modelo base desde la expresión
            model = eval(self.model_expression, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            raise ValueError(f"Error creando modelo: {e}")
        
        # Configurar parámetros si se especifican en YAML
        if 'min_values' in fit_params:
            min_vals = [float(v) if v != 'inf' else np.inf for v in fit_params['min_values']]
            model.min_values = min_vals
            
        if 'max_values' in fit_params:
            max_vals = [float(v) if v != 'inf' else np.inf for v in fit_params['max_values']]
            model.max_values = max_vals
            
        if 'default_values' in fit_params:
            model.default_values = fit_params['default_values']
            
        if 'fixed' in fit_params:
            model.fix = fit_params['fixed']
        
        # Guardar plantilla del modelo
        self.model_template = model
        
        # Obtener índices a evaluar
        self.eval_indices = fit_params.get('eval_indices')
        if not self.eval_indices:
            raise ValueError("Debe especificar 'eval_indices' en config.yaml")
        
        print(f"   Evaluando índices: {self.eval_indices}")
        
        return model
    
    def create_model_for_interval(self):
        """Crea una copia nueva del modelo para cada intervalo"""
        if self.model_template is None:
            self.initialize_model()
        
        # Crear una copia profunda del modelo para no modificar la plantilla
        return deepcopy(self.model_template)
    
    def precompute_responses(self, rsps: GbmDetectorCollection, 
                            time_ranges: List[Tuple[float, float]]) -> List:
        """Precalcula respuestas interpoladas"""
        print(f"Precalculando respuestas para {len(time_ranges)} intervalos...")
        
        rsps_interpoladas = []
        for t0, t1 in time_ranges:
            tcent = (t0 + t1) / 2
            rsps_interpoladas.append([rsp.interpolate(tcent) for rsp in rsps])
        
        print("Respuestas precalculadas")
        return rsps_interpoladas
    
    def fit_interval(self, phas: GbmDetectorCollection, 
                    bkgds: GbmDetectorCollection,
                    rsps_interp: List, 
                    t0: float, t1: float) -> Optional[Dict]:
        """
        Ajusta un intervalo
        """
        fit_params = self.config.get_fitting_params()
        fit_options = fit_params.get('fit_options', {})
        max_fit_time = fit_params.get('max_fit_time', 10)
        error_threshold = fit_params.get('relative_error_threshold', 0.2)
        
        try:
            # Crear una nueva instancia del modelo para este intervalo
            model = self.create_model_for_interval()
            
            # Crear fitter
            fitter = SpectralFitterCstat(
                phas, bkgds.to_list(), rsps_interp, method=fit_options.get('method', 'TNC')
            )
            
            start_time = time.time()
            
            # Intentar ajuste
            try:
                fitter.fit(model, options={'maxiter': fit_options.get('maxiter', 500)})
            except Exception as e:
                print(f"Error en ajuste {t0:.3f}-{t1:.3f}: {e}")
                return None
            
            # Timeout
            if time.time() - start_time > max_fit_time:
                print(f"Timeout en ajuste {t0:.3f}-{t1:.3f}")
                return None
            
            # Validar convergencia
            if not np.isfinite(fitter.statistic):
                print(f"Stat no finita en {t0:.3f}-{t1:.3f}")
                return None
            
            if fitter.covariance is None or np.isnan(fitter.covariance).any():
                print(f"Covarianza invalida en {t0:.3f}-{t1:.3f}")
                return None
            
            # Calcular errores
            errs = np.array(fitter.asymmetric_errors(cl=0.9), dtype=float)
            params = np.array(fitter.parameters)
            
            if len(errs) != len(params):
                print(f"Inconsistencia en errores/parametros en {t0:.3f}-{t1:.3f}")
                return None
            
            # Calcular errores relativos
            eps = 1e-10
            rel_errs = np.mean(
                np.abs(errs) / np.maximum(np.abs(params[:, None]), eps),
                axis=1
            )
            
            # Evaluar errores en los índices especificados
            rel_errs_eval = rel_errs[self.eval_indices]
            max_rel = np.max(rel_errs_eval)
            
            # Guardar si errores aceptables
            if max_rel < error_threshold:
                res = self._create_result_dict(t0, t1, fitter, model, errs, rel_errs)
                print(f"{t0:.3f}–{t1:.3f} s: Fit guardado")
                return res
            else:
                print(f"{t0:.3f}–{t1:.3f} s: Errores grandes ({max_rel*100:.1f}%)")
                return None
                
        except Exception as e:
            print(f"Error en {t0}-{t1}: {e}")
            return None
    
    def _create_result_dict(self, t0: float, t1: float, 
                          fitter: SpectralFitterCstat, model,
                          errs: np.ndarray, rel_errs: np.ndarray) -> Dict:
        """Crea diccionario con resultados"""
        res = {
            "t_start": t0,
            "t_stop": t1,
            "Cstat": fitter.statistic,
            "dof": fitter.dof,
        }
        
        # Parámetros y errores
        for name, val in zip(model.param_list, fitter.parameters):
            res[name[0]] = float(val)
        
        for i, (name, val) in enumerate(zip(model.param_list, fitter.parameters)):
            err_low, err_high = errs[i]
            rel_err_mean = rel_errs[i] * 100
            res[f"{name[0]}_err_low"] = float(err_low)
            res[f"{name[0]}_err_high"] = float(err_high)
            res[f"{name[0]}_err_rel(%)"] = float(rel_err_mean)
        
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
        print("Ejecutando analisis espectral...")
        
        # Inicializar el modelo una sola vez
        self.initialize_model()
        
        # Precalcular respuestas
        rsps_interpoladas = self.precompute_responses(rsps, time_ranges)
        
        results = []
        
        # Procesar cada intervalo
        for i, (t0, t1) in enumerate(time_ranges):
            print(f"   Intervalo {i+1}/{len(time_ranges)}: {t0:.3f}–{t1:.3f} s", end=" ")
            
            # Extraer datos PHA
            phas = cspecs.to_pha(
                time_ranges=[(t0, t1)],
                nai_kwargs={'energy_range': erange_nai},
                bgo_kwargs={'energy_range': erange_bgo}
            )
            
            # Obtener respuestas interpoladas
            rsps_interp = rsps_interpoladas[i]
            
            # Ajustar
            result = self.fit_interval(phas, bkgds, rsps_interp, t0, t1)
            
            if result:
                results.append(result)
                print("Fit guardado")
            else:
                print("Fit fallo")
        
        print(f"Analisis espectral completado: {len(results)}/{len(time_ranges)} ajustes validos")
        return results