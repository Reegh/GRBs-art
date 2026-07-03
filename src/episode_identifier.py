# src/episode_identifier.py
import numpy as np
import sys
from gdt.core.data_primitives import TimeBins
from astropy.stats import bayesian_blocks
import gdt.core.binning.binned as binned_methods

from config_manager import ConfigManager
from data_loader import DataLoader
from background_fitter import BackgroundFitterManager

def main():
    if len(sys.argv) < 2:
        print("Uso: python src/episode_identifier.py <detector_control>")
        print("Ejemplo: python src/episode_identifier.py n1")
        detector_control = 'n1'
    else:
        detector_control = sys.argv[1]
    
    detector_control = str(detector_control).strip().lower()

    config = ConfigManager("config.yaml")
    data_loader = DataLoader(config)
    bkg_fitter = BackgroundFitterManager(config)

    print("=" * 60)
    print(f"BÚSQUEDA DE INTERVALOS PARA YAML (Buscando: '{detector_control}')")
    print("=" * 60)

    cspecs = data_loader.load_cspec_data()
    t90_params = config.get_t90_params()
    t0 = float(t90_params["t0"])
    T90 = float(t90_params["T90"])

    burst_start = t0
    burst_end = t0 + T90

    print("-> Calculando background de referencia...")
    bkgds = bkg_fitter.fit_background(cspecs, burst_start, burst_end)

    print("-> Extrayendo detector y generando curva de luz neta...")
    
    llaves_cspecs = list(cspecs._data_dict.keys()) if hasattr(cspecs, '_data_dict') else []
    llaves_bkgds = list(bkgds._data_dict.keys()) if hasattr(bkgds, '_data_dict') else []

    index_match = None
    real_cspec_key = None
    
    for idx, k in enumerate(llaves_cspecs):
        if detector_control in str(k).lower():
            index_match = idx
            real_cspec_key = k
            break

    if index_match is not None and index_match < len(llaves_bkgds):
        real_bkg_key = llaves_bkgds[index_match]
    else:
        real_bkg_key = None

    if real_cspec_key is None or real_bkg_key is None:
        print(f"\n[!] Error: No se pudo emparejar '{detector_control}' posicionalmente.")
        return

    print(f"   [Mapeo Posicional Exitoso]")
    print(f"   - Archivo Procesado (Índice {index_match}): {real_cspec_key}")
    print(f"   - Llave BKGD        (Índice {index_match}): {real_bkg_key}")

    # Extraemos los objetos validados
    cspec_det = cspecs.get_item(real_cspec_key)
    bkg_det = bkgds.get_item(real_bkg_key)

    # 1. Rebinamos ÚNICAMENTE los datos a 1.024s para estabilizar el ruido de Poisson
    print("   [Suavizado] Rebinando datos experimentales a 1.024s...")
    cspec_rebinned = cspec_det.rebin_time(binned_methods.rebin_by_time, 1.024)
    lc_total = cspec_rebinned.to_lightcurve()

    # 2. Control adaptativo de canales utilizando integración nativa de la GDT
    energy_ranges = config.config.get("energy_ranges", {})
    if detector_control.startswith("b"):
        e_min, e_max = energy_ranges.get("bgo", [325.0, 25000.0])
    else:
        e_min, e_max = energy_ranges.get("nai", [50.0, 900.0])

    if bkg_det.num_chans <= 8:
        # Para CTIME usamos el objeto integrado completo para mantener la amplitud
        bkg_lc_raw = bkg_det.integrate_energy()
    else:
        # Para CSPEC recortamos el rango energético antes de integrar
        bkg_lc_raw = bkg_det.slice_energy(e_min, e_max).integrate_energy()
    
    # 3. INTERPOLACIÓN CON NUMPY: Mapeamos usando la propiedad 'time_centroids' validada
    print("   [Ajuste] Mapeando polinomio de background a la nueva grilla temporal...")
    bkg_rates_interp = np.interp(lc_total.centroids, bkg_lc_raw.time_centroids, bkg_lc_raw.rates)
    
    # Resta de vectores alineados en el tiempo
    net_rates = lc_total.rates - bkg_rates_interp
    net_counts = net_rates * lc_total.widths
    
    # Instanciamos la curva de luz neta suavizada para los bloques
    lc = TimeBins(
        counts=net_counts,
        lo_edges=lc_total.lo_edges,
        hi_edges=lc_total.hi_edges,
        exposure=lc_total.exposure
    )

    print("-> Segmentando con bloques bayesianos sobre datos suavizados...")
    
    t_centroids, unique_idx = np.unique(lc.centroids, return_index=True)
    x_counts = lc.counts[unique_idx]
    
    sigma_counts = np.sqrt(np.abs(lc_total.counts))[unique_idx]
    sigma_counts[sigma_counts == 0.0] = 1.0
    
    # Con el ruido mitigado por el rebinado, p0=0.05 ahora sí morderá la bajada completa del pulso
    block_edges = bayesian_blocks(
        t_centroids, 
        x_counts, 
        sigma=sigma_counts, 
        fitness='measures', 
        p0=0.05
    )
    
    new_lo = block_edges[:-1]
    new_hi = block_edges[1:]
    new_counts = []
    new_exposure = []
    
    for lo, hi in zip(new_lo, new_hi):
        mask = (lc.centroids >= lo) & (lc.centroids < hi)
        new_counts.append(np.sum(lc.counts[mask]))
        new_exposure.append(np.sum(lc.exposure[mask]))
        
    binned_lc = TimeBins(
        counts=np.array(new_counts),
        lo_edges=new_lo,
        hi_edges=new_hi,
        exposure=np.array(new_exposure)
    )
    
    print(f"-> Segmentación completada. Se generaron {len(binned_lc.rates)} bloques temporales.")
    print(f"   [Debug] Tasa neta máxima detectada en los bloques: {np.max(binned_lc.rates):.3f} cts/s")

    signal_intervals = []
    for lo, hi, r in zip(binned_lc.lo_edges, binned_lc.hi_edges, binned_lc.rates):
        if r > 0.0: 
            signal_intervals.append((float(lo), float(hi)))

    print(f"   [Debug] Bloques significativos que superaron el fondo (rate > 0): {len(signal_intervals)}")

    if not signal_intervals:
        print("\n[!] Alerta: Ningún bloque bayesiano posee tasas por encima del nivel del fondo.")
        return

    main_pulse = signal_intervals[0]
    soft_tail = signal_intervals[1:]

    print("\n" + "#" * 45)
    print("fitting_params:")
    print("  # Episodio 1: Pulso Principal (Modelo recomendado: Compt)")
    print(f"  analysis_start: {main_pulse[0]:.3f}")
    print(f"  analysis_stop: {main_pulse[1]:.3f}")
    print("  model_type: 'Compt'")
    
    if soft_tail:
        print("\n  # Episodios de la Cola Suave detectados (Modelo recomendado: BB)")
        for i, tail in enumerate(soft_tail):
            print(f"  # Bloque de cola #{i+1} -> start: {tail[0]:.3f}, stop: {tail[1]:.3f} (Sugerido: 'BB')")
    else:
        print("\n  # No se detectaron bloques posteriores significativos para una cola suave.")
    print("#" * 45 + "\n")

if __name__ == "__main__":
    main()