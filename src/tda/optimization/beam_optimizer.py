"""
Optimización de Forma de Viga (Voladizo 1D)

Solver matemático puro para una viga voladizo de acero con carga puntual.
Calcula el perfil de altura óptimo h(x) para minimizar el volumen mientras 
satisface las restricciones de tensión y rigidez.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

class BeamOptimizer:
    def __init__(self, b, h0, p, N, E_s=200e9, sigma_adm=250e6, max_iter=100):
        self.b = b
        self.h0 = h0
        self.p = p
        self.N = N
        self.E_s = E_s          # Módulo de Young (Pa)
        self.sigma_adm = sigma_adm  # Tensión admisible (Pa)
        self.max_iter = max_iter
        
        # Inercia inicial
        self.I0 = (b * h0**3) / 12.0
        self.I_min = 0.05 * self.I0  # Inercia mínima de seguridad

    def optimizar_viga_completo(self, L, F, callback=None):
        """
        Optimiza una viga voladizo de longitud L cargada con fuerza puntual F en el extremo libre.
        
        Parámetros:
            L: Longitud (m)
            F: Carga puntual en el extremo libre (N)
            callback: Función para visualización en tiempo real
            
        Retorna:
            dict: Resultados finales de la optimización
        """
        # Cuadrícula
        x = np.linspace(0, L, self.N)
        dx = L / (self.N - 1)
        
        # Momento Flector (Voladizo con carga puntual en x=L)
        # M(x) = -F * (L - x)
        M = -F * (L - x)
        
        # Deflexión original (inercia constante I0)
        # y''(x) = M(x) / (E * I0)
        # y(x) = F/(6*E*I0) * (x^3 - 3L x^2)  (analítica exacta)
        curvature_orig = M / (self.E_s * self.I0)
        slope_orig = cumulative_trapezoid(curvature_orig, x, initial=0)
        Y_original = cumulative_trapezoid(slope_orig, x, initial=0)
        
        # Inicializar inercia para la optimización
        I = np.ones(self.N) * self.I0
        
        error = 1.0
        iteracion = 0
        tol = 1e-4
        
        y_adm = (L / 250.0)  # Límite de deflexión estándar para voladizo L/250
        max_M = np.max(np.abs(M))
        if max_M == 0: max_M = 1.0
            
        while error > tol and iteracion < self.max_iter:
            I_vieja = I.copy()
            
            # 1. Deflexión con inercia actual
            curvature = M / (self.E_s * I)
            slope = cumulative_trapezoid(curvature, x, initial=0)
            Y = cumulative_trapezoid(slope, x, initial=0)
            
            # 2. Actualización SIMP basada en momento
            # Queremos más inercia donde el momento es alto
            I_target = np.clip(self.I_min + (self.I0 - self.I_min) * (np.abs(M) / max_M)**self.p, self.I_min, self.I0)
            I = 0.85 * I + 0.15 * I_target
            
            error = np.linalg.norm(I - I_vieja) / np.linalg.norm(I_vieja)
            
            # Perfil de altura actual
            h_v = (12 * I / self.b) ** (1/3)
            
            # Tensiones
            sigma = (6.0 * np.abs(M)) / (self.b * h_v**2) # Pa
            sigma_MPa = sigma / 1e6
            
            # KPIs
            V_orig = self.b * self.h0 * L
            V_opt = self.b * dx * np.sum(h_v)
            saving_pct = (1.0 - V_opt / V_orig) * 100.0
            weight_saved = (V_orig - V_opt) * 7850.0  # Densidad del acero ~7850 kg/m^3
            
            if callback and iteracion % 2 == 0:
                visualization_data = {
                    "iteration": iteracion,
                    "x": x,
                    "I": I,
                    "Y": Y,
                    "Y_original": Y_original,
                    "M": M,
                    "h_v": h_v,
                    "sigma_MPa": sigma_MPa,
                    "saving_pct": saving_pct,
                    "weight_saved": weight_saved,
                    "error": error,
                    "L": L,
                    "y_adm": y_adm,
                    "max_M": max_M,
                    "F": F
                }
                callback(visualization_data)
            
            iteracion += 1
        
        # Cálculos finales
        h_v_final = (12 * I / self.b) ** (1/3)
        sigma_final = (6.0 * np.abs(M)) / (self.b * h_v_final**2)
        sigma_MPa_final = sigma_final / 1e6
        
        return {
            "x": x,
            "I": I,
            "Y": Y,
            "Y_original": Y_original,
            "M": M,
            "h_v": h_v_final,
            "iterations": iteracion,
            "final_error": error,
            "saving_pct": saving_pct,
            "weight_saved": weight_saved,
            "sigma_MPa": sigma_MPa_final,
            "L": L,
            "y_adm": y_adm,
            "F": F
        }
