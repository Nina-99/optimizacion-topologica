"""Vocabulario de estados y veredictos del Cap. V.

El PDF distingue el ESTADO DE LA EVIDENCIA (Medido / Proyectado) del
VEREDICTO del contraste (Se sostiene / Acotada / No concluyente / No se
sostiene). El software no debe reducir todo a «cumple / no cumple».

Uso:
    from tda.analysis.estados import MEDIDO, CONCLUYENTE, NO_CONCLUYENTE
"""

from __future__ import annotations

MEDIDO = "Medido"
PROYECTADO = "Proyectado"

CONCLUYENTE = "Concluyente"
NO_CONCLUYENTE = "No concluyente"
ACOTADA = "Acotada"

SE_SOSTIENE = "Se sostiene"
NO_SE_SOSTIENE = "No se sostiene"


def estado_evidencia(medido: bool = True) -> str:
    """Etiqueta de evidencia: Medido si hay corrida; Proyectado si no."""
    return MEDIDO if medido else PROYECTADO


def veredicto_binario(se_sostiene: bool) -> str:
    """Veredicto para un contraste concluyente."""
    return SE_SOSTIENE if se_sostiene else NO_SE_SOSTIENE


def etiqueta(estado: str, veredicto: str) -> str:
    """Etiqueta compuesta, p. ej. «Medido · No concluyente»."""
    return f"{estado} · {veredicto}"
