# Réplica y justificación de las figuras de `8.Figuras/`

Este documento justifica cada figura del Capítulo V a partir de su **función generadora**, su **archivo de datos** y los **valores que la sostienen**. La carpeta `8.Figuras/` es la referencia visual; `data/figuras/replica/` contiene la réplica reproducible y `data/figuras/replica/valores.csv` el libro de valores.

## Cómo se replica

```bash
python scripts/replicar_figuras.py
```

Salidas:

```text
data/figuras/replica/fig5_1_calibracion.png
data/figuras/replica/fig5_2_tda_estabilidad.png
data/figuras/replica/fig5_3_cota_corregida.png
data/figuras/replica/fig5_4_simp_base.png
data/figuras/replica/fig5_4b_convergencia.png
data/figuras/replica/fig5_5_penalizacion.png
data/figuras/replica/fig5_6_malla.png
data/figuras/replica/fig5_7_umbral.png
data/figuras/replica/fig5_8_rmin_falsacion.png
data/figuras/replica/valores.csv
```

## Leyendas breves (las mismas en la app y en la réplica)

- **5.1.** (a) Nubes de `C₀` con lectura correcta de `(β₀, β₁)` frente a `τ`. (b) Acierto por nube en cada `τ`. (c) Margen de cada nube de `C`, sin ruido, respecto de `τ*`.
- **5.2.** (a1–a3) Tasa de acierto de `(β₀, β₁)` frente a `q` (`σ = q·diam X/√d`), por nube y dimensión. (b) `d_B` frente a `2d_H`, normalizados por `diam X`, en 1 680 réplicas. (c) Razón `d_B/(2d_H)`.
- **5.3.** (a) `d_B/diam X` del círculo con ruido antiguo, filtración truncada en `ε ≤ 0,6` y completa; banda de 20 réplicas. (b) `d_B` dividida por la cota probabilística `2σ(√d + √(2 ln(n/δ)))`, `δ=0,05`.
- **5.4.** (a) Diseño convergido del voladizo `60×30` (`p=3`, `f_V=0,5`, `r_min=2,4`); contorno de `u_V`, que conserva volumen. (b) Historial de compliance (log) con referentes físico, binarizado y cota convexa.
- **5.5.** (a) Compliance penalizada y del binarizado frente a `p` (log). (b) Grisura media `4ρ(1−ρ)`; bajo cada `p`, `β₁` del binarizado.
- **5.6.** (a) Diseños convergidos con tres resoluciones y `r_min` físico constante (`2,4; 4,8; 7,2`). (b) Compliance penalizada y del binarizado frente a la malla.
- **5.7.** (a) `β₀` y `β₁` del diseño base frente a `u`. (b) Fracción de volumen del binarizado. (c) Diagrama `H₁` de la filtración por subniveles de `f=1−ρ*`; banda de barras borrables con `‖f−g‖∞ ≤ 0,45`.
- **5.8.** `β₁` del binarizado (`u=0,5`) frente a `r_min` en voladizo `60×30` y MBB `120×40`. Círculos abiertos: corridas que agotaron 300 iteraciones.

## Trazabilidad figura → código → dato

| Figura | Función | Dato principal | Qué justifica |
|---|---|---|---|
| 5.1 | `fig_51_calibracion` | `fig51_calibracion_C0.csv`, `fig51_margenes_C.csv` | τ* = 0,175 es la única ventana 7/7; los márgenes de C explican la fragilidad del toro y los dos círculos. |
| 5.2 | `fig_52_estabilidad` | `fig52_replicas.csv`, `fig52_q0.csv` | 1 680 réplicas, 0 violaciones de `dB ≤ 2dH`; razones medias 0,31 / 0,25 / 0,36 y caída de la lectura escalar. |
| 5.3 | `fig_53_cota_corregida` | `fig53_truncamiento.csv`, `fig52_replicas.csv` | El truncado en ε ≤ 0,6 esconde el ciclo principal (0,042–0,117); sin truncar, `dB/diam` se satura en 0,407. Cota probabilística: 0 violaciones, razón máxima 0,44. |
| 5.4 | `fig_54a_diseno`, `fig_54b_historial` | corrida base 60×30 | Diseño convergido `β₀=1, β₁=3`; `u_V` conserva volumen; referentes `c_unif=79,1`, `c_bin=63,0`, `c_ast=55,1`. |
| 5.5 | `fig_55_penalizacion` | `data/barrido_HE2b.csv` (p) | Con `p=1` el diseño gris pierde la topología al binarizarse; `p=2,3,4` estabilizan `β₁=3` y la grisura. |
| 5.6 | `fig_56_malla` | `data/barrido_HE2b.csv` (malla) | `β₁=3` en 60×30, 120×60 y 180×90 con `r_min` físico constante; compliance del binarizado 62,6–63,0. |
| 5.7 | `fig_57_umbral` | corrida base + barrido de `u` | `β₁=3` en la meseta `u ∈ [0,05; 0,90]`; el diagrama H₁ es del campo `f=1−ρ̃`, con persistencia mínima 0,914. |
| 5.8 | `fig_58_rmin_falsacion` | `data/barrido_HE2b.csv` (voladizo y MBB) | En el voladizo `β₁=3` en los seis `r_min`; en MBB la meseta no es constante (9, 8, 8, 6, 2, 2 con `u=0,5`). |

## Comparación de valores

| Figura | Valor del PDF | Réplica actual | Estado |
|---|---:|---:|---|
| 5.1 τ* | 0,175 | 0,175 | Coincide |
| 5.2 réplicas / violaciones | 1 680 / 0 | 1 680 / 0 | Coincide |
| 5.2 razón media nativa / d=10 / d=100 | 0,31 / 0,25 / 0,36 | 0,31 / 0,25 / 0,36 | Coincide |
| 5.3 truncado q=0,05…0,20 | 0,042 / 0,079 / 0,106 / 0,117 | 0,042 / 0,079 / 0,106 / 0,117 | Coincide |
| 5.3 completo q=0,05…0,20 | 0,191 / 0,385 / 0,407 / 0,407 | 0,191 / 0,385 / 0,407 / 0,407 | Coincide |
| 5.4 iteraciones | 139 | 89 | **Difiere** (mismos valores finales; el solver converge antes) |
| 5.4 `u_V` | 0,554 | 0,556 | Diferencia < 0,002 |
| 5.4 `c_unif / c_bin / c_ast` | 79,1 / 63,1 / 55,1 | 79,09 / 63,08 / 55,15 | Diferencia < 0,1 % |
| 5.4 reducción / η | 20,2 % / 0,666 | 20,24 % / 0,669 | Diferencia < 0,2 puntos |
| 5.5 `c_bin` p=2,3,4 | 62,5 / 63,1 / 63,3 | 62,52 / 63,08 / 63,33 | Coincide |
| 5.6 `c_bin` mallas | 63,1 / 62,7 / 62,8 | 63,08 / 62,75 / 62,80 | Coincide |
| 5.7 persistencia mínima H₁ | 0,905 | 0,914 | Diferencia < 0,01 |
| 5.8 MBB `β₁(u=0,5)` | 9, 8, 8, 6, 2, 2 | 9, 8, 8, 6, 2, 2 | Coincide |

## La única diferencia estructural: iteraciones de la Fig. 5.4

La Fig. 5.4(b) del PDF dice **139 iteraciones**; la réplica actual converge en **89**. No cambia ninguna conclusión: `c_unif`, `c_bin`, `c_ast`, `β₁=3` y `u_V` coinciden dentro de la tolerancia. La diferencia proviene del criterio de parada histórico del solver, no de una métrica distinta.

## Qué se corrigió para poder replicar

- Se implementó `fig_58_rmin_falsacion`, que antes no existía.
- El barrido H.E.2 ahora guarda `beta1_u05`, porque la Fig. 5.8 del PDF usa el umbral fijo `u=0,5`, no `u_V`. Con esa convención, la MBB reproduce exactamente `9, 8, 8, 6, 2, 2`.
- `fig_54a_diseno` volvió al formato de un panel con contorno `u_V`, como el PDF.
- `fig_54b_historial` rotula el número de iteraciones y el estado de convergencia.
- `fig_56_malla` ya no imprime «β₁=3» fijo: usa los datos.
- La página H.E.2 muestra la Fig. 5.8 y la tabla MBB.

## Comparación visual

```bash
python scripts/comparar_figuras.py
```

Genera contact sheets `data/figuras/replica/comparacion/comparacion_5.*.png` con la figura original de `8.Figuras/` y la réplica. La diferencia visual esperable es de estilo (matplotlib, fuentes, resolución); la diferencia numérica está en `valores.csv` y en la tabla anterior.

## Fuente de verdad

Las figuras 5.1, 5.2, 5.3 y 5.8 son réplicas exactas o exactas dentro de la resolución. Las figuras 5.4–5.7 coinciden en los valores que sustentan el veredicto; la diferencia de iteraciones de 5.4 se reporta como diferencia de implementación, no como cambio de resultado.
