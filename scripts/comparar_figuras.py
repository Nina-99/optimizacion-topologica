"""Comparación visual 8.Figuras/ vs data/figuras/replica/.

Genera un contact sheet por figura (original | réplica) en
`data/figuras/replica/comparacion/`. No calcula SSIM: las réplicas usan
matplotlib y las originales pueden tener otro render; la comparación es
visual y se apoya en `valores.csv` para los números.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "8.Figuras"
REP = ROOT / "data" / "figuras" / "replica"
OUT = REP / "comparacion"

PARES = {
    "5.1": ("fig5_1_calibracion.png", "fig5_1_calibracion.png"),
    "5.2": ("fig5_2_tda_estabilidad.png", "fig5_2_tda_estabilidad.png"),
    "5.3": ("fig5_3_cota_corregida.png", "fig5_3_cota_corregida.png"),
    "5.5": ("fig5_5_penalizacion.png", "fig5_5_penalizacion.png"),
    "5.6": ("fig5_6_malla.png", "fig5_6_malla.png"),
    "5.7": ("fig5_7_umbral.png", "fig5_7_umbral.png"),
    "5.8": ("fig5_8_rmin_falsacion.png", "fig5_8_rmin_falsacion.png"),
}


def _cargar(path: Path, alto: int = 900) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w = int(im.width * alto / im.height)
    return im.resize((w, alto), Image.LANCZOS)


def _contacto(nombre: str, original: Path, replica: Path) -> Path:
    a = _cargar(original)
    b = _cargar(replica)
    margen = 24
    lienzo = Image.new("RGB", (a.width + b.width + margen * 3, a.height + 50), "white")
    lienzo.paste(a, (margen, 50))
    lienzo.paste(b, (a.width + margen * 2, 50))
    from PIL import ImageDraw
    d = ImageDraw.Draw(lienzo)
    d.text((margen, 15), "8.Figuras (objetivo)", fill="black")
    d.text((a.width + margen * 2, 15), "Réplica", fill="black")
    out = OUT / f"comparacion_{nombre}.png"
    lienzo.save(out)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fig, (orig, rep) in PARES.items():
        o, r = ORIG / orig, REP / rep
        if o.exists() and r.exists():
            print("[OK]", _contacto(fig, o, r))

    # Fig. 5.4: el objetivo combina (a) diseño y (b) convergencia.
    a = REP / "fig5_4_simp_base.png"
    b = REP / "fig5_4b_convergencia.png"
    if a.exists() and b.exists():
        ia, ib = _cargar(a, 500), _cargar(b, 500)
        comp = Image.new("RGB", (ia.width + ib.width + 24, 500), "white")
        comp.paste(ia, (0, 0))
        comp.paste(ib, (ia.width + 24, 0))
        tmp = OUT / "_fig5_4_combinada.png"
        comp.save(tmp)
        print("[OK]", _contacto("5.4", ORIG / "fig5_4_simp_base.png", tmp))


if __name__ == "__main__":
    main()
