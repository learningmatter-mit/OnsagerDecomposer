"""Generate docs/logo.svg: an ion inside a ring split into mechanism contributions.

python docs/make_logo.py
"""
import math
from pathlib import Path

CENTER, R_OUT, R_IN, GAP_DEG = 128, 104, 63, 3.0
# (share in %, colour, offset outward in px) - one wedge pulled out = "decomposed"
WEDGES = [(34, "#23215B", 0), (24, "#1E6A70", 0), (18, "#B8962E", 0), (14, "#D9674E", 9), (10, "#5FA8D3", 0)]


def point(r, deg, dx=0.0, dy=0.0):
    a = math.radians(deg)
    return CENTER + dx + r * math.cos(a), CENTER + dy + r * math.sin(a)


def wedge(start, end, offset):
    mid = math.radians((start + end) / 2)
    dx, dy = offset * math.cos(mid), offset * math.sin(mid)
    large = 1 if end - start > 180 else 0
    (x1, y1), (x2, y2) = point(R_OUT, start, dx, dy), point(R_OUT, end, dx, dy)
    (x3, y3), (x4, y4) = point(R_IN, end, dx, dy), point(R_IN, start, dx, dy)
    return (f"M{x1:.2f},{y1:.2f} A{R_OUT},{R_OUT} 0 {large} 1 {x2:.2f},{y2:.2f} "
            f"L{x3:.2f},{y3:.2f} A{R_IN},{R_IN} 0 {large} 0 {x4:.2f},{y4:.2f} Z")


def main():
    paths, angle = [], -90.0
    for share, colour, offset in WEDGES:
        span = 360 * share / 100
        paths.append(f'  <path d="{wedge(angle + GAP_DEG / 2, angle + span - GAP_DEG / 2, offset)}" fill="{colour}"/>')
        angle += span
    ion = (f'  <circle cx="{CENTER}" cy="{CENTER}" r="30" fill="#FFFFFF" stroke="#23215B" stroke-width="7"/>\n'
           f'  <rect x="{CENTER - 14}" y="{CENTER - 3.5}" width="28" height="7" rx="2" fill="#23215B"/>\n'
           f'  <rect x="{CENTER - 3.5}" y="{CENTER - 14}" width="7" height="28" rx="2" fill="#23215B"/>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256" '
           'role="img" aria-label="OnsagerDecomposer logo">\n'
           f'  <circle cx="{CENTER}" cy="{CENTER}" r="128" fill="#F6F4EE"/>\n' + "\n".join(paths) + "\n" + ion + "\n</svg>\n")
    out = Path(__file__).with_name("logo.svg")
    out.write_text(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
