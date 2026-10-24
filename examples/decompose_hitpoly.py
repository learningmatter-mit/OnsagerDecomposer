"""Decompose Li self-diffusion of a liquid or polymer electrolyte over a range of window lengths.

python examples/decompose_hitpoly.py <hitpoly_folder> --system liquid --timestep 2 --temperature 298 \
    --anion-center N --stride 1 --windows 1 2 5 10 20 50 100

Shells are computed once (every ``stride`` frames) and saved next to the output, then
subsampled for each window length dt = k * stride * timestep.
"""
import argparse
from pathlib import Path

import pandas as pd

import onsagerdecomposer as od


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--system", choices=["liquid", "polymer"], required=True)
    p.add_argument("--timestep", type=float, required=True, help="ps between saved frames")
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--anion-center", default="N", help="anion atom tracked for anion displacements (N: TFSI/FSI, P: PF6)")
    p.add_argument("--stride", type=int, default=1, help="read every n-th frame")
    p.add_argument("--windows", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100],
                   help="window lengths in units of stride")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="onsager_results")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    top = od.read_hitpoly_topology(args.folder)
    shells_file = out / f"shells_stride-{args.stride}.npz"
    if shells_file.exists():
        shells = od.Shells.load(shells_file)
    else:
        frames = od.iter_hitpoly_frames(args.folder, top, stride=args.stride)
        shells = od.compute_shells(top, frames, timestep=args.timestep * args.stride,
                                   anion_center=args.anion_center, n_workers=args.workers)
        shells.save(shells_file)

    rule = od.liquid_events if args.system == "liquid" else od.polymer_events
    tables, macro_tables = [], []
    for k in args.windows:
        s = shells.subsample(k)
        events = rule(s)
        corr = od.decompose(s.cation_displacements(), events.labels, events.names, s.anion_displacements(),
                            distinct=False)
        table = od.self_diffusion_table(corr, events, s.timestep, top.box**3, args.temperature)
        tables.append(table.assign(window_ps=s.timestep))
        print(f"dt = {s.timestep:g} ps\n{table}\n")
        if args.system == "liquid":
            states, names = od.solvation_states(s)
            tr = od.transitions(states, names)
            corr = od.decompose(s.cation_displacements(), tr.labels, tr.names, s.anion_displacements(),
                                distinct=False)
            macro = od.per_macrostate(od.self_diffusion_table(corr, tr, s.timestep, top.box**3, args.temperature),
                                      names)
            print(f"{macro}\n")
            macro_tables.append(macro.assign(window_ps=s.timestep))
    pd.concat(tables).to_csv(out / "events.csv")
    if macro_tables:
        pd.concat(macro_tables).to_csv(out / "macrostates.csv")


if __name__ == "__main__":
    main()
