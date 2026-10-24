"""OnsagerDecomposer: mechanism-resolved, additive decomposition of Onsager transport coefficients."""
from .events import Events, liquid_events, polymer_events, residue_changes, residue_entries_exits
from .io import iter_hitpoly_frames, read_hitpoly_topology
from .macrostates import solvation_states, transitions
from .onsager import Correlations, decompose
from .results import per_macrostate, self_diffusion_table
from .shells import Shells, compute_shells

__version__ = "1.0.0"
