import math
import pickle
import numpy as np
import pandas as pd
from utils import read_xyz, read_xyz_monomer_unit
from itertools import permutations
from scipy.fft import fft, ifft
from scipy.spatial import Voronoi, voronoi_plot_2d
from pymatgen.core import Structure
from pymatgen.core import Lattice
from scipy.spatial import ConvexHull
from scipy import stats
from pymatgen.io.zeopp import ZeoCssr
import pymatgen.io.zeopp as zeopp
from pymatgen.io.ase import AseAtomsAdaptor
import ase
import pickle as pkl
import copy
import matplotlib
import networkx as nx
from pymatgen.core.periodic_table import Element
import seaborn as sns
import scipy.constants as const
import warnings
matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['axes.titlesize'] = 15
matplotlib.rcParams['axes.labelsize'] = 15
matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['legend.fontsize'] = 15
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import itertools

#from analyze_env_st import *
# --- NEW (top-level): imports used by workers ---
import numpy as np
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


__version__ = "0.5.3"
"""
0.1.1 version: removed neighbor_list from complete_coord_info object to save memory
0.1.2 (future): reduce the anion_displacements object as ndarray.
0.1.3 (currently implementing): add distance matrix to PolymerEventAnalyzer (for Li and anion-N)
0.1.6 implemented subtraction of COM in the reading phase (unwrapped coordinates)
0.1.7 memory efficient onsager event analysis (not saving everything in the memory)
0.2.1 liquid finalized
0.2.2 liquid polished
0.2.3 SIC polished
0.3.1 polymer polished
0.3.2 polished more polymeranalyzer
0.4.1 for polymeranalyzer, change underlying data type from dictionaries to np.arrays for memory efficiency.
0.4.2 for liquidanalyzer, change underlying data type from dictionaries to np.arrays for memory efficiency.
also added u_methods which can take in any number of molecule types for analysis.
0.4.3 for sicanalyzer, change underlying data type from dictionaries to np.arrays for memory efficiency. untested
0.5.1 all three classes work well now.
0.5.2 parallelize liquid electrolyte state transitions. This can be extended to other systems in the future, but hasn't been done yet.
0.5.3 fix cip and agg definitions
TBD: in read_structures_parallel, deprecate "timestep". it is never used.
"""


# --- NEW (top-level): presort helpers (moved out so workers can use them) ---
def presort_time_res_val(arr):
    """
    arr: ndarray (N, 3) with columns [time, residue_id, value]
    Returns a presorted view and time-block indices so each per-time lookup
    is just a slice + grouping (no sorting inside the worker loop).
    """
    a = np.ascontiguousarray(arr)
    if not np.issubdtype(a[:, 0].dtype, np.integer):
        a = a.copy()
        a[:, 0] = a[:, 0].astype(np.int64)

    order = np.lexsort((a[:, 2], a[:, 1], a[:, 0]))  # time, residue, value
    arrs = a[order]
    times = arrs[:, 0]
    uniq_times, t_starts = np.unique(times, return_index=True)
    t_ends = np.r_[t_starts[1:], times.size]
    return arrs, uniq_times, t_starts, t_ends


def residue_dict_from_presorted(arrs, uniq_times, t_starts, t_ends, time):
    """
    Retrieve {residue_id: sorted list of values} for a given time
    using the presorted arrays (no per-call sorting).
    """
    if not np.issubdtype(uniq_times.dtype, np.integer):
        time = int(time)
    i = np.searchsorted(uniq_times, time)
    if i >= len(uniq_times) or uniq_times[i] != time:
        return {}

    block = arrs[t_starts[i]:t_ends[i]]  # already sorted by (res, value)
    if block.size == 0:
        return {}

    residues = block[:, 1].astype(int)
    vals     = block[:, 2]
    uniq_res, r_starts = np.unique(residues, return_index=True)
    r_ends = np.r_[r_starts[1:], residues.size]
    return {int(r): vals[s:e].tolist() for r, s, e in zip(uniq_res, r_starts, r_ends)}

# --- NEW (top-level): worker function run in child processes ---
def _process_one_li(payload):
    """
    payload: (Li_index, info_dict, methods, res_id_mapper, res_id_to_type, anion_coordinating_Li)
    info_dict must contain: 'flattened_O_providers', 'cartesian_positions', 'global_times'
    Returns: (Li_index, result_dict) where result_dict matches your total_state_info[Li_index]
    """
    (Li_index, info, methods, res_id_mapper, res_id_to_type, anion_coordinating_Li) = payload

    arr = info['flattened_O_providers']
    cart_positions = info['cartesian_positions']
    global_times   = info['global_times']

    # presort once per Li
    arrs, uniq_times, t_starts, t_ends = presort_time_res_val(arr)

    global_times_list = []
    cartesian_positions_list = []
    method_dict_lists = {m: [] for m in methods}

    # per time frame
    for timeindex in global_times:
        # reconstruct O_providers dict quickly
        O_providers = residue_dict_from_presorted(arrs, uniq_times, t_starts, t_ends, timeindex)

        # Build current_Li_state equivalents (only fields you actually use later)
        # Determine solvation_env (ssip/cip/agg)
        contains_anion = False
        is_agg = False
        anion_keys = []

        for res_id in O_providers.keys():
            if res_id_mapper[res_id] == "anion":
                contains_anion = True
                anion_keys.append(res_id)
        if len(list(set(anion_keys))) > 1:
            # Li binding to more than 2 TFSI anions. This is technically not a CIP, but an AGG now.
            is_agg = True
        if contains_anion:
            # check if an anion here coordinates >=2 Li at this frame → aggregate
            for anion_key in anion_keys:
                an_type = res_id_to_type[anion_key]
                an_arr = anion_coordinating_Li[an_type]  # expected shape: (M, 2) with [time, anion_res_id]
                if np.sum((an_arr[:, 0] == timeindex) & (an_arr[:, 1] == anion_key)) >= 2:
                    is_agg = True
                    break

        if not contains_anion:
            solvation_env = 'ssip'
        elif not is_agg:
            solvation_env = 'cip'
        else:
            solvation_env = 'agg'

        # Split solvent vs anion dicts
        solvent_dict = {}
        anion_dict = {}
        for res_id, vlist in O_providers.items():
            typ = res_id_mapper[res_id]
            if typ == 'solvent':
                solvent_dict[res_id] = vlist
            elif typ == 'anion':
                anion_dict[res_id] = vlist
            else:
                raise ValueError(f"Unknown res_id type: {res_id} -> {typ}")

        num_solvent = len(solvent_dict)
        num_anion = len(anion_dict)
        num_anion_atoms = sum(len(anion_dict[res_id]) for res_id in anion_dict)
        num_solvent_atoms = sum(len(solvent_dict[res_id]) for res_id in solvent_dict)

        # record times/positions
        global_times_list.append(global_times[timeindex])
        cartesian_positions_list.append(cart_positions[timeindex])

        # Build method_dict
        method_dict = {}

        if 1 in methods:
            # ssip=0, cip=1, agg=2
            method_dict[1] = {'ssip':0,'cip':1,'agg':2}[solvation_env]

        if 2 in methods:
            # num_coordinating_residues capped at 10
            s = num_anion + num_solvent
            if s >= 11:
                warnings.warn(f"method_2_state ({s}) exceeds 10, setting to 10")
                s = 10
            method_dict[2] = s

        if 3 in methods:
            s = num_solvent
            if s >= 8:
                warnings.warn(f"method_3_state ({s}) exceeds 7, setting to 7")
                s = 7
            method_dict[3] = s

        if 4 in methods:
            s = num_anion
            if s >= 4:
                warnings.warn(f"method_4_state ({s}) exceeds 3, setting to 3")
                s = 3
            method_dict[4] = s

        if 5 in methods:
            a, b = num_solvent, num_anion
            if a >= 8:
                warnings.warn(f"num_solvent ({a}) exceeds 7, setting to 7")
                a = 7
            if b >= 4:
                warnings.warn(f"num_anion ({b}) exceeds 3, setting to 3")
                b = 3
            method_dict[5] = a * 4 + b

        if 6 in methods:
            s = num_solvent_atoms
            if s >= 10:
                warnings.warn(f"method_6_state ({s}) exceeds 9, setting to 9")
                s = 9
            method_dict[6] = s

        if 7 in methods:
            s = num_anion_atoms
            if s >= 10:
                warnings.warn(f"method_7_state ({s}) exceeds 9, setting to 9")
                s = 9
            method_dict[7] = s

        if 8 in methods:
            a, b = num_solvent_atoms, num_anion_atoms
            if a >= 10:
                warnings.warn(f"num_solvent_atoms ({a}) exceeds 9, setting to 9")
                a = 9
            if b >= 10:
                warnings.warn(f"num_anion_atoms ({b}) exceeds 9, setting to 9")
                b = 9
            method_dict[8] = a * 10 + b

        # append
        for m in method_dict.keys():
            method_dict_lists[m].append(method_dict[m])

    # pack per-Li result
    result = {
        "global_times":       np.array(global_times_list, dtype=int),
        "cartesian_positions":np.array(cartesian_positions_list, dtype=float),
        "method_dicts":       method_dict_lists
    }
    print(f"Li_index {Li_index} time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return Li_index, result


def convert_D_to_sigma(volume, num_Li, z, temperature):
    """
    compute the factor to convert the diffusivity (cm^2/s) to sigma (mS/cm).
    sigma = D * factor
    volume : volume of the cell in Angstrom^3
    NUM_LI : number of Li ions in the cell
    z : charge of the ion (Li = 1)
    temperature : temperature in K
    """
    vol = volume * 1e-24  # units cm^3 (from Angstrom^3)
    return 1000 * num_Li / (vol * const.N_A) * z**2 * (const.N_A * const.e) ** 2 / (const.R * temperature)
def compressed_self_linear_fit(timestep, dt, avg_self,fitting_range=None):
    """
    perform a component-wise linear regression fit
    returns a slope matrix of the same size as the input matrix
    timestep : in ps (how frequently the data is sampled)
    dt : x-values
    avg_self : from avg_matrix -> (diagonal == 0), 1D vector. 
    fitting_range : (lower, upper) in fraction of the total length of dt.
    a typical value is (0.1, 0.2). default is fitting the entire range.
    outputs: same dimensions, but Diffusivity values.
    If the input was particle-based, then the result is particle-based diffusivity. 
    Otherwise, if it is averaged over particles, then it is Li and anion diffusivity.
    """
    self_slope  = np.zeros_like(avg_self[:,0])
    if fitting_range:
            t_lower = int(np.floor(len(dt)*fitting_range[0]))
            t_upper = int(np.floor(len(dt)*fitting_range[1]))
    else:
        t_lower = 0
        t_upper = len(dt)-1
    for i in range(avg_self.shape[0]):
        values = avg_self[i,t_lower:t_upper]
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(dt[t_lower:t_upper], values)
        self_slope[i] = slope
    self_matrix = self_slope/timestep/60 * (1/1000) 
    # from A^2/ps to cm^2/s; 1/1000 : slope per ps to slope per fs
    return self_matrix


class InorganicElectrolyteAnalyzer:
    def __init__(self, PATH, label):
        self.PATH = PATH
        self.label = label
    
    def read_structures_parallel(self, 
                                 structure_list, 
                                 analyze_anions=False,
                                 timestep=1, 
                                 stepskip=1, 
                                 method='site_definitions', 
                                 correct_com='solvent',
                                 mobile_species="Li", 
                                 coordinators=["O", "Cl", "F", "Br", "I"],
                                 site_definitions=None,
                                 cutoff=0,
                                 compute_distance_matrix=False):
        """
        structure_list : a list of pymatgen structures
        timestep : in ps 
        stepskip : number of steps to skip
        method : 'site_definitions' or 'convex'. only "site_definitions" is implemented.
        correct_com : 'solvent' or 'framework'
        site_definitions : path to the site_definitions pickle file
        """
        self.sampling_freq = timestep * stepskip
        self.structure = structure_list[0]
        self.res_ids = [i.specie.symbol for i in self.structure]
        self.min_O_index = min([index for index, i in enumerate(self.structure) if i.specie == Element("O")])
        self.mobile_species = mobile_species
        self.coordinators = [Element(i) for i in coordinators] # [Element("O"), Element("Cl"), Element("F"), Element("Br"), Element("I")]
        self.str_coordinators = coordinators # ["O", "Cl", "F", "Br", "I"]
        self.compute_distance_matrix = compute_distance_matrix
        if site_definitions is not None:
            self.site_definitions = pickle.load(open(site_definitions, "rb"))
        else:
            self.site_definitions = None
        if method == 'site_definitions':
            assert self.site_definitions is not None, "site_definitions must be provided if method is site_definitions"
        relevant_ions = []
        for index, i in enumerate(self.structure):
            if i.specie == Element(self.mobile_species):
                relevant_ions.append(index)
        print(f"number of diffusing ({self.mobile_species}) atoms: {sum(np.array(relevant_ions))}")
        lattice = structure_list[0].lattice
        atom_names = [i.specie.symbol for i in structure_list[0]]
        # Get dimensions for the fractional coordinates array
        n_frames = len(structure_list)
        n_atoms = len(structure_list[0])
        
        # Initialize array to store fractional coordinates
        xyz_rdf_frac = np.zeros((n_frames, n_atoms, 3))
        
        # Extract fractional coordinates for each frame
        for frame_idx, structure in enumerate(structure_list):
            xyz_rdf_frac[frame_idx] = structure.frac_coords
        xyz_rdf_frac = np.insert(xyz_rdf_frac, 0, xyz_rdf_frac[0], axis=0)
        dp = xyz_rdf_frac[1:,:,:] - xyz_rdf_frac[:-1,:,:]
        dp = dp - np.round(dp)
        print(f"dp.shape: {dp.shape}")
        #dp = np.einsum('ijk,kl->ijl', dp, lattice)
        dp = np.matmul(dp, lattice.matrix)
        print(f"dp.shape: {dp.shape}")
        f_disp = np.cumsum(dp, axis=0)
        #xyz_rdf = xyz_rdf[0] + f_disp
        xyz_rdf = np.matmul(xyz_rdf_frac[0], lattice.matrix) + f_disp
        xyz_rdf = xyz_rdf[::stepskip, ...]
        # coordinates are unwrapped now, and is sliced by stepskip.
        if correct_com == "framework":
            """
            geometric center of non-diffusing (Li) atoms
            not considering the mass of atoms
            TBD : generalize to other diffusing cations such as Na, K, Zn.
            """
            relevant_ions = []
            for index, i in enumerate(atom_names):
                if i == self.mobile_species:
                    relevant_ions.append(False)
                else:
                    relevant_ions.append(True)
            if len(relevant_ions) == 0:
                raise ValueError("no non-diffusing atoms found")
            print(sum(np.array(relevant_ions)))
            for frame_index, sliced_frame in enumerate(xyz_rdf):
                com = np.mean(sliced_frame[relevant_ions], axis=0)
                xyz_rdf[frame_index] = xyz_rdf[frame_index] - com
        elif correct_com == "solvent":
            """
            center of mass of non-diffusing (Li) atoms
            considering the mass of atoms
            This is what technically matches with EIS measurements
            (ref: 10.1021/jacs.2c02389)
            """
            mass_list = [Element(i).atomic_mass for i in atom_names]

            relevant_ions = []
            for index, i in enumerate(atom_names):
                if i == self.mobile_species:
                    relevant_ions.append(False)
                else:
                    relevant_ions.append(True)
            if sum(np.array(relevant_ions)) == 0:
                raise ValueError("no non-diffusing atoms found")
            coms = []
            for frame_index, sliced_frame in enumerate(xyz_rdf):
                val = sliced_frame*np.array(mass_list)[:,np.newaxis] # position * mass
                com = np.sum(val[relevant_ions], axis=0)/(np.array(mass_list)[relevant_ions].sum())
                xyz_rdf[frame_index] = xyz_rdf[frame_index] - com
                coms.append(com)
            self.coms = np.array(coms)

        def map_res_id_to_Li_or_O(structure):
            '''
            res_ids: a list of res_ids
            structure: initial structure or any example of structure
            returns a function that determines if the res_id is a polymer, TFSI, or Li
            for now, only works for TFSI (num_atom = 15), but can be improved to work for other anions by taking it as an argument
            '''
            Li_indices = []
            O_indices = []
            for index, i in enumerate(structure):
                if i.specie.symbol == self.mobile_species:
                    Li_indices.append(index)
                elif i.specie.symbol in self.str_coordinators: # ["O", "Cl"]
                    O_indices.append(index)
            
            res_id_mapper = {}
            for i in Li_indices:
                res_id_mapper[i] = self.mobile_species
            for j in O_indices:
                res_id_mapper[j] = "anion"
            return res_id_mapper

        self.res_id_mapper = map_res_id_to_Li_or_O(self.structure)
        
        def extract_coordination_environments(s, Li_index, global_time_index, distance_cutoff=4, method='site_definitions'):
            """
            s : pymatgen structure
            Li_index : index of lithium
            same as polymers, but to add additional Li neighbors within a distance cutoff, I make 
            extract_coordination_environments_Li function below.
            """
            site = s[Li_index]
            nlist = s.get_neighbors(site, 4.8)
            nlist.sort(key=lambda x:x.nn_distance)
            #print(f"analyzing Li at index: {Li_index}, frame_index: {global_time_index}")
            nlist_O = [i for i in nlist if i.specie in self.coordinators]
            H_plus_coordinators = self.coordinators + [Element("H")] # Hydrogen is ignored as a coordinator.
            nlist_non_H_O = [i for i in nlist if i.specie not in H_plus_coordinators]
            nearest_non_H_O = nlist_non_H_O[0]
            convex_nlist_O = nlist_O[:4]
            if method == 'convex':
                raise NotImplementedError("Convex hull method not implemented yet")
                for neighb_index, neighb in enumerate(nlist_O[4:]):
                    og_convex = ConvexHull([i.coords for i in convex_nlist_O])
                    new_convex = ConvexHull([i.coords for i in convex_nlist_O] + [neighb.coords])
                    new_non_H_convex = ConvexHull([i.coords for i in convex_nlist_O] + [nearest_non_H_O.coords] + [neighb.coords])
                    exist_non_H_O = False
                    for k in nlist_non_H_O:
                        if k.nn_distance < neighb.nn_distance:
                            exist_non_H_O = True
                    if (len(new_convex.vertices) == len(og_convex.vertices) + 1
                        and nearest_non_H_O.coords in new_non_H_convex.points[new_non_H_convex.vertices]
                        and len(new_convex.vertices) != len(new_non_H_convex.vertices)
                        and neighb.nn_distance < 1.3 * convex_nlist_O[0].nn_distance):
                        convex_nlist_O.append(neighb)
                    else:
                        for i in convex_nlist_O:
                            i.label = self.res_ids[i.index] # decorating pymatgen Neighbor object by labeling them by which res_id this oxygen comes from
                        each_env_info = {'cartesian_position': site.coords, 
                                            'global_time':global_time_index, 
                                            'cutoff':max([i.nn_distance for i in convex_nlist_O]), 
                                            'num_neighbor':len(convex_nlist_O), 
                                            'neighborlist':convex_nlist_O}
                        break
                    if neighb_index == len(nlist_O[4:])-1:
                        for i in convex_nlist_O:
                            i.label = self.res_ids[i.index] # decorating pymatgen Neighbor object by labeling them by which res_id this oxygen comes from
                        each_env_info = {'cartesian_position': site.coords, 
                                            'global_time':global_time_index, 
                                            'cutoff':max([i.nn_distance for i in convex_nlist_O]), 
                                            'num_neighbor':len(convex_nlist_O), 
                                            'neighborlist':convex_nlist_O}
            elif method == 'csm':
                raise NotImplementedError("CSM method not implemented yet")
                assigned_coord_env = extract_site(s, li_index=Li_index, envtype='both', center=True)
                print(assigned_coord_env)
                if assigned_coord_env['type'] == 'tet':
                    coord_env = nlist_O[:4]
                elif assigned_coord_env['type'] == 'oct':
                    coord_env = nlist_O[:6]
                else:
                    raise ValueError("Invalid coordination environment type")
                each_env_info = {'cartesian_position': site.coords, 
                                'global_time':global_time_index, 
                                'cutoff':max([i.nn_distance for i in coord_env]), 
                                'num_neighbor':len(coord_env), 
                                'neighborlist':coord_env}

                #assert each_env_info['num_neighbor'] in [4, 6]
            elif method == 'site_definitions':
                '''
                1. find the nearest oxygen to this Li.
                2. downselect the possible sites with that Li
                3. iterate through the possible sites and find the site where Li is within the convex hull. 
                find (N_withLi) - (N_withoutLi) == 0 (i.e. number of simplexes are the same with / without Li)
                4. if none of the convex hull holds the Li, then find the one with the smallest convex hull volume diff
                (V_withLi) - (V_withoutLi) minimize 
                '''
                def find_correct_site(s, 
                                      Li_index, 
                                      min_O_index,
                                      site_definitions, 
                                      nlist_O,
                                      vol_diff_threshold=10):
                    site = s[Li_index]
                    candidate_sites = [site_definitions.nodes[geom_site] 
                                       for geom_site in site_definitions.nodes 
                                       if len(site_definitions.nodes[geom_site]['neigh'].intersection(set([i.index - min_O_index for i in nlist_O[:4]]))) >= 1]
                    # compute the overlap of four nearest oxygen index of the target lithium, and candidate sites from site_definitions.
                    found = False
                    for geom in candidate_sites:
                        convex_site = [k.coords for k in nlist_O if k.index - min_O_index in geom['neigh']] # nlist_O oxygens that are part of the candidate site oxygens
                        if len(convex_site) not in [4, 6]:
                            #print("not enough neighbors, this isn't the correct site")
                            continue
                        convex_site_wLi = convex_site + [site.coords]
                        #print(convex_site_wLi)
                        if len(ConvexHull(convex_site).vertices) == len(ConvexHull(convex_site_wLi).vertices):
                            correct_node = geom
                            correct_site = [k for k in nlist_O if k.index - min_O_index in geom['neigh']]
                            assert len(correct_site) == 4 or len(correct_site) == 6
                            found = True
                            break
                    if not found:
                        vol_diff = []
                        common_O = []
                        for geom_index, geom in enumerate(candidate_sites):
                            convex_site = [k.coords for k in nlist_O if k.index - min_O_index in geom['neigh']] 
                            common_O.append(len(convex_site))
                            if len(convex_site) not in [4, 6]:
                                #print("not enough neighbors, this isn't the correct site-2")
                                vol_diff.append(1000000)
                                continue
                            else:
                                convex_site_wLi = convex_site + [site.coords]
                                vol_diff.append(ConvexHull(convex_site_wLi).volume - ConvexHull(convex_site).volume)
                        assert len(vol_diff) > 0
                        if len(vol_diff) == 0:
                            #print("NLIST_O: ",set([k.index  for k in nlist_O]))
                            #print("candidate_sites: ")
                            # len(vol_diff) cannot be zero.
                            for geom in candidate_sites:
                                print([p + min_O_index for p in geom['neigh']])
                            #print("no match here...")
                        assert len(vol_diff) == len(candidate_sites)
                        if min(vol_diff) >= 1000000:
                            print("NLIST_O: ",set([k.index  for k in nlist_O]))
                            print("common_O: ", common_O)
                            print("*** Problem detected ***")
                            raise ValueError("Site not found. The framework must have changed. Consider lowering the temperature.")
                        correct_site_index = np.argmin(vol_diff)
                        if vol_diff[correct_site_index] >= vol_diff_threshold:
                            print("*** Problem detected ***")
                            print("Volume diff: ", vol_diff)
                            print("NLIST_O: ",set([k.index  for k in nlist_O]))
                            print("assigned node: ", candidate_sites[correct_site_index])
                            print("assigned site: ", [k for k in nlist_O if k.index - min_O_index in candidate_sites[correct_site_index]['neigh']])
                            print("assigned site: ", [k.index - min_O_index for k in nlist_O if k.index - min_O_index in candidate_sites[correct_site_index]['neigh']])
                            print("common_O: ", common_O)
                            raise ValueError("Site has been assigned, but Li lies too far from any given set of sites. Printing some additional information.")
                            s.to(filename=f"problematic_structure_{Li_index}.vasp", fmt='poscar')
                        if vol_diff[correct_site_index] >= vol_diff_threshold:
                            s.to(filename=f"problematic_structure_{Li_index}.vasp", fmt='poscar')
                        assert vol_diff[correct_site_index] < vol_diff_threshold # reduce to 2 later.
                        correct_node = candidate_sites[correct_site_index]
                        correct_site = [k for k in nlist_O if k.index - min_O_index in candidate_sites[correct_site_index]['neigh']]
                        found = True
                    return correct_site, correct_node
                correct_site, correct_node = find_correct_site(s, Li_index, self.min_O_index, self.site_definitions, nlist_O) # this is a node in networkx graph
                # print(f"correct site {Li_index} : ", [k.index for k in correct_site])
                each_env_info = {'cartesian_position': site.coords, 
                                'global_time':global_time_index, 
                                'cutoff':max([i.nn_distance for i in correct_site]), 
                                'num_neighbor':len(correct_site), 
                                'neighborlist':correct_site,
                                'node':correct_node}
            # if assertion fails, then rerun the upper block for loop using distance factor of 2.0

            site = s[Li_index]
            nlist = s.get_neighbors(site, distance_cutoff) # neighboring Li within distance_cutoff
            nlist.sort(key=lambda x:x.nn_distance)
            nlist_Li = [i for i in nlist if i.specie == Element(self.mobile_species)]
            true_neighbors = []
            # print(f"analyzing neighboring {len(nlist_Li)} Li atoms at index: {Li_index}, frame_index: {global_time_index}")
            def find_nodes_within_distance(G, source_node, max_distance):
                """
                Find all nodes that are within 'max_distance' edges from source_node
                """
                # Get shortest path lengths from source to all nodes
                path_lengths = nx.single_source_shortest_path_length(G, source_node)
                
                # Filter nodes within the max distance
                nodes_within_distance = [node for node, length in path_lengths.items() 
                                    if length <= max_distance]
                
                return nodes_within_distance
            within_max_distance_nodes = find_nodes_within_distance(self.site_definitions, correct_node['node_id'], 2)
            for candidate_Li in nlist_Li:
                candidate_site = s[candidate_Li.index]
                # print("considering candidate site: ", candidate_site)
                nlist_neighb = s.get_neighbors(candidate_site, 4.8)
                nlist_neighb.sort(key=lambda x:x.nn_distance)
                nlist_neighb_O = [i for i in nlist_neighb if i.specie in self.coordinators]
                #convex_nlist_neighb_O = nlist_neighb_O[:4]
                # print(Li_index, candidate_Li.index)
                correct_neighbor_site, correct_neighbor_node = find_correct_site(s, candidate_Li.index, self.min_O_index, self.site_definitions, nlist_neighb_O)
                #print(f"correct neighboring site {candidate_Li.index}: ", [k.index for k in correct_neighbor_site])
                # print('HERE')
                #print(set([k.index for k in correct_neighbor_site]))
                # print(set([k.index for k in each_env_info['neighborlist']]))
                # if len(set([k.index for k in correct_neighbor_site]).intersection(set([k.index for k in each_env_info['neighborlist']]))) >= 2: # this is the problem here.
                if correct_neighbor_node['node_id'] in within_max_distance_nodes:
                    # face-sharing = 3 shared oxygens.
                    # print("found the correct neighbor site")
                    true_neighbors.append(candidate_Li)

            each_env_info['num_Li'] = len(true_neighbors)
            each_env_info['neighborlist'].extend(true_neighbors)
            # This Li must also be the face-sharing Li. three common oxygens.

            return each_env_info


        

        # combine these two environments. merge Li information into the above one.
        Li_indices = []
        envinfo = {}
        for site_index, site in enumerate(self.structure):
            if site.specie == Element(self.mobile_species):
                Li_indices.append(site_index)
                envinfo[site_index] = []
        if self.compute_distance_matrix:
            distance_matrix = []    
        


        per_Li_data = {}
        for site_index, site in enumerate(self.structure):
            if site.specie == Element(self.mobile_species):
                per_Li_data[site_index] = {
                    "cartesian_position": [],
                    "global_times": [],
                    "cutoffs": [],
                    "num_neighbors": [],
                    "flattened_O_providers": [],
                    "flattened_neighborset": [],
                    "nodes": [],
                    "num_Li": []
                }
        for frame_index, frame in enumerate(xyz_rdf):
            s = Structure(lattice = lattice, species = atom_names, coords = xyz_rdf[frame_index], coords_are_cartesian=True)
            if self.compute_distance_matrix:
                s_copy = s.copy()
                s_copy.remove_species([i for i in s_copy.species if i not in [Element(self.mobile_species)]])
                distance_matrix.append(s_copy.distance_matrix)
            for Li_index in Li_indices:
                each_env_info = extract_coordination_environments(s, Li_index, frame_index)
                O_providers = {}
                for jindex, j in enumerate(each_env_info['neighborlist']):
                    if j.specie.symbol in O_providers.keys():    # res_ids : Li or oxygen
                        O_providers[j.index].append(j.index)
                    else:
                        O_providers[j.index] = [j.index]
                        # basically, each O_provider is each oxygen index, and that list can only
                        # have its index in the list.
                        # i do it this way to use the same code as PEO + LiTFSI
                        # This basicaly means that each O, each Li is a different residue.
                        # in each residue, only one Li / O exist
                        # and ther residue id is the same as atom index.
                for j in O_providers.keys():
                    O_providers[j].sort()
                each_env_info['O_providers'] = O_providers
                each_env_info.pop('neighborlist')
                each_env_info['neighborset'] = set(each_env_info['O_providers'].keys())

                per_Li_data[Li_index]["cartesian_position"].append(each_env_info['cartesian_position'])
                per_Li_data[Li_index]["global_times"].append(each_env_info["global_time"])
                per_Li_data[Li_index]["cutoffs"].append(each_env_info["cutoff"])
                per_Li_data[Li_index]["num_neighbors"].append(len(each_env_info["O_providers"].keys()))
                per_Li_data[Li_index]["nodes"].append(each_env_info["node"])
                per_Li_data[Li_index]["num_Li"].append(each_env_info["num_Li"])

                for res_id, O_indices in each_env_info["O_providers"].items():
                    for O_index in O_indices:
                        per_Li_data[Li_index]["flattened_O_providers"].append([frame_index, res_id, O_index])
                for res_id in each_env_info["neighborset"]:
                    per_Li_data[Li_index]["flattened_neighborset"].append([frame_index, res_id])

        if self.compute_distance_matrix:
            self.distance_matrix = np.stack(distance_matrix)
        else:
            self.distance_matrix = None

        self.complete_coord_info = {}
        for Li_index, data in per_Li_data.items():
            self.complete_coord_info[Li_index] = {
                "cartesian_positions": np.array(data["cartesian_position"]),
                "global_times": np.array(data["global_times"]),
                "cutoffs": np.array(data["cutoffs"]),
                "num_neighbors": np.array(data["num_neighbors"]),
                "flattened_O_providers": np.array(data["flattened_O_providers"]) if data["flattened_O_providers"] else np.empty((0,3), dtype=int),
                "flattened_neighborset": np.array(data["flattened_neighborset"]) if data["flattened_neighborset"] else np.empty((0,2), dtype=int),
                "nodes": np.array(data["nodes"]), # this might not be np-arrayable. then make it just a list of nodes.
                "num_Li": np.array(data["num_Li"])
            }
        self.complete_node_info = {}
        for Li_index in self.complete_coord_info.keys():
            self.complete_node_info[Li_index] = [j['node_id'] for j in self.complete_coord_info[Li_index]["nodes"]]
        return
    
    def wrap_disp(self, disp):
        frac_disp =self.structure.lattice.get_fractional_coords(disp)
        #frac_disp = np.array(disp)/np.array(self.boxdim)
        rv = []
        for d in frac_disp:
            if d > 0.5:
                rv.append(d-1)
            elif d < -0.5:
                rv.append(d+1)
            else:
                rv.append(d)
        return self.structure.lattice.get_cartesian_coords(np.array(rv))
        #return np.array(rv)*np.array(self.boxdim)

    def get_total_entities():
        return
    
    def simplify_complete_coord_info_to_states(self, methods):
        """
        For inorganic crystalline conductors, the states are "activated local environment" and "isolated local environment".
        The activated local environment is the one with locally stuffed Li atoms nearby.
        The isolated local environment is the one with no locally neighboring Li atoms nearby.
        There are no segments in inorganic conductors, so "skip" is not used.
        When considering Li neighbors, the number of edges away default (N) is 2.
        methods: a list of integers from 1 to 3.
        
        method 1:regardless of which crystal site, check existence of neighboring Li within N edge distance.
            0: no Li within N edge distance.
            1: at least 1 Li within N edge distance.
        method 2:
            0: 6b, no neighbors within N edge distance
            1: 6b, at least 1 neighbor within N edge distance
            2: 18e, no neighbors within N edge distance
            3: 18e, at least 1 neighbor within N edge distance
            4: 36f, no neighbors within N edge distance
            5: 36f, at least 1 neighbor within N edge distance
        method 3: 
            0: 6b
            1: 18e
            2: 36f
        method 4: 
            0: octahedral; 6b or 18e
            1: tetrahedral; 36f
        method 5:
            0: octahedral, with no Li neighbors within N edge distance
            1: octahedral, with at least 1 Li neighbor within N edge distance
            2: tetrahedral, with no Li neighbors within N edge distance
            3: tetrahedral, with at least 1 Li neighbor within N edge distance
        """
        def construct_residue_dict(flattened_O_providers, time):
            unique_residues = np.unique(flattened_O_providers[(flattened_O_providers[:,0] == time)][:,1])
            rv = {}
            for i in unique_residues:
                vals = list(flattened_O_providers[(flattened_O_providers[:,0] == time) & (flattened_O_providers[:,1] == i), 2])
                vals.sort()
                rv[i] = vals
            #print(a[(a[:,0] == time)])
            return rv
        total_state_info = {}
        for Li_index in self.complete_coord_info.keys():
            #Li_state_list = []
            global_times_list = []
            cartesian_positions_list = []
            method_dict_lists = {}
            for method in methods:
                method_dict_lists[method] = []
            for timeindex in self.complete_coord_info[Li_index]["global_times"]:
                coord_env = {}
                coord_env['cartesian_position'] = self.complete_coord_info[Li_index]['cartesian_positions'][timeindex]
                coord_env['global_time'] = self.complete_coord_info[Li_index]['global_times'][timeindex]
                coord_env['O_providers'] = construct_residue_dict(self.complete_coord_info[Li_index]['flattened_O_providers'], timeindex)
                #coord_env['neighborset'] = [i.keys() for i in coord_env['O_providers']]
                coord_env['neighborset'] = list(coord_env['O_providers'].keys())
                #coord_env['neighborset'] = construct_residue_dict(self.complete_coord_info[Li_index]['flattened_neighborset'], timeindex)



                current_Li_state = {}
                current_Li_state['lithium'] = {}
                current_Li_state['neighbor_wyckoff'] = [] # just a list of wyckoff of neighbors
                current_Li_state['anion'] = {}
                current_Li_state['wyckoff'] = self.site_definitions.nodes[self.complete_node_info[Li_index][timeindex]]['wyckoff']
                # current_state : {res_id : }
                for res_id in coord_env['neighborset']:
                    if self.res_id_mapper[res_id] == self.mobile_species:
                        current_Li_state['lithium'][res_id] = coord_env['O_providers'][res_id]
                        this_wyckoff = self.site_definitions.nodes[self.complete_node_info[Li_index][timeindex]]['wyckoff']
                        assert this_wyckoff in ['6b', '18e', '36f']
                        current_Li_state['neighbor_wyckoff'].append(this_wyckoff)
                        """
                        note, self.complete_coord_info[last_lithium_index][last_time_index] is a dictionary of items as follows:
                        this is because each oxygen anion is considered a different residue.
                        'O_providers': {253: [253], 295: [295], 254: [254], 269: [269]}, (this includes oxygens and lithiums)
                        'neighborset': {253, 254, 269, 295}} (this includes oxygens and lithiums)
                        """
                    elif self.res_id_mapper[res_id] == "anion":
                        current_Li_state['anion'][res_id] = coord_env['O_providers'][res_id]
                    else:
                        raise ValueError(f"This element should not be a neighbor of Li: {res_id, self.res_id_mapper[res_id]}")
                num_li_neighbors = len(current_Li_state['lithium'].keys())
                current_Li_state['Li_index'] = Li_index
                current_Li_state['cartesian_position'] = coord_env['cartesian_position']
                current_Li_state['global_time'] = coord_env['global_time']
                global_times_list.append(current_Li_state['global_time'])
                cartesian_positions_list.append(current_Li_state['cartesian_position'])
                # total_state_info extraction done.
                # now go through possible methods and make a methods dictionary
                method_dict = {}

                if 1 in methods:
                    """
                    method 1:regardless of which crystal site, check existence of neighboring Li within N edge distance.
                    0: no Li within N edge distance.
                    1: at least 1 Li within N edge distance.
                    """
                    if num_li_neighbors == 0:
                        method_1_state = 0
                    elif num_li_neighbors > 0:
                        method_1_state = 1
                    method_dict[1] = method_1_state
                if 2 in methods:
                    """
                    method 2:
                    0: 6b, no neighbors within N edge distance
                    1: 6b, at least 1 neighbor within N edge distance
                    2: 18e, no neighbors within N edge distance
                    3: 18e, at least 1 neighbor within N edge distance
                    4: 36f, no neighbors within N edge distance
                    5: 36f, at least 1 neighbor within N edge distance
                    """
                    if current_Li_state['wyckoff'] == '6b':
                        if num_li_neighbors == 0:
                            method_2_state = 0
                        elif num_li_neighbors > 0:
                            method_2_state = 1
                    elif current_Li_state['wyckoff'] == '18e': 
                        if num_li_neighbors == 0:
                            method_2_state = 2
                        elif num_li_neighbors > 0:
                            method_2_state = 3
                    elif current_Li_state['wyckoff'] == '36f':
                        if num_li_neighbors == 0:
                            method_2_state = 4
                        elif num_li_neighbors > 0:
                            method_2_state = 5
                    method_dict[2] = method_2_state
                            
                if 3 in methods:
                    """
                    method 3: 
                    0: 6b
                    1: 18e
                    2: 36f
                    """
                    if current_Li_state['wyckoff'] == '6b':
                        method_3_state = 0
                    elif current_Li_state['wyckoff'] == '18e':
                        method_3_state = 1
                    elif current_Li_state['wyckoff'] == '36f':
                        method_3_state = 2
                    method_dict[3] = method_3_state
                if 4 in methods:
                    """
                    method 4: 
                    0: octahedral; 6b or 18e
                    1: tetrahedral; 36f
                    """
                    if current_Li_state['wyckoff'] == '6b' or current_Li_state['wyckoff'] == '18e':
                        method_4_state = 0
                    elif current_Li_state['wyckoff'] == '36f':
                        method_4_state = 1
                    method_dict[4] = method_4_state
                if 5 in methods:
                    """
                    method 5:
                    0: octahedral, with no Li neighbors within N edge distance
                    1: octahedral, with at least 1 Li neighbor within N edge distance
                    2: tetrahedral, with no Li neighbors within N edge distance
                    3: tetrahedral, with at least 1 Li neighbor within N edge distance
                    """
                    if current_Li_state['wyckoff'] == '6b' or current_Li_state['wyckoff'] == '18e':
                        if num_li_neighbors == 0:
                            method_5_state = 0
                        elif num_li_neighbors > 0:
                            method_5_state = 1
                    elif current_Li_state['wyckoff'] == '36f':
                        if num_li_neighbors == 0:
                            method_5_state = 2
                        elif num_li_neighbors > 0:
                            method_5_state = 3
                    method_dict[5] = method_5_state
                current_Li_state['method_dict'] = method_dict
                for method in method_dict_lists.keys():
                    method_dict_lists[method].append(method_dict[method])
                # Li_state_list.append(current_Li_state)
            #total_state_info[Li_index] = Li_state_list
            total_state_info[Li_index] = {"global_times":np.array(global_times_list, dtype=int),
                                          "cartesian_positions":np.array(cartesian_positions_list, dtype=float),
                                          "method_dicts":method_dict_lists}
        self.complete_states = total_state_info
        return self.complete_states

    def state_to_state_transitions(self):
        """
        take in self.complete_states.
        identify how many states there are. Then compute transitions between them. e.g. (0-0, 0-1, 0-2, ...)
        assign displacements to each transition like i do for extract_entity_events.
        """
        num_states_mapper = {
            1:2,
            2:6,
            3:3,
            4:2,
            5:4
        }
        state_to_state_recorder_per_Li = {}
        """
        data structure:
        key : Li_index, value : dictionary of keys: method_key, values: dictionary of key : transition_code, values: list of dictionarys of {'global_time': X, "disp": disp}

        """
        for Li_index in self.complete_states.keys():
            transition_dict = {}
            for method_key in self.complete_states[Li_index]['method_dicts'].keys():
                transition_recorder = {}
                for i in range(num_states_mapper[method_key]):
                    for j in range(num_states_mapper[method_key]):
                        transition_recorder[f"{i}-{j}"] = {"global_time":[], "disp":[]}
                assert(len(transition_recorder.keys()) == num_states_mapper[method_key]**2)
                for index, i in enumerate(self.complete_states[Li_index]['global_times']):
                    if index == len(self.complete_states[Li_index]['global_times']) - 1:
                        # end of the trajectory
                        continue
                    #displacement = self.wrap_disp(self.complete_states[Li_index][index+1]['cartesian_position'] - self.complete_states[Li_index][index]['cartesian_position'])
                    displacement = self.complete_states[Li_index]['cartesian_positions'][index + 1] - self.complete_states[Li_index]['cartesian_positions'][index]
                    # this should not be wrapped!
                    assert (len(displacement) == 3)
                    transition_code = f"{self.complete_states[Li_index]['method_dicts'][method_key][index]}-{self.complete_states[Li_index]['method_dicts'][method_key][index+1]}"
                    transition_recorder[transition_code]["global_time"].append(index)
                    transition_recorder[transition_code]["disp"].append(displacement)
                for transition_code in transition_recorder.keys():
                    transition_recorder[transition_code]["global_time"] = np.array(transition_recorder[transition_code]["global_time"], dtype=int)
                    transition_recorder[transition_code]["disp"] = np.array(transition_recorder[transition_code]["disp"], dtype=float)
                transition_dict[method_key] = transition_recorder
                #print(len(self.complete_states[Li_index]['global_times']))
                #print(sum([len(transition_recorder[i]["global_time"]) for i in transition_recorder.keys()]))
                for i in transition_recorder.keys():
                    print(i, len(transition_recorder[i]["global_time"]))
                assert(len(self.complete_states[Li_index]['global_times']) - 1 == sum([len(transition_recorder[i]["global_time"]) for i in transition_recorder.keys()]))

            state_to_state_recorder_per_Li[Li_index] = transition_dict
        self.state_transitions = state_to_state_recorder_per_Li
        return state_to_state_recorder_per_Li

    def extract_entity_events(self):
        # first number: polymer, second number: TFSI
        # 0 : no change, 1 : added, 2: removed, 3 : add and remove
        event_recorder_per_Li = {}
        for Li_index in self.complete_coord_info.keys():
            event_recorder = {rc: {"global_time":[], "disp":[]} for rc in [
                '0-0-0', '0-0-1', '0-1-0', '0-1-1', '0-2-0', '0-2-1', '0-3-0', '0-3-1',
                '1-0-0', '1-0-1', '1-1-0', '1-1-1', '1-2-0', '1-2-1', '1-3-0', '1-3-1',
                '2-0-0', '2-0-1', '2-1-0', '2-1-1', '2-2-0', '2-2-1', '2-3-0', '2-3-1',
                '3-0-0', '3-0-1', '3-1-0', '3-1-1', '3-2-0', '3-2-1', '3-3-0', '3-3-1'
            ]}
            cartesian_positions = self.complete_coord_info[Li_index]['cartesian_positions']
            flattened_O_providers = self.complete_coord_info[Li_index]['flattened_O_providers']
            n_frames = cartesian_positions.shape[0]
            for t in range(n_frames - 1):
                # Iterating through snapshots
                #if index == n_frames - 1:
                #    # end of the trajectory
                #    continue
                # Wrap disp is not necessary because I've already unwrapped the xyz coordinates when reading the structures.
                # displacement = self.wrap_disp(self.complete_coord_info[Li_index][index+1]['cartesian_position'] - self.complete_coord_info[Li_index][index]['cartesian_position'])
                neighborset_t = set(flattened_O_providers[flattened_O_providers[:,0] == t, 1])
                neighborset_tp1 = set(flattened_O_providers[flattened_O_providers[:,0] == t+1, 1])
                displacement = cartesian_positions[t+1] - cartesian_positions[t]
                assert (len(displacement) == 3)
                removed = neighborset_t - neighborset_tp1
                added = neighborset_tp1 - neighborset_t

                Li_added = False
                Li_removed = False
                Li_nochange = True
                O_added = False
                O_removed = False
                O_nochange = True
                contains_Li = self.complete_coord_info[Li_index][t]['num_Li']
                if contains_Li > 0:
                    Li_id = 1
                else:
                    Li_id = 0
                for j in removed:
                    if self.res_id_mapper[j] == self.mobile_species:
                        Li_removed = True
                        Li_nochange = False
                    if self.res_id_mapper[j] == 'anion':
                        O_removed = True
                        O_nochange = False
                for j in added:
                    if self.res_id_mapper[j] == self.mobile_species:
                        Li_added = True
                        Li_nochange = False
                    if self.res_id_mapper[j] == 'anion':
                        O_added = True
                        O_nochange = False

                # determine the polymer index (pid)
                if Li_added and Li_removed:
                    pid = 3
                if Li_added and not Li_removed:
                    pid = 1
                if not Li_added and Li_removed:
                    pid = 2
                if not Li_added and not Li_removed:
                    pid = 0
                if O_added and O_removed:
                    tid = 3
                if O_added and not O_removed:
                    tid = 1
                if not O_added and O_removed:
                    tid = 2
                if not O_added and not O_removed:
                    tid = 0
                returncode = f"{pid}-{tid}-{Li_id}"
                event_recorder[returncode]["global_time"].append(t)
                event_recorder[returncode]["disp"].append(displacement)
            for rc in event_recorder:
                event_recorder[rc]["global_time"] = np.array(event_recorder[rc]["global_time"], dtype=int)
                event_recorder[rc]["disp"] = np.array(event_recorder[rc]["disp"], dtype=float)
            event_recorder_per_Li[Li_index] = event_recorder
        self.entitychange = event_recorder_per_Li
        return event_recorder_per_Li


    def extract_events_with_graph(self, mode):
        """
        mode : "init", "final", "both". When determining the neighbor of a lithium, 
        you can compare the neighbors at init, final, or both, 
        mode == "init" : determine neighbors by only comparing the neighbors at init
        mode == "final" : determine neighbors by only comparing the neighbors at final
        mode == "both" : determine neighbors by comparing the neighbors at init and final
        (both : "OR" operation)
        default; both.
        """
        # first number: polymer, second number: TFSI
        # 0 : no change, 1 : added, 2: removed, 3 : add and remove
        event_recorder_per_Li = {}
        for Li_index in self.complete_coord_info.keys():
            event_recorder = {rc: {"global_time":[], "disp":[], "max_alignment":[]} for rc in [
                '0-0', '0-1', '1-0', '1-1'
            ]}
            """
            "x-y" : x and y can be 0 or 1 (boolean)
            0 means no hop, 1 means hop.
            x is the lithium of interest, y is the neighbor of the lithium of interest.
            """
            cartesian_positions = self.complete_coord_info[Li_index]['cartesian_positions']
            flattened_O_providers = self.complete_coord_info[Li_index]['flattened_O_providers']
            n_frames = cartesian_positions.shape[0]
            for t in range(n_frames - 1):
                # Iterating through snapshots
                #if index == n_frames - 1:
                #    # end of the trajectory
                #    continue
                # Wrap disp is not necessary because I've already unwrapped the xyz coordinates when reading the structures.
                # displacement = self.wrap_disp(self.complete_coord_info[Li_index][index+1]['cartesian_position'] - self.complete_coord_info[Li_index][index]['cartesian_position'])
                displacement = cartesian_positions[t+1] - cartesian_positions[t]
                assert (len(displacement) == 3)
                neighborset_t = set(flattened_O_providers[flattened_O_providers[:,0] == t, 1])
                neighborset_tp1 = set(flattened_O_providers[flattened_O_providers[:,0] == t+1, 1])
                self_hop = self.complete_node_info[Li_index][t + 1] != self.complete_node_info[Li_index][t]
                neighbor_hop = False
                # init_env
                if mode == "init":
                    compare_init = True
                    compare_final = False
                elif mode == "final":
                    compare_init = False
                    compare_final = True
                elif mode == "both":
                    compare_init = True
                    compare_final = True
                self_disp = self.complete_coord_info[Li_index]['cartesian_positions'][t+1] - self.complete_coord_info[Li_index]['cartesian_positions'][t]
                alignments = []
                if compare_init:
                    for neighbor in neighborset_t:
                        if self.structure[neighbor].specie != Element(self.mobile_species):
                            continue
                        if self.complete_node_info[neighbor][t + 1] != self.complete_node_info[neighbor][t]:
                            neighbor_hop = True
                            neighbor_disp = self.complete_coord_info[neighbor]['cartesian_positions'][t + 1] - self.complete_coord_info[neighbor]['cartesian_positions'][t]
                            alignments.append(np.dot(self_disp, neighbor_disp)/np.linalg.norm(self_disp)/np.linalg.norm(neighbor_disp))
                if compare_final:
                    for neighbor in neighborset_tp1:
                        if self.structure[neighbor].specie != Element(self.mobile_species):
                            continue
                        if self.complete_node_info[neighbor][t + 1] != self.complete_node_info[neighbor][t]:
                            neighbor_hop = True
                            neighbor_disp = self.complete_coord_info[neighbor]['cartesian_positions'][t + 1] - self.complete_coord_info[neighbor]['cartesian_positions'][t]
                            alignments.append(np.dot(self_disp, neighbor_disp)/np.linalg.norm(self_disp)/np.linalg.norm(neighbor_disp))
                if len(alignments) > 0:
                    max_alignment = max(alignments)
                else:
                    max_alignment = 0
                num_init_neighbor = len(neighborset_t)
                num_final_neighbor = len(neighborset_tp1)

                #returncode = f"{int(self_hop)}-{int(neighbor_hop)}-{int(num_init_neighbor)}-{int(num_final_neighbor)}"
                returncode = f"{int(self_hop)}-{int(neighbor_hop)}"
                event_recorder[returncode]["global_time"].append(t)
                event_recorder[returncode]["disp"].append(displacement)
                event_recorder[returncode]["max_alignment"].append(max_alignment)
            for rc in event_recorder:
                event_recorder[rc]["global_time"] = np.array(event_recorder[rc]["global_time"], dtype=int)
                event_recorder[rc]["disp"] = np.array(event_recorder[rc]["disp"], dtype=float)
                event_recorder[rc]["max_alignment"] = np.array(event_recorder[rc]["max_alignment"], dtype=float)
            event_recorder_per_Li[Li_index] = event_recorder
        self.entitychange = event_recorder_per_Li
        return event_recorder_per_Li                



class LiquidElectrolyteAnalyzer:
    def __init__(self, PATH, label):
        self.PATH = PATH
        self.label = label

    def read_structures_parallel(self, 
                                 analyze_anions=True, 
                                 PART_INDEX=0, 
                                 N_PART=1, 
                                 timestep=1, 
                                 stepskip=1, 
                                 correct_com='solvent', 
                                 coordinators=["O", "F", "N"],
                                 anion="PF6",
                                 mode="convex",
                                 cutoff=0,
                                 compute_distance_matrix=False):
        '''
        FOLDER_PATH : path of the directory that contains three files:
        atom_names_rdf.txt : file that contains the atom names in the system
        xyz_unwrapped_rdf_all.txt : file that contains the unwrapped xyz coordinates of the system
        box_len.txt : file that contains the box length of the system
        timestep : time step of the simulation (in ps). This is not used in the analysis.
        Everything is based on frames. Later on, in the post processing, the timescale can be 
        adjusted using this information. Just for recording.
        stepskip : how many frames to skip
        correct_com : which method of center of mass correction to use. either "solvent", "baro", or "salt_position"
        "solvent" centered is the default choice.
        coordinators : a list of str elements that can coordinate the cation, either from anion or solvents.
        anion : a str among "PF6", "FSI", "TFSI"
        mode : method to determine solvation environment, either convex or hard distance cutoff
        cutoff : distance cutoff for hard distance cutoff method (not used if mode == "convex")
        '''
        # FOLDER_PATH = "../featurize/actual_25_correct_frame"
        #FOLDER_PATH = "/Volumes/PROGRADE/353K_simulations_arash/2.25"
        # FOLDER_PATH = "/home/pleon/projects/collaborations/tri/vacancy_count/tfsi/0.75"

        # xyz_wrapped_rdf_beginning = first 100 frames, middle = middle 100 frames, end = final 100 frames
        self.mode = mode
        self.cutoff = cutoff
        self.sampling_freq = timestep * stepskip
        self.coordinators = [Element(i) for i in coordinators]
        self.anion = anion
        self.compute_distance_matrix = compute_distance_matrix
        xyz_rdf, atom_names, atom_names_long_rdf, res_ids = read_xyz(
            self.PATH, "atom_names_rdf_all.txt", "xyz_unwrapped_rdf_all.txt",
            include_resid=True
        )

        box_len = float(open(f"{self.PATH}/box_len.txt", "r").read())
        self.res_ids = res_ids
        xyz_rdf_frac = xyz_rdf/box_len # change to frac_coords
        xyz_rdf_frac = np.insert(xyz_rdf_frac, 0, xyz_rdf_frac[0], axis=0) 
        dp = xyz_rdf_frac[1:, ] - xyz_rdf_frac[:-1, ] # getting frac_disps
        dp = dp - np.round(dp) # unwrapping frac_disps
        dp = dp * box_len # changing back to real coords
        f_disp = np.cumsum(dp, axis=0) 
        xyz_rdf = xyz_rdf[0] + f_disp #* cell_dim[0][0]
        print('atom_names', atom_names, len(atom_names))
        print("atom_names_long_rdf", atom_names_long_rdf, len(atom_names_long_rdf))
        print('res_ids', res_ids, len(res_ids))
        self.atom_names_long_rdf = atom_names_long_rdf
        self.atom_names = atom_names
        self.types_rdf = [i.split('-')[1] for i in atom_names_long_rdf]
        unique_type_list = np.unique(self.types_rdf)
        unique_types = {"solvent": [i for i in unique_type_list if i.startswith("PL") or i.startswith("P")], "anion": [i for i in unique_type_list if i.startswith("AN")], "cation": [i for i in unique_type_list if i.startswith("CA")]}
        unique_types = {"solvent": list(set(unique_types["solvent"])), "anion": list(set(unique_types["anion"])), "cation": list(set(unique_types["cation"]))}
        self.unique_types = unique_types
        
        # for multi solvent cases
        def map_res_id_to_type(res_ids, types_rdf):
            res_id_to_type = {}
            for res_id, type_ in zip(res_ids, types_rdf):
                res_id_to_type[res_id] = type_
            return res_id_to_type


        self.res_id_to_type = map_res_id_to_type(self.res_ids, self.types_rdf)






        if len(unique_types["cation"]) > 1:
            raise NotImplementedError("Multi-cation simulations not implemented")

        if len(unique_types["anion"]) > 1:
            raise NotImplementedError("Multi-anion simulations not implemented")
        if anion not in ["PF6", "FSI", "TFSI"]:
            raise ValueError(f"Anion {anion} not implemented")

        for i in atom_names_long_rdf:
            if i.endswith("CA1"):
                self.mobile_cation = i.split("-")[0] # This should be things like "Li"
                break
        # for a given AN1 and fixed resid, what's the composition of that molecule? 
        # if that molecule is PF6, then anion_tracker = "AN-P"
        # if that molecule is FSI or TFSI, then anion_tracker = "N"
        anion_tracking_dict = {"PF6":"P", "FSI":"N", "TFSI":"N"}
        atom_name_long_anion = f"{anion_tracking_dict[anion]}-{list(unique_types['anion'])[0]}" 
        atom_name_long_cation = f"{self.mobile_cation}-{list(unique_types['cation'])[0]}"
        # this is atom_names_long_rdf value.
        # anion tracking atom must not be a specific element, but something from
        # atom_names_long_rdf because that element can also be in the solvent.

        xyz_rdf = xyz_rdf[::stepskip, ...] # Slicing by every "stepskip" frames 
        """
        After this, all of the frames are skipped.
        This is the only part where "stepskip" is used.
        """

        if correct_com == "salt_position":
            raise NotImplementedError("salt_position COM correction not yet developed")
            relevant_ions = []
            for index, i in enumerate(atom_names):
                if i == self.mobile_cation or atom_names_long_rdf[index] == atom_name_long_anion:
                    relevant_ions.append(True)
                else:
                    relevant_ions.append(False)
            print(sum(np.array(relevant_ions)))
            for frame_index, sliced_frame in enumerate(xyz_rdf):
                print(xyz_rdf.shape, sliced_frame.shape)
                com = np.mean(sliced_frame[relevant_ions], axis=0)
                xyz_rdf[frame_index] = xyz_rdf[frame_index] - com

        elif correct_com == "baro":
            mass_list = [Element(i).atomic_mass for i in atom_names]
            #print(sum(np.array(relevant_ions)))
            for frame_index, sliced_frame in enumerate(xyz_rdf):
                print(xyz_rdf.shape, sliced_frame.shape)
                com = np.sum(sliced_frame*np.array(mass_list)[:,np.newaxis], axis=0)/np.sum(np.array(mass_list)) # fixed from np.mean -> np.sum
                #com = np.mean(global_time_frame)
                xyz_rdf[frame_index] = xyz_rdf[frame_index] - com   
        elif correct_com == "solvent":
            mass_list = [Element(i).atomic_mass for i in atom_names]
            # determine solvent
            """
            res_ids_dict = {}
            for i in np.unique(np.array(res_ids)):
                res_ids_dict[i] = 0
            for i in np.array(res_ids):
                res_ids_dict[i] += 1
            solvent_res_id = [i for i in res_ids_dict.keys() if res_ids_dict[i] > 15]
            # TBD: check this... need to be generalized to other anions... 15 is for TFSI.
            relevant_ions = []
            for index, i in enumerate(res_ids):
                if i in solvent_res_id:
                    relevant_ions.append(True)
                else:
                    relevant_ions.append(False)
            """
            relevant_ions = []
            for index, i in enumerate(self.types_rdf):
                if i in unique_types["solvent"]:
                    relevant_ions.append(True)
                else:
                    relevant_ions.append(False)
            print(sum(np.array(relevant_ions)), len(relevant_ions))
            
            for frame_index, sliced_frame in enumerate(xyz_rdf):
                print(xyz_rdf.shape, sliced_frame.shape)
                val = sliced_frame*np.array(mass_list)[:,np.newaxis]
                com = np.sum(val[relevant_ions], axis=0)/(np.array(mass_list)[relevant_ions].sum())
                xyz_rdf[frame_index] = xyz_rdf[frame_index] - com
                
        atom_names_long_rdf = np.array(atom_names_long_rdf)

        EACH_PART_COUNT = int(len(xyz_rdf)/N_PART)+1
        START_INDEX = EACH_PART_COUNT * PART_INDEX
        if PART_INDEX == N_PART-1:
            END_INDEX = len(xyz_rdf)
        else:
            END_INDEX = EACH_PART_COUNT * (PART_INDEX+1)
        print(f"START_INDEX: {START_INDEX}, END_INDEX: {END_INDEX}, EACH_PART_COUNT: {EACH_PART_COUNT}")

        def get_structure_list(atom_names_rdf, xyz_rdf, box_dim):
            '''
            atom_names_rdf: list of atom names
            xyz_rdf: position of atoms
            box_dim: integer number for a cubic sized box in Angstrom
            returns a list of pymatgen structure objects.
            deprecatd because it is way too memory intesive.
            Note: the index of atoms are not modified.
            '''
            if type(box_dim)==float:
                # cubic box
                lattice = Lattice.cubic(box_dim)
            else:
                # non-cubic box
                lattice = Lattice(box_dim)

            structure_list = []
            for i in range(len(xyz_rdf)):
                structure_list.append(Structure(lattice=lattice , species=atom_names_rdf, coords=xyz_rdf[i], coords_are_cartesian=True))
            return structure_list
        def get_initial_final_structure(atom_names_rdf, xyz_rdf, box_dim):
            '''
            atom_names_rdf: list of atom names
            xyz_rdf: position of atoms
            box_dim: integer number for a cubic sized box in Angstrom
            returns pymatgen structure object of the first snapshot.

            Note: the index of atoms are not modified.
            '''
            if type(box_dim)==float:
                # cubic box
                lattice = Lattice.cubic(box_dim)
            else:
                # non-cubic box
                lattice = Lattice(box_dim)
            structure_init  = Structure(lattice= lattice , species=atom_names_rdf, coords=xyz_rdf[0], coords_are_cartesian=True)
            structure_final = Structure(lattice= lattice , species=atom_names_rdf, coords=xyz_rdf[-1], coords_are_cartesian=True)
            return structure_init, structure_final
        def map_res_id_to_solvent_or_anion(res_ids):
            '''
            res_ids: a list of res_ids
            returns a function that determines if the res_id is a solvent, anion or mobile cation (Li )
            '''
            #mapper = {}
            #for i in res_ids:
            #    if i in mapper.keys():
            #         mapper[i] +=1
            #    else:
            #        mapper[i] = 1
            res_id_mapper = {}
            #print(mapper)
            for index, i in enumerate(self.types_rdf):
                if i in unique_types["solvent"]:
                    res_id_mapper[res_ids[index]] = "solvent"
                elif i in unique_types["anion"]:
                    res_id_mapper[res_ids[index]] = "anion"
                elif i in unique_types["cation"]:
                    res_id_mapper[res_ids[index]] = "cation"

            #for i in mapper.keys():
            #    if mapper[i] == 10: # polymer, not TFSI
            #        res_id_mapper[i] = "solvent" # This is the solvent basically. 
            #    elif mapper[i] == 7: # TFSI (basically PF6)
            #        res_id_mapper[i] = "anion"
            #    elif mapper[i] == 1: # Li
            #        res_id_mapper[i] = "cation"
            #    else:
            #        print('unknown species in the structure', i, mapper[i]
            #              )
            #        raise ValueError(f"Unknown species in structure with res_id {i} and count {mapper[i]}. Expected counts: 1 (Li), 7 (TFSI), or 10 (polymer).")
                    
            return res_id_mapper

        def extract_coordination_environments_hard(s, Li_index, global_time_index, cutoff):
            """
            s : pymatgen structure
            Li_index : index of lithium
            cutoff : distance cutoff for hard distance cutoff method
            """
            site = s[Li_index]
            nlist = s.get_neighbors(site, cutoff)
            nlist.sort(key=lambda x:x.nn_distance)
            nlist_O = [i for i in nlist if i.specie in self.coordinators]
            each_env_info = {'cartesian_position': site.coords, 
                            'global_time':global_time_index, 
                            'cutoff':max([i.nn_distance for i in nlist_O]), 
                            'num_neighbor':len(nlist_O), 
                            'neighborlist':nlist_O}
            return each_env_info

        def extract_coordination_environments(s, Li_index, global_time_index):
            """
            s : pymatgen structure
            Li_index : index of lithium
            """
            site = s[Li_index]
            nlist = s.get_neighbors(site, 6)
            nlist.sort(key=lambda x:x.nn_distance)
            nlist_O = [i for i in nlist if i.specie in self.coordinators]
            H_plus_coordinators = self.coordinators + [Element("H")]
            nlist_non_H_O = [i for i in nlist if i.specie not in H_plus_coordinators]   
            # nlist_O = [i for i in nlist if i.specie in [Element("O"), Element("F")]] # check_this
            # nlist_non_H_O = [i for i in nlist if i.specie not in [Element("H"), Element("O"), Element("F")]]
            nearest_non_H_O = nlist_non_H_O[0]
            convex_nlist_O = nlist_O[:4]
            #print(len(nlist_O), convex_nlist_O)
            for neighb_index, neighb in enumerate(nlist_O[4:]):
                #print(neighb_index)
                og_convex = ConvexHull([i.coords for i in convex_nlist_O])
                new_convex = ConvexHull([i.coords for i in convex_nlist_O] + [neighb.coords])
                new_non_H_convex = ConvexHull([i.coords for i in convex_nlist_O] + [nearest_non_H_O.coords] + [neighb.coords])
                exist_non_H_O = False
                for k in nlist_non_H_O:
                    if k.nn_distance < neighb.nn_distance:
                        exist_non_H_O = True
                if (len(new_convex.vertices) == len(og_convex.vertices) + 1
                    and nearest_non_H_O.coords in new_non_H_convex.points[new_non_H_convex.vertices]
                    and len(new_convex.vertices) != len(new_non_H_convex.vertices)
                    and neighb.nn_distance < 1.75 * convex_nlist_O[0].nn_distance):
                    convex_nlist_O.append(neighb)
                else:
                    for i in convex_nlist_O:
                        i.label = self.res_ids[i.index] # decorating pymatgen Neighbor object by labeling them by which res_id this oxygen comes from
                    each_env_info = {'cartesian_position': site.coords, 
                                        'global_time':global_time_index, 
                                        'cutoff':max([i.nn_distance for i in convex_nlist_O]), 
                                        'num_neighbor':len(convex_nlist_O), 
                                        'neighborlist':convex_nlist_O}
                    break
                if neighb_index == len(nlist_O[4:])-1:
                    for i in convex_nlist_O:
                        i.label = self.res_ids[i.index] # decorating pymatgen Neighbor object by labeling them by which res_id this oxygen comes from
                    each_env_info = {'cartesian_position': site.coords, 
                                        'global_time':global_time_index, 
                                        'cutoff':max([i.nn_distance for i in convex_nlist_O]), 
                                        'num_neighbor':len(convex_nlist_O), 
                                        'neighborlist':convex_nlist_O}
            O_providers = {}
            for jindex, j in enumerate(each_env_info['neighborlist']):
                if self.res_ids[j.index] in O_providers.keys(): # self.res_ids takes in an atom index, and gives the res_id of that atom
                    # important note:
                    # here, I am appending the sequential index of the anion, not the true index within
                    # the supercell
                    O_providers[self.res_ids[j.index]].append(j.index)
                    #O_providers[self.res_ids[j.index]].append(self.anion_indices_forward[j.index]) # this problem
                else:
                    O_providers[self.res_ids[j.index]] = [j.index] # this uses the exact atom index.
                    #O_providers[self.res_ids[j.index]] = [self.anion_indices_forward[j.index]] # this problem. makes the anion indices sequential
            for j in O_providers.keys():
                O_providers[j].sort()
            each_env_info['O_providers'] = O_providers
            each_env_info.pop('neighborlist') # this is memory-intensive.
            each_env_info['neighborset'] = set(each_env_info['O_providers'].keys())
            return each_env_info


        def create_sequential_anion_mapping(self):
            """
            take in self.structure, identify the anion indices, and return a mapping of anion indices to sequential integer.
            This is necessary to identify sequential anion segments in a PEO chain.
            """
            anion_indices_forward = {}
            anion_indices_back = {}
            anion_indices = []
            for site_index, site in enumerate(self.structure):
                if site.specie in self.coordinators:
                    anion_indices.append(site_index)
            for index, site_index in enumerate(anion_indices):
                anion_indices_forward[site_index] = index
                anion_indices_back[index] = site_index
            assert(len(anion_indices) > 0)
            return anion_indices_forward, anion_indices_back

        #slist = get_structure_list(atom_names, xyz_rdf, box_len)
        #self.structurelist = slist
        self.res_ids = res_ids
        self.res_id_mapper = map_res_id_to_solvent_or_anion(self.res_ids)
        #print(f"Read {len(slist)} structures from {self.PATH}")
        #self.structure = self.structurelist[0]
        self.structure_init, self.structure_final = get_initial_final_structure(atom_names, xyz_rdf, box_len)
        self.structure_init.to(filename= "liquid_electrolyte_test_init.vasp", fmt="poscar")
        self.structure_final.to(filename= "liquid_electrolyte_test_final.vasp", fmt="poscar")
        self.structure = self.structure_init
        self.boxdim = self.structure.lattice.abc
        self.anion_indices_forward, self.anion_indices_back = create_sequential_anion_mapping(self)

        if analyze_anions:
            anion_center_indices = []
            for index, i in enumerate(self.atom_names_long_rdf):
                if i == atom_name_long_anion:
                    anion_center_indices.append(index)
            per_anion_data = {}
            for anion_index in anion_center_indices:
                per_anion_data[self.res_ids[anion_index]] = {"global_times":[], "disp":[]}
            #for site_index, site in enumerate(self.structure):
            #    if site.specie == Element('P'): # To be generalized to other anions, take anion species as input
            #        # this might be a part of solvent, so need to generalize.
            #        anion_center_indices.append(site_index)
            #         nitrogen_site = site
            #anion_displacements = {}
            #for anion_index in anion_center_indices:
            #    anion_displacements[self.res_ids[anion_index]] = []
            
        Li_indices = []
        #envinfo = {}

        # new data types
        per_Li_data = {}
        for site_index, site in enumerate(self.structure):
            if site.specie == Element(self.mobile_cation):
                Li_indices.append(site_index)
                per_Li_data[site_index] = {
                    "cartesian_positions": [],
                    "global_times": [],
                    "cutoffs": [],
                    "num_neighbors": [],
                    "flattened_O_providers": [],
                    "flattened_neighborset": [],
                }



        anion_coordinating_Li = {} # anion_dictionary make a anion index with key : anion-index, value : list of indices of lithium; 
        for i in self.unique_types['anion']:
            anion_coordinating_Li[i] = []
        for i in self.unique_types['solvent']:
            anion_coordinating_Li[i] = []
        # data structure: frames X anion_indices X list of Li indices

        if type(box_len)==float:
            # cubic box
            lattice = Lattice.cubic(box_len)
        else:
            # non-cubic box
            lattice = Lattice(box_len)
        s = self.structure
        if self.compute_distance_matrix:        
            distance_matrix = [] # this tracks the distance matrix of the entire trajectory.    
        for frame_index, frame in enumerate(xyz_rdf):
            # complete_coord_info: START_INDEX ~ END_INDEX-1 (END_INDEX is covered in the next round, TOTAL_INDEX is length of the trajectory)
            # anion_displacement: (1-0), (2-1), ... (END_INDEX-(END_INDEX-1))
            if frame_index < START_INDEX: # not current part
                continue
            if frame_index == END_INDEX: # not current part
                s = Structure(lattice= lattice, species=atom_names, coords=xyz_rdf[frame_index], coords_are_cartesian=True)
                #s_copy = s.copy()
                #s_copy.remove_species([i for i in s_copy.species if i not in [Element("Li"), Element("N")]])
                #distance_matrix.append(s_copy.distance_matrix)
                for anion_index in anion_center_indices:
                    disp = s[anion_index].coords - s_prev[anion_index].coords
                    each_anion_disp = {"global_time":(frame_index-1), "disp":disp}
                    per_anion_data[self.res_ids[anion_index]]['global_times'].append(each_anion_disp['global_time'])
                    per_anion_data[self.res_ids[anion_index]]['disp'].append(each_anion_disp['disp'])
                # everything done here.
                break
            s = Structure(lattice= lattice , species=atom_names, coords=xyz_rdf[frame_index], coords_are_cartesian=True)
            
            mobile_indices = [index for index, i in enumerate(atom_names_long_rdf) if i == atom_name_long_anion or i == atom_name_long_cation]
            # compute distance matrix for lithium and anion centers
            s_copy = s.copy()
            # index of mobile cation or anion tracking atom.
            s_copy.remove_sites([i for i in range(len(s_copy)) if i not in mobile_indices]) 
            if self.compute_distance_matrix:
                distance_matrix.append(s_copy.distance_matrix)
            # compute coordination environments for lithium
            for Li_index in Li_indices:
                if mode == "convex":
                    each_env_info = extract_coordination_environments(s, Li_index, frame_index)
                elif mode == "hard":
                    each_env_info = extract_coordination_environments_hard(s, Li_index, frame_index, self.cutoff)
                
                # ... get each_env_info as before ...
                per_Li_data[Li_index]["cartesian_positions"].append(each_env_info['cartesian_position'])
                per_Li_data[Li_index]["global_times"].append(each_env_info['global_time'])
                per_Li_data[Li_index]["cutoffs"].append(each_env_info['cutoff'])
                per_Li_data[Li_index]["num_neighbors"].append(each_env_info['num_neighbor'])

                for res_id, O_indices in each_env_info['O_providers'].items():
                    for O_index in O_indices:
                        per_Li_data[Li_index]["flattened_O_providers"].append([frame_index, res_id, O_index])
                for res_id in each_env_info['neighborset']:
                    per_Li_data[Li_index]["flattened_neighborset"].append([frame_index, res_id])

                # HERE CHECK
                """
                O_providers = {}
                for jindex, j in enumerate(each_env_info['neighborlist']):
                    if self.res_ids[j.index] in O_providers.keys():
                        O_providers[self.res_ids[j.index]].append(j.index)
                    else:
                        O_providers[self.res_ids[j.index]] = [j.index]
                for j in O_providers.keys():
                    O_providers[j].sort()
                each_env_info['O_providers'] = O_providers
                each_env_info.pop('neighborlist') # this is memory-intensive.
                each_env_info['neighborset'] = set(each_env_info['O_providers'].keys())
                envinfo[Li_index].append(each_env_info)
                """
                # frame_index : anion_index : list of Li_index
                for i in each_env_info['neighborset']:
                    anion_coordinating_Li[self.res_id_to_type[i]].append([frame_index, i, Li_index])
                    #if self.res_id_mapper[i] == 'anion':
                    #    anion_coordinating_Li.append([frame_index, i, Li_index])
                    #if i not in anion_dictionary.keys():
                    #    anion_dictionary[i] = [Li_index]
                    #else:
                    #    anion_dictionary[i].append(Li_index)
            #anion_dictionary_list.append(anion_dictionary)
            """
            self.anion_dictionary_list : a list of dictionaries, where each item is a dictionary of 
            anion_residue_id as key and value as a list of Li_index integer values
            this is used to determine whether an anion is bound to more than one lithium,
            which would indicate that those lithiums are considered "agg"
            """
            if frame_index == START_INDEX:
                s_prev = s
                continue
            else:
                for anion_index in anion_center_indices:
                    #disp = self.wrap_disp(s[anion_index].coords - s_prev[anion_index].coords)
                    disp = s[anion_index].coords - s_prev[anion_index].coords
                    each_anion_disp = {"global_time":(frame_index-1), "disp":disp}
                    per_anion_data[self.res_ids[anion_index]]['global_times'].append(each_anion_disp['global_time'])
                    per_anion_data[self.res_ids[anion_index]]['disp'].append(each_anion_disp['disp'])
                s_prev = s
        if self.compute_distance_matrix:
            self.distance_matrix = np.stack(distance_matrix)
        else:
            self.distance_matrix = None



        self.complete_coord_info = {}
        for Li_index, data in per_Li_data.items():
            self.complete_coord_info[Li_index] = {
                "cartesian_positions": np.array(data["cartesian_positions"]),
                "global_times": np.array(data["global_times"]),
                "cutoffs": np.array(data["cutoffs"]),
                "num_neighbors": np.array(data["num_neighbors"]),
                "flattened_O_providers": np.array(data["flattened_O_providers"], dtype=int) if data["flattened_O_providers"] else np.empty((0,3), dtype=int),
                #"flattened_neighborset": np.array(data["flattened_neighborset"], dtype=int) if data["flattened_neighborset"] else np.empty((0,2), dtype=int),
            }
        for anion_index, data in per_anion_data.items():
            per_anion_data[anion_index] = {
                "global_times": np.array(data["global_times"]),
                "disp": np.array(data["disp"]),
            }
        self.anion_displacements = per_anion_data


        #self.anion_displacements = anion_displacements
        self.anion_coordinating_Li = {i:np.array(anion_coordinating_Li[i]) for i in anion_coordinating_Li.keys()}

        # Li_index already in the key information, unnecessary overload.        
        #for index, Li_index in enumerate(envinfo):
        #    for j in range(len(envinfo[Li_index])):
        #        envinfo[Li_index][j]['Li_index'] = Li_index

        #self.complete_coord_info = envinfo

        return 


    def u_simplify_complete_coord_info_to_states(self, methods=[1], method2_solvent_anion_dict=None):
        """
        Simplify the complete coordination information to states.
        for n-molecule systems.

        """

        def construct_residue_dict(flattened_O_providers, time):
            unique_residues = np.unique(flattened_O_providers[(flattened_O_providers[:,0] == time)][:,1])
            rv = {}
            for i in unique_residues:
                vals = list(flattened_O_providers[(flattened_O_providers[:,0] == time) & (flattened_O_providers[:,1] == i), 2])
                vals.sort()
                rv[i] = vals
            #print(a[(a[:,0] == time)])
            return rv


        total_state_info = {}
        for Li_index in self.complete_coord_info.keys():
            global_times_list = []
            cartesian_positions_list = []
            method_dict_lists = {}
            for method in methods:
                method_dict_lists[method] = []
            for timeindex in self.complete_coord_info[Li_index]['global_times']:
                coord_env = {}
                coord_env['cartesian_position'] = self.complete_coord_info[Li_index]['cartesian_positions'][timeindex]
                coord_env['global_time'] = self.complete_coord_info[Li_index]['global_times'][timeindex]
                coord_env['O_providers'] = construct_residue_dict(self.complete_coord_info[Li_index]['flattened_O_providers'], timeindex)
                current_Li_state = {}

                sp_counter = 0
                for solvent in self.unique_types['solvent']:
                    current_Li_state[solvent] = {}
                    print(f"adding {sp_counter}th solvent {solvent}")
                    sp_counter += 1
                for anion in self.unique_types['anion']:
                    current_Li_state[anion] = {}
                    print(f"adding {sp_counter}th anion {anion}")
                    sp_counter += 1
                print(f"total species counter: {sp_counter}")

                # traditional agg, ssip, cip classifciation
                contains_anion = False
                is_agg = False
                anion_keys = []
                for res_id in coord_env['O_providers'].keys():
                    if self.res_id_mapper[res_id] == "anion":
                        contains_anion = True
                        anion_keys.append(res_id)
                        #break
                if contains_anion:
                    #summed_value = 0
                    for anion_key in anion_keys:
                        for anion_type in self.unique_types['anion']:
                            summed_value = np.sum((self.anion_coordinating_Li[anion_type][:,0] == timeindex) & (self.anion_coordinating_Li[anion_type][:,1] == anion_key))
                            # this is how many Li this anion of anion_key as residue and anion_type as type binds to.
                            # if any time summed_value is 2 ore more, it is an agg.
                            if summed_value >= 2:
                                is_agg = True
                                break
                if not contains_anion:
                    solvation_env = 'ssip'
                elif contains_anion and not is_agg:
                    solvation_env = 'cip'
                elif contains_anion and is_agg:
                    solvation_env = 'agg'
                current_Li_state['solvation_env_simple'] = solvation_env

                if 2 in methods:
                    # multi-solvent agg, ssip, cip classification
                    """
                    take dictionary as input. {"solvent":["PL1", "PL2"], "anion":["AN1", "PL3"]} something like this.
                    determine what is true solvent. in other words, find whether glyme is PL1 or PL2.
                    then, all others are considered anions.

                    ssip_0 : pure glyme solvation : always 1
                    cip_0,1,2 : contact that has only index0, or index1 anion or more than 2 indices of anions as contact points
                    agg_0,1,2 : agg of cip type 0,1, or 2

                    
                    ssip_0 : pure glyme solvation : always 1
                    cip_0 : Na that has at least one anion, but no zwits. 2**n_anion-1 cases (3 anions : 0,1,2,01,02,12,0123)
                    cip_1 : Na that has at least one zwit, but no anion.
                    cip_2 : Na that has at least one zwit and one anion.
                    agg_0 : cip_0 that is connected by molecule index 0 (n (1 contact) + n*2 + )
                    agg_1 : cip_1 that is connected by molecule index 1
                    agg_0_1-0 : cip_0_1 that is connected by anion, not zwit
                    agg_0_1-1 : cip_0_1 that is connected by zwit, not anion
                    agg_0_1-0_1 : cip_0_1 that is connected by zwit and anion
                    """
                    if method2_solvent_anion_dict is None:
                        raise ValueError("Please provide a dictionary of solvent and anion types for method 2.")
                    solvent_list = method2_solvent_anion_dict['solvent']
                    anion_list = method2_solvent_anion_dict['anion']
                    solvent_res_ids = {}
                    anion_res_ids = {}
                    for solvent_type in solvent_list:
                        solvent_res_ids[solvent_type] = []
                    for anion_type in anion_list:
                        anion_res_ids[anion_type] = []
                    for res_id in coord_env['O_providers'].keys():
                        if self.res_id_to_type[res_id] in solvent_list:
                            solvent_res_ids[self.res_id_to_type[res_id]].append(res_id)
                        elif self.res_id_to_type[res_id] in anion_list:
                            anion_res_ids[self.res_id_to_type[res_id]].append(res_id)
                    if sum([len(anion_res_ids[i]) for i in anion_res_ids.keys()]) == 0:
                        solvation_env = 'ssip'
                    else:
                        contacts = []
                        for anion_type in anion_list:
                            if len(anion_res_ids[anion_type]) > 0:
                                contacts.append(anion_type)
                        #solvation_env = f'cip_{"_".join(contacts)}'
                        connecting_contacts = []
                        for anion_type in contacts:
                            for residue in anion_res_ids[anion_type]:
                                if np.sum((self.anion_coordinating_Li[anion_type][:,0] == timeindex) & (self.anion_coordinating_Li[anion_type][:,1] == residue)) >= 2: 
                                    # This is to be checked. not done yet. needs to be done for all type_rdfs.
                                    connecting_contacts.append(anion_type)
                                    break
                        if len(connecting_contacts) == 0:
                            if len(contacts) == 1:
                                solvation_env = f'cip-{anion_list.index(contacts[0])}'
                            else:
                                solvation_env = f'cip-{len(anion_res_ids.keys())}'
                                #solvation_env = f'cip-{"_".join(contacts)}'
                        else:
                            if len(contacts) == 1:
                                solvation_env = f'agg-{anion_list.index(contacts[0])}'
                            else:
                                solvation_env = f'agg-{len(anion_res_ids.keys())}'
                                #solvation_env = f'agg-{"_".join(contacts)}-{"_".join(connecting_contacts)}'
                    current_Li_state['u_solvation_env'] = solvation_env
                    
                # agg_determination : TBD after discussion with Dongmin
                for res_id in coord_env['O_providers'].keys():
                    # determine which type_rdf this res_id is.
                    type_rdf = self.types_rdf[coord_env['O_providers'][res_id][0]]
                    current_Li_state[type_rdf][res_id] = coord_env['O_providers'][res_id]
                num_solvation_residue_dict = {}
                for specie in self.unique_types['solvent']:
                    num_solvation_residue_dict[specie] = len(current_Li_state[specie].keys())
                for specie in self.unique_types['anion']:
                    num_solvation_residue_dict[specie] = len(current_Li_state[specie].keys())
                num_solvation_atom_dict = {}
                for specie in self.unique_types['solvent']:
                    num_solvation_atom_dict[specie] = sum([len(current_Li_state[specie][res_id]) for res_id in current_Li_state[specie].keys()])
                for specie in self.unique_types['anion']:
                    num_solvation_atom_dict[specie] = sum([len(current_Li_state[specie][res_id]) for res_id in current_Li_state[specie].keys()])
                current_Li_state['Li_index'] = Li_index
                current_Li_state['cartesian_position'] = coord_env['cartesian_position']
                current_Li_state['global_time'] = coord_env['global_time']
                current_Li_state['num_solvation_residue_dict'] = num_solvation_residue_dict
                current_Li_state['num_solvation_atom_dict'] = num_solvation_atom_dict
                global_times_list.append(coord_env['global_time'])
                cartesian_positions_list.append(coord_env['cartesian_position'])
                method_dict = {}


                if 1 in methods:
                    """
                    method 1: ssip, cip, agg -> refer to "extract_entity_events_coordination_environments" function.
                    possible vlues : 0,1,2
                    """
                    if current_Li_state['solvation_env_simple'] == 'ssip':
                        method_1_state = 0
                    elif current_Li_state['solvation_env_simple'] == 'cip':
                        method_1_state = 1
                    elif current_Li_state['solvation_env_simple'] == 'agg':
                        method_1_state = 2
                    method_dict[1] = method_1_state
                if 2 in methods:
                    if current_Li_state['u_solvation_env'] == 'ssip':
                        method_2_state = 0
                    elif current_Li_state['u_solvation_env'].startswith('cip'):
                        method_2_state = int(current_Li_state['u_solvation_env'].split('-')[1]) + 1
                    elif current_Li_state['u_solvation_env'].startswith('agg'):
                        method_2_state = int(current_Li_state['u_solvation_env'].split('-')[1]) + 2 + len(anion_list) 
                        # ssip : always 0
                        # cip : num_anion + 1 count (start from 1 to num_anion + 1) (e.g. num_anion = 2, then 1, 2, 3)
                        # agg : num_anion + 1 count (start from 2 + num_anion) and goes up to (2 + 2*num_anion)
                        # (e.g. nm_anion = 2, then 4, 5, 6 -> total of 7 states)
                        # total number of states : 3 + 2*num_anion
                    method_dict[2] = method_2_state

                
                current_Li_state['method_dict'] = method_dict
                for method in method_dict_lists.keys():
                    method_dict_lists[method].append(method_dict[method])
                
                #Li_state_list.append(current_Li_state)
            total_state_info[Li_index] = {"global_times":np.array(global_times_list, dtype=int), 
                                        "cartesian_positions":np.array(cartesian_positions_list, dtype=float), 
                                        "method_dicts":method_dict_lists}        
        self.u_complete_states = total_state_info
        return self.u_complete_states           



    def u_state_to_state_transitions(self, method2_solvent_anion_dict=None):
        """
        take in self.complete_states.
        identify how many states there are. Then compute transitions between them. e.g. (0-0, 0-1, 0-2, ...)
        assign displacements to each transition like i do for extract_entity_events.
        """
        num_states_mapper = {
            1 : 3
        }
        if method2_solvent_anion_dict is not None:
            num_states_mapper[2] = 3 + 2*len(method2_solvent_anion_dict['anion'])
        state_to_state_recorder_per_Li = {}
        """
        data structure:
        key : Li_index, value : dictionary of keys: method_key, values: dictionary of key : transition_code, values: list of dictionarys of {'global_time': X, "disp": disp}

        """
        for Li_index in self.u_complete_states.keys():
            transition_dict = {}
            for method_key in self.u_complete_states[Li_index]['method_dicts'].keys():
                if method_key == 2 and method2_solvent_anion_dict is None:
                    raise ValueError("Please provide a dictionary of solvent and anion types for method 2.")
                transition_recorder = {}
                for i in range(num_states_mapper[method_key]):
                    for j in range(num_states_mapper[method_key]):
                        transition_recorder[f"{i}-{j}"] = {"global_time":[], "disp":[]}
                assert(len(transition_recorder.keys()) == num_states_mapper[method_key]**2)
                for index, i in enumerate(self.u_complete_states[Li_index]['global_times']):
                    if index == len(self.u_complete_states[Li_index]['global_times']) - 1:
                        # end of the trajectory
                        continue
                    #displacement = self.wrap_disp(self.complete_states[Li_index][index+1]['cartesian_position'] - self.complete_states[Li_index][index]['cartesian_position'])
                    displacement = self.u_complete_states[Li_index]['cartesian_positions'][index+1] - self.u_complete_states[Li_index]['cartesian_positions'][index]
                    assert (len(displacement) == 3)
                    transition_code = f"{self.u_complete_states[Li_index]['method_dicts'][method_key][index]}-{self.u_complete_states[Li_index]['method_dicts'][method_key][index+1]}"
                    transition_recorder[transition_code]["global_time"].append(index)
                    transition_recorder[transition_code]["disp"].append(displacement)
                for transition_code in transition_recorder.keys():
                    transition_recorder[transition_code]["global_time"] = np.array(transition_recorder[transition_code]["global_time"], dtype=int)
                    transition_recorder[transition_code]["disp"] = np.array(transition_recorder[transition_code]["disp"], dtype=float)
                transition_dict[method_key] = transition_recorder
                assert(len(self.u_complete_states[Li_index]['global_times']) - 1 == sum([len(transition_recorder[i]['global_time']) for i in transition_recorder.keys()]))

            state_to_state_recorder_per_Li[Li_index] = transition_dict
        self.u_state_transitions = state_to_state_recorder_per_Li
        return state_to_state_recorder_per_Li


    def u_extract_entity_events(self, method):
        """
        A-B-C-D if four solvents or anions in total.
        A-B-C-D-E if five solvents or anions in total.
        method : "add_or_remove", "change_or_nochange"
        """
        if method not in ["add_or_remove", "change_or_nochange"]:
            raise ValueError(f"Invalid method: {method}")
        event_recorder_per_Li = {}
        num_unique_types = len(self.unique_types['solvent']) + len(self.unique_types['anion'])
        for Li_index in self.complete_coord_info.keys():
            # Generate all possible combinations as strings directly
            if method == "add_or_remove":
                combinations = ['-'.join(str(x) for x in p) for p in itertools.product(range(4), repeat=num_unique_types)]
            elif method == "change_or_nochange":
                combinations = ['-'.join(str(x) for x in p) for p in itertools.product(range(2), repeat=num_unique_types)]
            # Create dictionary with combinations as keys
            event_recorder = {rc: {"global_time":[], "disp": []} for rc in combinations}
            cartesian_positions = self.complete_coord_info[Li_index]['cartesian_positions']
            flattened_O_providers = self.complete_coord_info[Li_index]['flattened_O_providers']
            n_frames = cartesian_positions.shape[0]
            type_list = self.unique_types['solvent'] + self.unique_types['anion']
            print(f"Order of types: {type_list}")
            for t in range(n_frames - 1):
                neighborset_t = set(flattened_O_providers[flattened_O_providers[:,0] == t, 1])
                neighborset_tp1 = set(flattened_O_providers[flattened_O_providers[:,0] == t+1, 1])
                removed = neighborset_t - neighborset_tp1
                added = neighborset_tp1 - neighborset_t
                displacement = cartesian_positions[t+1] - cartesian_positions[t]
                added_dict = {}
                removed_dict = {}
                for type_ in type_list:
                    added_dict[type_] = False
                    removed_dict[type_] = False
                for res_id in added:
                    added_dict[self.res_id_to_type[res_id]] = True
                for res_id in removed:
                    removed_dict[self.res_id_to_type[res_id]] = True
                def get_return_code_bool(added, removed):
                    """
                    added, removed: booleans
                    """
                    if added and removed:
                        return 3
                    if not added and not removed:
                        return 0
                    if added and (not removed):
                        return 1
                    if (not added) and removed:
                        return 2
                def get_return_code_str(added_dict, removed_dict, method=method):
                    """
                    method : "add_or_remove", "change_or_nochange"
                    """
                    if method not in ["add_or_remove", "change_or_nochange"]:
                        raise ValueError(f"Invalid method: {method}")
                    number_list = []
                    for type_ in type_list:
                        number_list.append(get_return_code_bool(added_dict[type_], removed_dict[type_]))
                    if method == "change_or_nochange":
                        number_list = [0 if x == 0 else 1 for x in number_list]
                    return_code = '-'.join(str(x) for x in number_list)
                    return return_code
                return_code = get_return_code_str(added_dict, removed_dict, method=method)
                event_recorder[return_code]["global_time"].append(t)
                event_recorder[return_code]["disp"].append(displacement)
            for rc in event_recorder.keys():
                if len(event_recorder[rc]["global_time"]) > 0:
                    event_recorder[rc]["global_time"] = np.array(event_recorder[rc]["global_time"], dtype=int)
                    event_recorder[rc]["disp"] = np.array(event_recorder[rc]["disp"], dtype=float)
            event_recorder_per_Li[Li_index] = event_recorder
        self.u_entitychange = event_recorder_per_Li
        return event_recorder_per_Li


    # Inside your class
    def simplify_complete_coord_info_to_states(self, methods=[1], n_workers=8, parallel=True):
        """
        Parallelizes per Li_index (default 8 workers). Set parallel=False for serial.

        method 1: ssip, cip, agg -> refer to "extract_entity_events_coordination_environments" function.
        method 2: num_coordinating_residues (from 0 to 8)
        method 3: num_solvent_residues (from 0 to 5)
        method 4: num_anion_residues (from 0 to 3)
        method 5: (num_solvent_residues, num_anion_residues) (6 X 4)
        method 6: num_solvent_atoms (from 0 to 9)
        method 7: num_anion_atoms (from 0 to 5)
        method 8: (num_solvent_atoms, num_anion_atoms) (0 to 9, 0 to 5)

        """
        total_state_info = {}

        # --- CHANGED: optional parallel branch ---
        if parallel and n_workers and n_workers > 1:
            # Build lightweight payloads so workers don’t receive the whole 'self'
            payloads = []
            for Li_index, info in self.complete_coord_info.items():
                # each 'info' must have the arrays we need; dicts below assumed small
                payloads.append((
                    Li_index,
                    {
                        'flattened_O_providers': info['flattened_O_providers'],
                        'cartesian_positions':   info['cartesian_positions'],
                        'global_times':          info['global_times'],
                    },
                    methods,
                    self.res_id_mapper,
                    self.res_id_to_type,
                    self.anion_coordinating_Li
                ))

            # NOTE (memory):
            # - On Linux (fork) this is efficient.
            # - On macOS/Windows (spawn) these arrays are copied; reduce n_workers if RAM is limited.

            max_workers = min(n_workers, os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_process_one_li, p) for p in payloads]
                for fut in as_completed(futures):
                    Li_index, result = fut.result()
                    total_state_info[Li_index] = result

        else:
            # --- ORIGINAL serial path (kept) ---
            for Li_index in self.complete_coord_info.keys():
                now = datetime.now()
                readable_format = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"markov, analysing Li index: {Li_index}, {readable_format}")

                info = self.complete_coord_info[Li_index]
                arrs, uniq_times, t_starts, t_ends = presort_time_res_val(info['flattened_O_providers'])

                global_times_list = []
                cartesian_positions_list = []
                method_dict_lists = {m: [] for m in methods}

                for timeindex in info['global_times']:
                    O_providers = residue_dict_from_presorted(arrs, uniq_times, t_starts, t_ends, timeindex)

                    contains_anion = False
                    is_agg = False
                    anion_keys = []
                    for res_id in O_providers.keys():
                        if self.res_id_mapper[res_id] == "anion":
                            contains_anion = True
                            anion_keys.append(res_id)
                    if len(list(set(anion_keys))) > 1:
                        # Li binding to more than 2 TFSI anions. This is technically not a CIP, but an AGG now.
                        is_agg = True
                    if contains_anion:
                        for anion_key in anion_keys:
                            if np.sum((self.anion_coordinating_Li[self.res_id_to_type[anion_key]][:,0] == timeindex) &
                                    (self.anion_coordinating_Li[self.res_id_to_type[anion_key]][:,1] == anion_key)) >= 2:
                                is_agg = True
                                break

                    solvation_env = 'ssip' if not contains_anion else ('cip' if not is_agg else 'agg')

                    solvent_dict = {}
                    anion_dict = {}
                    for res_id, vlist in O_providers.items():
                        typ = self.res_id_mapper[res_id]
                        if typ == 'solvent':
                            solvent_dict[res_id] = vlist
                        elif typ == 'anion':
                            anion_dict[res_id] = vlist
                        else:
                            raise ValueError(f"Unknown res_id: {res_id, self.res_id_mapper[res_id]}")

                    num_solvent = len(solvent_dict)
                    num_anion = len(anion_dict)
                    num_anion_atoms = sum(len(anion_dict[res_id]) for res_id in anion_dict)
                    num_solvent_atoms = sum(len(solvent_dict[res_id]) for res_id in solvent_dict)

                    global_times_list.append(info['global_times'][timeindex])
                    cartesian_positions_list.append(info['cartesian_positions'][timeindex])

                    method_dict = {}
                    if 1 in methods:
                        method_dict[1] = {'ssip': 0, 'cip': 1, 'agg': 2}[solvation_env]
                    if 2 in methods:
                        s = num_anion + num_solvent
                        if s >= 11:
                            warnings.warn(f"method_2_state ({s}) exceeds 10, setting to 10")
                            s = 10
                        method_dict[2] = s
                    if 3 in methods:
                        s = num_solvent
                        if s >= 8:
                            warnings.warn(f"method_3_state ({s}) exceeds 7, setting to 7")
                            s = 7
                        method_dict[3] = s
                    if 4 in methods:
                        s = num_anion
                        if s >= 4:
                            warnings.warn(f"method_4_state ({s}) exceeds 3, setting to 3")
                            s = 3
                        method_dict[4] = s
                    if 5 in methods:
                        a, b = num_solvent, num_anion
                        if a >= 8:
                            warnings.warn(f"num_solvent ({a}) exceeds 7, setting to 7")
                            a = 7
                        if b >= 4:
                            warnings.warn(f"num_anion ({b}) exceeds 3, setting to 3")
                            b = 3
                        method_dict[5] = a * 4 + b
                    if 6 in methods:
                        s = num_solvent_atoms
                        if s >= 10:
                            warnings.warn(f"method_6_state ({s}) exceeds 9, setting to 9")
                            s = 9
                        method_dict[6] = s
                    if 7 in methods:
                        s = num_anion_atoms
                        if s >= 10:
                            warnings.warn(f"method_7_state ({s}) exceeds 9, setting to 9")
                            s = 9
                        method_dict[7] = s
                    if 8 in methods:
                        a, b = num_solvent_atoms, num_anion_atoms
                        if a >= 10:
                            warnings.warn(f"num_solvent_atoms ({a}) exceeds 9, setting to 9")
                            a = 9
                        if b >= 10:
                            warnings.warn(f"num_anion_atoms ({b}) exceeds 9, setting to 9")
                            b = 9
                        method_dict[8] = a * 10 + b

                    for m in method_dict.keys():
                        method_dict_lists[m].append(method_dict[m])

                total_state_info[Li_index] = {
                    "global_times":        np.array(global_times_list, dtype=int),
                    "cartesian_positions": np.array(cartesian_positions_list, dtype=float),
                    "method_dicts":        method_dict_lists
                }

        self.complete_states = total_state_info
        return self.complete_states

    def state_to_state_transitions(self):
        """
        take in self.complete_states.
        identify how many states there are. Then compute transitions between them. e.g. (0-0, 0-1, 0-2, ...)
        assign displacements to each transition like i do for extract_entity_events.
        """
        num_states_mapper = {
            1 : 3,
            2 : 11,
            3 : 8,
            4 : 4,
            5 : 32,
            6 : 10,
            7 : 10,
            8 : 100
        }
        state_to_state_recorder_per_Li = {}
        """
        data structure:
        key : Li_index, value : dictionary of keys: method_key, values: dictionary of key : transition_code, values: list of dictionarys of {'global_time': X, "disp": disp}

        """
        for Li_index in self.complete_states.keys():
            transition_dict = {}
            for method_key in self.complete_states[Li_index]['method_dicts'].keys():
                transition_recorder = {}
                for i in range(num_states_mapper[method_key]):
                    for j in range(num_states_mapper[method_key]):
                        transition_recorder[f"{i}-{j}"] = {"global_time":[], "disp":[]}
                assert(len(transition_recorder.keys()) == num_states_mapper[method_key]**2)
                for index, i in enumerate(self.complete_states[Li_index]['global_times']):
                    if index == len(self.complete_states[Li_index]['global_times']) - 1:
                        # end of the trajectory
                        continue
                    #displacement = self.wrap_disp(self.complete_states[Li_index][index+1]['cartesian_position'] - self.complete_states[Li_index][index]['cartesian_position'])
                    displacement = self.complete_states[Li_index]['cartesian_positions'][index+1] - self.complete_states[Li_index]['cartesian_positions'][index]
                    assert (len(displacement) == 3)
                    transition_code = f"{self.complete_states[Li_index]['method_dicts'][method_key][index]}-{self.complete_states[Li_index]['method_dicts'][method_key][index+1]}"
                    transition_recorder[transition_code]["global_time"].append(index)
                    transition_recorder[transition_code]["disp"].append(displacement)
                for transition_code in transition_recorder.keys():
                    transition_recorder[transition_code]["global_time"] = np.array(transition_recorder[transition_code]["global_time"], dtype=int)
                    transition_recorder[transition_code]["disp"] = np.array(transition_recorder[transition_code]["disp"], dtype=float)
                transition_dict[method_key] = transition_recorder
                assert(len(self.complete_states[Li_index]['global_times']) - 1 == sum([len(transition_recorder[i]['global_time']) for i in transition_recorder.keys()]))

            state_to_state_recorder_per_Li[Li_index] = transition_dict
        self.state_transitions = state_to_state_recorder_per_Li
        return state_to_state_recorder_per_Li

    def extract_entity_events(self):
        # first number: polymer, second number: TFSI
        # 0 : no change, 1 : added, 2: removed, 3 : add and remove
        import numpy as np
        from typing import List, Set, Dict, Tuple

        # Cache: id(array) -> (order, time_slices, neighbors_by_time_sets)
        _NEIGHBOR_CACHE: Dict[int, Tuple[np.ndarray, Dict[int, Tuple[int,int]], List[Set[int]]]] = {}

        def _build_time_slices(a: np.ndarray):
            # Global sort by (time, neighbor) so neighbors within a time are grouped
            order = np.lexsort((a[:,1], a[:,0]))  # col0=time, col1=neighbor
            times_sorted = a[order, 0]
            times, t_idx, t_cnt = np.unique(times_sorted, return_index=True, return_counts=True)
            time_slices = {int(t): (int(i), int(c)) for t, i, c in zip(times, t_idx, t_cnt)}
            return order, time_slices

        def _build_neighbors_by_time_sets(a: np.ndarray, n_frames: int) -> Tuple[np.ndarray, Dict[int, Tuple[int,int]], List[Set[int]]]:
            order, time_slices = _build_time_slices(a)
            neighbors_by_time: List[Set[int]] = [set() for _ in range(n_frames)]
            for t, (start, count) in time_slices.items():
                # rows in this time slice
                idx = order[start:start+count]
                # unique neighbor ids for this time
                neighbors = np.unique(a[idx, 1])
                neighbors_by_time[t] = set(map(int, neighbors))
            return order, time_slices, neighbors_by_time

        def get_neighbors_by_time(flattened_O_providers: np.ndarray, n_frames: int) -> List[Set[int]]:
            key = id(flattened_O_providers)
            cached = _NEIGHBOR_CACHE.get(key)
            if cached is None:
                _NEIGHBOR_CACHE[key] = _build_neighbors_by_time_sets(flattened_O_providers, n_frames)
            return _NEIGHBOR_CACHE[key][2]

        def invalidate_neighbors_index(flattened_O_providers: np.ndarray) -> None:
            _NEIGHBOR_CACHE.pop(id(flattened_O_providers), None)

        event_recorder_per_Li = {}
        for Li_index in self.complete_coord_info.keys():
            event_recorder = {rc: {"global_time":[], "disp": []} for rc in [
                '0-0', '0-1', '0-2', '0-3', '1-0', '1-1', '1-2', '1-3',
                '2-0', '2-1', '2-2', '2-3', '3-0', '3-1', '3-2', '3-3'
            ]}
            cartesian_positions = self.complete_coord_info[Li_index]['cartesian_positions']
            flattened_O_providers = self.complete_coord_info[Li_index]['flattened_O_providers']
            n_frames = cartesian_positions.shape[0]
            neighbors_by_time = get_neighbors_by_time(flattened_O_providers, n_frames)

            for t in range(n_frames - 1):
                neighborset_t = neighbors_by_time[t]
                neighborset_tp1 = neighbors_by_time[t + 1]
                removed = neighborset_t - neighborset_tp1
                added = neighborset_tp1 - neighborset_t
                displacement = cartesian_positions[t+1] - cartesian_positions[t]
                solvent_added = False
                solvent_removed = False
                solvent_nochange = True
                anion_added = False
                anion_removed = False
                anion_nochange = True
                for j in removed:
                    if self.res_id_mapper[j] == 'solvent':
                        solvent_removed = True
                        solvent_nochange = False
                    if self.res_id_mapper[j] == 'anion':
                        anion_removed = True
                        anion_nochange = False
                for j in added:
                    if self.res_id_mapper[j] == 'solvent':
                        solvent_added = True
                        solvent_nochange = False
                    if self.res_id_mapper[j] == 'anion':
                        anion_added = True
                        anion_nochange = False

                # determine the polymer index (pid)
                if solvent_added and solvent_removed:
                    pid = 3
                if solvent_added and not solvent_removed:
                    pid = 1
                if not solvent_added and solvent_removed:
                    pid = 2
                if not solvent_added and not solvent_removed:
                    pid = 0
                if anion_added and anion_removed:
                    tid = 3
                if anion_added and not anion_removed:
                    tid = 1
                if not anion_added and anion_removed:
                    tid = 2
                if not anion_added and not anion_removed:
                    tid = 0
                returncode = f"{pid}-{tid}"
                event_recorder[returncode]["global_time"].append(t)
                event_recorder[returncode]["disp"].append(displacement)
            for rc in event_recorder:
                event_recorder[rc]["global_time"] = np.array(event_recorder[rc]["global_time"], dtype=int)
                event_recorder[rc]["disp"] = np.array(event_recorder[rc]["disp"], dtype=float)
            event_recorder_per_Li[Li_index] = event_recorder
        self.entitychange = event_recorder_per_Li
        return event_recorder_per_Li


    def extract_entity_events_coordination_environments(self, former_or_latter='former'): #TBD
        # first number: polymer, second number: TFSI
        # 0 : no change, 1 : added, 2: removed, 3 : add and remove
        """
        former_or_latter : 'former' or 'latter' (when to compute the coordination type; agg, ssip, cip)
        """
        event_recorder_per_Li = {}
        for Li_index in self.complete_coord_info.keys():
            event_recorder = {rc: {"global_time":[], "disp": []} for rc in [
                'ssip', 'cip', 'agg'
            ]}
            cartesian_positions = self.complete_coord_info[Li_index]['cartesian_positions']
            flattened_O_providers = self.complete_coord_info[Li_index]['flattened_O_providers']
            n_frames = cartesian_positions.shape[0]
            for t in range(n_frames - 1):
                neighborset_t = set(flattened_O_providers[flattened_O_providers[:,0] == t, 1])
                neighborset_tp1 = set(flattened_O_providers[flattened_O_providers[:,0] == t+1, 1])
                removed = neighborset_t - neighborset_tp1
                added = neighborset_tp1 - neighborset_t
                displacement = cartesian_positions[t+1] - cartesian_positions[t]
  
                # solvent separated IP : no anions nearby
                # CIP : 1 or more anions nearby
                # AG : aggregate (1 or more anions nearby, which is part of another aggregate)
                # how do i check this? loop through all of the Li at that frame
                # anion_dictionary make a anion index with key : anion-index, value : list of indices of lithium; 
                contains_anion = False
                is_agg = False
                if former_or_latter == 'former':
                    neighborset = neighborset_t
                elif former_or_latter == 'latter':
                    neighborset = neighborset_tp1
                anion_keys = []
                for res_id in neighborset: # current neighborset
                    if self.res_id_mapper[res_id] == 'anion':
                        contains_anion = True
                        anion_keys.append(res_id)
                        #break
                if len(list(set(anion_keys))) > 1:
                    # Li binding to more than 2 TFSI anions. This is technically not a CIP, but an AGG now.
                    is_agg = True
                if contains_anion:
                    for anion_key in anion_keys:
                        if np.sum((self.anion_coordinating_Li[self.res_id_to_type[anion_key]][:,0] == t) & (self.anion_coordinating_Li[self.res_id_to_type[anion_key]][:,1] == anion_key)) >= 2:
                            is_agg = True
                            break
                if not contains_anion:
                    return_code = 'ssip'
                elif contains_anion and not is_agg:
                    return_code = 'cip'
                elif contains_anion and is_agg:
                    return_code = 'agg'
                event_recorder[return_code]["global_time"].append(t)
                event_recorder[return_code]["disp"].append(displacement)
            for rc in event_recorder:
                event_recorder[rc]["global_time"] = np.array(event_recorder[rc]["global_time"], dtype=int)
                event_recorder[rc]["disp"] = np.array(event_recorder[rc]["disp"], dtype=float)
            event_recorder_per_Li[Li_index] = event_recorder
        self.entitychange = event_recorder_per_Li
        return event_recorder_per_Li


    def wrap_disp(self, disp):
        frac_disp = np.array(disp)/np.array(self.boxdim)
        rv = []
        for d in frac_disp:
            if d > 0.5:
                rv.append(d-1)
            elif d < -0.5:
                rv.append(d+1)
            else:
                rv.append(d)
        return np.array(rv)*np.array(self.boxdim)



class PolymerElectrolyteAnalyzer:
    def __init__(self, PATH, label):
        self.PATH = PATH
        self.label = label
    """
    def read_structures(self, analyze_anions=True, timestep=1, stepskip=1, correct_com=False):
        '''
        FOLDER_PATH : path of the directory that contains three files:
        atom_names_rdf.txt : file that contains the atom names in the system
        xyz_unwrapped_rdf_all.txt : file that contains the unwrapped xyz coordinates of the system
        box_len.txt : file that contains the box length of the system
        timestep : timestep of thte MD trajectory that is being read.
        stepskip : how frequently you want to save from the stepskip.
        in the class, I would only save sampling_freq (in ps)
        returns a list of pymatgen structure objects
        '''
        # FOLDER_PATH = "../featurize/actual_25_correct_frame"
        #FOLDER_PATH = "/Volumes/PROGRADE/353K_simulations_arash/2.25"
        # FOLDER_PATH = "/home/pleon/projects/collaborations/tri/vacancy_count/tfsi/0.75"

        # xyz_wrapped_rdf_beginning = first 100 frames, middle = middle 100 frames, end = final 100 frames
        self.sampling_freq = timestep * stepskip
        xyz_rdf, atom_names, atom_names_long_rdf, res_ids = read_xyz(
            self.PATH, "atom_names_rdf.txt", "xyz_unwrapped_rdf_all.txt",
            include_resid=True
        )
        atom_names_long_rdf = np.array(atom_names_long_rdf)
        box_len = float(open(f"{self.PATH}/box_len.txt", "r").read())
        
        if correct_com:
            for time_index, global_time_frame in enumerate(xyz_rdf):
                com = np.mean(global_time_frame, axis=0)
                xyz_rdf[time_index] = xyz_rdf[time_index] - com
        def get_structure_list(atom_names_rdf, xyz_rdf, box_dim):
            '''
            atom_names_rdf: list of atom names
            xyz_rdf: position of atoms
            box_dim: integer number for a cubic sized box in Angstrom
            returns a list of pymatgen structure objects.
            deprecatd because it is way too memory intesive.
            Note: the index of atoms are not modified.
            '''
            if type(box_dim)==float:
                # cubic box
                lattice = Lattice.cubic(box_dim)
            else:
                # non-cubic box
                lattice = Lattice(box_dim)

            structure_list = []
            for i in range(len(xyz_rdf)):
                structure_list.append(Structure(lattice= lattice , species=atom_names_rdf, coords=xyz_rdf[i], coords_are_cartesian=True))
            return structure_list
        def get_initial_structure(atom_names_rdf, xyz_rdf, box_dim):
            '''
            atom_names_rdf: list of atom names
            xyz_rdf: position of atoms
            box_dim: integer number for a cubic sized box in Angstrom
            returns pymatgen structure object of the first snapshot.

            Note: the index of atoms are not modified.
            '''
            if type(box_dim)==float:
                # cubic box
                lattice = Lattice.cubic(box_dim)
            else:
                # non-cubic box
                lattice = Lattice(box_dim)
            structure_init  = Structure(lattice= lattice , species=atom_names_rdf, coords=xyz_rdf[0], coords_are_cartesian=True)
            return structure_init
        def map_res_id_to_polymer_or_anion(res_ids):
            '''
            res_ids: a list of res_ids
            returns a function that determines if the res_id is a polymer, TFSI, or Li
            for now, only works for TFSI (num_atom = 15), but can be improved to work for other anions by taking it as an argument
            '''
            mapper = {}
            for i in res_ids:
                if i in mapper.keys():
                    mapper[i] +=1
                else:
                    mapper[i] = 1
            print(mapper)
            res_id_mapper = {}
            for i in mapper.keys():
                if mapper[i] > 15: # polymer, not TFSI
                    res_id_mapper[i] = "polymer"
                elif mapper[i] == 15: # TFSI
                    res_id_mapper[i] = "TFSI"
                elif mapper[i] == 1: # Li
                    res_id_mapper[i] = "Li"
            #def foo(x):
            #    return res_id_mapper[x]
            return res_id_mapper



        def extract_coordination_environments(s, Li_index, global_time_index):
            
            s : pymatgen structure
            Li_index : index of lithium
            
            site = s[Li_index]
            nlist = s.get_neighbors(site, 6)
            nlist.sort(key=lambda x:x.nn_distance)
            nlist_O = [i for i in nlist if i.specie == Element("O")] # Here, I am considering only oxygen as a coordinating anion. In other polymers, we can add Nitrogen, etc.
            nlist_non_H_O = [i for i in nlist if i.specie not in [Element("H"), Element("O")]]
            nearest_non_H_O = nlist_non_H_O[0]
            convex_nlist_O = nlist_O[:4]
            for neighb_index, neighb in enumerate(nlist_O[4:]):
                og_convex = ConvexHull([i.coords for i in convex_nlist_O])
                new_convex = ConvexHull([i.coords for i in convex_nlist_O] + [neighb.coords])
                new_non_H_convex = ConvexHull([i.coords for i in convex_nlist_O] + [nearest_non_H_O.coords] + [neighb.coords])
                exist_non_H_O = False
                for k in nlist_non_H_O:
                    if k.nn_distance < neighb.nn_distance:
                        exist_non_H_O = True
                if len(new_convex.vertices) == len(og_convex.vertices) + 1 and len(new_convex.vertices) != len(new_non_H_convex.vertices) and neighb.nn_distance < 1.75 * convex_nlist_O[0].nn_distance:
                    convex_nlist_O.append(neighb)
                else:
                    for i in convex_nlist_O:
                        i.label = self.res_ids[i.index] # decorating pymatgen Neighbor object by labeling them by which res_id this oxygen comes from
                    each_env_info = {'cartesian_position': site.coords, 
                                        'global_time':global_time_index, 
                                        'cutoff':max([i.nn_distance for i in convex_nlist_O]), 
                                        'num_neighbor':len(convex_nlist_O), 
                                        'neighborlist':convex_nlist_O}
                    break
                if neighb_index == len(nlist_O[4:])-1:
                    for i in convex_nlist_O:
                        i.label = self.res_ids[i.index] # decorating pymatgen Neighbor object by labeling them by which res_id this oxygen comes from
                    each_env_info = {'cartesian_position': site.coords, 
                                        'global_time':global_time_index, 
                                        'cutoff':max([i.nn_distance for i in convex_nlist_O]), 
                                        'num_neighbor':len(convex_nlist_O), 
                                        'neighborlist':convex_nlist_O}
            return each_env_info







        #slist = get_structure_list(atom_names, xyz_rdf, box_len)
        #self.structurelist = slist
        self.res_ids = res_ids
        self.res_id_mapper = map_res_id_to_polymer_or_anion(self.res_ids)
        #print(f"Read {len(slist)} structures from {self.PATH}")
        #self.structure = self.structurelist[0]
        self.structure = get_initial_structure(atom_names, xyz_rdf, box_len)
        self.boxdim = self.structure.lattice.abc




        if analyze_anions:
            anion_center_indices = []
            for site_index, site in enumerate(self.structure):
                if site.specie == Element('N'):
                    anion_center_indices.append(site_index)
                    nitrogen_site = site
            anion_displacements = {}
            for anion_index in anion_center_indices:
                anion_displacements[self.res_ids[anion_index]] = []
            

        Li_indices = []
        envinfo = {}
        for site_index, site in enumerate(self.structure):
            if site.specie == Element("Li"):
                Li_indices.append(site_index)
                envinfo[site_index] = []
        if type(box_len)==float:
            # cubic box
            lattice = Lattice.cubic(box_len)
        else:
            # non-cubic box
            lattice = Lattice(box_len)
        s = self.structure
        for global_time_index, frame in enumerate(xyz_rdf[::stepskip]):
            s_prev = s
            s = Structure(lattice= lattice , species=atom_names, coords=xyz_rdf[global_time_index], coords_are_cartesian=True)
            for Li_index in Li_indices:
                each_env_info = extract_coordination_environments(s, Li_index, global_time_index)
                O_providers = {}
                for jindex, j in enumerate(each_env_info['neighborlist']):
                    if self.res_ids[j.index] in O_providers.keys():
                        O_providers[self.res_ids[j.index]].append(j.index)
                    else:
                        O_providers[self.res_ids[j.index]] = [j.index]
                for j in O_providers.keys():
                    O_providers[j].sort()
                each_env_info['O_providers'] = O_providers
                each_env_info.pop('neighborlist') # this is memory-intensive.
                each_env_info['neighborset'] = set(each_env_info['O_providers'].keys())
                envinfo[Li_index].append(each_env_info)

            if global_time_index == 0:
                continue
            else:
                for anion_index in anion_center_indices:
                    disp = self.wrap_disp(s[anion_index].coords - s_prev[anion_index].coords)
                    each_anion_disp = {"global_time":global_time_index-1, "disp":disp}
                    anion_displacements[self.res_ids[anion_index]].append(each_anion_disp)
        self.anion_displacements = anion_displacements


        # below unnecessary info.        
        #for index, Li_index in enumerate(envinfo):
        #    for j in range(len(envinfo[Li_index])):
        #        envinfo[Li_index][j]['Li_index'] = Li_index

        self.complete_coord_info = envinfo

        return 
    """

    def save_distance_matrix_parallel(self, PART_INDEX=None, timestep=1, stepskip=1, N_PART=10):
        self.sampling_freq = timestep * stepskip
        xyz_rdf, atom_names, atom_names_long_rdf, res_ids = read_xyz(
            self.PATH, "atom_names_rdf.txt", "xyz_unwrapped_rdf_all.txt",
            include_resid=True
        )
        atom_names_long_rdf = np.array(atom_names_long_rdf)
        box_len = float(open(f"{self.PATH}/box_len.txt", "r").read())


        EACH_PART_COUNT = int(len(xyz_rdf)/N_PART)+1
        START_INDEX = EACH_PART_COUNT * PART_INDEX
        if PART_INDEX == N_PART-1:
            END_INDEX = len(xyz_rdf)
        else:
            END_INDEX = EACH_PART_COUNT * (PART_INDEX+1)
        print(f"START_INDEX: {START_INDEX}, END_INDEX: {END_INDEX}, EACH_PART_COUNT: {EACH_PART_COUNT}")

        def get_initial_structure(atom_names_rdf, xyz_rdf, box_dim):
            '''
            atom_names_rdf: list of atom names
            xyz_rdf: position of atoms
            box_dim: integer number for a cubic sized box in Angstrom
            returns pymatgen structure object of the first snapshot.

            Note: the index of atoms are not modified.
            '''
            if type(box_dim)==float:
                # cubic box
                lattice = Lattice.cubic(box_dim)
            else:
                # non-cubic box
                lattice = Lattice(box_dim)
            structure_init  = Structure(lattice= lattice , species=atom_names_rdf, coords=xyz_rdf[0], coords_are_cartesian=True)
            return structure_init
        def map_res_id_to_polymer_or_anion(res_ids):
            '''
            res_ids: a list of res_ids
            DEPRECATED
            returns a function that determines if the res_id is a polymer, TFSI, or Li
            for now, only works for TFSI (num_atom = 15), but can be improved to work for other anions by taking it as an argument
            '''
            print("DEPRECATED")
            mapper = {}
            for i in res_ids:
                if i in mapper.keys():
                    mapper[i] +=1
                else:
                    mapper[i] = 1
            res_id_mapper = {}
            for i in mapper.keys():
                if mapper[i] > 15: # polymer, not TFSI
                    res_id_mapper[i] = "polymer"
                elif mapper[i] == 15: # TFSI
                    res_id_mapper[i] = "anion"
                elif mapper[i] == 1: # Li
                    res_id_mapper[i] = "Li"
            #def foo(x):
            #    return res_id_mapper[x]
            return res_id_mapper

        #slist = get_structure_list(atom_names, xyz_rdf, box_len)
        #self.structurelist = slist
        self.res_ids = res_ids
        self.res_id_mapper = map_res_id_to_polymer_or_anion(self.res_ids)
        #print(f"Read {len(slist)} structures from {self.PATH}")
        #self.structure = self.structurelist[0]
        self.structure = get_initial_structure(atom_names, xyz_rdf, box_len)
        self.boxdim = self.structure.lattice.abc



        Li_indices = []
        envinfo = {}
        for site_index, site in enumerate(self.structure):
            if site.specie == Element("Li"):
                Li_indices.append(site_index)
                envinfo[site_index] = []
        if type(box_len)==float:
            # cubic box
            lattice = Lattice.cubic(box_len)
        else:
            # non-cubic box
            lattice = Lattice(box_len)
        s = self.structure
        distance_matrix = [] # this tra
        for global_time_index, frame in enumerate(xyz_rdf[::stepskip]):
            # complete_coord_info: START_INDEX ~ END_INDEX-1 (END_INDEX is covered in the next round, TOTAL_INDEX is length of the trajectory)
            # anion_displacement: (1-0), (2-1), ... (END_INDEX-(END_INDEX-1))
            if global_time_index < START_INDEX: # not current part
                continue
            if global_time_index == END_INDEX: # not current part

                break
            s = Structure(lattice= lattice , species=atom_names, coords=xyz_rdf[global_time_index], coords_are_cartesian=True)
            s_copy = s.copy()
            s_copy.remove_species([i for i in s_copy.species if i not in [Element("Li"), Element("N")]])
            distance_matrix.append(s_copy.distance_matrix)

        self.distance_matrix = np.stack(distance_matrix)

        return
    

    def observe_specific_frame(self, record_index, Li_index, PART_INDEX=0, N_PART=1, timestep=1, stepskip=1, correct_com='baro', coordinators=["O", "F", "N"]):
        self.sampling_freq = timestep * stepskip
        self.coordinators = coordinators
        xyz_rdf, atom_names, atom_names_long_rdf, res_ids = read_xyz(
            self.PATH, "atom_names_rdf.txt", "xyz_unwrapped_rdf_all.txt",
            include_resid=True
        )
        relevant_ions = []
        for index, i in enumerate(atom_names):
            if i == "Li" or i == "N":
                relevant_ions.append(True)
            else:
                relevant_ions.append(False)
        print(sum(np.array(relevant_ions)))
        box_len = float(open(f"{self.PATH}/box_len.txt", "r").read())

        xyz_rdf_frac = xyz_rdf/box_len # change to frac_coords
        xyz_rdf_frac = np.insert(xyz_rdf_frac, 0, xyz_rdf_frac[0], axis=0) 
        dp = xyz_rdf_frac[1:, ] - xyz_rdf_frac[:-1, ] # getting frac_disps
        dp = dp - np.round(dp) # unwrapping frac_disps
        dp = dp * box_len # changing back to real coords
        f_disp = np.cumsum(dp, axis=0) 
        xyz_rdf = xyz_rdf[0] + f_disp #* cell_dim[0][0]
        print('atom_names', atom_names, len(atom_names))
        print('res_ids', res_ids, len(res_ids))
        xyz_rdf = xyz_rdf[::stepskip, ...] # Slicing by every "stepskip" frames
        if correct_com == "salt_position":
            relevant_ions = []
            for index, i in enumerate(atom_names):
                if i == "Li" or i == "N":
                    relevant_ions.append(True)
                else:
                    relevant_ions.append(False)
            print(sum(np.array(relevant_ions)))
            for frame_index, sliced_frame in enumerate(xyz_rdf):
                print(xyz_rdf.shape, sliced_frame.shape)
                com = np.mean(sliced_frame[relevant_ions], axis=0)
                xyz_rdf[frame_index] = xyz_rdf[frame_index] - com

        elif correct_com == "baro":
            mass_list = [Element(i).atomic_mass for i in atom_names]
            #print(sum(np.array(relevant_ions)))
            for frame_index, sliced_frame in enumerate(xyz_rdf):
                print(xyz_rdf.shape, sliced_frame.shape)
                com = np.sum(sliced_frame*np.array(mass_list)[:,np.newaxis], axis=0)/np.sum(np.array(mass_list)) # fixed from np.mean -> np.sum
                #com = np.mean(global_time_frame)
                xyz_rdf[frame_index] = xyz_rdf[frame_index] - com   
        elif correct_com == "solvent":
            mass_list = [Element(i).atomic_mass for i in atom_names]
            # determine solvent
            res_ids_dict = {}
            for i in np.unique(np.array(res_ids)):
                res_ids_dict[i] = 0
            for i in np.array(res_ids):
                res_ids_dict[i] += 1
            solvent_res_id = [i for i in res_ids_dict.keys() if res_ids_dict[i] > 15]
            # check this...
            relevant_ions = []
            for index, i in enumerate(res_ids):
                if i in solvent_res_id:
                    relevant_ions.append(True)
                else:
                    relevant_ions.append(False)
            for frame_index, sliced_frame in enumerate(xyz_rdf):
                print(xyz_rdf.shape, sliced_frame.shape)
                val = sliced_frame*np.array(mass_list)[:,np.newaxis]
                com = np.sum(val[relevant_ions], axis=0)/(np.array(mass_list)[relevant_ions].sum())
                xyz_rdf[frame_index] = xyz_rdf[frame_index] - com
                
        atom_names_long_rdf = np.array(atom_names_long_rdf)
        def get_initial_structure(atom_names_rdf, xyz_rdf, box_dim):
            '''
            atom_names_rdf: list of atom names
            xyz_rdf: position of atoms
            box_dim: integer number for a cubic sized box in Angstrom
            returns pymatgen structure object of the first snapshot.

            Note: the index of atoms are not modified.
            '''
            if type(box_dim)==float:
                # cubic box
                lattice = Lattice.cubic(box_dim)
            else:
                # non-cubic box
                lattice = Lattice(box_dim)
            structure_init  = Structure(lattice= lattice , species=atom_names_rdf, coords=xyz_rdf[0], coords_are_cartesian=True)
            return structure_init

        EACH_PART_COUNT = int(len(xyz_rdf)/N_PART)+1
        START_INDEX = EACH_PART_COUNT * PART_INDEX
        if PART_INDEX == N_PART-1:
            END_INDEX = len(xyz_rdf)
        else:
            END_INDEX = EACH_PART_COUNT * (PART_INDEX+1)
        print(f"START_INDEX: {START_INDEX}, END_INDEX: {END_INDEX}, EACH_PART_COUNT: {EACH_PART_COUNT}")

        #slist = get_structure_list(atom_names, xyz_rdf, box_len)
        #self.structurelist = slist
        #self.res_ids = res_ids
        #self.res_id_mapper = map_res_id_to_polymer_or_anion(self.res_ids)
        #print(f"Read {len(slist)} structures from {self.PATH}")
        #self.structure = self.structurelist[0]
        #self.structure = get_initial_structure(atom_names, xyz_rdf, box_len)
        #self.boxdim = self.structure.lattice.abc
        #self.anion_indices_forward, self.anion_indices_back = create_sequential_anion_mapping(self)




        #Li_indices = []
        #envinfo = {}
        #for site_index, site in enumerate(self.structure):
        #    if site.specie == Element("Li"):
        #        Li_indices.append(site_index)
        #        envinfo[site_index] = []
        if type(box_len)==float:
            # cubic box
            lattice = Lattice.cubic(box_len)
        else:
            # non-cubic box
            lattice = Lattice(box_len)
        #distance_matrix = [] # this tra
        s = Structure(lattice= lattice , species=atom_names, coords=xyz_rdf[record_index], coords_are_cartesian=True)
        neighbor_list = s.get_neighbors(s[Li_index], 6)
        #s.remove_sites([i.index for i in neighbor_list])
        s.remove_sites([i for i in range(len(s)) if i not in [j.index for j in neighbor_list]+ [Li_index]])
        s.to(filename="local_env-Li_{}-frame_{}.cif".format(Li_index, record_index), fmt='cif')

    def read_structures_parallel(self, 
                                 analyze_anions=True, 
                                 PART_INDEX=0, 
                                 N_PART=1, 
                                 timestep=1, 
                                 stepskip=1, 
                                 correct_com='solvent',
                                 coordinators = ["O", "F", "N"],
                                 anion="TFSI",
                                 mode = "convex",
                                 cutoff = 0,
                                 compute_distance_matrix=False,
                                 is_copolymer=False):
        '''
        FOLDER_PATH : path of the directory that contains three files:
        atom_names_rdf.txt : file that contains the atom names in the system
        xyz_unwrapped_rdf_all.txt : file that contains the unwrapped xyz coordinates of the system
        box_len.txt : file that contains the box length of the system
        timestep : time step of the simulation (in ps). This is not used in the analysis.
        Everything is based on frames. Later on, in the post processing, the timescales can be
        adjusted using this information. Just for recording.
        stepskip : how many frames to skip.
        correct_com : which method of center of mass correction to us. either "solvent", "baro", or "salt_position"
        "solvent" cnetered is the default choice.
        coordinators : a list of str elements that can coordinate the cation, either from anions or polymers.
        anion_species : the same as coordinators. to be deprecated and replaced by coordinators.
        anion : a str among "PF6", "FSI", "TFSI"
        mode : method to determine solvation environment, either convex or hard distance cutoff
        cutoff : distance cutoff for hard distance cutoff (not used if mode == "convex")
        later on, implement a function to compute RDF, detect 1st minimum distance and use that value.
        '''
        # FOLDER_PATH = "../featurize/actual_25_correct_frame"
        #FOLDER_PATH = "/Volumes/PROGRADE/353K_simulations_arash/2.25"
        # FOLDER_PATH = "/home/pleon/projects/collaborations/tri/vacancy_count/tfsi/0.75"

        # xyz_wrapped_rdf_beginning = first 100 frames, middle = middle 100 frames, end = final 100 frames
        self.mode = mode
        self.cutoff = cutoff
        self.sampling_freq = timestep * stepskip
        self.coordinators = [Element(i) for i in coordinators]
        self.anion = anion
        self.compute_distance_matrix = compute_distance_matrix
        self.is_copolymer = is_copolymer
        xyz_rdf, atom_names, atom_names_long_rdf, res_ids = read_xyz(
            self.PATH, "atom_names_rdf_all.txt", "xyz_unwrapped_rdf_all.txt",
            include_resid=True
        )
        if is_copolymer:
            monomer_unit_ids = read_xyz_monomer_unit(
                self.PATH, "atom_names_all.rdf.txt"
            )
        else:
            monomer_unit_ids = None
        box_len = float(open(f"{self.PATH}/box_len.txt", "r").read())

        xyz_rdf_frac = xyz_rdf/box_len # change to frac_coords
        xyz_rdf_frac = np.insert(xyz_rdf_frac, 0, xyz_rdf_frac[0], axis=0) 
        dp = xyz_rdf_frac[1:, ] - xyz_rdf_frac[:-1, ] # getting frac_disps
        dp = dp - np.round(dp) # unwrapping frac_disps
        dp = dp * box_len # changing back to real coords
        f_disp = np.cumsum(dp, axis=0) 
        xyz_rdf = xyz_rdf[0] + f_disp #* cell_dim[0][0]
        print('atom_names', atom_names, len(atom_names))
        print('atom_names_long_rdf', atom_names_long_rdf, len(atom_names_long_rdf))
        print('res_ids', res_ids, len(res_ids))
        self.atom_names_long_rdf = atom_names_long_rdf
        self.atom_names = atom_names
        self.types_rdf = [i.split('-')[1] for i in atom_names_long_rdf]
        unique_type_list = np.unique(self.types_rdf)
        unique_types = {"polymer": [i for i in unique_type_list if i.startswith("PL") or i.startswith("P")], "anion": [i for i in unique_type_list if i.startswith("AN")], "cation": [i for i in unique_type_list if i.startswith("CA")]}
        unique_types = {"polymer": list(set(unique_types["polymer"])), "anion": list(set(unique_types["anion"])), "cation": list(set(unique_types["cation"]))}
        self.unique_types = unique_types

        
        if len(unique_types["cation"]) > 1:
            raise NotImplementedError("Multi-cation simulations not implemented")

        if len(unique_types["anion"]) > 1:
            raise NotImplementedError("Multi-anion simulations not implemented")
        if anion not in ["PF6", "FSI", "TFSI"]:
            raise ValueError(f"Anion {anion} not implemented")

        for i in atom_names_long_rdf:
            if i.endswith("CA1"):
                self.mobile_cation = i.split("-")[0] # This should be things like "Li"
                break
        # for a given AN1 and fixed resid, what's the composition of that molecule? 
        # if that molecule is PF6, then anion_tracker = "AN-P"
        # if that molecule is FSI or TFSI, then anion_tracker = "N"
        anion_tracking_dict = {"PF6":"P", "FSI":"N", "TFSI":"N"}
        atom_name_long_anion = f"{anion_tracking_dict[anion]}-{list(unique_types['anion'])[0]}" # e.g. "N-AN1"
        atom_name_long_cation = f"{self.mobile_cation}-{list(unique_types['cation'])[0]}" # e.g. "Li-CA1"


        # center of mass correction
        xyz_rdf = xyz_rdf[::stepskip, ...] # Slicing by every "stepskip" frames
        """
        After this, all of the frames are skipped.
        This is the only part where "stepskip" is used.
        """

        if correct_com == "salt_position":
            relevant_ions = []
            for index, i in enumerate(atom_names):
                if i == self.mobile_cation or atom_names_long_rdf[index] == atom_name_long_anion:
                    relevant_ions.append(True)
                else:
                    relevant_ions.append(False)
            print(sum(np.array(relevant_ions)))
            for frame_index, sliced_frame in enumerate(xyz_rdf):
                print(xyz_rdf.shape, sliced_frame.shape)
                com = np.mean(sliced_frame[relevant_ions], axis=0)
                xyz_rdf[frame_index] = xyz_rdf[frame_index] - com

        elif correct_com == "baro":
            mass_list = [Element(i).atomic_mass for i in atom_names]
            #print(sum(np.array(relevant_ions)))
            for frame_index, sliced_frame in enumerate(xyz_rdf):
                print(xyz_rdf.shape, sliced_frame.shape)
                com = np.sum(sliced_frame*np.array(mass_list)[:,np.newaxis], axis=0)/np.sum(np.array(mass_list)) # fixed from np.mean -> np.sum
                #com = np.mean(global_time_frame)
                xyz_rdf[frame_index] = xyz_rdf[frame_index] - com   
        elif correct_com == "solvent":
            mass_list = [Element(i).atomic_mass for i in atom_names]
            # determine solvent
            """
            res_ids_dict = {}
            for i in np.unique(np.array(res_ids)):
                res_ids_dict[i] = 0
            for i in np.array(res_ids):
                res_ids_dict[i] += 1
            solvent_res_id = [i for i in res_ids_dict.keys() if res_ids_dict[i] > 15]
            # check this...
            relevant_ions = []
            for index, i in enumerate(res_ids):
                if i in solvent_res_id:
                    relevant_ions.append(True)
                else:
                    relevant_ions.append(False)
            """
            relevant_ions = []
            for index, i in enumerate(self.types_rdf):
                if i in unique_types["polymer"]:
                    relevant_ions.append(True)
                else:
                    relevant_ions.append(False)
            print(sum(np.array(relevant_ions)), len(relevant_ions))

            for frame_index, sliced_frame in enumerate(xyz_rdf):
                print(xyz_rdf.shape, sliced_frame.shape)
                val = sliced_frame*np.array(mass_list)[:,np.newaxis]
                com = np.sum(val[relevant_ions], axis=0)/(np.array(mass_list)[relevant_ions].sum())
                xyz_rdf[frame_index] = xyz_rdf[frame_index] - com
                
        atom_names_long_rdf = np.array(atom_names_long_rdf)


        EACH_PART_COUNT = int(len(xyz_rdf)/N_PART)+1
        START_INDEX = EACH_PART_COUNT * PART_INDEX
        if PART_INDEX == N_PART-1:
            END_INDEX = len(xyz_rdf)
        else:
            END_INDEX = EACH_PART_COUNT * (PART_INDEX+1)
        print(f"START_INDEX: {START_INDEX}, END_INDEX: {END_INDEX}, EACH_PART_COUNT: {EACH_PART_COUNT}")

        def get_structure_list(atom_names_rdf, xyz_rdf, box_dim):
            '''
            atom_names_rdf: list of atom names
            xyz_rdf: position of atoms
            box_dim: integer number for a cubic sized box in Angstrom
            returns a list of pymatgen structure objects.
            deprecatd because it is way too memory intesive.
            Note: the index of atoms are not modified.
            '''
            if type(box_dim)==float:
                # cubic box
                lattice = Lattice.cubic(box_dim)
            else:
                # non-cubic box
                lattice = Lattice(box_dim)

            structure_list = []
            for i in range(len(xyz_rdf)):
                structure_list.append(Structure(lattice= lattice , species=atom_names_rdf, coords=xyz_rdf[i], coords_are_cartesian=True))
            return structure_list
        def get_initial_structure(atom_names_rdf, xyz_rdf, box_dim):
            '''
            atom_names_rdf: list of atom names
            xyz_rdf: position of atoms
            box_dim: integer number for a cubic sized box in Angstrom
            returns pymatgen structure object of the first snapshot.

            Note: the index of atoms are not modified.
            '''
            if type(box_dim)==float:
                # cubic box
                lattice = Lattice.cubic(box_dim)
            else:
                # non-cubic box
                lattice = Lattice(box_dim)
            structure_init  = Structure(lattice= lattice , species=atom_names_rdf, coords=xyz_rdf[0], coords_are_cartesian=True)
            structure_final = Structure(lattice= lattice , species=atom_names_rdf, coords=xyz_rdf[-1], coords_are_cartesian=True)
            return structure_init, structure_final
        def map_res_id_to_polymer_or_anion(res_ids):
            '''
            res_ids: a list of res_ids
            returns a function that determines if the res_id is a polymer, TFSI, or Li
            for now, only works for TFSI (num_atom = 15), but can be improved to work for other anions by taking it as an argument
            '''

            """
            mapper = {}
            for i in res_ids:
                if i in mapper.keys():
                    mapper[i] +=1
                else:
                    mapper[i] = 1
            res_id_mapper = {}
            for i in mapper.keys():
                if mapper[i] > 15: # polymer, not TFSI
                    res_id_mapper[i] = "polymer"
                elif mapper[i] == 15: # TFSI
                    res_id_mapper[i] = "TFSI"
                elif mapper[i] == 1: # Li
                    res_id_mapper[i] = "Li"
            """
            #def foo(x):
            #    return res_id_mapper[x]
            res_id_mapper = {}
            for index, i in enumerate(self.types_rdf):
                if i in unique_types["polymer"]:
                    res_id_mapper[res_ids[index]] = "polymer"
                elif i in unique_types["anion"]:
                    res_id_mapper[res_ids[index]] = "anion"
                elif i in unique_types["cation"]:
                    res_id_mapper[res_ids[index]] = "cation"
            return res_id_mapper

        def extract_coordination_environments_hard(s, Li_index, global_time_index, cutoff):
            """
            s : pymatgen structure
            Li_index : index of lithium
            cutoff : distance cutoff for hard distance cutoff method
            """
            site = s[Li_index]
            nlist = s.get_neighbors(site, cutoff)
            nlist.sort(key=lambda x:x.nn_distance)
            nlist_O = [i for i in nlist if i.specie in self.coordinators]
            each_env_info = {'cartesian_position':site.coords,
                             'global_time':global_time_index,
                             'cutoff':max([i.nn_distance for i in nlist_O]),
                             'num_neighbor':len(nlist_O),
                             'neighborlist':nlist_O}


            O_providers = {}
            for jindex, j in enumerate(each_env_info['neighborlist']):
                if self.res_ids[j.index] in O_providers.keys(): # self.res_ids takes in an atom index, and gives the res_id of that atom
                    # important note:
                    # here, I am appending the sequential index of the anion, not the true index within
                    # the supercell
                    #O_providers[self.res_ids[j.index]].append(j.index)
                    O_providers[self.res_ids[j.index]].append(self.anion_indices_forward[j.index]) # this problem
                else:
                    #O_providers[self.res_ids[j.index]] = [j.index]
                    O_providers[self.res_ids[j.index]] = [self.anion_indices_forward[j.index]] # this problem.
            for j in O_providers.keys():
                O_providers[j].sort()
            each_env_info['O_providers'] = O_providers
            each_env_info.pop('neighborlist') # this is memory-intensive.
            each_env_info['neighborset'] = set(each_env_info['O_providers'].keys())

            return each_env_info

        def extract_coordination_environments(s, Li_index, global_time_index):
            """
            s : pymatgen structure
            Li_index : index of lithium
            """
            site = s[Li_index]
            nlist = s.get_neighbors(site, 6)
            nlist.sort(key=lambda x:x.nn_distance)
            nlist_O = [i for i in nlist if i.specie in self.coordinators]
            H_plus_coordinators = self.coordinators + [Element("H")]
            nlist_non_H_O = [i for i in nlist if i.specie not in H_plus_coordinators]
            nearest_non_H_O = nlist_non_H_O[0]
            convex_nlist_O = nlist_O[:4]
            for neighb_index, neighb in enumerate(nlist_O[4:]):
                og_convex = ConvexHull([i.coords for i in convex_nlist_O])
                new_convex = ConvexHull([i.coords for i in convex_nlist_O] + [neighb.coords])
                new_non_H_convex = ConvexHull([i.coords for i in convex_nlist_O] + [nearest_non_H_O.coords] + [neighb.coords])
                exist_non_H_O = False
                for k in nlist_non_H_O:
                    if k.nn_distance < neighb.nn_distance:
                        exist_non_H_O = True
                if (len(new_convex.vertices) == len(og_convex.vertices) + 1
                    and nearest_non_H_O.coords in new_non_H_convex.points[new_non_H_convex.vertices]
                    and len(new_convex.vertices) != len(new_non_H_convex.vertices)
                    and neighb.nn_distance < 1.75 * convex_nlist_O[0].nn_distance):
                    convex_nlist_O.append(neighb)
                else:
                    for i in convex_nlist_O:
                        i.label = self.res_ids[i.index] # decorating pymatgen Neighbor object by labeling them by which res_id this oxygen comes from
                    each_env_info = {'cartesian_position': site.coords, 
                                        'global_time':global_time_index, 
                                        'cutoff':max([i.nn_distance for i in convex_nlist_O]), 
                                        'num_neighbor':len(convex_nlist_O), 
                                        'neighborlist':convex_nlist_O}
                    break
                if neighb_index == len(nlist_O[4:])-1:
                    for i in convex_nlist_O:
                        i.label = self.res_ids[i.index] # decorating pymatgen Neighbor object by labeling them by which res_id this oxygen comes from
                    each_env_info = {'cartesian_position': site.coords, 
                                        'global_time':global_time_index, 
                                        'cutoff':max([i.nn_distance for i in convex_nlist_O]), 
                                        'num_neighbor':len(convex_nlist_O), 
                                        'neighborlist':convex_nlist_O}
            
            O_providers = {}
            for jindex, j in enumerate(each_env_info['neighborlist']):
                if self.res_ids[j.index] in O_providers.keys(): # self.res_ids takes in an atom index, and gives the res_id of that atom
                    # important note:
                    # here, I am appending the sequential index of the anion, not the true index within
                    # the supercell
                    #O_providers[self.res_ids[j.index]].append(j.index)
                    O_providers[self.res_ids[j.index]].append(self.anion_indices_forward[j.index]) # this problem
                else:
                    #O_providers[self.res_ids[j.index]] = [j.index]
                    O_providers[self.res_ids[j.index]] = [self.anion_indices_forward[j.index]] # this problem.
            for j in O_providers.keys():
                O_providers[j].sort()
            each_env_info['O_providers'] = O_providers


            if self.is_copolymer:
                solvating_types = {}
                for jindex, j in enumerate(each_env_info['neighborlist']):
                    if self.monomer_unit_ids[j.index] in solvating_types.keys():
                        solvating_types[self.monomer_unit_ids[j.index]].append(j.index)
                    else:
                        solvating_types[self.monomer_unit_ids[j.index]] = [j.index]
                for j in solvating_types.keys():
                    solvating_types[j].sort()
                each_env_info['solvating_types'] = solvating_types

            else:
                pass

            each_env_info.pop('neighborlist') # this is memory-intensive.
            each_env_info['neighborset'] = set(each_env_info['O_providers'].keys())
            return each_env_info

        def create_sequential_anion_mapping(self):
            """
            take in self.structure, identify the anion indices, and return a mapping of anion indices to sequential integer.
            This is necessary to identify sequential anion segments in a PEO chain.
            """
            anion_indices_forward = {}
            anion_indices_back = {}
            anion_indices = []
            for site_index, site in enumerate(self.structure):
                if site.specie in self.coordinators:
                    anion_indices.append(site_index)
            for index, site_index in enumerate(anion_indices):
                anion_indices_forward[site_index] = index
                anion_indices_back[index] = site_index
            assert(len(anion_indices) > 0)
            return anion_indices_forward, anion_indices_back


        #slist = get_structure_list(atom_names, xyz_rdf, box_len)
        #self.structurelist = slist
        self.res_ids = res_ids
        self.monomer_unit_ids = monomer_unit_ids
        self.res_id_mapper = map_res_id_to_polymer_or_anion(self.res_ids)
        #print(f"Read {len(slist)} structures from {self.PATH}")
        #self.structure = self.structurelist[0]
        self.structure_init, self.structure_final = get_initial_structure(atom_names, xyz_rdf, box_len)
        self.structure_init.to(filename="polymer_test_init.vasp", fmt="poscar")
        self.structure_final.to(filename="polymer_test_final.vasp", fmt="poscar")
        self.structure = self.structure_init
        self.boxdim = self.structure.lattice.abc
        self.anion_indices_forward, self.anion_indices_back = create_sequential_anion_mapping(self)



        if analyze_anions:
            anion_center_indices = []
            for index, i in enumerate(self.atom_names_long_rdf):
                if i == atom_name_long_anion:
                    anion_center_indices.append(index)
            #for site_index, site in enumerate(self.structure):
            #    if site.specie == atom_name_long_anion:
            #        anion_center_indices.append(site_index)
            per_anion_data = {}
            for anion_index in anion_center_indices:
                per_anion_data[self.res_ids[anion_index]] = {"global_times":[], "disp":[]}
            

        Li_indices = []
        #envinfo = {}

        # new data types
        per_Li_data = {}
        for site_index, site in enumerate(self.structure):
            if site.specie == Element(self.mobile_cation):
                Li_indices.append(site_index)
                per_Li_data[site_index] = {
                    "cartesian_positions": [],
                    "global_times": [],
                    "cutoffs": [],
                    "num_neighbors": [],
                    "flattened_O_providers": [],
                    "flattened_solvating_types": [],
                    "flattened_neighborset": [],
                }

        if type(box_len)==float:
            # cubic box
            lattice = Lattice.cubic(box_len)
        else:
            # non-cubic box
            lattice = Lattice(box_len)
        s = self.structure
        if self.compute_distance_matrix:
            distance_matrix = [] # this tracks the distance matrix for each frame.
        for frame_index, frame in enumerate(xyz_rdf):
            # complete_coord_info: START_INDEX ~ END_INDEX-1 (END_INDEX is covered in the next round, TOTAL_INDEX is length of the trajectory)
            # anion_displacement: (1-0), (2-1), ... (END_INDEX-(END_INDEX-1))
            if frame_index < START_INDEX: # not current part
                continue
            if frame_index == END_INDEX: # not current part
                s = Structure(lattice= lattice , species=atom_names, coords=xyz_rdf[frame_index], coords_are_cartesian=True)
                #s_copy = s.copy()
                #s_copy.remove_species([i for i in s_copy.species if i not in [Element("Li"), Element("N")]])
                #distance_matrix.append(s_copy.distance_matrix)
                for anion_index in anion_center_indices:
                    #disp = self.wrap_disp(s[anion_index].coords - s_prev[anion_index].coords)
                    disp = s[anion_index].coords - s_prev[anion_index].coords
                    each_anion_disp = {"global_time":(frame_index-1), "disp":disp}
                    per_anion_data[self.res_ids[anion_index]]['global_times'].append(each_anion_disp['global_time'])
                    per_anion_data[self.res_ids[anion_index]]['disp'].append(each_anion_disp['disp'])
                # everything done here.
                break
            s = Structure(lattice= lattice , species=atom_names, coords=xyz_rdf[frame_index], coords_are_cartesian=True)
            mobile_indices = [index for index, i in enumerate(atom_names_long_rdf) if i == atom_name_long_anion or i == atom_name_long_cation]
            # compute distance matrix for lithium and anion centers.
            s_copy = s.copy()
            #s_copy.remove_species([i for i in s_copy.species if i not in [Element("Li"), Element("N")]])
            s_copy.remove_sites([i for i in range(len(s_copy)) if i not in mobile_indices])
            if self.compute_distance_matrix:
                distance_matrix.append(s_copy.distance_matrix)
            # compute coordination environments for lithium.
            for Li_index in Li_indices:
                if self.mode == "convex":
                    each_env_info = extract_coordination_environments(s, Li_index, frame_index)
                elif self.mode == "hard":
                    each_env_info = extract_coordination_environments_hard(s, Li_index, frame_index, self.cutoff)
                
                
                # ... get each_env_info as before ...
                per_Li_data[Li_index]["cartesian_positions"].append(each_env_info['cartesian_position'])
                per_Li_data[Li_index]["global_times"].append(each_env_info['global_time'])
                per_Li_data[Li_index]["cutoffs"].append(each_env_info['cutoff'])
                per_Li_data[Li_index]["num_neighbors"].append(each_env_info['num_neighbor'])

                for res_id, O_indices in each_env_info['O_providers'].items():
                    for O_index in O_indices:
                        per_Li_data[Li_index]["flattened_O_providers"].append([frame_index, res_id, O_index])
                for monomer_id, O_indices in each_env_info['solvating_types'].items():
                    for O_index in O_indices:
                        per_Li_data[Li_index]["flattened_solvating_types"].append([frame_index, monomer_id, O_index])
                for res_id in each_env_info['neighborset']:
                    per_Li_data[Li_index]["flattened_neighborset"].append([frame_index, res_id])
            if frame_index == START_INDEX:
                s_prev = s
                continue
            else:
                for anion_index in anion_center_indices:
                    #disp = self.wrap_disp(s[anion_index].coords - s_prev[anion_index].coords)
                    disp = s[anion_index].coords - s_prev[anion_index].coords
                    each_anion_disp = {"global_time":(frame_index-1), "disp":disp}
                    per_anion_data[self.res_ids[anion_index]]['global_times'].append(each_anion_disp['global_time'])
                    per_anion_data[self.res_ids[anion_index]]['disp'].append(each_anion_disp['disp'])
                s_prev = s
        if self.compute_distance_matrix:
            self.distance_matrix = np.stack(distance_matrix)
        else:
            self.distance_matrix = None


        # Li_index already in the key information, unnecessary overload.        
        #for index, Li_index in enumerate(envinfo):
        #    for j in range(len(envinfo[Li_index])):
        #        envinfo[Li_index][j]['Li_index'] = Li_index

        self.complete_coord_info = {}
        for Li_index, data in per_Li_data.items():
            self.complete_coord_info[Li_index] = {
                "cartesian_positions": np.array(data["cartesian_positions"]),
                "global_times": np.array(data["global_times"]),
                "cutoffs": np.array(data["cutoffs"]),
                "num_neighbors": np.array(data["num_neighbors"]),
                "flattened_O_providers": np.array(data["flattened_O_providers"], dtype=int) if data["flattened_O_providers"] else np.empty((0,3), dtype=int),
                "flattened_solvating_types": np.array(data["flattened_solvating_types"], dtype=int) if data["flattened_solvating_types"] else np.empty((0,3), dtype=int),
                #"flattened_neighborset": np.array(data["flattened_neighborset"], dtype=int) if data["flattened_neighborset"] else np.empty((0,2), dtype=int),
            }
        for anion_index, data in per_anion_data.items():
            per_anion_data[anion_index] = {
                "global_times": np.array(data["global_times"]),
                "disp": np.array(data["disp"]),
            }
        self.anion_displacements = per_anion_data

        return 



    def wrap_disp(self, disp):
        frac_disp = np.array(disp)/np.array(self.boxdim)
        rv = []
        for d in frac_disp:
            if d > 0.5:
                rv.append(d-1)
            elif d < -0.5:
                rv.append(d+1)
            else:
                rv.append(d)
        return np.array(rv)*np.array(self.boxdim)
    """
    def get_drift_corrected_structurelist(self):
        _p, _lattices = [], []
        for s in self.structurelist:
            _p.append(np.array(s.frac_coords)[:, None])
            _lattices.append(s.lattice.matrix)
        _p.insert(0, _p[0])
        _lattices.insert(0, _lattices[0])
        p = np.concatenate(_p, axis=1)
        dp = p[:, 1:] - p[:, :-1]
        dp = dp - np.round(dp)
        f_disp = np.cumsum(dp, axis=1)
        c_disp = []
        for i in f_disp:
            c_disp.append([np.dot(d, m) for d, m in zip(i, _lattices[1:])])
        disp = np.array(c_disp) #cartesian displacements
        drift = np.average(disp, axis=0) # averaging over all atoms the drift values.
        drift_corrected_structurelist = []
        for drift_each_time in drift:
            drift_corrected_structurelist.append(Structure(lattice=s.lattice, species=s.species, coords=[site.coords-drift_each_time for site in self.structure], coords_are_cartesian=True))
        self.drift_corrected_structurelist = drift_corrected_structurelist
        return
    """



    def extract_anion_displacements(self, drift_corrected=True):
        # deprecated!!!!!!!!
        '''
        track the position of nitrogen of each TFSI, and compute the displacement of it at each time step.
        Rather than N-index, better to track based on the res_id? or using the N-index, I can always retrieve res_ids. No problem.
        deprecated. read_structure takes care of everything now.
        '''
        anion_center_indices = []
        for site_index, site in enumerate(self.structure):
            if site.specie == Element('N'):
                anion_center_indices.append(site_index)
                nitrogen_site = site
        if drift_corrected:
            target_slist = self.drift_corrected_structurelist
        else:
            target_slist = self.structurelist
        anion_displacements = {}
        for anion_index in anion_center_indices:
            each_anion_disp = []
            for global_time, structure in enumerate(target_slist):
                if global_time == len(target_slist)-1:
                    continue
                disp = self.wrap_disp(target_slist[global_time+1][anion_index].coords - target_slist[global_time][anion_index].coords)
                each_anion_disp.append({"global_time":global_time, "disp":disp})
            anion_displacements[self.res_ids[anion_index]] = each_anion_disp
        self.anion_displacements = anion_displacements
        return anion_displacements





        def get_structure_list(atom_names_rdf, xyz_rdf, box_dim):
            '''
            atom_names_rdf: list of atom names
            xyz_rdf: position of atoms
            box_dim: integer number for a cubic sized box in Angstrom
            returns a list of pymatgen structure objects.
            deprecatd because it is way too memory intesive.
            Note: the index of atoms are not modified.
            '''
            if type(box_dim)==float:
                # cubic box
                lattice = Lattice.cubic(box_dim)
            else:
                # non-cubic box
                lattice = Lattice(box_dim)

            structure_list = []
            for i in range(len(xyz_rdf)):
                structure_list.append(Structure(lattice= lattice , species=atom_names_rdf, coords=xyz_rdf[i], coords_are_cartesian=True))
            return structure_list


    def extract_coordination_environments_deprecated(self, save_memory=True, drift_corrected=True):
        '''
        structurelist : list of pymatgen structure objects

        '''
        Li_indices = []
        for site_index, site in enumerate(self.structure):
            if site.specie == Element("Li"):
                Li_indices.append(site_index)
        envinfo = {}

        if drift_corrected:
            target_slist = self.drift_corrected_structurelist
        else:
            target_slist = self.structurelist

        for Li_index in Li_indices:
            each_env_info = []
            for sindex, s in enumerate(target_slist):
                site = s[Li_index]
                nlist = s.get_neighbors(site, 6)
                nlist.sort(key=lambda x:x.nn_distance)
                nlist_O = [i for i in nlist if i.specie == Element("O")]
                nlist_non_H_O = [i for i in nlist if i.specie not in [Element("H"), Element("O")]]
                nearest_non_H_O = nlist_non_H_O[0]
                convex_nlist_O = nlist_O[:4]
                for neighb_index, neighb in enumerate(nlist_O[4:]):
                    og_convex = ConvexHull([i.coords for i in convex_nlist_O])
                    new_convex = ConvexHull([i.coords for i in convex_nlist_O] + [neighb.coords])
                    new_non_H_convex = ConvexHull([i.coords for i in convex_nlist_O] + [nearest_non_H_O.coords] + [neighb.coords])
                    exist_non_H_O = False
                    for k in nlist_non_H_O:
                        if k.nn_distance < neighb.nn_distance:
                            exist_non_H_O = True
                    if len(new_convex.vertices) == len(og_convex.vertices) + 1 and len(new_convex.vertices) != len(new_non_H_convex.vertices) and neighb.nn_distance < 1.75 * convex_nlist_O[0].nn_distance:
                        convex_nlist_O.append(neighb)
                    else:
                        for i in convex_nlist_O:
                            i.label = self.res_ids[i.index] # decorating pymatgen Neighbor object by labeling them by which res_id this oxygen comes from
                        each_env_info.append({'cartesian_position': site.coords, 
                                            'global_time':sindex, 
                                            'cutoff':max([i.nn_distance for i in convex_nlist_O]), 
                                            'num_neighbor':len(convex_nlist_O), 
                                            'neighborlist':convex_nlist_O})
                        break
                    if neighb_index == len(nlist_O[4:])-1:
                        for i in convex_nlist_O:
                            i.label = self.res_ids[i.index] # decorating pymatgen Neighbor object by labeling them by which res_id this oxygen comes from
                        each_env_info.append({'cartesian_position': site.coords, 
                                            'global_time':sindex, 
                                            'cutoff':max([i.nn_distance for i in convex_nlist_O]), 
                                            'num_neighbor':len(convex_nlist_O), 
                                            'neighborlist':convex_nlist_O})

            envinfo[Li_index] = each_env_info


            for Li_index in envinfo.keys():
                for index, i in enumerate(envinfo[Li_index]):
                    # Iterating through snapshots
                    O_providers = {}
                    for jindex, j in enumerate(i['neighborlist']):
                        if self.res_ids[j.index] in O_providers.keys():
                            O_providers[self.res_ids[j.index]].append(j.index)
                        else:
                            O_providers[self.res_ids[j.index]] = [j.index]
                    for j in O_providers.keys():
                        O_providers[j].sort()
                    envinfo[Li_index][index]['O_providers'] = O_providers
                    
            for index, Li_index in enumerate(envinfo):
                for j in range(len(envinfo[Li_index])):
                    envinfo[Li_index][j]['Li_index'] = Li_index


        for Li_index in envinfo.keys():
            for index, i in enumerate(envinfo[Li_index]):
                # Iterating through snapshots
                envinfo[Li_index][index]['neighborset'] = set(envinfo[Li_index][index]['O_providers'].keys())
        self.complete_coord_info = envinfo
        if save_memory:
            self.structurelist = None
        return envinfo


    def get_total_entities(self):
        '''
        complete_coord_info : Li_index : [{snapshots}]
        For each lithium, I keep a list of current entities, and a list of archived entities.
        When a current entity leaves the coordinating shell for the specific lithium,
        then it is moved to the archived entities.
        return data: key : Li_index, value : list of entities. each entity is a sequence of consecutive times
        when a specific res_id is present in the coordinating shell of the specific Li
        '''
        total_entities = {}
        for Li_index in self.complete_coord_info.keys():
            archived_entities = [] # {res_id : {'coord_list':[], 'Li_positions':[], 'times':[]}}
            current_entities = {}
            for index, i in enumerate(self.complete_coord_info[Li_index]):
                if index == 0:
                    for j in i['O_providers'].keys():
                        # j : res_id
                        current_entities[j] = {}
                        current_entities[j]['coord_list'] = [i['O_providers'][j]]
                        current_entities[j]['Li_positions'] = [i['cartesian_position']]
                        current_entities[j]['global_time'] = [i['global_time']]
                        #current_entities[j]['Li_index'] = [i['Li_index']] Li index isn't being used.
                # Iterating through snapshots

                # here
                for j in i['O_providers'].keys():
                    if j in current_entities.keys():
                        # existing entity, add the next O-index list
                        current_entities[j]['coord_list'].append(i['O_providers'][j])
                        current_entities[j]['Li_positions'].append(i['cartesian_position'])
                        current_entities[j]['global_time'].append(i['global_time'])
                        #current_entities[j]['Li_index'].append(i['Li_index'])
                    else:
                        # new entity, create a new entity
                        current_entities[j] = {}
                        current_entities[j]['coord_list'] = [i['O_providers'][j]]
                        current_entities[j]['Li_positions'] = [i['cartesian_position']]
                        current_entities[j]['global_time'] = [i['global_time']]
                        #current_entities[j]['Li_index'] = [i['Li_index']]

                # iterate thru existing entities to detect disappeared keys, move to archived entities.
                to_be_popped = []
                for j in current_entities.keys():
                    if j not in i['O_providers'].keys(): # these are res_ids
                        archived_entities.append({j: current_entities[j]})
                        to_be_popped.append(j)
                for j in to_be_popped:
                    current_entities.pop(j)

            # now we've iterated through all trajectory; archive all of the current entities.
            for j in current_entities.keys():
                archived_entities.append({j: current_entities[j]})
            total_entities[Li_index] = archived_entities
        self.total_entities = total_entities
        return total_entities


    def get_total_entities_segment(self):
        # not implemented.

        '''
        complete_coord_info : Li_index : [{snapshots}]
        For each lithium, I keep a list of current entities, and a list of archived entities.
        When a current entity leaves the coordinating shell for the specific lithium,
        then it is moved to the archived entities.
        '''
        total_entities = {}
        for Li_index in self.complete_coord_info.keys():
            archived_entities = [] # {res_id : {'coord_list':[], 'Li_positions':[], 'times':[]}}
            current_entities = {}
            for index, i in enumerate(self.complete_coord_info[Li_index]):
                if index == 0:
                    for j in i['O_providers'].keys():
                        # j : res_id
                        current_entities[j] = {}
                        current_entities[j]['coord_list'] = [i['O_providers'][j]]
                        current_entities[j]['Li_positions'] = [i['cartesian_position']]
                        current_entities[j]['global_time'] = [i['global_time']]
                        #current_entities[j]['Li_index'] = [i['Li_index']] Li index isn't being used.
                # Iterating through snapshots

                # here
                for j in i['O_providers'].keys():
                    if j in current_entities.keys():
                        # existing entity, add the next O-index list
                        current_entities[j]['coord_list'].append(i['O_providers'][j])
                        current_entities[j]['Li_positions'].append(i['cartesian_position'])
                        current_entities[j]['global_time'].append(i['global_time'])
                        #current_entities[j]['Li_index'].append(i['Li_index'])
                    else:
                        # new entity, create a new entity
                        current_entities[j] = {}
                        current_entities[j]['coord_list'] = [i['O_providers'][j]]
                        current_entities[j]['Li_positions'] = [i['cartesian_position']]
                        current_entities[j]['global_time'] = [i['global_time']]
                        #current_entities[j]['Li_index'] = [i['Li_index']]

                # iterate thru existing entities to detect disappeared keys, move to archived entities.
                to_be_popped = []
                for j in current_entities.keys():
                    if j not in i['O_providers'].keys():
                        archived_entities.append({j: current_entities[j]})
                        to_be_popped.append(j)
                for j in to_be_popped:
                    current_entities.pop(j)


            for j in current_entities.keys():
                archived_entities.append({j: current_entities[j]})
            total_entities[Li_index] = archived_entities
        self.total_entities = total_entities
        return False # not implemented
        #return total_entities



    def classify_entities(self):
        """
        total_entities: output of get_total_entities
        res_id_mapping_function: function that maps res_id to polymer or anion
        returns two lists : polymer_entities, tfsi_entities
        """
        polymer_entities = {}
        anion_entities = {}
        for Li_index in self.total_entities.keys():
            Li_index_polymer = []
            Li_index_anion = []
            for j in self.total_entities[Li_index]:
                if self.res_id_mapper[list(j.keys())[0]] == 'polymer':
                    Li_index_polymer.append(j)
                elif self.res_id_mapper[list(j.keys())[0]] == 'anion':
                    Li_index_anion.append(j)
            polymer_entities[Li_index] = Li_index_polymer
            anion_entities[Li_index] = Li_index_anion
        self.polymer_entities = polymer_entities
        self.anion_entities = anion_entities
        return polymer_entities, anion_entities

    def simplify_complete_coord_info_to_states(self, skip, methods=[1]):
        """
        take in complete_coord_info. Then, build the same datastructure, but instead of O_providers and coordination indices, 
        compute the classification into different predefined states.
        skip : when determining segments, what is the max number of oxygen increments to determine a segment
        methods : a list of integers from 1 to 10.

        method 1: only TFSI, only chain, TFSI+chain (three states)
        method 2: zero-chain, 1-chain, 2-chain (or segment)
        method 3: 0-TFSI, 1-TFSI, 2-TFSI, 3 or more TFSI
        method 4 : zero segment, one segment, two segments, three or more segments
        method 5: (polymer_res_id, anion) = (1, False), (2, False), (3, False; unlikely), (1,True), (2, True), (3, True), (0, True), (0, False)
        method 6: (segment_id, anion) = (1, False), (2, False), (3, False; unlikely), (1,True), (2, True), (3, True), (0, True), (0, False)
        method 7: (polymer_res_id, num_anion_res_id)
        method 8: (segment_id, num_anion_res_id)
        method 9: (num_O_peo, num_O_anion) = (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2), (0,3), (1,3), (2,3), (3,3)...
        method 10 : num_total_coordinators (integers)
        either residue_id or segments.
        This will be used in a general "state_to_state_transitions" function.
        """
        def split_into_within_n_groups(nums, skip):
            """
            a helper function that
            splits the list of numbers into groups where the difference between the numbers is less than or equal to skip.
            the list does not have to be sorted.
            used for segment identification.
            """
            assert skip >= 1
            from collections import deque

            if not nums:
                return []

            nums = sorted(set(nums))  # Remove duplicates and sort
            visited = set()
            groups = []

            def bfs(start):
                group = []
                queue = deque([start])
                visited.add(start)
                while queue:
                    current = queue.popleft()
                    group.append(current)
                    for neighbor in nums:
                        if neighbor not in visited and 1 <= abs(neighbor - current) <= skip:
                            visited.add(neighbor)
                            queue.append(neighbor)
                return sorted(group)

            for num in nums:
                if num not in visited:
                    groups.append(bfs(num))

            return groups
        def construct_residue_dict(flattened_O_providers, time):
            unique_residues = np.unique(flattened_O_providers[(flattened_O_providers[:,0] == time)][:,1])
            rv = {}
            for i in unique_residues:
                vals = list(flattened_O_providers[(flattened_O_providers[:,0] == time) & (flattened_O_providers[:,1] == i), 2])
                vals.sort()
                rv[i] = vals
            #print(a[(a[:,0] == time)])
            return rv
        total_state_info = {}
        for Li_index in self.complete_coord_info.keys():
            #Li_state_list = []
            global_times_list = []
            cartesian_positions_list = []
            method_dict_lists = {}
            for method in methods:
                method_dict_lists[method] = []
            for timeindex in self.complete_coord_info[Li_index]['global_times']:
                coord_env = {}
                coord_env['cartesian_position'] = self.complete_coord_info[Li_index]['cartesian_positions'][timeindex]
                coord_env['global_time'] = self.complete_coord_info[Li_index]['global_times'][timeindex]
                coord_env['O_providers'] = construct_residue_dict(self.complete_coord_info[Li_index]['flattened_O_providers'], timeindex)
                current_Li_state = {}
                current_Li_state['polymer'] = {}
                current_Li_state['anion'] = {}
                current_Li_state['segments'] = {}
                # current_state : {res_id : }
                for res_id in coord_env['O_providers'].keys():
                    if self.res_id_mapper[res_id] == 'polymer':
                        current_Li_state['polymer'][res_id] = coord_env['O_providers'][res_id]
                    elif self.res_id_mapper[res_id] == 'anion':
                        current_Li_state['anion'][res_id] = coord_env['O_providers'][res_id]
                    else:
                        raise ValueError(f"Unknown res_id: {res_id, self.res_id_mapper[res_id]}")
                num_polymer = len(current_Li_state['polymer'].keys())
                num_anion = len(current_Li_state['anion'].keys())

                if num_polymer > 0:
                    for res_id in current_Li_state['polymer'].keys():
                        splits = split_into_within_n_groups(current_Li_state['polymer'][res_id], skip=skip)
                        for split_index, split in enumerate(splits):
                            segment_id = str(res_id) + "_" + str(split_index)
                            current_Li_state['segments'][segment_id] = split
                num_segments = len(current_Li_state['segments'].keys())
                num_anion_atoms = sum([len(current_Li_state['anion'][res_id]) for res_id in current_Li_state['anion'].keys()])
                num_polymer_atoms = sum([len(current_Li_state['polymer'][res_id]) for res_id in current_Li_state['polymer'].keys()])
                num_segment_atoms = sum([len(current_Li_state['segments'][segment_id]) for segment_id in current_Li_state['segments'].keys()])
                print(num_polymer, num_anion, num_polymer_atoms, num_anion_atoms)
                if (num_anion_atoms >= 3 and num_anion == 1) or (num_anion_atoms >= 5 and num_anion >= 2):
                    print(f"frame_index {timeindex}, Li_index {Li_index}")
                #print(num_polymer_atoms, num_segment_atoms)
                assert(num_polymer_atoms == num_segment_atoms)
                current_Li_state['Li_index'] = Li_index
                current_Li_state['cartesian_position'] = coord_env['cartesian_position']
                current_Li_state['global_time'] = coord_env['global_time']
                global_times_list.append(coord_env['global_time'])
                cartesian_positions_list.append(coord_env['cartesian_position'])

                # total_state_info extraction done.
                # now go through possible methods and make a methods dictionary
                method_dict = {}

                if 1 in methods:
                    """
                    method-1 : only TFSI, only chain, TFSI+chain (three states)
                    0 : only polymer
                    1 : polymer + anion
                    2 : only anion
                    3 : no coordinators (impossible)
                    possible vlues : 0,1,2,3
                    """

                    # method-1 : only anion, only chain, anion+chain (three states)
                    if num_polymer == 0 and num_anion == 0:
                        method_1_state = 3
                        warnings.warn(f"No coordinators at {Li_index}")
                    elif num_polymer == 0 and num_anion > 0:
                        method_1_state = 2
                    elif num_polymer > 0 and num_anion == 0:
                        method_1_state = 0
                    elif num_polymer > 0 and num_anion > 0:
                        method_1_state = 1
                    method_dict[1] = method_1_state
                if 2 in methods:
                    """
                    method-2 : zero-chain, 1-chain, 2-chain, 3 or more chains (not segments)
                    0 : zero chains
                    1 : one chain
                    2 : two chains
                    3 : three or more chains
                    possible_values = 0,1,2,3
                    """
                    if num_polymer == 0:
                        method_2_state = 0
                    elif num_polymer == 1:
                        method_2_state = 1
                    elif num_polymer == 2:
                        method_2_state = 2
                    elif num_polymer >= 3:
                        method_2_state = 3
                    method_dict[2] = method_2_state
                if 3 in methods:
                    """
                    method-3 : 0-anion, 1-anion, 2-anion, 3 or more anion residues (e.g. TFSI)
                    0 : zero anion
                    1 : one anion
                    2 : two anion
                    3 : three or more anion
                    possible_values = 0,1,2,3
                    """
                    if num_anion == 0:
                        method_3_state = 0
                    elif num_anion == 1:
                        method_3_state = 1
                    elif num_anion == 2:
                        method_3_state = 2
                    elif num_anion >= 3:
                        method_3_state = 3
                    method_dict[3] = method_3_state
                if 4 in methods:
                    """
                    method-4 : similar to method 2, but instead of polymer residues, we use unique segments.
                    0 : zero segments
                    1 : one segments
                    2 : two segments
                    3 : three or more segments
                    possible values: 0, 1, 2, 3
                    """
                    if num_segments == 0:
                        method_4_state = 0
                    elif num_segments == 1:
                        method_4_state = 1
                    elif num_segments == 2:
                        method_4_state = 2
                    elif num_segments >= 3:
                        method_4_state = 3
                    method_dict[4] = method_4_state
                if 5 in methods:
                    """
                    method 5: (polymer_res_id, anion) = (1, False), (2, False), (3, False; unlikely), (1,True), (2, True), (3, True), (0, True), (0, False)
                    possible_values : 2 (T/F) * 0~3 (num_Polymer_chains)
                    0,1,2,3,4,5,6,7
                    assert
                    """
                    a = num_polymer
                    if a >= 4:
                        warnings.warn(f"num_polymer >= 4 at {Li_index}, set to 3")
                        a = 3
                    method_dict[5] = a*2 + int(num_anion > 0)
                if 6 in methods:
                    """
                    method 6: (segment_id, anion) = (1, False), (2, False), (3, False; unlikely), (1,True), (2, True), (3, True), (0, True), (0, False)
                    possible_values : 2 (T/F) * 4 (num_segments)
                    0,1,2,3,4,5,6,7
                    """
                    a = num_segments
                    if a >= 4:
                        warnings.warn(f"num_segments >= 4 at {Li_index}, set to 3")
                        a = 3
                    method_dict[6] = a*2 + int(num_anion > 0)
                if 7 in methods:
                    """
                    method 7: (polymer_res_id, num_O_anion) = (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2), (0,3), (1,3), (2,3), (3,3)...
                    max number of peo : 0~3
                    max number of anion : 0~3
                    possible values : 0~15 (16 values)
                    """
                    a, b = num_polymer, num_anion
                    assert(num_anion <= 3)
                    assert(num_polymer <= 3)
                    if a >= 4:
                        warnings.warn(f"num_polymer >= 4 at {Li_index}, set to 3")
                        a = 3
                    if b >= 4:
                        warnings.warn(f"num_anion >= 4 at {Li_index}, set to 3")
                        b = 3
                    method_dict[7] = a*4 + b
                if 8 in methods:
                    """
                    method 8: (segment_id, num_O_anion) = (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2), (0,3), (1,3), (2,3), (3,3)...
                    possible values : 0~15 (16 values)
                    max number of anion res_id : 0~3
                    max_number of segment : 0~3
                    """
                    a, b = num_segments, num_anion
                    if a >= 4:
                        warnings.warn(f"num_segments >= 4 at {Li_index}, set to 3")
                        a = 3
                    if b >= 4:
                        warnings.warn(f"num_anion >= 4 at {Li_index}, set to 3")
                        b = 3
                    method_dict[8] = a*4 + b
                if 9 in methods:
                    """
                    method 9: (num_polymer_atoms, num_anion_atoms)
                    num_polymer_atoms : 0~9
                    num_anion_atoms : 0~7
                    """
                    a, b = num_polymer_atoms, num_anion_atoms
                    if a >= 10:
                        warnings.warn(f"num_polymer_atoms >= 10 at {Li_index}, set to 9")
                        a = 9
                    if b >= 8:
                        warnings.warn(f"num_anion_atoms >= 8 at {Li_index}, set to 7")
                        b = 7
                    method_dict[9] = a*8 + b
                if 10 in methods:
                    """
                    method 10: total number of coordinating atoms
                    0~14 (15 values)
                    """
                    a = num_polymer_atoms + num_anion_atoms
                    if a >= 15:
                        warnings.warn(f"num_polymer_atoms + num_anion_atoms >= 15 at {Li_index}, set to 14")
                        a = 14
                    method_dict[10] = a
                current_Li_state['method_dict'] = method_dict
                for method in method_dict_lists.keys():
                    method_dict_lists[method].append(method_dict[method])
                #Li_state_list.append(current_Li_state)
            total_state_info[Li_index] = {"global_times":np.array(global_times_list, dtype=int), 
                                          "cartesian_positions":np.array(cartesian_positions_list, dtype=float), 
                                          "method_dicts":method_dict_lists}
        self.complete_states = total_state_info
        return self.complete_states

    def state_to_state_transitions(self):
        """
        take in self.complete_states.
        identify how many states there are. Then compute transitions between them. e.g. (0-0, 0-1, 0-2, ... , (3-3)).
        assign displacements to each transition like i do for extract_entity_events.
        """
        num_states_mapper = {1:4,
                            2:4,
                            3:4,
                            4:4,
                            5:8,
                            6:8,
                            7:16,
                            8:16,
                            9:80,
                            10:15}

        # first number: polymer, second number: TFSI
        # 0 : no change, 1 : added, 2: removed, 3 : add and remove
        state_to_state_recorder_per_Li = {}
        """
        data structure:
        key : Li_index, value : dictionary of keys: method_key, values: dictionary of key : transition_code, values: list of dictionarys of {'global_time': X, "disp": disp}

        """
        for Li_index in self.complete_states.keys():
            transition_dict = {}
            for method_key in self.complete_states[Li_index]['method_dicts'].keys():
                transition_recorder = {}
                for i in range(num_states_mapper[method_key]):
                    for j in range(num_states_mapper[method_key]):
                        transition_recorder[f"{i}-{j}"] = {"global_time":[], "disp":[]}
                assert(len(transition_recorder.keys()) == num_states_mapper[method_key]**2)
                for index, i in enumerate(self.complete_states[Li_index]['global_times']):
                    if index == len(self.complete_states[Li_index]['global_times']) - 1:
                        # end of the trajectory
                        continue
                    #displacement = self.wrap_disp(self.complete_states[Li_index][index+1]['cartesian_position'] - self.complete_states[Li_index][index]['cartesian_position'])
                    displacement = self.complete_states[Li_index]['cartesian_positions'][index+1] - self.complete_states[Li_index]['cartesian_positions'][index]
                    assert (len(displacement) == 3)
                    transition_code = f"{self.complete_states[Li_index]['method_dicts'][method_key][index]}-{self.complete_states[Li_index]['method_dicts'][method_key][index+1]}"
                    transition_recorder[transition_code]["global_time"].append(index)
                    transition_recorder[transition_code]["disp"].append(displacement)
                for transition_code in transition_recorder.keys():
                    transition_recorder[transition_code]["global_time"] = np.array(transition_recorder[transition_code]["global_time"], dtype=int)
                    transition_recorder[transition_code]["disp"] = np.array(transition_recorder[transition_code]["disp"], dtype=float)
                transition_dict[method_key] = transition_recorder
                assert(len(self.complete_states[Li_index]['global_times']) - 1 == sum([len(transition_recorder[i]['global_time']) for i in transition_recorder.keys()]))

            state_to_state_recorder_per_Li[Li_index] = transition_dict
        self.state_transitions = state_to_state_recorder_per_Li
        return state_to_state_recorder_per_Li

    
    def state_populations(self):
        """
        take in self.complete_states.
        compute the probability of each state (population)
        """

    def extract_entity_events(self):
        """
        iterate through Li-index, then time. at each time, check the removed and added oxygen "residue ids"
        then, check if it is coming from anion or polymer.
        then assign that instantaneous displacement to each event recorder.
        This step requires "complete_coord_info" information to be computed from "read_structures_parallel"
        This function detects new entities entering or leaving the coordinating shell of a given lithium.
        """
        # first number: polymer, second number: TFSI
        # 0 : no change, 1 : added, 2: removed, 3 : add and remove
        event_recorder_per_Li = {}
        for Li_index in self.complete_coord_info.keys():
            event_recorder = {rc: {"global_time": [], "disp": []} for rc in [
                '0-0', '0-1', '0-2', '0-3', '1-0', '1-1', '1-2', '1-3',
                '2-0', '2-1', '2-2', '2-3', '3-0', '3-1', '3-2', '3-3'
            ]}
            cartesian_positions = self.complete_coord_info[Li_index]["cartesian_positions"]
            flattened_O_providers = self.complete_coord_info[Li_index]["flattened_O_providers"]
            n_frames = cartesian_positions.shape[0]
            for t in range(n_frames - 1):
                neighborset_t = set(flattened_O_providers[flattened_O_providers[:,0] == t, 1])
                neighborset_tp1 = set(flattened_O_providers[flattened_O_providers[:,0] == t+1, 1])
                removed = neighborset_t - neighborset_tp1
                added = neighborset_tp1 - neighborset_t
                displacement = cartesian_positions[t+1] - cartesian_positions[t]

                poly_added = False
                poly_removed = False
                poly_nochange = True
                anion_added = False
                anion_removed = False
                anion_nochange = True
                for j in removed:
                    if self.res_id_mapper[j] == 'polymer':
                        poly_removed = True
                        poly_nochange = False
                    if self.res_id_mapper[j] == 'anion':
                        anion_removed = True
                        anion_nochange = False
                for j in added:
                    if self.res_id_mapper[j] == 'polymer':
                        poly_added = True
                        poly_nochange = False
                    if self.res_id_mapper[j] == 'anion':
                        anion_added = True
                        anion_nochange = False

                # determine the polymer index (pid)
                if poly_added and poly_removed:
                    pid = 3
                if poly_added and not poly_removed:
                    pid = 1
                if not poly_added and poly_removed:
                    pid = 2
                if not poly_added and not poly_removed:
                    pid = 0
                if anion_added and anion_removed:
                    tid = 3
                if anion_added and not anion_removed:
                    tid = 1
                if not anion_added and anion_removed:
                    tid = 2
                if not anion_added and not anion_removed:
                    tid = 0
                returncode = f"{pid}-{tid}"
                event_recorder[returncode]["global_time"].append(t)
                event_recorder[returncode]["disp"].append(displacement)
            for rc in event_recorder:
                event_recorder[rc]["global_time"] = np.array(event_recorder[rc]["global_time"], dtype=int)
                event_recorder[rc]["disp"] = np.array(event_recorder[rc]["disp"], dtype=float)
            event_recorder_per_Li[Li_index] = event_recorder
        self.entitychange = event_recorder_per_Li
        return event_recorder_per_Li

    def detect_change_O_index_polymer(self):
        self.entity_O_change_polymer = self.detect_change_O_index(self.polymer_entities)
        return self.entity_O_change_polymer
    
    def detect_change_O_index_anion(self):
        self.entity_O_change_anion = self.detect_change_O_index(self.anion_entities)
        return self.entity_O_change_anion      

    def identify_const_entity_groups(self):
        '''
        similar to "extract_entity_events", but the no-change case "0-0", where the entities (set of residue_ids) stay the same.
        There would be time_indices of "0-0" where some of them are consecutive.
        The output is a dictionary of keys "Li_index", values: a list of consecutive const_entity coordination environments.
        This does not require "extract_entity_events" to be performed.
        '''
        # first number: polymer, second number: TFSI
        # 0 : no change, 1 : added, 2: removed, 3 : add and remove
        # initial part can be heavily trimmed 
        return_dict = {}

        for Li_index in self.entitychange.keys():
            event_recorder = self.entitychange[Li_index]
            def group_adjacent_integers(lst):
                '''
                must be converted from np.array to list first.
                For example, take lst = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 15]
                returns  [[1, 2, 3], [5, 6, 7], [9, 10, 11, 12], [15]]
                '''
                if not lst:
                    return []

                groups = []
                current_group = [lst[0]]

                for i in range(1, len(lst)):
                    if lst[i] == lst[i - 1] + 1:
                        current_group.append(lst[i])
                    else:
                        groups.append(current_group)
                        current_group = [lst[i]]

                groups.append(current_group)
                return groups
            grouped_const_entity = group_adjacent_integers(list(event_recorder['0-0']['global_time']))
            each_entity_list = []
            for group in grouped_const_entity:
                current_entities = {} # key : res_id, value : {'coord_list':[], 'Li_positions':[], 'times':[]}
                for time_index in self.complete_coord_info[Li_index]['global_times']:
                    
                #for time_index, coord_env in enumerate(self.complete_coord_info[Li_index]): # time enumeration
                    if time_index >= group[0] and time_index <= group[-1] + 1: 
                        # +1 needed here! so that we know what the transition is. even if 0-0 occured once, there will be
                        # before and after this event and we need those info.
                        if time_index == group[0]:
                            flattened_O_providers = self.complete_coord_info[Li_index]['flattened_O_providers']
                            residue_ids = set(flattened_O_providers[flattened_O_providers[:,0] == time_index, 1])
                            for residue_id in residue_ids:
                                # j : res_id
                                current_entities[residue_id] = {}
                                current_entities[residue_id]['coord_list'] = [flattened_O_providers[(flattened_O_providers[:,0] == time_index) & (flattened_O_providers[:,1] == residue_id), 2]]
                                current_entities[residue_id]['Li_positions'] = [self.complete_coord_info[Li_index]['cartesian_positions'][time_index]]#[coord_env['cartesian_position']]
                                current_entities[residue_id]['global_time'] = [self.complete_coord_info[Li_index]['global_times'][time_index]] #[coord_env['global_time']]
                                assert self.complete_coord_info[Li_index]['global_times'][time_index] == time_index
                                #current_entities[j]['Li_index'] = [i['Li_index']]
                        else:
                            flattened_O_providers = self.complete_coord_info[Li_index]['flattened_O_providers']
                            residue_ids = set(flattened_O_providers[flattened_O_providers[:,0] == time_index, 1])
                            for residue_id in residue_ids:
                                if residue_id in current_entities.keys():
                                    # existing entity, add the next O-index list
                                    current_entities[residue_id]['coord_list'].append(flattened_O_providers[(flattened_O_providers[:,0] == time_index) & (flattened_O_providers[:,1] == residue_id), 2])
                                    current_entities[residue_id]['Li_positions'].append(self.complete_coord_info[Li_index]['cartesian_positions'][time_index])
                                    current_entities[residue_id]['global_time'].append(self.complete_coord_info[Li_index]['global_times'][time_index])
                                    #current_entities[j]['Li_index'].append(i['Li_index'])
                                else:
                                    # new entity, create a new entity
                                    AssertionError("Bug. No new entities must appeaer during 0-0 duration.")
                #current_entities = [{j: current_entities[j]} for j in current_entities.keys()] # without this line, I get a dictionary with keys of res_ids.
                each_entity_list.append(current_entities)
            return_dict[Li_index] = each_entity_list
        self.const_entity_groups = return_dict
        return return_dict



    def classify_const_entity_events(self, mode):
        """ 
        This function is used to classify the constant entity events.
        It requires "identify_const_entity_groups" to be performed first.
        """
        if mode == "TOTAL_N_OXYGEN": # detect change of oxygen indices
            raise NotImplementedError
        
        if mode == "TOTAL_N_OXYGEN_POLYMER":
            raise NotImplementedError

        if mode == "TOTAL_N_OXYGEN_TFSI":
            raise NotImplementedError
        
        if mode == "CHANGE_OR_NOCHANGE": # 0 : no-change in any oxygen, 1 : change in any oxygen (boolean classifier)
            # easy, compare with next frame, if everything same, 0, anything change, 1
            event_recorder = {}
            for Li_index in self.const_entity_groups.keys():
                event_recorder_each_Li = {"0-Nochange":{"global_time":[], "disp":[]}, "0-Change":{"global_time":[], "disp":[]}}
                for entity_group in self.const_entity_groups[Li_index]: # entity group iteration
                    list_of_interest = list(entity_group.values())[0] # no ISSUE: Here, you can have multiple types of entities! I'm using the first one... (doesn't matter which one actually)
                    for local_time_index, _ in enumerate(list_of_interest['coord_list']): # time iteration
                        if local_time_index == len(list_of_interest['coord_list'])-1:
                            continue    
                        # just using the first entity in the group to track time
                        #Li_disp = self.wrap_disp(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])
                        Li_disp = list_of_interest['Li_positions'][local_time_index+1] - list_of_interest['Li_positions'][local_time_index]
                        #Li_disp = list(entity_group[0].values())[0]['Li_positions'][time_index+1] - list(entity_group[0].values())[0]['Li_positions'][time_index]
                    
                        no_change = True
                        for res_id in entity_group.keys(): # each entity iteration
                            now = entity_group[res_id]['coord_list'][local_time_index]
                            after = entity_group[res_id]['coord_list'][local_time_index+1]
                            if set(now) != set(after):
                                no_change = False
                                break
                        if no_change:
                            event_recorder_each_Li["0-Nochange"]["global_time"].append(list_of_interest['global_time'][local_time_index])
                            event_recorder_each_Li["0-Nochange"]["disp"].append(Li_disp)
                        else:
                            event_recorder_each_Li["0-Change"]["global_time"].append(list_of_interest['global_time'][local_time_index])
                            event_recorder_each_Li["0-Change"]["disp"].append(Li_disp)
                event_recorder[Li_index] = event_recorder_each_Li
            return event_recorder
        
        if mode == "ONLY_MAJORITY_POLYMER": # 0, 1, 2, 3 (no-change, add new, lose some, add and lose)
            # I only look at how the majority polymer evolves as a function of time.
            def get_sequential_index(index_list):
                '''convert index to sequential index (O-C-C-O-C-C chain)
                this needs to be generalized. rigiht now, we ignore hydrogen.
                To work for any type of polymer, this has to be generalized.
                '''
                return set(index_list)
                #return set([index//3 for index in index_list])
            def find_majority_polymer(entity_group, res_id_mapper):
                '''
                majority_polymer: polymer that provides most number of oxygens to the target lithium on average during the duration
                entity_group : [{res_id : {'coord_list':[], 'Li_positions':[], 'times':[], 'Li_index':[]}}]
                '''
                polymer_entities = [{res_id:entity_group[res_id]} for res_id in entity_group.keys() if res_id_mapper[res_id] == 'polymer']
                polymer_entities.sort(key=lambda x:list(x.keys())[0]) # deterministic. in case two entities have the exact same average coordinations.
                #print(f"number of polymer entities: {len(polymer_entities)} out of total {len(entity_group.keys())}")# , "number of polymers")
                if len(polymer_entities) == 0:
                    return None
                if len(polymer_entities) == 1:
                    return polymer_entities[0]
                majority_polymer = polymer_entities[0] # initializing as the first polymer
                majority_num_O = np.average([len(i) for i in list(majority_polymer.values())[0]['coord_list']])
                for entity in polymer_entities:
                    #print(np.average([len(i) for i in list(entity.values())[0]['coord_list']]))
                    if np.average([len(i) for i in list(entity.values())[0]['coord_list']]) > majority_num_O:
                        majority_num_O = np.average([len(i) for i in list(entity.values())[0]['coord_list']])
                        majority_polymer = entity
                #print("final", majority_num_O)
                return majority_polymer
            event_recorder = {}
            for Li_index in self.const_entity_groups.keys():
                event_recorder_each_Li = {"0-MP-0":{"global_time":[], "disp":[]}, "0-MP-1":{"global_time":[], "disp":[]}, "0-MP-2":{"global_time":[], "disp":[]}, "0-MP-3":{"global_time":[], "disp":[]}} 
                # 0-MP-0 : no O-index change for majority polymer, 0-MP-1 : new O added, 0-MP-2 : existing O removed, 0-MP-3 : new O added and existing removed
                for entity_group in self.const_entity_groups[Li_index]:
                    majority_polymer_entity = find_majority_polymer(entity_group, self.res_id_mapper)
                    if majority_polymer_entity == None:
                        ''' in case there is no polymer entity at all'''
                        list_of_interest = list(entity_group.values())[0] # taking any entity (this would be a TFSI, not a polymer)
                        sequence = list_of_interest['coord_list']
                        converted_sequence = [get_sequential_index(i) for i in sequence] # C-C-O-C-C-O -> divide by 3
                        for local_time_index, seq in enumerate(converted_sequence):
                            if local_time_index == len(converted_sequence)-1: # if
                                continue
                            else:
                                event_recorder_each_Li["0-MP-0"]["global_time"].append(list_of_interest['global_time'][local_time_index])
                                event_recorder_each_Li["0-MP-0"]["disp"].append(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])
                        continue
                    list_of_interest = list(majority_polymer_entity.values())[0] # {'coord_list':[], 'Li_positions':[], 'times':[], 'Li_index':[]}
                    sequence = list_of_interest['coord_list']
                    print(sequence)
                    converted_sequence = [get_sequential_index(i) for i in sequence] # C-C-O-C-C-O -> divide by 3, but this isn't necessary anymore because i use sequential index for anions only
                    # set is needed here to compute "removed" and "added".
                    print(converted_sequence)
                    for local_time_index, seq in enumerate(converted_sequence):
                        assert(len(converted_sequence) > 1)
                        if local_time_index == len(converted_sequence)-1: # if you are at the end of a sequence or if the length of a stay for a sequence is 1 frame 
                            # so here only if the entity stays more than 1 frame, i record as a change_in_O_index
                            # TBD here
                            '''
                            if length of "converted_sequence" is 1, it can't be here.
                            because it means an entity before and after this time is different, making this classified as a 
                            different event, not "0-0".
                            added an assertion above for this.
                            This is because for const_entity_groups, if it occured once, then we have states before and after the event.

                            '''
                            continue
                        removed = seq - converted_sequence[local_time_index+1]
                        added = converted_sequence[local_time_index+1] - seq
                        if len(removed) > 0 and len(added) > 0:
                            pid = "0-MP-3"
                        elif len(removed) > 0 and len(added) == 0:
                            pid = "0-MP-2"
                        elif len(removed) == 0 and len(added) > 0:
                            pid = "0-MP-1"
                        elif len(removed) == 0 and len(added) == 0:
                            pid = "0-MP-0"
                        else:
                            raise ValueError("Bug. No event type found.")
                        #event_recorder_each_Li[pid].append({"global_time":list_of_interest['global_time'][local_time_index] , "disp":self.wrap_disp(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])})
                        event_recorder_each_Li[pid]["global_time"].append(list_of_interest['global_time'][local_time_index])
                        event_recorder_each_Li[pid]["disp"].append(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])
                event_recorder[Li_index] = event_recorder_each_Li
            return event_recorder


        if mode == "ONLY_MAJORITY_POLYMER_ROLL_NOROLL": # 0, 1, 2, 3 (no-change, add new, lose some, add and lose)
            # I only look at how the majority polymer evolves as a function of time.
            def get_sequential_index(index_list):
                '''convert index to sequential index (O-C-C-O-C-C chain)'''
                return set(index_list)
                #return set([index//3 for index in index_list])
            def find_majority_polymer(entity_group, res_id_mapper):
                '''
                majority_polymer: polymer that provides most number of oxygens to the target lithium on average during the duration
                entity_group : [{res_id : {'coord_list':[], 'Li_positions':[], 'times':[], 'Li_index':[]}}]
                '''
                polymer_entities = [{res_id:entity_group[res_id]} for res_id in entity_group.keys() if res_id_mapper[res_id] == 'polymer']
                polymer_entities.sort(key=lambda x:list(x.keys())[0]) # deterministic. in case two entities have the exact same average coordinations.
                #print(f"number of polymer entities: {len(polymer_entities)} out of total {len(entity_group.keys())}")# , "number of polymers")
                if len(polymer_entities) == 0:
                    return None
                if len(polymer_entities) == 1:
                    return polymer_entities[0]
                majority_polymer = polymer_entities[0] # initializing as the first polymer
                majority_num_O = np.average([len(i) for i in list(majority_polymer.values())[0]['coord_list']])
                for entity in polymer_entities:
                    #print(np.average([len(i) for i in list(entity.values())[0]['coord_list']]))
                    if np.average([len(i) for i in list(entity.values())[0]['coord_list']]) > majority_num_O:
                        majority_num_O = np.average([len(i) for i in list(entity.values())[0]['coord_list']])
                        majority_polymer = entity
                #print("final", majority_num_O)
                return majority_polymer
            event_recorder = {}
            for Li_index in self.const_entity_groups.keys():
                event_recorder_each_Li = {"Roll":{"global_time":[], "disp":[]}, "Segmental":{"global_time":[], "disp":[]}}# "0-MP-2":[], "0-MP-3":[]} 
                # 0-MP-0 : no O-index change for majority polymer, 0-MP-1 : new O added, 0-MP-2 : existing O removed, 0-MP-3 : new O added and existing removed
                for entity_group in self.const_entity_groups[Li_index]:
                    majority_polymer_entity = find_majority_polymer(entity_group, self.res_id_mapper)
                    if majority_polymer_entity == None:
                        ''' in case there is no polymer entity at all'''
                        list_of_interest = list(entity_group.values())[0] # taking any entity (this would be a TFSI, not a polymer)
                        sequence = list_of_interest['coord_list']
                        converted_sequence = [get_sequential_index(i) for i in sequence] # C-C-O-C-C-O -> divide by 3
                        for local_time_index, seq in enumerate(converted_sequence):
                            if local_time_index == len(converted_sequence)-1: # if
                                continue
                            else:
                                #event_recorder_each_Li["Segmental"].append({"global_time":list_of_interest['global_time'][local_time_index], "disp":self.wrap_disp(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])})
                                event_recorder_each_Li["Segmental"]["global_time"].append(list_of_interest['global_time'][local_time_index])
                                event_recorder_each_Li["Segmental"]["disp"].append(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])
                        continue
                    list_of_interest = list(majority_polymer_entity.values())[0] # {'coord_list':[], 'Li_positions':[], 'times':[], 'Li_index':[]}
                    sequence = list_of_interest['coord_list']
                    converted_sequence = [get_sequential_index(i) for i in sequence] # C-C-O-C-C-O -> divide by 3
                    for local_time_index, seq in enumerate(converted_sequence):
                        if local_time_index == len(converted_sequence)-1: # if you are at the end of a sequence or if the length of a stay for a sequence is 1 frame 
                            # so here only if the entity stays more than 1 frame, i record as a change_in_O_index
                            # TBD here
                            continue
                        removed = seq - converted_sequence[local_time_index+1]
                        added = converted_sequence[local_time_index+1] - seq
                        if len(removed) > 0 and len(added) > 0:
                            pid = "Roll"
                        if len(removed) > 0 and len(added) == 0:
                            pid = "Segmental"
                        if len(removed) == 0 and len(added) > 0:
                            pid = "Segmental"
                        if len(removed) == 0 and len(added) == 0:
                            pid = "Segmental"
                        #event_recorder_each_Li[pid].append({"global_time":list_of_interest['global_time'][local_time_index] , "disp":self.wrap_disp(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])})
                        event_recorder_each_Li[pid]["global_time"].append(list_of_interest['global_time'][local_time_index])
                        event_recorder_each_Li[pid]["disp"].append(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])
                event_recorder[Li_index] = event_recorder_each_Li
            return event_recorder
        

        if mode == "ONLY_MAJORITY_POLYMER_THREE": # 0, 1, 2, 3 (no-change, add new, lose some, add and lose)
            # I only look at how the majority polymer evolves as a function of time.
            def get_sequential_index(index_list):
                '''convert index to sequential index (O-C-C-O-C-C chain)'''
                return set(index_list)
                #return set([index//3 for index in index_list])
            def find_majority_polymer(entity_group, res_id_mapper):
                '''
                majority_polymer: polymer that provides most number of oxygens to the target lithium on average during the duration
                entity_group : [{res_id : {'coord_list':[], 'Li_positions':[], 'times':[], 'Li_index':[]}}]
                '''
                polymer_entities = [{res_id:entity_group[res_id]} for res_id in entity_group.keys() if res_id_mapper[res_id] == 'polymer']
                polymer_entities.sort(key=lambda x:list(x.keys())[0]) # deterministic. in case two entities have the exact same average coordinations.
                #print(f"number of polymer entities: {len(polymer_entities)} out of total {len(entity_group.keys())}")# , "number of polymers")
                if len(polymer_entities) == 0:
                    return None
                if len(polymer_entities) == 1:
                    return polymer_entities[0]
                majority_polymer = polymer_entities[0] # initializing as the first polymer
                majority_num_O = np.average([len(i) for i in list(majority_polymer.values())[0]['coord_list']])
                for entity in polymer_entities:
                    #print(np.average([len(i) for i in list(entity.values())[0]['coord_list']]))
                    if np.average([len(i) for i in list(entity.values())[0]['coord_list']]) > majority_num_O:
                        majority_num_O = np.average([len(i) for i in list(entity.values())[0]['coord_list']])
                        majority_polymer = entity
                #print("final", majority_num_O)
                return majority_polymer
            event_recorder = {}
            for Li_index in self.const_entity_groups.keys():
                event_recorder_each_Li = {"Roll":{"global_time":[], "disp":[]}, "LOSE_OR_GAIN":{"global_time":[], "disp":[]},"Segmental":{"global_time":[], "disp":[]}}# "0-MP-2":[], "0-MP-3":[]} 
                # 0-MP-0 : no O-index change for majority polymer, 0-MP-1 : new O added, 0-MP-2 : existing O removed, 0-MP-3 : new O added and existing removed
                for entity_group in self.const_entity_groups[Li_index]:
                    majority_polymer_entity = find_majority_polymer(entity_group, self.res_id_mapper)
                    if majority_polymer_entity == None:
                        ''' in case there is no polymer entity at all'''
                        list_of_interest = list(entity_group.values())[0] # taking any entity (this would be a TFSI, not a polymer)
                        sequence = list_of_interest['coord_list']
                        converted_sequence = [get_sequential_index(i) for i in sequence] # C-C-O-C-C-O -> divide by 3
                        for local_time_index, seq in enumerate(converted_sequence):
                            if local_time_index == len(converted_sequence)-1: # if
                                continue
                            else:
                                event_recorder_each_Li["Segmental"]["global_time"].append(list_of_interest['global_time'][local_time_index])
                                event_recorder_each_Li["Segmental"]["disp"].append(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])
                                #event_recorder_each_Li["Segmental"].append({"global_time":list_of_interest['global_time'][local_time_index], "disp":self.wrap_disp(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])})
                        continue
                    list_of_interest = list(majority_polymer_entity.values())[0] # {'coord_list':[], 'Li_positions':[], 'times':[], 'Li_index':[]}
                    sequence = list_of_interest['coord_list']
                    converted_sequence = [get_sequential_index(i) for i in sequence] # C-C-O-C-C-O -> divide by 3
                    for local_time_index, seq in enumerate(converted_sequence):
                        if local_time_index == len(converted_sequence)-1: # if you are at the end of a sequence or if the length of a stay for a sequence is 1 frame 
                            # so here only if the entity stays more than 1 frame, i record as a change_in_O_index
                            # TBD here
                            continue
                        removed = seq - converted_sequence[local_time_index+1]
                        added = converted_sequence[local_time_index+1] - seq
                        if len(removed) > 0 and len(added) > 0:
                            pid = "Roll"
                        if len(removed) > 0 and len(added) == 0:
                            pid = "LOSE_OR_GAIN"
                        if len(removed) == 0 and len(added) > 0:
                            pid = "LOSE_OR_GAIN"
                        if len(removed) == 0 and len(added) == 0:
                            pid = "Segmental"
                        #event_recorder_each_Li[pid].append({"global_time":list_of_interest['global_time'][local_time_index] , "disp":self.wrap_disp(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])})
                        event_recorder_each_Li[pid]["global_time"].append(list_of_interest['global_time'][local_time_index])
                        event_recorder_each_Li[pid]["disp"].append(list_of_interest['Li_positions'][local_time_index+1]-list_of_interest['Li_positions'][local_time_index])
                event_recorder[Li_index] = event_recorder_each_Li
            return event_recorder



    @classmethod
    def detect_change_O_index(cls, entities):
        '''
        entities: {Li_index: [{res_id: {'coord_list':[], 'Li_positions':[], 'times':[], 'Li_index':[]}}]
        you may choose to use this class method for polymer_entities only, tfsi_entities only or everything.
        if entities is not provided, it uses "total_entities" within the object.
        '''
        def get_sequential_index(index_list):
            '''convert index to sequential index (O-C-C-O-C-C chain)'''
            return set([index//3 for index in index_list])
        return_dict = {}
        for Li_index in entities.keys():
            return_each_Li_dict = {0:[],1:[],2:[],3:[]}
            for entity in entities[Li_index]:
                list_of_interest = list(entity.values())[0] # {'coord_list':[], 'Li_positions':[], 'times':[], 'Li_index':[]}
                sequence = list_of_interest['coord_list']
                converted_sequence = [get_sequential_index(i) for i in sequence] # C-C-O-C-C-O -> divide by 3
                for time_index, seq in enumerate(converted_sequence):
                    if time_index == len(converted_sequence)-1: # if you are at the end of a sequence or if the length of a stay for a sequence is 1 frame 
                        # so here only if the entity stays more than 1 frame, i record as a change_in_O_index
                        # TBD here
                        continue
                    removed = seq - converted_sequence[time_index+1]
                    added = converted_sequence[time_index+1] - seq
                    if len(removed) > 0 and len(added) > 0:
                        pid = 3
                    if len(removed) > 0 and len(added) == 0:
                        pid = 2
                    if len(removed) == 0 and len(added) > 0:
                        pid = 1
                    if len(removed) == 0 and len(added) == 0:
                        pid = 0
                    #return_dict[pid].append([time_index, list(entity.keys())[0]])
                    #return_dict[pid].append({'disp':list_of_interest['Li_positions'][time_index+1]-list_of_interest['Li_positions'][time_index]})#, [time_index, list(entity.keys())[0]])
                    #return_each_Li_dict[pid].append(cls.wrap_disp(list_of_interest['Li_positions'][time_index+1]-list_of_interest['Li_positions'][time_index]))
                    return_each_Li_dict[pid].append(list_of_interest['Li_positions'][time_index+1]-list_of_interest['Li_positions'][time_index])
            return_dict[Li_index] = return_each_Li_dict
        return return_dict
    
    @classmethod
    def aggregate_detections(cls, entities):
        '''
        remove the Li_index distinction from any dictlist classified by Li_index
        '''
        rv = {}
        for Li_index in entities.keys():
            for event_type in entities[Li_index].keys():
                if event_type in rv.keys():
                    rv[event_type].extend(entities[Li_index][event_type])
                else:
                    rv[event_type] = entities[Li_index][event_type]
        return rv

    @classmethod
    def compute_event_msd_fft(cls, event_recorder):
        '''
        event_recorder: key : type of event, value : list of disps
        '''
        def autocorrelation_fft(x):
            N = x.shape[0]
            F = np.fft.fft(x, n=2 * N)
            PSD = F * F.conjugate()
            res = np.fft.ifft(PSD)
            res = (res[:N]).real
            n = N * np.ones(N) - np.arange(N)
            return res / n
        
        def one_ion_msd_fft(r, dt_indices):
            """
            r (np.array, shape is typically [n_step,3], n_step is number of steps, 3 is 3 dimentions)
            """
            # ------------ S1
            n_step, dim = r.shape
            r_square = np.square(r)
            r_square = np.append(r_square, np.zeros((1, dim)), axis=0)  # (n_step+1, 3)
            S1_component = np.zeros((dim, n_step))  # (dim, n_step)
            r_square_sum = 2 * np.sum(r_square, axis=0)  # (3)
            for i in range(n_step):
                r_square_sum = r_square_sum - r_square[i - 1, :] - r_square[n_step - i, :]
                S1_component[:, i] = r_square_sum / (n_step - i)
            S1 = np.sum(S1_component, axis=0)

            # ------------ S2
            S2_component = np.array([autocorrelation_fft(r[:, i]) for i in range(r.shape[1])])  # (dim, N)
            S2 = np.sum(S2_component, axis=0)

            # ------------ return
            return (S1 - 2 * S2)[dt_indices], (S1_component - 2 * S2_component)[:, dt_indices]

        event_msd = {}
        for i in event_recorder.keys():
            if len(event_recorder[i]) == 0:
                continue
            dp = np.array([j for j in event_recorder[i]])
            #random.shuffle(dp)
            c_disp = np.cumsum(np.array(dp), axis=0)

            n_steps = len(dp)
            dt_indices = np.arange(1, n_steps, 1)
            dt = dt_indices * 1 # 1 : how much time per snapshot, determined by step_skip and time_step
            n_dt = len(dt_indices)
            msd_by_ions = np.empty([0, n_dt])
            #print(msd_by_ions)
            msd_i, msd_component_i = one_ion_msd_fft(c_disp, dt_indices)
            
            '''print(msd_i, np.shape(msd_i))
            msd_by_ions = np.append(msd_by_ions,
                                                msd_i.reshape(1, n_dt),
                                                axis=0)
            print(msd_by_ions)
            break
            '''
            event_msd[i] = {"time":dt, "msds":msd_i}

        return event_msd

    @classmethod
    def compute_event_msd(cls, event_recorder):
        '''
        non-smoothed MSD (no time-window averaging) calculator
        '''
        event_msd = {}
        for i in event_recorder.keys():
            if len(event_recorder[i]) == 0:
                continue
            dp = np.array([j for j in event_recorder[i]])
            c_disp = np.cumsum(np.array([i for i in event_recorder[i]]), axis=0)
            #event_msd[i] = np.average([np.linalg.norm(i) for i in c_disp])
            dt = np.arange(0, len(c_disp), 1)
            msds = np.zeros_like(dt, dtype=np.double)
            for timeindex, time in enumerate(c_disp):
                sq_disp = time**2
                #print(msds[timeindex])
                #print(sq_disp)
                msds[timeindex] = np.average(sq_disp)
            event_msd[i] = {"time":dt, "msds":msds}
        return event_msd
    
    @classmethod
    def plot_freq_diffusivity(cls, detection_result, ax=None):
        total_freq_d_info = []
        color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"]
        color_dict = {}
        for index, i in enumerate(['0', '1', '2','3']):
            color_dict[i] = color_list[index]
        for Li_index in detection_result.keys():
            event_msd_fft = cls.compute_event_msd_fft(detection_result[Li_index])
            for event_type in event_msd_fft.keys():
                sliced = len(event_msd_fft[event_type]['time'])//20
                if sliced < 1:
                    continue
                slope, intercept, _, _, _ = stats.linregress(event_msd_fft[event_type]['time'][:sliced], event_msd_fft[event_type]['msds'][:sliced])
                diffusivity_event = slope / (20 * 3)
                frequency_event = len(event_msd_fft[event_type]['time'])/np.sum([len(event_msd_fft[i]['time']) for i in event_msd_fft.keys()])
                total_freq_d_info.append({'Li_index':Li_index, 'event_type':event_type, 'diffusivity':diffusivity_event, 'frequency':frequency_event})
        df_freq_d_info = pd.DataFrame(total_freq_d_info)
        if not ax:
            fig, ax = plt.subplots()
        sns.scatterplot(data=df_freq_d_info, x='frequency', y='diffusivity', hue = 'event_type', alpha=0.5, ax=ax)
        ax.set_xscale('log')
        ax.set_xlabel("Frequency of event (1/ps)")
        ax.set_ylabel("Diffusivity ($\AA^2$/ps)")
        ax.set_yscale('log')
        return df_freq_d_info

    @classmethod
    def get_long_lived_entities(cls, list_of_entities, time):
        '''
        list_of_entities : [entities]
        entities : {res_id : {'coord_list':[], 'Li_positions':[], 'times':[]}}
        '''
        long_lived_entities = {}
        for Li_index in list_of_entities.keys():
            long_lived_entities[Li_index] = []
            for j in list_of_entities[Li_index]:
                if len(list(j.values())[0]['Li_positions']) > time:
                    long_lived_entities[Li_index].append(j)
                    #print(j)
                #if j[list(j.keys())[0]]['times'][-1] - j[list(j.keys())[0]]['times'][0] > time:
                #    long_lived_entities.append(j)
        return long_lived_entities

    @classmethod
    def get_short_lived_entities(cls, list_of_entities, time):
        '''
        list_of_entities : [entities]
        entities : {res_id : {'coord_list':[], 'Li_positions':[], 'times':[]}}
        '''
        short_lived_entities = {}
        for Li_index in list_of_entities.keys():
            short_lived_entities[Li_index] = []
            for j in list_of_entities[Li_index]:
                if len(list(j.values())[0]['Li_positions']) < time:
                    short_lived_entities[Li_index].append(j)
                    #print(j)
                #if j[list(j.keys())[0]]['times'][-1] - j[list(j.keys())[0]]['times'][0] > time:
                #    long_lived_entities.append(j)
        return short_lived_entities

    @classmethod
    def get_less_than_n_coordinated_entities(cls, list_of_entities, cutoff_coord):
        '''
        list_of_entities : [entities]
        entities : {res_id : {'coord_list':[], 'Li_positions':[], 'times':[]}}
        '''
        highly_coordinating_entities = {}
        for Li_index in list_of_entities.keys():
            highly_coordinating_entities[Li_index] = []
            for j in list_of_entities[Li_index]:
                if np.average([len(k) for k in list(j.values())[0]['coord_list']]) < cutoff_coord:
                    highly_coordinating_entities[Li_index].append(j)
                #if j[list(j.keys())[0]]['times'][-1] - j[list(j.keys())[0]]['times'][0] > time:
                #    long_lived_entities.append(j)
        return highly_coordinating_entities

    @classmethod
    def get_more_than_n_coordinated_entities(cls, list_of_entities, cutoff_coord):
        '''
        list_of_entities : [entities]
        entities : {res_id : {'coord_list':[], 'Li_positions':[], 'times':[]}}
        '''
        highly_coordinating_entities = {}
        for Li_index in list_of_entities.keys():
            highly_coordinating_entities[Li_index] = []
            for j in list_of_entities[Li_index]:
                if np.average([len(k) for k in list(j.values())[0]['coord_list']]) > cutoff_coord:
                    highly_coordinating_entities[Li_index].append(j)
                #if j[list(j.keys())[0]]['times'][-1] - j[list(j.keys())[0]]['times'][0] > time:
                #    long_lived_entities.append(j)
        return highly_coordinating_entities
    
    @classmethod
    def from_parallel_pieces(cls, piecelist, compute_distance_matrix=False):
        """
        concatenate multiple 
        """
        return_object = PolymerElectrolyteAnalyzer(piecelist[0].PATH, piecelist[0].label)
        return_object.complete_coord_info = {}
        return_object.anion_displacements = {}
        if compute_distance_matrix:
            return_object.distance_matrix = np.concatenate([i.distance_matrix for i in piecelist], axis=0)
        else:
            return_object.distance_matrix = None
        for Li_index in piecelist[0].complete_coord_info.keys():
            return_object.complete_coord_info[Li_index] = {'cartesian_positions':np.concatenate([piece.complete_coord_info[Li_index]['cartesian_positions'] for piece in piecelist]),
                                                    'cutoffs':np.concatenate([piece.complete_coord_info[Li_index]['cutoffs'] for piece in piecelist]),
                                                    'num_neighbors':np.concatenate([piece.complete_coord_info[Li_index]['num_neighbors'] for piece in piecelist]),
                                                    'flattened_O_providers':np.concatenate([piece.complete_coord_info[Li_index]['flattened_O_providers'] for piece in piecelist]),
                                                    'global_times':np.concatenate([piece.complete_coord_info[Li_index]['global_times'] for piece in piecelist])}
        for anion_index in piecelist[0].anion_displacements.keys():
            return_object.anion_displacements[anion_index] = {'global_times':np.concatenate([piece.anion_displacements[anion_index]['global_times'] for piece in piecelist]),
                                                            'disp':np.concatenate([piece.anion_displacements[anion_index]['disp'] for piece in piecelist])}
        return_object.res_ids = piecelist[0].res_ids
        return_object.res_id_mapper = piecelist[0].res_id_mapper
        return_object.structure = piecelist[0].structure
        return_object.structure_init = piecelist[0].structure_init
        return_object.structure_final = piecelist[0].structure_final
        return_object.mobile_cation = piecelist[0].mobile_cation
        return_object.anion = piecelist[0].anion
        return_object.atom_names = piecelist[0].atom_names
        return_object.atom_names_long_rdf = piecelist[0].atom_names_long_rdf
        return_object.unique_types = piecelist[0].unique_types
        return_object.types_rdf = piecelist[0].types_rdf
        return_object.coordinators = piecelist[0].coordinators
        return_object.cutoff = piecelist[0].cutoff
        return_object.mode = piecelist[0].mode
        return_object.anion_indices_forward = piecelist[0].anion_indices_forward
        return_object.anion_indices_back = piecelist[0].anion_indices_back
        return_object.boxdim = piecelist[0].boxdim
        return_object.sampling_freq = piecelist[0].sampling_freq
        return_object.compute_distance_matrix = compute_distance_matrix
        return return_object



import numpy as np
from multiprocessing import Pool

def calculate_single_dt(instance, time, stepskip,distance_cutoff):
    """Helper function to calculate for single dt using the instance method."""
    return instance.calculate_for_single_dt(time, stepskip, distance_cutoff)

class OnsagerAnalyzer():
    def __init__(self, timestep, Li_disps, anion_disps, label=None, distance_matrix=None, distance_cutoff=None, mode="EVENT", binmethod="FULL"):
        '''
        distance_cutoff: None or integer. None is the conventional Onsager coefficients.
        mode: "EVENT" : using the event-based displacements for lithium, "PARTICLE" : concatenating the event-decomposed displacements 
        so that we obtain Li-Li, Li-anion, anion-anion only. default is "EVENT" mode.
        '''
        self.Li_disps = Li_disps
        self.anion_disps = anion_disps
        self.n_total_frames = len(list(self.anion_disps.values())[0])
        self.mode=mode
        if mode == "EVENT":
            self.event_keys = list(list(self.Li_disps.values())[0].keys())
            print(self.event_keys, 'here')
        elif mode == "PARTICLE":
            self.event_keys = ["Li"]
        self.timestep = timestep
        self.label = label
        self.distance_matrix = distance_matrix
        if distance_cutoff == None:
            self.distance_cutoff = 1000000
        else:
            self.distance_cutoff = distance_cutoff
        self.mode = mode
        if binmethod == "FULL":
            print("No binning. Distance matrix is not used.")
            self.distance_matrix = None
            self.extended_distance_matrix = None
            
        elif binmethod == "DISTANCE":
            distance_info_former = self.distance_matrix[:-1,:,:] < self.distance_cutoff # as a function of time, Li1 and Li2 adjacent?
            distance_info_latter = self.distance_matrix[1:,:,:] < self.distance_cutoff
        elif binmethod == "NUMPARTICLE":
            def keep_N_smallest(matrix, num_keep=int(distance_cutoff)):
                NUM_EACH_PARTICLE = int(matrix.shape[0]/2)
                group1 = np.arange(0, NUM_EACH_PARTICLE)
                group2 = np.arange(NUM_EACH_PARTICLE, 2*NUM_EACH_PARTICLE)
                n = matrix.shape[0]
                result_matrix = np.zeros_like(matrix)

                for i in range(n):
                    # Get the current row (distances of particle i to others)
                    row = matrix[i, :]

                    # Group 1: indices 0-24 (within group1)
                    if i in group1:
                        group1_distances = row[group1]
                        smallest_two_group1 = np.argsort(group1_distances)[:2]  # Smallest two in group1
                        result_matrix[i, group1[smallest_two_group1]] = group1_distances[smallest_two_group1]

                    # Group 2: indices 25-49 (within group2)
                    if i in group2:
                        group2_distances = row[group2]
                        smallest_two_group2 = np.argsort(group2_distances)[:2]  # Smallest two in group2
                        result_matrix[i, group2[smallest_two_group2]] = group2_distances[smallest_two_group2]

                    # Cross group 1 and group 2 (0-24 vs 25-49)
                    if i in group1:
                        group2_distances = row[group2]
                        smallest_two_group2 = np.argsort(group2_distances)[:2]  # Smallest two in group2
                        result_matrix[i, group2[smallest_two_group2]] = group2_distances[smallest_two_group2]
                    elif i in group2:
                        group1_distances = row[group1]
                        smallest_two_group1 = np.argsort(group1_distances)[:2]  # Smallest two in group1
                        result_matrix[i, group1[smallest_two_group1]] = group1_distances[smallest_two_group1]

                return result_matrix
            neighbor_masked = np.array([keep_N_smallest(self.distance_matrix[i]) + np.identity(self.distance_matrix.shape[1]) for i in range(self.distance_matrix.shape[0])])
            distance_info_former = neighbor_masked[:-1,:,:] > 0
            distance_info_latter = neighbor_masked[1:,:,:] > 0
                
        if binmethod in ["DISTANCE", "NUMPARTICLE"]:
            self.distance_matrix_cutoff = distance_info_latter | distance_info_former
            #self.distance_matrix_cutoff = np.random.choice([True, False], size=self.distance_matrix_cutoff.shape)
            intermediate = self.distance_matrix_cutoff.copy()
            NUM_CATION = int(self.distance_matrix.shape[1]/2)
            NUM_EVENT = len(self.event_keys)
            intermediate[:, :NUM_CATION, :] , intermediate[:, NUM_CATION:2*NUM_CATION, :] = intermediate[:, NUM_CATION:2*NUM_CATION, :], intermediate[:, :NUM_CATION, :].copy()
            intermediate[:, :, :NUM_CATION], intermediate[:, :, NUM_CATION:2*NUM_CATION] = intermediate[:, :, NUM_CATION:2*NUM_CATION], intermediate[:, :, :NUM_CATION].copy()
            print(intermediate.shape)
            
            # Step 1: Repeat rows 0-25 three times across the time dimension
            repeated_rows = np.repeat(intermediate[:, :NUM_CATION, :], NUM_EVENT, axis=1)

            # Step 2: Repeat columns 0-25 three times
            repeated_rows_and_columns = np.repeat(repeated_rows[:, :, :NUM_CATION], NUM_EVENT, axis=2)

            # Step 3: Combine the repeated part with the unchanged part for rows and columns from 25-50
            self.extended_distance_matrix = np.concatenate([
                np.concatenate([repeated_rows_and_columns, np.repeat(intermediate[:, :NUM_CATION, NUM_CATION:], NUM_EVENT, axis=1)], axis=2),
                np.concatenate([np.repeat(intermediate[:, NUM_CATION:, :NUM_CATION], NUM_EVENT, axis=2), intermediate[:, NUM_CATION:, NUM_CATION:]], axis=2)
            ], axis=1)
            print(self.extended_distance_matrix.shape)  # Should print (1000, 75, 75) 
        elif binmethod == "FULL":
            self.extended_distance_matrix = None
        total_simulation_time = len(list(self.anion_disps.values())[0]['global_times'])
        cation_disp_matrix = []
        if mode == "EVENT":
            for Li_index, events in self.Li_disps.items():
                specific_lithium = []
                for event_type in self.event_keys:
                    specific_lithium_specific_event = np.zeros((total_simulation_time,3))
                    #for specific_event in events:
                    print(event_type, events, self.event_keys,'here2')
                    if event_type in events:
                        for inside_index, global_time in enumerate(events[event_type]['global_time']):
                            specific_lithium_specific_event[global_time] = events[event_type]['disp'][inside_index]
                    specific_lithium.append(specific_lithium_specific_event)
                specific_lithium = np.stack(specific_lithium, axis=0)
                cation_disp_matrix.append(specific_lithium)
        elif mode == "PARTICLE":
            for Li_index, events in self.Li_disps.items():
                specific_lithium = []
                specific_lithium_specific_event = np.zeros((total_simulation_time,3))
                for event_type in events.keys(): # existing events ,not necessarily all possible keys needed because we add'm up.
                    for inside_index, global_time in enumerate(events[event_type]['global_time']):
                        specific_lithium_specific_event[global_time] = events[event_type]['disp'][inside_index]
                specific_lithium.append(specific_lithium_specific_event)
                specific_lithium = np.stack(specific_lithium, axis=0)
                cation_disp_matrix.append(specific_lithium)
            
            
        cation_disp_matrix = np.concatenate(cation_disp_matrix, axis=0)
        anion_disp_matrix = []
        for anion_index, anion in self.anion_disps.items():
            #specific_anion = np.array([anion[i]['disp'] for i in range(total_simulation_time)])
            specific_anion = np.array(anion['disp'])
            anion_disp_matrix.append(specific_anion)
        anion_disp_matrix = np.stack(anion_disp_matrix, axis=0)
        complete_disp_matrix = np.append(cation_disp_matrix, anion_disp_matrix, axis=0)
        self.complete_disp_matrix = np.transpose(complete_disp_matrix, (1, 0, 2))

        return
    
    def compute_self_diffusion_Li():
        return

    def compute_self_diffusion_anion():
        return

    def compute_cation_cation_cross():
        return 
    
    def compute_cation_anion_cross():
        return
    
    def compute_anion_anion_cross():
        return
    
    def compute_complete_matrix_concatenated(self, num_index=None, init_index=0):
        Li_disps = {}
        anion_disps = {}
        target_Li_disps = {}
        target_anion_disps = {}
        for Li_index in self.Li_disps:
            target_Li_disps[Li_index] = []
            for event_type in self.event_keys:
                for disp in self.Li_disps[Li_index][event_type]:
                    if disp['global_time'] < init_index:
                        continue
                    if disp['global_time'] >= init_index + num_index:
                        continue

                    target_Li_disps[Li_index].append(disp)
            target_Li_disps[Li_index].sort(key=lambda x: x['global_time'])        
        for anion_index in self.anion_disps.keys():
            target_anion_disps[anion_index] = []
            for disp in self.anion_disps[anion_index]:
                if disp['global_time'] < init_index:
                    continue
                if disp['global_time'] >= init_index + num_index:
                    continue
                target_anion_disps[anion_index].append(disp)
        cumsum_vector = []
        #event_orders = list(list(target_Li_disps.values())[0].keys())
        #event_orders.sort()
        for Li_index in target_Li_disps.keys():
            print(len(target_Li_disps[Li_index]))
            assert(len(target_Li_disps[Li_index]) == num_index)
            cumdisp = np.sum([i['disp'] for i in target_Li_disps[Li_index]], axis=0)
            cumsum_vector.append(list(cumdisp))
        for anion_index in target_anion_disps.keys():
            print(len(target_anion_disps[anion_index]))
            assert(len(target_anion_disps[anion_index]) == num_index)
            cumdisp = np.sum([i['disp'] for i in target_anion_disps[anion_index]], axis=0)
            cumsum_vector.append(list(cumdisp))
        return cumsum_vector
    
    # define 10% to 50% of the trajectory to perform a linear fit to get the Onsager coefficients.



    def compute_complete_matrix(self, num_index=None, init_index=0):
        '''
        num_index: number of indices to include in the slicing
        init_index: starting index of the slicing (everything before this are sliced out)
        '''
        Li_disps = {}
        anion_disps = {}
        target_Li_disps = {}
        target_anion_disps = {}
        for Li_index in self.Li_disps:
            target_Li_disps[Li_index] = {}
            for event_type in self.event_keys:
                for disp in self.Li_disps[Li_index][event_type]:
                    if disp['global_time'] < init_index:
                        continue
                    if disp['global_time'] >= init_index + num_index:
                        continue
                    if event_type not in target_Li_disps[Li_index].keys():
                        target_Li_disps[Li_index][event_type] = []
                    target_Li_disps[Li_index][event_type].append(disp)
                    
        for anion_index in self.anion_disps:
            target_anion_disps[anion_index] = []
            for disp in self.anion_disps[anion_index]:
                if disp['global_time'] < init_index:
                    continue
                if disp['global_time'] >= init_index + num_index:
                    continue
                target_anion_disps[anion_index].append(disp)
        cumsum_vector = []
        #event_orders = list(list(target_Li_disps.values())[0].keys())
        #event_orders.sort()
        for Li_index in target_Li_disps.keys():
            Li_disp_counter = 0
            for event in self.event_keys:
                if event not in target_Li_disps[Li_index].keys():
                    cumdisp = np.zeros(3)
                else:
                    cumdisp = np.sum([i['disp'] for i in target_Li_disps[Li_index][event]], axis=0)
                    Li_disp_counter += len(target_Li_disps[Li_index][event])
                cumsum_vector.append(list(cumdisp))
            assert(Li_disp_counter == num_index)
        for anion_index in target_anion_disps:
            cumdisp = np.sum([i['disp'] for i in target_anion_disps[anion_index]], axis=0)
            cumsum_vector.append(list(cumdisp))
        return cumsum_vector
    
    def compute_complete_matrix2(self, num_index=None, init_index=0):
        '''
        num_index: number of indices to include in the slicing
        init_index: starting index of the slicing (everything before this are sliced out)
        '''
        # Initialize dictionaries and list
        target_Li_disps = {}
        target_anion_disps = {}
        cumsum_vector = []

        # Process Li_disps
        for Li_index, events in self.Li_disps.items():
            target_Li_disps[Li_index] = {}
            target_events = target_Li_disps[Li_index]
            for event_type in self.event_keys:
                target_event_list = [disp for disp in events.get(event_type, [])
                                    if init_index <= disp['global_time'] < init_index + num_index]
                if target_event_list:
                    target_events[event_type] = target_event_list

        # Process anion_disps
        for anion_index, disps in self.anion_disps.items():
            filtered_disps = [disp for disp in disps 
                            if init_index <= disp['global_time'] < init_index + num_index]
            if filtered_disps:
                target_anion_disps[anion_index] = filtered_disps

        # Calculate cumulative displacement vectors
        for Li_index, events in target_Li_disps.items():
            Li_disp_counter = 0
            for event in self.event_keys:
                if event in events:
                    cumdisp = np.sum([i['disp'] for i in events[event]], axis=0)
                    Li_disp_counter += len(events[event])
                else:
                    cumdisp = np.zeros(3)
                cumsum_vector.append(list(cumdisp))
            assert(Li_disp_counter == num_index)

        for anion_index, disps in target_anion_disps.items():
            cumdisp = np.sum([i['disp'] for i in disps], axis=0)
            cumsum_vector.append(list(cumdisp))
        return cumsum_vector@np.transpose(cumsum_vector)
        #return cumsum_vector

    def build_complete_matrix(self):
        """
        cation_disp_matrix: {Li_index: event_index: ndarray}
        anion_disp_matrix: {anion_index: time, each-point: 3 dimensional}
        """
        total_simulation_time = len(list(self.anion_disps.values())[0])
        cation_disp_matrix = {}
        for Li_index, events in self.Li_disps.items():
            specific_lithium = []
            for event_type in self.event_keys:
                specific_lithium_specific_event = np.zeros((total_simulation_time,3))
                #for specific_event in events:
                if event_type in events:
                    for event in events[event_type]:
                        specific_lithium_specific_event[event['global_time']] = event['disp']
                specific_lithium.append(specific_lithium_specific_event)
            specific_lithium = np.stack(specific_lithium, axis=0)
            cation_disp_matrix[Li_index] = specific_lithium
        anion_disp_matrix = {}
        for anion_index, anion in self.anion_disps.items():
            specific_anion = np.array([anion[i]['disp'] for i in range(total_simulation_time)])
            anion_disp_matrix[anion_index] = specific_anion
        self.complete_event_matrix = cation_disp_matrix
        self.complete_anion_matrix = anion_disp_matrix
        return

    #def prepare_new(self):

    def compute_fft_onsager(self, self_only=True):

        # Existing function to compute the autocorrelation of a single particle (self-correlation)
        def autocorrelation_fft(x):
            N = x.shape[0]
            F = np.fft.fft(x, n=2 * N)
            PSD = F * F.conjugate()
            res = np.fft.ifft(PSD)
            res = (res[:N]).real
            n = N * np.ones(N) - np.arange(N)
            return res / n

        # Function to compute self-MSD for a single ion using FFT
        def one_ion_msd_fft(r, dt_indices):
            n_step, dim = r.shape
            r_square = np.square(r)
            r_square = np.append(r_square, np.zeros((1, dim)), axis=0)  # (n_step+1, 3)
            S1_component = np.zeros((dim, n_step))  # (dim, n_step)
            r_square_sum = 2 * np.sum(r_square, axis=0)  # (3)
            for i in range(n_step):
                r_square_sum = r_square_sum - r_square[i - 1, :] - r_square[n_step - i, :]
                S1_component[:, i] = r_square_sum / (n_step - i)
            S1 = np.sum(S1_component, axis=0)

            # Compute S2 using FFT (autocorrelation)
            S2_component = np.array([autocorrelation_fft(r[:, i]) for i in range(r.shape[1])])  # (dim, N)
            S2 = np.sum(S2_component, axis=0)
            #print("S1: ", S1)
            #print("S2", S2)
            # Return the MSD and its components for each time interval
            return (S1 - 2 * S2)[dt_indices], (S1_component - 2 * S2_component)[:, dt_indices]

        def crosscorrelation_fft(x1, x2):
            N = x1.shape[0]
            F1 = np.fft.fft(x1, n=2 * N)
            F2 = np.fft.fft(x2, n=2 * N)
            PSD = F1 * F2.conjugate()
            res = np.fft.ifft(PSD)
            res = (res[:N]).real
            n = N * np.ones(N) - np.arange(N)
            return res / n


        # New function to compute cross-MSD (distinct particles) using FFT
        def cross_ion_msd_fft(r1, r2, dt_indices):
            n_step, dim = r1.shape
            
            # Compute the cross-term S1
            #r1_square = np.square(r1)
            #r2_square = np.square(r2)
            r3_square = r1 * r2
            #r1_square = np.append(r1_square, np.zeros((1, dim)), axis=0)  # (n_step+1, 3)
            #r2_square = np.append(r2_square, np.zeros((1, dim)), axis=0)  # (n_step+1, 3)
            r3_square = np.append(r3_square, np.zeros((1, dim)), axis=0)  # (n_step+1)
            S1_component = np.zeros((dim, n_step))  # (dim, n_step)
            #r1_square_sum = np.sum(r1_square, axis=0)  # (3)
            #r2_square_sum = np.sum(r2_square, axis=0)  # (3)
            r3_square_sum = 2 * np.sum(r3_square, axis=0)
            for i in range(n_step):
                #r1_square_sum = r1_square_sum - r1_square[i - 1, :] - r1_square[n_step - i, :]
                #r2_square_sum = r2_square_sum - r2_square[i - 1, :] - r2_square[n_step - i, :]
                r3_square_sum = r3_square_sum - r3_square[i - 1] - r3_square[n_step - i]
                #S1_component[:, i] = (r1_square_sum + r2_square_sum) / (n_step - i)
                S1_component[:, i] = r3_square_sum / (n_step - i)

            S1 = np.sum(S1_component, axis=0)
            # Cross-correlation S2 using FFT
            S2_component1 = np.array([crosscorrelation_fft(r1[:, i], r2[:, i]) for i in range(r1.shape[1])])  # (dim, N)
            S2_component2 = np.array([crosscorrelation_fft(r2[:, i], r1[:, i]) for i in range(r1.shape[1])])  # (dim, N)
            S2_1 = np.sum(S2_component1, axis=0)
            S2_2 = np.sum(S2_component2, axis=0)
            #print("S1: ", S1)
            #print("S2_1", S2_1)
            #print("S2_2", S2_2)# S2_1, S2_2)
            """
            S2_component = np.zeros((dim, n_step))  # (dim, N)
            for d in range(dim):
                delta_r1 = r1[:, d] - np.mean(r1[:, d])
                delta_r2 = r2[:, d] - np.mean(r2[:, d])

                F1 = fft(delta_r1, n=2 * n_step)
                F2 = fft(delta_r2, n=2 * n_step)

                cross_corr = ifft(F1 * F2.conjugate())
                S2_component[d, :] = (cross_corr[:n_step]).real

            S2 = np.sum(S2_component, axis=0)
            """
            return (S1 - S2_1 - S2_2)[dt_indices], (S1_component - S2_component1 - S2_component2)[:, dt_indices]
            return (S1 - 2 * S2)[dt_indices], (S1_component - 2 * S2_component)[:, dt_indices]


        n_ions = self.complete_disp_matrix.shape[1]
        # Main loop to compute the self- and cross-MSD for all particle pairs
        #n_dt = len(self.anion_disps[16])
        n_dt = self.complete_disp_matrix.shape[0]
        num_event = len(self.event_keys)
        num_Li = len(self.Li_disps.keys())
        #print(n_dt)
        msd_by_pairs = np.empty([0, n_dt])  # Shape of n_pairs * n_dt
        msd_component_by_pairs = np.empty([3, 0, n_dt])  # Shape of 3 * n_pairs * n_dt
        displacements_final_diffusion_ions = self.complete_disp_matrix.transpose(1, 0, 2)
        #print(displacements_final_diffusion_ions.shape)
        #displacements_final_diffusion_ions = np.cumsum(displacements_final_diffusion_ions, axis=1)
        #print(displacements_final_diffusion_ions.shape)
        dt_indices = np.arange(0, n_dt)
        msd_matrix = []
        if self.mode == "EVENT":
            averaged_self = np.zeros((len(self.event_keys)+1, len(self.event_keys)+1, n_dt))  #np.zeros((n_ions, n_ions, n_dt))
            averaged_distinct = np.zeros((len(self.event_keys)+1, len(self.event_keys)+1, n_dt)) #np.zeros((n_ions, n_ions, n_dt))
        elif self.mode == "PARTICLE":
            averaged_self = np.zeros((2, 2, n_dt))
            averaged_distinct = np.zeros((2, 2, n_dt))
        for i in range(n_ions):
            print(f"FFT on column {i}th among{n_ions} columns")
            msd_by_pairs = np.empty([0, n_dt])  # Shape of n_pairs * n_dt
            # print(f"working on column {i}th")
            for j in range(n_ions):  # Loop over distinct particle pairs
                """
                if i < 25:
                    ik = i + 25
                else:
                    ik = i - 25
                if  j < 25:
                    jk = j + 25
                else:
                    jk = j - 25
                mask_tensor = self.extended_distance_matrix[:,ik,jk]
                """
                #i_k = np.random.randint(0,50)
                #j_k = np.random.randint(0,50)
                if self.extended_distance_matrix is not None:
                    mask_tensor = self.extended_distance_matrix[:,i,j]
                    ith = displacements_final_diffusion_ions[i,:,:].transpose()*mask_tensor
                    jth = displacements_final_diffusion_ions[j,:,:].transpose()*mask_tensor
                    ith = np.cumsum(ith.transpose(), axis=0)*sum(mask_tensor)/len(mask_tensor)
                    jth = np.cumsum(jth.transpose(), axis=0)*sum(mask_tensor)/len(mask_tensor)
                else:
                    # When no distance matrix is provided, treat all distances as valid
                    ith = np.cumsum(displacements_final_diffusion_ions[i,:,:], axis=0)
                    jth = np.cumsum(displacements_final_diffusion_ions[j,:,:], axis=0)
                if self.mode == "PARTICLE":
                    # start classification of where i, j correspond to
                    # particle-wise; Li-self, Li-distinct, anion-self, anion-distinct?
                    if self_only:
                        if i == j:
                            msd_ij, msd_component_ij = one_ion_msd_fft(ith, dt_indices)
                        else:
                            msd_ij, msd_component_ij = np.zeros(n_dt), np.zeros((3, n_dt))
                    else: # full computation of distinct
                        if i > j:
                            continue
                        elif i == j:
                            msd_ij, msd_component_ij = one_ion_msd_fft(ith, dt_indices)
                        else:
                            msd_ij, msd_component_ij = cross_ion_msd_fft(ith, jth, dt_indices)
                    # Now assigning the msd_ij to correct places.
                    if i == j:
                        if i < n_ions/2: 
                            averaged_self[0][0] += msd_ij
                        else:
                            averaged_self[1][1] += msd_ij
                    else:
                        if i < n_ions/2 and j < n_ions/2: # (Li, Li), distinct
                            averaged_distinct[0][0] += 2 * msd_ij
                        elif i < n_ions/2 and j >= n_ions/2: # (Li, anion), distinct
                            averaged_distinct[0][1] += msd_ij
                            averaged_distinct[1][0] += msd_ij
                        elif i >= n_ions/2 and j < n_ions/2: # (anion, Li), distinct
                            averaged_distinct[1][0] += msd_ij
                            averaged_distinct[1][0] += msd_ij
                        else:
                            averaged_distinct[1][1] += 2* msd_ij # remember, this is multiplication of two vectors; (i,j) and (j,i) exist
                elif self.mode == "EVENT":
                    if i > j:
                        continue # fill half because it's symmetric
                    if i == j:
                        msd_ij, msd_component_ij = one_ion_msd_fft(ith, dt_indices)
                        if i < num_Li * num_event: # Li event self
                            averaged_self[i%num_event][j%num_event] += msd_ij
                        else: # anion self
                            averaged_self[-1][-1] += msd_ij
                    else:
                        i_iscation = i < num_Li * num_event
                        j_iscation = j < num_Li * num_event
                        if i_iscation and j_iscation:
                            if i//num_event == j//num_event: # Li different event, but self (same particle)
                                msd_ij, msd_component_ij = cross_ion_msd_fft(ith, jth, dt_indices)
                                averaged_self[i%num_event][j%num_event] += msd_ij
                                averaged_self[j%num_event][i%num_event] += msd_ij

                            else:
                                if self_only:
                                    msd_ij, msd_component_ij = np.zeros(n_dt), np.zeros((3, n_dt))
                                else:
                                    msd_ij, msd_component_ij = cross_ion_msd_fft(ith, jth, dt_indices)
                                averaged_distinct[i%num_event][j%num_event] += msd_ij
                                averaged_distinct[j%num_event][i%num_event] += msd_ij
                        elif i_iscation and not j_iscation:
                            if self_only:
                                msd_ij, msd_component_ij = np.zeros(n_dt), np.zeros((3, n_dt))
                            else:
                                msd_ij, msd_component_ij = cross_ion_msd_fft(ith, jth, dt_indices)
                            averaged_distinct[i%num_event][-1] += msd_ij
                            averaged_distinct[-1][i%num_event] += msd_ij

                        elif j_iscation and not i_iscation:
                            if self_only:
                                msd_ij, msd_component_ij = np.zeros(n_dt), np.zeros((3, n_dt))
                            else:
                                msd_ij, msd_component_ij = cross_ion_msd_fft(ith, jth, dt_indices)
                            averaged_distinct[-1][j%num_event] += msd_ij
                            averaged_distinct[j%num_event][-1] += msd_ij

                        elif not i_iscation and not j_iscation:
                            if self_only:
                                msd_ij, msd_component_ij = np.zeros(n_dt), np.zeros((3, n_dt))
                            else:
                                msd_ij, msd_component_ij = cross_ion_msd_fft(ith, jth, dt_indices)
                            averaged_distinct[-1][-1] += 2*msd_ij
                #msd_by_pairs = np.append(msd_by_pairs, msd_ij.reshape(1, n_dt), axis=0)
                #msd_component_by_pairs = np.append(msd_component_by_pairs, msd_component_ij.reshape(3, 1, n_dt), axis=1)
            #print("here", msd_by_pairs.shape)
            #msd_matrix.append(msd_by_pairs)#, axis=0)
        return averaged_self, averaged_distinct
        msd_matrix = np.stack(msd_matrix)

        # Average MSD and MSD components over all pairs
        msd = np.average(msd_by_pairs, axis=0)
        msd_component = np.average(msd_component_by_pairs, axis=1)
        msd_matrix = np.where(msd_matrix,msd_matrix,msd_matrix.transpose(1,0,2))
        self.matrices_for_all_dt = msd_matrix
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
        #with open(f"{self.label}_{formatted_time}_d-{self.distance_cutoff}_mode-{self.mode}fft.pkl", "wb") as f:
        #    pickle.dump({"matrices":msd_matrix, "distance_cutoff":self.distance_cutoff, "mode":self.mode}, f)
        return msd_matrix
    





    def compute_fft_onsager_old2(self):

        # Existing function to compute the autocorrelation of a single particle (self-correlation)
        def autocorrelation_fft(x):
            N = x.shape[0]
            F = np.fft.fft(x, n=2 * N)
            PSD = F * F.conjugate()
            res = np.fft.ifft(PSD)
            res = (res[:N]).real
            n = N * np.ones(N) - np.arange(N)
            return res / n

        # Function to compute self-MSD for a single ion using FFT
        def one_ion_msd_fft(r, dt_indices):
            n_step, dim = r.shape
            r_square = np.square(r)
            r_square = np.append(r_square, np.zeros((1, dim)), axis=0)  # (n_step+1, 3)
            S1_component = np.zeros((dim, n_step))  # (dim, n_step)
            r_square_sum = 2 * np.sum(r_square, axis=0)  # (3)
            for i in range(n_step):
                r_square_sum = r_square_sum - r_square[i - 1, :] - r_square[n_step - i, :]
                S1_component[:, i] = r_square_sum / (n_step - i)
            S1 = np.sum(S1_component, axis=0)

            # Compute S2 using FFT (autocorrelation)
            S2_component = np.array([autocorrelation_fft(r[:, i]) for i in range(r.shape[1])])  # (dim, N)
            S2 = np.sum(S2_component, axis=0)
            #print("S1: ", S1)
            #print("S2", S2)
            # Return the MSD and its components for each time interval
            return (S1 - 2 * S2)[dt_indices], (S1_component - 2 * S2_component)[:, dt_indices]

        def crosscorrelation_fft(x1, x2):
            N = x1.shape[0]
            F1 = np.fft.fft(x1, n=2 * N)
            F2 = np.fft.fft(x2, n=2 * N)
            PSD = F1 * F2.conjugate()
            res = np.fft.ifft(PSD)
            res = (res[:N]).real
            n = N * np.ones(N) - np.arange(N)
            return res / n


        # New function to compute cross-MSD (distinct particles) using FFT
        def cross_ion_msd_fft(r1, r2, dt_indices):
            n_step, dim = r1.shape
            
            # Compute the cross-term S1
            #r1_square = np.square(r1)
            #r2_square = np.square(r2)
            r3_square = r1 * r2
            #r1_square = np.append(r1_square, np.zeros((1, dim)), axis=0)  # (n_step+1, 3)
            #r2_square = np.append(r2_square, np.zeros((1, dim)), axis=0)  # (n_step+1, 3)
            r3_square = np.append(r3_square, np.zeros((1, dim)), axis=0)  # (n_step+1)
            S1_component = np.zeros((dim, n_step))  # (dim, n_step)
            #r1_square_sum = np.sum(r1_square, axis=0)  # (3)
            #r2_square_sum = np.sum(r2_square, axis=0)  # (3)
            r3_square_sum = 2 * np.sum(r3_square, axis=0)
            for i in range(n_step):
                #r1_square_sum = r1_square_sum - r1_square[i - 1, :] - r1_square[n_step - i, :]
                #r2_square_sum = r2_square_sum - r2_square[i - 1, :] - r2_square[n_step - i, :]
                r3_square_sum = r3_square_sum - r3_square[i - 1] - r3_square[n_step - i]
                #S1_component[:, i] = (r1_square_sum + r2_square_sum) / (n_step - i)
                S1_component[:, i] = r3_square_sum / (n_step - i)

            S1 = np.sum(S1_component, axis=0)
            # Cross-correlation S2 using FFT
            S2_component1 = np.array([crosscorrelation_fft(r1[:, i], r2[:, i]) for i in range(r1.shape[1])])  # (dim, N)
            S2_component2 = np.array([crosscorrelation_fft(r2[:, i], r1[:, i]) for i in range(r1.shape[1])])  # (dim, N)
            S2_1 = np.sum(S2_component1, axis=0)
            S2_2 = np.sum(S2_component2, axis=0)
            #print("S1: ", S1)
            #print("S2_1", S2_1)
            #print("S2_2", S2_2)# S2_1, S2_2)
            """
            S2_component = np.zeros((dim, n_step))  # (dim, N)
            for d in range(dim):
                delta_r1 = r1[:, d] - np.mean(r1[:, d])
                delta_r2 = r2[:, d] - np.mean(r2[:, d])

                F1 = fft(delta_r1, n=2 * n_step)
                F2 = fft(delta_r2, n=2 * n_step)

                cross_corr = ifft(F1 * F2.conjugate())
                S2_component[d, :] = (cross_corr[:n_step]).real

            S2 = np.sum(S2_component, axis=0)
            """
            return (S1 - S2_1 - S2_2)[dt_indices], (S1_component - S2_component1 - S2_component2)[:, dt_indices]
            return (S1 - 2 * S2)[dt_indices], (S1_component - 2 * S2_component)[:, dt_indices]



        n_ions = self.complete_disp_matrix.shape[1]
        # Main loop to compute the self- and cross-MSD for all particle pairs
        #n_dt = len(self.anion_disps[16])
        n_dt = self.complete_disp_matrix.shape[0]
        num_event = len(self.event_keys)
        num_Li = len(self.Li_disps.keys())
        #print(n_dt)
        msd_by_pairs = np.empty([0, n_dt])  # Shape of n_pairs * n_dt
        msd_component_by_pairs = np.empty([3, 0, n_dt])  # Shape of 3 * n_pairs * n_dt
        displacements_final_diffusion_ions = self.complete_disp_matrix.transpose(1, 0, 2)
        #print(displacements_final_diffusion_ions.shape)
        #displacements_final_diffusion_ions = np.cumsum(displacements_final_diffusion_ions, axis=1)
        #print(displacements_final_diffusion_ions.shape)
        dt_indices = np.arange(0, n_dt)
        msd_matrix = []
        if self.mode == "EVENT":
            averaged_self = np.zeros((len(self.event_keys)+1, len(self.event_keys)+1, n_dt))  #np.zeros((n_ions, n_ions, n_dt))
            averaged_distinct = np.zeros((len(self.event_keys)+1, len(self.event_keys)+1, n_dt)) #np.zeros((n_ions, n_ions, n_dt))
        elif self.mode == "PARTICLE":
            averaged_self = np.zeros((2, 2, n_dt))
            averaged_distinct = np.zeros((2, 2, n_dt))
        for i in range(n_ions):
            msd_by_pairs = np.empty([0, n_dt])  # Shape of n_pairs * n_dt
            print(f"working on column {i}th")
            for j in range(n_ions):  # Loop over distinct particle pairs
                """
                if i < 25:
                    ik = i + 25
                else:
                    ik = i - 25
                if  j < 25:
                    jk = j + 25
                else:
                    jk = j - 25
                mask_tensor = self.extended_distance_matrix[:,ik,jk]
                """
                #i_k = np.random.randint(0,50)
                #j_k = np.random.randint(0,50)
                mask_tensor = self.extended_distance_matrix[:,i,j]
                ith = displacements_final_diffusion_ions[i,:,:].transpose()*mask_tensor
                jth = displacements_final_diffusion_ions[j,:,:].transpose()*mask_tensor
                ith = np.cumsum(ith.transpose(), axis=0)*sum(mask_tensor)/len(mask_tensor)
                jth = np.cumsum(jth.transpose(), axis=0)*sum(mask_tensor)/len(mask_tensor)

                #print(mask_tensor)
                if i > j:
                    #msd_ij = np.zeros(n_dt)
                    #msd_by_pairs = np.append(msd_by_pairs, msd_ij.reshape(1, n_dt), axis=0)
                    continue
                if i == j:
                    
                    # Self-correlation (diagonal terms)
                    #msd_ij, msd_component_ij = one_ion_msd_fft(displacements_final_diffusion_ions[i, :, :], dt_indices)
                    msd_ij, msd_component_ij = one_ion_msd_fft(ith, dt_indices)
                else:
                    
                    # Cross-correlation (off-diagonal terms)
                    #print(displacements_final_diffusion_ions[i, :, :].shape)
                    #ith = displacements_final_diffusion_ions[i, :, :].transpose()*mask_tensor
                    #jth = displacements_final_diffusion_ions[j, :, :].transpose()*mask_tensor
                    msd_ij, msd_component_ij = cross_ion_msd_fft(ith, jth, dt_indices)
                    #msd_ij, msd_component_ij = cross_ion_msd_fft(displacements_final_diffusion_ions[i, :, :]*mask_tensor, 
                    #                                            displacements_final_diffusion_ions[j, :, :][mask_tensor], dt_indices)
                if self.mode == "PARTICLE":
                    # start classification of where i, j correspond to
                    # particle-wise; Li-self, Li-distinct, anion-self, anion-distinct?
                    if i == j:
                        if i < n_ions/2:
                            averaged_self[0][0] += msd_ij
                        else:
                            averaged_self[1][1] += msd_ij
                    else:
                        if i < n_ions/2 and j < n_ions/2:
                            averaged_distinct[0][0] += 2 * msd_ij
                        elif i < n_ions/2 and j >= n_ions/2:
                            averaged_distinct[0][1] += msd_ij
                            averaged_distinct[1][0] += msd_ij
                        elif i >= n_ions/2 and j < n_ions/2:
                            averaged_distinct[1][0] += msd_ij
                            averaged_distinct[1][0] += msd_ij
                        else:
                            averaged_distinct[1][1] += 2* msd_ij
                elif self.mode == "EVENT":
                    if i == j:
                        if i < num_Li * num_event:
                            averaged_self[i%num_event][j%num_event] += msd_ij
                        else:
                            averaged_self[-1][-1] += msd_ij
                    else:
                        i_iscation = i < num_Li * num_event
                        j_iscation = j < num_Li * num_event
                        if i_iscation and j_iscation:
                            if i//num_event == j//num_event:
                                averaged_self[i%num_event][j%num_event] += msd_ij
                                averaged_self[j%num_event][i%num_event] += msd_ij

                            else:
                                averaged_distinct[i%num_event][j%num_event] += msd_ij
                                averaged_distinct[j%num_event][i%num_event] += msd_ij
                        elif i_iscation and not j_iscation:
                            averaged_distinct[i%num_event][-1] += msd_ij
                            averaged_distinct[-1][i%num_event] += msd_ij

                        elif j_iscation and not i_iscation:
                            averaged_distinct[-1][j%num_event] += msd_ij
                            averaged_distinct[j%num_event][-1] += msd_ij

                        elif not i_iscation and not j_iscation:
                            averaged_distinct[-1][-1] += 2*msd_ij
                #msd_by_pairs = np.append(msd_by_pairs, msd_ij.reshape(1, n_dt), axis=0)
                #msd_component_by_pairs = np.append(msd_component_by_pairs, msd_component_ij.reshape(3, 1, n_dt), axis=1)
            #print("here", msd_by_pairs.shape)
            #msd_matrix.append(msd_by_pairs)#, axis=0)
        return averaged_self, averaged_distinct
        msd_matrix = np.stack(msd_matrix)

        # Average MSD and MSD components over all pairs
        msd = np.average(msd_by_pairs, axis=0)
        msd_component = np.average(msd_component_by_pairs, axis=1)
        msd_matrix = np.where(msd_matrix,msd_matrix,msd_matrix.transpose(1,0,2))
        self.matrices_for_all_dt = msd_matrix
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
        #with open(f"{self.label}_{formatted_time}_d-{self.distance_cutoff}_mode-{self.mode}fft.pkl", "wb") as f:
        #    pickle.dump({"matrices":msd_matrix, "distance_cutoff":self.distance_cutoff, "mode":self.mode}, f)
        return msd_matrix
    

    def compute_fft_onsager_old(self):

        # Existing function to compute the autocorrelation of a single particle (self-correlation)
        def autocorrelation_fft(x):
            N = x.shape[0]
            F = np.fft.fft(x, n=2 * N)
            PSD = F * F.conjugate()
            res = np.fft.ifft(PSD)
            res = (res[:N]).real
            n = N * np.ones(N) - np.arange(N)
            return res / n

        # Function to compute self-MSD for a single ion using FFT
        def one_ion_msd_fft(r, dt_indices):
            n_step, dim = r.shape
            r_square = np.square(r)
            r_square = np.append(r_square, np.zeros((1, dim)), axis=0)  # (n_step+1, 3)
            S1_component = np.zeros((dim, n_step))  # (dim, n_step)
            r_square_sum = 2 * np.sum(r_square, axis=0)  # (3)
            for i in range(n_step):
                r_square_sum = r_square_sum - r_square[i - 1, :] - r_square[n_step - i, :]
                S1_component[:, i] = r_square_sum / (n_step - i)
            S1 = np.sum(S1_component, axis=0)

            # Compute S2 using FFT (autocorrelation)
            S2_component = np.array([autocorrelation_fft(r[:, i]) for i in range(r.shape[1])])  # (dim, N)
            S2 = np.sum(S2_component, axis=0)
            #print("S1: ", S1)
            #print("S2", S2)
            # Return the MSD and its components for each time interval
            return (S1 - 2 * S2)[dt_indices], (S1_component - 2 * S2_component)[:, dt_indices]

        def crosscorrelation_fft(x1, x2):
            N = x1.shape[0]
            F1 = np.fft.fft(x1, n=2 * N)
            F2 = np.fft.fft(x2, n=2 * N)
            PSD = F1 * F2.conjugate()
            res = np.fft.ifft(PSD)
            res = (res[:N]).real
            n = N * np.ones(N) - np.arange(N)
            return res / n


        # New function to compute cross-MSD (distinct particles) using FFT
        def cross_ion_msd_fft(r1, r2, dt_indices):
            n_step, dim = r1.shape
            
            # Compute the cross-term S1
            #r1_square = np.square(r1)
            #r2_square = np.square(r2)
            r3_square = r1 * r2
            #r1_square = np.append(r1_square, np.zeros((1, dim)), axis=0)  # (n_step+1, 3)
            #r2_square = np.append(r2_square, np.zeros((1, dim)), axis=0)  # (n_step+1, 3)
            r3_square = np.append(r3_square, np.zeros((1, dim)), axis=0)  # (n_step+1)
            S1_component = np.zeros((dim, n_step))  # (dim, n_step)
            #r1_square_sum = np.sum(r1_square, axis=0)  # (3)
            #r2_square_sum = np.sum(r2_square, axis=0)  # (3)
            r3_square_sum = 2 * np.sum(r3_square, axis=0)
            for i in range(n_step):
                #r1_square_sum = r1_square_sum - r1_square[i - 1, :] - r1_square[n_step - i, :]
                #r2_square_sum = r2_square_sum - r2_square[i - 1, :] - r2_square[n_step - i, :]
                r3_square_sum = r3_square_sum - r3_square[i - 1] - r3_square[n_step - i]
                #S1_component[:, i] = (r1_square_sum + r2_square_sum) / (n_step - i)
                S1_component[:, i] = r3_square_sum / (n_step - i)

            S1 = np.sum(S1_component, axis=0)
            # Cross-correlation S2 using FFT
            S2_component1 = np.array([crosscorrelation_fft(r1[:, i], r2[:, i]) for i in range(r1.shape[1])])  # (dim, N)
            S2_component2 = np.array([crosscorrelation_fft(r2[:, i], r1[:, i]) for i in range(r1.shape[1])])  # (dim, N)
            S2_1 = np.sum(S2_component1, axis=0)
            S2_2 = np.sum(S2_component2, axis=0)
            #print("S1: ", S1)
            #print("S2_1", S2_1)
            #print("S2_2", S2_2)# S2_1, S2_2)
            """
            S2_component = np.zeros((dim, n_step))  # (dim, N)
            for d in range(dim):
                delta_r1 = r1[:, d] - np.mean(r1[:, d])
                delta_r2 = r2[:, d] - np.mean(r2[:, d])

                F1 = fft(delta_r1, n=2 * n_step)
                F2 = fft(delta_r2, n=2 * n_step)

                cross_corr = ifft(F1 * F2.conjugate())
                S2_component[d, :] = (cross_corr[:n_step]).real

            S2 = np.sum(S2_component, axis=0)
            """
            return (S1 - S2_1 - S2_2)[dt_indices], (S1_component - S2_component1 - S2_component2)[:, dt_indices]
            return (S1 - 2 * S2)[dt_indices], (S1_component - 2 * S2_component)[:, dt_indices]



        n_ions = self.complete_disp_matrix.shape[1]
        # Main loop to compute the self- and cross-MSD for all particle pairs
        #n_dt = len(self.anion_disps[16])
        n_dt = self.complete_disp_matrix.shape[0]
        #print(n_dt)
        msd_by_pairs = np.empty([0, n_dt])  # Shape of n_pairs * n_dt
        msd_component_by_pairs = np.empty([3, 0, n_dt])  # Shape of 3 * n_pairs * n_dt
        displacements_final_diffusion_ions = self.complete_disp_matrix.transpose(1, 0, 2)
        #print(displacements_final_diffusion_ions.shape)
        #displacements_final_diffusion_ions = np.cumsum(displacements_final_diffusion_ions, axis=1)
        #print(displacements_final_diffusion_ions.shape)
        dt_indices = np.arange(0, n_dt)
        msd_matrix = []
        for i in range(n_ions):
            msd_by_pairs = np.empty([0, n_dt])  # Shape of n_pairs * n_dt
            print(f"working on column {i}th")
            for j in range(n_ions):  # Loop over distinct particle pairs
                """
                if i < 25:
                    ik = i + 25
                else:
                    ik = i - 25
                if  j < 25:
                    jk = j + 25
                else:
                    jk = j - 25
                mask_tensor = self.extended_distance_matrix[:,ik,jk]
                """
                #i_k = np.random.randint(0,50)
                #j_k = np.random.randint(0,50)
                mask_tensor = self.extended_distance_matrix[:,i,j]
                ith = displacements_final_diffusion_ions[i,:,:].transpose()*mask_tensor
                jth = displacements_final_diffusion_ions[j,:,:].transpose()*mask_tensor
                ith = np.cumsum(ith.transpose(), axis=0)*sum(mask_tensor)/len(mask_tensor)
                jth = np.cumsum(jth.transpose(), axis=0)*sum(mask_tensor)/len(mask_tensor)

                #print(mask_tensor)
                if i > j:
                    msd_ij = np.zeros(n_dt)
                    msd_by_pairs = np.append(msd_by_pairs, msd_ij.reshape(1, n_dt), axis=0)
                    continue
                if i == j:
                    
                    # Self-correlation (diagonal terms)
                    #msd_ij, msd_component_ij = one_ion_msd_fft(displacements_final_diffusion_ions[i, :, :], dt_indices)
                    msd_ij, msd_component_ij = one_ion_msd_fft(ith, dt_indices)
                else:
                    
                    # Cross-correlation (off-diagonal terms)
                    #print(displacements_final_diffusion_ions[i, :, :].shape)
                    #ith = displacements_final_diffusion_ions[i, :, :].transpose()*mask_tensor
                    #jth = displacements_final_diffusion_ions[j, :, :].transpose()*mask_tensor
                    msd_ij, msd_component_ij = cross_ion_msd_fft(ith, jth, dt_indices)
                    #msd_ij, msd_component_ij = cross_ion_msd_fft(displacements_final_diffusion_ions[i, :, :]*mask_tensor, 
                    #                                            displacements_final_diffusion_ions[j, :, :][mask_tensor], dt_indices)
                
                msd_by_pairs = np.append(msd_by_pairs, msd_ij.reshape(1, n_dt), axis=0)
                #msd_component_by_pairs = np.append(msd_component_by_pairs, msd_component_ij.reshape(3, 1, n_dt), axis=1)
            #print("here", msd_by_pairs.shape)
            msd_matrix.append(msd_by_pairs)#, axis=0)

        msd_matrix = np.stack(msd_matrix)

        # Average MSD and MSD components over all pairs
        msd = np.average(msd_by_pairs, axis=0)
        msd_component = np.average(msd_component_by_pairs, axis=1)
        msd_matrix = np.where(msd_matrix,msd_matrix,msd_matrix.transpose(1,0,2))
        self.matrices_for_all_dt = msd_matrix
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
        with open(f"{self.label}_{formatted_time}_d-{self.distance_cutoff}_mode-{self.mode}fft.pkl", "wb") as f:
            pickle.dump({"matrices":msd_matrix, "distance_cutoff":self.distance_cutoff, "mode":self.mode}, f)
        return msd_matrix        
    def compute_complete_matrix_masked(self, num_index=None, init_index=0):



        disp_tensor = self.complete_disp_matrix[init_index:init_index+num_index] # (20000, 350, 3 shape)
        mask_tensor = self.extended_distance_matrix[init_index:init_index+num_index] # (20000, 350, 350 shape)
        result_matrix = np.zeros((disp_tensor.shape[1], disp_tensor.shape[1]))
        for i in range(disp_tensor.shape[1]):
            for j in range(disp_tensor.shape[1]):
                if i >= j:
                    val = np.dot(np.sum(disp_tensor[:,i][mask_tensor[:,i,j]], axis=0),np.sum(disp_tensor[:,j][mask_tensor[:,i,j]], axis=0))
                    result_matrix[i,j] = val
                    result_matrix[j,i] = val
        return result_matrix



        """
        num_cation = len(self.complete_event_matrix.keys())
        anion_index_offset = len(self.event_keys)*num_cation
        num_anion = len(self.anion_disps.keys())
        starting_Li_index = min(self.complete_event_matrix.keys())
        starting_anion_index = min(self.anion_disps.keys())
        return_matrix = np.zeros((len(self.event_keys)*len(self.complete_event_matrix.keys())+len(self.anion_disps.keys()), len(self.event_keys)*num_cation+num_anion))

        for Li_index1, events1 in self.complete_event_matrix.items():
            for Li_index2, events2 in self.complete_event_matrix.items():
                print(f"Li_index1: {Li_index1}, Li_index2: {Li_index2}")
                if Li_index1 > Li_index2:
                    continue
                mask = self.distance_matrix_cutoff[:,num_anion + Li_index1 - starting_Li_index, num_anion + Li_index2 - starting_Li_index][init_index:init_index+num_index]

                for event_index1, event1 in enumerate(self.event_keys):
                    for event_index2, event2 in enumerate(self.event_keys):
                        val1 = self.complete_event_matrix[Li_index1][event_index1, init_index:init_index+num_index][mask==True]
                        val2 = self.complete_event_matrix[Li_index2][event_index2, init_index:init_index+num_index][mask==True]
                        sum_val1 = np.sum(val1, axis=0)
                        sum_val2 = np.sum(val2, axis=0)
                        result_value = np.dot(sum_val1, sum_val2)
                        return_matrix[(Li_index1-starting_Li_index)*len(self.event_keys) + event_index1, (Li_index2-starting_Li_index)*len(self.event_keys) + event_index2] = result_value
                        return_matrix[(Li_index2-starting_Li_index)*len(self.event_keys) + event_index2, (Li_index1-starting_Li_index)*len(self.event_keys) + event_index1] = result_value

        for Li_index, events in self.complete_event_matrix.items(): # Li-loop
            for anion_index, each_anion_disp in self.anion_disps.items(): # anion loop
                print(f"Li_index: {Li_index}, anion_index: {anion_index}")
                for event_index, event in enumerate(self.event_keys):
                    mask = self.distance_matrix_cutoff[:,num_anion + Li_index - starting_Li_index, anion_index-starting_anion_index][init_index:init_index+num_index]
                    val1 = self.complete_event_matrix[Li_index][event_index, init_index:init_index+num_index][mask==True]
                    val2 = self.complete_anion_matrix[anion_index][init_index:init_index+num_index][mask==True]
                    sum_val1 = np.sum(val1, axis=0)
                    sum_val2 = np.sum(val2, axis=0)
                    result_value = np.dot(sum_val1, sum_val2)
                    return_matrix[(Li_index-starting_Li_index)*len(self.event_keys) + event_index, (anion_index_offset + anion_index - starting_anion_index)] = result_value
                    return_matrix[(anion_index_offset + anion_index - starting_anion_index), (Li_index-starting_Li_index)*len(self.event_keys) + event_index] = result_value

        for anion_index1, disps1 in self.complete_anion_matrix.items():
            for anion_index2, disps2 in self.complete_anion_matrix.items():
                print(f"anion_index1: {anion_index1}, anion_index2: {anion_index2}")
                mask = self.distance_matrix_cutoff[:,anion_index1-starting_anion_index, anion_index2-starting_anion_index][init_index:init_index+num_index]
                val1 = self.complete_anion_matrix[anion_index1][init_index:init_index+num_index][mask==True]
                val2 = self.complete_anion_matrix[anion_index2][init_index:init_index+num_index][mask==True]
                sum_val1 = np.sum(val1, axis=0)
                sum_val2 = np.sum(val2, axis=0)
                result_value = np.dot(sum_val1, sum_val2)
                return_matrix[anion_index_offset + anion_index1 - starting_anion_index, anion_index_offset + anion_index2 - starting_anion_index] = result_value
                return_matrix[anion_index_offset + anion_index2 - starting_anion_index, anion_index_offset + anion_index1 - starting_anion_index] = result_value
        return return_matrix
        """

    def compute_complete_matrix_masked_sep13(self, num_index=None, init_index=0):
        num_cation = len(self.complete_event_matrix.keys())
        anion_index_offset = len(self.event_keys)*num_cation
        num_anion = len(self.anion_disps.keys())
        starting_Li_index = min(self.complete_event_matrix.keys())
        starting_anion_index = min(self.anion_disps.keys())
        return_matrix = np.zeros((len(self.event_keys)*len(self.complete_event_matrix.keys())+len(self.anion_disps.keys()), len(self.event_keys)*num_cation+num_anion))

        for Li_index1, events1 in self.complete_event_matrix.items():
            for Li_index2, events2 in self.complete_event_matrix.items():
                print(f"Li_index1: {Li_index1}, Li_index2: {Li_index2}")
                if Li_index1 > Li_index2:
                    continue
                mask = self.distance_matrix_cutoff[:,num_anion + Li_index1 - starting_Li_index, num_anion + Li_index2 - starting_Li_index][init_index:init_index+num_index]

                for event_index1, event1 in enumerate(self.event_keys):
                    for event_index2, event2 in enumerate(self.event_keys):
                        val1 = self.complete_event_matrix[Li_index1][event_index1, init_index:init_index+num_index][mask==True]
                        val2 = self.complete_event_matrix[Li_index2][event_index2, init_index:init_index+num_index][mask==True]
                        sum_val1 = np.sum(val1, axis=0)
                        sum_val2 = np.sum(val2, axis=0)
                        result_value = np.dot(sum_val1, sum_val2)
                        return_matrix[(Li_index1-starting_Li_index)*len(self.event_keys) + event_index1, (Li_index2-starting_Li_index)*len(self.event_keys) + event_index2] = result_value
                        return_matrix[(Li_index2-starting_Li_index)*len(self.event_keys) + event_index2, (Li_index1-starting_Li_index)*len(self.event_keys) + event_index1] = result_value

        for Li_index, events in self.complete_event_matrix.items(): # Li-loop
            for anion_index, each_anion_disp in self.anion_disps.items(): # anion loop
                print(f"Li_index: {Li_index}, anion_index: {anion_index}")
                for event_index, event in enumerate(self.event_keys):
                    mask = self.distance_matrix_cutoff[:,num_anion + Li_index - starting_Li_index, anion_index-starting_anion_index][init_index:init_index+num_index]
                    val1 = self.complete_event_matrix[Li_index][event_index, init_index:init_index+num_index][mask==True]
                    val2 = self.complete_anion_matrix[anion_index][init_index:init_index+num_index][mask==True]
                    sum_val1 = np.sum(val1, axis=0)
                    sum_val2 = np.sum(val2, axis=0)
                    result_value = np.dot(sum_val1, sum_val2)
                    return_matrix[(Li_index-starting_Li_index)*len(self.event_keys) + event_index, (anion_index_offset + anion_index - starting_anion_index)] = result_value
                    return_matrix[(anion_index_offset + anion_index - starting_anion_index), (Li_index-starting_Li_index)*len(self.event_keys) + event_index] = result_value

        for anion_index1, disps1 in self.complete_anion_matrix.items():
            for anion_index2, disps2 in self.complete_anion_matrix.items():
                print(f"anion_index1: {anion_index1}, anion_index2: {anion_index2}")
                mask = self.distance_matrix_cutoff[:,anion_index1-starting_anion_index, anion_index2-starting_anion_index][init_index:init_index+num_index]
                val1 = self.complete_anion_matrix[anion_index1][init_index:init_index+num_index][mask==True]
                val2 = self.complete_anion_matrix[anion_index2][init_index:init_index+num_index][mask==True]
                sum_val1 = np.sum(val1, axis=0)
                sum_val2 = np.sum(val2, axis=0)
                result_value = np.dot(sum_val1, sum_val2)
                return_matrix[anion_index_offset + anion_index1 - starting_anion_index, anion_index_offset + anion_index2 - starting_anion_index] = result_value
                return_matrix[anion_index_offset + anion_index2 - starting_anion_index, anion_index_offset + anion_index1 - starting_anion_index] = result_value
        return return_matrix

    def compute_complete_matrix_masked_old(self, num_index=None, init_index=0):
        '''
        num_index: number of indices to include in the slicing
        init_index: starting index of the slicing (everything before this are sliced out)
        '''
        # Initialize dictionaries and list
        target_Li_disps = {}
        target_anion_disps = {}
        cumsum_vector = []
        num_index = int(num_index)
        # Process Li_disps
        for Li_index, events in self.Li_disps.items():
            target_Li_disps[Li_index] = {}
            target_events = target_Li_disps[Li_index]
            for event_type in self.event_keys:
                target_event_list = [disp for disp in events.get(event_type, [])
                                    if init_index <= disp['global_time'] < init_index + num_index]
                if target_event_list:
                    target_events[event_type] = target_event_list

        # Process anion_disps
        for anion_index, disps in self.anion_disps.items():
            filtered_disps = [disp for disp in disps 
                            if init_index <= disp['global_time'] < init_index + num_index]
            if filtered_disps:
                target_anion_disps[anion_index] = filtered_disps

        return_matrix = np.zeros((len(self.event_keys)*len(target_Li_disps.keys())+len(target_anion_disps.keys()), len(self.event_keys)*len(target_Li_disps.keys())+len(target_anion_disps.keys())))
        anion_index_offset = len(self.event_keys)*len(target_Li_disps.keys())
        num_cation = len(target_Li_disps.keys())
        num_anion = len(target_anion_disps.keys())
        starting_Li_index = min(target_Li_disps.keys())
        starting_anion_index = min(target_anion_disps.keys())
        # Calculate cumulative displacement vectors

        def map_cation_anion_index_to_distance_matrix():
            return

        first_global_time_index = init_index
        last_global_time_index = init_index + num_index -1
        counter = 0
        # Li-Li correlations
        
        for Li_index1, events1 in target_Li_disps.items():
            for Li_index2, events2 in target_Li_disps.items():
                #print(f"Li_index1: {Li_index1}, Li_index2: {Li_index2}")
                if Li_index1 > Li_index2:
                    continue
                ##
                specific_events1 = {}
                specific_events2 = {}
                #distance_info = self.distance_matrix[:, num_anion + Li_index1 - starting_Li_index, num_anion + Li_index2 - starting_Li_index] # as a function of time, Li1 and Li2 adjacent?
                #distance_info_former = [distance_info[index] < distance_cutoff for index, i  in enumerate(distance_info) if index < len(distance_info)-1]
                #distance_info_latter = [distance_info[index+1] < distance_cutoff for index, i in enumerate(distance_info) if index < len(distance_info)-1]
                distance_info = self.distance_matrix_cutoff[:,num_anion + Li_index1 - starting_Li_index, num_anion + Li_index2 - starting_Li_index]
                #distance_info = [a and b for a, b in zip(distance_info_former, distance_info_latter)]
                for event_index1, event1 in enumerate(self.event_keys):
                    if event1 in events1.keys():
                        specific_events1[event1] = [i for i in events1[event1] if distance_info[i['global_time']]]
                for event_index2, event2 in enumerate(self.event_keys):
                    if event2 in events2.keys():
                        specific_events2[event2] = [i for i in events2[event2] if distance_info[i['global_time']]]
                ## generate events1, events2 dict that are tailored to the distance cutoff. Then, in the remaining steps, I don't have to care about distance anymore.
                for event_index1, event1 in enumerate(self.event_keys):
                    for event_index2, event2 in enumerate(self.event_keys):
                        if event1 in specific_events1 and event2 in specific_events2 and len(specific_events1[event1]) > 0 and len(specific_events2[event2]) > 0:
                            
                            #cation_disp_list1 = [i for i in events1[event1] if self.distance_matrix[i['global_time'], num_anion + Li_index1 - starting_Li_index, num_anion + Li_index2 - starting_Li_index] < distance_cutoff]
                            #HERE cation_disp_list1 = [i for i in cation_disp_list1 if self.distance_matrix[i['global_time']+1, num_anion + Li_index1 - starting_Li_index, num_anion + Li_index2 - starting_Li_index] < distance_cutoff]
                            cation_disp_list1 = specific_events1[event1]
                            if len(cation_disp_list1) == 0:
                                cation_disp1 = np.zeros(3)
                            else:
                                cation_disp1 = np.sum([i['disp'] for i in cation_disp_list1], axis=0)
                            #relevant_times = [i['global_time'] for i in cation_disp_list1]
                            #cation_disp_list2 = [i for i in events2[event2] if self.distance_matrix[i['global_time'], num_anion + Li_index1 - starting_Li_index, num_anion + Li_index2 - starting_Li_index] < distance_cutoff]
                            #HERE cation_disp_list2 = [i for i in cation_disp_list2 if self.distance_matrix[i['global_time']+1, num_anion + Li_index1 - starting_Li_index, num_anion + Li_index2 - starting_Li_index] < distance_cutoff]
                            cation_disp_list2 = specific_events2[event2]
                            if len(cation_disp_list2) == 0:
                                cation_disp2 = np.zeros(3)
                            else:
                                cation_disp2 = np.sum([i['disp'] for i in cation_disp_list2], axis=0)
                            assert(cation_disp1.shape[0] == 3)
                            assert(cation_disp2.shape[0] == 3)
                            #print(cation_disp1, cation_disp2)
                            if Li_index1 == Li_index2 and event1 == event2:
                                # self term
                                assert(np.linalg.norm(cation_disp1 - cation_disp2) < 1e-6)
                                counter += 1
                            else:
                                counter += 2
                            result_value = np.dot(cation_disp1, cation_disp2)
                            #print(result_value)
                            return_matrix[(Li_index1-starting_Li_index)*len(self.event_keys) + event_index1, (Li_index2-starting_Li_index)*len(self.event_keys) + event_index2] = result_value
                            return_matrix[(Li_index2-starting_Li_index)*len(self.event_keys) + event_index2, (Li_index1-starting_Li_index)*len(self.event_keys) + event_index1] = result_value
                        else:
                            return_matrix[(Li_index1-starting_Li_index)*len(self.event_keys) + event_index1, (Li_index2-starting_Li_index)*len(self.event_keys) + event_index2] = 0
                            return_matrix[(Li_index2-starting_Li_index)*len(self.event_keys) + event_index2, (Li_index1-starting_Li_index)*len(self.event_keys) + event_index1] = 0
                            if Li_index1 == Li_index2 and event1 == event2:
                                # self term
                                counter += 1
                            else:
                                counter += 2
        

        #print(target_Li_disps.keys())
        #print(target_Li_disps[5850])
        # Li-anion correlations
        for Li_index, events in target_Li_disps.items(): # Li-loop
            for anion_index, each_anion_disp in target_anion_disps.items(): # anion loop
                #print(f"Li_index: {Li_index}, anion index: {anion_index}")

                #is_adjacent_i_j = self.distance_matrix[first_global_time_index:last_global_time_index,Li_index-starting_Li_index, num_cation + anion_index - starting_anion_index] < distance_cutoff
                specific_events_Li = {}
                #distance_info = self.distance_matrix[:, num_anion + Li_index1 - starting_Li_index, anion_index - starting_anion_index] # as a function of time, Li1 and Li2 adjacent?
                #distance_info_former = [distance_info[index] < distance_cutoff for index, i  in enumerate(distance_info) if index < len(distance_info)-1]
                #distance_info_latter = [distance_info[index+1] < distance_cutoff for index, i in enumerate(distance_info) if index < len(distance_info)-1]
                #distance_info = [a and b for a, b in zip(distance_info_former, distance_info_latter)]
                distance_info = self.distance_matrix_cutoff[:,num_anion + Li_index - starting_Li_index, anion_index - starting_anion_index]

                for event_index, event in enumerate(self.event_keys):
                    if event in events.keys():
                        specific_events_Li[event] = [i for i in events[event] if distance_info[i['global_time']]]
                specific_anions = [i for i in each_anion_disp if distance_info[i['global_time']]]


                for event_index, event_type in enumerate(self.event_keys): # event loop
                    if event_type in specific_events_Li.keys():
                        cation_disp_list = specific_events_Li[event_type]
                        #cation_disp_list = [i for i in events[event_type] if self.distance_matrix[i['global_time'], num_anion + Li_index-starting_Li_index, anion_index - starting_anion_index] < distance_cutoff]
                        #HERE cation_disp_list = [i for i in cation_disp_list if self.distance_matrix[i['global_time']+1, num_anion + Li_index-starting_Li_index, anion_index - starting_anion_index] < distance_cutoff]

                        #assert(len(cation_disp_list) == len(events[event_type]))
                        if len(cation_disp_list) == 0:
                            cation_disp = np.zeros(3)
                        else:
                            cation_disp = np.sum([i['disp'] for i in cation_disp_list], axis=0)
                        relevant_times = [i['global_time'] for i in cation_disp_list] 
                        #print("from_cation", relevant_times)
                        #print("from_anion", [i['global_time'] for i in each_anion_disp])
                        #anion_disp_list = [i for i in each_anion_disp if self.distance_matrix[i['global_time'], num_anion + Li_index-starting_Li_index,  anion_index - starting_anion_index] < distance_cutoff]
                        #HERE anion_disp_list = [i for i in anion_disp_list if self.distance_matrix[i['global_time']+1, num_anion + Li_index-starting_Li_index,  anion_index - starting_anion_index] < distance_cutoff]
                        #anion_disp_list = [i['disp'] for i in anion_disp_list]
                        anion_disp_list = [i['disp'] for i in specific_anions]
                        #anion_disp_list = [i['disp'] for i in each_anion_disp]
                        if len(anion_disp_list) == 0:
                            anion_disp = np.zeros(3)
                        else:
                            anion_disp = np.sum(anion_disp_list, axis=0)
                        assert(len(cation_disp) == 3)
                        assert(len(anion_disp) == 3)
                        result_value = np.dot(cation_disp, anion_disp)
                        return_matrix[(Li_index-starting_Li_index)*len(self.event_keys) + event_index, anion_index_offset + anion_index - starting_anion_index] = result_value
                        return_matrix[anion_index_offset + anion_index - starting_anion_index, (Li_index-starting_Li_index)*len(self.event_keys) + event_index] = result_value
                        ###
                    else: # this event did not exist, therefore the dot product is exactly zero.
                        return_matrix[(Li_index-starting_Li_index)*len(self.event_keys) + event_index, anion_index_offset + anion_index - starting_anion_index] = 0
                        return_matrix[anion_index_offset + anion_index-starting_anion_index, (Li_index-starting_Li_index)*len(self.event_keys) + event_index] = 0


        # anion-anion correlations (done)
        for anion_index1, disps1 in target_anion_disps.items():
            for anion_index2, disps2 in target_anion_disps.items():
                #print(f"Li_index: {anion_index1}, anion index: {anion_index2}")

                #specific_anions1 = {}
                #specific_anions2 = {}
                #distance_info = self.distance_matrix[:, anion_index1 - starting_anion_index,  anion_index2 - starting_anion_index] # as a function of time, Li1 and Li2 adjacent?
                #distance_info_former = [distance_info[index] < distance_cutoff for index, i  in enumerate(distance_info) if index < len(distance_info)-1]
                #distance_info_latter = [distance_info[index+1] < distance_cutoff for index, i in enumerate(distance_info) if index < len(distance_info)-1]
                #distance_info = [a and b for a, b in zip(distance_info_former, distance_info_latter)]
                distance_info = self.distance_matrix_cutoff[:,anion_index1 - starting_anion_index,  anion_index2 - starting_anion_index]

                specific_anions1 = [i for i in disps1 if distance_info[i['global_time']]]
                specific_anions2 = [i for i in disps2 if distance_info[i['global_time']]]

                if len(specific_anions1) == 0:
                    result_value = 0
                else:
                    disp_1 = [i['disp'] for i in specific_anions1]
                    disp_2 = [i['disp'] for i in specific_anions2]
                    disp_1_sum = np.sum(np.stack(disp_1), axis=0)
                    disp_2_sum = np.sum(np.stack(disp_2), axis=0)
                    assert(len(disp_1_sum) == 3)
                    assert(len(disp_2_sum) == 3)
                    result_value = np.dot(disp_1_sum, disp_2_sum)
                """
                is_adjacent_i_j = self.distance_matrix[first_global_time_index:last_global_time_index+1,anion_index1 - starting_anion_index,  anion_index2 - starting_anion_index] < distance_cutoff
                #HERE is_adjacent_i_j_next = self.distance_matrix[first_global_time_index+1:last_global_time_index+2,anion_index1 - starting_anion_index,  anion_index2 - starting_anion_index] < distance_cutoff
                is_adjacent_i_j_next = self.distance_matrix[first_global_time_index:last_global_time_index+1,anion_index1 - starting_anion_index,  anion_index2 - starting_anion_index] < distance_cutoff #HERE_added

                #print([i['disp'] for i in disps1])
                #print(is_adjacent_i_j)
                #print(len([i['disp'] for i in disps1]))
                #print(len(is_adjacent_i_j))
                assert(len([i['disp'] for i in disps1]) == len(is_adjacent_i_j))
                #print([i['disp'] for i in disps1][is_adjacent_i_j])
                #i_disp = np.sum([i['disp'] for i in disps1][is_adjacent_i_j], axis=0)
                #j_disp = np.sum([j['disp'] for j in disps2][is_adjacent_i_j], axis=0)
                if sum(is_adjacent_i_j) == 0:
                    result_value = 0
                else:
                    i_stacks = [item for item, is_adjacent, is_adjacent_next in zip([i['disp'] for i in disps1], is_adjacent_i_j, is_adjacent_i_j_next) if is_adjacent and is_adjacent_next]
                    j_stacks = [item for item, is_adjacent, is_adjacent_next in zip([i['disp'] for i in disps2], is_adjacent_i_j, is_adjacent_i_j_next) if is_adjacent and is_adjacent_next]
                    if len(i_stacks) > 0:
                        i_disp = np.sum(np.stack(i_stacks), axis=0)
                    else:
                        i_disp = np.zeros(3)
                    if len(j_stacks) > 0:
                        j_disp = np.sum(np.stack(j_stacks), axis=0)
                    else:
                        j_disp = np.zeros(3)
                    assert(len(i_disp) == 3)
                    assert(len(j_disp) == 3)
                    result_value = np.dot(i_disp, j_disp)
                """
                return_matrix[anion_index_offset + anion_index1 - starting_anion_index, anion_index_offset + anion_index2 - starting_anion_index] = result_value
                return_matrix[anion_index_offset + anion_index2 - starting_anion_index, anion_index_offset + anion_index1 - starting_anion_index] = result_value
        return return_matrix

    # define 10% to 50% of the trajectory to perform a linear fit to get the Onsager coefficients.

    

    def calculate_for_single_dt(self, dt, stepskip=10,distance_cutoff=None):
        # iterate through constant interval, different starting point
        """
        dt: width of time window (# of frames)
        stepskip: choosing one out of 10 frames for moving-window averaging
        """
        matrices = []
        init_index = 0
        return_name = f"s-{self.label}_dt-{dt}_stepskip-{stepskip}_dcut-{distance_cutoff}.pickle"
        """
        for root, dirs, files in os.walk("."):
            if return_name in files:
                print(f"Already computed for dt = {dt}")
                return
        """    

        print(f"computing for dt = {dt}")
        
        while True:
            if init_index + dt > self.n_total_frames:
                break
            #cumvec = self.compute_complete_matrix2(num_index=dt, init_index=init_index)
            if distance_cutoff == None:
                matrices.append(self.compute_complete_matrix2(num_index=dt, init_index=init_index))
            else:
                matrices.append(self.compute_complete_matrix_masked(num_index=dt, init_index=init_index))##, distance_cutoff=distance_cutoff))
            init_index += stepskip
        matrices = np.array(matrices)
        return_matrix = np.average(matrices, axis=0)
        with open(return_name, "wb") as f:
            pickle.dump(return_matrix, f)
        return return_matrix
        #return matrices, return_matrix
        # return an averaged matrix
    """
    def calculate_for_multiple_dt_parallel(self, dt_lower_fraction, dt_upper_fraction, stepskip=10, windowfactor=1.05, specific_integers=None, num_processes=8):
        dt_lower = dt_lower_fraction * self.n_total_frames
        dt_upper = dt_upper_fraction * self.n_total_frames

        def generate_integers_with_bounds(lower_bound, upper_bound, windowfactor=1.05, specific_integers=None):
            generated_integers = []
            current_integer = lower_bound
            while current_integer <= upper_bound:
                generated_integers.append(current_integer)
                current_integer = math.ceil(current_integer * windowfactor)
            if specific_integers is not None:
                specific_integers = [i for i in specific_integers if lower_bound <= i <= upper_bound]
                generated_integers = list(set(generated_integers + specific_integers))
            generated_integers.sort()
            return generated_integers

        dt_values = generate_integers_with_bounds(dt_lower, dt_upper, windowfactor=windowfactor, specific_integers=specific_integers)

        with Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(lambda time: self.calculate_for_single_dt(time, stepskip=stepskip), dt_values):
                # Process the result immediately to reduce memory usage
                matrices_for_all_dt.append(result)  # Or save to disk
                del result  # Explicitly free the memory

        matrices_for_all_dt = np.stack(matrices_for_all_dt)
        return (dt_values, matrices_for_all_dt)
    """

    def calculate_for_multiple_dt_parallel(self, dt_lower_fraction, dt_upper_fraction, stepskip=10, windowfactor=1.05, specific_integers=None, num_processes=8):#, distance_cutoff=self.distance_cutoff):
        dt_lower = dt_lower_fraction * self.n_total_frames
        dt_upper = dt_upper_fraction * self.n_total_frames

        def generate_integers_with_bounds(lower_bound, upper_bound, windowfactor=1.05, specific_integers=None):
            generated_integers = []
            current_integer = lower_bound
            while current_integer <= upper_bound:
                generated_integers.append(current_integer)
                current_integer = math.ceil(current_integer * windowfactor)
            if specific_integers is not None:
                specific_integers = [i for i in specific_integers if lower_bound <= i <= upper_bound]
                generated_integers = list(set(generated_integers + specific_integers))
            
            generated_integers = [int(i) for i in generated_integers]
            generated_integers.sort()
            return generated_integers

        dt_values = generate_integers_with_bounds(dt_lower, dt_upper, windowfactor=windowfactor, specific_integers=specific_integers)

        matrices_for_all_dt = []
        with Pool(processes=num_processes) as pool:
            # Use the helper function instead of the lambda
            results = pool.starmap(calculate_single_dt, [(self, time, stepskip, self.distance_cutoff) for time in dt_values])

        # Combine the results
        matrices_for_all_dt = np.stack(results)
        self.dt_values = dt_values
        self.matrices_for_all_dt = matrices_for_all_dt
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
        with open(f"{self.label}_{formatted_time}_d-{self.distance_cutoff}.pkl", "wb") as f:
            pickle.dump({"dt_values":dt_values, "matrices":matrices_for_all_dt, "condition":{"dt_lower":dt_lower_fraction, "dt_upper":dt_upper_fraction, "stepskip":stepskip, "windowfactor":windowfactor,
                "specific_integers":specific_integers,"distance_cutoff":self.distance_cutoff}}, f)
        return (dt_values, matrices_for_all_dt)




    def calculate_for_multiple_dt(self, dt_lower_fraction, dt_upper_fraction, stepskip=10, windowfactor = 1.05, specific_integers=None, distance_cutoff=None):
        """
        dt_lower_fraction : lower fraction of the total frames to compute moving-window averaged Onsager matrix
        dt_upper_fraction : upper fraction of the total frames to compute moving-window averaged Onsager matrix
        windowskip : increase the window width by this amount of frames
        stepskip : passed to function "calculte_for_single_dt", moving the window by this amount of frames when performing moving-window averaging
        returns a tuple of (list of number of frame indexes (dt), window-averaged Onsager matrix for each numframes)
        """
        dt_lower = dt_lower_fraction * self.n_total_frames
        dt_upper = dt_upper_fraction * self.n_total_frames


        def generate_integers_with_bounds(lower_bound, upper_bound, windowfactor=1.05, specific_integers=None):
            # Initialize the list to store generated integers
            generated_integers = []
            current_integer = lower_bound

            # Generate integers by multiplying with the windowfactor
            while current_integer <= upper_bound:
                generated_integers.append(current_integer)
                current_integer = math.ceil(current_integer * windowfactor)

            # Include specific integers if provided
            if specific_integers is not None:
                # Filter specific integers to ensure they are within the bounds
                specific_integers = [i for i in specific_integers if lower_bound <= i <= upper_bound]
                # Merge the lists and remove duplicates
                generated_integers = list(set(generated_integers + specific_integers))

            # Sort the final list of integers
            generated_integers.sort()

            return generated_integers
        matrices_for_all_dt = []
        dt_values = generate_integers_with_bounds(dt_lower, dt_upper, windowfactor=windowfactor, specific_integers=specific_integers)
        for time in dt_values:
            matrices_for_all_dt.append(self.calculate_for_single_dt(time, stepskip=stepskip, distance_cutoff = distance_cutoff))
        matrices_for_all_dt = np.stack(matrices_for_all_dt)
        self.dt_values = dt_values
        self.matrices_for_all_dt = matrices_for_all_dt
        return (dt_values, matrices_for_all_dt)

    
    def calculate_for_multiple_constant_dt(self, dt_lower_fraction, dt_upper_fraction, windowskip=10, stepskip=10, windowfactor = 1):
        """
        dt_lower_fraction : lower fraction of the total frames to compute moving-window averaged Onsager matrix
        dt_upper_fraction : upper fraction of the total frames to compute moving-window averaged Onsager matrix
        windowskip : increase the window width by this amount of frames
        stepskip : passed to function "calculte_for_single_dt", moving the window by this amount of frames when performing moving-window averaging
        returns a tuple of (list of number of frame indexes (dt), window-averaged Onsager matrix for each numframes)
        """
        dt_lower = dt_lower_fraction * self.n_total_frames
        dt_upper = dt_upper_fraction * self.n_total_frames
        dt_values = []
        matrices_for_all_dt = []
        timewindow = dt_lower
        while True:
            if timewindow > dt_upper:
                break
            dt_values.append(timewindow)
            matrices_for_all_dt.append(self.calculate_for_single_dt(timewindow, stepskip=stepskip))
            if windowfactor == 1:           
                timewindow += windowskip
            else:
                timewindow = math.ceil(timewindow * windowfactor)
        matrices_for_all_dt = np.stack(matrices_for_all_dt)
        self.dt_values = dt_values
        self.matrices_for_all_dt = matrices_for_all_dt
        return (dt_values, matrices_for_all_dt)

        # from 10% to 50% of the total dt, calculate the window-averaged matrix
        # return dt, [list of averaged matrix]

    def linear_regression_component(self, dt, matrix_list, fitting_range=None):
        """
        perform a component-wise linear regression fit
        returns a slope matrix of the same size as the input matrix
        """
        slope_matrix = np.zeros_like(matrix_list[0])
        rvalue_matrix = np.zeros_like(matrix_list[0])
        stderr_matrix = np.zeros_like(matrix_list[0])
        if fitting_range:
            for tindex, t in enumerate(dt):
                if t > fitting_range[0]:
                    t_lower = tindex
                    break
            for tindex, t in enumerate(dt):
                if t > fitting_range[1]:
                    t_upper = tindex
                    break
        else:
            t_lower = 0
            t_upper = len(dt)-1
        for i in range(matrix_list.shape[1]):
            for j in range(matrix_list.shape[2]):
                values = matrix_list[t_lower:t_upper, i, j]
                slope, intercept, rvalue, pvalue, stderr = stats.linregress(dt[t_lower:t_upper], values)
                slope_matrix[i][j] = slope
                rvalue_matrix[i][j] = rvalue
                stderr_matrix[i][j] = stderr
        self.D_matrix = slope_matrix/self.timestep/60 * (1/1000) # from A^2/ps to cm^2/s
        return slope_matrix, rvalue_matrix, stderr_matrix







    def compress_dt_matrix(self, dt, matrix_list):
        """
        perform a component-wise linear regression fit
        returns a slope matrix of the same size as the input matrix
        """
        slope_matrix = np.zeros_like(matrix_list[0])
        rvalue_matrix = np.zeros_like(matrix_list[0])
        stderr_matrix = np.zeros_like(matrix_list[0])
        num_event = len(self.event_keys)
        num_Li = len(self.Li_disps.keys())
        num_anion = len(self.anion_disps.keys())
        print(num_event, num_Li, num_anion)
        self_matrix = []
        for i in range(num_event+1): # self anion is included
            self_matrix.append([])
            for j in range(num_event+1):
                self_matrix[i].append([]) # self anion is included
        distinct_matrix = []
        for i in range(num_event+1): # distinct must include anions
            distinct_matrix.append([])
            for j in range(num_event+1):
                distinct_matrix[i].append([])
        print(num_event)
        anion_Li_matrix = []
        for i in range(num_event):
            anion_Li_matrix.append([])
        anion_matrix = [[],[]]
        for i in range(matrix_list.shape[1]):
            for j in range(matrix_list.shape[0]):
                if i == j:
                    if i < num_Li * num_event:
                        self_matrix[i%num_event][j%num_event].append(matrix_list[i,j,:]) # Li-self
                    else:
                        self_matrix[-1][-1].append(matrix_list[i, j,:]) # anion-self
                        #return 'anion_self'
                else:
                    i_iscation = i < num_Li * num_event
                    j_iscation = j < num_Li * num_event
                    if i_iscation and j_iscation:
                        if i//num_event == j//num_event:
                            self_matrix[i%num_event][j%num_event].append(matrix_list[i, j,:]) # Li-Li self
                            #return 'event_event_self'
                        else:
                            distinct_matrix[i%num_event][j%num_event].append(matrix_list[i, j,:]) # Li-Li distinct
                            #return 'event_event_distinct'
                    elif i_iscation and not j_iscation:
                        distinct_matrix[i%num_event][-1].append(matrix_list[i, j,:])
                        #return 'cation_anion'
                    elif j_iscation and not i_iscation:
                        distinct_matrix[-1][j%num_event].append(matrix_list[i, j,:])
                        #return 'cation_anion'
                    elif not i_iscation and not j_iscation:
                        distinct_matrix[-1][-1].append(matrix_list[i, j,:])

                        #return 'anion_anion'

        averaged_self = []
        for i in range(len(self_matrix)):
            templist = []
            for j in range(len(self_matrix[i])):
                if len(self_matrix[i][j]) > 0:
                    templist.append(np.sum(self_matrix[i][j], axis=0))
                else:
                    templist.append(np.zeros(len(self_matrix[0][0][0])))
            averaged_self.append(templist)
        averaged_self = np.array(averaged_self)
        print(averaged_self.shape)

        averaged_distinct = []
        for i in range(len(distinct_matrix)):
            templist = []
            for j in range(len(distinct_matrix[i])):
                templist.append(np.sum(distinct_matrix[i][j], axis=0))
            averaged_distinct.append(templist)
        averaged_distinct = np.array(averaged_distinct)
        averaged_distinct.shape
        return dt, averaged_self, averaged_distinct




def simplify_entitychange_new(entitychange):
    """
    Li_index : {event_type : {'global_time' : [], 'disp' : []}}
    """
    def merge_events(event_lists):
        event_lists = [i for i in event_lists if len(i['global_time']) > 0] # is not None]
        if len(event_lists) == 0:
            return {'global_time': np.array([]), 'disp': np.zeros((1,3))}
        global_times = np.concatenate([i['global_time'] for i in event_lists])
        displacements = np.concatenate([i['disp'] for i in event_lists])
        # Sort by global_times and reorder displacements accordingly
        sort_indices = np.argsort(global_times)
        global_times = global_times[sort_indices]
        displacements = displacements[sort_indices]
        return {'global_time': global_times, 'disp': displacements}
    rv = {}
    for Li_index in entitychange.keys():
        rv[Li_index] = {'0-0':entitychange[Li_index]['0-0'],
                        '0-1':merge_events([entitychange[Li_index]['0-1'], entitychange[Li_index]['0-2'], entitychange[Li_index]['0-3']]), 
                        '1-0':merge_events([entitychange[Li_index]['1-0'], entitychange[Li_index]['2-0'], entitychange[Li_index]['3-0']]), 
                        '1-1':merge_events([entitychange[Li_index]['1-1'], entitychange[Li_index]['1-2'], entitychange[Li_index]['1-3'],
                                             entitychange[Li_index]['2-1'], entitychange[Li_index]['2-2'], entitychange[Li_index]['2-3'],
                                             entitychange[Li_index]['3-1'], entitychange[Li_index]['3-2'], entitychange[Li_index]['3-3']]) 
                        }
    return rv

def simplify_entitychange_Li(entitychange):
    rv = {}
    for Li_index in entitychange.keys():
        rv[Li_index] = {'vibration': [], 'single_ion_hop': [], 'concerted_hop': [], 'counter_ion_hop': [], 'X': []}
        for event_type in entitychange[Li_index].keys():
            # Vibration events
            if event_type in ["0-0-0", "0-0-1"]:
                rv[Li_index]['vibration'].extend(entitychange[Li_index][event_type])
            # Single-ion hops (no Li neighbor or Li neighbor leaves during hop)
            elif event_type in ["0-3-0", "1-3-0", "2-3-1", "3-3-1"]:
                rv[Li_index]['single_ion_hop'].extend(entitychange[Li_index][event_type])
            # Concerted hops (with Li neighbor during hop)
            elif event_type in ["0-3-1", "1-3-1"]:
                rv[Li_index]['concerted_hop'].extend(entitychange[Li_index][event_type])
            # Counter ion hops (Li neighbor approaches or leaves without self hop)
            elif event_type in ["1-0-0", "1-0-1", "2-0-1", "3-0-1"]:
                rv[Li_index]['counter_ion_hop'].extend(entitychange[Li_index][event_type])
            else:
                rv[Li_index]['X'].extend(entitychange[Li_index][event_type])
            # All other combinations marked with X in comments are ignored
    return rv

def concatenate_events(entity_change_dict, within_entity_dict):
    """
    remove 0-0 event from entity_change_dict and add all of the new events from within_entity_dict
    """
    rv = {}
    for Li_index in entity_change_dict.keys():
        rv[Li_index] = {}
        for event_type in entity_change_dict[Li_index].keys():
            if event_type == "0-0":
                continue
            rv[Li_index][event_type] = entity_change_dict[Li_index][event_type]
        for event_type in within_entity_dict[Li_index].keys():
            rv[Li_index][event_type] = within_entity_dict[Li_index][event_type]
    return rv


