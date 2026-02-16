import numpy as np

# import torch
import os, sys

# print(sys.path)
# sys.path.append('/home/gridsan/pleon/proj/repos')
# from ForceFieldNet.ffnet.utils.trajectory_analysis import (plot_calc_rdf,
# get_coords_PDB_rdf,read_xyz, save_params)
from itertools import permutations
import itertools
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
#import torch
import ase
from pymatgen.core import Structure

MASS_TO_NAME = {
    12.0112: "C",
    15.9994: "O",
    1.00797: "H",
    6.941: "Li",
    18.9984: "F",
    32.064: "S",
    14.0067: "N",
}
TYPE_TO_NAME_TRI = {
    6: "O,PL1",
    1: "C,PL1",
    2: "C,PL1",
    3: "C,PL1",
    4: "C,PL1",
    90: "Li,CA1",
    91: "C,AN1",
    92: "F,AN1",
    93: "N,AN1",
    94: "O,AN1",
    95: "S,AN1",
}
TYPE_TO_NAME_TRI_FSI = {
    6: "O,PL1",
    1: "C,PL1",
    2: "C,PL1",
    3: "C,PL1",
    4: "C,PL1",
    90: "Li,CA1",
    91: "F,AN1",
    92: "N,AN1",
    93: "O,AN1",
    94: "S,AN1",
}
# for fsi, F = 91, N = 92, O=93, S = 94
# so currently,
"""
F -> C
N -> F
O -> N
S -> O
"""


def get_coords_lammpstrj_rdf(
    folder,
    simu_length,  # ns
    save_interval=5,  # ps
    frame_save_amount=100,
    anion="",
):
    """
    Saves wrapped 100 xyz frames at the beginning of the simulation
    at the middle of it and at the end.
    """
    if anion == "":
        type_to_name = TYPE_TO_NAME_TRI
    elif anion == "_fsi":
        type_to_name = TYPE_TO_NAME_TRI_FSI
    else:
        raise NameError("need to specify TRI types for this anion!!")
    xyz_1 = []
    xyz_2 = []
    xyz_3 = []
    atom_names = []
    curr_frame = 0

    frame_count = int(simu_length * 1000 / save_interval)
    frame_intervals = [frame_save_amount, int(frame_count / 2), frame_count]

    frame_int_ind = 0
    num_col = -5
    with open(f"{folder}/traj.lammpstrj", "r") as f:
        for ind, line in enumerate(f):
            if ind == 5:  # get box bounds to unwrap, assuming that constant volume
                values = line.split()
                box_len = float(values[1]) - float(values[0])
            if "ITEM: ATOMS id mol" in line:
                num_col = len(line.split()) - 2
                curr_frame += 1
                # if curr_frame % 10 == 0:
                #     print(curr_frame)
                continue
            values = line.split()
            if len(values) == num_col:
                if (
                    curr_frame > frame_intervals[0] - frame_save_amount
                    and curr_frame < frame_intervals[0]
                ):
                    x = float(values[5])
                    y = float(values[6])
                    z = float(values[7])
                    xyz_1.append(np.array([x, y, z], dtype=np.float32))
                    if curr_frame < 2:
                        values = line.split()
                        if int(values[2]) in type_to_name:
                            atom_names.append(type_to_name[int(values[2])])
                        else:
                            atom_names.append("H,PL1")
                elif (
                    curr_frame > frame_intervals[1] - frame_save_amount
                    and curr_frame < frame_intervals[1]
                ):
                    x = float(values[5])
                    y = float(values[6])
                    z = float(values[7])
                    xyz_2.append(np.array([x, y, z], dtype=np.float32))
                elif (
                    curr_frame > frame_intervals[2] - frame_save_amount
                    and curr_frame < frame_intervals[2]
                ):
                    x = float(values[5])
                    y = float(values[6])
                    z = float(values[7])
                    xyz_3.append(np.array([x, y, z], dtype=np.float32))
                elif curr_frame > frame_intervals[2]:
                    break
    with open(f"{folder}/xyz_wrapped_rdf_beginning.txt", "w") as f:
        for xyz in xyz_1:
            f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")
    with open(f"{folder}/xyz_wrapped_rdf_middle.txt", "w") as f:
        for xyz in xyz_2:
            f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")
    with open(f"{folder}/xyz_wrapped_rdf_end.txt", "w") as f:
        for xyz in xyz_3:
            f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")
    with open(f"{folder}/atom_names_rdf.txt", "w") as f:
        for i in atom_names:
            f.write(f"{i}\n")
    with open(f"{folder}/box_len.txt", "w") as f:
        f.write(str(box_len))
    print(f"RDF frames have been saved at {folder}")


def get_unwrapped_coords_lammpstrj_rdf(
    folder,
    simu_length,  # ns
    save_interval=5,  # ps
    frame_save_amount=100,
    anion="",
):
    """
    Saves unwrapped 100 xyz frames at the beginning of the simulation
    at the middle of it and at the end.
    """
    xyz_1 = []
    xyz_2 = []
    xyz_3 = []
    atom_names = []
    curr_frame = 0

    if anion == "":
        type_to_name = TYPE_TO_NAME_TRI
    elif anion == "_fsi":
        type_to_name = TYPE_TO_NAME_TRI_FSI
    else:
        raise NameError("need to specify TRI types for this anion!!")
    frame_count = int(simu_length * 1000 / save_interval)
    num_col = -5
    if frame_save_amount is None:
        with open(f"{folder}/traj.lammpstrj", "r") as f:
            for ind, line in enumerate(f):
                if ind == 5:  # get box bounds to unwrap, assuming that constant volume
                    values = line.split()
                    box_len = float(values[1]) - float(values[0])
                if "ITEM: ATOMS id mol" in line:
                    num_col = len(line.split()) - 2
                    curr_frame += 1
                    continue
                values = line.split()
                if len(values) == num_col:
                    x = float(values[5]) + box_len * float(values[-3])
                    y = float(values[6]) + box_len * float(values[-2])
                    z = float(values[7]) + box_len * float(values[-1])
                    xyz_1.append(np.array([x, y, z], dtype=np.float32))
                    import pdb

                    pdb.set_trace()
                    if curr_frame < 2:
                        values = line.split()
                        if int(values[2]) in type_to_name:
                            atom_names.append(
                                type_to_name[int(values[2])] + f",{values[1]}"
                            )
                        else:
                            atom_names.append(f"H,PL1,{values[1]}")
        with open(f"{folder}/xyz_unwrapped_rdf_all.txt", "w") as f:
            for xyz in xyz_1:
                f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")
        with open(f"{folder}/atom_names_rdf.txt", "w") as f:
            for i in atom_names:
                f.write(f"{i}\n")
        print(f"RDF frames have been saved at {folder}")
        return

    # frame_intervals = [frame_save_amount, int(frame_count / 2), frame_count]

    # frame_int_ind = 0
    # num_col = -5
    # with open(f"{folder}/traj.lammpstrj", "r") as f:
    #     for ind, line in enumerate(f):
    #         if ind == 5:  # get box bounds to unwrap, assuming that constant volume
    #             values = line.split()
    #             box_len = float(values[1]) - float(values[0])
    #         if "ITEM: ATOMS id mol" in line:
    #             num_col = len(line.split()) - 2
    #             curr_frame += 1
    #             continue
    #         values = line.split()
    #         if len(values) == num_col:
    #             if (
    #                 curr_frame > frame_intervals[0] - frame_save_amount
    #                 and curr_frame < frame_intervals[0]
    #             ):
    #                 x = float(values[5]) + box_len * float(values[-3])
    #                 y = float(values[6]) + box_len * float(values[-2])
    #                 z = float(values[7]) + box_len * float(values[-1])
    #                 xyz_1.append(np.array([x, y, z], dtype=np.float32))
    #                 if curr_frame < 2:
    #                     values = line.split()
    #                     if int(values[2]) in type_to_name:
    #                         atom_names.append(type_to_name[int(values[2])])
    #                     else:
    #                         atom_names.append("H,PL1")
    #             elif (
    #                 curr_frame > frame_intervals[1] - frame_save_amount
    #                 and curr_frame < frame_intervals[1]
    #             ):
    #                 x = float(values[5]) + box_len * float(values[-3])
    #                 y = float(values[6]) + box_len * float(values[-2])
    #                 z = float(values[7]) + box_len * float(values[-1])
    #                 xyz_2.append(np.array([x, y, z], dtype=np.float32))
    #             elif (
    #                 curr_frame > frame_intervals[2] - frame_save_amount
    #                 and curr_frame < frame_intervals[2]
    #             ):
    #                 x = float(values[5]) + box_len * float(values[-3])
    #                 y = float(values[6]) + box_len * float(values[-2])
    #                 z = float(values[7]) + box_len * float(values[-1])
    #                 xyz_3.append(np.array([x, y, z], dtype=np.float32))
    #             elif curr_frame > frame_intervals[2]:
    #                 break
    # with open(f"{folder}/xyz_unwrapped_rdf_beginning.txt", "w") as f:
    #     for xyz in xyz_1:
    #         f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")
    # with open(f"{folder}/xyz_unwrapped_rdf_middle.txt", "w") as f:
    #     for xyz in xyz_2:
    #         f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")
    # with open(f"{folder}/xyz_unwrapped_rdf_end.txt", "w") as f:
    #     for xyz in xyz_3:
    #         f.write(f"{xyz[0]},{xyz[1]},{xyz[2]}\n")
    # with open(f"{folder}/atom_names_rdf.txt", "w") as f:
    #     for i in atom_names:
    #         f.write(f"{i}\n")
    # with open(f"{folder}/box_len.txt", "w") as f:
    #     f.write(str(box_len))
    # print(f"RDF frames have been saved at {folder}")


def torch_nbr_list(
    xyz, cutoff, cell, directed=True, pbc_flag=True, requires_large_offsets=True
):
    """Pytorch implementations of nbr_list for minimum image convention, the offsets are only limited to 0, 1, -1:
    it means that no pair interactions is allowed for more than 1 periodic box length. It is so much faster than
    neighbor_list algorithm in ase.
    It is similar to the output of neighbor_list("ijS", atomsobject, cutoff) but a lot faster
    Args:
        atomsobject (TYPE): Description
        cutoff (float): cutoff for
        device (str, optional): Description
        requires_large_offsets: to get offsets beyond -1,0,1
    Returns:
        i, j, cutoff: just like ase.neighborlist.neighbor_list
    """

    if pbc_flag:
        # check if sufficiently large to run the "fast" nbr_list function
        # otherwise, default to the "robust" nbr_list function below for small cells
        if np.all(2 * cutoff < cell.diagonal()):
            # fidn the distance from one to another
            dis_mat = xyz[None, :, :] - xyz[:, None, :]
            cell_dim = cell.diagonal()
            # cell_dim = torch.Tensor(np.array(atomsobject.get_cell())).diag().to(device)
            if requires_large_offsets:
                shift = np.round(np.divide(dis_mat, cell_dim))
                offsets = -shift
            else:
                offsets = -dis_mat.ge(0.5 * cell_dim).to(torch.float) + dis_mat.lt(
                    -0.5 * cell_dim
                ).to(torch.float)

            dis_mat = dis_mat + offsets * cell_dim
            dis_sq = np.sum(np.power(dis_mat, 2), axis=-1)
            mask = (dis_sq < cutoff**2) & (dis_sq != 0)
            # print('min:', np.min(np.sqrt(dis_sq[dis_sq!=0])), np.max(np.sqrt(dis_sq[dis_sq!=0])))
            nbr_list = mask.nonzero()
            offsets = offsets[nbr_list[0], nbr_list[1], :]
        # nbr_list[:, 1], :].detach().to("cpu").numpy()
        else:
            raise NameError("only works if cutoff << cell_param for now")
            # "robust" nbr_list function for all cells (pbc)
            xyz = torch.Tensor(atomsobject.get_positions(wrap=True)).to(device)

            # since we are not wrapping
            # retrieve the shift vectors that would be equivalent to wrapping
            positions = atomsobject.get_positions(wrap=True)
            unwrapped_positions = atomsobject.get_positions(wrap=False)
            shift = positions - unwrapped_positions
            cell = atomsobject.cell
            cell = np.broadcast_to(
                cell.T, (shift.shape[0], cell.shape[0], cell.shape[1])
            )
            shift = np.linalg.solve(cell, shift).round().astype(int)

            # estimate getting close to the cutoff with supercell expansion
            cell = atomsobject.cell
            a_mul = int(np.ceil(cutoff / np.linalg.norm(cell[0]))) + 1
            b_mul = int(np.ceil(cutoff / np.linalg.norm(cell[1]))) + 1
            c_mul = int(np.ceil(cutoff / np.linalg.norm(cell[2]))) + 1
            supercell_matrix = np.array([[a_mul, 0, 0], [0, b_mul, 0], [0, 0, c_mul]])
            supercell = clean_matrix(supercell_matrix @ cell)

            # cartesian lattice points
            lattice_points_frac = lattice_points_in_supercell(supercell_matrix)
            lattice_points = np.dot(lattice_points_frac, supercell)
            # need to get all negative lattice translation vectors but remove duplicate 0 vector
            zero_idx = np.where(
                np.all(lattice_points.__eq__(np.array([0, 0, 0])), axis=1)
            )[0][0]
            lattice_points = np.concatenate(
                [lattice_points[zero_idx:, :], lattice_points[:zero_idx, :]]
            )

            N = len(lattice_points)
            # perform lattice translation vectors on positions
            lattice_points_T = torch.tile(
                torch.from_numpy(lattice_points),
                (len(xyz),) + (1,) * (len(lattice_points.shape) - 1),
            ).to(device)
            xyz_T = torch.repeat_interleave(xyz.view(-1, 1, 3), N, dim=1)
            xyz_T = xyz_T + lattice_points_T.view(xyz_T.shape)
            diss = xyz_T[None, :, None, :, :] - xyz_T[:, None, :, None, :]
            diss = diss[:, :, 0, :, :]
            dis_sq = diss.pow(2).sum(-1)
            mask = (dis_sq < cutoff**2) & (dis_sq != 0)
            nbr_list = mask.nonzero()[:, :2]
            offsets = lattice_points_T.view(xyz_T.shape)[
                mask.nonzero()[:, 1], mask.nonzero()[:, 2]
            ]

            # get offsets as original integer multiples of lattice vectors
            cell = np.broadcast_to(
                cell.T, (offsets.shape[0], cell.shape[0], cell.shape[1])
            )
            # offsets = offsets.detach().to("cpu").numpy()
            offsets = np.linalg.solve(cell, offsets).round().astype(int)

            # add shift to offsets with the right indices according to pairwise nbr_list
            offsets = torch.from_numpy(offsets).int().to(device)
            shift = torch.from_numpy(shift).int().to(device)

            # index shifts by atom but then apply shifts to pairwise interactions
            # get shifts for each atom i and j that would be equivalent to wrapping
            # convention is j - i for get_rij with NNs
            shift_i = shift[nbr_list[:, 0]]
            shift_j = shift[nbr_list[:, 1]]
            offsets = (shift_j - shift_i + offsets).detach().to("cpu").numpy()

    else:
        raise NameError("only works if pbc for now")
        xyz = torch.Tensor(atomsobject.get_positions(wrap=False)).to(device)
        dis_mat = xyz[None, :, :] - xyz[:, None, :]
        dis_sq = dis_mat.pow(2).sum(-1)
        mask = (dis_sq < cutoff**2) & (dis_sq != 0)
        nbr_list = mask.nonzero(as_tuple=False)

    if not directed:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]
    # i, j = nbr_list[:, 0].detach().to("cpu").numpy(
    # ), nbr_list[:, 1].detach().to("cpu").numpy()
    i, j = nbr_list[0], nbr_list[1]

    if pbc_flag:
        offsets = offsets
    else:
        offsets = np.zeros((nbr_list.shape[0], 3))

    return i, j, offsets


# def get_molecules(nbr_list, all_ion_atoms, an1, an2):
#     clusters = np.zeros(len(all_ion_atoms), dtype=int)
#     for i,cur in enumerate(all_ion_atoms):
#         mm=max(clusters)
#         #oxy_neighbors contains all the atoms that the current atom is in the cutoff of
#         oxy_neighbors = nbr_list[nbr_list[:,0] == cur].reshape(-1)
#         #if this has no neighbors, then the current atom will belong to a new cluster
#         if len(oxy_neighbors)==0:
#             clusters[i]=mm+1
#             continue
#         #if all the neighbors are in unassigned clusters and the current atom has already been assigned,
#         #assign the neighbors to the current cluster
#         if (clusters[oxy_neighbors]==0).all() and clusters[i]!=0:
#             clusters[oxy_neighbors]=clusters[i]
#         #if all the niehbors are in unassigned clusters and the curent cluster is also unassigned
#         #assign the current atom and all the neighbors to a new cluster
#         elif (clusters[oxy_neighbors]==0).all() and clusters[i]==0:
#             clusters[oxy_neighbors]=mm+1
#             clusters[i]=mm+1
#         #if not all the neighbors are in unassigned clusters (so if at least one neighbor has been assigned)
#         #and the current atom has not been assigned yet, assign the current atom and all neighbors to
#         #the lowest cluster name (choosing the minimum cluster name as default is arbitrary,
#         #as long as all have same)
#         elif (clusters[oxy_neighbors]==0).all() == False and clusters[i]==0:
#             clusters[i]=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])
#             clusters[oxy_neighbors]=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])
#         #if not all the neighbors are in unassigned clusters and current atom has been assigned,
#         elif (clusters[oxy_neighbors]==0).all() == False and clusters[i]!=0:
#             #temporary is the first you only take the clusters that are neighbors
#             #then you only take the nonzero of those neighboring clusters
#             #then you only take the ones that are not the minimum of those nonzero neighboring clusters
#             # tmp=clusters[oxy_neighbors]
#             #     [clusters[oxy_neighbors]!=0]
#             #     [clusters[oxy_neighbors][clusters[oxy_neighbors]!=0]!=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])]
#             tmp = clusters[oxy_neighbors]
#             tmp = tmp[tmp != 0]
#             tmp = tmp[tmp != min(tmp)]
#             #set the current cluster to the minimum of these nonzero, neighboring clusters
#             clusters[i]=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])
#             #then, set all the neighboring atoms to this same minimum cluster name
#             clusters[oxy_neighbors]=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])
#             #temp contains all the nonzero neighboring cluster types
#             #for each of these nonzero neighboring cluster types, change them to the overall minimum same clustering name
#             for tr in tmp:
#                 clusters[np.where(clusters==tr)[0]]=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])
#     molecules=[]
#     for i in range(1,max(clusters)+1):
#         if np.size(np.where(clusters==i)[0])==0:
#             continue
#         molecules.append(np.where(clusters==i)[0])
#     #only keep one oxygen from each anion in molecule for proper counting of Li vs TFSI
#     for i, cur_mol in enumerate(molecules):
#         for j, cur in enumerate(cur_mol):
#             curan1 = np.where(an1==cur)
#             curan2 = np.where(an2==cur)
#             #if the current atom in the molecule is one of anion 1,
#             #set the atom to the first atom of that anion
#             #or do the same if the atom is part of anion 2
#             if len(curan1) > 0 and  len(np.where(an1==cur)[0]) > 0:
#                 molecules[i][j] = an1[curan1[0][0]][0]
#             elif len(curan2) > 0 and  len(np.where(an2==cur)[0]) > 0:
#                 molecules[i][j] = an2[curan2[0][0]][0]
#         #only keep one atom of each anion
#         molecules[i] = np.unique(molecules[i])
#     return molecules


def wrap_xyz(xyz, cell):
    for i, cur in enumerate(xyz):
        for j, cur_atom in enumerate(cur):
            # wrap the coordinates into the cell box
            image = cur_atom // cell.diagonal()
            cur_atom = cur_atom - cell.diagonal() * image
            xyz[i][j] = cur_atom
    return xyz


def read_xyz(
    folder_path,
    atom_name,
    xyz_name,
    include_resid=False,
):
    with open(f"{folder_path}/{atom_name}", "r") as f:
        atom_names = []
        atom_names_long = []
        res_ids = []
        for line in f:
            value = line.split(",")
            atom_names.append(value[0])
            if include_resid:  # can only be used with atom_names_rdf.txt
                res_ids.append(int(value[2]))
            atom_names_long.append(value[0] + "-" + value[1].strip())
    with open(f"{folder_path}/{xyz_name}", "r") as f:
        xyz = []
        for line in f:
            values = line.split(",")
            xyz.append(np.array(values[0:3], dtype=np.float64))
    xyz = np.concatenate(xyz).reshape(-1, len(atom_names), 3)
    if include_resid:
        return xyz, atom_names, atom_names_long, res_ids
    else:
        return xyz, atom_names, atom_names_long

def read_xyz_monomer_unit(
    folder_path,
    atom_name,
):
    """
    right now, atom_names_all.rdf.txt 3rd column is the monomer unit index.
    0 : monomer-A
    1 : monomer-B
    2 : carbon cap
    3 : Li or TFSI
    """
    with open(f"{folder_path}/{atom_name}", "r") as f:
        monomer_unit_ids = []
        for line in f:
            value = line.split(",")
            monomer_unit_ids.append(int(value[2]))
    return monomer_unit_ids


def plot_calc_rdf(
    xyz_rdf_unwrp,
    folder,
    one_name: list,
    two_names: list,
    cell,
    atom_names_long_rdf,
    names=None,
    temperature=353,
    plot_name=None,
    plot_flag=True,
):
    rdfs = []
    bins = []
    for i, two_name in enumerate(two_names):
        figsize = (6, 5)
        fig_scalingfactor = figsize[1] / 5
        fontsize = 20 * fig_scalingfactor
        plt.rcParams.update({"font.size": fontsize})
        labelsize = 20 * fig_scalingfactor
        titelsize = labelsize * 1.2
        pad = labelsize / 3
        tickwidth = 2.5 * fig_scalingfactor
        maj_tick_size = 6 * fig_scalingfactor
        min_tick_size = 3 * fig_scalingfactor
        dpi = 400

        axs2 = []
        # min_idx = 70
        min_idx = 68
        idxs_one = []
        idxs_two = []
        for ind, atom in enumerate(atom_names_long_rdf):
            for name in one_name:
                if name == atom:
                    idxs_one.append(ind)
            for name in two_name:
                if name == atom:
                    idxs_two.append(ind)
        if len(idxs_one) == 0 or len(idxs_two) == 0:
            print(
                "the combination of",
                one_name,
                "and",
                two_name,
                "is not in this simulation",
            )
            continue
        # fig, subplot_dict = plt.subplot(mosaic_str, figsize=figsize, sharex=True, sharey=False)
        fig, axs = plt.subplots(1, 1, figsize=figsize, sharex=False, sharey=False)
        num_frames = xyz_rdf_unwrp.shape[0]
        cell_lengths = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])

        if one_name == two_name:
            # if you are getting the rdf between an atom and other atoms of the same type, you do not want repeats, so use
            # itertools.combinations
            idxs = np.array(list(itertools.combinations(idxs_one, 2)))
        else:
            idxs = np.array(list(itertools.product(idxs_one, idxs_two)))
        dist_mat = np.zeros(shape=(num_frames, idxs.shape[0]), dtype=np.float32)

        for ii, idx in enumerate(tqdm(idxs)):
            dist_mat[:, ii] = mini_dist(
                xyz_rdf_unwrp[:, idx[0]], xyz_rdf_unwrp[:, idx[1]], cell_lengths
            )

        # dist_mat = dist_mat.reshape((num_frames, len(idxs_one), len(idxs_two)))
        dx = 0.05
        V = cell_lengths[0] * cell_lengths[1] * cell_lengths[2]
        centers, gr, coord = RDF(
            dist_mat,
            dx=dx,
            Na=len(idxs_one),
            Nb=len(idxs_two),
            V=V,
            same_flag=(one_name == two_name),
        )
        bins.append(centers)
        rdfs.append(gr)
        center_val = centers[min_idx]
        coord_val = coord[min_idx]

        axs.plot(centers, gr, linewidth=3, label=r"$g(r)$", color="blue")

        ax02 = axs.twinx()
        axs2.append(ax02)
        ax02.plot(
            centers, coord, linewidth=3, label=r"$\int dr g(r)$", color="darkorange"
        )

        axs.text(
            0.375,
            0.9,
            "CN(%1.1f $\AA$): %1.2f" % (center_val, coord_val),
            transform=axs.transAxes,
            fontsize=labelsize * 0.8,
        )
        # axs.set_ylim([0, gr.max()*1.1])
        axs.set_ylim([0, 5.5])

        if not plot_name:
            plot_name = folder.split("/")[-1]
            title_name = f"path = {plot_name}, temp = {int(temperature)}"
        else:
            title_name = f"{plot_name}, temp = {int(temperature)}"
        axs.set_title(f"{title_name}", fontsize=titelsize)

        for ax in [axs]:
            print(one_name)
            if one_name == ["Li-CA1"] and two_name == ["Li-CA1"]:
                ax.set_xlim([0, 20])
            else:
                ax.set_xlim([0, 10])
            ax.axhline(1.0, linewidth=2, color="black")
            ax.axvline(3.25, linewidth=1, color="grey", linestyle="--")
            ax.axvline(2, linewidth=1, color="grey", linestyle="--")
            # ax.axvline(3, linewidth=1, color='grey', linestyle='--')

            ax.tick_params(
                axis="y",
                length=maj_tick_size,
                width=tickwidth,
                labelsize=labelsize,
                pad=pad,
                direction="in",
                labelcolor="blue",
            )
            ax.tick_params(
                axis="x",
                length=maj_tick_size,
                width=tickwidth,
                labelsize=labelsize,
                pad=pad,
                direction="in",
            )

            for key in ax.spines.keys():
                ax.spines[key].set_linewidth(tickwidth)

        for ax in axs2:
            ax.tick_params(
                axis="y",
                length=maj_tick_size,
                width=tickwidth,
                labelsize=labelsize,
                pad=pad,
                direction="in",
                labelcolor="darkorange",
            )
            ax.set_ylim([0, 10])

        plt.tight_layout()
        # if names:
        #     plt.savefig(f"{folder}/RDF_{names[i]}.png", dpi=dpi)
        #     plt.savefig(f"{folder}/RDF_{names[i]}.svg", dpi=dpi)
        # else:
        #     plt.savefig(f"{folder}/RDF_{i}.png", dpi=dpi)
        #     plt.savefig(f"{folder}/RDF_{i}.svg", dpi=dpi)
        if plot_flag:
            plt.show()
        else:
            plt.close()
    return bins, rdfs


def mini_dist(pos1, pos2, cell_vecs):
    abs_dist = np.abs(pos1 - pos2) % (cell_vecs)
    pbc_dist = np.stack([np.abs(abs_dist - cell_vecs), abs_dist]).min(axis=0)
    return np.linalg.norm(pbc_dist, axis=-1)


def RDF(dists, dx, Na, Nb, V, same_flag=False):
    data = dists.flatten()
    maxv = data.max()
    dx2 = dx / 2
    nbins = int(np.ceil(maxv / dx))
    binc = np.arange(nbins) * dx + dx2
    hist = np.histogram(data, bins=nbins, range=(0, maxv))[0].astype(float)
    hist /= dists.shape[0]

    gr = hist * (V / (4.0 * np.pi * Na * Nb))
    gr /= np.power(binc, 2) * dx
    crd = np.zeros(shape=(nbins,), dtype=float)
    if same_flag:
        gr = 2 * gr
        for binn in range(nbins):
            crd[binn] = 2 * hist[: (binn + 1)].sum() / Na
    else:
        for binn in range(nbins):
            crd[binn] = hist[: (binn + 1)].sum() / Na

    return binc, gr, crd


def get_molecules(nbr_list, all_ion_atoms, an1=None, an2=None):
    """
    nbr_list is the list of pairs of atoms within cutoff of each other
    all_ion_atoms is the list of the indices of all the atoms
    an1 = which atoms are together

    """
    clusters = np.zeros(len(all_ion_atoms), dtype=int)
    for i, cur in enumerate(all_ion_atoms):
        mm = max(clusters)
        # oxy_neighbors contains all the atoms that the current atom is in the cutoff of
        oxy_neighbors = nbr_list[nbr_list[:, 0] == cur].reshape(-1)
        # if this has no neighbors, then the current atom will belong to a new cluster
        if len(oxy_neighbors) == 0:
            clusters[i] = mm + 1
            continue
        # if all the neighbors are in unassigned clusters and the current atom has already been assigned,
        # assign the neighbors to the current cluster
        if (clusters[oxy_neighbors] == 0).all() and clusters[i] != 0:
            clusters[oxy_neighbors] = clusters[i]
        # if all the niehbors are in unassigned clusters and the curent cluster is also unassigned
        # assign the current atom and all the neighbors to a new cluster
        elif (clusters[oxy_neighbors] == 0).all() and clusters[i] == 0:
            clusters[oxy_neighbors] = mm + 1
            clusters[i] = mm + 1
        # if not all the neighbors are in unassigned clusters (so if at least one neighbor has been assigned)
        # and the current atom has not been assigned yet, assign the current atom and all neighbors to
        # the lowest cluster name (choosing the minimum cluster name as default is arbitrary,
        # as long as all have same)
        elif (clusters[oxy_neighbors] == 0).all() == False and clusters[i] == 0:
            clusters[i] = min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])
            clusters[oxy_neighbors] = min(
                clusters[oxy_neighbors][clusters[oxy_neighbors] != 0]
            )
        # if not all the neighbors are in unassigned clusters and current atom has been assigned,
        elif (clusters[oxy_neighbors] == 0).all() == False and clusters[i] != 0:
            # temporary is the first you only take the clusters that are neighbors
            # then you only take the nonzero of those neighboring clusters
            # then you only take the ones that are not the minimum of those nonzero neighboring clusters
            # tmp=clusters[oxy_neighbors]
            #     [clusters[oxy_neighbors]!=0]
            #     [clusters[oxy_neighbors][clusters[oxy_neighbors]!=0]!=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])]
            tmp = clusters[oxy_neighbors]
            tmp = tmp[tmp != 0]
            tmp = tmp[tmp != min(tmp)]
            # set the current cluster to the minimum of these nonzero, neighboring clusters
            clusters[i] = min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])
            # then, set all the neighboring atoms to this same minimum cluster name
            clusters[oxy_neighbors] = min(
                clusters[oxy_neighbors][clusters[oxy_neighbors] != 0]
            )
            # temp contains all the nonzero neighboring cluster types
            # for each of these nonzero neighboring cluster types, change them to the overall minimum same clustering name
            for tr in tmp:
                clusters[np.where(clusters == tr)[0]] = min(
                    clusters[oxy_neighbors][clusters[oxy_neighbors] != 0]
                )
    molecules = []
    for i in range(1, max(clusters) + 1):
        if np.size(np.where(clusters == i)[0]) == 0:
            continue
        molecules.append(np.where(clusters == i)[0])
    # only keep one oxygen from each anion in molecule for proper counting of Li vs TFSI
    if an1:
        for i, cur_mol in enumerate(molecules):
            for j, cur in enumerate(cur_mol):
                curan1 = np.where(an1 == cur)
                if an2:
                    curan2 = np.where(an2 == cur)
                # if the current atom in the molecule is one of anion 1,
                # set the atom to the first atom of that anion
                # or do the same if the atom is part of anion 2
                if len(curan1) > 0 and len(np.where(an1 == cur)[0]) > 0:
                    if len(an1.shape) > 1:
                        molecules[i][j] = an1[curan1[0][0]][0]
                    else:
                        molecules[i][j] = an1[curan1[0][0]]
                elif an2 and len(curan2) > 0 and len(np.where(an2 == cur)[0]) > 0:
                    if len(an2.shape) > 1:
                        molecules[i][j] = an2[curan2[0][0]][0]
                    else:
                        molecules[i][j] = an2[curan2[0][0]]
            # only keep one atom of each anion
            molecules[i] = np.unique(molecules[i])
    return molecules


def rdf_pablo(
    traj,
    index1,
    index2,
    box_len,
    N_atoms,
    device,
    comb=True,
    bond_index=None,
    nbins=100,
    lag=0,
):
    # traj = list of xyz where it is of shape: (#frames, #atoms, 3)
    # index1 = list of all the index of atom type 1
    # index2 = list of all the index of atom type 2
    # N_atoms = total number of atoms in your simulation

    traj = torch.Tensor(traj).to(device)
    box_len = torch.Tensor(box_len).to(device)

    if comb is True:
        superindex = torch.LongTensor(list(itertools.combinations(index1, index2)))
    else:
        superindex = torch.LongTensor(list(itertools.product(index1, index2)))

    self_filter = (superindex[:, 0] - superindex[:, 1]) != 0
    superindex = superindex[self_filter]

    A = torch.zeros(N_atoms, N_atoms, dtype=torch.long)
    A[superindex[:, 0], superindex[:, 1]] = 1
    # A[superindex[:,1], superindex[:,0]] = 1

    if bond_index is not None:
        # A[bond_index[:,1], bond_index[:,0]] = 0
        A[bond_index[:, 0], bond_index[:, 1]] = 0
    superindex = torch.nonzero(A.triu(), as_tuple=False)
    # print(superindex)
    # print(print('superindex_shape = {}'.format(superindex.shape[0])))
    # compute dis mat
    dis_3d = traj[superindex[:, 0], :] - traj[superindex[:, 1], :]

    # minimum image convention
    dis_3d_mic = dis_3d - box_len * np.around(dis_3d / box_len)
    dis = dis_3d_mic.pow(2).sum(-1).sqrt()
    # import ipdb
    # ipdb.set_trace()
    # print(dis)
    # print('max distance [A]:', dis.max().item(), 'box length:', box_len[0].item())
    # cutoff
    R_max = (box_len.min() * 0.5).detach().cpu().numpy()
    # R_max = 2

    # import ipdb
    # ipdb.set_trace()
    # N = traj.shape[0] * superindex.shape[0]
    N = dis.view(-1).shape[0]
    # N = index2.shape[0]
    box_vol = (box_len[0] * box_len[1] * box_len[2]).detach().cpu().numpy()
    # print(box_vol)

    hist = np.histogram(dis.detach().cpu().numpy(), bins=nbins, range=(0.0, R_max))

    count, edge = np.histogram([-1], bins=nbins, range=(0.0, R_max))  # [1]
    count += hist[0]
    # count = hist[0]

    bins = 0.5 * (edge[:-1] + edge[1:])
    vol = 4 / 3.0 * np.pi * (np.power(edge[1:], 3) - np.power(edge[:-1], 3))

    density = N / (box_vol)

    rdf = count / (vol * density)
    return bins, rdf


def get_neighbors_two_atoms(
    xyz1, xyz2, bond_length, cell=[None, None, None], periodic=True
):
    """
    in this case, atom1 = vornoi points, atom2 = peo oxygens. So what this code will do is
    iterate through the voronoi points and find the oxygens that are within the neighborhood of each according to bond_length"""
    # A=np.lexsort((xyz[:,2],xyz[:,1],xyz[:,0]))
    dis_mat = xyz2[None, :, :] - xyz1[:, None, :]
    if periodic == True:
        shift = np.round(np.divide(dis_mat, cell))
        offsets = -shift
        dis_mat = dis_mat + offsets * cell
    dis_sq = np.power(np.array(dis_mat), 2).sum(-1)
    """
    dis_sq is now the shape of [# atom1, # atom2], where each row is the distance between the current
    atom1 and all the atom2 atoms
    """
    dis_sq = dis_sq**0.5
    clusters = np.array([0 for i in range(xyz1.shape[0])])  # one cluster for each atom1
    all_neighbors, all_dists = [], []
    for i in range(xyz1.shape[0]):  # iterate through each xyz1
        mm = max(clusters)
        oxy_neighbors = []
        oxy_neighbors.extend(list(np.where(dis_sq[i] <= bond_length)[0]))
        oxy_neighbors = np.array(oxy_neighbors)
        if len(oxy_neighbors) == 0:
            clusters[i] = mm + 1
            continue
        # elif xyz1.shape[0] == 1:
        #     all_neighbors = [oxy_neighbors]
        #     break
        all_neighbors.append(
            np.vstack(
                [
                    i * np.ones_like(oxy_neighbors),
                    oxy_neighbors,
                    dis_sq[i, oxy_neighbors],
                ]
            ).transpose(1, 0)
        )
    all_neighbors = np.vstack(all_neighbors)

    return all_neighbors


def analyze_psd(lines):
    data = [line.strip("\n").split(" ") for line in lines[11:]]
    data = [[float(d[0]), float(d[1]), float(d[2]), float(d[3])] for d in data]
    psd_df = pd.DataFrame(data, columns=["bin", "count", "cum_dist", "deriv_dist"])

    CUTOFF = BIN_CUTOFF * 10 + 1

    bins = psd_df["bin"].values[:CUTOFF].flatten().tolist()
    cum_dist = (
        psd_df["cum_dist"].fillna(FILL_PSD_NA_VAL).values[:CUTOFF].flatten().tolist()
    )
    deriv_dist = (
        psd_df["deriv_dist"].fillna(FILL_PSD_NA_VAL).values[:CUTOFF].flatten().tolist()
    )
    return bins, cum_dist, deriv_dist


def analyze_psd_pts(lines):
    data = [line.strip("\n").split() for line in lines[2:]]
    data = [[float(d[1]), float(d[2]), float(d[3]), float(d[4])] for d in data]
    psd_df = pd.DataFrame(data, columns=["x", "y", "z", "max_radius"])
    return psd_df[["x", "y", "z"]].values, psd_df["max_radius"].values


def combine_voro_by_o_neighbors(voronoi_atoms, o_atoms, num_neighbors=6):
    neighbors = get_neighbors_two_atoms(
        voronoi_atoms.positions,
        o_atoms.positions,
        bond_length=10,
        cell=voronoi_atoms.cell.diagonal(),
    )

    neigh_clus = [
        np.array(neighbors[neighbors[:, 0] == cur])
        for cur in np.unique(neighbors[:, 0])
    ]
    # print(neigh_clus[0])
    neigh_clus = [
        cur[np.argsort(cur[:, 2])][:num_neighbors] for cur in neigh_clus
    ]  # sort according to O-distance and only chose closest 5 oxygens
    neigh_clus = [
        cur[np.argsort(cur[:, 1])] for cur in neigh_clus
    ]  # sort according to closet oxygen indices
    neigh_clus = np.array(neigh_clus)
    oxys = neigh_clus[:, :, 1]
    # print(oxys.shape)
    new_voronoi = []
    for cur_unique in np.unique(oxys, axis=0):
        mask = [cur.tolist() == cur_unique.tolist() for cur in oxys]
        curxyz = voronoi_atoms.positions[mask][0]
        if curxyz.shape[0] > 1:
            # these xyz can be across a pbc, so they must be combined appropriately
            curxyz = voronoi_atoms.positions[mask, :]
            box_len = voronoi_atoms.cell.array.diagonal()
            # this ends up adding a box length to anything below 1/2*box_len in any dimension
            offsets = ~np.greater_equal(curxyz, (0.5 * box_len)) + np.less_equal(
                curxyz, -0.5 * box_len
            )
            curxyz = curxyz + offsets * box_len
            # this mean is from like 1/2 - 3/2 of the box_len, so we need to rewrap back to the box!
            curxyz = curxyz.mean(axis=0)
            # wrapped = current - box_len*(number of times it is past the box_len aka the image number)
            curxyz = curxyz - box_len * (curxyz // box_len)
            # now curxyz is wrapped back into the box, and can be saved as a new, combined voronoi point
        new_voronoi.append(curxyz)
    new_voronoi = np.vstack(new_voronoi)

    new_voronoi_atoms = ase.Atoms(
        ["X" for cur in range(len(new_voronoi))],
        positions=new_voronoi,
        cell=o_atoms.cell,
        pbc=[1, 1, 1],
    )
    return new_voronoi_atoms


def combine_voro_by_o_neighbors_struc(voronoi_struc, o_struc, num_neighbors=6):
    neighbors = get_neighbors_two_atoms(
        voronoi_struc.cart_coords,
        o_struc.cart_coords,
        bond_length=10,
        cell=np.array(voronoi_struc.lattice.abc),
    )

    neigh_clus = [
        np.array(neighbors[neighbors[:, 0] == cur])
        for cur in np.unique(neighbors[:, 0])
    ]
    # print(neigh_clus[0])
    neigh_clus = [
        cur[np.argsort(cur[:, 2])][:num_neighbors] for cur in neigh_clus
    ]  # sort according to O-distance and only chose closest 5 oxygens
    neigh_clus = [
        cur[np.argsort(cur[:, 1])] for cur in neigh_clus
    ]  # sort according to closet oxygen indices
    neigh_clus = np.array(neigh_clus)
    oxys = neigh_clus[:, :, 1]
    # print(oxys.shape)
    new_voronoi = []
    for cur_unique in np.unique(oxys, axis=0):
        mask = [cur.tolist() == cur_unique.tolist() for cur in oxys]
        curxyz = voronoi_struc.cart_coords[mask][0]
        if curxyz.shape[0] > 1:
            # these xyz can be across a pbc, so they must be combined appropriately
            curxyz = voronoi_struc.cart_coords[mask, :]
            box_len = np.array(voronoi_struc.lattice.abc)
            # this ends up adding a box length to anything below 1/2*box_len in any dimension
            offsets = ~np.greater_equal(curxyz, (0.5 * box_len)) + np.less_equal(
                curxyz, -0.5 * box_len
            )
            curxyz = curxyz + offsets * box_len
            # this mean is from like 1/2 - 3/2 of the box_len, so we need to rewrap back to the box!
            curxyz = curxyz.mean(axis=0)
            # wrapped = current - box_len*(number of times it is past the box_len aka the image number)
            curxyz = curxyz - box_len * (curxyz // box_len)
            # now curxyz is wrapped back into the box, and can be saved as a new, combined voronoi point
        new_voronoi.append(curxyz)
    new_voronoi = np.vstack(new_voronoi)

    new_voronoi_struc = Structure(
        lattice=voronoi_struc[0].lattice,
        species=[voronoi_struc[0].species for cur in new_voronoi],
        coords=new_voronoi,
        coords_are_cartesian=True,
    )
    return new_voronoi_struc


def find_mic(D, cell, pbc=True):
    """Finds the minimum-image representation of vector(s) D"""
    # Calculate the 4 unique unit cell diagonal lengths
    diags = np.sqrt(
        (
            np.dot(
                [
                    [1, 1, 1],
                    [-1, 1, 1],
                    [1, -1, 1],
                    [-1, -1, 1],
                ],
                cell,
            )
            ** 2
        ).sum(1)
    )

    # calculate 'mic' vectors (D) and lengths (D_len) using simple method
    Dr = np.dot(D, np.linalg.inv(cell))
    D = np.dot(Dr - np.round(Dr) * pbc, cell)
    D_len = np.sqrt((D**2).sum(1))
    # return mic vectors and lengths for only orthorhombic cells,
    # as the results may be wrong for non-orthorhombic cells
    if (max(diags) - min(diags)) / max(diags) > 1e-9:
        return D, D_len

    # The cutoff radius is the longest direct distance between atoms
    # or half the longest lattice diagonal, whichever is smaller
    cutoff = min(max(D_len), max(diags) / 2.0)
    # The number of neighboring images to search in each direction is
    # equal to the ceiling of the cutoff distance (defined above) divided
    # by the length of the projection of the lattice vector onto its
    # corresponding surface normal. a's surface normal vector is e.g.
    # b x c / (|b| |c|), so this projection is (a . (b x c)) / (|b| |c|).
    # The numerator is just the lattice volume, so this can be simplified
    # to V / (|b| |c|). This is rewritten as V |a| / (|a| |b| |c|)
    # for vectorization purposes.
    latt_len = np.sqrt((cell**2).sum(1))
    V = abs(np.linalg.det(cell))
    n = pbc * np.array(np.ceil(cutoff * np.prod(latt_len) / (V * latt_len)), dtype=int)

    # Construct a list of translation vectors. For example, if we are
    # searching only the nearest images (27 total), tvecs will be a
    # 27x3 array of translation vectors. This is the only nested loop
    # in the routine, and it takes a very small fraction of the total
    # execution time, so it is not worth optimizing further.
    tvecs = []
    for i in range(-n[0], n[0] + 1):
        latt_a = i * cell[0]
        for j in range(-n[1], n[1] + 1):
            latt_ab = latt_a + j * cell[1]
            for k in range(-n[2], n[2] + 1):
                tvecs.append(latt_ab + k * cell[2])
    tvecs = np.array(tvecs)

    # Translate the direct displacement vectors by each translation
    # vector, and calculate the corresponding lengths.
    D_trans = tvecs[np.newaxis] + D[:, np.newaxis]
    D_trans_len = np.sqrt((D_trans**2).sum(2))

    # Find mic distances and corresponding vector(s) for each given pair
    # of atoms. For symmetrical systems, there may be more than one
    # translation vector corresponding to the MIC distance; this finds the
    # first one in D_trans_len.
    D_min_len = np.min(D_trans_len, axis=1)
    D_min_ind = D_trans_len.argmin(axis=1)
    D_min = D_trans[range(len(D_min_ind)), D_min_ind]

    return D_min, D_min_len
