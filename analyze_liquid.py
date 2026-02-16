# %%
from MarkovTransportAnalyzer import *
import pickle
import os
import argparse
import time

# %%
if __name__ == "__main__":
    # user_specified information (for now)
    method2_solvent_anion_dict = {"solvent":['PL1'], "anion":['AN1']}
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('-p', '--path', type=str, default="")
    parser.add_argument('-n', '--name', type=str, default="")
    parser.add_argument('-t', '--timestep', type=int)
    parser.add_argument('--temperature', type=int)
    parser.add_argument('-a', '--anion', type=str, default="PF6")
    parser.add_argument('-m', '--method', type=str, default="convex")
    parser.add_argument('-s', '--self_only', type=bool, default=False)  
    parser.add_argument('-c', '--correct_com', type=str, default="solvent")
    parser.add_argument('--skip', type=int, default=1)
    args = parser.parse_args()
    step = int(args.step)
    PATH = str(args.path)
    NAME = str(args.name)
    anion = str(args.anion)
    method = str(args.method)
    TIMESTEP = float(args.timestep)
    correct_com = str(args.correct_com)
    TEMPERATURE = int(args.temperature)
    if type(args.self_only) == str:
        raise ValueError("self_only must be True or False")
    else:
        self_only = args.self_only
    SKIP = int(args.skip)
    # step 1 : setting the parameters.
    #stepskip_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    #stepskip_values = [200, 100, 50, 20, 10, 5, 2, 1]
    #TEMPERATURE = 298
    #self_only = bool(args.self_only)

    print("step: ", step, type(step))
    FILENAME = f"{NAME}_skip-{SKIP}_{TEMPERATURE}K_method-{method}_com-{correct_com}_selfonly-{self_only}"
    if step in [0, 1]:
        print("i hope not here?")
        if os.path.exists(f"raw_{FILENAME}.pickle"):
            if step == 0:
                with open(f"raw_{FILENAME}.pickle", "rb") as f:
                    le = pickle.load(f)
            elif step == 1:
                print(f"raw_{FILENAME}.pickle already exists")
                exit()
        else:
            le = LiquidElectrolyteAnalyzer(PATH, NAME)
            le.read_structures_parallel(analyze_anions=True, 
                                        timestep=50, 
                                        stepskip=SKIP, 
                                        correct_com=correct_com,
                                        coordinators = ["O", "F", "N"],
                                        anion = anion,
                                        mode = method,
                                        cutoff = 0
                                        )
            with open(f"raw_{FILENAME}.pickle", "wb") as f:
                pickle.dump(le, f)
    print("start to read")    
    if step == 2:
        with open(f"raw_{FILENAME}.pickle", "rb") as f:
            le = pickle.load(f)
    print("finished reading")
    # step 2 : extract entity events
    if step in [0, 2]:
        le.extract_entity_events()
        le.simplify_complete_coord_info_to_states(methods=[1], n_workers=16, parallel=True)
        le.state_to_state_transitions()
        #le.u_simplify_complete_coord_info_to_states(methods=[1, 2], method2_solvent_anion_dict=method2_solvent_anion_dict)
        #le.u_state_to_state_transitions(method2_solvent_anion_dict=method2_solvent_anion_dict)
        #le.u_extract_entity_events(method="add_or_remove")
        with open(f"raw_{FILENAME}.pickle", "wb") as f:
            pickle.dump(le, f)
    if step in [3,4,5]:
        le = pickle.load(open(f"raw_{FILENAME}.pickle", "rb"))


    if step in [0, 3]:
        # simplified transitions
        if os.path.exists(f"onsager_simplified_{FILENAME}.pickle"):
            print(f"onsager_simplified_{FILENAME}.pickle already exists")
        else:
            cation_disps = simplify_entitychange_new(le.entitychange)
            oa = OnsagerAnalyzer(1, cation_disps, le.anion_displacements, label = NAME, distance_matrix=None, mode="EVENT", distance_cutoff = None, binmethod="FULL")
            self_values, distinct_values = oa.compute_fft_onsager(self_only=self_only)
            dt_values = np.arange(self_values.shape[2])
            return_dict = {"dt_values":dt_values, "self_values":self_values, "distinct_values":distinct_values}
            simplified_transitions = {}
            simplified_transitions["event"] = return_dict
            oa = OnsagerAnalyzer(1, cation_disps, le.anion_displacements, label = NAME, distance_matrix=None, mode="PARTICLE", distance_cutoff = None, binmethod="FULL")
            self_values, distinct_values = oa.compute_fft_onsager(self_only=self_only)
            dt_values = np.arange(self_values.shape[2])
            return_dict = {"dt_values":dt_values, "self_values":self_values, "distinct_values":distinct_values}
            simplified_transitions["particle"] = return_dict
            with open(f"onsager_simplified_{FILENAME}.pickle", "wb") as f:
                pickle.dump(simplified_transitions, f)
        # simplified u-transitions
        if os.path.exists(f"onsager_simplified_u{FILENAME}.pickle"):
            print(f"onsager_simplified_u{FILENAME}.pickle already exists")
        elif False:
            cation_disps = le.u_entitychange
            oa = OnsagerAnalyzer(1, cation_disps, le.anion_displacements, label = NAME, distance_matrix=None, mode="EVENT", distance_cutoff = None, binmethod="FULL")
            self_values, distinct_values = oa.compute_fft_onsager(self_only=self_only)
            dt_values = np.arange(self_values.shape[2])
            return_dict = {"dt_values":dt_values, "self_values":self_values, "distinct_values":distinct_values}
            simplified_transitions = {}
            simplified_transitions["event"] = return_dict
            oa = OnsagerAnalyzer(1, cation_disps, le.anion_displacements, label = NAME, distance_matrix=None, mode="PARTICLE", distance_cutoff = None, binmethod="FULL")
            self_values, distinct_values = oa.compute_fft_onsager(self_only=self_only)
            dt_values = np.arange(self_values.shape[2])
            return_dict = {"dt_values":dt_values, "self_values":self_values, "distinct_values":distinct_values}
            simplified_transitions["particle"] = return_dict
            with open(f"onsager_simplified_u{FILENAME}.pickle", "wb") as f:
                pickle.dump(simplified_transitions, f)


    if step in [0, 4]:
        # markov transitions
        for method_key in list(le.state_transitions.values())[0].keys():
            if os.path.exists(f"onsager_markov_{method_key}_{FILENAME}.pickle"):
                print(f"onsager_markov_{method_key}_{FILENAME}.pickle already exists")
            else:
                markov_transitions = {}
                cation_disps = {}
                for Li_index in le.state_transitions.keys():
                    cation_disps[Li_index] = le.state_transitions[Li_index][method_key]
                oa_particle = OnsagerAnalyzer(1, cation_disps, le.anion_displacements, label=f"{NAME}_markov", distance_matrix=None, mode="PARTICLE", distance_cutoff=None, binmethod="FULL")
                self_values_particle, distinct_values_particle = oa_particle.compute_fft_onsager()
                dt_values_particle = np.arange(self_values_particle.shape[2])
                return_dict_particle = {"dt_values": dt_values_particle, "self_values": self_values_particle, "distinct_values": distinct_values_particle}

                oa_event = OnsagerAnalyzer(1, cation_disps, le.anion_displacements, label=f"{NAME}_markov", distance_matrix=None, mode="EVENT", distance_cutoff=None, binmethod="FULL")
                self_values_event, distinct_values_event = oa_event.compute_fft_onsager()
                dt_values_event = np.arange(self_values_event.shape[2])
                return_dict_event = {"dt_values": dt_values_event, "self_values": self_values_event, "distinct_values": distinct_values_event}

                markov_transitions["particle"] = return_dict_particle
                markov_transitions["event"] = return_dict_event
                with open(f"onsager_markov_{method_key}_{FILENAME}.pickle", "wb") as f:
                    pickle.dump(markov_transitions, f)

    #if step in [0, 5]:
    if False:
        # markov transitions
        for method_key in list(le.state_transitions.values())[0].keys():
            if os.path.exists(f"onsager_markov_u{method_key}_{FILENAME}.pickle"):
                print(f"onsager_markov_u{method_key}_{FILENAME}.pickle already exists")
            else:
                markov_transitions = {}
                cation_disps = {}
                for Li_index in le.u_state_transitions.keys():
                    cation_disps[Li_index] = le.u_state_transitions[Li_index][method_key]
                oa_particle = OnsagerAnalyzer(1, cation_disps, le.anion_displacements, label=f"{NAME}_markov", distance_matrix=None, mode="PARTICLE", distance_cutoff=None, binmethod="FULL")
                self_values_particle, distinct_values_particle = oa_particle.compute_fft_onsager()
                dt_values_particle = np.arange(self_values_particle.shape[2])
                return_dict_particle = {"dt_values": dt_values_particle, "self_values": self_values_particle, "distinct_values": distinct_values_particle}

                oa_event = OnsagerAnalyzer(1, cation_disps, le.anion_displacements, label=f"{NAME}_markov", distance_matrix=None, mode="EVENT", distance_cutoff=None, binmethod="FULL")
                self_values_event, distinct_values_event = oa_event.compute_fft_onsager()
                dt_values_event = np.arange(self_values_event.shape[2])
                return_dict_event = {"dt_values": dt_values_event, "self_values": self_values_event, "distinct_values": distinct_values_event}

                markov_transitions["particle"] = return_dict_particle
                markov_transitions["event"] = return_dict_event
                with open(f"onsager_markov_u{method_key}_{FILENAME}.pickle", "wb") as f:
                    pickle.dump(markov_transitions, f)
