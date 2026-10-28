


# Testing Graph theory
import networkx as nx



def graph_from_pairs(pairs):

    # Find all unique graph points and construct from edges
    point_conversions = [] # (spot_ind, ref_ind)
    edge_list = []
    for pair in pairs:
        edge_points = []
        for point_values in pair:
            if tuple(point_values) not in point_conversions:
                point_conversions.append(tuple(point_values))
            edge_points.append(point_conversions.index(tuple(point_values)))
        edge_list.append(tuple(edge_points))

    G = nx.Graph(edge_list)
    cliques = list(nx.find_cliques(G))

    return G


def graph_from_pairs(pairs):
    pairs = np.asarray(pairs)
    
    # 1. Map each unique (spot_ind, ref_ind) assignment to a unique point_conversion node
    point_conversions = []
    
    # Flatten the 3D array to easily extract all [spot, ref] rows
    flat_mappings = pairs.reshape(-1, 2)
    for row in flat_mappings:
        point = (int(row[0]), int(row[1]))
        if point not in point_conversions:
            point_conversions.append(point)
            
    # 2. Replicate your edge creation loop using the new structure
    edge_list = []
    for pair in pairs:
        # Extract the two explicit points making up this specific pair
        pt_a = (int(pair[0, 0]), int(pair[0, 1]))
        pt_b = (int(pair[1, 0]), int(pair[1, 1]))
        
        # Look up their index locations exactly like your original code did
        idx_a = point_conversions.index(pt_a)
        idx_b = point_conversions.index(pt_b)
        
        if idx_a != idx_b:  # Avoid self-loops
            edge_list.append((idx_a, idx_b))
        
    # 3. Generate the Graph object
    G = nx.Graph(edge_list)
    
    return G, point_conversions








def connections_to_graph(pairs):

    point_conversions = [] # (spot_ind, ref_ind)

    # Find all graph points
    for spot_ind in range(pairs.shape[1]):
        ref_inds = np.unique(pairs[:, spot_ind][~np.isnan(pairs[:, spot_ind])])

        for ref_ind in ref_inds:
            point_conversions.append((int(spot_ind), int(ref_ind)))

    # Decomposes connections into points and find edges
    edge_list= []
    for pair in pairs:
        spot_inds = np.nonzero(~np.isnan(pair))[0]
        ref_inds = pair[spot_inds]

        points = [point_conversions.index((int(spot_ind), int(ref_ind)))
                  for spot_ind, ref_ind in zip(spot_inds, ref_inds)]
        
        edge_list.append(tuple(points))

        # G.add_edge(points)
    
    G = nx.Graph(edge_list)
    cliques = list(nx.find_cliques(G))

    qofs = []
    point_conversions = np.asarray(point_conversions)
    for clique in tqdm(cliques):

        spot_inds, ref_inds = point_conversions[clique].T

        conn_spots = all_spot_qs[spot_inds]
        conn_refs = all_ref_qs[ref_mask][ref_inds]
        orientation, _ = Rotation.align_vectors(conn_spots,
                                                conn_refs)
        all_rot_qs = orientation.apply(all_ref_qs[ref_mask], inverse=False)
        temp_qmask = qmask.generate(all_rot_qs)

        qof = get_quality_of_fit(
            all_spot_qs[spot_inds], # fit_spot_qs
            all_spot_ints[spot_inds], # fit_spot_ints
            all_rot_qs[ref_inds], # fit_rot_qs
            all_ref_fs[ref_mask][ref_inds], # fit_ref_fs
            all_spot_qs, # all_spot_qs
            all_spot_ints, # all_spot_ints
            all_rot_qs[temp_qmask], # all_rot_qs
            all_ref_fs[ref_mask][temp_qmask], # all_ref_fs
            sigma=0.1)
        qofs.append(qof)
    
    
    return point_conversions, edge_list, cliques, qofs, G


def connection_from_clique(clique, point_conversions, spot_num):

    conn = [np.nan,] * spot_num

    for spot_ind, ref_ind in point_conversions[clique]:
        conn[spot_ind] = ref_ind
    
    return np.asarray(conn)


from xrdmaptools.utilities.utilities import tic, toc
spot_nums = []
times = []

def tic(output=False):
    global _t0
    _t0 = ttime.monotonic()
    if output:
        return _t0


def toc(string='', print_msg=True, output=False):
    global _t0
    dt = ttime.monotonic() - _t0
    s = f'{string}\ndt = {dt:.3f} sec'
    if print_msg:
        print(s, end='\n')
    if output:
        return dt

def timed(func):
    def wrapper(*args, **kwargs):
        tic()
        out = func(*args, **kwargs)
        times.append(toc(output=True, print_msg=False))
        return out
    return wrapper

# from xrdmaptools.reflections.spot_blob_indexing_3D import pair_casting_index_best_grain, pair_casting_index_full_pattern
# pair_casting_index_best_grain = timed(pair_casting_index_best_grain)
# pair_casting_index_full_pattern = timed(pair_casting_index_full_pattern)

@timed
def test_indexing(all_spot_qs,
                  all_spot_ints,
                  all_ref_qs,
                  all_ref_hkls,
                  all_ref_fs,
                  near_q,
                  near_angle,
                  min_q,
                  qmask,
                  qof_minimum=0,
                  max_grains=20,
                  degrees=False,
                  symmetrize=True,
                  verbose=True):
    
    if near_q > min_q * 0.85:
        err_str = ("'near_q' threshold is greater than 85% of minimum "
                   + "lattice spacing. This seems unwise.")
        raise ValueError(err_str)    
    
    # Find vector magnitudes for measured and reference reflections
    spot_q_mags = np.linalg.norm(all_spot_qs, axis=1)
    ref_q_mags = np.linalg.norm(all_ref_qs, axis=1)

    # Find difference between measured and calculated q magnitudes
    mag_diff_arr = np.abs(spot_q_mags[:, np.newaxis]
                          - ref_q_mags[np.newaxis, :])
    
    # Eliminate any reflections outside of phase-allowed spots
    phase_mask = np.any(mag_diff_arr < near_q, axis=1)
    phase_inds = np.nonzero(phase_mask)[0]
    mag_diff_arr = mag_diff_arr[phase_mask]
    all_spot_qs = all_spot_qs[phase_mask]
    spot_q_mags = spot_q_mags[phase_mask]

    # Generate all pairs of spots which are crystallographically feasible
    spot_pair_indices = list(combinations(range(len(all_spot_qs)), 2))
    spot_pair_dist = distance_matrix(all_spot_qs, all_spot_qs) 
    allowed_pairs = [spot_pair_dist[tuple(indices)] > min_q * 0.85
                      for indices in spot_pair_indices]
    spot_pair_indices = np.asarray(spot_pair_indices)[allowed_pairs]
    
    # Determine all angles
    spot_angles = multi_vector_angles(all_spot_qs, all_spot_qs, degrees=degrees)
    ref_angles = multi_vector_angles(all_ref_qs, all_ref_qs, degrees=degrees)

    # Construct iterable
    if verbose:
        iterate = lambda x : tqdm(x, position=0, leave=True)
    else:
        iterate = lambda x : x

    if verbose:
        print('Finding all valid pairs...', flush=True)

    blank_connection = np.array([np.nan,] * len(phase_mask))
    connection_pairs = []
    pairs = []    
    for pair in iterate(spot_pair_indices):
        ref_combos = list(product(*[np.nonzero(mag_diff_arr[i] < near_q)[0]
                          for i in pair]))

        angle_mask = [np.abs(spot_angles[tuple(pair)]
                      - ref_angles[tuple(combo)]) < near_angle
                      for combo in ref_combos]
        doublet_mask = [combo[0] != combo[1] for combo in ref_combos]
        collinear_mask = [ref_angles[tuple(combo)] > 0 for combo in ref_combos]

        ref_combos = np.asarray(ref_combos)[(np.asarray(angle_mask)
                                             & np.asarray(doublet_mask)
                                             & np.asarray(collinear_mask))]
        
        # Rotation based symmeterization
        if len(ref_combos) > 0:
            temp_connections = []
            temp_pairs, temp_pair_mask = [], []
            pair_chars, pair_mags, pair_matches = [], [], []
            match_id = 0
            
            # Check all possible combintations from pair
            for combo in ref_combos:
                
                # Remove orientationally invariant combinations
                if are_collinear(all_ref_hkls[combo]):
                    continue

                # Add pairs
                temp_pairs.append([(s, r) for s, r in zip(pair, combo)])
                temp_pair_mask.append(True)

                # Add connections (MAY BE REMOVED LATER)
                connection = blank_connection.copy()
                connection[phase_inds[pair[0]]] = combo[0]
                connection[phase_inds[pair[1]]] = combo[1]
                temp_connections.append(connection)

                # Qualify potential fits and their orientationally magnitude
                if symmetrize:
                    # Characterize combo
                    combo_char = (ref_q_mags[combo[0]].round(10),
                                  ref_q_mags[combo[1]].round(10),
                                  ref_angles[tuple(combo)].round(10))
                    if combo_char not in pair_chars:
                        match_id += 1
                        pair_matches.append(match_id)
                    else:
                        pair_matches.append(pair_chars.index(combo_char))
                    pair_chars.append(combo_char)
                    
                    # Orientation magnitude of combo. Will chose smallest-ish later
                    combo_mag = Rotation.align_vectors(
                                        all_spot_qs[pair],
                                        all_ref_qs[combo])[0].magnitude()
                    if degrees:
                        combo_mag = np.degrees(combo_mag)
                    pair_mags.append(combo_mag)
            
            # Pick 
            if symmetrize:
                temp_pair_mask = np.asarray([False,] * len(temp_pair_mask))
                for idx in np.unique(pair_matches):
                    equi_mask = pair_matches == idx
                    min_angle = np.min(np.asarray(pair_mags)[equi_mask])
                    keep_mask = np.asarray(pair_mags)[equi_mask] < min_angle + near_angle
                    temp_pair_mask[np.nonzero(equi_mask)[0][keep_mask]] = True
            
            pairs.extend(np.asarray(temp_pairs)[temp_pair_mask])
            connection_pairs.extend(np.asarray(temp_connections)[temp_pair_mask])


        # Magnitude base symmeterization - BAD        
        # if len(ref_combos) > 0:
        #     temp_pairs, temp_pair_mask = [], []
        #     pair_characteristics, pair_matches, pair_evaluators = [], [], []
        #     match_id = 0
        #     for combo in ref_combos:
        #         temp_pairs.append([(s, r) for s, r in zip(pair, combo)])
        #         temp_pair_mask.append(True)
        #         if symmetrize:
        #             characteristics = (ref_q_mags[combo[0]].round(10),
        #                                ref_q_mags[combo[1]].round(10),
        #                                ref_angles[tuple(combo)].round(10))
        #             if characteristics not in pair_characteristics:
        #                 match_id += 1
        #             pair_characteristics.append(characteristics)
        #             pair_matches.append(match_id)
        #             # Simple way to get close to fundamental zone
        #             pair_evaluators.append(np.sum([mag_diff_arr[m] for m in temp_pairs[-1]]))
                
        #     if symmetrize:
        #         temp_pair_mask = [False,] * len(temp_pair_mask)
        #         for idx in np.unique(pair_matches):
        #             equi_mask = pair_matches == idx
        #             keep_ind = np.argmin(np.asarray(pair_evaluators)[equi_mask])
        #             temp_pair_mask[np.nonzero(equi_mask)[0][keep_ind]] = True
            
        #     pairs.extend(np.asarray(temp_pairs)[temp_pair_mask])

        # # No symmeterization
        # if len(ref_combos) > 0:
        #     pair_characteristics = []
        #     for combo in ref_combos:
        #         pairs.append([(s, r) for s, r in zip(pair, combo)])
            
        #         connection = blank_connection.copy()
        #         connection[phase_inds[pair[0]]] = combo[0]
        #         connection[phase_inds[pair[1]]] = combo[1]
        #         connection_pairs.append(connection)

    # if len(connection_pairs) > 0:
    #     red_pairs = reduce_symmetric_equivalents(np.asarray(connection_pairs),
    #                                             all_spot_qs,
    #                                             all_ref_qs,
    #                                             all_ref_hkls,
    #                                             near_angle,
    #                                             min_q,
    #                                             degrees=degrees,
    #                                             verbose=verbose)

    #     pairs = []
    #     for pair in red_pairs:
    #         (spot_inds, ref_inds) = _get_connection_indices(pair)
    #         pairs.append([(s, r) for s, r in zip(spot_inds, ref_inds)])

    # return pairs

    if verbose:
        print('Constructing graph and finding best cliques...')
    
    # Find all unique graph points and construct from edges
    point_conversions = [] # (spot_ind, ref_ind)
    edge_list = []
    for pair in pairs:
        edge_points = []
        for point_values in pair:
            if tuple(point_values) not in point_conversions:
                point_conversions.append(tuple(point_values))
            edge_points.append(point_conversions.index(tuple(point_values)))
        edge_list.append(tuple(edge_points))

    G = nx.Graph(edge_list)
    cliques = list(nx.find_cliques(G))

    # Remove symmetrically equivalent cliques
    if symmetrize:
        pass
        # clique_character = []
        # clique_ids = []
        # match_id = 0
        # for i, clique in enumerate(cliques):
        #     character = []
        #     for point in clique:
        #         spot_ind, ref_ind = point_conversions[point]
        #         character.append(ref_q_mags[ref_ind].round(10))
        #     character = tuple(character)

        #     if character in clique_character:
        #         clique_ids.append(clique_ids[clique_character.index(character)])
        #     else:
        #         match_id += 1
        #         clique_ids.append(match_id)
        #     clique_character.append(character)

        # sym_cliques = []
        # clique_ids = np.asarray(clique_ids)
        # for idx in np.unique(clique_ids):
        #     sym_inds = np.nonzero(clique_ids == idx)[0]
        #     min_ind = np.argmin([clique_character[ind][-1] for ind in sym_inds])

        #     sym_cliques.append(cliques[sym_inds[min_ind]])

        # cliques = sym_cliques

        # clique_character = []
        # sym_cliques = []
        # for clique in cliques:
        #     character = []
        #     for point in clique:
        #         spot_ind, ref_ind = point_conversions[point]
        #         character.append([spot_ind, ref_q_mags[ref_ind].round(10)])
        #     character = tuple(character)

        #     if character not in clique_character:
        #         sym_cliques.append(clique)
        #         clique_character.append(character)
    
        # cliques = sym_cliques

        # clique_character = []
        # clique_mags = []
        # sym_cliques = []
        # for clique in cliques:
        #     character, spots = [], []
        #     for point in clique:
        #         spot_ind, ref_ind = point_conversions[point]
        #         character.extend([spot_ind, ref_q_mags[ref_ind].round(10)])
        #     character = tuple(character)
        #     mag = np.sum([mag_diff_arr[point_conversions[point]] for point in clique])

        #     if character not in clique_character:
        #         sym_cliques.append(clique)
        #         clique_character.append(character)
        #         clique_mags.append(mag)
        #     else:
        #         ind = clique_character.index(character)
        #         if clique_mags[ind] > mag:
        #             clique_mags[ind] = mag
        #             sym_cliques[ind] = clique
    
        # cliques = sym_cliques

    # return cliques, point_conversions

    qofs = []
    point_conversions = np.asarray(point_conversions)

    for clique in iterate(cliques):

        spot_inds, ref_inds = point_conversions[clique].T

        conn_spots = all_spot_qs[spot_inds]
        conn_refs = all_ref_qs[ref_inds]
        orientation, _ = Rotation.align_vectors(conn_spots,
                                                conn_refs)
        all_rot_qs = orientation.apply(all_ref_qs, inverse=False)
        temp_qmask = qmask.generate(all_rot_qs)

        qof = get_quality_of_fit(
            all_spot_qs[spot_inds], # fit_spot_qs
            all_spot_ints[spot_inds], # fit_spot_ints
            all_rot_qs[ref_inds], # fit_rot_qs
            all_ref_fs[ref_inds], # fit_ref_fs
            all_spot_qs, # all_spot_qs
            all_spot_ints, # all_spot_ints
            all_rot_qs[temp_qmask], # all_rot_qs
            all_ref_fs[temp_qmask], # all_ref_fs
            sigma=0.1)
        qofs.append(qof)
    
    # return point_conversions, cliques, qofs

    # Dummy check
    if len(cliques) < 1:
        return [], []

    # Decompose cliques into best fitting
    best_cliques, best_qofs = [], []
    iterations = 1
    while True:

        # Add best clique
        best_ind = np.argmax(qofs)
        points = cliques[best_ind]
        best_cliques.append(points)
        best_qofs.append(qofs[best_ind])

        # Update cliques and qofs
        used_spots = [point_conversions[point][0] for point in points]
        removed_cliques = []
        for i, clique in enumerate(cliques):
            for point in clique:
                if point_conversions[point][0] in used_spots:
                    removed_cliques.append(i)
                    break
        cliques = [c for i, c in enumerate(cliques) if i not in removed_cliques]
        qofs = [q for i, q in enumerate(qofs) if i not in removed_cliques]

        if (len(cliques) < 1
            or best_qofs[-1] < qof_minimum
            or iterations > max_grains):
            break
            # pass
        else:
            iterations += 1

    # return best_cliques, best_qofs

    best_connections = [connection_from_clique(c, point_conversions, len(all_spot_qs)) for c in best_cliques]

    return best_connections, best_qofs



def lazy_writing(dset, image_data):
    for row_idx, row in tqdm(enumerate(image_data), total=len(image_data)):
        if isinstance(row, np.ndarray):
            dset[row_idx] = row
        else:
            for img_idx, img in enumerate(row.iter_chunks()):
                mask, comp = row.id.read_direct_chunk((img_idx, 0, 0))
                dset.id.write_direct_chunk((row_idx, img_idx, 0, 0), comp, mask)


def estimate_compression(data):
    _, counts = np.unique(data, return_counts=True)
    probs = counts / data.size
    entropy = -np.sum(probs * np.log2(probs))

    return {
        'entropy_bits' : entropy,
        'estimated_ratio' : 32 / max(entropy, 0.001),
        'should_compress' : entropy < 24exit
    }