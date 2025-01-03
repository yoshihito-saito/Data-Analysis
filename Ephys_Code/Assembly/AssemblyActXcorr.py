def get_region(index, unit_BLA_idx, unit_M2_idx, unit_S1_idx):
    if index in unit_BLA_idx:
        return "BLA"
    elif index in unit_M2_idx:
        return "M2"
    elif index in unit_S1_idx:
        return "S1"
    else:
        return "Unknown"

# Function to find unique pairs and their region combinations
from itertools import combinations
def find_pairs_with_regions(matrix, unit_BLA_idx, unit_M2_idx, unit_S1_idx):
    total_members = matrix.shape[1]  # Total number of possible members
    all_pairs = set()  # Set to store all possible pairs
    appeared_pairs = set()  # Set to store all appeared pairs

    # Generate all possible pairs from all members
    for pair in combinations(range(total_members), 2):
        all_pairs.add(pair)

    # Loop through each row in the matrix
    for row in matrix:
        # Find indices of members in the ensemble
        members = np.where(row == 1)[0]
        # Generate all pairs for this ensemble
        member_pairs = combinations(members, 2)
        # Add these pairs to appeared_pairs
        appeared_pairs.update(member_pairs)

    # Find pairs that never appeared
    never_appeared_pairs = all_pairs - appeared_pairs

    # Create a list of tuples (pair, region_combination) and sort it
    appeared_pairs_with_regions = sorted([
        (pair, (get_region(pair[0], unit_BLA_idx, unit_M2_idx, unit_S1_idx), get_region(pair[1], unit_BLA_idx, unit_M2_idx, unit_S1_idx))) for pair in appeared_pairs
    ])

    never_pairs_with_regions = sorted([
        (pair, (get_region(pair[0], unit_BLA_idx, unit_M2_idx, unit_S1_idx), get_region(pair[1], unit_BLA_idx, unit_M2_idx, unit_S1_idx))) for pair in never_appeared_pairs
    ])

    # Extract sorted pairs and their regions
    appeared_pairs_array = np.array([pair[0] for pair in appeared_pairs_with_regions])
    appeared_regions_array = np.array([pair[1] for pair in appeared_pairs_with_regions])

    never_pairs_array = np.array([pair[0] for pair in never_pairs_with_regions])
    never_regions_array = np.array([pair[1] for pair in never_pairs_with_regions])

    return appeared_pairs_array, appeared_regions_array, never_pairs_array, never_regions_array


def detectReactivation(R):
    React_Event = np.where(R>5)[0]
    return React_Event

def reactCorrelogram(binraster_A, binraster_B, React_Event, binwidth_react=0.025, binwidth_fine=0.001):
    #ver 2
    window=0.05
    nCells_A = binraster_A.shape[0]
    nCells_B = binraster_B.shape[0]


    React_Event_fine = ((React_Event)*binwidth_react/binwidth_fine).astype('int64')
    react_idx_fine = np.vstack([React_Event_fine, React_Event_fine+(binwidth_react/binwidth_fine)]).T.astype('int64')

    #delete index of outside window

    del_idx=np.where((react_idx_fine-(window/binwidth_fine))[:,0]<0)[0]
    react_idx_fine=np.delete(react_idx_fine,del_idx,0)
    del_idx=np.where((react_idx_fine+(window/binwidth_fine))[:,1]>len(binraster_A[0]))[0]
    react_idx_fine=np.delete(react_idx_fine,del_idx,0)

    #numbers of reactivation
    nReact = react_idx_fine.shape[0]
    #z-scoring
    binraster_B_z = stats.zscore(binraster_B, axis=1)

    #reference spike index within reactivations
    ref_spiketime = []
    for k in range(binraster_A.shape[0]):
        spiketime=np.where(binraster_A[k]>=1)[0]
        ref_spiketime_tmp=[]
        for i in range(nReact):
            ref_spiketime_tmp.extend(spiketime[np.where((react_idx_fine[i][0]<=spiketime) & (spiketime<=react_idx_fine[i][1]))[0]])
        ref_spiketime.append(ref_spiketime_tmp)

    ReactCCG=[]
    for i in range(nCells_A):
        if ref_spiketime[i]!=[]:
            norm_ReactCCG_tmp=[]
            for ii in range(nCells_B):
                tmp = []
                for iii in range(len(ref_spiketime[i])):
                    tmp.append(binraster_B_z[ii][int(ref_spiketime[i][iii]-(window/binwidth_fine)):int(ref_spiketime[i][iii]+(window/binwidth_fine))])
                norm_ReactCCG_tmp.append(np.sum(tmp,axis=0)/nReact) #normilized by reactivation number

            ReactCCG.extend(norm_ReactCCG_tmp)
    ReactCCG = np.array(ReactCCG)
    
    return ReactCCG