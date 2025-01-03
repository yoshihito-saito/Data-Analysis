def calc_HFO_spikes(binraster, HFOs, window, binwidth):
    #z-scoring of binraster
    z_binraster = stats.zscore(binraster, axis=1).T
    #extract HFO coupling window
    HFOs_idx = np.where(HFOs==1)[0]
    pre_HFO = HFOs_idx-(window/binwidth)
    post_HFO = HFOs_idx+(window/binwidth)+1
    del_id_pre = np.where(pre_HFO<=0)
    pre_HFO = np.delete(pre_HFO, del_id_pre)
    post_HFO = np.delete(post_HFO, del_id_pre)
    del_id_post = np.where(post_HFO>binraster.shape[1])
    pre_HFO = np.delete(pre_HFO, del_id_post)
    post_HFO = np.delete(post_HFO, del_id_post)
    HFO_window = np.array([pre_HFO, post_HFO]).T
    HFO_window = np.array([np.arange(start, end, (end - start)/((window/binwidth)*2)) for start, end in HFO_window]).astype('int')
    HFO_spikes = z_binraster[HFO_window,:]
    
    return HFO_spikes

def calc_SW_spikes(binraster, SW_peak, window, binwidth):
    #z-scoring of binraster
    z_binraster = stats.zscore(binraster, axis=1).T
    #extract HFO coupling window
    SWup_idx = np.where(SW_peak==1)[0]
    pre_SWup = SWup_idx-(window/binwidth)
    post_SWup = SWup_idx+(window/binwidth)+1
    del_id_pre = np.where(pre_SWup<=0)
    pre_SWup = np.delete(pre_SWup, del_id_pre)
    post_SWup = np.delete(post_SWup, del_id_pre)
    del_id_post = np.where(post_SWup>z_binraster.shape[0])
    pre_SWup = np.delete(pre_SWup, del_id_post)
    post_SWup = np.delete(post_SWup, del_id_post)
    SW_window = np.array([pre_SWup, post_SWup]).T
    SW_window = np.array([np.arange(start, end+1, (end - start)/((window/binwidth)*2)) for start, end in SW_window]).astype('int')
    SW_spikes = z_binraster[SW_window,:]
    SW_spikes = np.mean(SW_spikes, axis=0).T
    return SW_spikes

def cellpairCofire(binraster_A, binraster_B):
    '''
        This function returns cell pair cofiring matrix

        inputs
        ----------
        binraster_A: binraster of region A (cell x time matrix)
        binraster_B: binraster of region B (cell x time matrix)

        Output
        ----------
        cofire: cofiring matrix of cell pairs between region A and B (cell x time matrix)
    '''    
    if (binraster_A!=[]) & (binraster_B!=[]):
        cofire = np.zeros([binraster_A.shape[0]*binraster_B.shape[0], binraster_A.shape[1]])
        count=0
        for i in range(binraster_A.shape[0]):
            fr_idx_A = np.zeros(binraster_A.shape[1])
            fr_idx_A[np.where(binraster_A[i]>=1)[0]]=1
            for k in range(binraster_B.shape[0]):
                fr_idx_B = np.zeros(binraster_B.shape[1])
                fr_idx_B[np.where(binraster_B[k]>=1)[0]]=1
                cofire_idx = np.where((fr_idx_A + fr_idx_B)==2)[0]
                cofire[count][cofire_idx]=1
                count=count+1
    else:
        cofire=[]

    return cofire
def HFO_SO_coupling(binned_HFO, binned_SO_peak, binwidth, coupling_window):
    """
        A function to detect coupling between High-Frequency Oscillations (HFO) and Slow Oscillations (SO).

        inputs:
        ----------
        binned_HFO : numpy array
            Binned data for HFO (an array of 0s and 1s). A value of 1 indicates the presence of an HFO.
        binned_SO_peak : numpy array
            Binned data for SO peaks. A value of -1 indicates the presence of an SO peak.
        binwidth : float
            The width of each bin in seconds, representing the time span each bin covers.
        coupling_window : list of float
            The time window in seconds for detecting coupling between SO and HFO. 
            It should be a list like [min_time_diff, max_time_diff].

        outputs:
        -------
        HFO_coupled_SO : numpy array
            A binary array indicating SO peaks that are coupled with HFOs (1s where coupling occurs).
        HFO_noncoupled_SO : numpy array
            A binary array indicating SO peaks that are not coupled with HFOs (1s where no coupling occurs).
    """
    binned_SO_peak_up = np.zeros(len(binned_SO_peak))
    binned_SO_peak_up[binned_SO_peak==-1]=1
    SO_peak_up_idx = np.where(binned_SO_peak_up==1)[0]
    HFO_idx = np.where(binned_HFO==1)[0]
    time_differences = []
    for i in SO_peak_up_idx:
        # Find the closest HFO peak
        closest_HFO = min(HFO_idx, key=lambda x: abs(x - i))
        # Calculate the time difference between the HFO and the  SO
        time_diff = i - closest_HFO
        time_differences.append(time_diff)
    time_diff = np.array(time_differences)

    nesting_time_window = (coupling_window[0]/binwidth, coupling_window[1]/binwidth)

    # Find the indices of the time differences that fall within the nesting time window
    HFO_coupled_SO_idx = np.where(np.logical_and(time_diff >= nesting_time_window[0], time_diff <= nesting_time_window[1]))[0]
    HFO_coupled_SO_idx = SO_peak_up_idx[HFO_coupled_SO_idx]
    HFO_noncoupled_SO_idx = np.setdiff1d(SO_peak_up_idx, HFO_coupled_SO_idx)
    HFO_coupled_SO = np.zeros(binned_SO_peak_up.shape[0])
    HFO_noncoupled_SO = np.zeros(binned_SO_peak_up.shape[0])
    HFO_coupled_SO[HFO_coupled_SO_idx] = 1
    HFO_noncoupled_SO[HFO_noncoupled_SO_idx] = 1

    return HFO_coupled_SO, HFO_noncoupled_SO

def SO_HFO_coupling(binned_HFO, binned_SO_peak, binwidth, coupling_window):
    """
        A function to detect coupling between High-Frequency Oscillations (HFO) and Slow Oscillations (SO).

        inputs:
        ----------
        binned_HFO : numpy array
            Binned data for HFO (an array of 0s and 1s). A value of 1 indicates the presence of an HFO.
        binned_SO_peak : numpy array
            Binned data for SO peaks. A value of -1 indicates the presence of an SO peak.
        binwidth : float
            The width of each bin in seconds, representing the time span each bin covers.
        coupling_window : list of float
            The time window in seconds for detecting coupling between SO and HFO. 
            It should be a list like [min_time_diff, max_time_diff].

        outputs:
        -------
        SO_coupled_HFO : numpy array
            A binary array indicating HFO_onset that are coupled with HFOs (1s where coupling occurs).
        SO_noncoupled_HFO : numpy array
            A binary array indicating HFO_onset that are not coupled with HFOs (1s where no coupling occurs).
    """
    binned_SO_peak_up = np.zeros(len(binned_SO_peak))
    binned_SO_peak_up[binned_SO_peak==-1]=1
    SO_peak_up_idx = np.where(binned_SO_peak_up==1)[0]
    HFO_idx = np.where(binned_HFO==1)[0]
    time_differences = []
    for i in HFO_idx:
        # Find the closest HFO peak
        closest_SO = min(SO_peak_up_idx, key=lambda x: abs(x - i))
        # Calculate the time difference between the HFO and the  SO
        time_diff = i - closest_SO
        time_differences.append(time_diff)
    time_diff = np.array(time_differences)

    nesting_time_window = (coupling_window[0]/binwidth, coupling_window[1]/binwidth)

    # Find the indices of the time differences that fall within the nesting time window
    SO_coupled_HFO_idx = np.where(np.logical_and(time_diff >= nesting_time_window[0], time_diff <= nesting_time_window[1]))[0]
    SO_coupled_HFO_idx = HFO_idx[SO_coupled_HFO_idx]
    SO_noncoupled_HFO_idx = np.setdiff1d(HFO_idx, SO_coupled_HFO_idx)
    SO_coupled_HFO = np.zeros(binned_HFO.shape[0])
    SO_noncoupled_HFO = np.zeros(binned_HFO.shape[0])
    SO_coupled_HFO[SO_coupled_HFO_idx] = 1
    SO_noncoupled_HFO[SO_noncoupled_HFO_idx] = 1

    return SO_coupled_HFO, SO_noncoupled_HFO