def HFO_SO_coupling(binned_HFO, binned_SO_peak):
    binwidth=0.025
    binned_SO_peak_up = np.zeros(len(binned_SO_peak))
    binned_SO_peak_up[binned_SO_peak==-1]=1
    HFO_idx = np.where(binned_HFO==1)[0]
    SO_peak_up_idx = np.where(binned_SO_peak_up==1)[0]
    
    time_differences = []
    for i in HFO_idx:
        # Find the closest SO peak
        closest_so = min(SO_peak_up_idx, key=lambda x: abs(x - i))
        # Calculate the time difference between the HFO and the  SO
        time_diff = closest_so-i
        time_differences.append(time_diff)
    time_diff = np.array(time_differences)
    
    nesting_time_window = (-0.1/binwidth, 0.1/binwidth)

    # Find the indices of the time differences that fall within the nesting time window
    SO_coupled_HFO_idx = np.where(np.logical_and(time_diff >= nesting_time_window[0], time_diff <= nesting_time_window[1]))[0]
    
    return time_diff, SO_coupled_HFO_idx