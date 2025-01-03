def bandpass(data, low, high, sf, order):
    '''
        bandpass filter
        Parameters
        ----------
        low: bandpass low
        high: bandpas high
        sf: sampling frequency
    '''
    from scipy.signal import butter, lfilter, filtfilt, hilbert
    nyq = sf/2
    low = low/nyq
    high = high/nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def detect_serial_outliers(outliers):
    temp_group = [outliers[0]]
    result = []
    for i in range(1, len(outliers)):
        if outliers[i] == outliers[i - 1] + 1:
            temp_group.append(outliers[i])
        else:
            result.append(temp_group)
            temp_group = [outliers[i]]

    result.append(temp_group)
    
    return result

def correct_outliers(postion_XY, threshold):
    outliers=[]
    for i in range(postion_XY.shape[0]):
        filt_pos_ = bandpass(postion_XY[i], low=0.001, high=0.1, sf=fps, order=1)+np.mean(postion_XY[i])
        outliers.append(np.where(abs(postion_XY[i]-filt_pos_)>threshold)[0])
    cr_pos_XY=postion_XY.copy()
    for j in range(postion_XY.shape[0]):
        if outliers[j]!=[]:
            serial_outliers=detect_serial_outliers(outliers[j])
            for i in range(len(serial_outliers)):
                if len(cr_pos_XY[j]) > max(serial_outliers[i])+1:
                    if len(serial_outliers[i]) ==1:
                        cr_pos_XY[j][serial_outliers[i][0]]=np.linspace(cr_pos_XY[j][serial_outliers[i][0]-1],cr_pos_XY[j][serial_outliers[i][0]+1],1)
                    else:
                        cr_pos_XY[j][serial_outliers[i]]=np.linspace(cr_pos_XY[j][min(serial_outliers[i])-1],cr_pos_XY[j][max(serial_outliers[i])+1],len(serial_outliers[i]))
    return cr_pos_XY

def resample_data_to_target_rate(data, source_rate, target_rate):
    from scipy.interpolate import interp1d
    source_duration = len(data) / source_rate
    target_length = int(source_duration * target_rate)
    
    source_time = np.linspace(0, source_duration, num=len(data))
    target_time = np.linspace(0, source_duration, num=target_length)
    
    f_source_to_target = interp1d(source_time, data, kind='linear')
    resampled_data = f_source_to_target(target_time)
    
    return resampled_data

def pos_XY_2_binned_pos(pos_X, pos_Y, box_size, bin_size):
    '''
    input
        pos_X: animal X position vector in pixel
        pos_Y: animal Y position vector in pixel
        box_size: box size in pixel e.g., box_size=[1200, 1200]
        bin_size: bin size in pixel e.g., bin_size=30 *30 pixels/cm it dependes on system
    output
        position_id: list of which frame indices the animal was in each spatial bin
    '''
    xbins = np.arange(0, box_size[0] + bin_size, bin_size)
    ybins = np.arange(0, box_size[1] + bin_size, bin_size)
    position_id = []
    for i in range(len(ybins)-1):
        for k in range(len(xbins)-1):
            a = np.where((pos_X>=xbins[k]) & (pos_X<xbins[k+1]) & (pos_Y>=ybins[i]) & (pos_Y<ybins[i+1]))[0].tolist()
            position_id.append(a)
            
    return position_id

def calc_occupancy_map(position_id, binwidth):
    '''
    input
        position_id: list of which frame indices the animal was in each spatial bin
        binwidth: time window for binninng
    output
        occupacy_map
    '''
    occupacy=[]
    for i in range(len(position_id)):
        time_spent=len(position_id[i])*binwidth #staying time in each bins
        occupacy.append(time_spent)
    occupacy_map=np.array(occupacy).reshape(int(math.sqrt(len(position_id))), int(math.sqrt(len(position_id))))
    return occupacy_map

def calc_firing_map(act, position_id):
    '''
    input
        act: binned spikes for a neuron 
        position_id: list of which frame indices the animal was in each spatial bin
    output
        firing_map: spike count map of a neuron
    '''
    firing_map = np.zeros(len(position_id))
    for i in range(len(firing_map)):
        tmp=[]
        for k in range(len(position_id[i])):
            a = act[position_id[i][k]]
            tmp.append(a)
        tmp2=np.nansum(tmp)
        firing_map[i]=tmp2
    firing_map = firing_map.reshape(int(math.sqrt(len(position_id))), int(math.sqrt(len(position_id))))
    
    return firing_map

def calc_firing_rate_map(occupacy_map, firing_map, occupacy_th=0.2):
    occupacy_map_copy = occupacy_map.copy()
    occupacy_map_copy[np.where(occupacy_map<occupacy_th)]=np.nan
    firing_rate_map=firing_map/occupacy_map_copy
    
    return firing_rate_map



def calculate_information(occupancy_map, firing_rate_map):
    """
    Calculate spatial information per spike and per second.

    Parameters:
    - occupancy_map: 2D array representing time spent in each position bin.
    - firing_rate_map: 2D array representing the firing rate in each position bin.

    Returns:
    - information: Spatial information per spike (bits/spike).
    """
    import scipy
    occupancy_map=np.nan_to_num(occupancy_map)
    firing_rate_map=np.nan_to_num(firing_rate_map)   
    firing_rate_map=scipy.ndimage.gaussian_filter(firing_rate_map,2)
    total_time = np.sum(occupancy_map) 
    mean_firing_rate = np.sum(occupancy_map*firing_rate_map)/total_time
    information = 0
    for i in range(occupancy_map.shape[0]):
        for j in range(occupancy_map.shape[1]):
            pi = occupancy_map[i, j] / total_time  # Proportion of time spent in bin
            lambda_i = firing_rate_map[i, j]   # Firing rate in bin

            if pi > 0 and lambda_i > 0 and mean_firing_rate > 0:
                information += pi * (lambda_i / mean_firing_rate) * np.log2(lambda_i / mean_firing_rate)
    
    return information


def calc_FR_mod(firingrate_map,h_middle_line, v_middle_line, box_size=[1200,1200], bin_size=[30,30]):
    
    # the direction of box was invarted -> bottom right: female, upper left: empty
    FR_mod={}
    for i in firingrate_map.keys():
        firingrate_map_tmp = firingrate_map[i]
        all_mean_rate = np.nanmean(firingrate_map_tmp)
        bottom_right_mean_rate = np.nanmean(firingrate_map_tmp[int(h_middle_line/bin_size[0]-1):int(box_size[0]/bin_size[0]), 
                                                           int(v_middle_line/bin_size[1]):int(box_size[1]/bin_size[1])])
        FR_mod_tmp ={i: (bottom_right_mean_rate/all_mean_rate)}
        FR_mod.update(FR_mod_tmp)
    return FR_mod