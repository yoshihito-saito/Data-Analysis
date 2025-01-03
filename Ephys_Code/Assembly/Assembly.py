def AssemblyWeight(Qref):
    '''
        This function returns normarized weight vectors of the assembly patterns
        Parameters
        ----------
        Qref: binned spike trains during reference epoch
    '''    
    from sklearn.decomposition import FastICA
    
    Qref = stats.zscore(Qref, axis=1)
    nCells = len(Qref)
    
    def pcacov(C):
        lambdas, PCs = np.linalg.eigh(C)
        for i in range(len(PCs[0])):
            if PCs[:,i][np.argmax(abs(PCs[:,i]))] < 0:
                PCs[:,i] = PCs[:,i]*-1
        PCs = np.fliplr(PCs)
        PCs = np.round(PCs, 4)
        return lambdas, PCs

    Cref = np.corrcoef(Qref);
    lambdas, PCs = pcacov(Cref);

    lMax = (1 + math.sqrt(nCells / len(Qref[1])))**2
    nPCs = sum(lambdas>lMax);
    if nPCs > 0:
        phi = lambdas[np.array(np.where(lambdas>lMax)).min():np.array(np.where(lambdas>lMax)).max()+1]/lMax
        phi = phi[::-1]
        PCs = PCs[:,:nPCs]
        ##van de Ven et al 2016
        Zproj = Qref.T.dot(PCs)
        ica = FastICA(max_iter=500000) 
        source = ica.fit_transform(Zproj)
        icaW = ica._unmixing.T  
        Vec=(PCs.dot(icaW)) 
        nW=Vec/np.sqrt(sum(Vec**2))
        for i in range(len(nW[0])):
            if nW[:,i][np.argmax(abs(nW[:,i]))] < 0:
                nW[:,i] = nW[:,i]*-1
    else:
        print('no significant PCs')
        nW=np.zeros([Qref.shape[0],1])
    return nW

def AssemblyStrength(nW, Qtar):
    '''
        This function returns assembly strength
        Parameters
        ----------
        nW: normilized weight vector
        Qtar: binned spike trains during target epoch
    '''    
    
    Qtar = stats.zscore(Qtar, axis=1)
    Pk = np.outer(nW, nW)
    Pk[[range(len(nW)), range(len(nW))]] = 0
    Rk = Qtar.T.dot(Pk)*Qtar.T
    react = np.sum(Rk, axis=1)
    
    return react

def AssemblyStrength_intregion(nW, unit_region_N, Qtar):
    '''
        This function returns assembly strength
        Parameters
        ----------
        nW: normilized weight vector
        Qtar: binned spike trains during target epoch
    '''  
    unit_BLA_N=unit_region_N['BLA']
    unit_M2_N=unit_region_N['M2']
    unit_S1_N=unit_region_N['S1']
    
    Qtar = stats.zscore(Qtar, axis=1)
    Pk = np.outer(nW, nW)
    
    Pk[:unit_BLA_N,:unit_BLA_N]=0
    Pk[unit_BLA_N:unit_BLA_N+unit_M2_N,unit_BLA_N:unit_BLA_N+unit_M2_N]=0
    Pk[unit_BLA_N+unit_M2_N:unit_BLA_N+unit_M2_N+unit_S1_N,unit_BLA_N+unit_M2_N:unit_BLA_N+unit_M2_N+unit_S1_N]=0
    Pk[[range(len(nW)), range(len(nW))]] = 0
    
    Rk = Qtar.T.dot(Pk)*Qtar.T
    react = np.sum(Rk, axis=1)
    
    return react


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

def pos_XY_2_binned_pos(pos_X, pos_Y, box_size=[1200, 1200], bin_size=30):
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


def calc_act_map(act, position_id):
    '''
    input
        act: activation raster time series of an assembly (binarized by threshold of 5)
        position_id: list of which frame indices the animal was in each spatial bin
    output
        activation_map: activation count map of a neuron
    '''
    act_map = np.zeros(len(position_id))
    for i in range(len(act_map)):
        tmp=[]
        for k in range(len(position_id[i])):
            a = act[position_id[i][k]]
            tmp.append(a)
        tmp2=np.nansum(tmp)
        act_map[i]=tmp2
    act_map = act_map.reshape(int(math.sqrt(len(position_id))), int(math.sqrt(len(position_id))))
    
    return act_map

def calc_activation_rate_map(occupacy_map, activation_map, occupacy_th=0.2):
    occupacy_map_copy = occupacy_map.copy()
    occupacy_map_copy[np.where(occupacy_map<occupacy_th)]=np.nan
    activation_rate_map=activation_map/occupacy_map_copy
    
    return activation_rate_map


def calc_HFO_react(react, onset_HFO, window=1, binwidth=0.025, sigma=1):
    pre_HFO = onset_HFO-(window/binwidth)
    post_HFO = onset_HFO+(window/binwidth)

    onset_HFO = np.round(onset_HFO)
    pre_HFO = np.round(pre_HFO)
    post_HFO = np.round(post_HFO)

    del_id_pre = np.where(pre_HFO<=0)
    pre_HFO = np.delete(pre_HFO, del_id_pre)
    post_HFO = np.delete(post_HFO, del_id_pre)

    del_id_post = np.where(post_HFO>react.shape[1])
    pre_HFO = np.delete(pre_HFO, del_id_post)
    post_HFO = np.delete(post_HFO, del_id_post)

    HFO_react = []
    for k in range(react.shape[0]):
        tmp=[]
        for i in range(len(pre_HFO)):
            a = react[k][int(pre_HFO[i]):int(post_HFO[i])]
            tmp.append(a)
        tmp_mean = np.mean(tmp,axis=0)
        tmp_mean = scipy.ndimage.filters.gaussian_filter(tmp_mean, sigma=sigma).tolist()
        HFO_react.append(tmp_mean)

    return HFO_react