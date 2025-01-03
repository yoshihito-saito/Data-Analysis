def EMGrms(EMG, epoch, sf, emg_th, on_emg_th, output_path, plot=True, save_fig=False):
    '''
        This function returns the rms value of EMG in the defined epoch.
        Parameters
        ----------
        EMG: EMG signal
        epoch: epoch for rms calculation (sec)
        sf: sampling frequency
        emg_th: threshold for sleep scoring
        on_eng_th: threshold for sleep scoring during online scoring
    '''
    
    epochFrames = int(epoch*sf)
    L = len(EMG)
    if L%epochFrames>0:
        dellist = list(range(int(len(EMG)-len(EMG)%epochFrames),len(EMG)))
        EMG = np.delete(EMG, dellist)
    epochs = int(L/epochFrames)
    EMGpow = np.reshape(EMG,(epochFrames, epochs), order="F")
    EMGrms = np.power(EMGpow,2)
    EMGrms = np.sum(EMGrms, axis = 0)/epochFrames
    EMGrms = np.sqrt(EMGrms)
    
    if plot==True:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=100)
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        plt.plot(EMGrms, lw=0.5, color='royalblue')
        plt.plot([0, epochs],[on_emg_th, on_emg_th], lw = 1, color='gray', label='online')
        plt.plot([0, epochs],[emg_th, emg_th], lw = 1, color='r', label='offline')
        plt.xlim([0, epochs])
        plt.ylim([0, 50])
        plt.xlabel('epochs')
        plt.ylabel('EMGrms')
        plt.legend(loc='upper right')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        if save_fig==True:
            fig.savefig(output_path+'/EMGrms.png')
        else:
            print('No save')
    else:
        print('No Figure')
    
    return EMGrms



def sleep_scoring(EEG, EMGrms, epoch, sf, emg_th, eeg_th):
    '''
        This function returns the result of sleep scoring from EEG and EMG data.
        Parameters
        ----------
        EEG: EEG signal
        EMGrms: rms value of EMG in the epoch
        epoch: epoch for sleep scoring (sec)
        sf: sampling frequency
        emg_th: threshold for sleep scoring
        eeg_th: threshold of delta/theta ratio
    '''
    epochFrames = int(epoch*sf)
    L = len(EEG)
    if L%epochFrames>0:
        dellist = list(range(int(len(EEG)-len(EEG)%epochFrames),len(EEG)))
        EEG = np.delete(EEG, dellist)
    w = signal.hann(epochFrames, 'True')
    spectrum, freq, t = mlab.specgram(EEG, window=w, NFFT=epochFrames, Fs=sf, noverlap = 0, mode="psd")
   
    ix_D = (freq>1) & (freq<4) #delta band
    ix_T = (freq>6) & (freq<9) #theta band
    ix_N = (freq>1) & (freq<50) #using for the normilization

    pow_D = sum(abs(spectrum[ix_D,:]))
    pow_T = sum(abs(spectrum[ix_T,:]))
    pow_N = sum(abs(spectrum[ix_N,:]))

    nrm_pow_D = pow_D/pow_N
    nrm_pow_T = pow_T/pow_D
    dtratio = nrm_pow_D/nrm_pow_T
    
    offline_pre = []
    for i in range(len(EMGrms)):
        if EMGrms[i] >= emg_th:
            state = 0
        elif dtratio[i] >= eeg_th:
            state = 1
        else:
            state = 2
        offline_pre.append(state)
        
    #trim under 3epochs    
    trim = np.zeros(len(offline_pre))
    for i in range(2, len(offline_pre)):
        if (offline_pre[i] == offline_pre[i-1]) & (offline_pre[i] == offline_pre[i-2]):
            trim[i] = offline_pre[i]
            trim[i-1] = offline_pre[i]
            trim[i-2] = offline_pre[i]
        else:
            trim[i] = trim[i-1]
            
    #Omit awake -> REM transition
    sleep_state = np.zeros(len(trim))
    for i in range(1, len(trim)):
        if trim[i] - sleep_state[i-1] == 2:
            sleep_state[i] = 0
        else:
            sleep_state[i] = trim[i]
    #Convert to original sampling number
    sleep_state_org = np.zeros(len(EEG))
    for i in range(0, np.size(sleep_state)):
        if (sleep_state[i] == 0):
            sleep_state_org[i*epochFrames: (i+1)*epochFrames] = 0
        elif (sleep_state[i] == 1):
            sleep_state_org[i*epochFrames: (i+1)*epochFrames] = 1
        elif (sleep_state[i] == 2):
            sleep_state_org[i*epochFrames: (i+1)*epochFrames] = 2        
    if L%epochFrames>0:
        appendlist = [sleep_state_org[-1] for _ in range(L%epochFrames)]
        sleep_state_org = np.append(sleep_state_org, appendlist)
        sleep_state_org = sleep_state_org.astype(np.int16)
        
    return dtratio, sleep_state_org


def plot_sleepstate(EMGrms, emg_th, EEG, eeg_th, dtratio, sleep_state, epoch, sf, output_path, save_fig=False):
    '''
        plot sleep scoring result
        Parameters
        ----------
        EMGrms: rms value of EMG in the epoch
        EEG: EEG signal
        eeg_th: threshold for delta theta ratio
        dtratio: delta theta ratio
        sleep_state: sleep state
        epoch: epoch for sleep scoring (sec)
        sf: sampling frequency
    '''   
    fig = plt.figure(figsize=(10, 8), dpi=100)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    ax1 = fig.add_subplot(4,1,1)
    ax1.plot(EMGrms, lw=1, color='k')
    ax1.plot([0, len(EMGrms)],[emg_th, emg_th], lw = 1, color='r', label=emg_th)
    plt.xlim([0, len(EMGrms)])
    plt.xlabel('epoch')
    plt.ylabel('EMGrms')
    plt.legend(loc='upper right')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

    ax2 = fig.add_subplot(4,1,2)
    ax2.plot(np.log10(dtratio), lw=1, color='k')
    ax2.plot([0, len(dtratio)],[math.log10(eeg_th), math.log10(eeg_th)], lw = 1, color='r', label=eeg_th)
    plt.xlim([0, len(dtratio)])
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('Dleta/Theta')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    
    epochFrames = int(epoch*sf)
    ax3 = fig.add_subplot(4,1,3)
    w = signal.hann(epochFrames, 'True')
    spectrum = plt.specgram(EEG, window=w, NFFT=epochFrames, Fs=sf, noverlap=0, mode='psd', cmap='magma')
    plt.ylim([0, 50])
    plt.xlabel('time (s)')
    plt.ylabel('Frequency (Hz)')
    divider = make_axes_locatable(ax3)
    ax_cb = divider.append_axes("right", size="0%", pad=0)
    cbar = plt.colorbar(cax=ax_cb)
    plt.clim(35, 0)
    cbar.ax.axis('off')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('right')
    plt.gca().xaxis.set_ticks_position('bottom')

    ax4 = fig.add_subplot(4,1,4)
    ax4.plot(sleep_state, lw=1, color='k')
    plt.xlim([0, len(sleep_state)])
    plt.xlabel('frames')
    plt.ylabel('sleep state (offline)')
    plt.yticks(np.arange(0, 3, 1),['Awake','NREM','REM'])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

    fig.tight_layout()
    fig.canvas.draw()
    axpos1 = ax1.get_position() 
    axpos2 = ax2.get_position()
    axpos3 = ax3.get_position()
    axpos4 = ax4.get_position()

    ax2.set_position([axpos2.x0, axpos2.y0, axpos1.width, axpos2.height])
    ax3.set_position([axpos3.x0, axpos3.y0, axpos1.width, axpos3.height])
    ax4.set_position([axpos4.x0, axpos4.y0, axpos1.width, axpos4.height])
    if save_fig==True:
        fig.savefig(output_path+'/sleep_scoring.png')
    else:
        print('No save')
        

        
def art_check(LFP, art_epoch, sleep_state, lfp_th_w, lfp_th_s, art_emg_th, sf, plot=True):
    '''
        artifact check
        Parameters
        ----------
        LFP: LFP signal
        EMG: EMG signal
        sleep_state: sleep state
        art_LFP: artifact threshold for awake EEG
        art_EMG: artifact threshold for sleep EMG
        epoch: epoch for calculation (sec)
        sf: sampling frequecny
    '''   
    
    L = len(LFP)
    epochFrames = int(art_epoch*sf)
    epochs = int(L/epochFrames)
    
    if L%epochFrames>0:
        dellist = list(range(int(len(LFP)-len(LFP)%epochFrames),len(LFP)))
        LFP_trim = np.delete(LFP, dellist)
        w = signal.hann(epochFrames, 'True')
        LFP_pow = np.reshape(LFP_trim,(epochFrames,epochs), order="F")
        LFPrms = np.power(LFP_pow,2)
        LFPrms = np.sum(LFPrms, axis = 0)/epochFrames
        LFPrms = np.sqrt(LFPrms)
    else:
        w = signal.hann(epochFrames, 'True')
        LFP_pow = np.reshape(LFP,(epochFrames,epochs), order="F")
        LFPrms = np.power(LFP_pow,2)
        LFPrms = np.sum(LFPrms, axis = 0)/epochFrames
        LFPrms = np.sqrt(LFPrms)
    
    convert_rate=100 #100Hz
    sleep_state_d = sleep_state[::int(sf/convert_rate)]
    LFPrms_d = LFPrms[::int(sf/convert_rate)]
    epochFrames_d = epochFrames[::int(sf/convert_rate)]
    epochs_d = int(epochs/(sf/convert_rate))
    state_epoch = sleep_state_d[0:len(sleep_state_d):epochFrames_d]
    if plot==True:
        fig = plt.figure(figsize=(8, 6), dpi=100)
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        ax1 = fig.add_subplot(3,1,1)
        ax1.plot(LFPrms_d, lw=0.5, color='k')
        ax1.plot([0, epochs_d],[lfp_th_w, lfp_th_w], lw = 1, color='g', label=lfp_th_w)
        ax1.plot([0, epochs_d],[lfp_th_s, lfp_th_s], lw = 1, color='r', label=lfp_th_s)
        ax1.plot(state_epoch*400, lw=1, color='b', label='state')
        plt.xlim([0, epochs_d])
        plt.ylim([-100, lfp_th_s+300])
        plt.xlabel('epoch')
        plt.ylabel('LFP rms')
        plt.legend(loc='upper right')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        
        fig.tight_layout()
        fig.canvas.draw()

    else:
        print('No Figure')
    
    return LFPrms


def art_rej(EMG, art_emg_th, sleep_state, art_epoch, sf):
    '''
        remove artifact epochs
        Parameters
        ----------
        EMG: EMG signal
        lfp_th_w
        lfp_th_s
        art_emg_th
        sleep_state: sleep state
        art_epoch: epoch for calculation (sec)
        sf: sampling frequecny
    '''       
    L = len(EMG)
    epochFrames = int(art_epoch*sf)
    epochs = int(L/epochFrames)

    if L%epochFrames>0:
        dellist = list(range(int(len(EMG)-len(EMG)%epochFrames),len(EMG)))
        EMG = np.delete(EMG, dellist)
    EMGpow = np.reshape(EMG,(epochFrames, epochs), order="F")
    EMGrms = np.power(EMGpow,2)
    EMGrms = np.sum(EMGrms, axis = 0)/epochFrames
    EMGrms = np.sqrt(EMGrms)
    
    state_epoch = sleep_state[0:L-L%epochFrames:epochFrames]
    
    offline_rej = []
    for i in range(len(state_epoch)):
        if (state_epoch[i]==1) & (EMGrms[i] > art_emg_th):
            art_rej = 4
        elif (state_epoch[i]==1) & (EMGrms[i] <= art_emg_th):
            art_rej = 1
        elif (state_epoch[i]==2) & (EMGrms[i] > art_emg_th):
            art_rej = 4
        elif (state_epoch[i]==2) & (EMGrms[i] <= art_emg_th):
            art_rej = 2
        else:
            art_rej = 0
        offline_rej.append(art_rej)
        #0: awake, 1:NREM, 2:REM, 4:sleep artifact
        
    #Convert to original sampling number
    sleep_state_rej = np.zeros(len(EMG))
    for i in range(len(state_epoch)):
        if (offline_rej[i] == 0):
            sleep_state_rej[i*epochFrames: (i+1)*epochFrames] = 0
        elif (offline_rej[i] == 1):
            sleep_state_rej[i*epochFrames: (i+1)*epochFrames] = 1
        elif (offline_rej[i] == 2):
            sleep_state_rej[i*epochFrames: (i+1)*epochFrames] = 2
        elif (offline_rej[i] == 3):
            sleep_state_rej[i*epochFrames: (i+1)*epochFrames] = 3
        elif (offline_rej[i] == 4):
            sleep_state_rej[i*epochFrames: (i+1)*epochFrames] = 4

    if L%epochFrames>0:
        appendlist = [sleep_state_rej[-1] for _ in range(L%epochFrames)]
        sleep_state_rej = np.append(sleep_state_rej, appendlist)
        sleep_state_rej = sleep_state_rej.astype(np.int16)
            
    return sleep_state_rej