class unitPreprocess:
    def __init__(self, mouseID, session_info, classified_unit, binwidth, out_Dir):
        self.session_info = session_info
        
        #load unit data
        def unit_load(mouseID, session_info):
            M2_unitDir = glob.glob(session_info['sessionDir'][mouseID]+'/04_spike_output_MS/unit_spikes_M2.h5')
            S1_unitDir = glob.glob(session_info['sessionDir'][mouseID]+'/04_spike_output_MS/unit_spikes_S1.h5')
            BLA_unitDir = glob.glob(session_info['sessionDir'][mouseID]+'/04_spike_output_MS/unit_spikes_BLA.h5')

            M2_unith5 = h5py.File(M2_unitDir[0], 'r') 
            S1_unith5 = h5py.File(S1_unitDir[0], 'r')
            BLA_unith5 = h5py.File(BLA_unitDir[0], 'r')

            sf = BLA_unith5['sf'].value
            frames = BLA_unith5['frames'].value

            unit_id_M2_RS = classified_unit[(classified_unit['quality']=='single') & (classified_unit['type']=='RS') & (classified_unit['location']=='true')
                                          & (classified_unit['region']=='M2') & (classified_unit['mouseID']==mouseID)]['unitID'].tolist()
            unit_id_S1_RS = classified_unit[(classified_unit['quality']=='single') & (classified_unit['type']=='RS') & (classified_unit['location']=='true')
                                          & (classified_unit['region']=='S1') & (classified_unit['mouseID']==mouseID)]['unitID'].tolist()
            unit_id_BLA_RS = classified_unit[(classified_unit['quality']=='single') & (classified_unit['type']=='RS') & (classified_unit['location']=='true')
                                           & (classified_unit['region']=='BLA') & (classified_unit['mouseID']==mouseID)]['unitID'].tolist()

            unit_id_M2_FS = classified_unit[(classified_unit['quality']=='single') & (classified_unit['type']=='FS') & (classified_unit['location']=='true')
                                          & (classified_unit['region']=='M2') & (classified_unit['mouseID']==mouseID)]['unitID'].tolist()
            unit_id_S1_FS = classified_unit[(classified_unit['quality']=='single') & (classified_unit['type']=='FS') & (classified_unit['location']=='true')
                                          & (classified_unit['region']=='S1') & (classified_unit['mouseID']==mouseID)]['unitID'].tolist()
            unit_id_BLA_FS = classified_unit[(classified_unit['quality']=='single') & (classified_unit['type']=='FS') & (classified_unit['location']=='true')
                                           & (classified_unit['region']=='BLA') & (classified_unit['mouseID']==mouseID)]['unitID'].tolist()           
            unit_id = {'unit_id_M2_RS': unit_id_M2_RS, 'unit_id_S1_RS': unit_id_S1_RS, 'unit_id_BLA_RS': unit_id_BLA_RS,
                        'unit_id_M2_FS': unit_id_M2_FS, 'unit_id_S1_FS': unit_id_S1_FS, 'unit_id_BLA_FS': unit_id_BLA_FS}
            
            return unit_id, sf, frames, M2_unith5, S1_unith5, BLA_unith5
        
        unit_id, sf, frames, M2_unith5, S1_unith5, BLA_unith5 = unit_load(mouseID, session_info)
        
        self.sf=sf
        self.frames=frames
        
        #load TTL timing
        
        ttlDir = glob.glob(self.session_info['sessionDir'][mouseID]+'/00_TTL_output/*.h5')
        ttlh5 = h5py.File(ttlDir[0],'r')
        epoch_TTL_timing = ttlh5['epoch_TTL_timing'].value
        pre_epoch_id = list([0,epoch_TTL_timing[1]-epoch_TTL_timing[0]])
        task_epoch_id = list([epoch_TTL_timing[1]-epoch_TTL_timing[0], epoch_TTL_timing[1]-epoch_TTL_timing[0]+epoch_TTL_timing[3]-epoch_TTL_timing[2]])
        post_epoch_id = list([epoch_TTL_timing[1]-epoch_TTL_timing[0]+epoch_TTL_timing[3]-epoch_TTL_timing[2],epoch_TTL_timing[1]-epoch_TTL_timing[0]+epoch_TTL_timing[3]-epoch_TTL_timing[2]+epoch_TTL_timing[5]-epoch_TTL_timing[4]])   
        ttlh5.flush()
        ttlh5.close()
        
        #load sleep data
        stateDir = glob.glob(self.session_info['sessionDir'][mouseID]+'/05_sleepstate_output/*.h5')
        stateh5 = h5py.File(stateDir[0], 'r')
        sleep_state = stateh5['sleep_state'].value
        stateh5.flush()
        stateh5.close()
        sleep_state_pre = sleep_state[pre_epoch_id[0]:pre_epoch_id[1]]
        sleep_state_task = sleep_state[task_epoch_id[0]:task_epoch_id[1]]
        sleep_state_post = sleep_state[post_epoch_id[0]:post_epoch_id[1]]
        
        state = {'pre':sleep_state_pre, 'task':sleep_state_task, 'post':sleep_state_post}
        self.state=state
        def unit_id_list(unit_id):
            unit_id_list = []
            for i in range(len(unit_id)):
                a = 'unit'+ str('{0:03d}'.format(unit_id[i]))
                unit_id_list.append(a)
                
            return unit_id_list
        
        unit_id_list_M2_RS = unit_id_list(unit_id['unit_id_M2_RS'])
        unit_id_list_S1_RS = unit_id_list(unit_id['unit_id_S1_RS'])
        unit_id_list_BLA_RS = unit_id_list(unit_id['unit_id_BLA_RS'])
        unit_id_list_M2_FS = unit_id_list(unit_id['unit_id_M2_FS'])
        unit_id_list_S1_FS = unit_id_list(unit_id['unit_id_S1_FS'])
        unit_id_list_BLA_FS = unit_id_list(unit_id['unit_id_BLA_FS'])
        unit_id_list = {'M2_RS': unit_id_list_M2_RS, 'S1_RS': unit_id_list_S1_RS, 'BLA_RS': unit_id_list_BLA_RS, 
                        'M2_FS': unit_id_list_M2_FS, 'S1_FS': unit_id_list_S1_FS, 'BLA_FS': unit_id_list_BLA_FS}
        self.unit_id_list=unit_id_list
        
        def unit_sptrain(unith5, unit_id_list, region, unitType):
            #read unit firing index
            sort_id = region+'_'+unitType
            if unit_id_list[sort_id]!=[]:
                unit_spikes = {}
                for i in unit_id_list[sort_id]:
                    a = unith5['unit'][i]['sp_train'].value
                    b = {i:a}
                    unit_spikes.update(b)
            
                #firing index -> 0 or 1 spike train
                unit_sptrain = {}            
                for i in list(unit_spikes.keys()):
                    a = np.zeros((frames), dtype=int)
                    b = {i:a}
                    unit_sptrain.update(b)
                for k in list(unit_spikes.keys()):
                    for i in unit_spikes[k]:
                        unit_sptrain[k][i]=1
            else:
                unit_sptrain=[]
            return unit_sptrain


        unit_sptrain_M2_RS=unit_sptrain(M2_unith5, unit_id_list, region='M2', unitType='RS')
        unit_sptrain_S1_RS=unit_sptrain(S1_unith5, unit_id_list, region='S1', unitType='RS')
        unit_sptrain_BLA_RS=unit_sptrain(BLA_unith5, unit_id_list, region='BLA', unitType='RS')     
        unit_sptrain_M2_FS=unit_sptrain(M2_unith5, unit_id_list, region='M2', unitType='FS')
        unit_sptrain_S1_FS=unit_sptrain(S1_unith5, unit_id_list, region='S1', unitType='FS')
        unit_sptrain_BLA_FS=unit_sptrain(BLA_unith5, unit_id_list, region='BLA', unitType='FS')
        M2_unith5.close, S1_unith5.close, BLA_unith5.close()
        
        
        def unit_sptrain_epoch(unit_sptrain, pre_epoch_id, task_epoch_id, post_epoch_id):
            if unit_sptrain!=[]:
                unit_sptrain_pre={}
                unit_sptrain_task={}
                unit_sptrain_post={}
                for i in list(unit_sptrain.keys()):
                    if sum(unit_sptrain[i][task_epoch_id[0]: task_epoch_id[1]])!=0:
                        a = unit_sptrain[i][pre_epoch_id[0]: pre_epoch_id[1]]
                        b = {i:a}
                        unit_sptrain_pre.update(b)
                        a = unit_sptrain[i][task_epoch_id[0]: task_epoch_id[1]]
                        b = {i:a}
                        unit_sptrain_task.update(b)            
                        a = unit_sptrain[i][post_epoch_id[0]: post_epoch_id[1]]
                        b = {i:a}
                        unit_sptrain_post.update(b)
                unit_sptrain_epoch = {'pre':unit_sptrain_pre, 'task':unit_sptrain_task, 'post':unit_sptrain_post}
            else:
                unit_sptrain_epoch = {'pre':[], 'task':[], 'post':[]}
                
            return unit_sptrain_epoch
         
        unit_sptrain_M2_RS_epoch = unit_sptrain_epoch(unit_sptrain_M2_RS, pre_epoch_id=pre_epoch_id, task_epoch_id=task_epoch_id, post_epoch_id=post_epoch_id)
        unit_sptrain_S1_RS_epoch = unit_sptrain_epoch(unit_sptrain_S1_RS, pre_epoch_id=pre_epoch_id, task_epoch_id=task_epoch_id, post_epoch_id=post_epoch_id)
        unit_sptrain_BLA_RS_epoch = unit_sptrain_epoch(unit_sptrain_BLA_RS, pre_epoch_id=pre_epoch_id, task_epoch_id=task_epoch_id, post_epoch_id=post_epoch_id)
        
        unit_sptrain_M2_FS_epoch = unit_sptrain_epoch(unit_sptrain_M2_FS, pre_epoch_id=pre_epoch_id, task_epoch_id=task_epoch_id, post_epoch_id=post_epoch_id)
        unit_sptrain_S1_FS_epoch = unit_sptrain_epoch(unit_sptrain_S1_FS, pre_epoch_id=pre_epoch_id, task_epoch_id=task_epoch_id, post_epoch_id=post_epoch_id)
        unit_sptrain_BLA_FS_epoch = unit_sptrain_epoch(unit_sptrain_BLA_FS, pre_epoch_id=pre_epoch_id, task_epoch_id=task_epoch_id, post_epoch_id=post_epoch_id)
        
        
        def binraster_epoch(unit_sptrain_epoch, binwidth, sf):
            def binned_spike(unit_sptrain, binwidth, sf):
                binwidth = int(binwidth*sf)
                unit_id_list = list(unit_sptrain.keys())
                binnend_spikes={}
                for i in unit_id_list:
                    sptrain = unit_sptrain[i][:-int(unit_sptrain[i].shape[0]%int(0.025*sf))]
                    a = sptrain.reshape(-1, binwidth).sum(axis=1)
                    b = {i:a}
                    binnend_spikes.update(b) 
                return binnend_spikes
            
            if unit_sptrain_epoch['task']!=[]:
                binraster_PRE=binned_spike(unit_sptrain_epoch['pre'], binwidth=binwidth, sf=int(sf))
                binraster_Task=binned_spike(unit_sptrain_epoch['task'], binwidth=binwidth, sf=int(sf))
                binraster_POST=binned_spike(unit_sptrain_epoch['post'], binwidth=binwidth, sf=int(sf))
                binraster = {'PRE':binraster_PRE,'Task':binraster_Task,'POST':binraster_POST}
            else:
                binraster=[]
            return binraster

        M2_RS_binraster = binraster_epoch(unit_sptrain_M2_RS_epoch, binwidth=binwidth, sf=self.sf)
        S1_RS_binraster = binraster_epoch(unit_sptrain_S1_RS_epoch, binwidth=binwidth, sf=self.sf)
        BLA_RS_binraster = binraster_epoch(unit_sptrain_BLA_RS_epoch, binwidth=binwidth, sf=self.sf)      
        M2_FS_binraster = binraster_epoch(unit_sptrain_M2_FS_epoch, binwidth=binwidth, sf=self.sf)
        S1_FS_binraster = binraster_epoch(unit_sptrain_S1_FS_epoch, binwidth=binwidth, sf=self.sf)
        BLA_FS_binraster = binraster_epoch(unit_sptrain_BLA_FS_epoch, binwidth=binwidth, sf=self.sf)
        
        def binning_sleep_state(sleep_state, pre_epoch_id, task_epoch_id, post_epoch_id, binwidth, sf):
            def binupsampling(signal_d, binwidth_wide, binwidth):
                signal_up = np.zeros(int(len(signal_d)*(binwidth_wide/binwidth)))
                for i,k in enumerate(signal_d):
                    signal_up[int(i*(binwidth_wide/binwidth)):int((i+1)*(binwidth_wide/binwidth))] = k
                return signal_up
            sleep_state_pre = sleep_state[pre_epoch_id[0]:pre_epoch_id[1]]
            sleep_state_task = sleep_state[task_epoch_id[0]:task_epoch_id[1]]
            sleep_state_post = sleep_state[post_epoch_id[0]:post_epoch_id[1]]
            
            #binning in 25 ms binwidth
            if sleep_state_pre.shape[0]%(sf*0.025) != 0:
                sleep_state_pre = sleep_state_pre[:-int(sleep_state_pre.shape[0]%int(0.025*sf))]
            if sleep_state_task.shape[0]%(sf*0.025) != 0:
                sleep_state_task = sleep_state_task[:-int(sleep_state_task.shape[0]%int(0.025*sf))]
            if sleep_state_post.shape[0]%(sf*0.025) != 0:
                sleep_state_post = sleep_state_post[:-int(sleep_state_post.shape[0]%int(0.025*sf))]

            binned_sleep_state_pre = sleep_state_pre[::int(sf*0.025)]
            binned_sleep_state_task = sleep_state_task[::int(sf*0.025)]
            binned_sleep_state_post = sleep_state_post[::int(sf*0.025)]
            
            # up sampling to keep state indices
            binned_sleep_state_pre = binupsampling(binned_sleep_state_pre,  0.025,binwidth)
            binned_sleep_state_task = binupsampling(binned_sleep_state_task, 0.025,binwidth)
            binned_sleep_state_post = binupsampling(binned_sleep_state_post, 0.025,binwidth)
            
            
            binned_sleep_state = {'PRE':binned_sleep_state_pre,'Task':binned_sleep_state_task,'POST':binned_sleep_state_post}
            return binned_sleep_state
        
        binned_sleep_state = binning_sleep_state(sleep_state, pre_epoch_id, task_epoch_id, post_epoch_id, binwidth=binwidth, sf=self.sf)
        
        binraster = {'M2_RS': M2_RS_binraster, 'S1_RS': S1_RS_binraster, 'BLA_RS': BLA_RS_binraster,
                     'M2_FS': M2_FS_binraster, 'S1_FS': S1_FS_binraster, 'BLA_FS': BLA_FS_binraster,
                     'State': binned_sleep_state}

        os.makedirs(out_Dir, exist_ok=True)
        f_name = out_Dir+'/'+ mouseID +'_binraster.pkl'
        
        # self.binraster=binraster
        with open(f_name, mode='wb') as f:
            pickle.dump(binraster,f)
        
        
        
        del unit_sptrain_M2_RS, unit_sptrain_S1_RS, unit_sptrain_BLA_RS, unit_sptrain_M2_FS, unit_sptrain_S1_FS, unit_sptrain_BLA_FS,\
        unit_sptrain_M2_RS_epoch, unit_sptrain_S1_RS_epoch, unit_sptrain_BLA_RS_epoch,\
        unit_sptrain_M2_FS_epoch, unit_sptrain_S1_FS_epoch, unit_sptrain_BLA_FS_epoch,\
        M2_RS_binraster, S1_RS_binraster, BLA_RS_binraster, M2_FS_binraster, S1_FS_binraster, BLA_FS_binraster, sleep_state, binned_sleep_state