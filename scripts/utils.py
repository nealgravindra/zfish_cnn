# utils

def parse_filenames(file_list, n_test=5, verbose=True) :
    '''Train/test split for 3 timepoint classes'''
    dpf3 = []
    dpf4 = []
    dpf5 = []
    count = 0
    for f in file_list :
        if '3dpf' in f :
            dpf3.append(f)
        elif '4dpf' in f :
            dpf4.append(f)
        elif '5dpf' in f :
            dpf5.append(f)
        else :
            count+=1
            print('{} files in data folder not matched.\n...Problem file: {}'.format(count,f))

    if verbose :
        print('\n3dpf has {}-images'.format(len(dpf3)))
        print('4dpf has {}-images'.format(len(dpf4)))
        print('5dpf has {}-images\n'.format(len(dpf5)))

    dpf3_test = random.sample(dpf3,n_test)
    dpf3_train= [i for i in dpf3 if i not in dpf3_test]
    dpf4_test = random.sample(dpf4,n_test)
    dpf4_train= [i for i in dpf4 if i not in dpf4_test]
    dpf5_test = random.sample(dpf5,n_test)
    dpf5_train= [i for i in dpf5 if i not in dpf5_test]

    if verbose :
        # for reproducibility, hard code test set op to get train
        print('3 DPF test set...')
        for i in dpf3_test :
            print('    {}'.format(os.path.split(i)[1].split('.npy')[0]))
        print('\n4 DPF test set...')
        for i in dpf4_test :
            print('    {}'.format(os.path.split(i)[1].split('.npy')[0]))
        print('\n5 DPF test set...')
        for i in dpf5_test :
            print('    {}'.format(os.path.split(i)[1].split('.npy')[0]))

    return dpf3_train,dpf4_train,dpf5_train,dpf3_test,dpf4_test,dpf5_test

def get_im(fname) :
    '''returns numpy array on range [0,1]'''
    img = tifffile.imread(fname)
    if len(img.shape)>2 : #Cxy
        # 2-channel image, 2nd channel has vasculature
        img = img[1,:,:]
    img = img/(2**12 - 1)
    img = img/np.mean(img)
    img = img/20
    img[img>1] = 1
#     img = (img - np.mean(img))/np.std(img) # z-score
#     img = img/np.mean(img)
#     img = img/np.max(img) # scale [0,1]
    if False :
        # add noise
        img = img + np.random.normal(0,0.1,[img.shape[0],img.shape[0]])
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) # range [0,1]
    return img
