import logging

logger = logging.getLogger(__name__)



## TODO -- this should warn about poorly separated means
## TODO -- this should warn when not bimodal
## TODO -- this should wran about
def get_state_means(signal):
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(signal.reshape(-1,1))

    means = gmm.means_.ravel()
    stds  = np.sqrt(gmm.covariances_).ravel()

    return means, stds

def compute_threshold(state_means, state_stdevs, mode):
    if mode == 'bayes':
        return compute_bayes_threshold()
    elif mode == 'mean':
        return state_means.sum()/2

from scipy.ndimage import median_filter
def find_edges(signal, filter=True, mode='mean'):
    if filter == True:
        signal = median_filter(signal, size=5, mode='nearest')
        
    state_means, state_stdevs = get_state_means(signal)
    threshold = compute_threshold(state_means, state_stdevs, mode=mode)
    binarized = signal > threshold

    edge_indices = np.where(np.abs(np.diff(binarized))==1)

    return signal, threshold, edge_indices 
    ## TODO -- we only want to return signal if filter == Tue --> maybe we need to separate that out
    ## TODO -- ok we definitely don't want find_edges doing anything other than finding edges. it should NOT do additional signal processing, it should take a processed signal and maybe the threshold. It should only handle binarziation and the np.where

