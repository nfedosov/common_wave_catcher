from scipy.linalg import pinv, block_diag
import numpy as np
from sklearn.decomposition import PCA

Fs = 1e3
N_comps = 2
Nhigh = 3
Nlow = 1
DoWhitening = False
pre_time = 20
post_time = 30
T = pre_time + post_time + 1
Nsim = 200
SNR = 3
rnd_state=0

def Detect_waves(waves,Nlow=Nlow,Nhigh=Nhigh,N_comps=N_comps,DoWhitening=DoWhitening):
    F_storage = []

    def apply_pca_to_2d_array(array_2d,DoWhitening=DoWhitening):
            pca = PCA(n_components=N_comps,random_state=rnd_state,whiten=DoWhitening)
            transformed_array = pca.fit_transform(array_2d)
            return transformed_array

    waves_pca = np.stack([apply_pca_to_2d_array(arr) for arr in (waves.swapaxes(2,1))])



    row_indices, col_indices = np.triu_indices(N_comps)
    Hsym = np.zeros((N_comps**2, (N_comps**2 - N_comps) // 2 + N_comps))
    Hsym[row_indices * N_comps + col_indices, np.arange(len(row_indices))] = 1
    Hsym[col_indices * N_comps + row_indices, np.arange(len(col_indices))] = 1


    Mask = np.zeros(waves_pca.shape[1])
    Mask[Nlow:Nhigh + 1] = 1
    Mask[Mask.shape[0]-Nhigh:Mask.shape[0]-Nlow + 1] = 1

    for k in range(waves_pca.shape[0]):
        XF = np.fft.fft(waves_pca[k], axis=0) * (Mask[:, np.newaxis])

        j = np.complex(0, 1)
        kk = np.arange(waves_pca.shape[1])

        XFW = XF * np.exp(-j * 2 * np.pi * kk / waves_pca.shape[1])[:, np.newaxis]
        dXF = XF - XFW

        XFWblk = block_diag(*[XFW] * N_comps)

        mopt_ful_fft = pinv(XFWblk.T @ XFWblk) @ XFWblk.T @ dXF.flatten()
        mopt_sym_fft = pinv((XFWblk @ Hsym).T @ (XFWblk @ Hsym)) @ (XFWblk @ Hsym).T @ dXF.flatten()

        Chi2_ful_fft = np.linalg.norm(XFWblk @ mopt_ful_fft - dXF.flatten())
        Chi2_sym_fft = np.linalg.norm(XFWblk @ Hsym @ mopt_sym_fft - dXF.flatten())

        F_fft = (Chi2_sym_fft**2 - Chi2_ful_fft**2) / Chi2_ful_fft**2

        F_storage.append(F_fft)

    return np.stack(F_storage)
