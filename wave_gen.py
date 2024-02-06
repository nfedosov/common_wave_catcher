import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

import scipy.sparse as sp
from scipy.signal import butter, filtfilt


import warnings

rng = np.random.default_rng(seed=0) # random Generator with seed

params = {
    'duration': T / Fs,
    'sampling_rate': Fs,
    'nspoints': 51,
    'speed': 0.8,
    'draw_paths': False,
    'draw_wave': False,
}

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
channel_type = 'grad'
if channel_type == 'grad':
    channel_idx = np.setdiff1d(np.arange(0, 305), np.arange(2, 305, 3))
elif channel_type == 'mag':
    channel_idx = np.arange(2, 305, 3)




def gain_orient(G3, channel_idx):
    Gain = G3['Gain'][channel_idx, :]
    # Create a sparse block diagonal matrix for orientations
    GridOrient = sp.block_diag(np.hsplit(G3['GridOrient'].T,G3['GridOrient'].shape[0]))
    # Apply the orientation to the Gain matrix
    Gain = Gain @ GridOrient
    return Gain

def wave_on_sensor(cortex, PARAMS, G):
    strt = rng.integers(0, G.shape[1])

    vertices = cortex['Vertices']
    VertConn = cortex['VertConn']
    speed = PARAMS['speed']
    duration = PARAMS['duration']
    nspoints = PARAMS['nspoints']
    SR = PARAMS['sampling_rate']

    indn = np.nonzero(VertConn[strt])[1]
    max_step = 151
    num_dir = len(indn)

    nn = rng.integers(num_dir) ###########  0

    IND = np.zeros(max_step, dtype=int)

    IND[0] = strt
    ind0 = indn[nn]
    IND[1] = ind0
    norm0 = np.mean(cortex['VertNormals'][indn], axis=0)
    norm0 /= np.linalg.norm(norm0)
    Pnorm0 = np.eye(3) - np.outer(norm0, norm0)
    dr0 = (vertices[ind0] - vertices[strt]) @ Pnorm0.T
    dr0 /= np.linalg.norm(dr0)

    third_vector = np.cross(dr0, norm0)
    third_vector /= np.linalg.norm(third_vector)

    abc = np.dot(np.outer(third_vector, vertices[strt]),third_vector)  #######
    Ptpon = np.eye(3) - np.outer(third_vector, third_vector)
    d = 3

    while d <= max_step:
        indn1 = np.nonzero(VertConn[ind0])[1]
        cs = np.zeros(len(indn1))
        for n1 in range(len(indn1)):
            dr1 = vertices[indn1[n1]] - vertices[ind0]  # next minus prev
            dr1 = np.dot(dr1, Ptpon.T)
            dr1 = dr1 / np.linalg.norm(dr1)
            cs[n1] = np.dot(dr1, dr0.T)

        cs_positive_ind = np.where(cs > 0)[0]
        if len(cs_positive_ind) > 0:
            tang = np.zeros(len(cs_positive_ind))
            projected_ind0 = abc + np.dot(vertices[ind0], Ptpon)
            for n1 in range(len(cs_positive_ind)):
                projected_next = abc + np.matmul(vertices[indn1[cs_positive_ind[n1]]], Ptpon)
                tang[n1] = np.linalg.norm(projected_next - vertices[indn1[cs_positive_ind[n1]]]) / \
                            np.linalg.norm(projected_next - projected_ind0)

            csminind = np.argmin(tang)

            dr0 = (vertices[indn1[cs_positive_ind[csminind]]] - vertices[ind0])
            dr0 = dr0 / np.linalg.norm(dr0)
            ind0 = indn1[cs_positive_ind[csminind]]
            IND[d-1] = ind0
        else:
            csmaxind = np.argmax(cs)
            dr0 = (vertices[indn1[csmaxind]] - vertices[ind0])
            dr0 = dr0 / np.linalg.norm(dr0)
            ind0 = indn1[csmaxind]
            IND[d-1] = ind0

        d += 1

    DIST = np.zeros(max_step - 1)

    IND_shifted = np.roll(IND, shift=-1, axis=0)
    DIST = np.linalg.norm(vertices[IND_shifted] - vertices[IND], axis=1)[:-1]


    ntpoints = round(SR * duration)
    PATH = np.zeros((nspoints, 3))

    FM = np.zeros((G.shape[0], nspoints))
    tstep = 1 / SR
    l = speed * tstep
    PATH[0] = vertices[strt]
    FM[:, 0] = G[:, strt]
    res = 0
    v1 = 0
    v2 = 1
    for t in range(1, nspoints):

        if l < res:
            alpha = 1 - l / res
            PATH[t] = alpha * PATH[t - 1, :] + (1 - alpha) * vertices[IND[v2], :]
            FM[:, t] = alpha * FM[:, t - 1] + (1 - alpha) * G[:, IND[v2]]
            res = res - l

        elif l > res:
            if res == 0:
                if l < DIST[v2 - 1]:
                    alpha = 1 - l / DIST[v2 - 1]
                    PATH[t, :] = alpha * vertices[IND[v1], :] + (1 - alpha) * vertices[IND[v2], :]
                    FM[:, t] = alpha * G[:, IND[v1]] + (1 - alpha) * G[:, IND[v2]]
                    res = DIST[v2 - 1] - l

                elif l == DIST[v2 - 1]:
                    PATH[t, :] = vertices[IND[v2], :]
                    FM[:, t] = G[:, IND[v2]]
                    v1 += 1
                    v2 += 1

                else:
                    l2 = l - DIST[v2 - 1]
                    v1 += 1
                    v2 += 1
                    res = DIST[v2 - 1] - l2
                    while res < 0:
                        l2 = -res
                        v1 += 1
                        v2 += 1
                        res = DIST[v2 - 1] - l2
                    alpha = 1 - l2 / DIST[v2 - 1]
                    PATH[t, :] = alpha * vertices[IND[v1], :] + (1 - alpha) * vertices[IND[v2], :]
                    FM[:, t] = alpha * G[:, IND[v1]] + (1 - alpha) * G[:, IND[v2]]

            else:
                l2 = l - res
                v1 += 1
                v2 += 1
                res = DIST[v2 - 1] - l2

                while res < 0:
                    l2 = -res
                    v1 += 1
                    v2 += 1
                    res = DIST[v2 - 1] - l2
                alpha = 1 - l2 / DIST[v2 - 1]
                PATH[t, :] = alpha * vertices[IND[v1]] + (1 - alpha) * vertices[IND[v2]]
                FM[:, t] = alpha * G[:, IND[v1]] + (1 - alpha) * G[:, IND[v2]]

        else:  # l == res
            PATH[t, :] = vertices[IND[v2], :]
            FM[:, t] = G[:, IND[v2]]
            v1 += 1
            v2 += 1


    if PARAMS['draw_paths'] == 1:
        fig = px.scatter_3d(vertices, x=0, y=1, z=2, width=800, height=600,opacity=0.00099) #to improve speed performance - reduce number of vertices

        # Add scatter plot for 'Vertices'
        fig.add_trace(go.Scatter3d(x=vertices[IND, 0], y=vertices[IND, 1], z=vertices[IND, 2], mode='markers', name='Vertices',opacity=0.8))

        # Add scatter plot for 'PATHWAY'
        fig.add_trace(go.Scatter3d(x=PATH[:, 0], y=PATH[:, 1], z=PATH[:, 2], mode='markers', name='PATHWAY',marker=dict(color=PATH[:, 2], colorscale='tealgrn')))
        fig.show()


    wave = np.zeros((ntpoints, nspoints))
    t = np.arange(ntpoints)[:, np.newaxis]
    n = np.arange(nspoints)[np.newaxis, :]
    wave = np.cos(2 * np.pi * (n - t) / nspoints)


    if  PARAMS['draw_wave']:
        plt.figure()
        plt.plot(np.arange(wave.shape[1]),wave[:,:])

    sensor_waves = np.tensordot(FM,wave.T,axes=1)

    return sensor_waves, strt, nn


def blob_on_sensor(cortex, PARAMS, G, strt, nn):
    vertices = cortex['Vertices']
    VertConn = cortex['VertConn']
    speed = PARAMS['speed']
    duration = PARAMS['duration']
    nspoints = PARAMS['nspoints']
    SR = PARAMS['sampling_rate']

    indn = np.nonzero(VertConn[strt])[1]
    max_step = 100
    num_dir = len(indn)

    IND = np.zeros(max_step, dtype=int)
    IND[0] = strt
    ind0 = indn[nn]
    IND[1] = ind0
    norm0 = np.mean(cortex['VertNormals'][indn], axis=0)
    norm0 /= np.linalg.norm(norm0)
    Pnorm0 = np.eye(3) - np.outer(norm0, norm0)

    dr0 = (vertices[ind0] - vertices[strt]) @ Pnorm0.T
    dr0 /= np.linalg.norm(dr0)

    third_vector = np.cross(dr0, norm0)
    third_vector /= np.linalg.norm(third_vector)
    abc = np.dot(np.outer(third_vector, vertices[strt]), third_vector)  #######
    Ptpon = np.eye(3) - np.outer(third_vector, third_vector)

    d = 3

    while d <= max_step:
        indn1 = np.nonzero(VertConn[ind0])[1]
        cs = np.zeros(len(indn1))

        for n1 in range(len(indn1)):
            dr1 = (vertices[indn1[n1]] - vertices[ind0]) @ Ptpon.T
            dr1 /= np.linalg.norm(dr1)
            cs[n1] = np.dot(dr1, dr0.T)

        cs_positive_ind = np.where(cs > 0)[0]

        if len(cs_positive_ind) > 0:
            tang = np.zeros(len(cs_positive_ind))
            projected_ind0 = abc + np.dot(vertices[ind0, :], Ptpon)
            for n1 in range(len(cs_positive_ind)):
                projected_next = abc + np.dot(vertices[indn1[cs_positive_ind[n1]], :], Ptpon)
                tang[n1] = np.linalg.norm(projected_next - vertices[indn1[cs_positive_ind[n1]], :]) / \
                           np.linalg.norm(projected_next - projected_ind0)

            csminind = np.argmin(tang)
            dr0 = (vertices[indn1[cs_positive_ind[csminind]], :] - vertices[ind0, :])
            dr0 /= np.linalg.norm(dr0)
            ind0 = indn1[cs_positive_ind[csminind]]
            IND[d - 1] = ind0
        else:
            csmaxind = np.argmax(cs)
            dr0 = (vertices[indn1[csmaxind]] - vertices[ind0])
            dr0 /= np.linalg.norm(dr0)
            ind0 = indn1[csmaxind]
            IND[d - 1] = ind0

        d += 1

    DIST = np.linalg.norm(vertices[IND[1:]] - vertices[IND[:-1]], axis=1)
    IND_shifted = np.roll(IND, shift=-1, axis=0)
    DIST = np.linalg.norm(vertices[IND_shifted] - vertices[IND], axis=1)[:-1]

    ntpoints = round(SR * duration)
    PATH = np.zeros((nspoints, 3))

    FM = np.zeros((G.shape[0], nspoints))
    tstep = 1 / SR

    l = speed * tstep

    PATH[0, :] = vertices[strt, :]
    FM[:, 0] = G[:, strt]
    res = 0
    v1 = 0
    v2 = 1

    for t in range(1, nspoints):
        if l < res:
            alpha = 1 - l / res
            PATH[t] = alpha * PATH[t - 1, :] + (1 - alpha) * vertices[IND[v2], :]
            FM[:, t] = alpha * FM[:, t - 1] + (1 - alpha) * G[:, IND[v2]]
            res = res - l
        elif l > res:
            if res == 0:
                if l < DIST[v2 - 1]:
                    alpha = 1 - l / DIST[v2 - 1]
                    PATH[t, :] = alpha * vertices[IND[v1], :] + (1 - alpha) * vertices[IND[v2], :]
                    FM[:, t] = alpha * G[:, IND[v1]] + (1 - alpha) * G[:, IND[v2]]
                    res = DIST[v2 - 1] - l
                elif l == DIST[v2 - 1]:
                    PATH[t, :] = vertices[IND[v2], :]
                    FM[:, t] = G[:, IND[v2]]
                    v1 += 1
                    v2 += 1
                else:
                    l2 = l - DIST[v2 - 1]
                    v1 += 1
                    v2 += 1
                    res = DIST[v2 - 1] - l2
                    while res < 0:
                        l2 = -res
                        v1 += 1
                        v2 += 1
                        res = DIST[v2 - 1] - l2
                    alpha = 1 - l2 / DIST[v2 - 1]
                    PATH[t, :] = alpha * vertices[IND[v1], :] + (1 - alpha) * vertices[IND[v2], :]
                    FM[:, t] = alpha * G[:, IND[v1]] + (1 - alpha) * G[:, IND[v2]]
            else:
                l2 = l - res
                v1 += 1
                v2 += 1
                res = DIST[v2 - 1] - l2

                while res < 0:
                    l2 = -res
                    v1 += 1
                    v2 += 1
                    res = DIST[v2 - 1] - l2
                alpha = 1 - l2 / DIST[v2 - 1]
                PATH[t, :] = alpha * vertices[IND[v1]] + (1 - alpha) * vertices[IND[v2]]
                FM[:, t] = alpha * G[:, IND[v1]] + (1 - alpha) * G[:, IND[v2]]
        else:  # l == res
            PATH[t, :] = vertices[IND[v2], :]
            FM[:, t] = G[:, IND[v2]]
            v1 += 1
            v2 += 1

    if PARAMS['draw_paths'] == 1:
        fig = px.scatter_3d(vertices, x=0, y=1, z=2, width=800, height=600,
                            opacity=0.00099)  # to improve speed performance - reduce number of vertices

        # Add scatter plot for 'Vertices'
        fig.add_trace(
            go.Scatter3d(x=vertices[IND, 0], y=vertices[IND, 1], z=vertices[IND, 2], mode='markers', name='Vertices',
                         opacity=0.8))

        # Add scatter plot for 'PATHWAY'
        fig.add_trace(go.Scatter3d(x=PATH[:, 0], y=PATH[:, 1], z=PATH[:, 2], mode='markers', name='PATHWAY',
                                   marker=dict(color=PATH[:, 2], colorscale='tealgrn')))
        fig.show()

    wave = np.cos(2 * np.pi * np.arange(nspoints)[:, None] / nspoints) * np.cos(2 * np.pi * np.arange(ntpoints) / ntpoints)
    sensor_waves = np.tensordot(FM, wave.T, axes=1)

    if PARAMS['draw_wave']:
        plt.figure()
        plt.plot(np.arange(wave.shape[1]), wave[:, :])

    return sensor_waves

def generate_brain_noise(G, N, T, Fs):
    """
    Generate brain noise based on the given forward matrix G.

    Parameters:
        G (numpy.ndarray): Forward matrix.
        N (int): Number of sources.
        T (int): Duration of the brain noise in samples.
        Fs (float): Sampling frequency.

    Returns:
        numpy.ndarray: Generated brain noise.
    """

    Nsrc = G.shape[1]
    src_idx = rng.integers(0, Nsrc, N)

    q = rng.random((T + 1000, N))

    alpha_band = [8, 12]
    beta_band = [15, 30]
    gamma1_band = [30, 50]
    gamma2_band = [50, 70]
    theta_band = [4, 7]

    def apply_band_filter(data, band, fs):
        b, a = butter(4, np.array(band)/(fs / 2), btype='band')
        return filtfilt(b, a, data, axis=0)

    A = apply_band_filter(q, alpha_band, Fs)
    B = apply_band_filter(q, beta_band, Fs)
    C = apply_band_filter(q, gamma1_band, Fs)
    D = apply_band_filter(q, gamma2_band, Fs)
    Y = apply_band_filter(q, theta_band, Fs)

    source_noise = (
        (1 / np.mean(alpha_band)) * A
        + (1 / np.mean(beta_band)) * B
        + (1 / np.mean(gamma1_band)) * C
        + (1 / np.mean(gamma2_band)) * D
        + (1 / np.mean(theta_band)) * Y
    )

    brain_noise = G[:, src_idx] @ source_noise[500 - round(T / 2) + 1 : 500 + round(T / 2), :].T

    return brain_noise

def generate_wave(mat2, params, G,Nsim):
    warnings.filterwarnings("ignore")
    waves = np.zeros((Nsim * 2, channel_idx.shape[0], T))
    for i in tqdm(range(Nsim),desc='Synthesizing data'):
        wave, strt, nn = wave_on_sensor(mat2, params, G)
        blob = blob_on_sensor(mat2, params, G, strt, nn)
        noise = generate_brain_noise(G,1000,T,Fs)
        noise_norm = noise/np.linalg.norm(noise)

        waves[i] = SNR*(wave/np.linalg.norm(wave)) + noise_norm
        waves[i+Nsim] = SNR*(blob/np.linalg.norm(blob)) + noise_norm
    return waves


