function [brain_noise]= generate_brain_noise(G, N, T, Fs)
 

    Nsrc = size(G,2);
    src_idx = randi(Nsrc, N,1);

    q = randn(T+1000,N);

    alpha_band = [8, 12];
    beta_band = [15, 30];
    gamma1_band = [30, 50];
    gamma2_band = [50, 70];
    theta_band = [4,7];

    
    [b,a] = butter(4,alpha_band/(Fs/2));
    A = filtfilt(b,a,q);
    [b,a] = butter(4,beta_band/(Fs/2));
    B = filtfilt(b,a,q);
    [b,a] = butter(4,gamma1_band/(Fs/2));
    C = filtfilt(b,a,q);
    [b,a] = butter(4,gamma2_band/(Fs/2));
    D = filtfilt(b,a,q);
    [b,a] = butter(4,theta_band/(Fs/2));
    Y = filtfilt(b,a,q);
    
    source_noise = (1/mean(alpha_band))*A + (1/mean(beta_band))*B +...
        (1/mean(gamma1_band))*C + (1/mean(gamma2_band))*D+ (1/mean(theta_band))*Y;
    brain_noise = G(:,src_idx)*source_noise(500-round(T/2)+2:500+round(T/2),:)';

   
end