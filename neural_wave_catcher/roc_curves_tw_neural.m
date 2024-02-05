close all

Fs = 1000;
allow_save = true;



T = 51; %num of samples
Nsim = 50; %Nsim per parameters set % Nsim for wave and Nsim for static
SNR = [5];%[1,2,3,5];


params = struct();
params.duration = T/Fs;
params.sampling_rate = Fs;
params.nspoints = [51];%[11,26,51];%for 0.5 mm/ms - 20 mm
params.speed = [0.5];%[0.2, 0.3, 0.5, 0.8];
params.draw_paths = 0;
params.draw_wave = 1;
params.draw_wos = 1;
params.vicinity = 0.005;

Nstart_areas = 4;




channel_type = 'grad'; % channels you want to analyse ('grad' or 'mag')
  
cortex = load('C:/Users/Fedosov/Documents/projects/waves/tess_cortex_pial_high.mat');


% Initial parameters
if strcmp(channel_type, 'grad') == 1
    channel_idx = setdiff(1:306, 3:3:306);
elseif strcmp(channel_type, 'mag') == 1
    channel_idx = 3:3:306;
end


G3_dir=  'C:/Users/Fedosov/Documents/projects/headmodel_surf_os_meg.mat';
G = gain_orient(G3_dir,channel_idx);


Atlas = cortex.Atlas(5).Scouts;

valid_idc = [];

for i = 1:length(Atlas)
    label =  lower(Atlas(i).Label);
    if ((length(strfind(label,'temp')) > 0)||(length(strfind(label,'front')) > 0))
        valid_idc = [valid_idc, Atlas(i).Vertices];
    end
end

valid_idc = unique(valid_idc);




rng(2)


total_num_sim = Nsim*length(SNR)*length(params.nspoints)*length(params.speed)*Nstart_areas;
waves = zeros(total_num_sim*2,length(channel_idx),T);


parameters_storage = [];
cumsum = 1;
for n = 1:length(SNR)
    
    for ns = 1:length(params.nspoints)
        ns
        for s = 1:length(params.speed)
            
            for start_area = 1: Nstart_areas
            central_strt = valid_idc(randi(length(valid_idc)));
            strt_coord = cortex.Vertices(central_strt, :);
            
            const_dir = [rand()*2-1, rand()*2-1, rand()*2-1];
            const_dir = const_dir/norm(const_dir);
      
            
            indn = find(cortex.VertConn(central_strt,:)); %indices for the first neighbours
            num_dir = length(indn); %number of directions
            norm0 = mean(cortex.VertNormals(indn,:));
            norm0 = norm0/norm(norm0); %mean normal for the set of vertices
     
            
            third_vector = cross(const_dir,norm0);
            third_vector = third_vector/norm(third_vector);

            dr0 = cross(norm0,third_vector);
            dr0 = dr0/norm(dr0);

            
            for i = 1:Nsim

                
                [wave_raw, strt, third_vector2, nspoints, speed]= wave_on_sensor(cortex,params, G,strt_coord,params.speed(s),...
                    params.nspoints(ns), third_vector, dr0);
                [blob_raw, strt]= blob_on_sensor(cortex,params, G,strt,speed,...
                    nspoints, third_vector2, dr0);
                
               
                wave_norm = wave_raw/norm(wave_raw);
                blob_norm = blob_raw/norm(blob_raw);
             
                noise = generate_brain_noise(G, 1000, T, Fs);
                noise_norm = noise/norm(noise);
                waves(cumsum,:,:) = SNR(n)*wave_norm +noise_norm;
                waves(total_num_sim+cumsum,:,:) = SNR(n)*blob_norm +noise_norm;
         
              
                parameters_storage = [parameters_storage, [SNR(n), params.nspoints(ns), params.speed(s), strt, dr0]'];
                
                cumsum = cumsum + 1;
            end
            end
        end
    end
end

if allow_save
    save('C:/Users/Fedosov/Documents/projects/waves/PDE_data/simulated_spikes.mat','waves','parameters_storage')
end


