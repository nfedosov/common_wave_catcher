function [sensor_waves, strt, third_vector, nspoints, speed] = wave_on_sensor(cortex,  PARAMS, G, strt_coord,speed,nspoints, third_vector, dr0)
    err_third_vector = 0.1;
    err_speed = 0.1;
    err_nspoints = 0.1;
    
    speed = speed + speed*(rand()*2-1)*err_speed;
    
    nspoints = nspoints + nspoints*round((rand()*2-1)*err_nspoints);
    third_vector = third_vector+[rand()*2-1, rand()*2-1, rand()*2-1]*err_third_vector;  
    third_vector = third_vector/norm(third_vector);
    Ptpon = eye(3) - third_vector'*third_vector;

    vertices = cortex.Vertices;
 
    area_idc = find(sqrt(sum((strt_coord - vertices).^2,2)) < PARAMS.vicinity);
    strt =area_idc(randi(length(area_idc)));
    
    
    faces = cortex.Faces;

    duration = PARAMS.duration;
    
    VertConn = cortex.VertConn;
    
    indn = find(VertConn(strt,:)); %indices for the first neighbours
    max_step = 150;  %?????????????????????? WAS 9
    num_dir = length(indn); %number of directions


    
   
        % find the direction vector
     IND(1) = strt; %first vertex
     ind0 = strt;
   
     
     abc = (third_vector*cortex.Vertices(strt,:)')*third_vector;
     

     d = 2; %number of the step
        
     while(d <= max_step)
     indn1 = find(VertConn(ind0,:));%new neighbours
        clear cs
        for n1 = 1:length(indn1)
            dr1 = (cortex.Vertices(indn1(n1),:) - cortex.Vertices(ind0,:)); %next minus prev
            dr1 = dr1*Ptpon';
            dr1 = dr1/norm(dr1);
            cs(n1) = dr1*dr0';
        end
     
            cs_positive_ind = find(cs>0);
            if ~isempty(cs_positive_ind)
                
                clear tang
                projected_ind0 = abc+cortex.Vertices(ind0,:)*Ptpon;
           
                for n1 = 1:length(cs_positive_ind)
                    projected_next = abc + cortex.Vertices(indn1(cs_positive_ind(n1)),:)*Ptpon;%%%%%%%%
                    
                    tang(n1) = norm(projected_next-cortex.Vertices(indn1(cs_positive_ind(n1)),:))/...
                               norm(projected_next-projected_ind0);
                end
                [csminval,csminind] = min(tang);%neighbour with max incremention
                
                dr0 = (cortex.Vertices(indn1(cs_positive_ind(csminind)),:) - cortex.Vertices(ind0,:)); 
                dr0 = dr0/norm(dr0);
                
                ind0 = indn1(cs_positive_ind(csminind));
                IND(d) = ind0;

             

            else
                
                [csmaxval,csmaxind] = max(cs);%neighbour with max incremention
                dr0 = (cortex.Vertices(indn1(csmaxind),:) - cortex.Vertices(ind0,:)); 
                dr0 = dr0/norm(dr0);
                ind0 = indn1(csmaxind);
                IND(d) = ind0;
            end
      
            
            d = d+1;
      end
        
    DIST = zeros(1,(max_step-1));
    for d = 1:num_dir
        for i = 2:max_step
            DIST(i-1) = norm(vertices(IND(i),:)-vertices(IND(i-1),:));
        end
    end
        
    SR = PARAMS.sampling_rate;
    ntpoints = round(SR*duration);%number of points in time for wave
    PATH = zeros(nspoints,3);
 
    FM = zeros(size(G,1),nspoints);
    tstep = (1/SR);
    
   
    
    l = speed*tstep;

          PATH(1,:) = vertices(strt,:); %path - direct x num of step x speed
            
            FM(:,1) = G(:,strt); %corresponding forward coef
            res = 0;
            v1 = 1;%prev vertex
            v2 = 2;%next vertex
            for t = 2:nspoints
%                 if (strt == 218301) && (s == 5) && (d == 4) && (t ==5)
%                     bbb = 1;
%                 end
%                 if (alpha > 1) || (alpha < 0)
%                     aaa = 1;
%                 end
                if l < res
                   alpha = 1-l/res;
                   PATH(t,:) = alpha*PATH((t-1),:)+(1-alpha)*vertices(IND(v2),:);
                  
                   FM(:,t) = alpha*FM(:,(t-1))+ (1-alpha)*G(:,IND(v2));
                   res = res-l;
                elseif l > res
                    if res == 0
                        if l < DIST(v2-1)
                            alpha = 1-l/DIST(v2-1);
                            PATH(t,:) = alpha*vertices(IND(v1),:)+(1-alpha)*vertices(IND(v2),:);
                           
                            FM(:,t) = alpha*G(:,IND(v1)) + (1-alpha)*G(:,IND(v2));
                            res = DIST(v2-1)-l;
                        elseif l == DIST(v2-1)
                            PATH(t,:)= vertices(IND(v2),:);
                           
                            FM(:,t) = G(:, IND(v2));
                            v1 = v1 + 1;
                            v2 = v2 + 1;
                        else
                            l2 = l-DIST(v2-1);
                            v1 = v1 + 1;
                            v2 = v2 + 1;
                            res = DIST(v2-1)-l2;
                            while res < 0
                                l2 = -res;
                                v1 = v1 + 1;
                                v2 = v2 + 1;
                                res = DIST(v2-1)-l2;
                            end
                            alpha = 1-l2/DIST(v2-1);
                            PATH(t,:) = alpha*vertices(IND(v1),:)+(1-alpha)*vertices(IND(v2),:);
                            
                            FM(:,t) = alpha*G(:, IND(v1)) + (1-alpha)*G(:,IND(v2));
                            
                        end
                    else
         
                        l2 = l-res;
                        v1 = v1 + 1;
                        v2 = v2 + 1;
                        res = DIST(v2-1)-l2;
                        
                        while res < 0
                            l2 = -res;
                            v1 = v1 + 1;
                            v2 = v2 + 1;
                            res = DIST(v2-1)-l2;
                        end
                        alpha = 1-l2/DIST(v2-1);
                        PATH(t,:) = alpha*vertices(IND(v1),:)+(1-alpha)*vertices(IND(v2),:);
                       
                        FM(:,t) = alpha*G(:,IND(v1))+ (1-alpha)*G(:,IND(v2));
                              
                    end
                else %l == res
                    PATH(t,:) = vertices(IND(v2),:);
                    
                    FM(:,t) = G(:,IND(v2));
                    v1 = v1 + 1;
                    v2 = v2 + 1;
                end
            end

            
   close all
%h = trimesh(faces,vertices(:,1),vertices(:,2),vertices(:,3));   
if PARAMS.draw_paths == 1

    
    figure
    %h = trimesh(faces,vertices(:,1),vertices(:,2),vertices(:,3));
    %set(h,'FaceAlpha',0.5);
   
    hold on
    for i = 1:num_dir
        hplot = plot3(vertices(IND,1), vertices(IND,2), vertices(IND,3),'ko','MarkerFaceColor','r')
    end%plot vertices
    for j = 1:num_dir
        for i = 1:nspoints
            hplot = plot3(PATH(:,1), PATH(:,2), PATH(:,3),'ko','MarkerFaceColor','g')
        end%plot path
    end
    hplot = plot3(vertices(strt,1),vertices(strt,2),vertices(strt,3),'ko','MarkerFaceColor','b')
end
 
%     
% %%%%%%%%%%%
% 

% 
    t = 0:(ntpoints-1);
    n = 0:nspoints-1;
    
    
    
    for i = t
       if nspoints == 51
           wave((i+1),:) = exp(-((n-i)*1.8/51).^2);
       else
           wave((i+1),:) = exp(-((n+26*51/51-i)*1.8/51).^2);%sin(2*pi*(n+i-51)/51).*exp(-((n-20-51+i)/20).^2);
       end
    end%wave - t (0 ... T) x 
    if PARAMS.draw_wave
     
        figure
        lgnd = {};
        count = 1;
        for i = [0:10:t(end)-1]%[0,round(t(end)/2),t(end)-1]
            plot(wave(i+1,:))
            hold on
            lgnd{count,1} = ['sample point = ', int2str(i)];
            count = count+1;
        end
        
        xlabel('x, points')
        
        
        legend(lgnd)
    end
    

%     figure
%     for i = 1:6
%     plot(t, wave(i,:))
%     hold on
%     end
       
    % waves on sensors  

            for t = 1:ntpoints
                for k = 1:nspoints
                    FM_s(:,k) = FM(:,k);
                end
                sensor_waves(:,t) = FM_s*wave(t,:)';
            end
    
    if PARAMS.draw_wos
        figure()
        plot(sensor_waves')
    end
    %sensor waves - {dir x speed} (sensors x time)
end





