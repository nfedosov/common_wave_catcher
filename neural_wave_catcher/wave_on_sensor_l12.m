function [sensor_waves, strt, nn] = wave_on_sensor_l12(cortex,  PARAMS, G)

    strt =randi(size(G,2));
    
    vertices = cortex.Vertices;
    faces = cortex.Faces;
    speed = PARAMS.speed;
    duration = PARAMS.duration;
    nspoints = PARAMS.nspoints;
    
    VertConn = cortex.VertConn;
    
    indn = find(VertConn(strt,:)); %indices for the first neighbours
    max_step = 150;  %?????????????????????? WAS 9
    num_dir = length(indn); %number of directions


    
    
    n = randi(num_dir);
   
    nn= n;
        % find the direction vector
     IND(1) = strt; %first vertex
     ind0 = indn(n); %second vertex (for given direction)
     IND(2) = ind0; %second vert into vert array
     norm0 = mean(cortex.VertNormals(indn,:));
     norm0 = norm0/norm(norm0); %mean normal for the set of vertices
     Pnorm0 = eye(3)-norm0'*norm0; %projector out of the norm (on the plane)
     
     
    
     dr0 = (cortex.Vertices(ind0,:) - cortex.Vertices(strt,:));
     dr0 = dr0*Pnorm0';
     dr0 = dr0/norm(dr0);% define direction vector
     
     %define the transversal plane (plane to project on)
     
     third_vector = cross(dr0,norm0);
     third_vector = third_vector/norm(third_vector);
     
     abc = (third_vector*cortex.Vertices(strt,:)')*third_vector;
     
     Ptpon = eye(3) - third_vector'*third_vector;
     
   
     d = 3; %number of the step
        
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
    h = trimesh(faces,vertices(:,1),vertices(:,2),vertices(:,3));
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
       wave((i+1),:) = (cos(2*pi * (n - i) / nspoints));
    end%wave - t (0 ... T) x 
    if PARAMS.draw_wave
     
        figure
        for i = t
            plot(wave(i+1,:))
            hold on
        end
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
    
   
 
    %sensor waves - {dir x speed} (sensors x time)
end





