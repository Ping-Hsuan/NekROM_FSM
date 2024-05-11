clear all; close all;

% This matlab script is to eveolve ROM in time using backward Euler method
% Created by Ping-Hsuan Tsai on Feb 1st 2024

global au0 bu0 cu u0;
global au bu;
global u ua u2a ru eu hufac;

global nb mb;

path = "./";

mb=dlmread(path+"nb");

mu = 1./15000
T_final = 40;
%Dt_list = [0.0001 0.00001 0.000001 0.0000001 0.00000001];
Dt_list = [0.001];
%Dt_list = [0.001];
%Dt_list = [0.0000001 0.00000001];

for k=1:size(Dt_list,2)

    Dt = Dt_list(k);
    nsteps = int64(T_final/Dt);
    iostep = int64(nsteps/2000);
    if (iostep == 0)
        iostep=1;
    end
    err_list = table;
    err_list_h10 = table;
    avg_err_list = table;

%nb_list = [17 18 19 20];
%for i=1:size(nb_list,2) 
%for nb=2:2:40
for nb=45:5:80
%nb = 21 % Specified by User
%nb = nb_list(i);
    
    index  = [1:nb+1];
    index1 = [1:nb];
    index2 = [2:nb+1];
    
    % load stiffness matrix
    a0_full = dlmread(path+"au");
    a0_full = reshape(a0_full,mb+1,mb+1);
    au0     = a0_full(index,index);
    
    % load mass matrix
    b0_full = dlmread(path+"bu");
    b0_full = reshape(b0_full,mb+1,mb+1);
    bu0     = b0_full(index,index);
    
    % load advection tensor
    cu_full = dlmread(path+"cu");
    cu_full = reshape(cu_full,mb,mb+1,mb+1);
    cutmp     = cu_full(index1,index,index);
    cutmp1    = cu_full(index1,index2,index2);
    cu      = reshape(cutmp1,nb*(nb),nb);
    
    % load initial condition
    u0_full = dlmread(path+"u0");
    u0      = u0_full(index);
    
    ns = dlmread(path+"ns");
    
    uk = dlmread(path+"uk");
    uk = reshape(uk,mb+1,ns)';
    
    au = au0(2:end,2:end);
    bu = bu0(2:end,2:end);
    
    u     = zeros(nb+1,3);
    u(:,1)=u0;

    extended_vec = zeros(ns,mb+1);
    
    fprintf("done loading ... \n");
    
    ucoef=zeros(((nsteps/iostep)+1),nb+1);
    ucoef(1,:) = u0;
    u_n = u0(2:nb+1);
    options = optimoptions('fsolve', 'TolFun', 1e-12, 'TolX', 1e-12, 'Algorithm', 'trust-region-reflective');
    
    for istep=1:nsteps
        tmp = fsolve(@(u)reduced_F(u,au0,bu,cu,cutmp,u_n,mu,nb,Dt),u_n,options);
        u_n = tmp;
    
        if (mod(istep,iostep) == 0)
            ucoef((istep/iostep)+1,:) = [1; u_n];
        end
    end
    
    if (nsteps >= 1000)
	    extended_vec(:,1:nb+1) = ucoef;
            err_wproj = extended_vec-uk; 
            err_l2 = diag(err_wproj*b0_full*err_wproj');
            err_l2_avg = sum(err_l2)/ns;
            err_h10 = diag(err_wproj*a0_full*err_wproj');
            err_h10_avg = sum(err_h10)/ns;
            column_names = strcat('N', num2str(nb));
            err_list = [err_list, array2table(err_l2, 'VariableNames', {column_names})];
            err_list_h10 = [err_list_h10, array2table(err_h10, 'VariableNames', {column_names})];
            cn1 = strcat('N', num2str(nb),'_l2');
            cn2 = strcat('N', num2str(nb),'_h10');
	    avg_err_list = [avg_err_list, array2table([err_l2_avg,err_h10_avg], 'VariableNames', {cn1,cn2})];
    end

	dir = sprintf('nsteps%d_dt%d',nsteps,Dt); 
    mkdir(dir);   

	coefname = sprintf("/ucoef_nb%d",nb);
	fileID = fopen(dir+coefname,'w');
	fprintf(fileID,"%24.15e\n",ucoef');
	fclose(fileID);

	coefname = sprintf("/ufinal_nb%d",nb);
	fileID = fopen(dir+coefname,'w');
	fprintf(fileID,"%24.15e\n",ucoef(end,:)');
	fclose(fileID);
end
if (nsteps >= 1000)
        table_name = sprintf('err_l2_square_dt%d_nsteps%d_mu%d.csv', Dt, nsteps,mu);
        writetable(err_list,table_name);
        table_name = sprintf('err_h10_square_dt%d_nsteps%d_mu%d.csv', Dt, nsteps,mu);
        writetable(err_list_h10,table_name);
        table_name = sprintf('avg_err_dt%d_nsteps%d_mu%d.csv', Dt, nsteps,mu);
        writetable(avg_err_list,table_name);
end
end

function G = reduced_F(u,a,b,c,ctmp,u_n,mu,nb,Dt)
    F = - mu * inv(b) * a(2:nb+1,2:nb+1) * u; % Contribution from the stiffness matrix 
    F = F - mu * inv(b) * a(2:nb+1,1); % Contribution from the stiffness matrix 
    F = F - inv(b) * reshape(c*u,nb,nb)*u; % Contribution from the advection tensor
%   F = F + inv(b) * ctmp(:,1,1)*1*1;
    F = F + inv(b) * ctmp(:,1,1)*1*1;
    F = F - inv(b) * reshape(ctmp(:,1,:),nb,nb+1)*[1;u];                                                                                                                                                                   
    F = F - inv(b) * reshape(ctmp(:,:,1),nb,nb+1)*[1;u];
    G = u - Dt * F - u_n;
end
