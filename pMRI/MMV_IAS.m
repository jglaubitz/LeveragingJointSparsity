%% MMV_IAS
%
% Description: 
%  Function for the MMV-IAS algorithm 
%  We assume iid noise
%
% INPUT: 
%  J :          number of MMVs 
%  F_fun :      list of forward operators
%  y :          list of vectorized MMVs  
%  variance :   list of noise variances 
%  R_fun :      regularization operator (function)
%  c, d :       regualrization hyper-hyper-parameters 
%
% OUTPUT: 
%  x :          list of vectorized MAP estimates 
%  theta :     	vector of MAP estimate for theta 
%  history :    structure containing residual norms and the tolerances at each iteration
%
% Author: Jan Glaubitz 
% Date: Feb 01, 2023 
% 

function [x, theta, history] = MMV_IAS( J, F_fun, y, noise_var, R_fun, r, beta, vartheta, QUIET )

    t_start = tic; % measure time 

    %% Global constants and defaults  
    MAX_ITER = 1000; 
    TOL_x = 1e-4;
    
    %% Data preprocessing & initial values 
    parfor j=1:J 
        M{j} = length(y{j}); % number of measurements 
        Fty{j} = F_fun{j}(y{j},'transp'); % forward operator applied to the data 
        N = length( Fty{j} ); % number of parameters 
        x{j} = zeros(N,1); x_OLD{j} = zeros(N,1); % initialize parameter vectors 
        FtF{j} = @(x) F_fun{j}( F_fun{j}(x,'notransp'), 'transp' ); % product corresponding to the forward operator 
    end
    K = length( R_fun(x_OLD{1},'notransp') ); % number of outputs of the regularization operator 
    theta = ones(K,1); % initial value for theta
   
    %% Outputting the learning progress
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\n', 'iter', 'max. rel. change in x', 'tol x');
    end
    
    %% Iterate between the update steps until convergence of max number of iterations 
    for counter = 1:MAX_ITER

        % 1) x-updates 
        parfor j=1:J
            A_fun{j} = @(x) (1/noise_var{j})*FtF{j}(x) + R_fun( (theta.^(-1)).*R_fun(x,'notransp'), 'transp'); % forward operator 
            [x{j},flag] = pcg( A_fun{j}, (1/noise_var{j})*Fty{j},[],[],[],[],x_OLD{j});
        end

        % 2) theta-update 
        % auxilary value 
        aux = 0;
        for j=1:J 
            aux = aux + real( R_fun(x{j},'notransp') ).^2; 
        end
        % actual update 
        eta = r*beta - (J/2 + 1); % auxilary parameter eta
        if r==1 
            theta = 0.5*vartheta*( eta + sqrt( eta^2 + 2*aux/vartheta ) ); 
        elseif r==-1 
            theta = ( aux/2 + vartheta )/( -eta ); 
        else 
            error('Only r=1 and r=-1 are implemented yet!'); 
        end

        % store certain values in history structure 
        history.change_x(counter) = 0; 
        change_x = zeros(J,1); 
        parfor j=1:J 
            change_x(j) = norm( x{j} - x_OLD{j} )/norm( x_OLD{j} ); % relative change in x{j}
            x_OLD{j} = x{j}; % store value of mu 
        end
        max_change = max(change_x); % maximum change 
        history.change_x(counter) = max_change; % sum of relative changes

        % display these values if desired 
        if ~QUIET
            fprintf('%3d\t%0.2e\t%0.2e\n', counter, history.change_x(counter), TOL_x);
        end
        
        % check for convergence 
        if ( history.change_x(counter) < TOL_x ) 
             break;
        end
        
    end

    % output the time it took to perform all operations 
    if ~QUIET
        toc(t_start);
    end
    
end