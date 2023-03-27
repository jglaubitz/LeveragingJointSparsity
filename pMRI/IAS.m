%% IAS
%
% Description: 
%  Function for the IAS algorithm.  
%  We assume iid noise
%
% INPUT: 
%  F_fun :              forward operator (function)
%  y :                  measurement vector 
%  variance :           fixed noise variances 
%  R_fun :              regularization operator (function)
%  r, beta, vartheta :  regualrization hyper-hyper-parameters 
%  x_init :             initial guess 
%
% OUTPUT: 
%  x :          MAP estimate for x 
%  theta :     	MAP estimate for theta 
%  history :    structure containing residual norms and the tolerances at each iteration
%
% Author: Jan Glaubitz 
% Date: Feb 01, 2023 
% 

function [x, theta, history] = IAS( F_fun, y, variance, R_fun, r, beta, vartheta, QUIET )

    t_start = tic; % measure time 

    %% Global constants and defaults   
    MAX_ITER = 1000; 
    TOL_x = 1e-4;

    %% Data preprocessing & initial values 
    M = length(y); % number of measurements 
    Fty = F_fun(y,'transp'); % forward operator applied to the indirect data  
    N = length( Fty ); % number of parameters
    K = length( R_fun(Fty,'notransp') ); % number of outputs of the regularization operator 
    FtF = @(x) F_fun( F_fun(x,'notransp'), 'transp' ); % product corresponding to the forward operator  
    theta = ones(K,1); % initial value for theta
    x_OLD = zeros(N,1); % auxilary variable to compute change in x
    
    %% Outputting the learning progress
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\n', 'iter', 'rel. change in x', 'tol x');
    end
    
    %% Iterate between the update steps until convergence of max number of iterations 
    for counter = 1:MAX_ITER

        % 1) x-updates 
        A_fun = @(x) (1/variance)*FtF(x) + R_fun( ( theta.^(-1) ).*R_fun(x,'notransp'), 'transp');
        [x,flag] = pcg( A_fun, (1/variance)*Fty ,[],[],[],[], x_OLD); 

        % 2) theta-update 
        eta = r*beta - (1/2 + 1); % auxilary parameter eta
        if r==1 
            theta = 0.5*vartheta*( eta + sqrt( eta^2 + 2*real( R_fun(x,'notransp') ).^2/vartheta ) ); 
        elseif r==-1 
            theta = ( real( R_fun(x,'notransp') ).^2/2 + vartheta )/( -eta ); 
        else 
            error('Only r=1 and r=-1 are implemented yet!'); 
        end

        % store certain values in history structure 
        history.change_x(counter) = norm( x - x_OLD )/norm( x_OLD ); % absolute error 
        x_OLD = x; 

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