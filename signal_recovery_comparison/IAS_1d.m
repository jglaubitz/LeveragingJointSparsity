%% IAS_1d
%
% Description: 
%  Function for the IAS algorithm.  
%  We assume a one-dimensional signal and iid noise
%
% INPUT: 
%  F :                  forward operator 
%  y :                  measurement vector 
%  variance :           fixed noise variances 
%  R :                  regularization matrix
%  r, beta, vartheta :  regualrization hyper-hyper-parameters 
%  x_init :             initial guess 
%
% OUTPUT: 
%  x :          MAP estimate for x 
%  theta :     	MAP estimate for theta 
%  history :    structure containing residual norms and the tolerances at each iteration
%
% Author: Jan Glaubitz 
% Date: Dec 23, 2022 
% 

function [x, theta, history] = IAS_1d( F, y, variance, R, r, beta, vartheta, QUIET )

    t_start = tic; % measure time 

    %% Global constants and defaults  
    %MIN_ITER = 10; 
    MAX_ITER = 1000; 
    TOL_x = 1e-6;
    TOL_G = 1e-6;
    
    %% Data preprocessing 
    M = size(F,1); % number of measurements 
    N = size(F,2); % number of parameters
    K = size(R,1); % number of outputs of the regularization operator 
    FtF = F'*F; % product corresponding to the forward operator 
    Fty = F'*y; % forward operator applied to the indirect data  
    G = @(x,theta) (1/variance)*norm( F*x - y ).^2/2 + ... 
        norm( diag(theta.^(-1/2))*R*x ).^2/2 + ... 
        sum( (theta./vartheta).^r ) + ... 
        ( 1/2 + 1 - r*beta )*sum( log(theta) ); 

    %% Initial values for the inverse variances and the mean  
    alpha = 1/variance; % go over from variance to precision 
    theta = ones(K,1); % initial value for beta
    theta_OLD = ones(K,1); % auxilary variable to compute change in beta 
    x_OLD = zeros(N,1); % auxilary variable to compute change in x
    
    %% Outputting the learning progress
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', ... 
            'iter', 'change in x', 'tol x', 'change in G', 'tol G');
    end
    
    %% Iterate between the update steps until convergence of max number of iterations 
    for counter = 1:MAX_ITER

        % 1) x-updates  
        Gamma_inv = sparse(real( (1/variance)*FtF + R'*diag(theta.^(-1))*R )); % coefficient matrix
        x = real( Gamma_inv\( (1/variance)*Fty ) ); % update x 

        % 2) theta-update 
        eta = r*beta - (1/2 + 1); % auxilary parameter eta
        if r==1 
            theta = 0.5*vartheta*( eta + sqrt( eta^2 + 2*real(R*x).^2/vartheta ) ); 
        elseif r==-1 
            theta = ( real(R*x).^2/2 + vartheta )/( -eta ); 
        else 
            error('Only r=1 and r=-1 are implemented yet!'); 
        end

        % store certain values in history structure 
        history.change_x(counter) = norm( x - x_OLD )/norm( x_OLD ); % absolute error 
        history.change_G(counter) = norm( G(x,theta) - G(x_OLD,theta_OLD) )/norm( G(x_OLD,theta_OLD) ); % relative error        
        x_OLD = x; theta_OLD = theta; % store new value of x and theta 

        % display these values if desired 
        if ~QUIET
            fprintf('%3d\t%0.2e\t%0.2e\t%0.2e\t%0.2e\n', ... 
                counter, history.change_x(counter), TOL_x, ... 
                history.change_G(counter), TOL_G);
        end
        
        % check for convergence 
        if ( history.change_x(counter) < TOL_x || ...
                history.change_G(counter) < TOL_G )%&& ... 
                %counter > MIN_ITER )
             break;
        end
        
    end

    % output the time it took to perform all operations 
    if ~QUIET
        toc(t_start);
    end
    
end