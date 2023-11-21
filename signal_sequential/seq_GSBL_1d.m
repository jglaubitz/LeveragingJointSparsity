%% seq_GSBL_1d
%
% Description: 
%  Function for the GSBL algorithm 
%  We assume a one-dimensional signal and iid noise
%
% INPUT: 
%  F :          forward operator 
%  y :          measurement vector 
%  variance :   fixed noise variance 
%  R :          regularization matrix
%  c, d :       regualrization hyper-hyper-parameters 
%  beta_init : initial guess for theta 
%
% OUTPUT: 
%  x :          MAP estimate for x 
%  beta :     	MAP estimate for hyper-parameter vector beta 
%  history :    structure containing residual norms and the tolerances at each iteration
%
% Author: Jan Glaubitz 
% Date: Nov 09, 2023 
% 

function [x, beta, history] = seq_GSBL_1d( F, y, variance, R, c, d, beta_init, QUIET )

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
    G = @(x,beta) (1/variance)*norm( F*x - y ).^2/2 + norm( diag(beta.^(1/2))*R*x ).^2/2 + ... 
        sum( beta.*d ) - sum( ( c - 1/2 ).*log(beta) ); % objective function 
    
    %% Initial values for the inverse variances and the mean  
    alpha = 1/variance; % go over from variance to precision 
    beta = beta_init; % initial value for beta
    beta_OLD = ones(K,1); % auxilary variable to compute change in beta 
    x_OLD = zeros(N,1); % auxilary variable to compute change in x
    
    %% Outputting the learning progress 
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', ... 
            'iter', 'change in x', 'tol x', 'change in G', 'tol G');
    end
    
    %% Iterate between the update steps until convergence of max number of iterations 
    for counter = 1:MAX_ITER

        % 1) x-update  
        Gamma_inv = sparse(real( alpha*FtF + R'*diag(beta)*R )); % coefficient matrix
        x = Gamma_inv\( alpha*Fty ); % update x

        % 2,3) alpha- and beta-update 
        %alpha = ( M/2 + c - 1 )./( norm(F*x-y).^2/2 + d );
        beta = ( c - 1/2 )./( real(R*x).^2/2 + d );

        % store certain values in history structure 
        history.change_x(counter) = norm( x - x_OLD )/norm( x_OLD ); % absolute error 
        history.change_G(counter) = norm( G(x,beta) - G(x_OLD,beta_OLD) )/norm( G(x_OLD,beta_OLD) ); % relative error        
        x_OLD = x; beta_OLD = beta; % store new value of x and theta 

        % display these values if desired 
        if ~QUIET
            fprintf('%3d\t%0.2e\t%0.2e\t%0.2e\t%0.2e\n', ... 
                counter, history.change_x(counter), TOL_x, ... 
                history.change_G(counter), TOL_G);
        end
        
        % check for convergence 
        if ( history.change_x(counter) < TOL_x || ...
                history.change_G(counter) < TOL_G ) %&& ... 
%                counter > MIN_ITER )
             break;
        end
        
    end

    % output the time it took to perform all operations 
    if ~QUIET
        toc(t_start);
    end
    
end