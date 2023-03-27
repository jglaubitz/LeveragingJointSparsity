%% MMV_GSBL_1d
%
% Description: 
%  Function for the MMV-IAS algorithm.  
%  We assume a one-dimensional signal and iid noise
%
% INPUT: 
%  J :                  number of MMVs
%  F :                  list of forward operator 
%  y :                  MMVs 
%  variance :           list of noise variances 
%  R :                  regularization matrix
%  r, beta, vartheta :  regualrization hyper-hyper-parameters 
%
% OUTPUT: 
%  x :          list of MAP estimates for x 
%  theta :     	MAP estimate for common hyper-parameter vector theta 
%  history :    structure containing residual norms and the tolerances at each iteration
%
% Author: Jan Glaubitz 
% Date: Jan 18, 2023 
% 

function [x, theta, history] = MMV_IAS_1d( J, F, y, variance, R, r, beta, vartheta, QUIET )

    t_start = tic; % measure time 

    %% Global constants and defaults  
    %MIN_ITER = 10; 
    MAX_ITER = 1000; 
    TOL_x = 1e-6;
    TOL_G = 1e-6;
    
    %% Data preprocessing & initial values 
    N = size(R,2); % number of pixels 
    K = size(R,1); % number of outputs of the regularization operator 
    theta = ones(K,1); % initial value for common hyper-parameter vector 
    theta_OLD = ones(K,1); % auxilary variable to compute change in beta 
    parfor j=1:J 
        FtF{j} = F{j}'*F{j}; % product corresponding to the forward operator 
        Fty{j} = F{j}'*y{j}; % forward operator applied to the indirect data 
        M{j} = size(F{j},1); % number of (indirect) measurements
        alpha{j} = 1/variance{j}; 
        x_OLD{j} = zeros(N,1); % mean 
        G{j} = @(x,theta) (1/variance{j})*norm( F{j}*x - y{j} ).^2/2 + ... 
            norm( diag(theta.^(-1/2))*R*x ).^2/2 + ... 
            sum( (theta./vartheta).^r ) + ... 
            ( 1/2 + 1 - r*beta )*sum( log(theta) ); % objective function
    end 
   
    %% Outputting the learning progress 
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', ... 
            'iter', 'change in x', 'tol x', 'change in G', 'tol G');
    end
    
    %% Iterate between the update steps until convergence of max number of iterations 
    for counter = 1:MAX_ITER

        % 1) x-updates 
        D_theta_inv = sparse(diag(1./theta)); % precision matrix 
        parfor j=1:J
            Gamma_inv{j} = sparse(real( alpha{j}*FtF{j} + R'*D_theta_inv*R )); % coefficient matrix
            x{j} = Gamma_inv{j}\( alpha{j}*Fty{j} ); % update x 
        end

        % 2) theta-update 
        % auxilary value 
        aux = 0;
        for j=1:J 
            aux = aux + (real(R*x{j})).^2; 
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
        history.change_G(counter) = 0; 
        for j=1:J 
            aux_x = norm( x{j} - x_OLD{j} )/norm( x_OLD{j} ); % relative change in x{j}
            history.change_x(counter) = history.change_x(counter) + aux_x; % sum of relative changes
            aux_G = norm( G{j}(x{j},theta) - G{j}(x_OLD{j},theta_OLD) )/norm( G{j}(x_OLD{j},theta_OLD) ); % relative change in G
            history.change_G(counter) = history.change_G(counter) + aux_G; % sum of relative changes
            x_OLD{j} = x{j}; % store value of mu 
        end

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