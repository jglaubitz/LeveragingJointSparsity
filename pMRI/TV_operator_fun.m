%% TV_operator_fun
%
% Description: 
%  Function to give a function for the TV operator 
% 
% INPUT: 
%  x :      input vector
%  order :	order of the TV operator 
%
% OUTPUT: 
%  y :    	output vector
%
% Author: Jan Glaubitz 
% Date: Jan 30, 2023  
%

function y = TV_operator_fun( N, x, order, flag )

    % Compute the univariate TV operator matrix 
    R = TV_operator( N, order ); % TV operator 
    R_tr = R'; % Its transpose 
    K = size(R,1); 

    if strcmp(flag,'notransp') % compute R*x 
        x = reshape(x,N,N);
        y_hor = R*x; % vertical/column differences  
        y_ver = x*R_tr; % vertical differences  
        y = [y_hor(:);y_ver(:)]; % vectorize and stack together 
    
    elseif strcmp(flag,'transp') % compute R'*x 
        z1 = reshape( x(1:K*N), K, N ); 
        z2 = reshape( x(K*N+1:end), N, K );
        y = R_tr*z1 + z2*R; 
        y = y(:); % vectrize

    end

end