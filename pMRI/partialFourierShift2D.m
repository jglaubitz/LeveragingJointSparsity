function y = partialFourierShift2D(idx1,idx2,n,x,mode)
%PARTIALFOURIER  Partial Fourier operator
%
% Y = PARTIALFOURIER(IDX,N,X,MODE)

%if mode==1
if strcmp(mode,'notransp') % compute R*x 
   x = reshape(x,n,n); % transform vector into matrix 
   z = fft2(x)/sqrt(n*n); % apply normalized 2d fast Fourier transform 
   z = fftshift(z); % re-order the Fourier models
   y = z(sub2ind(size(z),idx1,idx2)); % subsample 
   
else
   z = zeros(n,n);
   z(sub2ind(size(z),idx1,idx2)) = x;
   z = ifftshift(z);
   y = ifft2(z)*sqrt(n*n);
   y = reshape(y,n*n,1);

end