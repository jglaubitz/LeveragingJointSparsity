function [idx1,idx2, R ] = MMV_SampMatrixRadial(n,beams,c,C)
%
% produces the fan MRI mask, of size n*n,
% beams is the number of angles 
% c is the coil index 
% C is the total number of coils 
%
%m = round(sqrt(2)*n);
m = ceil(sqrt(2)*n);
aux = zeros(m,m); 
ima = aux;
aux(round(m/2+1),:) = 1;
aux(round(m/2),:)=1;
%aux(round(m/2+1),:) = 1;
angle = 180/beams; 
angle_init = 0.5*(c-1)*angle/C; 
angles = [angle_init:angle:180-angle];
for a = 1:length(angles)
    ang = angles(a);
    a = imrotate(aux,ang,'crop');
    ima = ima + a;
end
ima = ima(round(m/2+1) - n/2:round(m/2+1) + n/2-1,...
          round(m/2+1) - n/2:round(m/2+1) + n/2-1);

R = (ima > 0);

[idx1,idx2] = ind2sub(size(R), find(R == 1));

end
