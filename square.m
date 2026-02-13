%----------------------------------------------------------
% Copyright (C) 2025 Mauricio Kugler & Etore Maloso Tronconi
% Embedded Systems & Instrumentation Department
% École Supérieure d'Ingénieurs - ESIGELEC, France
% This software is intended for research purposes only;
% its redistribution is forbidden under any circumstances.
%----------------------------------------------------------

%==========================================================
size = 512; 
I = zeros(size,size,'uint8');
numImages = 20; 
imgIdx = 1;

%One square to be segmentated
max_num_sq = 1;
max_num_c = 4;
path = "./dataset/train";
while imgIdx <= numImages

    [S, Labels,nS] = SquareCoordinates(size,max_num_sq);
    [C, nC] = CirclesCoordinates(size,max_num_c);

    RGB = I;
    Label = I;

    for i=1:nS
        RGB = insertShape(RGB, "filled-rectangle", S(i,:), 'Color', randi([80, 255], 1, 3), 'Opacity', 1);
        Label = insertShape(Label, "filled-rectangle", S(i,:), 'Color', randi([80, 255], 1, 3), 'Opacity', 1);
    end
    
    for i = 1:nC
        RGB = insertShape(RGB, 'filled-circle', C(i,:), 'Color', randi([80, 255], 1, 3), 'Opacity', 1);
    end
    
    Imag = rgb2gray(RGB);
    num = bwconncomp(Imag,4).NumObjects;
   
    if num ~= nC + nS + 0 
        fprintf('Different quantities of shapes: Detected %d, Expected %d\n',num, nC + nS + 0);
        
    else
        % Show and save
        % figure
        % imshow(RGB);
        imwrite(RGB, sprintf(path+'/image_%d.png', imgIdx));
        imwrite(Label, sprintf(path+'/label_%d.png', imgIdx));
        imgIdx = imgIdx + 1;

    end    
    
end

function [S_out, Labels, n] = SquareCoordinates(size,nS)
    n = randi([1,nS]); 
    Sq = zeros(1,5); 
    for i=1:n
        k = randi([4, 16]);
        a = size/k;
        omega =  pi/2 * rand();
        x = randi([round(a), round(size-a)]);
        y = randi([round(a), round(size-a)]);
        
        
        Labels = [
            - a/2,  - a/2;
            + a/2,  - a/2;
            - a/2,  + a/2;
            + a/2,  + a/2;
        ];
        
        RotationMatrix = [
            cos(omega), -sin(omega);
            sin(omega),  cos(omega)
        ];
        
        Labels = (RotationMatrix * Labels')';
        Labels(:,1) = (Labels(:,1) + x)/size;
        Labels(:,2) = (Labels(:,2) + y)/size;
    
        
        omega = rad2deg(omega);
        Sq(i, :) = [x, y, a, a, omega];
    end
    S_out = Sq;
    
end


function [C_out, n_out] = CirclesCoordinates(size,nC)
    n = randi([1, nC]);
    C = zeros(n, 3);
    for i = 1:n
        k = randi([8, 16]);
        radius = size/k;
        x = randi([round(radius), round(size-radius)]);
        y = randi([round(radius), round(size-radius)]);
        C(i, :) = [x, y, radius];
    end
    C_out = C;
    n_out = n;

 end