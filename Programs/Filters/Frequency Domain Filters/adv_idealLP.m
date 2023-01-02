% Clear Command Window, Workspace, close all Figures
clc;
clear;
close all;

% Read an image
[filename, filepath] = uigetfile({'*.*'}, 'Select an image');
a = imread(fullfile(filepath, filename));

% Display the original image
figure, imshow(a);
title('Original Image');

% Handle RGB images
if ndims(a) == 3
    a = rgb2gray(a);
end

a = imresize(a, [256,256]); % resize the image to whatever size you like

% find the size
[r,c] = size(a);

% Apply Fourier Transform to the original image
im_f = fft2(a);

% Display image in the Fourier transform
figure, imshow(log(abs(im_f)), []);
title('Fourier Transform');

% Shift image to the center
f_shift = fftshift(im_f);

% Display shifted image
figure, imshow(log(abs(f_shift)), []);
title('Shifted Image');

% Find the center of the frequancy domain
p = r ./ 2;
q = c ./ 2;

% Cut-off Frequancy
d0 = 35;

% Initialize ILPF
idealLP = zeros(c,r);

% Create ILPF
for i=1:r
    for j=1:c
        D = sqrt((i-p).^2 + (j-q).^2);
        idealLP(i,j) = D <= d0;
    end
end

% Display ILPF
figure, imshow(idealLP);
title('Ideal Low-Pass Filter');

% Display ILPF using meshc
figure, meshc(idealLP);
title('3D Ideal Low-Pass Filter');

% Convolve shifted image with ILPF
convolveF = f_shift .* idealLP;

% Shifted back the image
original_image = ifftshift(convolveF);

% Convert image to the spatial domain
RImage = abs(ifft2(original_image));

% Display image in the spatial domain
figure, imshow(RImage, []);
title('Result Image');
