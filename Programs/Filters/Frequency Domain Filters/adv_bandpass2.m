% Clear Command Window, Workspace, close all Figures
clc;
clear;
close all;

% Read an image (e.g., 'instructor.tif')
[filename, filepath] = uigetfile({'*.*'}, 'Select an image');
a = imread(fullfile(filepath, filename));

% Show input image
imshow(a); title('Original Image');

% Handle RGB images
if ndims(a) == 3
    a = rgb2gray(a);
end

a = imresize(a, [450,450]);  % resize the image

% Find image size
[r,c] = size(a);

% Apply Fourier Transform to the original image
im_f = fft2(a);

% Display image in the Fourier transform
figure, imshow(log(abs(im_f)), []);
title('Fourier Transform');

% Shift image to the center (f_shift = FT_img)
f_shift = fftshift(im_f);

% Find the center of the frequancy domain
p = r ./ 2;
q = c ./ 2;

% Cut-off Frequancy
D01 = 10;
D02 = 20;

% Initialize BPF
bandPass = zeros(c,r);

% Create BPF
for i=1:r
    for j=1:c
        D = sqrt((i-p).^2 + (j-q).^2);
        bandPass(i,j) = (D >= D01 && D <= D02);
    end
end

% Display BPF
figure, imshow(bandPass);
title('Bandpass Filter');

% Display BPF using meshc
figure, meshc(bandPass);
title('3D Bandpass Filter');

% Convolve shifted image with BPF
convolveF = f_shift .* bandPass;

% Shift back the image 
original_image = ifftshift(convolveF);

% Convert image to the spatial domain
RImage = real(ifft2(original_image));

% Display Image in the spatial domain
figure, imshow(RImage, []);
title('Result Image');
