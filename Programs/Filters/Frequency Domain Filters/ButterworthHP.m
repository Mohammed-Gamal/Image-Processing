% Clear Command Window, Workspace, close all Figures
clc;
clear;
close all;

% Input image
[filename, filepath] = uigetfile({'*.*'}, 'Select an image');
a = imread(fullfile(filepath, filename));

% Show input image
figure, imshow(a); title('Original Image');

% Handle RGB images
if ndims(a) == 3
    a = rgb2gray(a);
end

a = imresize(a, [700,700]); % resize the image to whatever size you like

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

% Create Butterworth HP filter
bwhp = butterhp(a, 15, 1);

% Display BWHF
figure, imshow(bwhp);
title('Buttworth HP Filter');

% Display 3D BWHF using meshc
figure, meshc(bwhp);
title('3D Buttworth HP Filter');

% Convolve shifted image with BWHF
convolveF = f_shift .* bwhp;

% Shift back the image 
original_image = ifftshift(convolveF);

% Convert image to the spatial domain
RImage = abs(ifft2(original_image));

% Display Image in the spatial domain
figure, imshow(RImage, []);
title('Result Image');
