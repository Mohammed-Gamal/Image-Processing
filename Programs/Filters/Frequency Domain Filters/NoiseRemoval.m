% Clear Command Window, Workspace, close all Figures
clc;
clear;
close all;

% Input image
[filename, filepath] = uigetfile({'*.*'}, 'Select an image');
a = imread(fullfile(filepath, filename));

% Show input image
subplot(1,3,1);
imshow(a);

% Handle RGB images
if ndims(a) == 3
    a = rgb2gray(a);
end

a = imresize(a, [700,700]); % resize the image to whatever size you like


% define noise
noisy_image = imnoise(a, 'gaussian', 0.15);
subplot(1,3,2); imshow(noisy_image);

% Apply Fourier Transform, then shift to the origin
ft = fftshift(fft2(noisy_image));

% use butterworth low-pass filter
H = butterlp(ft, 20, 2);

conv = H .* ft;

% Apply Inverse Fourier Transform
out = ifft2(conv);

% display denoised image
subplot(1,3,3);
ifftshow(out);
