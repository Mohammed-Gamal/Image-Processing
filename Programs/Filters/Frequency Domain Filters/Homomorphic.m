% Clear Command Window, Workspace, close all Figures
clc;
clear all;
close all;

% Input image
i = imread('cameraman.tif');
subplot(1,2,1); imshow(i);

d_im = im2double(i);

% log of the image
log_Img = log2(1+ d_im);

% DFT of logged image
fftlog = fft2(log_Img);

% Filter (Butterworth HP) Applying DFT image
H = butterhp(d_im, 15, 2);
c = fftlog .* H;

% Inverse DFT of filtered image
h = real(ifft2(c));

% Inverse log
h1 = exp(h);

subplot(1,2,2); ifftshow(h1);
