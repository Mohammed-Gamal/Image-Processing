% Clear Command Window, Workspace, close all Figures
clc;
clear;
close all;

% Input image
[filename, filepath] = uigetfile({'*.*'}, 'Select an image');
a = imread(fullfile(filepath, filename));

% Show input image
subplot(2,3,1);
imshow(a); title('Input Image');

% Handle RGB images
if ndims(a) == 3
    a = rgb2gray(a);
end

a = imresize(a, [256,256]); % resize the image to whatever size you like

% Convert image to double format
a = im2double(a);


% log of the image
log_Img = log2(1 + a);

subplot(2,3,2);
imshow(log_Img); title('log image');


% DFT of logged image, after shifting
fftlog = fft2(log_Img);

subplot(2,3,3);
fftshow(fftlog); title('DFT spectrum');

% Filter Applying DFT image
H = butterhp(a, 15, 2);  % Butterworth HP filter

c = fftlog .* H;  % apply filter

subplot(2,3,4);
imshow(c); title('Applied filter');

% Inverse DFT of filtered image
im_n = real(ifft2(c));

subplot(2,3,5);
imshow(im_n); title('Inverse DFT');

% Inverse log 
im_e = exp(im_n);

subplot(2,3,6);
ifftshow(im_e); title('anti-logarithm');
