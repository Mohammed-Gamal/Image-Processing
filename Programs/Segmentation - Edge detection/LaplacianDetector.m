% Clear Command Window, Workspace, close all Figures
clc;
clear;
close all;

% Read an image
[filename, pathname] = uigetfile({'*.*'}, 'Select an image', 'C:\Users\admin\Documents\MATLAB\Images');
a = imread(fullfile(pathname, filename));

% Show input image
subplot(1,2,1);
imshow(a);

% Handle RGB images
if ndims(a) == 3
    a = rgb2gray(a);
end

a = imresize(a, [450,450]); % resize the image to whatever size you like


% Define the Laplacian filter mask.
Laplacian = [0 1 0; 1 -4 1; 0 1 0];

% Convolve the image using Laplacian Filter
k1 = conv2(double(a), Laplacian, 'same');

% Display the image.
subplot(1,2,2);
imshow(abs(k1), []);
