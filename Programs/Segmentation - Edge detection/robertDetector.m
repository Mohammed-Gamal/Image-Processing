% Clear Command Window, Workspace, close all Figures
clc;
clear;
close all;

% Read an image
[filename, pathname] = uigetfile({'*.*'}, 'Select an image');
a = imread(fullfile(pathname, filename));

% Show input image
subplot(2,2,1);
imshow(a); title('Input Image');

% Handle RGB images
if ndims(a) == 3
    a = rgb2gray(a);
end

a = imresize(a, [450,450]); % resize the image to whatever size you like
a = im2double(a);

% Roberts Operator Masks
hx = [+1 0; 0 -1];
hy = [0 +1; -1 0];

% Compute Gx and Gy
Gx = imfilter(a, hx);

subplot(2,2,2);
imshow(Gx); title('Gx');


Gy = imfilter(a, hy);

subplot(2,2,3);
imshow(Gy); title('Gy');

% Calculate the magnitude
Gxy = sqrt(Gx.^2 + Gy.^2);

% Display output image
subplot(2,2,4);
imshow(Gxy); title('Gxy - Output Image');
