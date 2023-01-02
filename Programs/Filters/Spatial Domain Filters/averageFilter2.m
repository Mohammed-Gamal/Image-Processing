% Neighborhod averaging filtering
% Clear Command Window, Workspace, close all Figures
clc;
clear;
close all;

% Read an image
[filename, pathname] = uigetfile({'*.*'}, 'Select an image', 'C:\Users\admin\Documents\MATLAB\Images');
a = imread(fullfile(pathname, filename));

% Show input image
subplot(1,3,1);
imshow(a); title('Input Image');

% Handle RGB images
if ndims(a) == 3
    a = rgb2gray(a);
end

% Method #1
mask = ones(3,3)/9;
filtered = filter2(mask, a);

subplot(1,3,2);
imshow(uint8(filtered)); title('Method #1');


% Method #2
mask = fspecial('average', [3 3]);
filtered = filter2(mask, a, 'valid');

subplot(1,3,3);
imshow(uint8(filtered)); title('Method #2');
% Or imshow(filtered/255); title('Method #2');
