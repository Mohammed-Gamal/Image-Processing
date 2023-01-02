% Clear Command Window, Workspace, close all Figures
clc;
clear;
close all;

% Read an image
[filename, pathname] = uigetfile({'*.*'}, 'Select an image');
a = imread(fullfile(pathname, filename));

% Show input image
subplot(1,3,1);
imshow(a); title('Input Image');

% Convert image into double format
a = im2double(a(:,:,1));

% filter the image using DoG filter
H1 = fspecial('gaussian', 21, 15);
H2 = fspecial('gaussian', 21, 20);

% DoG filter
DoG = H1 - H2;

subplot(1,3,2);
imshow(DoG,[]);

dogFilterImage = conv2(a, DoG, 'same');


% Display output image
subplot(1,3,3);
imshow(dogFilterImage, []);

