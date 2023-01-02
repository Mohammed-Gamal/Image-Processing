% Color RGB images - Histogram Equalization

% Read the input image
im = imread('football.jpg');

subplot(2,2,1),
imshow(im); title('Before');


% Convert the RGB image into HSV image format
HSV = rgb2hsv(im);

% Perform Histogram Equalization on intensity component
Heq = histeq(HSV(:,:,3));

HSV_mod = HSV;
HSV_mod(:,:,3) = Heq;

% Convert the HSV image back into RGB
RGB = hsv2rgb(HSV_mod);


subplot(2,2,2);
imshow(RGB); title('After');



%****** DISPLAY THE HISTOGRAM OF THE ORIGINAL AND THE EQUALIZED IMAGE ******%
HIST_IN = zeros([256 3]);
HIST_OUT = zeros([256 3]);


% HISTOGRAM OF THE RED,GREEN AND BLUE COMPONENTS
mymap = [1 0 0; 0.2 1 0; 0 0.2 1];

HIST_IN(:,1) = imhist(I(:,:,1),256);  % RED
HIST_IN(:,2) = imhist(I(:,:,2),256);  % GREEN
HIST_IN(:,3) = imhist(I(:,:,3),256);  % BLUE

subplot(2,2,3), bar(HIST_IN); colormap(mymap);


HIST_OUT(:,1) = imhist(RGB(:,:,1),256);  % RED
HIST_OUT(:,2) = imhist(RGB(:,:,2),256);  % GREEN
HIST_OUT(:,3) = imhist(RGB(:,:,3),256);  % BLUE

subplot(2,2,4), bar(HIST_OUT); colormap(mymap);
