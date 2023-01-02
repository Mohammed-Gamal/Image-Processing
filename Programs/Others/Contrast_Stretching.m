% Contrast/Histogram Stretching
i = imread('cameraman.tif');

% Original image
subplot(2,2,1);
imshow(i); title('Input Image');

subplot(2,2,2);
imhist(i); title('Hist. of input image');


% Histogram stretching
j = imadjust(i, [80/255 200/255], [0 1]);

subplot(2,2,3);
imshow(j);  title('Stretched Image');

subplot(2,2,4);
imhist(j); title('Hist. of stretched image');
