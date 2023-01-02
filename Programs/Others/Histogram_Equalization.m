% Histogram Equalization

i = imread('pout.tif');

subplot(2,2,1);
imshow(i);

subplot(2,2,2);
imhist(i);

j = histeq(i);

subplot(2,2,3);
imshow(j);

subplot(2,2,4);
imhist(j);
