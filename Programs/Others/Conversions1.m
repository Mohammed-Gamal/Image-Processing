% Conversion between Ind to RGB
[i, colormap] = imread('kids.tif');

subplot(2,2,1);
imshow(i, colormap); title('Original image');

% Convert image from indexed into RGB
rgbImage = ind2rgb(i, colormap);

subplot(2,2,2);
imshow(rgbImage); title('ind2rgb');

% Convert image from RGB to grayscale
k = rgb2gray(rgbImage);

subplot(2,2,3);
imshow(k);  title('rgb2gray');

% Convert image from indexed to graysacle
l = ind2gray(i, colormap);

subplot(2,2,4);
imshow(l); title('ind2gray');
