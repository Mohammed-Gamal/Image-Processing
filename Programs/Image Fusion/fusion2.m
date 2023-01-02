% Load two fuzzy versions of an original image.
x1 = imread('clock1.png');
x2 = imread('clock2.png');


% Merge the two images from level 5 wavelet decompositions using the smy4 wavelet.
% Perform the fusion by taking the maximum of the absolute value of the coefficients for both 'approximations' and 'details'.
wv = 'sym4';
lv = 5;
xfus = wfusimg(x1, x2, wv, lv, 'max', 'max');


% Plot the original and fused images.
subplot(2,2,1);
image(x1); title('Catherine 1');

subplot(2,2,2);
image(x2); title('Catherine 2');

subplot(2,2,3)
image(xfus); title('Synthesized Image');

colormap(map)
