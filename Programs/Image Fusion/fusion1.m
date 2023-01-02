% Load the mask and bust images.
load mask
x1 = X;

load bust
x2 = X;

% Merge the two images from level 5 wavelet decompositions using the db2 wavelet.
% Perform the fusion by taking the mean for both 'approximations' and 'details'.
wv = 'db2';
lv = 5;
xfus_mean = wfusimg(x1, x2, wv, lv, 'mean', 'mean');

% Merge the two images again, but this time perform the fusion by
% taking the 'maximum of the approximations' and the 'minimum for the details'.
xfus_maxmin = wfusimg(x1, x2, wv, lv, 'max', 'min');

% Plot the original and fused images.
subplot(2,2,1);
image(x1); title('Mask');

subplot(2,2,2);
image(x2); title('Bust');

subplot(2,2,3);
image(xfus_mean); title('Synthesized Image: mean-mean');

subplot(2,2,4);
image(xfus_maxmin); title('Synthesized Image: max-min');

colormap(map);
