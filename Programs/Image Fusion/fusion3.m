% Load two images of the same size.
load mask
a = X;

load bust
b = X;


% Define the fusion method and call the fusion function helperUserFusion
% The source code for helperUserFusion is listed in the appendix.
fus_method = struct('name', 'userDEF', 'param', 'helperUserFusion');


% Merge the images twice with the user-defined method.
% First use wfusmat, which fuses the images themselves and not their wavelet decompositions.
% Then use wfusimg, which fuses the wavelet decompositions.
c = wfusmat(a, b, fus_method);
d = wfusimg(a, b, 'db4', 5, fus_method, fus_method);


% Plot the original and fused images.
subplot(2,2,1);
image(a); title('Original Image 1');

subplot(2,2,2);
image(b); title('Original Image 2');

subplot(2,2,3);
image(c); title('Fused Images');


subplot(2,2,4);
image(d); title('Fused Decompositions');

colormap(pink(220))


% Visualize the differences between the merged images.
figure, image(c - d);

colormap(pink(220))
