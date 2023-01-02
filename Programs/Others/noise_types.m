% Noise types
i = imread('cameraman.tif');

% Salt & Pepper noise
iNoisy = imnoise(i, 'salt & pepper', 0.1);

subplot(2,2,1);
imshow([i, iNoisy]); title('Salt & Pepper');

% Impulse noise
i = im2double(rgb2gray(imread('peppers.png')));
p = 0.5; % p between 0 and 1
iNoisy = (i + p*rand(size(i)))/(1+p);

subplot(2,2,2);
imshow([i, iNoisy]); title('Impulse Noise');

% Gaussian noise
i = imread('pout.tif');
iNoisy = imnoise(i, 'gaussian', 0, 0.01);

subplot(2,2,3);
imshow([i, iNoisy]); title('Gaussian Noise');

% Multiplicative noise
i = imread('concordorthophoto.png');
iNoisy = imnoise(i,'speckle', 0.3);

subplot(2,2,4);
imshow([i, iNoisy]); title('Speckle Noise');
