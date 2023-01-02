% Median filtering
i = imread('cameraman.tif');

subplot(1,3,1);
imshow(i); title('Input Image');

% add noise to input image
iNoisy = imnoise(i,'salt & pepper', 0.2);

% filter noisy image using 'median filter'
filtered = medfilt2(iNoisy);

% display output
subplot(1,3,2);
imshow(iNoisy); title('Salt & Pepper noise');

subplot(1,3,3);
imshow(filtered); title('Med. Filtered');
