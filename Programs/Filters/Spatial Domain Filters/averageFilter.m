% Neighborhod averaging filtering
i = imread('cameraman.tif');

subplot(1,3,1);
imshow(i); title('Input Image');

% Method #1
mask = ones(3,3)/9;
filtered = filter2(mask, i);

subplot(1,3,2);
imshow(uint8(filtered)); title('Method #1');


% Method #2
mask = fspecial('average', [3 3]);
filtered = filter2(mask, i, 'valid');

subplot(1,3,3);
imshow(uint8(filtered)); title('Method #2');
% Or imshow(filtered/255); title('Method #2');
