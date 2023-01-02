% Convert RGB images into grayscale
rgb = imread('peppers.png');

subplot(3,1,1);
imshow(rgb); title('Original Image');

% Method #1
grayImage = rgb(:,:,1)*0.2989 + rgb(:,:,2)*0.5870+rgb(:,:,3)*0.1140;

subplot(3,1,2);
imshow(grayImage); title('grayscale image - method #1');


% Method #2
grayImage2 = rgb2gray(rgb);

subplot(3,1,3);
imshow(grayImage2); title('grayscale image - method #2');
