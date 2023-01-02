% Segementation - Edge detection using 'edge' builtin matlab function
% edge(image, 'filtername', threshold)

% Read the image
rgbImage = imread('circuit.tif');

subplot(3, 3, 1), 
imshow(rgbImage); 
title('Original Image'); 

% Convert RGB image to grayscale, if not
if ndims(rgbImage) == 3
    grayImage = rgb2gray(rgbImage); 
else
    grayImage = rgbImage;
end

subplot(3, 3, 2), 
imshow(grayImage); 
title('Grayscale Image'); 

%--------------------------------------------
% Sobel Edge Detection

J = edge(grayImage, 'Sobel'); 

subplot(3, 3, 3), 
imshow(J); 
title('Sobel'); 

%--------------------------------------------
% Prewitt Edge detection 

K = edge(grayImage, 'Prewitt'); 
subplot(3, 3, 4), 

imshow(K); 
title('Prewitt'); 

%--------------------------------------------
% Robert Edge Detection 

L = edge(grayImage, 'Roberts'); 

subplot(3, 3, 5), 
imshow(L); 
title('Robert'); 

%--------------------------------------------
% Log Edge Detection 

M = edge(grayImage, 'log'); 

subplot(3, 3, 6), 
imshow(M); 
title('Log'); 

%--------------------------------------------
% Zerocross Edge Detection 

M = edge(grayImage, 'zerocross'); 

subplot(3, 3, 7), 
imshow(M); 
title('Zerocross'); 

%--------------------------------------------
% Canny Edge Detection 

N = edge(grayImage, 'Canny'); 

subplot(3, 3, 8), 
imshow(N); 
title('Canny');
