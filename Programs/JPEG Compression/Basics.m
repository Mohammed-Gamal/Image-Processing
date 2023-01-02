% http://pi.math.cornell.edu/~web6140/TopTenAlgorithms/JPEG.html

% Read an image
I = imread('peppers.png');
% imshow(I)

ImageSize = 8*prod(size(I))


% Color transform: Convert RGB to YCbCr
Y = I;
A = [.229 .587 .114 ; -.168736 -.331264 .5 ; .5 -.418688 -.081312];
Y(:,:,1) = A(1,1)*I(:,:,1) + A(1,2)*I(:,:,2) + A(1,3)*I(:,:,3) + 0;
Y(:,:,2) = A(2,1)*I(:,:,1) + A(2,2)*I(:,:,2) + A(2,3)*I(:,:,3) + 128;
Y(:,:,3) = A(3,1)*I(:,:,1) + A(3,2)*I(:,:,2) + A(3,3)*I(:,:,3) + 128;


% However, Let's ignore the above method and use MATLAB's inbuilt convert (because of the normalizations):
Y = rgb2ycbcr(I);


% Plot Y'Cb'Cr colorspace
% Let's see what that colorspace looks like.
lb = {'Y (Luminance)', 'Cb (Blue Chrominance)', 'Cr (Red Chrominance)'};

for channel = 1:3
    Y_C = Y;
    Y_C(:, :, setdiff(1:3, channel)) = intmax(class(Y_C))/2;
    
    subplot(3,1,channel)
    imshow(ycbcr2rgb(Y_C)); title(lb{channel})
end


% Our eyes are senstitive to illuminance, not chrominance (color).
% Since our eyes are not particularly sensitive to chrominance, we can "downsample" the chrominance.
% Here, we remove x100 amount of "chrominance (color)" from the image and see that it has barely changed:
% subplot(1,2,1)
% imshow( I ); title('Original')

Y_d = Y;
Y_d(:,:,2) = 10*round(Y_d(:,:,2)/10);  % Cb Component
Y_d(:,:,3) = 10*round(Y_d(:,:,3)/10);  % Cr Component        (Chrominance Components)

% subplot(1,2,2)
% imshow(ycbcr2rgb(Y_d)); title('Downsample Chrominance')


% Our eyes are senstitive to illuminance (intensity).
% So, if we downsample the illuminance by x10, then there is a noticeable difference. (You'll have to zoom in to see it.)
% subplot(1,2,1)
% imshow( I ); title('Original')

Y_d = Y;
Y_d(:,:,1) = 10*round(Y_d(:,:,1)/10);  % Illuminance Component

% subplot(1,2,2)
% imshow(ycbcr2rgb(Y_d)); title('Downsample illuminance')
