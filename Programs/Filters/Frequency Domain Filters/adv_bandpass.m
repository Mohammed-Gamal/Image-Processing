% input image
input_image = imread('cameraman.tif');
subplot(1,3,1); imshow(input_image);

% Convert image to double
input_image = double(input_image);

% Getting Fourier Transform of the input image, then shifting to the origin
FT_img = fftshift(fft2(input_image));


% function meshgrid(v, u) returns 2D grid
% which contains the coordinates of vectors v and u.
[u,v] = meshgrid(-128:127, -128:127);

% Calculating Euclidean Distance
duv = sqrt(u.^2 + v.^2);

% Assign Cut-off Frequency
D01 = double(duv <= 100);
D02 = double(duv >= 50);

H = D01 .* D02;

% Convolve with the filter
G = FT_img .* H;
subplot(1,3,2); fftshow(G);

% Inverse Fourier Transform of the convoluted image
output_image = real(ifft2(double(G)));

subplot(1,3,3); ifftshow(output_image);

