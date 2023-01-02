clc
clear all
close all


% Read an input image
[filename, filepath] = uigetfile({'*.*'}, 'Select an image', 'C:\Users\admin\Documents\MATLAB\Image Processing\Images');
I = imread(fullfile(filepath, filename));


% Display original image
figure, imshow(I, []);


% Convert image to double format
I = im2double(I);
    
    
%------------- If an RGB image -------------
if ndims(I) == 3
    
    % log of the image
    log_I = log(1 + I);


    %-- Homomorphic Filtering (Gaussian Highpass) --
    M = 2*size(log_I,1) + 1;
    N = 2*size(log_I,2) + 1;
    
    
    % Cut-off frequency
    D0 = 10;
    
    
    % Gaussian Lowpass
    [X,Y] = meshgrid(1:N, 1:M);
    
    centerX = ceil(N/2);
    centerY = ceil(M/2);
    
    Duv = (X - centerX).^2 + (Y - centerY).^2; % D(u,v)
    H = exp( -Duv ./ (2 .* D0.^2) );
    
    
    % Gaussian Highpass = 1 - Gaussian Lowpass
    H = 1 - H;
    
    % Display the filter spectrum
    figure, imshow(H, []);
    
    % // Change - Shift then Transform
    H = fftshift(H);
    If = fft2(log_I, M, N); % fourier transform
    
    
    %// Change - Apply the filter (using repmat), ifftshift, then cast to real
    Iout = real(ifft2(repmat(H, [1, 1, 3]) .* If));
    
    % Reconstruct the RGB image
    Iout = Iout(1:size(log_I,1), 1:size(log_I,2), :);

    
    % Inverse log 
    Ihmf = exp(Iout) - 1;
    
    %// Visulaization - Display the final image in spatial domain
    figure, imshow(Ihmf, []);

    
%------------- If not an RGB image -------------
else 
    
    % log of the image
    log_I = log(1 + I);

    
    %-- Homomorphic Filtering (Gaussian Highpass) --
    [M,N] = size(log_I);
    
    % Cut-off frequency
    D0 = 7;
    
    % Gaussian Lowpass
    [X,Y] = meshgrid(1:N, 1:M);
    
    centerX = ceil(N/2);
    centerY = ceil(M/2);
    
    Duv = (X - centerX).^2 + (Y - centerY).^2; % D(u,v)
    H = exp( -Duv ./ (2 .* D0.^2) );
    
    
    % Gaussian Highpass = 1 - Gaussian Lowpass
    H = 1 - H;
    
    % Display the filter spectrum
    figure, imshow(H, []);
    
    
    % // Change - Shift the filter to the center
    H = fftshift(H);
    
    If = fft2(log_I, M, N); % fourier transform
    
    % Convolve shifted image with the filter
    convolveF = If .* H;
    
    
    %// Change - Apply the filter (using repmat), ifftshift, then cast to real
    Iout = real(ifft2(convolveF));

    
    % Inverse log 
    Ihmf = exp(Iout) - 1;
    
    %// Visulaization - Display the final image in spatial domain
    figure, imshow(Ihmf, []);
end
