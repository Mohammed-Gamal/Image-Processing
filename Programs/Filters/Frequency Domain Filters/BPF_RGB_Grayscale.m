clc
clear all
close all

% Read an input image
[filename, filepath] = uigetfile({'*.*'}, 'Select an image', 'C:\Users\admin\Documents\MATLAB\Image Processing\Images');
I = imread(fullfile(filepath, filename));

% Display original image
figure, imshow(I, []);


%------------- If an RGB image -------------
if ndims(I) == 3
   % Extract three images 
    Red   =  I(: , : , 1);
    Green =  I(: , : , 2);
    Blue  =  I(: , : , 3);

    % Get the size of any channel (they'all are the same size)
    [r,c] = size(Red);


    % // Change - Transform each channel, then shift
    F_r = fftshift(fft2(Red)); 
    F_g = fftshift(fft2(Green));
    F_b = fftshift(fft2(Blue)); 


    
    %-- ILPF Filtering --
    
    % Find the center of the frequancy domain
    p = r ./ 2;
    q = c ./ 2;

    % Cut-off Frequancy
    D01 = 1;
    D02 = 9;

    % Initialize BPF
    bandPass = zeros(c,r);

    % Create BPF
    for i=1:r
        for j=1:c
            D = sqrt((i-p).^2 + (j-q).^2);
            bandPass(i,j) = (D >= D01 && D <= D02);
        end
    end

    
    %// Change - To match size
    bandPass = imresize(bandPass, [r c]);
    
    % Display BPF
    figure, imshow(bandPass);


    % // Now filter
    FF_R = bandPass .* F_r; 
    FF_G = bandPass .* F_g; 
    FF_B = bandPass .* F_b; 


    %// Change - perform ifftshift, then ifft2, then cast to real
    % Inverse IFFT _RED 
    Ir = ifftshift(FF_R);
    Irr = real(ifft2(Ir));


    % Inverse IFFT _Green 
    Ig = ifftshift(FF_G);
    Igg = real(ifft2(Ig));


    % Inverse IFFT _Blue
    Ib = ifftshift(FF_B);
    Ibb = real(ifft2(Ib));


    %// Visualize the red, green and blue components
    b = zeros(r, c, 'uint8');
    
    image_red = cat(3,Irr, b, b);
    image_green = cat(3, b, Igg, b);
    image_blue = cat(3, b, b, Ibb);
    
    
    % Combine the three components together
    %// Change - Removed fluff
    b = uint8(cat(3, Irr, Igg, Ibb));
    
    %// Visulaization - Display the final image in a new figure
    figure;

    subplot(1,4,1);
    imshow(image_red), title('R');

    subplot(1,4,2);
    imshow(image_green), title('G');

    subplot(1,4,3);
    imshow(image_blue), title('G');

    subplot(1,4,4);
    imshow(b), title('Out');

    
%------------- If not an RGB image -------------
else 
    
    % find the size
    [r,c] = size(I);

    % Apply Fourier Transform to the original image
    im_f = fft2(I);

    % Shift image to the center
    f_shift = fftshift(im_f);
    
    % Find the center of the frequancy domain
    p = r ./ 2;
    q = c ./ 2;

    % Cut-off Frequancy
    D01 = 10;
    D02 = 20;

    % Initialize BPF
    bandPass = zeros(c,r);

    % Create BPF
    for i=1:r
        for j=1:c
            D = sqrt((i-p).^2 + (j-q).^2);
            bandPass(i,j) = (D >= D01 && D <= D02);
        end
    end

    
    %// Change - To match size
    bandPass = imresize(bandPass, [r c]);
    
    % Display BPF
    figure, imshow(bandPass);
    title('Bandpass Filter');
    
    
    % Convolve shifted image with IHPF
    convolveF = f_shift .* bandPass;

    % Shifted back the image
    original_image = ifftshift(convolveF);

    % Convert image to the spatial domain
    RImage = real(ifft2(original_image));

    % Display image in the spatial domain
    figure, imshow(RImage, []);
end
