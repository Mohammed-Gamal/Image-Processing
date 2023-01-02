% Histogram of RGB images
i = imread('peppers.png');

% Red component histogram
red_hist = imhist(i(:,:,1));
subplot(1,3,1);
plot(red_hist, 'red'); title('Red');

% Green component histogram
green_hist = imhist(i(:,:,2));
subplot(1,3,2);
plot(green_hist, 'green'); title('Green');

% Blue component histogram
blue_hist = imhist(i(:,:,3));
subplot(1,3,3);
plot(blue_hist, 'blue'); title('Blue');
