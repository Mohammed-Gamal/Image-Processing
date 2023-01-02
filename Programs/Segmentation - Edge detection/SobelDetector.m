% Edge Detection using Sobel Detector
a = imread('peppers.png');

% Display input image
subplot(1,2,1);
imshow(a); title('Input Image');

b = rgb2gray(a);
c = double(b);

for i = 1 : size(c, 1)-2
    for j = 1:size(c, 2)-2
        gx = (1*c(i,j)+2*c(i, j+1)+1*c(i, j+2))-(1*c(i+2, j)+2*c(i+2, j+1)+1*c(i+2, j+2));
        gy = (1*c(i,j)+2*c(i+1, j)+1*c(i+2, j))-(1*c(i, j+2)+2*c(i+1, j+2)+1*c(i+2, j+2));
        b(i,j) = sqrt(gx.^2 + gy.^2);
    end
end

% Display output image
subplot(1,2,2);
imshow(b); title('Output Image');
