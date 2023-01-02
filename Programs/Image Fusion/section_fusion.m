image1 = imread(fullfile('C:\Users\admin\Documents\MATLAB\Image Processing\Images', 'ct.jpg'));
image2 = imread(fullfile('C:\Users\admin\Documents\MATLAB\Image Processing\Images', 'mri.jpeg'));

image1 = double(image1);
image2 = double(image2);

fusedImage = (image1 + image2) / 2;

subplot(1,3,1);
imshow(image1/255, []); title('CT Image');

subplot(1,3,2);
imshow(image2/255, []); title('MRI Image');

subplot(1,3,3);
imshow(fusedImage/255, []); title('Fused Image');
