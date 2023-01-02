clc;
clear all;
close all;

% Read input image
[filename, filepath] = uigetfile({'*.*'}, 'Select an image');
i = imread(fullfile(filepath, filename));

% Display input image
subplot(1,2,1);
imshow(i); title('Input Image');

% Histogram stretching
if isa(i, 'logical')
    i = double(i);
else
    i = uint8(i);
end

j = imadjust(i, [80/255 200/255], [0 1]);


% Display output image
subplot(1,2,2);
imshow(j);  title('Stretched Image');

