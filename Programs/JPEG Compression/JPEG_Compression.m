% Read an input image
[filename, filepath] = uigetfile({'*.jpg'; '*.jpeg'; '*.jfif'; '*.pjpeg'; '*.pjp'}, 'Select a JPEG type image', 'C:\Users\admin\Documents\MATLAB\Image Processing\Images');
i = imread(fullfile(filepath, filename));

% Apply a lossy JPEG image compression
dct1(i, 8, 8)
