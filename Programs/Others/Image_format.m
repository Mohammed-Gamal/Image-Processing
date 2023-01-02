clc;
clear all;
close all;

% i = imread('football.jpg');

% valid extensions
valid_extensions = {'jpg', 'jpeg', 'jfif', 'pjpeg', 'pjp'};
valid_extensions

% read image info
info = imfinfo('football.jpg');
info
info.Format

% check image extension format
isValid = false;

for j=1:size(valid_extensions)
   if strcmp(info.Format, valid_extensions(1, j))
       isValid = true;
   end
end


% Display results
if isValid
    disp('Valid');
else
    disp('Not valid!');
end
