clc
clear all
close all

% Read an input image
[filename, filepath] = uigetfile({'*.*'}, 'Select an image', 'C:\Users\admin\Documents\MATLAB\Image Processing\Images');
A = imread(fullfile(filepath, filename));

subplot(1,2,1);
imshow(A); title('Input Image');

% PAD THE MATRIX WITH ZEROS ON ALL SIDES
modifyA = zeros(size(A,1)+2, size(A,2)+2, 3);  % zeros(rows, cols, no. matrices - default is 1);
B = zeros(size(A));


% COPY THE ORIGINAL IMAGE MATRIX TO THE PADDED MATRIX
for x=1:size(A,1)
    for y=1:size(A,2)
        for z=1:size(A,3)
            modifyA(x+1, y+1, z) = A(x, y, z);
        end
    end
end


% LET THE WINDOW BE AN ARRAY
% STORE THE 3x3 NEIGHBOUR VALUES IN THE ARRAY
% SORT AND FIND THE MIDDLE ELEMENT
for i=1:size(modifyA, 1)-2
    for j=1:size(modifyA, 2)-2
        for k=1:size(modifyA, 3)
            window = zeros(9);
            inc = 1;
            for x=0:2
                for y=0:2
                    window(inc) = modifyA(i+x, j+y, k);
                    inc = inc + 1;
                end
            end
        end

        med = sort(window);
        % PLACE THE MEDIAN ELEMENT IN THE OUTPUT MATRIX
        B(i, j, k) = med(5);
        
        out = cat(3, B, ); 

    end
end

% CONVERT THE OUTPUT MATRIX TO 0-255 RANGE IMAGE
B = uint8(out);



subplot(1,2,2);
imshow(B, []); title('IMAGE AFTER MEDIAN FILTERING');
