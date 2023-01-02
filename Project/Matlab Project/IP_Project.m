%{

        Copyright © 2022/2023. All rights reserved.


        Suez Canal University
        Faculty of Computers & Informatics - Computer Science Department
        Image Processing Practical Project 2022/2023 - 4th year


        Dr. Ghada El-Taweel


        Authors:

            - Mohamed Gamal Abd El-Nasser Abd El-Motalleb
            - Mohamed Ali Abd El-Karim Ali
            - Mohamed Hamdy Ahmed Hamdy
            - Albraa Hany Hemdan Fishawy
            - Karim Kamal Mohamed Ahmed         



        ? Project description and used functions:
            
             ? Preview Section (Original Image ? Filtered Image)

             ? Menu Section (Import, apply, reset and exit)

             ? Operation time in seconds

             ? Appliend Operations section:

                   ? Main Operations Sub-section
                       • High-pass filter
                       • Low-pass filter
                       • Band-pass filter
                       • Homomorphic High-pass filter
                       • Noise Removal

                       + Threshold values for each filter
                       + Default threshold values


                   ? Miscellaneos Operations
                       • Histogram Equalization
                       • Contrast Stretching
                       • Robert Edge Detection
                       • Sobel Edge Detection
                       • Prewitt Edge Detecion
                       • Laplacian Edge Detection
                       • Canny Edge Detection
                       • Zerocross Edge Detection
                       • Laplacian of Gaussian  (LoG) Edge Detection
                       • Difference of Gaussian (DoG) Edge Detection
                       • Lossy JPEG Compression

%}



function varargout = IP_Project(varargin)

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @IP_Project_OpeningFcn, ...
                   'gui_OutputFcn',  @IP_Project_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before IP_Project is made visible.
function IP_Project_OpeningFcn(hObject, eventdata, handles, varargin)

% Choose default command line output for IP_Project
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = IP_Project_OutputFcn(hObject, eventdata, handles)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
%%%***** Main buttons functions *****%%%
% --------------------------------------------------------------------

% --- Executes on 'import button' press
function import_Callback(hObject, eventdata, handles)

global i
global filename


[filename, pathname] = uigetfile({
    '*.*', 'All files (*.*)';
    '*.jpg','jpg Files (*.jpg)';
    '*.JPG','JPG Files (*.JPG)';
    '*.png','png Files (*.png)';
    '*.PNG','PNG Files (*.PNG)';
    '*.jpeg','jpeg Files (*.jpeg)';
    '*.JPEG','JPEG Files (*.JPEG)';
    '*.img','img Files (*.img)';
    '*.IMG','IMG Files (*.IMG)';
    '*.tif','tif Files (*.tif)';
    '*.TIF','TIF Files (*.TIF)';
    '*.tiff','tiff Files (*.tiff)';
    '*.TIFF','TIFF Files (*.TIFF)'}, 'Select an image', 'C:\Users\admin\Desktop\Image Processing\Practical\Images');

% Error check - if no filename, there is an error
if isequal(filename,0)
    msgbox('Load Error. No files selected!', 'Error', 'error');
else
    i = imread(fullfile(pathname, filename));
    axes(handles.axes1);
    imshow(i);
end


% --- Executes on 'reset button' press
function reset_Callback(hObject, eventdata, handles)
clearAxis(handles.axes1);
clearAxis(handles.axes2);
clearAxis(handles.axes3);
clearAxis(handles.axes4);
clearAxis(handles.axes5);
clearAxis(handles.axes6);
clearAxis(handles.axes7);
clearAxis(handles.axes8);
clearAxis(handles.axes9);
clearAxis(handles.axes10);
clearAxis(handles.axes11);
clearAxis(handles.axes12);
clearAxis(handles.axes13);


set(handles.lp_threshold, 'String', 'Threshold');
set(handles.hp_threshold, 'String', 'Threshold');
set(handles.bp_threshold1, 'String', 'Threshold 1');
set(handles.bp_threshold2, 'String', 'Threshold 2');
set(handles.homo_threshold, 'String', 'Threshold');
set(handles.noise_threshold, 'String', 'Noise');
set(handles.filter_threshold, 'String', 'Filter');

set(handles.miscellaneous, 'String', 'Miscellaneous');

set(handles.time, 'String', '0.0');


% --- Executes on 'exit button' press.
function exit_Callback(hObject, eventdata, handles)

msgbox('Thank you for using image processing tool!');
pause(1);
close();
close();




% --------------------------------------------------------------------
%%% ***** Main operations function *****%%%
% --------------------------------------------------------------------

% --- Executes on 'apply button' press
function apply_Callback(hObject, eventdata, handles)

% Check if the user has input a threshold value

% Thresholds array
t = [handles.hp_threshold,    ...
     handles.lp_threshold,    ...
     handles.bp_threshold1,   ...
     handles.bp_threshold2,   ...
     handles.homo_threshold,  ...
     handles.noise_threshold, ...
     handles.filter_threshold];

isEmptyThreshold = '';

for i=1:length(t)
    value = str2double(get(t(i), 'String'));
    
    if isnan(value)
        isEmptyThreshold = 'Empty';
    end

end



% Apply the operations only if a threshold value is provided or in case of a default threshold
default_threshold = get(handles.default_thresholds, 'Value');


if default_threshold == 1

    % Noise threshold
    noise_threshold = 0.15;
    filter_threshold = 20;
    
    set(handles.noise_threshold, 'String', noise_threshold);
    set(handles.filter_threshold, 'String', filter_threshold);
    

    % Low-pass threshold
    lp_threshold = 35;
    set(handles.lp_threshold, 'String', lp_threshold);
    

    % High-pass threshold
    hp_threshold = 5;
    set(handles.hp_threshold, 'String', hp_threshold);
    

    % Band-pass threshold
    bp_threshold1 = 1;
    bp_threshold2 = 15;
    set(handles.bp_threshold1, 'String', bp_threshold1);
    set(handles.bp_threshold2, 'String', bp_threshold2);
    

    % Homomorphic threshold
    homo_threshold = 0.5;
    set(handles.homo_threshold, 'String', homo_threshold);
    
else
    
    % Noise threshold
    noise_threshold = str2double(get(t(6), 'String'));
    filter_threshold = str2double(get(t(7), 'String'));
   
    % Low-pass threshold
    lp_threshold = str2double(get(t(2), 'String'));
 
    % High-pass threshold
    hp_threshold = str2double(get(t(1), 'String'));
   
    % Band-pass threshold
    bp_threshold1 = str2double(get(t(3), 'String'));
    bp_threshold2 = str2double(get(t(4), 'String'));
   
    % Homomorphic threshold
    homo_threshold = str2double(get(t(5), 'String'));
    
end



% Apply the operations
if ~strcmp(isEmptyThreshold, 'Empty') || default_threshold ~= 0
    
    global i
    img = i;

    
    % Operations start time
    tStart = tic;

    
    % Noise removal (we use gaussian noise in our case)
    NF = NoiseRemoval(img, 'gaussian', noise_threshold, filter_threshold, handles);  % Note: noise type is changeable

    % Low-pass filtering
    LPF = lpfilter(img, lp_threshold, handles);
    lpfilter(img, lp_threshold, handles);

    % High-pass filtering
    HPF = hpfilter(LPF, hp_threshold, handles);
    lpfilter(img, hp_threshold, handles);

    % Band-pass filtering
    BPF = bpfilter(HPF, bp_threshold1, bp_threshold2, handles);
    bpfilter(img, bp_threshold1, bp_threshold2, handles);

    % Homomorphic filtering
    HMF = homomorphic(BPF, homo_threshold, handles);
    homomorphic(img, homo_threshold, handles);

    % Show final manipulated image
    Res = HMF;
    axes(handles.axes2);
    imshow(Res, []);


    % Operations end time
    tEnd = toc(tStart);

    % Display operations time
    set(handles.time, 'String', tEnd);
else 
  msgbox('A threshold value is required or check the default values!', 'Error', 'error');
end




% --------------------------------------------------------------------
%%% ***** Required functions *****%%%
% --------------------------------------------------------------------

% Noise removal function (using Butterworth LP filter)
function [out] = NoiseRemoval(input_image, type, noise_threshold, filter_threshold, handle)
a = handleRGB(input_image);

% define noise
noisy_image = imnoise(uint8(a), type, noise_threshold);

axes(handle.axes13);
imshow(noisy_image);

% Apply Fourier Transform, then shift to the origin
ft = fftshift(fft2(noisy_image));

% use butterworth low-pass filter
H = butterlp(ft, filter_threshold, 2);

conv = H .* ft;

% Apply Inverse Fourier Transform
out_im = ifft2(conv);

% display denoised image
ifftshow(out_im, handle.axes12);


% return manipulated image
out = out_im;

% Ideal Low-Pass Filter
function [out] = lpfilter(input_image, threshold, handle)

I = input_image;

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
    d0 = threshold;

    % Initialize ILPF
    idealLP = zeros(c,r);

    % Create ILPF
    for i=1:r
        for j=1:c
            D = sqrt((i-p).^2 + (j-q).^2);
            idealLP(i,j) = D <= d0;
        end
    end

    % Display ILPF
    axes(handle.axes6);
    imshow(idealLP);
    % 3D: meshc(idealLP);


    %// Change - To match size
    idealLP = imresize(idealLP, [r c]);


    % // Now filter
    FF_R = idealLP .* F_r; 
    FF_G = idealLP .* F_g; 
    FF_B = idealLP .* F_b; 


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


    % Combine the three components together
    %// Change - Removed fluff
    b = uint8(cat(3, Irr, Igg, Ibb));
    
    %// Visulaization - Display the final image in the spatial domain
    axes(handle.axes5);
    imshow(b, []);


    % return manipulated image
    out = b;
    
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
    d0 = threshold;

    % Initialize ILPF
    idealLP = zeros(c,r);

    % Create ILPF
    for i=1:r
        for j=1:c
            D = sqrt((i-p).^2 + (j-q).^2);
            idealLP(i,j) = D <= d0;
        end
    end

    
    %// Change - To match size
    idealLP = imresize(idealLP, [r c]);
    
    % Display the filter spectrum
    axes(handle.axes6);
    imshow(idealLP, []);
    
    
    % Convolve shifted image with ILPF
    convolveF = f_shift .* idealLP;

    % Shifted back the image
    original_image = ifftshift(convolveF);

    % Convert image to the spatial domain
    RImage = real(ifft2(original_image));

    % Display image in the spatial domain
    axes(handle.axes5);
    imshow(RImage, []);
    
    
    % return manipulated image
    out = RImage;
end


% Ideal High-Pass Filter
function [out] = hpfilter(input_image, threshold, handle)

I = input_image;

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
    d0 = threshold;

    % Initialize IHPF
    idealHP = zeros(c,r);

    % Create IHPF
    for i=1:r
        for j=1:c
            D = sqrt((i-p).^2 + (j-q).^2);
            idealHP(i,j) = D >= d0;
        end
    end
    
    
    %// Change - To match size
    idealHP = imresize(idealHP, [r c]);
    
    
    % Display IHPF
    axes(handle.axes4);
    imshow(idealHP);
    % 3D: meshc(idealHP);


    % // Now filter
    FF_R = idealHP .* F_r; 
    FF_G = idealHP .* F_g; 
    FF_B = idealHP .* F_b; 


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
    
    
    % Combine the three components together
    %// Change - Removed fluff
    b = uint8(cat(3, Irr, Igg, Ibb));
    
    %// Visulaization - Display the final image in spatial domain
    axes(handle.axes3);
    imshow(b, []);

    
    % return manipulated image
    out = b;
    
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
    d0 = threshold;

    % Initialize IHPF
    idealHP = zeros(c,r);

    % Create IHPF
    for i=1:r
        for j=1:c
            D = sqrt((i-p).^2 + (j-q).^2);
            idealHP(i,j) = D >= d0;
        end
    end

    
    %// Change - To match size
    idealHP = imresize(idealHP, [r c]);
    
    % display the filter spectrum
    axes(handle.axes4);
    imshow(idealHP, []);
    
    
    % Convolve shifted image with IHPF
    convolveF = f_shift .* idealHP;

    % Shifted back the image
    original_image = ifftshift(convolveF);

    % Convert image to the spatial domain
    RImage = real(ifft2(original_image));

    % Display image in the spatial domain
    axes(handle.axes3);
    imshow(RImage, []);
    
    
    % return manipulated image
    out = RImage;
end


% Bandpass Filter
function [out] = bpfilter(input_image, threshold1, threshold2, handle)

I = input_image;

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
    D01 = threshold1;
    D02 = threshold2;

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
    axes(handle.axes8);
    imshow(bandPass, []);
    % 3D: meshc(bandPass);


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
    
    
    % Combine the three components together
    %// Change - Removed fluff
    b = uint8(cat(3, Irr, Igg, Ibb));
    
    %// Visulaization - Display the final image in spatial domain
    axes(handle.axes7);
    imshow(b, []);

    % return manipulated image
    out = b;

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
    D01 = threshold1;
    D02 = threshold2;

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
    axes(handle.axes8);
    imshow(bandPass);
    
    
    % Convolve shifted image with IHPF
    convolveF = f_shift .* bandPass;

    % Shifted back the image
    original_image = ifftshift(convolveF);

    % Convert image to the spatial domain
    RImage = real(ifft2(original_image));

    % Display image in the spatial domain
    axes(handle.axes7);
    imshow(RImage, []);
    
    
    % return manipulated image
    out = RImage;
end


% Homomorphic Filter (using Butterworth HP filter)
function [out] = homomorphic(input_image, threshold, handle)

I = input_image;

% Convert image to double format
I = im2double(I);
    
    
%------------- If an RGB image -------------
if ndims(I) == 3
    
    % log of the image
    log_I = log(1 + I);


    %-- Homomorphic Filtering (Gaussian Highpass) --
    M = 2*size(log_I,1) + 1;
    N = 2*size(log_I,2) + 1;
    
    
    % Cut-off frequency (D0)
    D0 = threshold;
    
    
    % Gaussian Lowpass
    [X,Y] = meshgrid(1:N, 1:M);
    
    centerX = ceil(N/2);
    centerY = ceil(M/2);
    
    Duv = (X - centerX).^2 + (Y - centerY).^2; % D(u,v)
    H = exp( -Duv ./ (2 .* D0.^2) );
    
    
    % Gaussian Highpass = 1 - Gaussian Lowpass
    H = 1 - H;
    
    % Display the filter spectrum
    axes(handle.axes10);
    imshow(H, []);
    
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
    axes(handle.axes9);
    imshow(Ihmf, []);
    
    
    % return manipulated image
    out = Ihmf;
    
%------------- If not an RGB image -------------
else 
    
    % log of the image
    log_I = log(1 + I);

    
    %-- Homomorphic Filtering (Gaussian Highpass) --
    [M,N] = size(log_I); % dimensions
    
    % Cut-off frequency
    D0 = threshold;
    
    % Gaussian Lowpass
    [X,Y] = meshgrid(1:N, 1:M);
    
    centerX = ceil(N/2);
    centerY = ceil(M/2);
    
    Duv = (X - centerX).^2 + (Y - centerY).^2; % D(u,v)
    H = exp( -Duv ./ (2 .* D0.^2) );
    
    
    % Gaussian Highpass = 1 - Gaussian Lowpass
    H = 1 - H;
    
    % Display the filter spectrum
    axes(handle.axes10);
    imshow(H, []);
    
    
    % // Change - Shift the filter to the center
    H = fftshift(H);
    
    If = fft2(log_I, M, N); % fourier transform
    
    % Convolve shifted image with the filter
    convolveF = If .* H;
    
    
    %// Change - Apply the filter (using repmat), ifftshift, then cast to real
    Iout = real(ifft2(convolveF));

    
    % Inverse log 
    Ihmf = exp(Iout) - 1;
    
    %// Display the final image in spatial domain
    axes(handle.axes9);
    imshow(Ihmf, []);
   
    
    % return manipulated image
    out = Ihmf;
end


% Butterworth High-Pass Filter
function [out] = butterhp(im, d, n)
h = size(im,2);
w = size(im,1);
[u,v] = meshgrid(-floor(w/2):floor((w-1)/2), -floor(h/2):floor((h-1)/2));
out = 1./(1.+(d./sqrt(u.^2+v.^2)).^2*n);

% Butterworth Low-Pass Filter
function [out] = butterlp(im, d, n)
out = 1 - butterhp(im, d, n);




% function to handle RGB images
function [out] = handleRGB(input_image)
a = input_image;

% Handle RGB images
if ndims(a) == 3
    a = rgb2gray(a);
end

a = imresize(a, [256,256]); % resize the image to whatever size you like

% return the result
out = a;

% function to clear axis
function clearAxis(h)
  axesHandlesToChildObjects = findobj(h, 'Type', 'image');
  if ~isempty(axesHandlesToChildObjects)
    delete(axesHandlesToChildObjects);
  end
  return;

% function to display fourier transform spectrum
function [] = fftshow(f, handle)
f1 = log(1+abs(f));
fm = max(f1(:));
axes(handle);
imshow(im2uint8(f1/fm));

% function to display inverse fourier transform spectrum
function [] = ifftshow(f, handle)
f1 = abs(f);
fm = max(f1(:));
axes(handle);
imshow(f1/fm);




% --------------------------------------------------------------------
%%%***** Miscellaneous Operations function *****%%%
% --------------------------------------------------------------------

% --- Executes on selection change in popupmenu1
function popupmenu1_Callback(hObject, eventdata, handles)
global i;
img = i;

axes(handles.axes11);

% Get currently selected option from menu
option = get(handles.popupmenu1, 'Value');
choice = handles.popupmenu1.String{option};

% Operation start time
tStart = tic;

switch option
    % Histogram Equalization
    case 2
        histogram_equalization(img);
        set(handles.miscellaneous, 'String', choice);
        msgbox('Histogram Equalization successfully applied!');

    % Contrast Stretching
    case 3
        contrast_stretching(img);
        set(handles.miscellaneous, 'String', choice);
        msgbox('Contrast Stretching successfully applied!');

    % Edge Detection using 'Robert Operator' detector
    case 4
        RobertDetector(img);
        set(handles.miscellaneous, 'String', choice);
        msgbox('Robert Edge Detection successfully applied!');
        
        % Built-in:  edge(img, 'Roberts')
        
    % Edge Detection using 'Sobel Operator' detector
    case 5
        SobelDetector(img);
        set(handles.miscellaneous, 'String', choice);
        msgbox('Sobel Edge Detection successfully applied!');
        
        % Built-in:  edge(img, 'Sobel')
        
    % Edge Detection using 'Prewitt Operator' Detector
    case 6
        PrewittDetector(img);
        set(handles.miscellaneous, 'String', choice);
        msgbox('Prewitt Edge Detecion successfully applied!');
        
        % Built-in:  edge(img, 'Prewitt')
        
    % Edge Detection using 'Laplacian' Detector
    case 7
        LaplacianDetector(img);
        set(handles.miscellaneous, 'String', choice);
        msgbox('Laplacian Edge Detection successfully applied!');
        
    % Edge Detection using 'Canny Operator' Detector
    case 8
        CannyDetector(img);
        set(handles.miscellaneous, 'String', choice);
        msgbox('Canny Edge Detection successfully applied!');
        
        % Built-in:  edge(img, 'Canny')
        
    % Edge Detection using 'Zerocross' Detector
    case 9
        % Handle RGB images
        new_img = handleRGB(img);
        
        % Convert image into double format
        new_img = im2double(new_img);
        
        % Apply a zerocross edge detection
        Zerocross_Detection = edge(new_img, 'zerocross', 0.002);
        
        set(handles.miscellaneous, 'String', choice);
        
        % Display output image
        imshow(Zerocross_Detection, []);
        
        msgbox('Zerocross Edge Detection successfully applied!');
        
    % Edge Detection using 'LoG' Detector
    case 10
        % Handle RGB images
        new_img = handleRGB(img);
        
        % Convert image into double format
        new_img = im2double(new_img);
        
        % Apply a LoG edge detection
        Log_Detection = edge(new_img, 'log', 0.002);
        
        % Display output image
        imshow(Log_Detection, []);
        
        set(handles.miscellaneous, 'String', choice);
        
        msgbox('LoG Edge Detection successfully applied!');
        
    % Edge Detection using 'DoG' Detector
    case 11
        DoG_Detector(img);
        set(handles.miscellaneous, 'String', choice);
        msgbox('DoG Edge Detection successfully applied!');
        
    % Median filter
    case 12
        
        % Apply median filter
        medianFilt(img, handles);

        set(handles.miscellaneous, 'String', choice);
        
        msgbox('Median filter successfully applied!');
        
    % Lossy JPEG Image Compression
    case 13
        global filename;
        
        % valid extensions
        valid_extensions = {'jpg', 'jpeg', 'jfif', 'pjpeg', 'pjp'};

        % read image info
        info = imfinfo(filename);

        % check image extension format
        isValid = false;

        for j=1:size(valid_extensions)
           if strcmp(info.Format, valid_extensions(1, j))
               isValid = true;
           end
        end


        % if image format is accepted
        if isValid
            set(handles.miscellaneous, 'String', choice);
            msgbox('Applying Lossy JPEG Image compression, please wait!', 'JPEG Compression', 'warn');
            dct1(img, 8, 8, handles.axes11);
            
        % if not
        else
            msgbox('Only JPEG image formats are accepted!', 'error');
        end

    % Histogram Plotting
    case 14
        img = i;
        
        % Histogram of RGB images
        if ndims(img) == 3
            % Red component histogram
            red_hist = imhist(img(:,:,1));
            figure, plot(red_hist, 'red');

            % Green component histogram
            green_hist = imhist(img(:,:,2));
            figure, plot(green_hist, 'green');

            % Blue component histogram
            blue_hist = imhist(img(:,:,3));
            figure, plot(blue_hist, 'blue');
            
        % Histogram of grayscale images
        else
            im_hist = imhist(img);
            figure, plot(im_hist);
        end
        
    otherwise
        msgbox('Please, select an option!');
end

% Operation end time
tEnd = toc(tStart);

% Display operation time
set(handles.time, 'String', tEnd);



% Color RGB images - Histogram Equalization
function [] = histogram_equalization(input_image)
im = input_image;

% If RGB image
if ndims(im) == 3
    % Convert the RGB image into HSV image format
    HSV = rgb2hsv(im);

    % Perform Histogram Equalization on intensity component
    Heq = histeq(HSV(:,:,3));

    HSV_mod = HSV;
    HSV_mod(:,:,3) = Heq;

    % Convert the HSV image back into RGB
    im = hsv2rgb(HSV_mod);
% if grayscale image
else
    im = histeq(uint8(im));
    
end

% display equalized image
imshow(im);


% Contrast/Histogram Stretching function
function [] = contrast_stretching(input_image)
i = input_image;

if isa(i, 'logical')
    i = double(i);
else
    i = uint8(i);
end

j = imadjust(i, [80/255 200/255], [0 1]);

imshow(j);

% Roberts edge detector function
function [] = RobertDetector(input_image)
a = handleRGB(input_image);

% Convert image into double format
a = im2double(a);

% Roberts Operator Masks
hx = [+1 0; 0 -1];
hy = [0 +1; -1 0];

% Compute Gx and Gy
Gx = imfilter(a, hx);
Gy = imfilter(a, hy);

% Calculate the magnitude
Gxy = sqrt(Gx.^2 + Gy.^2);

% Display output image
imshow(Gxy);

% Sobel edge detector function
function [] = SobelDetector(input_image)
a = handleRGB(input_image);

% Convert image into double format
a = im2double(a);

% Sobel Operator Masks
hx = [ 1,  2,  1;
       0,  0,  0;
      -1, -2, -1];
  
hy = [ 1,  0,  -1;
       2,  0,  -2;
       1,  0,  -1];

% Compute Gx and Gy
Gx = imfilter(a, hx);
Gy = imfilter(a, hy);

% Calculate the magnitude
Gxy = sqrt(Gx.^2 + Gy.^2);

% Display output image
imshow(Gxy);

% Prewitt edge detector function
function [] = PrewittDetector(input_image)
a = handleRGB(input_image);

% Convert image into double format
a = im2double(a);

% Prewitt Operator Masks
hx = [-1  0  1;
      -1  0  1;
      -1  0  1];
  
hy = [-1  -1  -1;
       0   0   0;
       1   1   1];

   
% Compute Gx and Gy
Gx = imfilter(a, hx);
Gy = imfilter(a, hy);

% Calculate the magnitude
Gxy = sqrt(Gx.^2 + Gy.^2);


% Display output image
imshow(Gxy);

% Laplacian edge detector function
function [] = LaplacianDetector(input_image)
a = handleRGB(input_image);

% Define the Laplacian filter mask.
Laplacian = [0 1 0; 1 -4 1; 0 1 0];

% Convolve the image using Laplacian Filter
k1 = conv2(double(a), Laplacian, 'same');

% Display the output image
imshow(abs(k1), []);

% Canny edge detector function
function [] = CannyDetector(input_image)
a = handleRGB(input_image);

% Convert the image into double
a = double(a);

% Value for Thresholding
T_Low = 0.075;
T_High = 0.175;

% Gaussian Filter Coefficient
B = [2, 4, 5, 4, 2; 4, 9, 12, 9, 4;5, 12, 15, 12, 5;4, 9, 12, 9, 4;2, 4, 5, 4, 2 ];
B = 1/159.* B;

% Convolution of image by Gaussian Coefficient
A = conv2(a, B, 'same');

% Filter for horizontal and vertical direction
KGx = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
KGy = [1, 2, 1; 0, 0, 0; -1, -2, -1];

% Convolution by image by horizontal and vertical filter
Filtered_X = conv2(A, KGx, 'same');
Filtered_Y = conv2(A, KGy, 'same');

% Calculate directions/orientations
arah = atan2(Filtered_Y, Filtered_X);
arah = arah*180/pi;

pan = size(A,1);
leb = size(A,2);

% Adjustment for negative directions, making all directions positive
for i=1:pan
    for j=1:leb
        if (arah(i,j)<0) 
            arah(i,j)=360+arah(i,j);
        end;
    end;
end;

arah2 = zeros(pan, leb);

% Adjusting directions to nearest 0, 45, 90, or 135 degree
for i = 1  : pan
    for j = 1 : leb
        if ((arah(i, j) >= 0 ) && (arah(i, j) < 22.5) || (arah(i, j) >= 157.5) && (arah(i, j) < 202.5) || (arah(i, j) >= 337.5) && (arah(i, j) <= 360))
            arah2(i, j) = 0;
        elseif ((arah(i, j) >= 22.5) && (arah(i, j) < 67.5) || (arah(i, j) >= 202.5) && (arah(i, j) < 247.5))
            arah2(i, j) = 45;
        elseif ((arah(i, j) >= 67.5 && arah(i, j) < 112.5) || (arah(i, j) >= 247.5 && arah(i, j) < 292.5))
            arah2(i, j) = 90;
        elseif ((arah(i, j) >= 112.5 && arah(i, j) < 157.5) || (arah(i, j) >= 292.5 && arah(i, j) < 337.5))
            arah2(i, j) = 135;
        end;
    end;
end;

% Calculate magnitude
magnitude = (Filtered_X.^2) + (Filtered_Y.^2);
magnitude2 = sqrt(magnitude);

BW = zeros(pan, leb);

% Non-Maximum Supression
for i=2:pan-1
    for j=2:leb-1
        if (arah2(i,j)==0)
            BW(i,j) = (magnitude2(i,j) == max([magnitude2(i,j), magnitude2(i,j+1), magnitude2(i,j-1)]));
        elseif (arah2(i,j)==45)
            BW(i,j) = (magnitude2(i,j) == max([magnitude2(i,j), magnitude2(i+1,j-1), magnitude2(i-1,j+1)]));
        elseif (arah2(i,j)==90)
            BW(i,j) = (magnitude2(i,j) == max([magnitude2(i,j), magnitude2(i+1,j), magnitude2(i-1,j)]));
        elseif (arah2(i,j)==135)
            BW(i,j) = (magnitude2(i,j) == max([magnitude2(i,j), magnitude2(i+1,j+1), magnitude2(i-1,j-1)]));
        end;
    end;
end;

BW = BW .* magnitude2;

% Hysteresis Thresholding
T_Low = T_Low * max(max(BW));
T_High = T_High * max(max(BW));

T_res = zeros(pan, leb);

for i = 1  : pan
    for j = 1 : leb
        if (BW(i, j) < T_Low)
            T_res(i, j) = 0;
        elseif (BW(i, j) > T_High)
            T_res(i, j) = 1;
        % Using 8-connected components
        elseif ( BW(i+1,j)>T_High || BW(i-1,j)>T_High || BW(i,j+1)>T_High || BW(i,j-1)>T_High || BW(i-1, j-1)>T_High || BW(i-1, j+1)>T_High || BW(i+1, j+1)>T_High || BW(i+1, j-1)>T_High)
            T_res(i,j) = 1;
        end;
    end;
end;

edge_final = uint8(T_res.*255);


% Show final edge detection result
imshow(edge_final);

% DoG edge detector function
function [] = DoG_Detector(input_image)
a = input_image;

% Convert image into double format
a = im2double(a(:,:,1));

% filter the image using DoG filter
H1 = fspecial('gaussian', 21, 15);
H2 = fspecial('gaussian', 21, 20);

% DoG filter
DoG = H1 - H2;

dogFilterImage = conv2(a, DoG, 'same');

% Display output image
imshow(dogFilterImage, []);

function [] = medianFilt(input_image, handle)

I = input_image;

%------------- If an RGB image ---------------
if ndims(I) == 3

% Extract the individual red, green, and blue color channels.
redChannel   = I(:, :, 1);
greenChannel = I(:, :, 2);
blueChannel  = I(:, :, 3);


% Median Filter the channels:
redMF   = medfilt2(redChannel,   [3 3]);
greenMF = medfilt2(greenChannel, [3 3]);
blueMF  = medfilt2(blueChannel,  [3 3]);


% Reconstruct the noise free RGB image
rgbFixed = cat(3, redMF, greenMF, blueMF);


% Display the image
axes(handle.axes11);
imshow(rgbFixed, []);


%------------- If not an RGB image -------------
else
    
    % Handle RGB images
    new_img = handleRGB(I);

    
    % Convert image into double format
    new_img = im2double(new_img);

    
    % Apply median filter
    med_filter = medfilt2(new_img, [3 3]);
    
    
    % Display output image
    axes(handle.axes11);
    imshow(med_filter, []);
end

% Lossy JPEG image compression
function[] = dct1(input, n, m, handle)
% "input": input image.
% "n": denotes the number of bits per pixel.
% "m": denotes the number of most significant bits (MSB) of DCT Coefficients. 

% Matrix Intializations.
N = 8;                        % Block size for which DCT is Computed.
M = 8;

I = input;                    % Reading the input image and storing intensity values in 2-D matrix I.
I_dim = size(I);              % Finding the dimensions of the image file.
I_Trsfrm.block = zeros(N,M);  % Initializing the DCT Coefficients Structure Matrix "I_Trsfrm" with the required dimensions.

Norm_Mat = [16  11  10  16  24  40  51  61       % Normalization matrix (8 X 8) used to Normalize the DCT Matrix.
           12  12  14  19  26  58  60  55
           14  13  16  24  40  57  69  56
           14  17  22  29  51  87  80  62
           18  22  37  56  68  109 103 77
           24  35  55  64  81  104 113 92
           49  64  78  87  103 121 120 101
           72  92  95  98  112 100 103 99];


%---------------------------------------------------%
%********** PART-1: COMPRESSION TECHNIQUE **********%
%---------------------------------------------------%

% Computing the Quantized & Normalized Discrete Cosine Transform.

% Y(k,l)=(2/root(NM))*c(k)*c(l)*sigma(i=0:N-1)sigma(j=0:M-1)y(i,j)cos(pi(2i+1)k/(2N))cos(pi(2j+1)l/(2M))
% where c(u) = 1/root(2), if u=0
%            = 1        , if u>0

for a=1:I_dim(1)/N
    for b=1:I_dim(2)/M
        for k=1:N
            for l=1:M
                prod=0;
                for i=1:N
                    for j=1:M
                        prod=prod+double(I(N*(a-1)+i,M*(b-1)+j))*cos(pi*(k-1)*(2*i-1)/(2*N))*cos(pi*(l-1)*(2*j-1)/(2*M));
                    end
                end
                if k == 1
                    prod = prod * sqrt(1/N);
                else
                    prod = prod * sqrt(2/N);
                end
                if l == 1
                    prod = prod * sqrt(1/M);
                else
                    prod = prod * sqrt(2/M);
                end
                I_Trsfrm(a,b).block(k,l) = prod;
            end
        end
        % Normalizing the DCT Matrix and Quantizing the resulting values.
        I_Trsfrm(a,b).block = round(I_Trsfrm(a,b).block ./ Norm_Mat);
    end
end

% zig-zag coding of the each 8x8 Block.
for a=1:I_dim(1)/N
    for b=1:I_dim(2)/M
        I_zigzag(a,b).block = zeros(1,0);
        freq_sum=2:(N+M);
        counter = 1;
        for i=1:length(freq_sum)
            if i<=((length(freq_sum)+1)/2)
                if rem(i,2)~=0
                    x_indices=counter:freq_sum(i)-counter;
                else
                    x_indices=freq_sum(i)-counter:-1:counter;
                end
                    index_len=length(x_indices);
                    y_indices=x_indices(index_len:-1:1); % Creating reverse of the array as "y_indices".
                    for p=1:index_len
                        if I_Trsfrm(a,b).block(x_indices(p),y_indices(p))<0
                            bin_eq = dec2bin(bitxor(2^n-1,abs(I_Trsfrm(a,b).block(x_indices(p),y_indices(p)))),n);
                        else
                            bin_eq = dec2bin(I_Trsfrm(a,b).block(x_indices(p),y_indices(p)),n);
                        end
                        I_zigzag(a,b).block = [I_zigzag(a,b).block, bin_eq(1:m)];
                    end
            else
                counter=counter+1;
                if rem(i,2)~=0
                    x_indices=counter:freq_sum(i)-counter;
                else
                    x_indices=freq_sum(i)-counter:-1:counter;
                end
                    index_len=length(x_indices);
                    y_indices=x_indices(index_len:-1:1); % Creating reverse of the array as "y_indices".
                    for p=1:index_len
                        if I_Trsfrm(a,b).block(x_indices(p),y_indices(p))<0
                            bin_eq=dec2bin(bitxor(2^n-1,abs(I_Trsfrm(a,b).block(x_indices(p),y_indices(p)))),n);
                        else
                            bin_eq=dec2bin(I_Trsfrm(a,b).block(x_indices(p),y_indices(p)),n);
                        end
                        I_zigzag(a,b).block=[I_zigzag(a,b).block,bin_eq(1:m)];
                    end
            end
        end
    end
end

% Clearing unused variables from Memory space
clear I_Trsfrm prod; 
clear x_indices y_indices counter;

% Run-Length Encoding the resulting code.
for a=1:I_dim(1)/N
    for b=1:I_dim(2)/M
        
        % Computing the Count values for the corresponding symbols and
        % savin them in "I_run" structure.
        count=0;
        run=zeros(1,0);
        sym=I_zigzag(a,b).block(1);
        j=1;
        block_len=length(I_zigzag(a,b).block);
        for i=1:block_len
            if I_zigzag(a,b).block(i)==sym
                count=count+1;
            else
                run.count(j)=count;
                run.sym(j)=sym;
                j=j+1;
                sym=I_zigzag(a,b).block(i);
                count=1;
            end
            if i==block_len
                run.count(j)=count;
                run.sym(j)=sym;
            end
        end 
        
        % Computing the codelength needed for the count values.
        dim        = length(run.count);    % calculates number of symbols being encoded.
        maxvalue   = max(run.count);       % finds the maximum count value in the count array of run structure.
        codelength = log2(maxvalue) + 1;
        codelength = floor(codelength);
        
        % Encoding the count values along with their symbols.
        I_runcode(a,b).code = zeros(1,0);
        for i=1:dim
            I_runcode(a,b).code = [I_runcode(a,b).code, dec2bin(run.count(i),codelength), run.sym(i)];
        end
    end
end

% Clearing unused variables from Memory Space.
clear I_zigzag run;



%-----------------------------------------------------%
%********** PART-2: DECOMPRESSION TECHNIQUE **********%
%-----------------------------------------------------%

% Run-Length Decoding of the compressed image.
for a=1:I_dim(1)/N
    for b=1:I_dim(2)/M
        enc_str=I_runcode(a,b).code;
        
        % Computing the length of the encoded string.
        enc_len=length(enc_str);
        
        % Since Max. Count is unknown at the receiver, Number of bits used for each 
        % count value is unknown and hence cannot be decoded directly. Number of bits 
        % used for each count can be found out by trial and error method for all 
        % the possible lengths => factors of encoded string length.

        % Computing the non-trivial factors of the "enc_len" (length of encoded
        % string) i.e., factors other than 1 & itself.
        factors_mat=zeros(1,0);
        if enc_len<=(n+1)
            realfact=enc_len;
        else
            for i=2:enc_len-2       % "enc_len-1" is always not a divisor of "enc_len".
                if(rem(enc_len,i)==0)
                    factors_mat=[factors_mat,i];
                end
            end

            % Trial and Error Method to Find the Exact count value.
            for i=1:length(factors_mat)
                flagcntr=0;
                temp_dim=enc_len/factors_mat(i);
                for j=1:temp_dim
                    if strcmp(enc_str(1+(j-1)*factors_mat(i):j*factors_mat(i)),dec2bin(0,factors_mat(i)))==0
                        if j==1
                            flagcntr=flagcntr+1;
                        else
                            if enc_str((j-1)*factors_mat(i))~=enc_str(j*factors_mat(i))
                                flagcntr=flagcntr+1;
                            else
                                break;
                            end
                        end
                    else
                        break;
                    end
                end
                if flagcntr==temp_dim
                    realfact=factors_mat(i);
                    break;
                end
            end
        end
        
        % Clearing unused variables from Memory space
        clear factors_mat flagcntr j 

        % Finding out the count values of corresponding symbols in the encoded
        % string and then decoding it accordingly.
        dec_str=zeros(1,0);
        temp_dim=enc_len/realfact;
        for i=1:temp_dim
            count_str=enc_str(1+(i-1)*realfact:(i*realfact)-1);
            countval=bin2dec(count_str);
            for j=1:countval
                dec_str=[dec_str,enc_str(i*realfact)];
            end
        end
        I_runcode(a,b).code=dec_str;
    end
end

% Clearing unused variables from Memory space
clear enc_str dec_str temp_dim realfact enc_len
clear countval count_str

% Reconstructing the 8x8 blocks in Zig-Zag fashion.
I_rec_Trnsfm.block = zeros(N,M);
for a=1:I_dim(1)/N
    for b=1:I_dim(2)/M
        bpp=length(I_runcode(a,b).code)/(N*M);  % "bpp" is the bits-per-pixel in reconstruction of image.
        bpp_diff=n-bpp; 
        freq_sum=2:(N+M);
        counter=1;
        c_indx=1;
        for i=1:length(freq_sum)
            if i<=((length(freq_sum)+1)/2)
                if rem(i,2)~=0
                    x_indices=counter:freq_sum(i)-counter;
                else
                    x_indices=freq_sum(i)-counter:-1:counter;
                end
                    index_len=length(x_indices);
                    y_indices=x_indices(index_len:-1:1); % Creating reverse of the array as "y_indices".
                    for p=1:index_len
                        decm_eq=bin2dec([I_runcode(a,b).code(1+m*(c_indx-1):m*c_indx),dec2bin(0,bpp_diff)]);
                        if decm_eq>(2^(n-1))-1
                            decm_eq=decm_eq-(2^n-1);
                        end
                        I_rec_Trnsfm(a,b).block(x_indices(p),y_indices(p))=decm_eq;
                       c_indx=c_indx+1;
                    end
            else
                counter=counter+1;
                if rem(i,2)~=0
                    x_indices=counter:freq_sum(i)-counter;
                else
                    x_indices=freq_sum(i)-counter:-1:counter;
                end
                    index_len=length(x_indices);
                    y_indices=x_indices(index_len:-1:1); % Creating reverse of the array as "y_indices".
                    for p=1:index_len
                        decm_eq=bin2dec([I_runcode(a,b).code(1+m*(c_indx-1):m*c_indx),dec2bin(0,bpp_diff)]);
                        if decm_eq>(2^(n-1))-1
                            decm_eq=decm_eq-(2^n-1);
                        end
                        I_rec_Trnsfm(a,b).block(x_indices(p),y_indices(p))=decm_eq;
                        c_indx=c_indx+1;
                    end
            end
        end
    end
end

% Clearing unused variables from Memory space
clear I_runcode x_indices y_indices
clear c_indx freq_sum

% Denormalizing the Reconstructed Tranform matrix using the same
% Normalization matrix.
for a=1:I_dim(1)/N
    for b=1:I_dim(2)/M
        I_rec_Trnsfm(a,b).block=(I_rec_Trnsfm(a,b).block).*Norm_Mat;
    end
end


% Inverse-Discrete Cosine Transform on the reconstructed Matrix.
% y(i,j)=(2/root(NM))*sigma(i=0:N-1)sigma(j=0:M-1) Y(k,l)c(k)*c(l)*cos(pi(2i+1)k/(2N))cos(pi(2j+1)l/(2M))
% where c(u)=1/root(2) if u=0
%            = 1       if u>0
for a=1:I_dim(1)/N
    for b=1:I_dim(2)/M
        for i=1:N
            for j=1:M
                prod=0;
                for k=1:N
                    for l=1:M
                        if k==1
                           temp=double(sqrt(1/2)*I_rec_Trnsfm(a,b).block(k,l))*cos(pi*(k-1)*(2*i-1)/(2*N))*cos(pi*(l-1)*(2*j-1)/(2*M));
                        else
                            temp=double(I_rec_Trnsfm(a,b).block(k,l))*cos(pi*(k-1)*(2*i-1)/(2*N))*cos(pi*(l-1)*(2*j-1)/(2*M));
                        end
                        if l==1
                            temp=temp*sqrt(1/2);
                        end
                        prod=prod+temp;
                    end
                end
                prod=prod*(2/sqrt(M*N));
                I_rec((a-1)*N+i,(b-1)*M+j)=prod;
            end
        end
    end
end

% Clearing unused variables from Memory Space.
clear I_rec_Trnsfm

% Displaying the Reconstructed Image.
I_rec = I_rec / max(max(I_rec));
I_rec = im2uint8(I_rec);

axes(handle);
imshow(I_rec, [0,2^n-1]);
msgbox('Lossy JPEG Image compression successfully applied!', 'success');

























% --- Executes during object creation, after setting all properties
function popupmenu1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function hp_threshold_Callback(hObject, eventdata, handles)
% hObject    handle to hp_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of hp_threshold as text
%        str2double(get(hObject,'String')) returns contents of hp_threshold as a double


% --- Executes during object creation, after setting all properties.
function hp_threshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hp_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function lp_threshold_Callback(hObject, eventdata, handles)
% hObject    handle to lp_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of lp_threshold as text
%        str2double(get(hObject,'String')) returns contents of lp_threshold as a double



% --- Executes during object creation, after setting all properties.
function lp_threshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lp_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function bp_threshold1_Callback(hObject, eventdata, handles)
% hObject    handle to bp_threshold1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bp_threshold1 as text
%        str2double(get(hObject,'String')) returns contents of bp_threshold1 as a double


% --- Executes during object creation, after setting all properties.
function bp_threshold1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bp_threshold1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function homo_threshold_Callback(hObject, eventdata, handles)
% hObject    handle to homo_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of homo_threshold as text
%        str2double(get(hObject,'String')) returns contents of homo_threshold as a double


% --- Executes during object creation, after setting all properties.
function homo_threshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to homo_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function noise_threshold_Callback(hObject, eventdata, handles)
% hObject    handle to noise_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of noise_threshold as text
%        str2double(get(hObject,'String')) returns contents of noise_threshold as a double


% --- Executes during object creation, after setting all properties.
function noise_threshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to noise_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function bp_threshold2_Callback(hObject, eventdata, handles)
% hObject    handle to bp_threshold2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bp_threshold2 as text
%        str2double(get(hObject,'String')) returns contents of bp_threshold2 as a double


% --- Executes during object creation, after setting all properties.
function bp_threshold2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bp_threshold2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function filter_threshold_Callback(hObject, eventdata, handles)
% hObject    handle to filter_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of filter_threshold as text
%        str2double(get(hObject,'String')) returns contents of filter_threshold as a double


% --- Executes during object creation, after setting all properties.
function filter_threshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to filter_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in default_thresholds.
function default_thresholds_Callback(hObject, eventdata, handles)
% hObject    handle to default_thresholds (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of default_thresholds
