% input image
i = imread('cameraman.tif');
subplot(1,3,1); imshow(i);

% bandpass filter
[u,v] = meshgrid(-128:127, -128:127);
duv = sqrt(u.^2 + v.^2);

D01 = duv <= 100;
D02 = duv >= 50;

D0 = D01 .* D02;


% fourier transform
aFreq = fftshift(fft2(i));

% filter applying
bp = aFreq .* D0;
subplot(1,3,2); fftshow(bp)

% inverse fourier transform
out = ifft2(bp);

subplot(1,3,3); ifftshow(out);
