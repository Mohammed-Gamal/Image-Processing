[u,v] = meshgrid(-128:127,-128:127);
duv = sqrt(u.^2 + v.^2);
D0 = duv < 100;

a = imread('cameraman.tif');
subplot(2,2,1); imshow(a);

% Fourier transform
aFreq = fftshift(fft2(a));  % shift zero-frequency component to the origin
subplot(2,2,2); fftshow(aFreq);

% Filter Applying
lp = aFreq .* D0;
subplot(2,2,3); fftshow(lp);

% Inverse fourier transform
lpi = ifft2(lp);
subplot(2,2,4); ifftshow(lpi);
