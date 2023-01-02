[u,v] = meshgrid(-128:127, -128:127);
duv = sqrt(u.^2 + v.^2);  % duv ? d(u,v)= sqrt(u^2 + v^2)
D0 = duv > 100;

% input image
a = imread('cameraman.tif');
subplot(2,2,1); imshow(a);

% fourier transform
aFreq = fftshift(fft2(a));  % shift zero-frequency component to the origin
subplot(2,2,2); fftshow(aFreq);

% filter applying
hp = aFreq .* D0;
subplot(2,2,3); fftshow(hp);

% inverse fourier transform
aFreqi = ifft2(hp);
subplot(2,2,4); ifftshow(aFreqi)
