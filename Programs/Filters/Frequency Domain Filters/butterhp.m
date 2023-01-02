function [ out ] = butterhp(im, d, n)
h = size(im,2);  % width
w = size(im,1);  % height
[u,v] = meshgrid(-floor(w/2):floor((w-1)/2), -floor(h/2):floor((h-1)/2));
out = 1./(1.+(d./sqrt(u.^2+v.^2)).^2*n);
end

