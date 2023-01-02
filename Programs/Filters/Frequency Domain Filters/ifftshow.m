function [] = ifftshow(f)
f1 = abs(f);
fm = max(f1(:));
imshow(f1/fm);
end

