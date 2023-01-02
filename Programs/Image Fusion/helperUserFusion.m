function c = helperUserFusion(A,B)

% create an upper triangular logical array the same size as A.
d = logical(triu(ones(size(A))));

% set a threshold
t = 0.3;

c = A;

% set the upper triangular portion of the output to a blend of A and B
c(d) = t*A(d) + (1-t)*B(d);

% set the lower triangular portion of the output to a different blend of A and B
c(~d) = t*B(~d) + (1-t)*A(~d);

end