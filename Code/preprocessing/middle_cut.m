function [ min_z, max_z ] = middle_cut(z)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

[N, edges] = histcounts(z);
inds = round(edges(N > 35));

max_z = max(inds);
min_z = min(inds);

end

