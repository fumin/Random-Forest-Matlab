function [ line ] = sample_hyperplane( X )
dimension = size(X, 2);
if size(X, 1) == 1
    line = rand(dimension, 1);
    return
end
mins = min(X); maxs = max(X);

points = zeros(dimension);
for i = 1:dimension
    points(:, i) = rand(dimension, 1) * (maxs(i) - mins(i)) + mins(i);
end
warning off all
line = points \ ones(dimension, 1);
warning on all
if isnan(line)
    line = rand(dimension, 1);
end

%plot(points(:, 1), points(:, 2), 'r+');
%hold on
%ys = (1 - line(1) * (mins(1):maxs(1))) / line(2);
%plot((mins(1):maxs(1)), ys, 'bo');

end
