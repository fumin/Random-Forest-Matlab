data = load('../data/iris_sepal_length_sepal_width');
X = data(:, 2:end);
Y = data(:, 1);
%scatter(X(:,1), X(:,2), 10, Y);

opts = struct;
opts.depth = 2;
opts.numTrees = 500;
opts.numSplits = 5;
opts.verbose = true;
opts.classifierID = 2;

tic;
m = forestTrain(X, Y, opts);
timetrain = toc;
tic;
yhatTrain = forestTest(m, X);
timetest = toc;
fprintf('tree depth = %d, reverse-randomness = %d\n', opts.depth, opts.numSplits);
fprintf('Training accuracy = %.2f\n', mean(yhatTrain == Y));

% Look at classifier distribution for fun, to see what classifiers were
% chosen at split nodes and how often
fprintf('Classifier distributions:\n');
classifierDist= zeros(1, 4);
unused= 0;
for i=1:length(m.treeModels)
    for j=1:length(m.treeModels{i}.weakModels)
        cc= m.treeModels{i}.weakModels{j}.classifierID;
        if cc>1 %otherwise no classifier was used at that node
            classifierDist(cc)= classifierDist(cc) + 1;
        else
            unused= unused+1;
        end
    end
end
fprintf('%d nodes were empty and had no classifier.\n', unused);
for i=1:4
    fprintf('Classifier with id=%d was used at %d nodes.\n', i, classifierDist(i));
end

%% plot results
real_xrange = [min(X(:, 1)), max(X(:, 1))];
xrange_len = real_xrange(2) - real_xrange(1);
real_yrange = [min(X(:, 2)), max(X(:, 2))];
yrange_len = real_yrange(2) - real_yrange(1);
xrange = [real_xrange(1)-xrange_len*0.2 real_xrange(2)+xrange_len*0.2];
yrange = [real_yrange(1)-yrange_len*0.2 real_yrange(2)+yrange_len*0.2];
xinc = (xrange(2) - xrange(1)) / 75;
yinc = (yrange(2) - yrange(1)) / 75;
[x, y] = meshgrid(xrange(1):xinc:xrange(2), yrange(1):yinc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];
[yhat, ysoft] = forestTest(m, xy);
decmaphard= reshape(yhat, image_size);
decmap= reshape(ysoft, [image_size 3]);

% decision boundary
subplot(121);
imagesc(xrange,yrange,decmaphard);
hold on;
set(gca,'ydir','normal');
cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
colormap(cmap);
plot(X(Y==1,1), X(Y==1,2), 'o', 'MarkerFaceColor', [.9 .3 .3], 'MarkerEdgeColor','k');
plot(X(Y==2,1), X(Y==2,2), 'o', 'MarkerFaceColor', [.3 .9 .3], 'MarkerEdgeColor','k');
plot(X(Y==3,1), X(Y==3,2), 'o', 'MarkerFaceColor', [.3 .3 .9], 'MarkerEdgeColor','k');
hold off;

% probabilities
subplot(122)
imagesc(xrange,yrange,decmap);
hold on;
set(gca,'ydir','normal');
plot(X(Y==1,1), X(Y==1,2), 'o', 'MarkerFaceColor', [.9 .3 .3], 'MarkerEdgeColor','k');
plot(X(Y==2,1), X(Y==2,2), 'o', 'MarkerFaceColor', [.3 .9 .3], 'MarkerEdgeColor','k');
plot(X(Y==3,1), X(Y==3,2), 'o', 'MarkerFaceColor', [.3 .3 .9], 'MarkerEdgeColor','k');
