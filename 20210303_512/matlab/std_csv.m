M = readmatrix('img-std.csv');
rm = M(:, 3);
gm = M(:, 5);
bm = M(:, 7);
x = [1:18];

figure(1);
plot(x, rm, '-or');
ylabel('R‰æ‘f•½‹Ï');
xlabel('‰æ‘œ”Ô†');
title('R‰æ‘f•½‹Ï');
xlim([0 19]);
ylim([120 230]);

figure(2);
plot(x, gm, '-og');
ylabel('G‰æ‘f•½‹Ï');
xlabel('‰æ‘œ”Ô†');
title('G‰æ‘f•½‹Ï');
xlim([0 19]);
ylim([120 230]);

figure(3);
plot(x, bm, '-ob');
ylabel('B‰æ‘f•½‹Ï');
xlabel('‰æ‘œ”Ô†');
title('B‰æ‘f•½‹Ï');
xlim([0 19]);
ylim([120 230]);