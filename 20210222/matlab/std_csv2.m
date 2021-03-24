M = readmatrix('img-std.csv');
rm = M(:, 3);
gm = M(:, 5);
bm = M(:, 7);
x = [1:18];



hold on
plot(x, rm, '-+r');
plot(x, gm, '-+g');
plot(x, bm, '-+b');
legend('R‰æ‘f•½‹Ï', 'G‰æ‘f•½‹Ï', 'B‰æ‘f•½‹Ï')
ylabel('‰æ‘f•½‹Ï');
xlabel('‰æ‘œ”Ô†');
title('‰æ‘f•½‹Ï');
xlim([0 19]);
ylim([120 250]);
