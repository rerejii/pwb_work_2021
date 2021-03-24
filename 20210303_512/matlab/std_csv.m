M = readmatrix('img-std.csv');
rm = M(:, 3);
gm = M(:, 5);
bm = M(:, 7);
x = [1:18];

figure(1);
plot(x, rm, '-or');
ylabel('R��f����');
xlabel('�摜�ԍ�');
title('R��f����');
xlim([0 19]);
ylim([120 230]);

figure(2);
plot(x, gm, '-og');
ylabel('G��f����');
xlabel('�摜�ԍ�');
title('G��f����');
xlim([0 19]);
ylim([120 230]);

figure(3);
plot(x, bm, '-ob');
ylabel('B��f����');
xlabel('�摜�ԍ�');
title('B��f����');
xlim([0 19]);
ylim([120 230]);