M = readmatrix('img-std.csv');
rm = M(:, 3);
gm = M(:, 5);
bm = M(:, 7);
x = [1:18];



hold on
plot(x, rm, '-+r');
plot(x, gm, '-+g');
plot(x, bm, '-+b');
legend('R��f����', 'G��f����', 'B��f����')
ylabel('��f����');
xlabel('�摜�ԍ�');
title('��f����');
xlim([0 19]);
ylim([120 250]);
