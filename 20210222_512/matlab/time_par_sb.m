name = {'drop1' 'drop2' 'drop3' 'drop4' 'drop5' 'drop6'};
s = [16 14 12 10 8 6 4];
x = [0:6];
time = [391.4 307.7 279 259.7 223 177.9 131.8];
sb = [475.6 671.4 616.2 569.5 665.6 1046.1 1903.2];

yyaxis left
plot(x, sb);
ylabel('’füE’Z—”','Color','black')

yyaxis right
plot(x, time);
ylabel('ˆ—ŠÔ‚Ì’Zk•b”[•b]','Color','black')

xlabel('‘w”')
title('’füE’Z—”‚Æˆ—ŠÔ‚ÌŠÖŒW')

xticks(x)
xticklabels(s)

% name = {'drop1' 'drop2' 'drop3' 'drop4' 'drop5' 'drop6'};
% x = [1:6];
% time = [83.7 112.4 131.7 168.4 213.5 259.6];
% acc = [0.000177693 0.002034356 0.000312307 0.006956071 0.007398045 0.008607325];
% sb = [195.8 140.6 93.9 190 570.5 1427.6];
% 
% yyaxis left
% plot(x, sb);
% title('’füE’Z—”‚Æˆ—ŠÔ‚ÌŠÖŒW')
% xlabel('Network')
% ylabel('ˆ«‰»‚µ‚½’füE’Z—”','Color','black')
% 
% yyaxis right
% plot(x, time);
% ylabel('ˆ—ŠÔ‚Ì’Zk•b”[•b]','Color','black')
% 
% xticks([1:6])
% xticklabels(name)