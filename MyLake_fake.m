L = 100;
x = 0:0.1:L;

D = 400;
w = 0.1;
t = 1/365;

C = sediment_v2(D,w,t);


% C_sol = 1/2*( erfc( (x-w*t)./2./sqrt(D*t) ) + exp(w.*x/D).*erfc((x+w.*t)/2/sqrt(D*t)));


% figure                 % Creates a figure
% plot(x, C(:,end),'kx', 'MarkerSize', 8,'LineWidth', 2);hold on; plot(x, C_sol, 'LineWidth', 3);legend('numerical','analytical')
% set(gca,'FontSize',18) % Creates an axes and sets its FontSize to 18
% grid on
% grid minor
