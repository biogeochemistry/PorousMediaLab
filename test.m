L = 30
x = 0:0.1:L

D = 0.5
w = 3
t = 1

C = sediment(D,w,t);
C_sol = 1/2*( erfc( (x-w*t)./2./sqrt(D*t) ) + exp(w.*x/D).*erfc((x+w.*t)/2/sqrt(D*t)));

plot(C(:,end),'kx');hold on; plot(C_sol);legend('numerical','analytical')
