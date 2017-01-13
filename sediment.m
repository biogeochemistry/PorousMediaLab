function  [C1_res, time, x] = sediment(D,w,years)
global dt dx phi N

BC1_top    = 1;  % umol per L
F_bottom  = 0;    % umol per L/y/cm^2
L         = 30; % cm
tend      = years;
C1_init    = 0;


phi =1;
dx = 0.1; % cm
dt = 0.001;% h
x  = 0:dx:L;
N  = size(x,2);

C1_init  = C1_init*ones(N,1);
C1_init(1) = BC1_top;


[AL, AR] = AL_AR_dirichlet(D,w);


C1_old      = C1_init;

time       = 0:dt:tend;
C1_res      = zeros(N,size(time,2));
C1_res(:,1) = C1_init;


for i=2:size(time,2)
    B = AR*C1_old;
    [C1_old, B] = update_bc_dirichlet(C1_old, B, BC1_top, F_bottom, D);
    C1_new      = linalg_solver(AL, B);
    C1_res(:,i) = C1_new;
    C1_old      = C1_new;
    Cold(1)     = BC1_top;
end

%% linalg_solver: x = A \ b
function [x] = linalg_solver(A, b)
    x =  A \ b;

%% update_bc_dirichlet: function description
function [C, B] = update_bc_dirichlet(C, B, BC_top, F_bottom, D)
    global dt dx phi N
    C(1)    = BC_top;
    s       = phi*D*dt/dx/dx;
    B(N) = B(N) + 2*F_bottom*s*dx/phi/D;

%% AL_AR_dirichlet: creates AL and AR matrices with Dirichlet BC
function [AL, AR] = AL_AR_dirichlet(D,w)
    global dt dx phi N
    s       = phi*D*dt/dx/dx;
    q       = phi*w*dt/dx;
    e1      = ones(N,1);
    AL      = spdiags([e1.*(-s/2-q/4) e1.*(1+s) e1.*(-s/2+q/4)],[-1 0 1],N,N);
    AR      = spdiags([e1*(s/2+q/4) e1*(1-s) e1*(s/2-q/4)],[-1 0 1],N,N);
    AL(1,1) = 1; AL(1,2) = 0; AL(N,N) = 1+s; AL(N,N-1) = -s;
    AR(1,1) = 1; AR(1,2) = 0; AR(N,N) = 1-s; AR(N,N-1) = s;


