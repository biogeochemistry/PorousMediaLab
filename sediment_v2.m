function [Sol_C, time, x] = sediment_v2(D_C,w_C,years)
% General:
F_bottom  = 0;    % umol per L/y/cm^2

% domain
L    = 100; % cm
tend = years;
phi = 1;
dx = 0.1; % cm
dt = 0.001;% h
x  = 0:dx:L;
N  = size(x,2);
time = 0:dt:tend;

% initializations
% species:
[AL, AR] = AL_AR_dirichlet(D_C, w_C, phi, dt, dx, N);

% species
BC_top_C  = 1;  % umol per L
Init_C    = 0;
Init_C  = Init_C*ones(N,1);
Init_C(1) = BC_top_C;
Sol_C      = zeros(N,size(time,2));
Sol_C(:,1) = Init_C;


for i=2:size(time,2)
    [Sol_C(:,i-1), B] = update_bc_dirichlet(Sol_C(:,i-1), AR, BC_top_C, F_bottom, D_C, phi, dt, dx, N);
    Sol_C(:,i)      = linalg_solver(AL, B);
    Sol_C(1,i) = BC_top_C;
end

%% linalg_solver: x = A \ b
function [x] = linalg_solver(A, b)
    x =  A \ b;

%% update_bc_dirichlet: function description
function [C, B] = update_bc_dirichlet(C, AR, BC_top_C, F_bottom, D_C, phi, dt, dx, N)
    B = AR*C;
    C(1)    = BC_top_C;
    s       = phi*D_C*dt/dx/dx;
    B(N) = B(N) + 2*F_bottom*s*dx/phi/D_C;

%% AL_AR_dirichlet: creates AL and AR matrices with Dirichlet BC
function [AL, AR] = AL_AR_dirichlet(D_C, w_C, phi, dt, dx, N)
    s       = phi*D_C*dt/dx/dx;
    q       = phi*w_C*dt/dx;
    e1      = ones(N,1);
    AL      = spdiags([e1.*(-s/2-q/4) e1.*(1+s) e1.*(-s/2+q/4)],[-1 0 1],N,N);
    AR      = spdiags([e1*(s/2+q/4) e1*(1-s) e1*(s/2-q/4)],[-1 0 1],N,N);
    AL(1,1) = 1; AL(1,2) = 0; AL(N,N) = 1+s; AL(N,N-1) = -s;
    AR(1,1) = 1; AR(1,2) = 0; AR(N,N) = 1-s; AR(N,N-1) = s;
