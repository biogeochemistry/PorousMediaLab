function [ sediment_concentrations, sediment_params, sediment_matrix_templates, species_sediment] = sediment_init( pH, max_depth, temperature )
  % Input:
  % bottom pH of the lake, temperature at SWI.

  % Output:
  % init sediment_concentrations - concentrations read from file
  % sediment_params - all sediment params
  % sediment_matrix_templates - all matrix templates for sediment module
  % species_sediment - which species to simulate
    global sed_par_file sediment_params
    sediment_params = params(max_depth, temperature);
    sediment_concentrations = init_concentrations(pH);
    sediment_matrix_templates = templates();
    species_sediment = species();
end

function [species_sediment] = species()
  % which species to simulate 1 = true , 0 = false
  species_sediment = {
      true, 'Oxygen';
      true, 'OM1';
      true, 'OM2';
      true, 'NO3';
      true, 'FeOH3';
      true, 'SO4';
      true, 'NH4';
      true, 'Fe2';
      true, 'FeOOH';
      true, 'H2S';
      true, 'HS';
      true, 'FeS';
      true, 'S0';
      true, 'PO4';
      true, 'S8';
      true, 'FeS2';
      true, 'AlOH3';
      true, 'PO4adsa';
      true, 'PO4adsb';
      true, 'Ca2';
      true, 'Ca3PO42';
      true, 'OMS';
      true, 'H';
      true, 'OH';
      true, 'CO2';
      true, 'CO3';
      true, 'HCO3';
      true, 'NH3';
      true, 'H2CO3';
  };
  species_sediment = containers.Map({species_sediment{:,2}},{species_sediment{:,1}});
end

function [sediment_params] = params(max_depth, temperature)
    global sed_par_file
    f=fopen(sed_par_file);
    % f=fopen('calibration_k_values.txt');
    data = textscan(f,'%s%f', 50,'Delimiter', '\t');
    fclose(f);

    % Estimation of params:
    temperature = 4.8497; % mean value of observed bottom temp; NOTE: We need to make better estimation of temp for diff coef.
    abs_temp = temperature+273.15; % [K]
    P_atm = 1.01; % [Bar]
    P_water = 998 * 9.8 * max_depth/10^5; % [Bar]
    pressure = P_atm + P_water; % [Bar]
    salinity = 0;
    viscosity = viscosity(temperature,pressure,salinity);

    % Linear regression of Diffusion coefficients for cations and anions (Boudreau, 1997):
    D_H    = lr_ion_diffusion(54.4, 1.555, temperature);
    D_OH   = lr_ion_diffusion(25.9, 1.094, temperature);
    D_HCO3 = lr_ion_diffusion(5.06, 0.275, temperature);
    D_CO3  = lr_ion_diffusion(4.33, 0.199, temperature);
    D_NO3  = lr_ion_diffusion(9.5,  0.388, temperature);
    D_SO4  = lr_ion_diffusion(4.88, 0.232, temperature);
    D_NH4  = lr_ion_diffusion(9.5,  0.413, temperature);
    D_Fe2  = lr_ion_diffusion(3.31, 0.15,  temperature);
    D_PO4  = lr_ion_diffusion(2.62, 0.143, temperature);
    D_Ca2  = lr_ion_diffusion(3.6,  0.179, temperature);
    D_HS   = lr_ion_diffusion(10.4, 0.273, temperature);

    % Empirical correlation of Wilke and Chang (1955) as corrected by Hayduk and Laudie (1974)
    D_NH3 = hayduk_laudie_diffusion(viscosity, abs_temp, 24.5);
    D_O2  = hayduk_laudie_diffusion(viscosity, abs_temp, 27.9);
    D_CO2 = hayduk_laudie_diffusion(viscosity, abs_temp, 37.3);

    % Diffusion coefficient based on Einstein relation:
    D_H2CO3 = einstein_diffusion(410.28, abs_temp, viscosity);

    % User specified diffusion coefficients and other params:
    D_H2S = 284;
    D_HS  = 284;
    D_S0  = 100;
    Db    = 5;
    D_A   = 200;

    w     = data{2}(39);
    n     = data{2}(40);
    depth = data{2}(41);
    F     = data{2}(42);
    alfa0 = data{2}(43);

    % OM composition
    Cx1 = data{2}(44);
    Ny1 = data{2}(45);
    Pz1 = data{2}(46);
    Cx2 = data{2}(47);
    Ny2 = data{2}(48);
    Pz2 = data{2}(49);

    ts  = data{2}(50);


    % Porosity modeling according to Rabouille, C. & Gaillard, J.-F., 1991:
    % NOTE: the experimental function. Checking the result of non-constant profile.
    x  = linspace(0,depth,n);

    fi_in = data{2}(35);
    fi_f  = data{2}(36);
    X_b   = data{2}(37);
    tortuosity = data{2}(38);
    fi = ( fi_in - fi_f ) * exp( -x' / X_b ) + fi_f;

    alfax = alfa0*exp(-0.25*x);
    alfax = alfax';

    sediment_params = {...
        % Spatial domain:
        n, 'n'; % points in spatial grid
        depth, 'depth'; % sediment depth
        1/365, 'years'; % 1 day #35
        ts, 'ts'; % time step
        x,     'x'; % x-axis

        % Scheme properties:
        % if alpha = betta = 0   then it is fully explicit
        % if alpha = betta = 1   then it is fully implicit
        % if alpha = betta = 1/2 then it is Crank-Nicolson
        % if alpha != betta      then it is additive Runge-Kutta method
        1, 'alpha'; % Diffusion shift
        1, 'betta'; % Advection shift
        1, 'gama';  % Reaction shift

        % Physical properties:
        w,    'w';  % time-dependent burial rate w = 0.1
        F,    'F';  % conversion factor = rhob * (1-fi) / fi ; where fi = porosity and rhob = solid phase density
        viscosity,   'viscosity';
        temperature, 'temperature';
        pressure,    'pressure';
        salinity,    'salinity';

        % Porosity profile  Rabouille, C. & Gaillard, J.-F., 1991.
        fi,     'fi'; % porosity
        tortuosity, 'tortuosity'; % tortuosity

        % Bio properties:
        Db,    'Db'; %'effective diffusion due to bioturbation, Canavan et al D_bio between 0-5, 5 in the top layers'; % #41
        alfax, 'alfax';  % bioirrigation

        % effective molecular diffusion
        D_O2,   'D_O2';
        D_NO3,  'D_NO3';
        D_SO4,  'D_SO4';
        D_NH4,  'D_NH4';
        D_Fe2,  'D_Fe2';
        D_H2S,  'D_H2S';
        D_S0,   'D_S0';
        D_PO4,  'D_PO4';
        D_Ca2,  'D_Ca2';
        D_HS,   'D_HS';
        D_H,    'D_H';
        D_OH,   'D_OH';
        D_CO2,  'D_CO2';
        D_CO3,  'D_CO3';
        D_HCO3, 'D_HCO3';
        D_NH3,  'D_NH3';
        D_H2CO3,'D_H2CO3';
        D_A,    'D_A'; % new added species here

        % OM composition
        Cx1, 'Cx1';
        Ny1, 'Ny1';
        Pz1, 'Pz1';
        Cx2, 'Cx2';
        Ny2, 'Ny2';
        Pz2, 'Pz2';

        % pH module. NOTE: experimental feature
        % Specify pH algorithm:
        % 0. Disabled
        % 1. Stumm & Morgan, 1995. Aquatic Chemistry. MATLAB
        % 2. Stumm & Morgan, 1995. Aquatic Chemistry. C++
        % 3. Phreeqc
        % 4. Delta function by Markelov (under test)

        0,    'pH algorithm';

        % chemical constants from file
        data{2}(1), 'k_OM';
        data{2}(2), 'k_OMb';
        data{2}(3), 'Km_O2';
        data{2}(4), 'Km_NO3';
        data{2}(5), 'Km_FeOH3';
        data{2}(6), 'Km_FeOOH';
        data{2}(7), 'Km_SO4';
        data{2}(8), 'Km_oxao';
        data{2}(9), 'Km_amao';
        data{2}(10), 'Kin_O2';
        data{2}(11), 'Kin_NO3';
        data{2}(12), 'Kin_FeOH3';
        data{2}(13), 'Kin_FeOOH';
        data{2}(14), 'k_amox';
        data{2}(15), 'k_Feox';
        data{2}(16), 'k_Sdis';
        data{2}(17), 'k_Spre';
        data{2}(18), 'k_FeS2pre';
        data{2}(19), 'k_alum';
        data{2}(20), 'k_pdesorb_a';
        data{2}(21), 'k_pdesorb_b';
        data{2}(22), 'k_rhom';
        data{2}(23), 'k_tS_Fe';
        data{2}(24), 'Ks_FeS';
        data{2}(25), 'k_Fe_dis';
        data{2}(26), 'k_Fe_pre';
        data{2}(27), 'k_apa';
        data{2}(28), 'kapa';
        data{2}(29), 'k_oms';
        data{2}(30), 'k_tsox';
        data{2}(31), 'k_FeSpre';
        data{2}(32), 'accel';
        data{2}(33), 'f_pfe';
        data{2}(34), 'k_pdesorb_c'};
    sediment_params = containers.Map({sediment_params{:,2}},{sediment_params{:,1}});
end

function [sediment_concentrations ] = init_concentrations(pH)
    global sediment_params
    n = sediment_params('n');

    % Init concentrations of sediment species
    O2      = ones(n,1) * 0;
    OM      = ones(n,1) * 0;
    OMb     = ones(n,1) * 0;
    NO3     = ones(n,1) * 0;
    FeOH3   = ones(n,1) * 0;
    SO4     = ones(n,1) * 0;
    NH4     = ones(n,1) * 0;
    Fe2     = ones(n,1) * 0;
    FeOOH   = ones(n,1) * 0;
    H2S     = ones(n,1) * 0;
    HS      = ones(n,1) * 0;
    FeS     = ones(n,1) * 0;
    S0      = ones(n,1) * 0;
    PO4     = ones(n,1) * 0;
    S8      = ones(n,1) * 0;
    FeS2    = ones(n,1) * 0;
    AlOH3   = ones(n,1) * 0;
    PO4adsa = ones(n,1) * 0;
    PO4adsb = ones(n,1) * 0;
    Ca2     = ones(n,1) * 0;
    Ca3PO42 = ones(n,1) * 0;
    OMS     = ones(n,1) * 0;
    OH      = ones(n,1) * 0;
    CO2     = ones(n,1) * 0;
    CO3     = ones(n,1) * 0;
    HCO3    = ones(n,1) * 0;
    NH3     = ones(n,1) * 0;
    H       = ones(n,1) * 10^-pH*10^3;
    H2CO3   = ones(n,1) * 0;


    sediment_concentrations = {...
        O2,      'Oxygen';
        OM,      'OM1';
        OMb,     'OM2';
        NO3,     'NO3';
        FeOH3,   'FeOH3';
        SO4,     'SO4';
        NH4,     'NH4';
        Fe2,     'Fe2';
        FeOOH,   'FeOOH';
        H2S,     'H2S';
        HS,      'HS';
        FeS,     'FeS';
        S0,      'S0';
        PO4,     'PO4';
        S8,      'S8';
        FeS2,    'FeS2';
        AlOH3,   'AlOH3';
        PO4adsa, 'PO4adsa';
        PO4adsb, 'PO4adsb';
        Ca2,     'Ca2';
        Ca3PO42, 'Ca3PO42';
        OMS,     'OMS';
        H ,      'H';
        OH ,     'OH';
        CO2,     'CO2';
        CO3,     'CO3';
        HCO3,    'HCO3';
        NH3,     'NH3';
        H2CO3,   'H2CO3';
    };
    sediment_concentrations = containers.Map({sediment_concentrations{:,2}},{sediment_concentrations{:,1}});
end

function [D] = einstein_diffusion(D_ref, abs_temp, viscosity)
  %% einstein_diffusion: Einstein's relation of diffusion coefficient derived from Einstein's formula D/D0
  % input:
  T_ref = 273.15 + 25; % [K]
  viscosity_ref = 1.002*10^-2; % [g s-1 cm-1]
  % D_ref - reference diffusion at 25'C [cm^2 year-1]
  % viscosity_ref - reference viscosity at 25'C
  % viscosity - current viscosity
  % abs_temp - current temperature in K

  % Output:
  % D - diffusion coefficient [cm^2 year-1]

  D = D_ref*(abs_temp/T_ref)*(viscosity_ref/viscosity);
end

function [D] = hayduk_laudie_diffusion(viscosity, abs_temp, molar_volume)
  %% hayduk_laudie_diffusion: empirical correlation of Wilke and Chang (1955) as corrected by Hayduk and Laudie (1974):

  % Input:
  % viscosity
  % abs_temp - temperature in K
  % molar_volume - is the molar volume of the nonelectrolyte (at the normal boiling temperature of that solute). [experimental]
  % 3.156*10^7 [seconds] ---> [year]

  % Output:
  % Diffusion coeff. in [cm^2 year-1]
  D = 4.71978 * 10^-9 * abs_temp / ( viscosity * molar_volume^0.6 ) * 3.156 * 10^7;
end

function [D] = lr_ion_diffusion(m0, m1, t)
  %%  lr_ion_diffusion: Linear Regressions† of the Infinite- Dilution Diffusion Coefficients Do for Anions and Cations against Temperature (Boudreau, B.P., 1997. Diagenetic Models and Their Implementation)

  % Output:
  % Diffusion coeff. in [cm^2 year-1]
  % 3.156*10^7 [seconds] ---> [year]

  D = ( m0 + m1*t ) * 10^-6 * 3.156 * 10^7;
end

function [u] = viscosity(temperature,pressure,salinity)
  %% viscosity: Values of the dynamic viscosity can be calculated from an empirical equation
  % developed by Matthaus (as quoted in Kukulka et al., 1987), which is claimed to be accurate to within 0.7% for the temperature, t, salinity, S, and pressure, P, ranges of 0
  % ≤ C ≤ 30, 0 ≤ S ≤ 36, and 1 to 1000 bars, respectively (Boudreau, 1997):

  % Input units
  t = temperature; % [Celsius]
  p = pressure; % [Bar]
  s = salinity; % [ppt] NOTE:not sure about units here!

  % Output:
  % viscosity [g cm-1 s-1] = Poise units
  u = (1.7910 - (6.144*10^-2)*t + (1.4510*10^-3)*t^2 - (1.6826*10^-5)*t^3 - (1.5290*10^-4)*p + (8.3885*10^-8)*p^2 + (2.4727*10^-3)*s + t*(6.0574*10^-6 *p - 2.6760*10^-9*p^2) + s*(4.8429*10^-5*t - 4.7172*10^-6*t^2 + 7.5986*10^-8*t^3))*10^-2;

end

function [sediment_matrix_templates] = templates()
    global sediment_params
    n  = sediment_params('n');
    depth = sediment_params('depth');
    x = linspace(0,depth,n);
    dx = x(2)-x(1);
    t = 0:sediment_params('ts'):sediment_params('years');
    dt    = t(2)-t(1);
    alpha = sediment_params('alpha'); % Diffusion shift
    betta = sediment_params('betta'); % Advection shift
    gama  = sediment_params('gama'); % Reaction shift
    v = sediment_params('w');
    fi  = sediment_params('fi');
    tortuosity = sediment_params('tortuosity');
    Db    = sediment_params('Db');

    % formation of templates:
    % Solid template the same for all solid species due to diffusion and advection coef the same for all.
    [LU_solid,  RK_solid,  LD_solid,  LA_solid,  RD_solid,  RA_solid] = matrices_template_solid(Db, tortuosity, v, fi, dx, dt, alpha, betta, n);

    % solute templates:
    [LU_ox0,  RK_ox0,  LD_ox0,  LA_ox0,  RD_ox0,  RA_ox0]        = matrices_template_solute(sediment_params('D_O2') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_NO30, RK_NO30, LD_NO30, LA_NO30, RD_NO30, RA_NO30]       = matrices_template_solute(sediment_params('D_NO3') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_SO40, RK_SO40, LD_SO40, LA_SO40, RD_SO40, RA_SO40]       = matrices_template_solute(sediment_params('D_SO4') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_NH40, RK_NH40, LD_NH40, LA_NH40, RD_NH40, RA_NH40]       = matrices_template_solute(sediment_params('D_NH4') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_Fe20, RK_Fe20, LD_Fe20, LA_Fe20, RD_Fe20, RA_Fe20]       = matrices_template_solute(sediment_params('D_Fe2') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_H2S0, RK_H2S0, LD_H2S0, LA_H2S0, RD_H2S0, RA_H2S0]       = matrices_template_solute(sediment_params('D_H2S') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_S00, RK_S00, LD_S00, LA_S00, RD_S00, RA_S00]             = matrices_template_solute(sediment_params('D_S0') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_PO40, RK_PO40, LD_PO40, LA_PO40, RD_PO40, RA_PO40]       = matrices_template_solute(sediment_params('D_PO4') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_Ca20, RK_Ca20, LD_Ca20, LA_Ca20, RD_Ca20, RA_Ca20]       = matrices_template_solute(sediment_params('D_Ca2') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_HS0, RK_HS0, LD_HS0, LA_HS0, RD_HS0, RA_HS0]             = matrices_template_solute(sediment_params('D_HS') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_H0, RK_H0, LD_H0, LA_H0, RD_H0, RA_H0]                   = matrices_template_solute(sediment_params('D_H') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_OH0, RK_OH0, LD_OH0, LA_OH0, RD_OH0, RA_OH0]             = matrices_template_solute(sediment_params('D_OH') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_CO20, RK_CO20, LD_CO20, LA_CO20, RD_CO20, RA_CO20]       = matrices_template_solute(sediment_params('D_CO2') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_CO30, RK_CO30, LD_CO30, LA_CO30, RD_CO30, RA_CO30]       = matrices_template_solute(sediment_params('D_CO3') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_HCO30, RK_HCO30, LD_HCO30, LA_HCO30, RD_HCO30, RA_HCO30] = matrices_template_solute(sediment_params('D_HCO3') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_NH30, RK_NH30, LD_NH30, LA_NH30, RD_NH30, RA_NH30]       = matrices_template_solute(sediment_params('D_NH3') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);
    [LU_H2CO30, RK_H2CO30, LD_H2CO30, LA_H2CO30, RD_H2CO30, RA_H2CO30]= matrices_template_solute(sediment_params('D_H2CO3') + Db, tortuosity, v, fi, dx, dt, alpha, betta, n);

    sediment_matrix_templates = {...

        LU_solid,  RK_solid,  LD_solid,  LA_solid,  RD_solid,  RA_solid, 'Solid';  % 1
        LU_ox0,  RK_ox0,  LD_ox0,  LA_ox0,  RD_ox0,  RA_ox0, 'Oxygen'; % 2
        LU_NO30, RK_NO30, LD_NO30, LA_NO30, RD_NO30, RA_NO30, 'NO3'; % 3
        LU_SO40, RK_SO40, LD_SO40, LA_SO40, RD_SO40, RA_SO40, 'SO4'; % 4
        LU_NH40, RK_NH40, LD_NH40, LA_NH40, RD_NH40, RA_NH40, 'NH4'; % 5
        LU_Fe20, RK_Fe20, LD_Fe20, LA_Fe20, RD_Fe20, RA_Fe20, 'Fe2'; % 6
        LU_H2S0, RK_H2S0, LD_H2S0, LA_H2S0, RD_H2S0, RA_H2S0, 'H2S'; % 7
        LU_S00, RK_S00, LD_S00, LA_S00, RD_S00, RA_S00, 'S0'; % 8
        LU_PO40, RK_PO40, LD_PO40, LA_PO40, RD_PO40, RA_PO40, 'PO4'; % 9
        LU_Ca20, RK_Ca20, LD_Ca20, LA_Ca20, RD_Ca20, RA_Ca20, 'Ca2'; % 10
        LU_HS0, RK_HS0, LD_HS0, LA_HS0, RD_HS0, RA_HS0, 'HS'; % 11
        LU_H0, RK_H0, LD_H0, LA_H0, RD_H0, RA_H0, 'H'; % 12
        LU_OH0, RK_OH0, LD_OH0, LA_OH0, RD_OH0, RA_OH0, 'OH'; % 13
        LU_CO20, RK_CO20, LD_CO20, LA_CO20, RD_CO20, RA_CO20, 'CO2'; % 14
        LU_CO30, RK_CO30, LD_CO30, LA_CO30, RD_CO30, RA_CO30, 'CO3'; % 15
        LU_HCO30, RK_HCO30, LD_HCO30, LA_HCO30, RD_HCO30, RA_HCO30, 'HCO3'; % 16
        LU_NH30, RK_NH30, LD_NH30, LA_NH30, RD_NH30, RA_NH30, 'NH3'; % 17
        LU_H2CO30, RK_H2CO30, LD_H2CO30, LA_H2CO30, RD_H2CO30, RA_H2CO30, 'H2CO3'; %18
    };

end

function [LU_solute, RK_solute, LD, LA, RD, RA] = matrices_template_solute(D_m, theta, v, fi, dx, dt, alpha, betta, n)
  %MATRICES Formation of matrices for species
  % ======================================================================

  D = D_m / theta^2;

  % NOTE: Removed porosity from advective term

  % Coefficients solute:
  LD = fi(2:end) * D * dt * betta / dx^2 ;
  % LA = fi(2:end) * v * dt * alpha / (2 * dx) ;
  LA = v * dt * alpha / (2 * dx) ;
  RD = fi(2:end) * D * dt * (1 - betta) / dx^2 ;
  % RA = fi(2:end) * v * dt * (1 - alpha) / (2 * dx);
  RA =  v * dt * (1 - alpha) / (2 * dx);

  % Constructing left matrix LU for solute species(left unknowns):
  LUdiags = ones(n-1,3); % n-1 due to BC at the toLU
  LUdiags(:,1) = LUdiags(:,1).*(-LA-LD);
  LUdiags(n-2,1) = -2*LD(n-2);
  LUdiags(:,2) = LUdiags(:,2).*(1+2*LD);
  LUdiags(:,3) = LUdiags(:,3).*(LA-LD);
  LUnum = [-1 0 1];
  LU_solute = spdiags(LUdiags,LUnum,n-1,n-1);


  % Constructing right matrix RK (right knowns):
  RKdiags = ones(n-1,3); % n-1 due to BC at the top
  RKdiags(:,1) = RKdiags(:,1) .* (RA+RD);
  RKdiags(n-2,1) = 2*RD(n-2);
  RKdiags(:,2) = RKdiags(:,2).*(1-2*RD);
  RKdiags(:,3) = RKdiags(:,3).*(-RA+RD);
  RKnum = [-1 0 1];
  RK_solute = spdiags(RKdiags,RKnum,n-1,n-1);
end

function [LU_solid, RK_solid, LD, LA, RD, RA] = matrices_template_solid(D_m, theta, v, fi, dx, dt, alpha, betta, n)
  %MATRICES Formation of matrices for species
  % ======================================================================

  D = D_m / theta^2;

  % NOTE: Removed porosity from advective term

  % Coefficients solid:
  LD = (1-fi) * D * dt * betta / dx^2 ;
  % LA = (1-fi) * v * dt * alpha / (2 * dx) ;
  LA =  v * dt * alpha / (2 * dx) ;
  RD = (1-fi) * D * dt * (1 - betta) / dx^2 ;
  % RA = (1-fi) * v * dt * (1 - alpha) / (2 * dx);
  RA =  v * dt * (1 - alpha) / (2 * dx);


  % Constructing left matrix LU_solid for solid species(left unknowns):
  LUdiags_solid = ones(n,3); % n-1 due to BC at the top
  LUdiags_solid(:,1) = LUdiags_solid(:,1).*(-LA-LD);
  LUdiags_solid(n-1,1) = -2*LD(n-1);
  LUdiags_solid(:,2) = LUdiags_solid(:,2).*(1+2*LD);
  LUdiags_solid(:,3) = LUdiags_solid(:,3).*(LA-LD);
  LUnum_solid = [-1 0 1];
  LU_solid = spdiags(LUdiags_solid,LUnum_solid,n,n);
  LU_solid(1,2) = - 2*LD(1);
  LU_solid(1,1) = ((LD(1)+LA(1))*2*dx*v/D+1+2*LD(1));


  % Constructing right matrix RK_solid for solid species(right knowns)
  RKdiags_solid = ones(n,3); % n-1 due to BC at the top
  RKdiags_solid(:,1) = RKdiags_solid(:,1) .* (RA+RD);
  RKdiags_solid(n-1,1) = 2*RD(n-1);
  RKdiags_solid(:,2) = RKdiags_solid(:,2).*(1-2*RD);
  RKdiags_solid(:,3) = RKdiags_solid(:,3).*(-RA+RD);
  RKnum_solid = [-1 0 1];
  RK_solid = spdiags(RKdiags_solid,RKnum_solid,n,n);
  RK_solid(1,2) = 2* RD(1);
  RK_solid(1,1) = ((-RA(1) - RD(1)) * 2*dx*v/D + 1-2*RD(1));
end

