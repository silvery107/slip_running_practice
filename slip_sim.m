clc; clear; close all;
import casadi.*

% Robot parameters
p = struct();
p.m = 1.;
p.g = 9.81;
p.l0 = 0.8; % Spring rest length
p.ks = 200; % Spring constant
p.Ts = 0.2; % Stance duration
p.max_phase_dt = 1; % Maximum phase duration
p.plot = false;

% Global parameters
k_raibert = 0.1;
ctrl_flag = 2;
num_jump = 10;
vx_des = 2.;

% Initial state and control
u0 = 0;
pc0 = [0; 1.3];
dpc0 = [0; 0];
pf0 = pc0 + rot_mat(u0) * [0; -p.l0];
x0 = [pc0; dpc0; pf0]; 
t0 = 0;
% x: [pc_x, pc_z, dpc_x, dpc_z, pf_x, pf_z]
% u: [th] % TODO include Ks as control?

% Build gait library
gait_library = build_gait_library(0, 3, 0.1, p);

p.plot = true;
p.fig = figure();
% exportgraphics(p.fig, "my_slip.gif", "Append", false);

%% Main loop
x = x0;
u = u0;
t = t0;
for ii=1:num_jump
    % Unpack states
    pc = x(1:2);
    dpc = x(3:4);
    x_apex = x(2:3);
    % Calculate control input
    switch ctrl_flag
        case 1 % Raibert heuristic
            pfx_err = dpc(1) * p.Ts / 2 + k_raibert * (dpc(1) - vx_des);
            u = asin(max(min(pfx_err, p.l0), -p.l0) / p.l0);
        case 2 % Linear deadbeat
            vx_tgt = min(x_apex(2)+0.1, vx_des);
            gait = gait_library(vx_tgt);
            u = gait.u_star + gait.K * (x_apex - gait.x_star);
        case 3 % Open loop
            u = u_star;
    end
    % Apply control
    pf_tgt = pc + rot_mat(u) * [0; -p.l0];
    x(5:6) = pf_tgt;
    % Simulate apex to apex
    [t, x] = simulate(t, x, p);
    % Early return
    if x(6) < -1e-5 || x(2) < -1e-5
        return;
    end
end

%% Functions
function gait_library = build_gait_library(v_lb, v_ub, v_delta, p)
    % Build gait library maps vx to (K, x*, u*)
    keys = v_lb:v_delta:v_ub;
    values = [];
    for vx=keys
        [x_star, u_star] = find_periodic_gait(vx, p);
        K = calc_feedback_gain(x_star, u_star, p);
        gait = struct('K', K, 'x_star', x_star, 'u_star', u_star);
        values = [values, gait];
    end
    gait_dict = dictionary(string(keys), values);
    gait_library = @(vx) gait_dict(string(round(vx, 1)));
end

function K = calc_feedback_gain(x_star, u_star, p)
    % Calculate K matrix via centered finite difference
    epsilon = 1e-4;
    dim_x = length(x_star);
    dim_u = length(u_star);

    Jx = zeros(dim_x, dim_x);
    for idx=1:size(Jx, 2)
        x_perturb = x_star;
        x_perturb(idx) = x_star(idx) - epsilon;
        x_prev = apex_to_apex_map(x_perturb, u_star, p);
        x_perturb = x_star;
        x_perturb(idx) = x_star(idx) + epsilon;
        x_next = apex_to_apex_map(x_perturb, u_star, p);
        Jx(:, idx) = (x_next - x_prev) / 2*epsilon;
    end

    Ju = zeros(dim_x, dim_u);
    for idx=1:size(Ju, 2)
        u_perturb = u_star;
        u_perturb(idx) = u_star(idx) - epsilon;
        x_prev = apex_to_apex_map(x_star, u_perturb, p);
        u_perturb = u_star;
        u_perturb(idx) = u_star(idx) + epsilon;
        x_next = apex_to_apex_map(x_star, u_perturb, p);
        Ju(:, idx) = (x_next - x_prev) / 2*epsilon;
    end

    K = -Ju\Jx;
end

function [x_star, u_star] = find_periodic_gait(vx, p)
    var_init = [1; 0]; % opt var: pc_z, u
    var = lsqnonlin(@(var)min_apex_map(var, vx, p), var_init);
    x_star = [var(1); vx];
    u_star = var(2);
end

function obj = min_apex_map(var, vx, p)
    % min_{pc_z, u} ||x_apex_next - x_apex||^2
    pc_z = var(1);
    u = var(2);
    x_apex = [pc_z; vx];
    x_apex_next = apex_to_apex_map(x_apex, u, p);
    obj = x_apex_next - x_apex;
end

function x_apex_next = apex_to_apex_map(x_apex, u, p)
    pc_z = x_apex(1);
    vx = x_apex(2);
    pc0 = [0; pc_z];
    dpc0 = [vx; 0];
    pf0 = pc0 + rot_mat(u) * [0; -p.l0];
    x0 = [pc0; dpc0; pf0]; 
    [~, x_next] = simulate(0, x0, p);
    x_apex_next = [x_next(2); x_next(3)];
end

function [t0, x] = simulate(t0, x, p)
    % assert(abs(x(4)) < 1e-6)

    % Simulate flight phase from apex
    tf = t0 + p.max_phase_dt;
    options = odeset("Events", @(t, x)event_td(t, x, p));
    sol = ode45(@(t, x)dynamics_flight(t, x, p), [t0, tf], x, options);
    t0 = sol.x(end);
    x = sol.y(:, end);

    for idx=1:length(sol.x)
        plot_robot(sol.y(:, idx), p);
    end
    if x(6) < -1e-5 || x(2) < -1e-5
        return;
    end
    
    % Simulate stance phase
    tf = t0 + p.max_phase_dt;
    options = odeset("Events", @(t, x)event_lo(t, x, p));
    sol = ode45(@(t, x)dynamics_stance(t, x, p), [t0, tf], x, options);
    t0 = sol.x(end);
    x = sol.y(:, end);

    for idx=1:length(sol.x)
        plot_robot(sol.y(:, idx), p);
    end
    % Update stance duration
    p.Ts = sol.x(end) - sol.x(1);

    % Simulate flight phase to apex
    tf = t0 + p.max_phase_dt;
    options = odeset("Events", @(t, x)event_apex(t, x, p));
    sol = ode45(@(t, x)dynamics_flight(t, x, p), [t0, tf], x, options);
    t0 = sol.x(end);
    x = sol.y(:, end);

    for idx=1:length(sol.x)
        plot_robot(sol.y(:, idx), p);
    end
end

function [value, isterminal, direction] = event_td(t, x, p)
    pf = x(5:6);
    value = pf(2);
    isterminal = 1;
    direction = -1;
end

function [value, isterminal, direction] = event_lo(t, x, p)
    pc = x(1:2);
    pf = x(5:6);
    l = pc - pf;
    value = norm(l) - p.l0;
    isterminal = 1;
    direction = 1;
end

function [value, isterminal, direction] = event_apex(t, x, p)
    dpc = x(3:4);
    value = dpc(2);
    isterminal = 1;
    direction = -1;
end

function dxdt = dynamics_flight(t, x, p)
    dpc = x(3:4);
    ddp = [0; -p.g];
    dpf = dpc;
    dxdt = [dpc; ddp; dpf];
end

function dxdt = dynamics_stance(t, x, p)
    pc = x(1:2);
    dpc = x(3:4);
    pf = x(5:6);
    l = pc - pf;
    l_norm = norm(l);
    l_dir = l / l_norm;
    if l_norm > p.l0
        l_dir = l_dir.*0;
    end
    ddp = p.ks * (p.l0 - l_norm) * l_dir / p.m + [0; -p.g];
    dpf = [0; 0];
    dxdt = [dpc; ddp; dpf];
end

function plot_robot(x, p)
    % TODO better visualization
    if ~p.plot
        return;
    end
    clf;
    hold on; grid on; axis equal;
    yline(0, LineWidth=3)
    pc = x(1:2);
    pf = x(5:6);
    scatter(pc(1), pc(2), 1000, 'r', LineWidth=3);
    plot([pc(1); pf(1)], [pc(2); pf(2)], 'k', LineWidth=2);
    ylim([-0.5, 2]);
    % xlim([-0.5, 5]);
    text(pc(1), pc(2)+0.3, "v_x = "+string(x(3)))
    pause(0.05);
    % exportgraphics(p.fig, "my_slip.gif", "Append", true);
end

function R = rot_mat(th)
    R = [cos(th), -sin(th);
         sin(th), cos(th)];
end

