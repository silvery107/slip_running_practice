clc; clear; close all;
import casadi.*

p = struct();
p.mass = 1.;
p.l0 = 0.9;
p.ks = 200;
p.g = 9.81;
p.kr = 0.1;
p.vx_des = 1.;
p.num_jump = 5;
p.max_phase_dt = 1;
p.Ts = 0.2;
p.ctrl_flag = 2;
p.plot = false;

[x_apex_star, u_star] = find_periodic_gait(p.vx_des, p)
K = calc_feedback_gain(x_apex_star, u_star, p)

u0 = u_star;
pc0 = [0; x_apex_star(1)];
dpc0 = [p.vx_des; 0];
pf0 = pc0 + rot_mat(u0) * [0; -p.l0];
x0 = [pc0; dpc0; pf0]; 
t0 = 0;
% x: [pc_x, pc_z, dpc_x, dpc_z, pf_x, pf_z]
% u: [th] % TODO include Ks as control?

p.plot = true;
p.fig = figure();
% exportgraphics(p.fig, "my_slip.gif", "Append", false);

x = x0;
t = t0;
u = u0;
%% Main loop
for ii=1:p.num_jump
    % Unpack states
    pc = x(1:2);
    dpc = x(3:4);
    x_apex = x(2:3);
    % Calculate control input
    switch p.ctrl_flag
        case 1 % Raibert heuristic
            pfx_err = dpc(1) * p.Ts / 2 + p.kr * (dpc(1) - p.vx_des);
            u = asin(max(min(pfx_err, p.l0), -p.l0) / p.l0);
        case 2 % Linear deadbeat
            u = u_star + K * (x_apex - x_apex_star);
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
function build_gait_library(p)
    % TODO build a search tree and map x_apex to (K, x*, u*) tuple
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
    ddp = p.ks * (p.l0 - l_norm) * l_dir / p.mass + [0; -p.g];
    dpf = [0; 0];
    dxdt = [dpc; ddp; dpf];
end

function plot_robot(x, p)
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
    xlim([-0.5, 5]);
    text(0, 1.5, "v_x = "+string(x(3)))
    pause(0.05);
    % exportgraphics(p.fig, "my_slip.gif", "Append", true);
end

function R = rot_mat(th)
    R = [cos(th), -sin(th);
         sin(th), cos(th)];
end

