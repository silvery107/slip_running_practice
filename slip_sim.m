clc; clear; close all;
import casadi.*

% Robot parameters
p = struct();
p.m = 1.;
p.g = 9.81;
p.ks = 500; % Spring constant
p.l0 = 0.8; % Spring rest length
p.l1 = 0.5; % Hopper link 1 length
p.l2 = 0.5; % Hopper link 2 length
p.Ts = 0.2; % Stance duration
p.max_phase_dt = 1; % Maximum phase duration
p.plot = false;

% Global parameters
k_raibert = 0.1;
ctrl_flag = 2;
num_jump = 10;
vx_des = 3.;

% Initial state and control
u0 = 0;
pc0 = [0; 1.3];
dpc0 = [1; 0];
pf0 = pc0 + rot_mat(u0) * [0; -p.l0];
x0 = [pc0; dpc0; pf0]; 
t0 = 0;
% x: [pc_x, pc_z, dpc_x, dpc_z, pf_x, pf_z]
% u: [th] % TODO include Ks as control?

% Build gait library
gait_library = build_gait_library(0, 3, 0.1, p);

p.plot = true;
p.fig = figure();
% exportgraphics(p.fig, "slip_demo.gif", "Append", false);

%% Main loop
clf;
p = init_renderer(x0, p);
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
        case 2 % Linear deadbeat gait library
            vx_tgt = min(x_apex(2)+0.1, vx_des);
            gait = gait_library(vx_tgt);
            u = gait.u_star + gait.K * (x_apex - gait.x_star);
            p.t_vx_des.String = "v_{x,d} = "+ string(round(vx_tgt, 1));
            drawnow
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
    options = optimoptions('lsqnonlin', 'Display', 'off');
    var_init = [1; 0]; % opt var: pc_z, u
    var = lsqnonlin(@(var)min_apex_map(var, vx, p), var_init, [], [], options);
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
    x_apex_next = x_next(2:3);
end

function [t0, x] = simulate(t0, x, p)
    % assert(abs(x(4)) < 1e-6)

    % Simulate flight phase from apex
    tf = t0 + p.max_phase_dt;
    options = odeset("Events", @(t, x)event_td(x, p));
    sol = ode45(@(t, x)dynamics_flight(x, p), [t0, tf], x, options);
    t0 = sol.x(end);
    x = sol.y(:, end);

    for idx=1:length(sol.x)
        dt = sol.x(min(idx+1, length(sol.x))) - sol.x(idx);
        plot_robot(sol.y(:, idx), dt, p);
    end
    if x(6) < -1e-5 || x(2) < -1e-5
        return;
    end
    
    % Simulate stance phase
    tf = t0 + p.max_phase_dt;
    options = odeset("Events", @(t, x)event_lo(x, p));
    sol = ode45(@(t, x)dynamics_stance(x, p), [t0, tf], x, options);
    t0 = sol.x(end);
    x = sol.y(:, end);

    for idx=1:length(sol.x)
        dt = sol.x(min(idx+1, length(sol.x))) - sol.x(idx);
        plot_robot(sol.y(:, idx), dt, p);
    end
    % Update stance duration
    p.Ts = sol.x(end) - sol.x(1);

    % Simulate flight phase to apex
    tf = t0 + p.max_phase_dt;
    options = odeset("Events", @(t, x)event_apex(x, p));
    sol = ode45(@(t, x)dynamics_flight(x, p), [t0, tf], x, options);
    t0 = sol.x(end);
    x = sol.y(:, end);

    for idx=1:length(sol.x)
        dt = sol.x(min(idx+1, length(sol.x))) - sol.x(idx);
        plot_robot(sol.y(:, idx), dt, p);
    end
end

function [value, isterminal, direction] = event_td(x, p)
    pf = x(5:6);
    value = pf(2);
    isterminal = 1;
    direction = -1;
end

function [value, isterminal, direction] = event_lo(x, p)
    pc = x(1:2);
    pf = x(5:6);
    l = pc - pf;
    value = norm(l) - p.l0;
    isterminal = 1;
    direction = 1;
end

function [value, isterminal, direction] = event_apex(x, p)
    dpc = x(3:4);
    value = dpc(2);
    isterminal = 1;
    direction = -1;
end

function dxdt = dynamics_flight(x, p)
    dpc = x(3:4);
    ddp = [0; -p.g];
    dpf = dpc;
    dxdt = [dpc; ddp; dpf];
end

function dxdt = dynamics_stance(x, p)
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

function p = init_renderer(x0, p)
    % Axis
    hold on; grid on; axis equal;
    % xlim([-0.5, 10]);
    ylim([-0.5, 2]);
    xlabel("x (m)");
    ylabel("z (m)");
    % Init robot pose
    pc = x0(1:2);
    pf = x0(5:6);
    l0_vec = pf - pc;
    l0_norm = norm(l0_vec);
    alpha = acos((p.l2^2 - l0_norm^2 - p.l1^2) / (-2*l0_norm*p.l1));
    theta = atan2(l0_vec(1), -l0_vec(2));
    pc_knee = pc + rot_mat(theta - alpha) * [0; -p.l1];
    % Init plot handlers
    p.s_com = scatter(pc(1), pc(2), 1000, 'r', LineWidth=4);
    p.p_thigh = plot([pc(1); pc_knee(1)], [pc(2); pc_knee(2)], 'k', LineWidth=2);
    p.p_calf = plot([pc_knee(1); pf(1)], [pc_knee(2); pf(2)], 'k', LineWidth=2);
    % p.p_spring = plot([pc(1); pf(1)], [pc(2); pf(2)], 'k', LineWidth=2);
    p.p_com_traj = plot(pc(1), pc(2), 'b', 'LineWidth', 2);
    p.t_vx = text(pc(1)+0.15, 1.3, "v_x = "+string(x0(3)), "FontSize",18);
    p.t_vx_des = text(pc(1)+0.15, 1.5, "v_{x,d} = "+string(x0(3)), "FontSize",18);
    yline(0, LineWidth=3);
    hold off;
end

function plot_robot(x, dt, p)
    if ~p.plot
        return;
    end
    % Retrieve robot pose
    pc = x(1:2);
    pf = x(5:6);
    l0_vec = pf - pc;
    l0_norm = norm(l0_vec);
    alpha = acos((p.l2^2 - l0_norm^2 - p.l1^2) / (-2*l0_norm*p.l1));
    theta = atan2(l0_vec(1), -l0_vec(2));
    p_knee = pc + rot_mat(theta - alpha) * [0; -p.l1];
    % Update plot handlers
    p.s_com.XData = pc(1);
    p.s_com.YData = pc(2);
    p.p_thigh.XData = [pc(1); p_knee(1)];
    p.p_thigh.YData = [pc(2); p_knee(2)];
    p.p_calf.XData = [p_knee(1); pf(1)];
    p.p_calf.YData = [p_knee(2); pf(2)];
    % p.p_spring.XData = [pc(1); pf(1)];
    % p.p_spring.YData = [pc(2); pf(2)];
    p.p_com_traj.XData(end+1) = pc(1);
    p.p_com_traj.YData(end+1) = pc(2);
    p.t_vx.Position = [pc(1)+0.15, 1.3];
    p.t_vx.String = "v_x = "+string(x(3));
    p.t_vx_des.Position = [pc(1)+0.15, 1.5];
    % Center robot
    xlim([pc(1) - 2.25, pc(1) + 1.75]);
    pause(dt);
    % exportgraphics(p.fig, "slip_demo.gif", "Append", true);
end

function R = rot_mat(th)
    R = [cos(th), -sin(th);
         sin(th), cos(th)];
end

