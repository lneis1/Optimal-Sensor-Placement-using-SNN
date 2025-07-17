%% 2D Sensor Placement via SNN-QP with Random Obstacles and Tradeoff Objective

% PARAMETERS
Lx = 10; Ly = 6;           % Room dimensions
nx = 12; ny = 8;           % Grid resolution
r = 1.5;                   % Sensor radius
%nx = ceil(Lx / 0.5*r);
%ny = ceil(Ly / 0.5*r);
k_spring = 5;              % Overlap penalty strength
samples = 1000;             % For coverage estimation
num_obstacles = randi(10);         % Number of random obstacles

% SNN-QP solver parameters
t_end = 500;
k0 = 0.09;
k1 = 0.05;

%% 1) ROOM AND RANDOM OBSTACLES
room = polyshape([0 Lx Lx 0], [0 0 Ly Ly]);
obstacles = polyshape();
for i = 1:num_obstacles
    w = 0.5 + rand();  % width
    h = 0.5 + rand();  % height
    x = (Lx - w) * rand();  % keep inside room
    y = (Ly - h) * rand();
    obs = polyshape([x x+w x+w x], [y y y+h y+h]);
    obstacles = union(obstacles, obs);
end


free_space = subtract(room, obstacles);

%% 2) CANDIDATE SENSOR LOCATIONS (in free space only)
[xg, yg] = meshgrid(linspace(0, Lx, nx), linspace(0, Ly, ny));
locs2d = [xg(:), yg(:)];
in_free = isinterior(free_space, locs2d(:,1), locs2d(:,2));
locs = locs2d(in_free, :);
assert(all(isinterior(free_space, locs(:,1), locs(:,2))), 'Error: Some candidate sensors are inside obstacles');
N = size(locs, 1);

%% 3) COVERAGE WEIGHTS
theta = 2*pi*rand(samples,1);
rad = r * sqrt(rand(samples,1));
unit_disk = [cos(theta), sin(theta)].*rad;
w = zeros(N,1);
for i = 1:N
    pts = locs(i,:) + unit_disk;
    w(i) = pi*r^2 * mean(isinterior(free_space, pts(:,1), pts(:,2)));
end

%% 4) SPRING PENALTY MATRIX
D = pdist2(locs, locs);
P = zeros(N);
for i = 1:N
    for j = i+1:N
        if D(i,j) < 2.5*r
            P(i,j) = k_spring * (2*r - D(i,j))^2;
            P(j,i) = P(i,j);
        end
    end
end

%% 5) COVERAGE CONSTRAINTS OVER FREE SPACE ONLY
[Xg2, Yg2] = meshgrid(linspace(0, Lx, 40), linspace(0, Ly, 24));
eval_pts = [Xg2(:), Yg2(:)];
in_free_eval = isinterior(free_space, eval_pts(:,1), eval_pts(:,2));
eval_pts = eval_pts(in_free_eval, :);  % Ensure coverage only for free space
J = size(eval_pts, 1);
M = zeros(J, N);
for j = 1:J
    for i = 1:N
        if norm(eval_pts(j,:) - locs(i,:)) <= r
            M(j,i) = 1;
        end
    end
end

%% QP FORMULATION WITH TRADEOFF OBJECTIVE
lambda = 0.001;  % weight on sensor count
A = P;
b = -w + lambda * ones(N,1);

C = [-M; eye(N); -eye(N)];
d = [-ones(J,1); -ones(N,1); zeros(N,1)];
x0 = rand(N,1);

%% 6) SOLVE
[t, X] = snn_solver(A, b, C, d, t_end, x0, k0, k1);
y_final = X(end,:)';
%% (Optional) Track objective function value over time
%% Compute and plot QP objective over time
obj_vals = zeros(length(t), 1);
for ti = 1:length(t)
    y_t = X(ti, :)';
    obj_vals(ti) = 0.5 * y_t' * A * y_t + b' * y_t;
end

% Compute convergence rate
initial_obj = obj_vals(1);
final_obj = obj_vals(end);
rel_drop = (initial_obj - final_obj) / abs(initial_obj);
fprintf('Initial QP objective: %.4f\n', initial_obj);
fprintf('Final QP objective: %.4f\n', final_obj);
fprintf('Relative drop (convergence rate): %.4f%%\n', 100 * rel_drop);

% Plot convergence curve
figure;
plot(t, obj_vals, 'LineWidth', 1.5);
xlabel('Time'); ylabel('Objective Function Value');
title(sprintf('Convergence of QP Objective (Drop: %.2f%%)', 100 * rel_drop));
grid on;

selected = find(y_final > 0.5);


obj_vals = zeros(length(t), 1);
for ti = 1:length(t)
    y_t = X(ti, :)';
    obj_vals(ti) = 0.5 * y_t' * A * y_t + b' * y_t;
end

figure;
plot(t, obj_vals, 'LineWidth', 1.5);
xlabel('Time'); ylabel('Objective Function Value');
title('Convergence of QP Objective');
grid on;

%% 7) PLOTS
%figure;
plot(t, X, 'LineWidth', 1.2);
xlabel('Time'); ylabel('y_i');
title('SNN-QP Dynamics of Sensor Variables');
legend(arrayfun(@(i) sprintf('y_{%d}', i), 1:N, 'UniformOutput', false));

%figure;
stem(X(end, :), 'filled');
xlabel('Sensor index'); ylabel('Final y_i value');
title('Optimized Sensor Activations');

%figure;
plot(free_space, 'FaceColor', [0.9 0.9 0.9]); hold on;
plot(obstacles, 'FaceColor', [0.5 0.5 0.5]);
scatter(locs(:,1), locs(:,2), 30, 'k');
scatter(locs(selected,1), locs(selected,2), 100, 'r', 'filled');
viscircles(locs(selected,:), r*ones(size(selected)), 'Color', 'r');
axis equal tight;
xlabel('X'); ylabel('Y');
title('Selected Sensor Placements in 2D Room');
legend({'Free space','Obstacle','Candidates','Selected'});

%% 8) COVERAGE METRIC + VISUALIZATION
test_xy = [Lx*rand(10000,1), Ly*rand(10000,1)];
in_free = isinterior(free_space, test_xy(:,1), test_xy(:,2));
in_coverage = false(10000,1);
for i = 1:length(selected)
    dist = vecnorm(test_xy - locs(selected(i),:), 2, 2);
    in_coverage = in_coverage | (dist <= r);
end
coverage_rate = sum(in_free & in_coverage) / sum(in_free);

fprintf('Selected sensors: %s\n', mat2str(selected'));
fprintf('Coverage over free space: %.2f%%\n', 100 * coverage_rate);
fprintf('Candidate sensor locations: %d\n', N);
fprintf('Selected sensor count: %d\n', length(selected));

figure;
plot(free_space, 'FaceColor', [0.9 0.9 0.9]); hold on;
plot(obstacles, 'FaceColor', [0.5 0.5 0.5]);
scatter(test_xy(in_free & in_coverage,1), test_xy(in_free & in_coverage,2), 10, [0 0.6 0], 'filled');
scatter(test_xy(in_free & ~in_coverage,1), test_xy(in_free & ~in_coverage,2), 10, [1 0 0], 'filled');
scatter(locs(:,1), locs(:,2), 30, 'k');
scatter(locs(selected,1), locs(selected,2), 100, 'b', 'filled');
viscircles(locs(selected,:), r*ones(size(selected)), 'Color', 'b');
axis equal tight;
xlabel('X'); ylabel('Y');
title(sprintf('Coverage: %.2f%% | Candidates: %d | Selected: %d', ...
    100 * coverage_rate, N, length(selected)));
legend({'Free space','Obstacle','Covered','Uncovered','Candidates','Selected'});
