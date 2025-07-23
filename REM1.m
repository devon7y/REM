% REM Memory Simulation Script (Shiffrin & Steyvers, 1997)
% This script simulates a basic recognition memory experiment
% using the core principles of the REM model, including feature
% representation, probabilistic storage, likelihood ratio calculation,
% and decision making based on the average likelihood ratio (phi)

% Set a seed for testing purposes
rng(69); % Keep this for reproducibility of a specific run

% --- Define Model and Simulation Parameters ---
disp('--- REM Model Simulation ---');
disp('Parameters:');

% Model Parameters (from Shiffrin & Steyvers, 1997, default values)
% The number of features that the geometric distribution contains
% Default 1000
param.num_total_features = 20; % Set to 20 as per your request
fprintf('  Number of total features: %d\n', param.num_total_features);

% 'w' - The number of non-zero features that define a word
% Default 20
param.w_word_features = 20; % Set to 20 as per your request
fprintf('  Number of word features (w): %d\n', param.w_word_features);

% 'g' - The geometric distribution parameter for feature values
% Default 0.4
param.g = 0.4;
fprintf('  Geometric distribution parameter (g): %.2f\n', param.g);

% 'u*' - The probability of storing a feature per storage attempt
% Default 0.04
param.u_star = 0.04;
fprintf('  Probability of storing a feature per attempt (u*): %.2f\n', param.u_star);

% 't' - The number of storage attempts (the units of time a word is studied)
% Default 10
param.t = 10;
fprintf('  Number of storage attempts (t): %d\n', param.t);

% 'c' - The probability that a word is copied correctly, assuming it the feature is stored
% Default 0.7
param.c = 0.7;
fprintf('  Probability of correct copy (c): %.2f\n', param.c);

% The threshold for 'old' versus 'new' responses (phi >= criterion)
% Default 1.0
% This is the single criterion for P(H)/P(F) and d' calculations
% The NRS slope calculations use different criteria defined in log_criteria
param.criterion = 1.0;
fprintf('  Recognition criterion: %.1f\n', param.criterion);

% Define the specific log-criteria used in the paper for ROC curve generation (for NRS)
log_criteria = [1.0, -0.8, -0.7, -0.2, 0, 0.2, 0.7, 1.0, 1.25, 1.75, 2.5];
roc_criteria = exp(log_criteria);

% List lengths to simulate
% Default [4, 10, 20, 40, 80] from Figure 3
sim.list_lengths_to_simulate = [4, 10, 20, 40, 80];

% Number of internal simulation runs to average results over for a single point
% IMPORTANT: With your changes, this now represents the number of 'blocks'
% of study/test lists, not individual test trials.
sim.num_simulations_per_point = 1000;
fprintf('  Number of internal simulation blocks per point: %d\n', sim.num_simulations_per_point);

% Number of repetitions to run the entire simulation for standard error of the mean error bar calculation
sim.num_repetitions_for_error_bars = 10;
fprintf('  Number of repetitions for error bars: %d\n', sim.num_repetitions_for_error_bars);

% --- Initialize Storage for Data and Error Bars ---
num_list_lengths = length(sim.list_lengths_to_simulate);

% Store results for each repetition for each list length
all_reps_PH = zeros(num_list_lengths, sim.num_repetitions_for_error_bars);
all_reps_PF = zeros(num_list_lengths, sim.num_repetitions_for_error_bars);
all_reps_d_prime = zeros(num_list_lengths, sim.num_repetitions_for_error_bars);
all_reps_NRS = zeros(num_list_lengths, sim.num_repetitions_for_error_bars);

disp('Starting simulations for different list lengths...');

% --- Start Parallel Pool (if not already open) ---
if isempty(gcp('nocreate'))
    parpool;
end

% --- Loop to Simulate For Each List Length ---
for ll_idx = 1:num_list_lengths
    
    current_list_length = sim.list_lengths_to_simulate(ll_idx);
    
    fprintf('\n--- List Length: %d ---\n', current_list_length);
    
    % --- Loop to Repeat Simulations For Each List Length (to calculate Error Bars) ---
    parfor rep_idx = 1:sim.num_repetitions_for_error_bars
        
        % --- Initialize Performance Metrics for current list length and repetition ---
        total_hits = 0;
        total_false_alarms = 0;
        
        % Initialize arrays to store phi values for d' and NRS calculation for THIS list length and THIS repetition
        % Now stores (num_simulations_per_point * current_list_length) phis for targets and lures
        total_test_trials_per_rep = sim.num_simulations_per_point * current_list_length;
        all_phi_targets = zeros(1, total_test_trials_per_rep);
        all_phi_lures = zeros(1, total_test_trials_per_rep);
        
        phi_target_idx_counter = 0; % Counter for filling all_phi_targets
        phi_lure_idx_counter = 0;   % Counter for filling all_phi_lures
        
        % --- Loop to Simulate Each Point (now each 'block' of study/test) ---
        for sim_idx = 1:sim.num_simulations_per_point
            % --- Generate Study List and Lure Pool for this block ---
            % Generate 2 * list_length words in total for this block
            total_words_for_block = 2 * current_list_length;
            all_generated_words_for_block = zeros(total_words_for_block, param.num_total_features);

            for j = 1:total_words_for_block
                all_generated_words_for_block(j, :) = generate_word_vector(param.num_total_features, param.w_word_features, param.g);
            end

            % First half are study words (targets), second half are lures for this block
            study_words = all_generated_words_for_block(1:current_list_length, :);
            lure_words_pool = all_generated_words_for_block(current_list_length + 1 : end, :);

            % Initialize episodic images (only for the studied words)
            memory_images = zeros(current_list_length, param.num_total_features);
            
            % For the number of words in the study list, store them in memory
            for i = 1:current_list_length
                memory_images(i, :) = store_word_into_memory(study_words(i, :), param.u_star, param.t, param.c, param.g);
            end
            
            % --- Simulate Target Trials (All Studied Words in this block) ---
            for target_word_idx = 1:current_list_length
                probe_target = study_words(target_word_idx, :);
                
                % Calculate the overall odds (phi) for the target probe
                phi_target = calculate_overall_odds(probe_target, memory_images, param.c, param.g);
                
                % Make recognition decision
                if phi_target >= param.criterion
                    total_hits = total_hits + 1;
                end
                
                % Store phi_target
                phi_target_idx_counter = phi_target_idx_counter + 1;
                all_phi_targets(phi_target_idx_counter) = phi_target;
            end
            
            % --- Simulate Lure Trials (All Lures from the pre-generated pool in this block) ---
            for lure_word_idx = 1:current_list_length
                probe_lure = lure_words_pool(lure_word_idx, :);
                
                % Calculate the overall odds (phi) for the lure probe
                phi_lure = calculate_overall_odds(probe_lure, memory_images, param.c, param.g);
                
                % Make recognition decision
                if phi_lure >= param.criterion
                    total_false_alarms = total_false_alarms + 1;
                end
                
                % Store phi_lure
                phi_lure_idx_counter = phi_lure_idx_counter + 1;
                all_phi_lures(phi_lure_idx_counter) = phi_lure;
            end
        end % End of sim_idx loop
        
        % --- Calculate and Store Results for Current List Length and Repetition ---
        % Total number of *effective* trials is now sim.num_simulations_per_point * current_list_length
        total_effective_trials_targets = sim.num_simulations_per_point * current_list_length;
        total_effective_trials_lures = sim.num_simulations_per_point * current_list_length;

        % Probability of hits for the current criterion
        P_H_current = total_hits / total_effective_trials_targets; % Denominator changed
        all_reps_PH(ll_idx, rep_idx) = P_H_current; % Sliced variable
        
        % Probability of false alarms for the current criterion
        P_F_current = total_false_alarms / total_effective_trials_lures; % Denominator changed
        all_reps_PF(ll_idx, rep_idx) = P_F_current; % Sliced variable
        
        % Calculate d' for the current list length and repetition
        z_hit_current = norminv(P_H_current, 0, 1);
        z_false_alarm_current = norminv(P_F_current, 0, 1);
        d_prime_current = z_hit_current - z_false_alarm_current;
        all_reps_d_prime(ll_idx, rep_idx) = d_prime_current; % Sliced variable
        
        % --- NRS (Normal-ROC Slope) Calculation for Current List Length and Repetition ---
        % Calculate P(H) and P(F) for each ROC criterion from the collected phi distributions
        roc_PH_current_ll_rep = zeros(1, length(roc_criteria));
        roc_PF_current_ll_rep = zeros(1, length(roc_criteria));
        for i = 1:length(roc_criteria)
            current_roc_criterion = roc_criteria(i);
            roc_PH_current_ll_rep(i) = sum(all_phi_targets >= current_roc_criterion) / total_effective_trials_targets; % Denominator changed
            roc_PF_current_ll_rep(i) = sum(all_phi_lures >= current_roc_criterion) / total_effective_trials_lures;     % Denominator changed
        end
        
        % Convert ROC points to z-scores
        z_roc_PH = norminv(roc_PH_current_ll_rep, 0, 1);
        z_roc_PF = norminv(roc_PF_current_ll_rep, 0, 1);
        
        % Filter out non-finite values (Inf or -Inf) as they cannot be used in linear regression
        valid_indices_roc = isfinite(z_roc_PH) & isfinite(z_roc_PF);
        z_roc_PH_filtered = z_roc_PH(valid_indices_roc);
        z_roc_PF_filtered = z_roc_PF(valid_indices_roc);
        
        nrs_current = NaN; % Initialize NRS as NaN in case calculation fails
        if length(z_roc_PF_filtered) >= 2
            % Fit a linear regression line to the z-ROC points
            coefficients_roc = polyfit(z_roc_PF_filtered, z_roc_PH_filtered, 1);
            nrs_current = coefficients_roc(1); % The first coefficient is the slope
        else
            % disp('Warning: Not enough valid (P(F), P(H)) pairs for NRS calculation for this list length and repetition.');
        end
        all_reps_NRS(ll_idx, rep_idx) = nrs_current; % Sliced variable
    end
end

disp('--- All Simulations Complete ---');

% --- Calculate Mean and Standard Deviation Across Repetitions ---
mean_PH = mean(all_reps_PH, 2)';
std_PH = std(all_reps_PH, 0, 2)';

mean_PF = mean(all_reps_PF, 2)';
std_PF = std(all_reps_PF, 0, 2)';

mean_d_prime = mean(all_reps_d_prime, 2)';
std_d_prime = std(all_reps_d_prime, 0, 2)';

mean_NRS = mean(all_reps_NRS, 2)';
std_NRS = std(all_reps_NRS, 0, 2)';

fprintf('\n--- Final Mean Results Across Repetitions ---\n');
for i = 1:num_list_lengths
    fprintf('List Length %d:\n', sim.list_lengths_to_simulate(i));
    fprintf('  Mean P(H): %.4f (SD: %.4f)\n', mean_PH(i), std_PH(i));
    fprintf('  Mean P(F): %.4f (SD: %.4f)\n', mean_PF(i), std_PF(i));
    fprintf('  Mean d'': %.4f (SD: %.4f)\n', mean_d_prime(i), std_d_prime(i));
    fprintf('  Mean NRS: %.4f (SD: %.4f)\n', mean_NRS(i), std_NRS(i));
end

% --- Plotting Results with Error Bars ---
figure; 
set(gcf, 'Units', 'normalized', 'Position', [0.1 0.1 0.5 0.9]);

x_labels = arrayfun(@num2str, sim.list_lengths_to_simulate, 'UniformOutput', false);

subplot(3, 1, 1);
errorbar(sim.list_lengths_to_simulate, mean_d_prime, std_d_prime, 'o-', 'LineWidth', 1.5, 'Color', 'g');
title('d-prime (d'')');
xlabel('List Length (L)');
ylabel('d''');
grid on;
xticks(sim.list_lengths_to_simulate);
xticklabels(x_labels);
xlim([min(sim.list_lengths_to_simulate) - 2, max(sim.list_lengths_to_simulate) + 2]);

subplot(3, 1, 2);
errorbar(sim.list_lengths_to_simulate, mean_PH, std_PH, 'o-', 'LineWidth', 1.5, 'Color', 'b', 'DisplayName', 'Hit Rate P(H)');
hold on;
errorbar(sim.list_lengths_to_simulate, mean_PF, std_PF, 's-', 'LineWidth', 1.5, 'Color', 'r', 'DisplayName', 'False Alarm Rate P(F)');
hold off;
title('Hit Rate and False Alarm Rate');
xlabel('List Length (L)');
ylabel('Probability');
grid on;
xticks(sim.list_lengths_to_simulate);
xticklabels(x_labels);
xlim([min(sim.list_lengths_to_simulate) - 2, max(sim.list_lengths_to_simulate) + 2]);
legend('show', 'Location', 'best');

subplot(3, 1, 3);
errorbar(sim.list_lengths_to_simulate, mean_NRS, std_NRS, 'o-', 'LineWidth', 1.5, 'Color', 'm');
title('Normal ROC Slope (NRS)');
xlabel('List Length (L)');
ylabel('NRS');
grid on;
xticks(sim.list_lengths_to_simulate);
xticklabels(x_labels);
xlim([min(sim.list_lengths_to_simulate) - 2, max(sim.list_lengths_to_simulate) + 2]);
sgtitle('REM.1 List Length and Predictions with SEM Error Bars');

% --- Helper Functions ---
function word_vector = generate_word_vector(num_total_features, w_word_features, g_param)
    word_vector = zeros(1, num_total_features);
    feature_indices = randperm(num_total_features, w_word_features);
    feature_values = geornd(g_param, 1, w_word_features) + 1;
    word_vector(feature_indices) = feature_values;
end

function episodic_image = store_word_into_memory(word_vector, u_star, t_attempts, c_copy, g_param)
    episodic_image = zeros(size(word_vector));
    word_nz_indices = find(word_vector ~= 0);

    if isempty(word_nz_indices)
        return;
    end

    num_active_features = length(word_nz_indices);
    p_stored_at_all = 1 - (1 - u_star)^t_attempts;
    is_stored_mask = rand(1, num_active_features) < p_stored_at_all;
    stored_features_indices_in_nz = word_nz_indices(is_stored_mask);

    if isempty(stored_features_indices_in_nz)
        return;
    end

    is_correct_copy_mask = rand(1, length(stored_features_indices_in_nz)) < c_copy;
    correct_copy_global_indices = stored_features_indices_in_nz(is_correct_copy_mask);
    erroneous_copy_global_indices = stored_features_indices_in_nz(~is_correct_copy_mask);

    episodic_image(correct_copy_global_indices) = word_vector(correct_copy_global_indices);

    if ~isempty(erroneous_copy_global_indices)
        num_erroneous = length(erroneous_copy_global_indices);
        erroneous_feature_values = geornd(g_param, 1, num_erroneous) + 1;
        episodic_image(erroneous_copy_global_indices) = erroneous_feature_values;
    end
end

function lambda_j = calculate_lambda_single_image(probe_vector, image_vector, c_copy, g_param)
    lambda_j = 1.0;
    common_nz_indices = find(probe_vector ~= 0 & image_vector ~= 0);

    for k_idx = 1:length(common_nz_indices)
        k = common_nz_indices(k_idx);
        probe_val = probe_vector(k);
        image_val = image_vector(k);

        if probe_val == image_val
            denominator = g_param * ((1 - g_param)^(probe_val - 1));
            if denominator > eps
                 lambda_j = lambda_j * ((1 - c_copy) + c_copy / denominator);
            else
                lambda_j = lambda_j * 1e10; % Using a large constant to represent a very high likelihood
            end
        else
            lambda_j = lambda_j * (1 - c_copy);
        end
    end
end

function phi = calculate_overall_odds(probe_vector, memory_images, c_copy, g_param)
    sum_lambda = 0;
    n_images = size(memory_images, 1);
    for i = 1:n_images
        current_image = memory_images(i, :);
        sum_lambda = sum_lambda + calculate_lambda_single_image(probe_vector, current_image, c_copy, g_param);
    end
    if n_images > 0
        phi = sum_lambda / n_images;
    else
        phi = 0;
    end
end