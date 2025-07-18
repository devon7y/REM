%% Representation

% 1. Environmental Feature Distribution
% Global parameter for the environmental geometric distribution
gen_g_env = 0.1;

% Define a sufficiently large maximum feature value for the environmental distribution
max_feature_value_possible = 100; 

% Pre-calculate the probabilities P(v) for v = 1 to max_feature_value_possible
feature_base_rates_distribution = zeros(1, max_feature_value_possible);
for v = 1:max_feature_value_possible
    feature_base_rates_distribution(v) = gen_g_env * (1 - gen_g_env)^(v - 1);
end
% Normalize the distribution to ensure it sums to 1 (important for sampling later)
feature_base_rates_distribution = feature_base_rates_distribution / sum(feature_base_rates_distribution);


% 2. Lexical Prototypes (Word Representations)
% Model parameters for word generation
number_of_prototypes = 1000; % n: Total size of the vocabulary
number_of_word_features = 20; % w: Number of non-zero features per word
gen_g_H = 0.25; % g_H: Parameter for high-frequency words
gen_g_L = 0.05; % g_L: Parameter for low-frequency words
prop_high_freq = 0.5; % Proportion of high-frequency words in the lexicon

% Error handling for n and w
if number_of_prototypes <= 0 || mod(number_of_prototypes, 1) ~= 0
    error('number_of_prototypes must be a positive integer.');
end
if number_of_word_features <= 0 || mod(number_of_word_features, 1) ~= 0
    error('number_of_word_features must be a positive integer.');
end
if number_of_word_features > max_feature_value_possible
    error('number_of_word_features cannot exceed max_feature_value_possible.');
end

word_prototypes = zeros(number_of_prototypes, max_feature_value_possible);
word_frequencies = cell(1, number_of_prototypes); % To store 'high' or 'low' frequency type

% Create cumulative distribution functions (CDFs) for g_H and g_L to enable sampling
% The range of values for a geometric distribution can be large, so ensure CDF covers sufficient range
cdf_g_H = cumsum(gen_g_H * (1 - gen_g_H).^(0:max_feature_value_possible-1));
cdf_g_L = cumsum(gen_g_L * (1 - gen_g_L).^(0:max_feature_value_possible-1));

for i = 1:number_of_prototypes
    % Assign frequency type
    if rand() < prop_high_freq
        word_frequencies{i} = 'high';
        current_cdf = cdf_g_H;
    else
        word_frequencies{i} = 'low';
        current_cdf = cdf_g_L;
    end
    
    % Generate w non-zero features
    % Initialize a temporary vector to hold the w non-zero features
    non_zero_features_temp = zeros(1, number_of_word_features);
    for j = 1:number_of_word_features
        r = rand();
        % Sample a feature value from the geometric distribution using its CDF
        % find returns the first index where the condition is true
        non_zero_features_temp(j) = find(current_cdf >= r, 1, 'first');
    end
    
    % Place these features into randomly chosen positions within the word_prototype vector
    % Ensure the selected positions are unique for this word
    feature_indices = randperm(max_feature_value_possible, number_of_word_features);
    word_prototypes(i, feature_indices) = non_zero_features_temp;
end

disp('Representation section complete.');

%% Storage

% Model parameters for storage
number_of_words_to_study = 50; % z: Number of words to be studied (targets)
units_of_time = 10; % x: Number of storage attempts (e.g., for 'strong' words)
probability_of_storage = 0.5; % u*: Probability of a feature being stored in an attempt
copying_accuracy = 0.9; % c: Probability of correct feature transfer

% 1. Select Study Words
% Randomly select z word prototypes from the word_prototypes array
study_indices = randperm(number_of_prototypes, number_of_words_to_study);
studied_word_prototypes = word_prototypes(study_indices, :);

% 2. Initialize Stored Traces
% Initialize z empty vectors, each of length max_feature_value_possible, with all values 0
stored_episodic_traces = zeros(number_of_words_to_study, max_feature_value_possible);

% 3. Simulate Storage Attempts
for i = 1:number_of_words_to_study
    current_word_prototype = studied_word_prototypes(i, :);
    current_stored_trace = zeros(1, max_feature_value_possible); % Temporary trace for this word
    
    for attempt = 1:units_of_time
        for feature_pos = 1:max_feature_value_possible
            % If the feature at this position has NOT already been stored
            if current_stored_trace(feature_pos) == 0
                % Probability of storage attempt success (u*)
                if rand() < probability_of_storage
                    % Copying accuracy (c)
                    if rand() < copying_accuracy
                        % Correctly copy feature from prototype
                        current_stored_trace(feature_pos) = current_word_prototype(feature_pos);
                    else
                        % Error: Store a random feature value from environmental distribution
                        % Sample from the environmental feature_base_rates_distribution
                        % Generate a random value between 0 and 1
                        r_val = rand();
                        % Find which bin this random value falls into based on CDF
                        cumulative_env_dist = cumsum(feature_base_rates_distribution);
                        random_feature_value = find(cumulative_env_dist >= r_val, 1, 'first');
                        current_stored_trace(feature_pos) = random_feature_value;
                    end
                end
            end
        end
    end
    % Store the finalized episodic trace for this word
    stored_episodic_traces(i, :) = current_stored_trace;
end

disp('Storage section complete.');

%% Retrieval

% 1. Create Probe Traces
number_of_test_probes = 2 * number_of_words_to_study; % 2z probes (z targets + z distractors)
probe_vectors = zeros(number_of_test_probes, max_feature_value_possible);
probe_types = cell(1, number_of_test_probes); % To store 'target' or 'distractor' type

% Add targets (studied words)
probe_vectors(1:number_of_words_to_study, :) = studied_word_prototypes;
probe_types(1:number_of_words_to_study) = {'target'};

% Add distractors (randomly selected non-studied words)
% Ensure distractors are not from the studied set
non_studied_indices = setdiff(1:number_of_prototypes, study_indices);
% Use datasample to pick without replacement
distractor_indices = datasample(non_studied_indices, number_of_words_to_study, 'Replace', false);
probe_vectors(number_of_words_to_study+1:end, :) = word_prototypes(distractor_indices, :);
probe_types(number_of_words_to_study+1:end) = {'distractor'};

% Initialize storage for all likelihood ratios
all_probe_likelihood_ratios = cell(1, number_of_test_probes);

% Likelihood Ratio Calculation Function logic (implemented inline for clarity)
for p = 1:number_of_test_probes
    current_probe_vector = probe_vectors(p, :);
    lambda_j_values_for_probe = zeros(1, size(stored_episodic_traces, 1)); % One lambda_j for each stored trace
    
    for j = 1:size(stored_episodic_traces, 1) % Iterate through each stored episodic trace
        current_stored_trace = stored_episodic_traces(j, :);
        total_likelihood_ratio = 1.0; % Initialize for multiplicative combination
        
        for feature_pos = 1:max_feature_value_possible % Iterate through each feature position
            probe_feature = current_probe_vector(feature_pos);
            trace_feature = current_stored_trace(feature_pos);
            
            local_lambda = 1.0; % Default local contribution (for features providing no evidence)
            
            % Get P(v) for the probe feature value from the environmental distribution
            % Ensure valid index and value exists in distribution
            prob_probe_feature_env = 0;
            if probe_feature > 0 && probe_feature <= max_feature_value_possible
                prob_probe_feature_env = feature_base_rates_distribution(probe_feature);
            end

            % Implement Equations 2, 3, 4A, 4B logic based on Shiffrin & Steyvers (1997)
            if trace_feature > 0 % Feature is present in the stored trace (I_j(f) > 0)
                if probe_feature == trace_feature % Match: I_j(f) = V(f) > 0 (Equation 2)
                    % P(V(f)|s) / P(V(f)|d) = 1 / P(V(f))
                    if prob_probe_feature_env > 0
                        local_lambda = 1 / prob_probe_feature_env;
                    else
                        local_lambda = 1e-10; % Avoid division by zero, very small likelihood for extremely rare features
                    end
                else % Mismatch: I_j(f) > 0 and V(f) != I_j(f) (Equation 3)
                    % P(V(f)|s) / P(V(f)|d) = 0 / P(V(f)) = 0
                    local_lambda = 1e-10; % Effectively zero, representing a strong mismatch
                end
            else % Feature is absent in the stored trace (I_j(f) = 0)
                if probe_feature > 0 % Probe feature present (V(f) > 0), trace feature absent (Equation 4A)
                    % This represents a feature from the studied word that was not stored or stored incorrectly.
                    % P(V(f)|s) / P(V(f)|d) where P(V(f)|s) is the probability of this configuration.
                    % The paper often implies this is (1 - probability_of_correct_storage_of_this_feature)
                    % which is (1 - u*c) for a feature that was originally present.
                    local_lambda = (1 - probability_of_storage * copying_accuracy);
                    if local_lambda <= 0
                        local_lambda = 1e-10; % Ensure it's not zero or negative
                    end
                else % Probe feature absent (V(f) = 0), trace feature absent (Equation 4B)
                    % This corresponds to a feature position that is zero in both the probe and the trace.
                    % These positions provide no unique evidence for old/new.
                    local_lambda = 1.0; 
                end
            end
            
            total_likelihood_ratio = total_likelihood_ratio * local_lambda;
        end
        lambda_j_values_for_probe(j) = total_likelihood_ratio;
    end
    all_probe_likelihood_ratios{p} = lambda_j_values_for_probe;
end

disp('Retrieval section complete.');

%% Bayesian decision

% Initialize results storage
num_hits = 0;
num_false_alarms = 0;
num_misses = 0;
num_correct_rejections = 0;

% Store individual probe decisions and odds for later analysis (e.g., ROC)
probe_decisions = cell(1, number_of_test_probes);
probe_odds = zeros(1, number_of_test_probes);

% Decision criterion: Odds > 1.0 corresponds to P(Old|Data) > 0.5
criterion_odds = 1.0; 

for p = 1:number_of_test_probes
    current_lambda_j_values = all_probe_likelihood_ratios{p};
    
    % 1. Calculate Odds of 'Old'
    % Average of all lambda_j values for the current probe across all stored traces
    odds_old = mean(current_lambda_j_values);
    probe_odds(p) = odds_old;
    
    % 2. Make Decision
    if odds_old > criterion_odds
        decision = 'Old';
    else
        decision = 'New';
    end
    probe_decisions{p} = decision;
    
    % 3. Record Results for performance evaluation
    if strcmp(probe_types{p}, 'target')
        if strcmp(decision, 'Old')
            num_hits = num_hits + 1;
        else
            num_misses = num_misses + 1;
        end
    else % probe_type is 'distractor'
        if strcmp(decision, 'Old')
            num_false_alarms = num_false_alarms + 1;
        else
            num_correct_rejections = num_correct_rejections + 1;
        end
    end
end

% Display summary results (for basic verification of simulation outcome)
fprintf('\nBayesian Decision Section Complete.\n');
fprintf('Total Targets: %d\n', number_of_words_to_study);
fprintf('Total Distractors: %d\n', number_of_words_to_study);
fprintf('Hits: %d\n', num_hits);
fprintf('Misses: %d\n', num_misses);
fprintf('False Alarms: %d\n', num_false_alarms);
fprintf('Correct Rejections: %d\n', num_correct_rejections);

% Calculate and display basic performance metrics
hit_rate = num_hits / number_of_words_to_study;
fa_rate = num_false_alarms / number_of_words_to_study;
fprintf('Hit Rate: %.4f\n', hit_rate);
fprintf('False Alarm Rate: %.4f\n', fa_rate);

disp('Simulation complete.');