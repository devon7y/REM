% REM Memory Simulation Script (Shiffrin & Steyvers, 1997)
% This script simulates a basic recognition memory experiment
% using the core principles of the REM model, including feature
% representation, probabilistic storage, likelihood ratio calculation,
% and decision making based on the average likelihood ratio (phi)

% Set a seed for testing purposes
rng(69);

% --- 1. Define Model and Simulation Parameters ---

disp('--- REM Model Simulation ---');
disp('Defining parameters...');

% Model Parameters (from Shiffrin & Steyvers, 1997, default values)

% The number of features that the geometric distribution contains
% Default 1000
param.num_total_features = 1000; 

% 'w' - The number of non-zero features that define a word
% Default 20
param.w_word_features = 20;    

% 'g' - The geometric distribution parameter for feature values
% Default 0.4
param.g = 0.4;                   

% 'u*' - The probability of storing a feature per storage attempt
% Default 0.04
param.u_star = 0.04;

% 't' - The number of storage attempts (the units of time a word is studied)
% Default 10
param.t = 10;

% 'c' - The probability that a word is copied correctly, assuming it the feature is stored
% Default 0.7
param.c = 0.7;

% The threshold for 'old' versus 'new' responses (phi >= criterion)
% Default 1.0
param.criterion = 1.0;

% 'n' - Number of words in the study list
sim.list_length = 80;           

% Number of simulation runs to average results over
sim.num_simulations = 1000; 

% --- 2. Initialize Performance Metrics ---

total_hits = 0;
total_false_alarms = 0;

disp(['Running ' num2str(sim.num_simulations) ' simulations with list length ' num2str(sim.list_length) '...']);

% --- 3. Main Simulation Loop ---

for sim_idx = 1:sim.num_simulations
    % --- 3.1. Generate Study List and Simulate Storage ---
    % Thought: First, we need a set of unique words to represent our study list
    % Then, for each word, we simulate the storage process to create an
    % "episodic image" in memory, which is an incomplete and error-prone copy
    
    % Initialize an empty array of word vectors
    study_words = cell(sim.list_length, 1);

    % Initialize an empty array of episodic images
    memory_images = cell(sim.list_length, 1);

    % For the number of words in the list length
    for i = 1:sim.list_length
        % Create a word vector representation from the geometric distribution
        study_words{i} = generate_word_vector(param.num_total_features, param.w_word_features, param.g);
        
        % Store the word in memory as an episodic image, which may have errors
        memory_images{i} = store_word_into_memory(study_words{i}, param.u_star, param.t, param.c, param.g);
    end

    % --- 3.2. Simulate a Target Trial (Recognition of a Studied Word) ---
    % Thought: Pick one word from the study list as the probe. Calculate phi
    % based on comparing it to all stored images. Check if it's a 'hit'

    % Randomly select one studied word to be the target probe
    target_idx = randi(sim.list_length);
    probe_target = study_words{target_idx};

    % Calculate the overall odds (phi) for the target probe
    phi_target = calculate_overall_odds(probe_target, memory_images, param.c, param.g);

    % Make recognition decision
    % Since the probe was in the study list, if the average likelihood
    % ratio is above 1, it is a hit
    if phi_target >= param.criterion
        total_hits = total_hits + 1;
    end

    % --- 3.3. Simulate a Distractor Trial (Recognition of an Unstudied Word) ---
    % Thought: Generate a brand new word that was not studied. Calculate phi
    % based on comparing it to all stored images. Check if it's a 'false alarm'.

    % Generate a new word that was NOT in the study list (a distractor)
    % For large num_total_features, generating a truly unique word is highly probable.
    probe_distractor = generate_word_vector(param.num_total_features, param.w_word_features, param.g);

    % Calculate the overall odds (phi) for the distractor probe
    phi_distractor = calculate_overall_odds(probe_distractor, memory_images, param.c, param.g);

    % Make recognition decision
    % Since the probe was NOT in the study list, if the average likelihood
    % ratio is above 1, it is a false alarm
    if phi_distractor >= param.criterion
        total_false_alarms = total_false_alarms + 1;
    end

end % End of main simulation loop

% --- 4. Calculate and Display Results ---
P_H = total_hits / sim.num_simulations;
P_F = total_false_alarms / sim.num_simulations;

disp('--- Simulation Results ---');
disp(['Hit Rate (P(H)): ' num2str(P_H)]);
disp(['False Alarm Rate (P(F)): ' num2str(P_F)]);
disp('--------------------------');

% --- 4. Helper Functions ---

% Function to generate a word vector with 'w' non-zero features; the remaining features are zeros
% Thought: Words are defined by a subset of features
% The non-zero features have values drawn from a geometric distribution
function word_vector = generate_word_vector(num_total_features, w_word_features, g_param)
    % Initialize all features to zero
    word_vector = zeros(1, num_total_features);
    
    % Randomly choose 'w' unique positions in the vector for non-zero features
    % Thought: These are the 'active' features that define this specific word
    feature_indices = randperm(num_total_features, w_word_features);
    
    % Generate feature values using a geometric distribution
    % geornd(p) gives number of failures before first success, so +1
    % Requires Statistics and Machine Learning Toolbox
    feature_values = geornd(g_param, 1, w_word_features) +1;
    
    % Assign the geometrically sampled features to the word_vector at the
    % randomly selected positions in the vector
    word_vector(feature_indices) = feature_values;
end

% Function to simulate storing a word into memory
% Thought: The stored image is an incomplete and error-prone copy.
function episodic_image = store_word_into_memory(word_vector, u_star, t_attempts, c_copy, g_param)
    % Initialize episodic image with zeros
    episodic_image = zeros(size(word_vector));
    
    % Get indices of the word's non-zero features
    % ~= means not equal to; so, this is finding values in the vector not equal to 0
    word_nz_indices = find(word_vector ~= 0);
    
    % Iterate through each of the word's non-zero features for storage attempts
    % Thought: Only non-zero features from the original word can be stored
    % Once a feature is stored, storage is not attempted again
    for k = word_nz_indices
        original_val = word_vector(k);
        
        % Initially, the feature has not been stored
        feature_stored = false;
        
        % For each storage attempt
        for attempt = 1:t_attempts

            % Pick a random number between U(0, 1)
            % There is a 4% of it being less than u_star if u_star is 0.04.
            % This means that there is a 4% chance of a feature being stored
            if rand() < u_star

                % Pick a random number between U(0, 1)
                % There is a 70% of it being less than c_copy if c_copy is 0.7
                % This means that there is a 70% chance of a feature being stored correctly
                if rand() < c_copy

                    % Store the feature correctly
                    episodic_image(k) = original_val;
                else 
                    % Generate feature values using a geometric distribution
                    % geornd(p) gives number of failures before first success, so +1
                    % Requires Statistics and Machine Learning Toolbox
                    random_val = geornd(g_param) +1; 

                    % Store the feature incorrectly
                    episodic_image(k) = random_val;
                end
                
                % Mark as stored so no more attempts occur for this feature
                feature_stored = true;

                break;
            end
        end
        % If feature_stored is false, it means after 't' attempts, nothing was stored
        % In this case, episodic_image(k) will remain 0 (its initial value)
    end
end

% Function to calculate the likelihood ratio for a single image given a probe
% This is Equation 4A
function lambda_j = calculate_lambda_single_image(probe_vector, image_vector, c_copy, g_param)
    % The overall likelihood ratio for an episodic image is is formed by multiplying 
    % the individual likelihood ratios of each feature
    % It is set to 1 by default so that it has no effect if not changed
    lambda_j = 1.0;
    
    % Iterate through all possible feature positions
    % Thought: We compare feature by feature
    for k = 1:length(probe_vector)
        probe_val = probe_vector(k);
        image_val = image_vector(k);
        
        % Case 1: If both probe and image have non-zero features at this position
        if probe_val ~= 0 && image_val ~= 0
            
            % If the probe and image match, the feature is in M_j
            if probe_val == image_val
                % The term: (c + (1-c)*g*(1-g)^(V-1)) / (g*(1 - g)^(V - 1))
                % This simplifies to: (1 - c) + c / (g*(1 - g)^(V - 1))
                
                % Denominator of the fractional part: g*(1 - g)^(V - 1)
                denominator = g_param * ((1 - g_param)^(probe_val - 1));
                
                % Ensure denominator is not zero to avoid Inf/NaN
                % This shouldn't happen with g > 0 and probe_val >= 1, but good practice
                if denominator > eps % eps is a small positive number
                     lambda_j = lambda_j * ((1 - c_copy) + c_copy / denominator);
                else
                    % If denominator is effectively zero (extremely rare feature), lambda goes to infinity
                    % Cap it at 1e10 for practical simulation though conceptually it's very large
                    lambda_j = lambda_j * 1e10;
                end
               
            % If the probe and image mismatch, the feature is in Q_j
            else
                % The term: (1-c)
                lambda_j = lambda_j * (1 - c_copy);
            end
        end
        % Case 2: One or both are zero (probe has non-zero, image has zero; or vice versa; or both zero)
        % Thought: These features do not contribute to the products in Equation 4A
        % because they are not considered "matching" or "mismatching"
        % Their contribution to the product is implicitly 1. So, nothing to do here
    end
end

% Function to calculate the overall odds (phi) for a probe
% Thought: Phi is the average of all individual likelihood ratios
function phi = calculate_overall_odds(probe_vector, memory_images, c_copy, g_param)
    
    % Initialize at 0
    sum_lambda = 0;

    % Get the number of images
    n_images = length(memory_images);
    
    % Sum up lambda_j for each image in memory
    for i = 1:n_images
        current_image = memory_images{i};
        sum_lambda = sum_lambda + calculate_lambda_single_image(probe_vector, current_image, c_copy, g_param);
    end
    
    % Calculate the average
    if n_images > 0
        phi = sum_lambda / n_images;
    else
        phi = 0; % Should not happen if list_length > 0
    end
end