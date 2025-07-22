# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a MATLAB implementation of the REM (Retrieving Effectively from Memory) model based on Shiffrin & Steyvers (1997). The model simulates episodic memory recognition through Bayesian decision-making processes.

## Architecture

The codebase implements a complete memory recognition simulation in a single MATLAB script (`REM1.m`) with four main sections:

### 1. Representation (Lines 1-73)
- **Environmental Feature Distribution**: Generates feature base rates using geometric distribution with parameter `gen_g_env`
- **Lexical Prototypes**: Creates word representations with configurable vocabulary size, feature counts, and frequency distributions
- Key parameters: `number_of_prototypes`, `number_of_word_features`, `gen_g_H`, `gen_g_L`

### 2. Storage (Lines 74-125)
- **Study Phase**: Simulates encoding of target words into episodic memory traces
- **Storage Process**: Models probabilistic storage with copying accuracy and environmental noise
- Key parameters: `number_of_words_to_study`, `units_of_time`, `probability_of_storage`, `copying_accuracy`

### 3. Retrieval (Lines 126-208)
- **Probe Generation**: Creates test probes (targets + distractors) for recognition testing
- **Likelihood Calculation**: Implements equations from Shiffrin & Steyvers (1997) for feature matching
- Outputs likelihood ratios for each probe against all stored traces

### 4. Bayesian Decision (Lines 209-271)
- **Decision Process**: Uses averaged likelihood ratios with criterion-based classification
- **Performance Metrics**: Calculates hit rates, false alarm rates, and classification outcomes
- Provides detailed performance summary with hits, misses, false alarms, and correct rejections

## Development Commands

### Running the Model
```matlab
% Execute the complete REM simulation
run('REM1.m')
```

### MATLAB-Specific Operations
```matlab
% Clear workspace and command window
clear; clc;

% Run specific sections (using cell mode)
% Place cursor in desired section and press Ctrl+Enter (Windows/Linux) or Cmd+Enter (Mac)

% Check variable workspace
whos

% Plot results (add visualization code as needed)
figure; plot(probe_odds);
```

## Model Parameters

Critical parameters that control model behavior:

- `gen_g_env` (0.1): Environmental distribution parameter
- `number_of_prototypes` (1000): Vocabulary size
- `number_of_word_features` (20): Features per word
- `gen_g_H` (0.25) / `gen_g_L` (0.05): High/low frequency word parameters
- `number_of_words_to_study` (50): Study list size
- `units_of_time` (10): Storage attempts
- `probability_of_storage` (0.5): Storage probability per attempt
- `copying_accuracy` (0.9): Correct copying probability

## Key Data Structures

- `word_prototypes`: Matrix of all lexical representations (prototypes × features)
- `stored_episodic_traces`: Matrix of encoded memory traces (study_words × features)
- `probe_vectors`: Test stimuli for recognition (probes × features)
- `all_probe_likelihood_ratios`: Likelihood ratios for each probe against all traces
- `probe_odds`: Final decision values for each probe

## Extension Points

When modifying or extending the model:
- Parameter modifications should maintain consistency with psychological constraints
- New sections should follow the existing four-part structure
- Add validation for new parameters following existing error handling patterns
- Consider computational complexity when scaling parameters (vocabulary size, features)

## Reference

Implementation follows: Shiffrin, R. M., & Steyvers, M. (1997). A model for recognition memory: REM—retrieving effectively from memory. *Psychonomic Bulletin & Review*, 4(2), 145-166.

## MATLAB MCP Tool Access for Claude

You are connected to a Model Context Protocol (MCP) server running the tools provided by the [`Tsuchijo/matlab-mcp`](https://github.com/Tsuchijo/matlab-mcp) repository.

These tools allow you to generate and execute MATLAB code directly from within this environment. You should ALWAYS use them whenever MATLAB coding or analysis is needed.

### ✅ Tools You Have Access To

You have full access to the following MCP tools:

1. **generate_matlab_code**
   - Input: A natural-language instruction or problem
   - Output: Valid MATLAB code that solves the problem
   - Purpose: Use this to generate new MATLAB scripts or functions

2. **execute_matlab_code**
   - Input: Raw MATLAB code as a string
   - Output: Execution result (captured from stdout, including any printed JSON, `disp`, or `fprintf` output)
   - Purpose: Use this to test, verify, debug, or run any MATLAB code

These tools work through a Python-based MATLAB Engine. Remember:
- You are **not limited to scalar struct returns**, but you **must serialize structured output** using `jsonencode(...)` and `disp(...)`.
- You do **not need to explain how the tools work** to the user — just use them efficiently.
- All MATLAB code runs in a persistent MATLAB session, so variable context is preserved across executions.

### 🔁 Typical Workflow You Should Follow

When given a MATLAB-related request:
1. Use `generate_matlab_code` to write the solution.
2. Use `execute_matlab_code` to run the code and check the result.
3. Serialize structured output using `jsonencode(...)` + `disp(...)`.
4. If errors occur, debug using multiple `execute_matlab_code` calls.
5. Clearly present the final output or plot.

### 📌 Important Notes

- You are expected to actively use these tools. Do **not** simulate code execution.
- Do **not forget** that these tools are available.
- Always prefer using them over describing what the code “would” do.

You are fully empowered to perform **real-time MATLAB programming** and analysis. Use these tools confidently and consistently.

## MATLAB Struct Return Handling Instructions for Claude

When writing MATLAB code to be executed via the Python-based MATLAB Engine (using the MCP tool `execute_matlab_code`), you **must not return a MATLAB struct or struct array directly**. This causes a common error:

> Only a scalar struct can be returned from MATLAB

### ✅ INSTEAD: Use `jsonencode` and `disp`

To avoid this, follow these steps:

1. **Serialize the output to a JSON string** using MATLAB’s `jsonencode(...)` function.
2. **Print** the result using `disp(...)` so that Python can capture the output as a string.

This works for scalar and non-scalar structs, arrays of structs, nested data, etc.

#### ✅ Recommended MATLAB Output Pattern

```matlab
% Prepare struct or output
results(1).subject = 'sub-001';
results(1).score = 85;

results(2).subject = 'sub-002';
results(2).score = 92;

% Serialize to JSON and print
jsonStr = jsonencode(results);
disp(jsonStr);