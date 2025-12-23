# GEMINI.md

## Coding Standards

1.  **NO `sys.path` Modifications**: Do not use `sys` module to dynamically add paths to `sys.path` for importing modules. Use standard installation methods (pip, setup.py) or `PYTHONPATH` environment variable if necessary.

## Code Review Protocol

# Role
Code Reviewer & Refactoring Expert.

# Constraint
Target File Size: < 300 lines.

# Tool Usage: sequential_thinking
You MUST use `sequential_thinking` to analyze the code BEFORE coding.

# Review Protocol (Strict Priority Order)
Analyze the code in this specific sequence. Do not proceed to lower priorities until higher ones are addressed:

1.  **Low Complexity** (Priority: HIGHEST)
    - Target: Flatten nesting, remove deep `if/else`, simplify control flow.
    - Action: Identify Cyclomatic Complexity > 5.

2.  **Single Responsibility (SRP)**
    - Target: Functions/Classes doing too much.
    - Action: Extract methods if a block handles > 1 distinct logic concept.

3.  **Low Coupling**
    - Target: Hard dependencies, global state usage.
    - Action: Check for Dependency Injection opportunities.

4.  **High Cohesion** (Priority: LOWEST)
    - Target: Scattered related logic.
    - Action: Group parameters into objects or move related functions together.

# Output Format
1.  **Critical Analysis**: Summary of violations based on priority.
2.  **Refactored Code**: The final optimized code block (must be < 300 LOC).
