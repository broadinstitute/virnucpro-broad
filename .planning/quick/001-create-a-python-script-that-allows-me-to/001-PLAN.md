---
phase: quick-001
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: [compare_virnucpro_outputs.py]
autonomous: true
must_haves:
  truths:
    - "Script can load and parse prediction files from both implementations"
    - "Script compares predictions and reports differences"
    - "Script provides useful metrics and statistics"
  artifacts:
    - path: "compare_virnucpro_outputs.py"
      provides: "Comparison script for VirNucPro outputs"
      min_lines: 150
  key_links:
    - from: "compare_virnucpro_outputs.py"
      to: "CSV/TSV parsing"
      via: "pandas.read_csv"
      pattern: "pd\.read_csv"
---

<objective>
Create a Python script to compare VirNucPro prediction outputs between the refactored (optimized) version and the original vanilla implementation.

Purpose: Enable validation that the optimized implementation produces equivalent results to the original
Output: Standalone Python script that compares prediction files and reports differences
</objective>

<execution_context>
@~/.claude/get-shit-done/workflows/execute-plan.md
@~/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
The VirNucPro pipeline produces predictions in two key formats:
1. prediction_results.txt - Per-frame predictions (seq_id_frame, prediction, score_0, score_1)
2. prediction_results_highestscore.csv - Consensus predictions (Modified_ID, Is_Virus, max_score_0, max_score_1)

The script needs to compare these outputs between implementations and identify:
- Prediction mismatches (virus vs others)
- Score differences (with tolerance for floating point)
- Missing/extra sequences
- Overall accuracy metrics
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create comparison script with comprehensive validation</name>
  <files>compare_virnucpro_outputs.py</files>
  <action>
    Create a Python script that:
    1. Loads prediction files from both implementations (vanilla and refactored)
    2. Supports both consensus CSV and per-frame TXT formats
    3. Compares predictions with configurable tolerances:
       - Exact match for binary predictions (virus/others)
       - Float tolerance for probability scores (default 1e-5)
    4. Reports detailed comparison metrics:
       - Total sequences compared
       - Matching predictions count and percentage
       - Score differences (mean, max, distribution)
       - Lists of mismatched sequences with details
    5. Handles edge cases:
       - Different sequence ordering
       - Missing sequences in either output
       - Different frame suffixes (F1-F3, R1-R3)
    6. Provides summary statistics and optional detailed diff output

    Use pandas for efficient data loading and comparison, with proper error handling.
    Include argparse CLI interface with options for:
    - Input file paths (vanilla and refactored outputs)
    - Output format (summary, detailed, csv)
    - Score tolerance threshold
    - Whether to compare consensus or per-frame predictions
  </action>
  <verify>python compare_virnucpro_outputs.py --help shows usage information</verify>
  <done>Script exists with comprehensive comparison capabilities and clear output</done>
</task>

</tasks>

<verification>
1. Script can load both CSV and TXT prediction formats
2. Comparison identifies matching and differing predictions
3. Output provides clear summary of differences
4. Edge cases handled gracefully
</verification>

<success_criteria>
- Python script `compare_virnucpro_outputs.py` exists and runs
- Script can compare prediction outputs from both implementations
- Reports useful metrics for validation
- Handles different file formats and edge cases
</success_criteria>

<output>
After completion, script will be at project root: `compare_virnucpro_outputs.py`
</output>