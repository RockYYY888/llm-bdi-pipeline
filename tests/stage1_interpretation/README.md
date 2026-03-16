# Stage 1 Test Harness

This directory contains the diagnostic harness for Stage 1 natural-language to LTLf generation.
It is useful for prompt evaluation and regression analysis, but it is not the canonical
benchmark-acceptance harness. That role belongs to `tests/test_pipeline.py`.

## Files

- `nl_to_ltlf_test_cases.csv`
  - hand-authored Stage 1 query cases with expected outputs
- `test_nl_to_ltlf_generation.py`
  - live LLM evaluation script for the CSV cases
- `test_results_nl_to_ltlf_*.csv`
  - generated local run artifacts; these are outputs, not normative fixtures

## What This Harness Is For

Use this directory when you want to inspect Stage 1 behaviour in isolation:

1. prompt changes in `src/stage1_interpretation/prompts.py`
2. object extraction quality
3. operator interpretation quality
4. structured JSON output stability

## Running It

```bash
cd tests/stage1_interpretation
python test_nl_to_ltlf_generation.py
```

The script writes a timestamped CSV report to the same directory.

## Output Columns

The generated CSV records:

- the input query and expected target
- the actual LTLf and extracted objects
- a success flag and match category
- a short reflection field for debugging failures

## Notes

- This harness uses live LLM calls and therefore has cost and latency.
- Results are useful for diagnosis, not as a substitute for end-to-end benchmark acceptance.
- Timestamped result CSVs should be treated as disposable run artifacts rather than repository
  fixtures.
