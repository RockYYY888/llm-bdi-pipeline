# Quick Start

Get the LTL pipeline running in 2 minutes.

---

## Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd /path/to/llm-bdi-pipeline-dev
uv sync
```

---

## Run Your First Test

```bash
uv run python src/main.py "Put block A on block B"
```

**Expected Output**:
```
Stage 1: Natural Language → LTL
  Objects: ['a', 'b']
  LTL Formulas:
    1. F(on(a, b))      # Eventually a is on b
    2. F(clear(a))      # Eventually a is clear

Stage 2: LTL → PDDL Problem
  Generated: output/20251022_170921/problem.pddl

Stage 3: PDDL → Action Plan
  Plan:
    1. pickup(a)
    2. stack(a, b)

✓ Pipeline Complete
```

**What Happened**:
1. Natural Language → LTL formulas (`F(on(a,b))`)
2. LTL → PDDL problem file
3. PDDL → Action plan (`pickup(a)`, `stack(a,b)`)

Check `output/YYYYMMDD_HHMMSS/` for generated files.

---

## Use with LLM (Optional)

```bash
# 1. Configure API key
cp .env.example .env
# Edit .env: OPENAI_API_KEY=your-key-here

# 2. Run with real LLM
uv run python src/main.py "Build a tower with C on B on A"
```

---

## Next Steps

- **Full Guide**: See [README.md](README.md)
- **Learn LTL**: See [LTL_TUTORIAL.md](LTL_TUTORIAL.md)
- **Detailed Usage**: See [USAGE_GUIDE.md](USAGE_GUIDE.md)

---

**That's it!** For more details, see [README.md](README.md).
