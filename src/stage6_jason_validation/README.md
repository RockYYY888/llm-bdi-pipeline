# Stage 6: Jason Runtime Validation

Stage 6 is part of the default pipeline mainline.
After Stage 5 renders `agentspeak_generated.asl`, Stage 6 validates runtime execution in Jason.

## Current Backend

- **Backend**: `RunLocalMAS` (`jason.infra.local.RunLocalMAS`)
- **Entry implementation**: `src/stage6_jason_validation/jason_runner.py`
- **Result policy**: hard-fail (Stage 6 failure makes the full pipeline fail)

## Runtime Flow

For each pipeline run, Stage 6 writes and executes:

1. `agentspeak_generated.asl`
   - Stage 5 output rewritten into the Jason runtime form
   - Appended entry goal `!execute`
   - Seeds runtime facts from `problem.hddl :init` when available, otherwise from Stage 4 witnesses
   - Executes `!run_dfa`
   - Validates accepting DFA state and target-literal context before success marker
   - Emits markers:
     - `execute success`
     - `execute failed`

2. `jason_runner.mas2j`
   - Single-agent MAS file for `agentspeak_generated`
   - `aslSourcePath: "."`

3. Java command:

```bash
java -cp <jason-cli-all-*.jar> \
  jason.infra.local.RunLocalMAS \
  jason_runner.mas2j \
  --log-conf <console-info-logging.properties>
```

4. Artifacts written in pipeline log dir:
   - `jason_stdout.txt`
   - `jason_stderr.txt`
   - `action_path.txt`
   - `jason_validation.json`

## Success Criteria

Stage 6 succeeds only if all checks pass:

1. Process finished (not timeout)
2. Exit code is `0`
3. `stdout` contains `execute success`
4. `stdout` does **not** contain `execute failed`
5. Runtime checks inside ASL pass:
   - `dfa_state(FINAL_STATE)` and `accepting_state(FINAL_STATE)` both hold
   - `!verify_targets` context matches all Stage 3 target literals

## Java Discovery

Stage 6 selects Java in this order:

1. `STAGE6_JAVA_BIN`
2. `STAGE6_JAVA_HOME/bin/java`
3. `JAVA_HOME/bin/java`
4. `PATH` (`java`)
5. macOS JVM directories

Supported versions: **17-23**.
If no supported Java is found, Stage 6 fails.

## Jason Build Discovery

Stage 6 checks:

- `src/stage6_jason_validation/jason_src/jason-cli/build/bin/jason-cli-all-*.jar`

If missing, Stage 6 runs:

```bash
cd src/stage6_jason_validation/jason_src
./gradlew config
```

If build still does not produce the jar, Stage 6 fails.

## Notes on Negation (Current)

Stage 6 uses a single negation policy end-to-end:

- all negative literals are rendered as `not p(...)`
- no `~p(...)` strong-negation track is maintained

In the generated Java symbolic environment:

- facts are stored in one set: `world`
- negative preconditions are checked as absence from `world`
- negative effects only remove positive facts from `world`
