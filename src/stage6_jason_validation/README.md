# Stage 6: Jason Runtime Validation

Stage 6 is part of the default pipeline mainline.
After Stage 5 renders `agentspeak_generated.asl`, Stage 6 validates runtime execution in Jason.

## Current Backend

- **Backend**: `RunLocalMAS` (`jason.infra.local.RunLocalMAS`)
- **Entry implementation**: `src/stage6_jason_validation/jason_runner.py`
- **Result policy**: hard-fail (Stage 6 failure makes the full pipeline fail)

## Runtime Flow

For each pipeline run, Stage 6 writes and executes:

1. `jason_runner_agent.asl`
   - Copy of Stage 5 output
   - Appended wrapper goal `!stage6_exec`
   - Seeds positive target beliefs from Stage 3 literals
   - Executes `!run_dfa`
   - Emits markers:
     - `stage6 exec success`
     - `stage6 exec failed`

2. `jason_runner.mas2j`
   - Single-agent MAS file for `jason_runner_agent`
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
   - `jason_validation.json`

## Success Criteria

Stage 6 succeeds only if all checks pass:

1. Process finished (not timeout)
2. Exit code is `0`
3. `stdout` contains `stage6 exec success`
4. `stdout` does **not** contain `stage6 exec failed`

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

Current Stage 6 follows Stage 5 semantics:

- `not p(...)` uses Jason NAF behavior (fact absence)
- No `~p(...)` strong-negation migration in this version

Strong-negation support is planned for a future iteration.
