# Stage 4: Jason AgentSpeak Validation

This stage validates AgentSpeak (.asl) code generated from Stage 3 using the Jason BDI framework.

## Directory Structure

```
stage4_jason_validation/
├── jason_src/              # Jason framework source code (built from GitHub)
│   ├── jason-cli/          # Jason command-line interface
│   └── jason-interpreter/  # Jason interpreter core
└── jason_project/          # Test project for validating AgentSpeak code
    ├── src/asl/            # AgentSpeak source files
    │   ├── test_agent.asl      # Comprehensive test agent
    │   └── simple_test.asl     # Simple validation agent
    ├── test_agent.mas2j    # Jason MAS configuration (comprehensive)
    ├── simple_test.mas2j   # Jason MAS configuration (simple)
    ├── run_jason.sh        # Wrapper script to run Jason with correct Java
    └── test_jason.sh       # Test script for validation
```

## Prerequisites

### 1. Java Installation

Jason requires Java 17-23. This project uses **Amazon Corretto 23**:
- **Java Home**: `/Users/lyw/Library/Java/JavaVirtualMachines/corretto-23.0.2/Contents/Home`
- Java 24 is **NOT** supported by the current Jason Gradle build system

### 2. Gradle Installation

Gradle is used to build Jason from source:
```bash
brew install gradle
```

## Setup

### Building Jason

Jason has been built from source and is located in `jason_src/`:

```bash
cd jason_src
export JAVA_HOME=/Users/lyw/Library/Java/JavaVirtualMachines/corretto-23.0.2/Contents/Home
./gradlew config  # Builds Jason CLI
```

The built Jason JAR is located at:
```
jason_src/jason-cli/build/bin/jason-cli-all-3.3.1.jar
```

## Running AgentSpeak Code

### Method 1: Using the Test Script (Recommended)

```bash
cd jason_project
./test_jason.sh
```

This script:
1. Sets the correct Java version (Java 23)
2. Runs the simple_test agent
3. Validates that Jason BDI framework is working

### Method 2: Manual Execution

```bash
cd jason_project
export JAVA_HOME=/Users/lyw/Library/Java/JavaVirtualMachines/corretto-23.0.2/Contents/Home
$JAVA_HOME/bin/java -jar ../jason_src/jason-cli/build/bin/jason-cli-all-3.3.1.jar <your_file>.mas2j --console
```

### Method 3: Using run_jason.sh Wrapper

```bash
cd jason_project
./run_jason.sh test_agent.mas2j
```

## Test Agents

### simple_test.asl
A minimal agent that:
- Starts with an initial goal
- Prints success messages
- Validates basic BDI operations
- Stops the MAS

### test_agent.asl
A more comprehensive agent that:
- Tests goal achievement
- Tests belief operations (addition and querying)
- Validates all basic BDI operations
- Includes failure handling

## Usage for Stage 3 Output

To validate AgentSpeak code generated from Stage 3:

1. Place the generated `.asl` file in `jason_project/src/asl/`
2. Create or modify the `.mas2j` configuration file:
```
MAS your_system_name {
    infrastructure: Centralised
    agents: your_agent_name;
    aslSourcePath: "src/asl";
}
```
3. Run the agent:
```bash
./test_jason.sh  # after modifying to use your .mas2j file
# or
$JAVA_HOME/bin/java -jar ../jason_src/jason-cli/build/bin/jason-cli-all-3.3.1.jar your_file.mas2j --console
```

## Troubleshooting

### Java Version Issues

If you encounter "Unsupported class file major version" errors:
- Ensure you're using Java 23 (Corretto)
- Check `JAVA_HOME` is set correctly:
```bash
export JAVA_HOME=/Users/lyw/Library/Java/JavaVirtualMachines/corretto-23.0.2/Contents/Home
```

### Jason Build Issues

If Jason fails to build:
```bash
cd jason_src
export JAVA_HOME=/Users/lyw/Library/Java/JavaVirtualMachines/corretto-23.0.2/Contents/Home
./gradlew clean
./gradlew config
```

### No Output from Agent

Jason runs in console mode with `--console` flag. If no agent output appears:
- Check that `.stopMAS` is called in your agent
- Verify the .mas2j file points to the correct .asl file
- Check for syntax errors in the AgentSpeak code

## Notes

- **Maven was initially attempted** but Jason artifacts are not available in Maven Central or JaCaMo repositories
- **Gradle is the official build tool** for Jason 3.x
- **Jason CLI** is used instead of the deprecated Jason IDE
- The test agents demonstrate that the Jason BDI framework environment is properly configured and functional
