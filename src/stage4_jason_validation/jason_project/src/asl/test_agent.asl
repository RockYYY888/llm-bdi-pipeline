/*
 * Stage 4 - Test AgentSpeak Agent
 *
 * A minimal BDI agent to verify Jason environment is working correctly.
 * This agent simply achieves a test goal and prints confirmation.
 */

// Initial goal: test the system
!test_system.

// Plan to achieve test_system goal
+!test_system
    : true
    <- .print("Jason BDI Framework is working!");
       .print("Stage 4 environment validation: SUCCESS");
       !verify_beliefs.

// Plan to verify belief operations
+!verify_beliefs
    : true
    <- +test_belief(success);
       .print("Belief added: test_belief(success)");
       !verify_queries.

// Plan to verify query operations
+!verify_queries
    : test_belief(success)
    <- .print("Query successful: test_belief(success) exists");
       .print("All basic BDI operations verified!");
       .stopMAS.  // Stop the multi-agent system

// Fallback plan if query fails
+!verify_queries
    : true
    <- .print("ERROR: Query failed - belief not found");
       .stopMAS.

// Handle failure of main test
-!test_system
    : true
    <- .print("ERROR: test_system goal failed");
       .stopMAS.
