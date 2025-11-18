# Legacy Code

This directory contains previous implementations that have been superseded or are no longer actively used.

## Files

### forward_planner.py
**Moved Date**: 2025-01-18

**Reason**: Superseded by the refactored `lifted_planner.py` which implements true lifted planning with unification.

**Original Purpose**:
- Forward state-space planner with optional variable support
- Implements "forward destruction" planning from goal states
- Uses pre-computed grounded actions for fixed object sets

**Why Deprecated**:
- User explicitly requested to use only lifted planning approach
- Grounded/object-based planning doesn't scale well with large object sets
- Lifted planning (when correctly implemented) has better theoretical scalability

**Recovery**:
If needed, this file can be restored from git history or this legacy directory.

---

**Note**: Files in this directory are kept for reference and recovery purposes. They are not used in the current codebase.
