# Stage 0 Completion Summary

**Date**: 2025-10-25
**Status**: ✅ COMPLETE
**Git Tag**: `pre-refactor-baseline`

## Overview

Stage 0 establishes a comprehensive safety net before beginning refactoring work. All deliverables have been completed successfully.

## Deliverables

### 1. Unit Test Suite ✅

Created `tests/unit/` directory with comprehensive test coverage:

- **test_kernels.py** (16 tests)
  - Tests for numba kernels: `_are_collinear`, `get_valid_moves_nb`, `get_valid_moves_subset_nb`, `simulate_nb`, `check_collinear_nb`, `filter_top_priority_moves`
  - Validates kernel correctness, parity, and determinism

- **test_env.py** (14 tests)
  - Tests for N3il and N3il_with_symmetry environment API
  - Validates state management, valid moves computation, subset parity
  - Documents existing bugs (duplicate initialization) for Stage 1 fixes

- **test_symmetry.py** (8 tests)
  - Tests for D4 symmetry transformations and filtering
  - Validates `apply_element_to_action`, `filter_actions_by_stabilizer_nb`
  - Tests symmetry-aware valid moves

- **test_rewards.py** (9 tests)
  - Tests for reward functions: `get_value_nb`, `get_value_exp_norm_nb`
  - Validates determinism, value ranges, and consistency

- **test_integration_smoke.py** (6 tests)
  - Integration tests for MCTS `evaluate()` and `search()`
  - Tests determinism with fixed seeds
  - Tests both N3il and N3il_with_symmetry environments

**Test Results**: All 53 tests passing in ~3 seconds

### 2. Baseline Data Verification ✅

Verified baseline data in `results/baseline/pre_refactor/2025-10-25_17-*`:

- **n=6**: 20 runs (4.6 KB)
- **n=8**: 20 runs (4.6 KB)
- **n=10**: 20 runs (4.7 KB)
- **n=20**: 20 runs (4.7 KB)

Total: 80 runs across 4 grid sizes, with complete metadata (git commit, environment, timing, results).

### 3. Git Setup ✅

- **Tag created**: `pre-refactor-baseline` - marks current state before refactoring
- **Commits**:
  - `782209d`: "test: Add Stage 0 unit test suite for pre-refactor baseline"
  - `415326d`: "chore: Add numba cache files to .gitignore"
- **.gitignore updated**: Added `*.nbc` and `*.nbi` for numba caches

### 4. Pre-Refactor Checklist ✅

- [x] Unit tests created and passing (53 tests, 100% pass rate)
- [x] Baseline data verified (80 runs for n ∈ {6, 8, 10, 20})
- [x] Git tagged `pre-refactor-baseline`
- [x] .gitignore updated for `__pycache__`, `*.nbc`, `*.nbi`
- [x] Code review: identified duplicate init bugs in N3il classes
- [x] Determinism verified: Tests with fixed seeds pass consistently

## Known Issues Documented

These issues will be fixed in Stage 1:

1. **Bug**: Duplicate initialization in `N3il.__init__` (src/envs/collinear_for_mcts.py:857-862)
2. **Bug**: Double `super().__init__()` call in `N3il_with_symmetry.__init__` (line 1438)
3. **Design**: `get_next_state()` mutates input array (documented behavior, may refactor later)
4. **Design**: Required args `TopN` and `simulate_with_priority` not well-documented (tests now enforce)

## Test Coverage

### Critical Paths Covered:
- ✅ Kernel correctness (collinearity detection, valid moves computation)
- ✅ Environment API (state management, transitions, termination)
- ✅ Symmetry filtering (D4 transformations, stabilizers)
- ✅ Reward functions (value computation, normalization)
- ✅ MCTS search (end-to-end integration with both environments)
- ✅ Determinism (seeded randomness reproducibility)

### Not Covered (intentional):
- Parallel MCTS variants (legacy, not primary focus)
- Tree visualization (optional feature)
- Experiment orchestration details (tested via integration tests)

## Next Steps: Stage 1

With Stage 0 complete, we can now safely proceed to Stage 1 refactoring:

1. Extract seeding utilities to `src/utils/seed.py`
2. Fix duplicate initialization bugs
3. Add `simulate()` method to environments
4. Remove unused code (`check_collinear_nb` from mcts.py)
5. Extract visualization to `src/viz/tree.py`
6. Extract experiment runner to `src/experiments/runner.py`

All changes will be validated against:
- Unit tests (must all pass)
- Baseline comparison (results within ±2%)
- Determinism (fixed seed → identical outputs)

## Performance Baseline

From existing baseline data (n=20, 20 runs):
- Mean terminal points: ~27-30 (varies by run)
- Mean time per run: ~5-10 seconds
- All runs complete successfully with no crashes

Stage 1 refactoring must preserve these characteristics within ±2% tolerance.

---

**Ready to proceed to Stage 1** ✅
