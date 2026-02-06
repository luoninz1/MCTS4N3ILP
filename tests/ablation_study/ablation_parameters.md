# Ablation Study Configuration

This document lists the parameter configurations for each algorithm variant used in the incremental ablation study.

| Variant | Description | Environment | Symmetric Action (`--symmetric_action`) | Max Symmetry Level (`--max_symmetry_level`) | Save Optimal State (`--save_optimal_terminal_state`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M0** | Baseline/Vanilla (FVAS) | `N3il_with_FVAS` | `None` | `None` (Equivalent to -1) | `False` |
| **M1** | M0 + SVAS | `N3il_with_SVAS_wo_inc` | `None` | `None` (Equivalent to -1) | `False` |
| **M2** | M1 + Incremental AS Update | `N3il` | `None` | `None` (Equivalent to -1) | `False` |
| **M3** | M2 + Dihedral Pruning | `N3il_with_symmetry` | `None` | `1` | `False` |
| **M4** | M3 + Batch Action ($C_4$) | `N3il_with_symmetry_and_symmetric_actions` | `rotation_90_then_rotation_180` | `None` (Defaults to $2n$) | `False` |
| **Ours**| M4 + Optimal Record | `N3il_with_symmetry_and_symmetric_actions` | `rotation_90_then_rotation_180` | `None` (Defaults to $2n$) | `True` |

## Experiment Parameters
- **Grid Sizes ($n$):** 30, 40, 50
- **Repeats:** 10 trials per grid size per variant (Seeds 0-9).
- **Number of Searches:** $10n^2$
- **Algorithm:** `MCTS_Tree_Reuse`

### Implementation Details
- **Max Symmetry Level Logic:** In `ablation.py`, if `max_symmetry_level` is `None`:
  - It defaults to `2*n` if environment is `N3il_with_symmetry_and_symmetric_actions`.
  - It defaults to `-1` otherwise.
