# from .base_env import Point
# from .base_env_3d import Point3D
# from .isosceles_triangle import NoIsoscelesEnv, NoStrictIsoscelesEnv
# from .colinear import (NoThreeCollinearEnv, NoThreeCollinearEnvWithPriority,
#                        FastNoThreeCollinearEnv, NoThreeInLineRemovalEnv, NoThreeInLineDominatingEnv)
# from .colinear_3d import NoThreeCollinear3DEnv, NoThreeCollinear3DEnvWithPriority
from .collinear_for_mcts import N3il, N3il_with_symmetry, supnorm_priority, supnorm_priority_array, get_value_nb