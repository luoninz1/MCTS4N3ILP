import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.algos import evaluate, MCTS
from src.rewards.n3il_rewards import set_reward_strategy

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MCTS tests for a range of n values.")
    parser.add_argument("--start", type=int, default=20, help="Starting value of n (inclusive)")
    parser.add_argument("--end", type=int, default=51, help="Ending value of n (exclusive)")
    parser.add_argument("--step", type=int, default=5, help="Step size for n values")
    parser.add_argument("--repeat", type=int, default=1, help="Number of runs for each n value")
    parser.add_argument("--symmetric_action", type=str, default="rotation_90_then_rotation_180", help="Symmetric action mode (e.g., 'vertical_flip_then_horizontal_flip_then_diagonal_flip')")
    parser.add_argument("--environment", type=str, default="N4il", help="Environment name")
    parser.add_argument("--algorithm", type=str, default="MCTS_Tree_Reuse", help="Algorithm name")
    parser.add_argument("--reward", type=str, default="exp_growth", 
                        choices=["default", "exp_reverse", "exp_growth", "linear", "gaussian", "optimal_3x3"],
                        help="Reward function strategy to use")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results (tables and figures). If None, uses current directory.")
    
    args_cli = parser.parse_args()
    print(f"Configuration: Setting Reward Strategy to '{args_cli.reward}'")
    set_reward_strategy(args_cli.reward)

    # Determine output directories
    if args_cli.output_dir:
        table_dir = args_cli.output_dir
        figure_dir = os.path.join(args_cli.output_dir, 'figure')
    else:
        table_dir = os.path.dirname(__file__)
        figure_dir = os.path.join(os.path.dirname(__file__), 'figure')
    
    # Ensure directories exist
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    if args_cli.symmetric_action == "None":
        args_cli.symmetric_action = None # Convert string "None" to actual None for easier handling in code

    # Generate list of n values
    n_list = range(args_cli.start, args_cli.end, args_cli.step)

    # Clear any previous global data before starting new experiments
    MCTS.clear_global_data()

    # Store results for all trials
    all_results = {n: [] for n in n_list}
    
    for i in range(args_cli.repeat):
        print(f"\n=== Starting Trial {i+1}/{args_cli.repeat} ===")
        
        for n in n_list:
            print(f"Trial {i+1}/{args_cli.repeat} for n={n}...", end=" ")

            """
                Available subgroups of D4 for 'symmetric_action':
                - Order 1: None (Identity only)
                - Order 2: 
                    'rotation_180', 'horizontal_flip', 'vertical_flip', 
                    'diagonal_flip', 'anti_diagonal_flip'
                - Order 4:
                    'rotation_90_then_rotation_180' (Cyclic C4: 90, 180, 270 rotations)
                    'vertical_flip_then_horizontal_flip' (V4-A: Rectangular symmetry)
                    'diagonal_flip_then_anti_diagonal_flip' (V4-B: Rhombic symmetry)
                - Order 8:
                    'vertical_flip_then_horizontal_flip_then_diagonal_flip' (Full D4 symmetry)
            """

            args = {
                'environment': args_cli.environment,  # Specify the environment
                'algorithm': args_cli.algorithm,
                'save_optimal_terminal_state': True,  # Save optimal terminal states found
                'save_all_optimal_terminal_states': True if n < 11 else False,  # Save all optimal terminal states found for n < 31 (to avoid memory issues for larger n)
                'symmetric_action': args_cli.symmetric_action,  # Specify symmetric action mode
                # horizontal_flip, vertical_flip, diagonal_flip, anti_diagonal_flip, rotation_90/180/270
                'node_compression': True,  # Enable node compression
                'max_level_to_use_symmetry': 2*n if args_cli.symmetric_action else 1,  # Use symmetry for first 2 levels (helps find compact solutions)
                'n': n,
                'C': 1.41,  # sqrt(2)
                'num_searches': 100*(n**2),  # Reduced for testing tree visualization
                'num_workers': 1,      # >1 ⇒ parallel
                'virtual_loss': 1.0,     # magnitude to subtract at reservation
                'process_bar': False,
                'display_state': True,
                'logging_mode': True,  # Enable logging mode to get return value
                'TopN': n,  # Without Priority
                "simulate_with_priority": False,
                'table_dir': table_dir,  # Directory to save tables
                'figure_dir': figure_dir,  # Directory to save figures
                'random_seed': i,  # Use the loop index as a seed for reproducibility
                'tree_visualization': False,  # Enable tree visualization
                'pause_at_each_step': False,  # Disable interactive prompts for automation
                'continue_from_existing_state': None, # None: continuation; str: path to load
            }
            
            # Get the result from evaluate function
            num_points = evaluate(args)
            all_results[n].append(num_points)
            print(f"Final points: {num_points}")
    
    # Process summaries after all trials
    for n in n_list:
        results_for_n = all_results[n]
        # Print summary for this grid size
        print(f"\n--- Summary for n={n} ---")
        print(f"Results: {results_for_n}")
        if results_for_n:
            print(f"Min points: {min(results_for_n)}")
            print(f"Max points: {max(results_for_n)}")
            print(f"Average points: {sum(results_for_n)/len(results_for_n):.2f}")
            print(f"Median points: {sorted(results_for_n)[len(results_for_n)//2]}")
        else:
            print("No results collected.")
    
    # Print overall summary
    print("\n" + "="*60)
    print("FINAL SUMMARY OF ALL EXPERIMENTS")
    print("="*60)
    for n, results in all_results.items():
        print(f"Grid size n={n}: {results}")
        print(f"  → Min: {min(results)}, Max: {max(results)}, Avg: {sum(results)/len(results):.4f}")
        
        # Count frequency of each result
        from collections import Counter
        freq = Counter(results)
        print(f"  → Frequency: {dict(freq)}")
        
        # Check if we found the expected minimum for 3x3 (should be 4)
        if n == 3:
            optimal_count = freq.get(4, 0)
            print(f"  → Found optimal 4-point solution: {optimal_count}/{len(results)} times ({100*optimal_count/len(results):.1f}%)")
        print()
    
    # Generate comprehensive visualization after all trials
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE TREE VISUALIZATION")
    print("="*60)
    
    web_viz_dir = os.path.join(os.path.dirname(__file__), 'web_visualization')
    experiment_name = f"mcts_n{args_cli.start}-{args_cli.end-1}_trials{args_cli.repeat}"
    
    final_html = MCTS.save_final_visualization(web_viz_dir, experiment_name)
    
    if final_html:
        print(f"🎉 Comprehensive visualization saved to: {final_html}")
        print(f"📊 Open this file in a web browser to explore all trials and steps interactively!")
    else:
        print("❌ No visualization data found.")                            