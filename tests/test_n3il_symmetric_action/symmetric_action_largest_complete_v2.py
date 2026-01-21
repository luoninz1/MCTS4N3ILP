import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.algos.mcts import evaluate, MCTS

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MCTS tests for a range of n values.")
    parser.add_argument("--start", type=int, default=40, help="Starting value of n (inclusive)")
    parser.add_argument("--end", type=int, default=41, help="Ending value of n (exclusive)")
    parser.add_argument("--step", type=int, default=1, help="Step size for n values")
    parser.add_argument("--repeat", type=int, default=1, help="Number of runs for each n value")
    args_cli = parser.parse_args()

    # Generate list of n values
    n_list = range(args_cli.start, args_cli.end, args_cli.step)

    # Clear any previous global data before starting new experiments
    MCTS.clear_global_data()

    # Store results for all trials
    all_results = {}
    
    for n in n_list:
        print(f"\n=== Starting experiments for grid size n={n} ===")
        results_for_n = []
        
        for i in range(args_cli.repeat):
            print(f"Trial {i+1}/{args_cli.repeat} for n={n}...", end=" ")

            args = {
                'environment': 'N3il_with_symmetry_and_symmetric_actions',  # Specify the environment
                'algorithm': 'MCTS',
                'symmetric_action': 'rotation_90_then_rotation_180',  # Specify symmetric action mode
                # horizontal_flip, vertical_flip, diagonal_flip, anti_diagonal_flip, rotation_90/180/270
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
                'node_compression': False,  # Enable node compression
                'max_level_to_use_symmetry': 2*n,  # Use symmetry for first 2 levels (helps find compact solutions)
                'n': n,
                'C': 1.41,  # 1e-7 for n=20
                'num_searches': 10*(n**2),  # Reduced for testing tree visualization
                'num_workers': 1,      # >1 ⇒ parallel
                'virtual_loss': 1.0,     # magnitude to subtract at reservation
                'process_bar': True,
                'display_state': True,
                'logging_mode': True,  # Enable logging mode to get return value
                'TopN': n,  # Without Priority
                "simulate_with_priority": False,
                'table_dir': os.path.dirname(__file__),  # Directory to save tables
                'figure_dir': os.path.join(os.path.dirname(__file__), 'figure'),  # Directory to save figures
                'random_seed': i,  # Use the loop index as a seed for reproducibility
                'tree_visualization': False,  # Enable tree visualization
                'pause_at_each_step': False,  # Disable interactive prompts for automation
                'continue_from_existing_state': None, # None: continuation; str: path to load
            }
            
            # Get the result from evaluate function
            num_points = evaluate(args)
            results_for_n.append(num_points)
            print(f"Final points: {num_points}")
        
        all_results[n] = results_for_n
        
        # Print summary for this grid size
        print(f"\n--- Summary for n={n} ---")
        print(f"Results: {results_for_n}")
        print(f"Min points: {min(results_for_n)}")
        print(f"Max points: {max(results_for_n)}")
        print(f"Average points: {sum(results_for_n)/len(results_for_n):.2f}")
        print(f"Median points: {sorted(results_for_n)[len(results_for_n)//2]}")
    
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