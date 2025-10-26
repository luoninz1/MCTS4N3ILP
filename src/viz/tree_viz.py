"""
Tree visualization utilities for MCTS.

This module contains the global trial data storage and HTML generation utilities
that are used by the MCTS class for optional tree visualization.

Note: The visualization methods remain in the MCTS class as they need access
to instance variables. This module provides the shared storage and final HTML
generation capabilities.
"""

import os
import json
import datetime


class MCTSVisualizationStorage:
    """Global storage for MCTS tree visualization data across all trials."""

    # Global storage for all trials and steps (class variable)
    global_trial_data = []

    @classmethod
    def clear_global_data(cls):
        """Clear all global trial data. Call this at the start of a new experiment set."""
        cls.global_trial_data.clear()
        print("Global trial data cleared.")

    @classmethod
    def save_final_visualization(cls, web_viz_dir=None, experiment_name="mcts_experiment"):
        """
        Save the final comprehensive visualization at the end of all trials.

        Args:
            web_viz_dir: Directory to save the visualization HTML
            experiment_name: Name prefix for the output file

        Returns:
            str: Path to the saved HTML file, or None if no data
        """
        if not cls.global_trial_data:
            print("No global trial data to save.")
            return None

        if web_viz_dir is None:
            web_viz_dir = './web_visualization'

        # Create the web visualization directory
        os.makedirs(web_viz_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(web_viz_dir, f"{experiment_name}_comprehensive_{timestamp}.html")

        # Save comprehensive visualization
        cls.save_comprehensive_html(filename)

        return filename

    @classmethod
    def save_comprehensive_html(cls, filename="mcts_comprehensive_visualization.html"):
        """
        Create a comprehensive HTML file with all trials and steps using JSON data.
        Includes trial selection, step selection, and navigation.

        Note: This is a placeholder - the actual implementation remains in MCTS class
        for now to avoid breaking dependencies. Future refactoring can move the full
        HTML generation here.

        Args:
            filename: Path to save the HTML file
        """
        # Import here to avoid circular dependency
        from src.algos.mcts import MCTS

        # Delegate to MCTS class method (which has the full implementation)
        return MCTS.save_comprehensive_html(filename)


__all__ = ['MCTSVisualizationStorage']
