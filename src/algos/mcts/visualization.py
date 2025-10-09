import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import datetime
import json
from pyvis.network import Network


def _state_to_image_base64(game, state):
    """
    Convert a game state to a base64-encoded image using the game's display_state method.
    """
    # Create a temporary figure
    plt.figure(figsize=(4, 4))

    # Get the grid info
    rows, cols = game.row_count, game.column_count

    # Plot the state (simplified version of display_state)
    y_idx, x_idx = np.nonzero(state)
    y_disp = rows - 1 - y_idx
    plt.scatter(x_idx, y_disp, s=200, c='blue', linewidths=0.5)

    # Draw grid
    plt.xticks(range(cols))
    plt.yticks(range(rows))
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, cols - 0.5)
    plt.ylim(-0.5, rows - 0.5)
    plt.gca().set_aspect('equal')

    # Remove axes labels and ticks for cleaner look
    plt.xticks([])
    plt.yticks([])

    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=80, pad_inches=0.1)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()

    return f"data:image/png;base64,{img_base64}"


def _get_node_label(node, iter_num=None):
    """
    Generate a label for a node showing its statistics.
    """
    avg_value = node.value_sum / node.visit_count if node.visit_count > 0 else 0

    # Calculate UCB if this node has a parent
    ucb = 0
    if node.parent is not None and node.visit_count > 0:
        try:
            ucb = node.parent.get_ucb(node, iter_num or 0)
        except:
            ucb = 0

    label = f"Visits: {node.visit_count}\n"
    label += f"Value Sum: {node.value_sum:.3f}\n"
    label += f"Avg Value: {avg_value:.3f}\n"
    label += f"UCB: {ucb:.3f}"

    return label


def tree_visualization(mcts_instance, root, snapshot_name="MCTS Tree"):
    """
    Create a tree visualization using pyvis and save it as a snapshot.
    Ensures all expanded nodes are captured, including nodes with no children.
    """
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.barnes_hut()

    # Collect all nodes using DFS to ensure we capture everything
    all_nodes = []
    visited = set()

    def collect_nodes_dfs(node, level=0):
        if id(node) in visited:
            return
        visited.add(id(node))
        all_nodes.append((node, level))

        # Recursively collect children
        for child in node.children:
            collect_nodes_dfs(child, level + 1)

    # Start DFS from root
    collect_nodes_dfs(root)

    print(f"Tree visualization: Found {len(all_nodes)} nodes total")

    # Prepare JSON-serializable data for nodes and edges
    json_nodes = []
    json_edges = []
    node_mapping = {}

    # Add nodes to the network
    for i, (node, level) in enumerate(all_nodes):
        # Generate unique node ID
        current_id = f"node_{i}"
        node_mapping[id(node)] = current_id

        # Get node image and label
        img_base64 = _state_to_image_base64(mcts_instance.game, node.state)
        label = _get_node_label(node)

        # Determine node color based on properties
        color = "#4CAF50"  # Default green
        if node.is_fully_expanded():
            color = "#2196F3"  # Blue for fully expanded
        elif len(node.children) == 0 and not node.is_fully_expanded():
            color = "#FF9800"  # Orange for leaf nodes that could expand
        elif np.sum(node.valid_moves) == 0:
            color = "#F44336"  # Red for terminal nodes

        # Add node to pyvis network
        net.add_node(
            current_id,
            label=label,
            image=img_base64,
            shape="image",
            size=30,
            level=level,
            color=color,
            title=f"Action: {node.action_taken}\n{label}\nChildren: {len(node.children)}\nValid moves left: {np.sum(node.valid_moves)}"
        )

        # Prepare JSON data for this node - split label into lines for proper display
        label_lines = label.split('\n')
        # Create title with proper newline escaping
        escaped_label = label.replace('\n', '\\n')
        title_text = f"Action: {node.action_taken}\\n{escaped_label}\\nChildren: {len(node.children)}\\nValid moves left: {np.sum(node.valid_moves)}"

        json_nodes.append({
            "id": current_id,
            "label": label_lines,  # Use array of lines instead of single string
            "image": img_base64,
            "shape": "image",
            "size": 30,
            "level": level,
            "color": color,
            "title": title_text,
            "x": i * 100,  # Simple layout
            "y": level * 150
        })

    # Add edges between nodes
    for node, _ in all_nodes:
        current_id = node_mapping[id(node)]
        for child in node.children:
            if id(child) in node_mapping:  # Ensure child was also collected
                child_id = node_mapping[id(child)]
                net.add_edge(current_id, child_id)
                json_edges.append({
                    "from": current_id,
                    "to": child_id,
                    "smooth": {"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4}
                })

    print(f"Tree visualization: Added {len(net.nodes)} nodes and {len(net.edges)} edges")

    # Configure layout
    net.set_options("""
    var options = {
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "UD",
                "sortMethod": "directed",
                "shakeTowards": "roots",
                "levelSeparation": 150,
                "nodeSpacing": 100
            }
        },
        "physics": {
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 100,
                "springConstant": 0.01,
                "nodeDistance": 120,
                "damping": 0.09
            },
            "maxVelocity": 50,
            "solver": "hierarchicalRepulsion",
            "stabilization": {"iterations": 100}
        },
        "nodes": {
            "font": {
                "size": 12,
                "color": "white"
            }
        },
        "edges": {
            "smooth": {
                "type": "cubicBezier",
                "forceDirection": "vertical",
                "roundness": 0.4
            }
        }
    }
    """)

    # Store this snapshot with trial information
    snapshot_data = {
        'name': snapshot_name,
        'network': net,
        'html': net.generate_html(),
        'trial_id': getattr(mcts_instance, 'trial_id', 'unknown'),
        'step_number': len(mcts_instance.snapshots),
        'total_nodes': len(all_nodes),
        'args': mcts_instance.args.copy(),  # Store configuration for reference
        'json_nodes': json_nodes,  # Add JSON data for better HTML generation
        'json_edges': json_edges
    }

    mcts_instance.snapshots.append(snapshot_data)

    # Also add to global trial data for multi-trial viewing
    # Import here to avoid circular dependency
    from src.algos.mcts.tree_search import MCTS
    MCTS.global_trial_data.append(snapshot_data)

    return net


def save_final_visualization(web_viz_dir=None, experiment_name="mcts_experiment"):
    """
    Save the final comprehensive visualization at the end of all trials.
    """
    # Import here to avoid circular dependency
    from src.algos.mcts.tree_search import MCTS

    if not MCTS.global_trial_data:
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
    save_comprehensive_html(filename)

    return filename


def save_comprehensive_html(filename="mcts_comprehensive_visualization.html"):
    """
    Create a comprehensive HTML file with all trials and steps using JSON data.
    Includes trial selection, step selection, and navigation.
    """
    # Import here to avoid circular dependency
    from src.algos.mcts.tree_search import MCTS

    if not MCTS.global_trial_data:
        print("No global trial data to save.")
        return

    # Organize data by trial and step
    trials_data = {}
    json_snapshots = []

    for snapshot in MCTS.global_trial_data:
        trial_id = snapshot['trial_id']
        step_num = snapshot['step_number']

        if trial_id not in trials_data:
            trials_data[trial_id] = {}
        trials_data[trial_id][step_num] = snapshot

        # Prepare JSON snapshot data
        json_snapshots.append({
            "id": f"t{trial_id}_s{step_num}",
            "title": f"Trial {trial_id} - {snapshot['name']}",
            "trial_id": trial_id,
            "step_number": step_num,
            "total_nodes": snapshot['total_nodes'],
            "nodes": snapshot.get('json_nodes', []),
            "edges": snapshot.get('json_edges', []),
            "grid_size": snapshot.get('args', {}).get('n', 'unknown')
        })

    print(f"Organizing {len(MCTS.global_trial_data)} snapshots across {len(trials_data)} trials")

    # Also save JSON data to separate file
    json_filename = filename.replace('.html', '_data.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_snapshots, f, indent=2, ensure_ascii=False)
    print(f"JSON data saved to: {json_filename}")

    # Create the comprehensive HTML using the improved approach
    html_content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>MCTS Comprehensive Tree Visualization</title>
  <style>
    body {{
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background-color: #1a1a1a;
        color: #ffffff;
    }}
    #toolbar {{
        display: flex;
        gap: 12px;
        align-items: center;
        padding: 16px;
        border-bottom: 1px solid #333;
        background-color: #2d2d2d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }}
    #mynetwork {{
        height: calc(100vh - 120px);
        background-color: #222222;
        border-radius: 8px;
        margin: 16px;
        border: 1px solid #444;
    }}
    button, select {{
        padding: 8px 16px;
        border-radius: 6px;
        border: 1px solid #555;
        background: #3a3a3a;
        color: #ffffff;
        cursor: pointer;
        transition: background-color 0.2s;
        font-size: 14px;
    }}
    button:hover, select:hover {{
        background: #4a4a4a;
    }}
    button:disabled {{
        background: #2a2a2a;
        color: #666;
        cursor: not-allowed;
    }}
    #title {{
        font-weight: 600;
        margin-left: 16px;
        font-size: 16px;
        color: #4CAF50;
    }}
    .stats {{
        display: flex;
        gap: 16px;
        margin-left: auto;
        font-size: 12px;
        color: #aaa;
    }}
    .stat-item {{
        background: #333;
        padding: 4px 8px;
        border-radius: 4px;
    }}
    .keyboard-help {{
        position: fixed;
        bottom: 16px;
        right: 16px;
        background: #333;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 12px;
        color: #aaa;
        border: 1px solid #444;
    }}
  </style>
</head>
<body>
  <div id="toolbar">
    <button id="prev">← Prev Step</button>
    <button id="next">Next Step →</button>
    <select id="trialSelect">
        <option value="">Select Trial...</option>"""

    # Add trial options
    for trial_id in sorted(trials_data.keys()):
        trial_steps = len(trials_data[trial_id])
        html_content += f'<option value="{trial_id}">Trial {trial_id} ({trial_steps} steps)</option>'

    html_content += f"""
    </select>
    <select id="stepSelect">
        <option value="">Select Step...</option>
    </select>
    <button id="autoPlay">⏯ Auto Play</button>
    <span id="title">MCTS Tree Visualization</span>
    <div class="stats">
        <div class="stat-item">Nodes: <span id="nodeCount">-</span></div>
        <div class="stat-item">Grid: <span id="gridSize">-</span></div>
        <div class="stat-item">Step: <span id="currentStep">-</span></div>
    </div>
  </div>
  <div id="mynetwork"></div>
  <div class="keyboard-help">
    ⌨️ Use ← → for steps, ↑ ↓ for trials, Space for auto-play
  </div>

  <!-- vis-network (vis.js) -->
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <script>
    const snapshots = {json.dumps(json_snapshots, ensure_ascii=False, indent=2)};
    let currentIdx = 0;
    let autoPlayInterval = null;
    let isAutoPlaying = false;

    // Initialize DataSet and Network
    const nodes = new vis.DataSet([]);
    const edges = new vis.DataSet([]);
    const container = document.getElementById('mynetwork');
    const network = new vis.Network(container, {{ nodes, edges }}, {{
      layout: {{
        hierarchical: {{
          enabled: true,
          direction: "UD",
          sortMethod: "directed",
          levelSeparation: 150,
          nodeSpacing: 100,
          treeSpacing: 200
        }}
      }},
      physics: {{
        hierarchicalRepulsion: {{
          centralGravity: 0.0,
          springLength: 100,
          springConstant: 0.01,
          nodeDistance: 120,
          damping: 0.09
        }},
        maxVelocity: 50,
        solver: "hierarchicalRepulsion",
        stabilization: {{iterations: 100}}
      }},
      nodes: {{
        font: {{
          size: 11,
          color: '#ffffff',
          multi: false,
          align: 'center'
        }},
        borderWidth: 2,
        shadow: true,
        widthConstraint: {{ maximum: 150 }},
        heightConstraint: {{ minimum: 80 }}
      }},
      edges: {{
        color: {{ color: '#666666' }},
        smooth: {{
          type: "cubicBezier",
          forceDirection: "vertical",
          roundness: 0.4
        }},
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }}
      }},
      interaction: {{ hover: true }},
      configure: {{ enabled: false }}
    }});

    // Load a specific snapshot
    function loadSnapshot(idx) {{
      if (idx < 0 || idx >= snapshots.length) return;

      currentIdx = idx;
      const snapshot = snapshots[idx];

      // Clear and add new data
      nodes.clear();
      edges.clear();

      // Process nodes - convert label arrays to multi-line text
      if (snapshot.nodes && snapshot.nodes.length > 0) {{
        const processedNodes = snapshot.nodes.map(node => {{
          if (Array.isArray(node.label)) {{
            // Convert array of lines to newline-separated text for vis.js
            node.label = node.label.join('\\n');
          }}
          return node;
        }});
        nodes.add(processedNodes);
      }}
      if (snapshot.edges && snapshot.edges.length > 0) {{
        edges.add(snapshot.edges);
      }}

      // Update UI
      document.getElementById('title').textContent = snapshot.title;
      document.getElementById('nodeCount').textContent = snapshot.total_nodes;
      document.getElementById('gridSize').textContent = snapshot.grid_size;
      document.getElementById('currentStep').textContent = `${{snapshot.step_number + 1}}/${{getStepsForTrial(snapshot.trial_id)}}`;

      // Update dropdowns
      document.getElementById('trialSelect').value = snapshot.trial_id;
      updateStepDropdown(snapshot.trial_id);
      document.getElementById('stepSelect').value = snapshot.step_number;

      // Update button states
      updateButtons();

      // Fit the network view
      setTimeout(() => network.fit({{ animation: true }}), 100);
    }}

    function getStepsForTrial(trialId) {{
      return snapshots.filter(s => s.trial_id === trialId).length;
    }}

    function updateStepDropdown(trialId) {{
      const stepSelect = document.getElementById('stepSelect');
      stepSelect.innerHTML = '<option value="">Select Step...</option>';

      const trialSnapshots = snapshots.filter(s => s.trial_id === trialId).sort((a, b) => a.step_number - b.step_number);
      trialSnapshots.forEach(snapshot => {{
        const option = document.createElement('option');
        option.value = snapshot.step_number;
        option.textContent = `Step ${{snapshot.step_number + 1}} (${{snapshot.total_nodes}} nodes)`;
        stepSelect.appendChild(option);
      }});
    }}

    function updateButtons() {{
      const prevBtn = document.getElementById('prev');
      const nextBtn = document.getElementById('next');

      prevBtn.disabled = currentIdx <= 0;
      nextBtn.disabled = currentIdx >= snapshots.length - 1;
    }}

    function toggleAutoPlay() {{
      isAutoPlaying = !isAutoPlaying;
      const btn = document.getElementById('autoPlay');

      if (isAutoPlaying) {{
        btn.textContent = '⏸ Pause';
        autoPlayInterval = setInterval(() => {{
          if (currentIdx < snapshots.length - 1) {{
            loadSnapshot(currentIdx + 1);
          }} else {{
            toggleAutoPlay(); // Stop at end
          }}
        }}, 2000);
      }} else {{
        btn.textContent = '⏯ Auto Play';
        if (autoPlayInterval) {{
          clearInterval(autoPlayInterval);
          autoPlayInterval = null;
        }}
      }}
    }}

    // Event handlers
    document.getElementById('prev').onclick = () => loadSnapshot(currentIdx - 1);
    document.getElementById('next').onclick = () => loadSnapshot(currentIdx + 1);
    document.getElementById('autoPlay').onclick = toggleAutoPlay;

    document.getElementById('trialSelect').onchange = (e) => {{
      if (e.target.value) {{
        updateStepDropdown(e.target.value);
        const firstStep = snapshots.find(s => s.trial_id === e.target.value);
        if (firstStep) {{
          const idx = snapshots.indexOf(firstStep);
          loadSnapshot(idx);
        }}
      }}
    }};

    document.getElementById('stepSelect').onchange = (e) => {{
      const trialId = document.getElementById('trialSelect').value;
      if (trialId && e.target.value !== '') {{
        const stepNum = parseInt(e.target.value);
        const snapshot = snapshots.find(s => s.trial_id === trialId && s.step_number === stepNum);
        if (snapshot) {{
          const idx = snapshots.indexOf(snapshot);
          loadSnapshot(idx);
        }}
      }}
    }};

    // Keyboard controls
    document.addEventListener('keydown', (e) => {{
      switch(e.key) {{
        case 'ArrowLeft':
          e.preventDefault();
          loadSnapshot(currentIdx - 1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          loadSnapshot(currentIdx + 1);
          break;
        case 'ArrowUp':
          e.preventDefault();
          // Previous trial
          const currentTrial = snapshots[currentIdx]?.trial_id;
          const trials = [...new Set(snapshots.map(s => s.trial_id))].sort();
          const currentTrialIdx = trials.indexOf(currentTrial);
          if (currentTrialIdx > 0) {{
            const prevTrial = trials[currentTrialIdx - 1];
            const firstStepOfPrevTrial = snapshots.find(s => s.trial_id === prevTrial);
            if (firstStepOfPrevTrial) {{
              loadSnapshot(snapshots.indexOf(firstStepOfPrevTrial));
            }}
          }}
          break;
        case 'ArrowDown':
          e.preventDefault();
          // Next trial
          const currentTrial2 = snapshots[currentIdx]?.trial_id;
          const trials2 = [...new Set(snapshots.map(s => s.trial_id))].sort();
          const currentTrialIdx2 = trials2.indexOf(currentTrial2);
          if (currentTrialIdx2 < trials2.length - 1) {{
            const nextTrial = trials2[currentTrialIdx2 + 1];
            const firstStepOfNextTrial = snapshots.find(s => s.trial_id === nextTrial);
            if (firstStepOfNextTrial) {{
              loadSnapshot(snapshots.indexOf(firstStepOfNextTrial));
            }}
          }}
          break;
        case ' ':
          e.preventDefault();
          toggleAutoPlay();
          break;
      }}
    }});

    // Initialize
    if (snapshots.length > 0) {{
      loadSnapshot(0);
    }} else {{
      document.getElementById('title').textContent = 'No snapshots available';
    }}
  </script>
</body>
</html>
"""

    # Write the file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Comprehensive visualization saved to: {filename}")
    print(f"Total snapshots: {len(MCTS.global_trial_data)}")
    print(f"Trials: {len(trials_data)}")

    return filename
