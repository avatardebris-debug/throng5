"""
Request Tetra to create the initial concept library structure.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from throng4.llm_policy.openclaw_bridge import OpenClawBridge

def main():
    bridge = OpenClawBridge(game="ConceptBootstrap")
    
    query = """
You offered to create the initial concepts/library.json structure. I'm ready!

CONTEXT:
We've successfully trained Tetris curriculum L2-7:
- L4: 6.4 ± 8.8 lines/episode (max 50)
- L5: 6.7 ± 10.4 lines/episode (max 62)
- L6: 6.6 ± 13.3 lines/episode (max 69)  
- L7: 11.1 ± 20.3 lines/episode (max 114)

The agent learned to:
- Avoid creating holes
- Stack pieces relatively flat
- Clear lines when possible
- Struggle with patience (no Tetris strategy mastered)

WORKSPACE STRUCTURE CREATED:
~/.openclaw/workspace/concepts/
├─ library.json (needs creation)
├─ meta_concepts.json (needs creation)
└─ game_mappings/
   └─ tetris.json (needs creation)

PLEASE CREATE:

1. **library.json** - Initial concept taxonomy with:
   - Meta-concepts you identified (avoid_danger, shape_optimization, etc.)
   - Empty instances initially
   - Confidence tracking structure
   - Transfer potential metadata

2. **meta_concepts.json** - Top-level abstractions:
   - avoid_danger (spatial, temporal, resource variants)
   - goal_optimization
   - exploration_vs_exploitation
   - etc.

3. **tetris.json** - Tetris-specific policy instances based on the training results above

Use the JSON format you specified in your response:
```json
{
  "concept_id": "avoid_danger_spatial",
  "label": "Spatial Danger Avoidance",
  "description": "...",
  "abstraction_level": "meta",
  "parent": "avoid_danger",
  "parameters": {...},
  "instances": [...],
  "transfer_confidence": 0.85,
  "success_rate": {...}
}
```

Please provide the complete JSON files I should save. I'll save them and we can iterate.
"""
    
    print("Requesting initial concept library from Tetra...")
    print("=" * 70)
    response = bridge.query(query)
    
    print("\n" + "=" * 70)
    print("TETRA'S RESPONSE:")
    print("=" * 70)
    print(response.raw)
    
    # Save response to file for review
    output_file = os.path.expanduser("~/.openclaw/workspace/concepts/tetra_bootstrap_response.txt")
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(response.raw)
    
    print("\n" + "=" * 70)
    print(f"Response saved to: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
