"""
Query Tetra about baseline design for meta-learning validation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from throng4.llm_policy.openclaw_bridge import OpenClawBridge

def main():
    bridge = OpenClawBridge(game="MetaLearningValidation")
    
    query = """
CONTEXT: We want to validate that the concept library system enables "learning to learn to learn" (Meta^3). We need rigorous baselines to prove the system works.

PROPOSED TESTING APPROACH:

**Test Suite:** Atari games (57 diverse games available)
- Progressive difficulty: Breakout → Pong → Space Invaders → Freeway → Montezuma's Revenge
- Measure learning speedup across games 1, 2, 5, 10, 20
- Track concept library growth and saturation

**Proposed Baselines:**

1. **Tabula Rasa** - Fresh agent each game, no concepts, no LLM
   - Measures: raw learning capability

2. **Weight Transfer Only** - Transfer NN weights between games, no concepts, no LLM
   - Measures: low-level feature reuse (Meta^0)

3. **Concepts Only (Frozen Library)** - Load concepts as fixed heuristics, no LLM queries
   - Measures: value of concepts themselves vs LLM reasoning

4. **Full System** - Concept library + Tetra queries + adaptive selection
   - Measures: complete meta-learning stack

**Key Metrics:**
- Episodes to reach 75% of baseline performance (should decrease over time)
- Transfer effectiveness: (baseline_episodes - concept_episodes) / baseline_episodes
- Concept hit rate: applicable_concepts / total_concepts
- Library saturation: new concepts per game (should decrease)

**Atari Technical Challenge:**
Atari uses visual input (210x160x3 pixels), but our agent expects feature vectors. Options:
1. Use RAM state instead of screen (simpler, 128 bytes)
2. Hand-crafted feature extraction (detect objects, positions)
3. CNN pre-processing layer

MY QUESTIONS:

1. **Baseline Design**: Are these the right baselines to isolate what's working? What am I missing?

2. **Frozen Library Mode**: Should I create a mode where concepts are loaded as fixed heuristics but you're not queried? This would test "concepts without LLM reasoning."

3. **Atari Input**: RAM state or screen pixels? RAM is simpler but less realistic. Screen requires CNN but tests visual transfer.

4. **Quick Validation**: Before full Atari suite, should we do a quick test:
   - Tetris → Breakout transfer (both spatial games)
   - Measure if Breakout learns 20%+ faster with Tetris concepts
   - If yes, proceed to full suite

5. **Statistical Rigor**: How many runs per baseline to prove significance? 5 runs? 10 runs?

6. **Concept Library Evolution**: When should concepts get:
   - Promoted to meta-concepts (after N successful transfers?)
   - Archived as failed (after N failed transfers?)
   - Merged with similar concepts?

7. **Phase 5 Clarification**: You mentioned "Policy Composition with anti-policy priority and confidence-gated A/B testing" earlier. Is this:
   - Something YOU do (LLM composes policies)?
   - Something the SYSTEM does (infrastructure for A/B testing)?
   - Part of this validation framework?

Please be specific about experimental design. We want to publish-quality validation of the meta-learning claim.
"""
    
    print("Querying Tetra about meta-learning validation design...")
    print("=" * 70)
    response = bridge.query(query)
    
    print("\n" + "=" * 70)
    print("TETRA'S RESPONSE:")
    print("=" * 70)
    print(response.raw)
    
    # Save response
    output_file = "tetra_metalearning_validation.txt"
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(response.raw)
    
    print("\n" + "=" * 70)
    print(f"Response saved to: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
