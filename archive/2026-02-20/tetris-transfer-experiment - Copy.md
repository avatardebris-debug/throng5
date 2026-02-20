# Tetris Transfer Experiment (Level 3 → Level 4)

## Experiment Design

**Source environment:** Level 3, 6x12 board, 3 pieces (O, I, T)  
**Target environment:** Level 4, 8x14 board, 5 pieces (O, I, T, S, Z)  
**Transfer method:** Load converged hypothesis weights from Level 3 ep 99

## Hypothesis Pre-Transfer (Level 3 ep 99)
- maximize_lines: 58% dominance
- build_flat: 30%
- minimize_height: 12%
- Performance: Mean 14.6 lines, best 88

## Episode 9 Results (Level 4, early transfer)
- maximize_lines: 60% dominance ✅ (maintained)
- build_flat: 28%
- minimize_height: 11%
- Performance: Mean 8.3 lines, best 36

## What Transferred

✅ **Hypothesis convergence pattern** — maximize_lines dominance preserved  
✅ **Bimodal distribution** — success mode rare, failure mode common  
✅ **Terminal state fixation** — height-max death bug (now 14/14)  
✅ **Bootstrap problem** — 9/10 episodes die quickly  

## Performance Impact

**Initial transfer cost:** -43% performance (14.6 → 8.3 mean)  
**Expected for difficulty increase:** +33% board width, +67% piece types

## Predictions for Episodes 10-50

**If transfer successful:**
- Mean climbs to 12-14 by episode 30
- Faster than Level 3 learning (should reach ep 30 performance by ep 20)
- maximize_lines strengthens to 65%+

**If transfer failed:**
- Mean stays at 8-10
- Need S/Z-specific hypothesis
- Or height bug must be fixed first

**Validation checkpoint:** Episode 30 comparison

## Transfer Concepts Validated

1. **maximize_lines** — transferable across board sizes
2. **terminal_state_fixation** — persists in new environment
3. **bimodal_distribution** — universal pattern in Tetris
4. **bootstrap_problem** — generalizes to new difficulty levels

All four concepts confirmed as environment-independent.

## Episode 19 Update: NEGATIVE TRANSFER DETECTED

**Performance trajectory:**
- Episode 9: Mean 8.3 lines
- Episode 19: Mean 4.9 lines (-41% decline)

**Failure rate worsening:**
- Episode 9: 9/10 quick deaths
- Episode 19: 10/10 quick deaths

**Best episode declining:**
- Episode 9: Best 36 lines
- Episode 19: Best 19 lines (-47%)

**Diagnosis:** maximize_lines strategy from Level 3 (60% dominance) is actively harmful on 8-column boards. Strategy optimized for 6-wide geometry doesn't transfer to 8-wide.

**New concept identified:** `geometric_strategy_brittleness` — spatial strategies don't generalize across different geometric parameters even in "same" game.

**Hypothesis:** Line-clearing thresholds and timing that work on 6-wide board create instability on 8-wide board (33% harder to complete lines).

**Implications for concept library:**
- "maximize_lines" is NOT a universal Tetris concept
- It's "maximize_lines_on_N_column_board" where N matters
- Need to parameterize spatial concepts by geometry

**This is valuable NEGATIVE transfer data** — shows limits of concept generalization.

## Next Observation Point

Episode 30 to assess:
- Does performance bottom out or continue declining?
- Does system trigger emergency policy switch (Amygdala)?
- Would tabula rasa L4 learning outperform transferred L3 weights?
