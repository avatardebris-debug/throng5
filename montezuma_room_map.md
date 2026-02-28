# Montezuma's Revenge — Room 0 Coordinate Map
**Source: Human playthrough 2026-02-25. RAM[42]=x, RAM[43]=y. Y increases downward.**

Jump height ≈ +20 y units.

## Platform Layout

| Platform              | Y      | X range    | Notes                                      |
|-----------------------|--------|------------|--------------------------------------------|
| Top platform          | 235    | 0–160      | Agent spawns at x=77                       |
| ↳ Left door           | 235    | x < 7      | Exit room left (key required)              |
| ↳ Left safe zone      | 235    | 0–48       |                                            |
| ↳ Gap (conveyor belt) | 235    | 48–102     | Conveyor pushes left; safe to cross        |
| ↳ Right safe zone     | 235    | 102+       |                                            |
| ↳ Right door          | 235    | x ≈ 130    | Exit room right (key required)             |
| Center (mid) platform | ~220   | 67–88      | Conveyor belt platform below top           |
| Rope                  | 198–212| x ≈ 109    | Grab from right by jumping right ~x=109    |
| Right platform        | ~210   | 127–145    | Safe — connects to right side              |
| Left platform (key)   | 195–215| 9–33       | Left wall area — safe. Key at x=15, y=200 |
| Left ladder           | 195–235| x ≈ 20     | Vertical ladder connecting platforms       |
| Floor (ground)        | 148    | 0–160      | Main traversal level                       |

## Key Location
- **Key**: x=13–17, y=195–215 (approximately x=15, y=200)

## Lethal / Death Zones

> [!WARNING]
> These zones cause instant death when entered at wrong heights.

| Zone                          | Condition                              | Why                                     |
|-------------------------------|----------------------------------------|-----------------------------------------|
| **Mid-air void**              | y=195–215 AND x=34–108                 | No floor, no rope — falls to y=148     |
| Exception (rope grab)         | y=198–212 AND x=105–113                | Rope is here — safe if grabbed          |
| **Right platform gap**        | y=195–215 AND x>145                    | Nothing to the right of right platform  |
| **Top platform right drop**   | y=235 AND x > 160                      | Off-screen                              |

## Skull Zone
- Skull patrols: x=60–111, y≈148 (floor level, moving)
- Safe crossing: time it or use gaps in patrol pattern

## Conveyor Belt
- Location: center platform x≈65–105, y≈220
- Effect: pushes agent LEFT automatically (faster than walking)
- Strategy: jump RIGHT to make rightward progress against conveyor

## Subgoal Coordinates (Calibrated)

| Subgoal               | X         | Y         | Notes                              |
|-----------------------|-----------|-----------|------------------------------------|
| Spawn                 | 77        | 235       | Episode start                      |
| Rope grabbed          | ~109      | 198–212   | Jump right from ~x=90+             |
| Right platform        | 127–145   | ~210      |                                    |
| Left ladder descent   | ~20       | 195–235   | Ladder column                      |
| Left platform         | 9–33      | 195–215   |                                    |
| Key collected         | 13–17     | ~200      | RAM[56] / RAM[65] bit changes      |
| Back on floor         | any       | ~148      | After key, descend left ladder     |
| Floor right traverse  | 111+      | ~148      | Past skull zone                    |
| Left door (exit)      | x < 7     | 235       | Requires key                       |
| Right door (exit)     | x ≈ 150+  | ~235      | Requires key, exact x TBD          |

## MCTS Action Mask Rules (room_constants.py)

```
LETHAL if: 192 < y < 220  AND  34 < x < 106   (void below center/left platforms)
  EXCEPT:  105 < x < 113  AND  196 < y < 214   (rope — safe zone)

LETHAL if: 192 < y < 220  AND  x > 145          (right of right platform)
```

> [!NOTE]
> Top platform (y=235) is entirely safe for lateral movement.
> Conveyor belt does NOT cause death, just pushes left.
> The agent CAN jump right against the conveyor to make progress.
