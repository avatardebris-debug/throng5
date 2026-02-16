"""
Test Rule Archive — SQLite persistence, decay, cross-game retrieval.

Tests:
1. Store and retrieve rules
2. Persistence across sessions (close/reopen DB)
3. Confidence decay on all rules
4. Cross-game retrieval (transferable rules)
5. Staleness queries
6. Search by description/feature
7. Statistics
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import os
import time
import numpy as np
from throng4.llm_policy.rule_archive import RuleArchive
from throng4.llm_policy.hypothesis import DiscoveredRule, RuleStatus, RuleLibrary, TestResult


def test_store_and_retrieve():
    """Test basic store and retrieve operations."""
    print("="*70)
    print("TEST 1: Store and Retrieve")
    print("="*70)
    
    # Clean slate
    db_path = "test_archive.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    archive = RuleArchive(db_path)
    
    # Create some rules
    rule1 = DiscoveredRule(
        id="gridworld_move_right",
        description="Moving right increases X coordinate",
        feature="position_x",
        direction="increase",
        status=RuleStatus.ACTIVE,
        confidence=0.9,
        environment_context="GridWorld_5x5"
    )
    
    rule2 = DiscoveredRule(
        id="tetris_avoid_gaps",
        description="Avoid leaving gaps in rows",
        feature="gaps",
        direction="minimize",
        status=RuleStatus.TENTATIVE,
        confidence=0.6,
        environment_context="Tetris_6x10",
        transferable=True
    )
    
    # Store
    archive.store_rule(rule1)
    archive.store_rule(rule2)
    
    print(f"\n  Stored 2 rules")
    
    # Retrieve
    retrieved1 = archive.get_rule("gridworld_move_right")
    retrieved2 = archive.get_rule("tetris_avoid_gaps")
    
    assert retrieved1 is not None, "Should retrieve rule1"
    assert retrieved2 is not None, "Should retrieve rule2"
    assert retrieved1.description == rule1.description
    assert retrieved2.transferable == True
    
    print(f"  Retrieved: {retrieved1.description}")
    print(f"  Retrieved: {retrieved2.description}")
    
    # Get by environment
    gridworld_rules = archive.get_rules_for_env("GridWorld_5x5")
    assert len(gridworld_rules) == 1
    print(f"\n  GridWorld rules: {len(gridworld_rules)}")
    
    # Get active
    active = archive.get_active_rules()
    assert len(active) == 1
    print(f"  Active rules: {len(active)}")
    
    # Get transferable
    transferable = archive.get_transferable_rules()
    assert len(transferable) == 1
    print(f"  Transferable rules: {len(transferable)}")
    
    archive.close()
    
    print("\n[OK] Store and retrieve test passed!")
    return db_path


def test_persistence(db_path):
    """Test that rules persist across sessions."""
    print("\n" + "="*70)
    print("TEST 2: Persistence Across Sessions")
    print("="*70)
    
    # Reopen the same database
    archive = RuleArchive(db_path)
    
    # Should still have the 2 rules from previous test
    all_rules = archive.get_rules_for_env("GridWorld_5x5") + archive.get_rules_for_env("Tetris_6x10")
    
    print(f"\n  Rules after reopening DB: {len(all_rules)}")
    assert len(all_rules) == 2, "Should persist across sessions"
    
    for rule in all_rules:
        print(f"    - {rule.description} ({rule.environment_context})")
    
    archive.close()
    
    print("\n[OK] Persistence test passed!")


def test_confidence_decay(db_path):
    """Test confidence decay on all archived rules."""
    print("\n" + "="*70)
    print("TEST 3: Confidence Decay")
    print("="*70)
    
    archive = RuleArchive(db_path)
    
    # Get current rules
    rule = archive.get_rule("gridworld_move_right")
    initial_confidence = rule.confidence
    
    print(f"\n  Initial confidence: {initial_confidence:.3f}")
    
    # Apply decay (simulate 10 hours passing)
    archive.apply_decay_all(decay_hours=10.0)
    
    # Retrieve again
    rule_after = archive.get_rule("gridworld_move_right")
    
    print(f"  After 10h decay: {rule_after.confidence:.3f}")
    print(f"  Status: {rule_after.status.value}")
    
    assert rule_after.confidence < initial_confidence, "Confidence should have decayed"
    assert rule_after.confidence > 0.01, "Should never be zero"
    
    archive.close()
    
    print("\n[OK] Confidence decay test passed!")


def test_cross_game_retrieval(db_path):
    """Test cross-game pattern matching via transferable rules."""
    print("\n" + "="*70)
    print("TEST 4: Cross-Game Retrieval")
    print("="*70)
    
    archive = RuleArchive(db_path)
    
    # Add a new game's rules
    new_rule = DiscoveredRule(
        id="puzzle_game_avoid_gaps",
        description="Avoid gaps in placement (similar to Tetris)",
        feature="gaps",
        direction="minimize",
        status=RuleStatus.TENTATIVE,
        confidence=0.5,
        environment_context="PuzzleGame_8x8",
        transferable=True
    )
    
    archive.store_rule(new_rule)
    
    # Search for transferable rules about "gaps"
    gap_rules = archive.search_by_feature("gaps")
    
    print(f"\n  Rules about 'gaps': {len(gap_rules)}")
    for rule in gap_rules:
        print(f"    - {rule.description} ({rule.environment_context})")
    
    assert len(gap_rules) >= 2, "Should find gap rules from multiple games"
    
    # Get all transferable
    transferable = archive.get_transferable_rules()
    print(f"\n  Total transferable rules: {len(transferable)}")
    
    archive.close()
    
    print("\n[OK] Cross-game retrieval test passed!")


def test_staleness_queries(db_path):
    """Test staleness and re-evaluation queries."""
    print("\n" + "="*70)
    print("TEST 5: Staleness Queries")
    print("="*70)
    
    archive = RuleArchive(db_path)
    
    # Create a stale rule (last tested long ago)
    old_rule = DiscoveredRule(
        id="old_rule",
        description="Old rule that needs re-evaluation",
        feature="test",
        status=RuleStatus.ACTIVE,
        confidence=0.7,
        environment_context="TestEnv"
    )
    old_rule.last_tested = time.time() - (100 * 3600)  # 100 hours ago
    
    archive.store_rule(old_rule)
    
    # Query stale rules (> 48 hours)
    stale = archive.get_stale_rules(max_age_hours=48.0)
    
    print(f"\n  Stale rules (>48h): {len(stale)}")
    for rule in stale:
        hours_old = (time.time() - rule.last_tested) / 3600
        print(f"    - {rule.description} (last tested {hours_old:.1f}h ago)")
    
    assert len(stale) >= 1, "Should find stale rule"
    
    archive.close()
    
    print("\n[OK] Staleness queries test passed!")


def test_search(db_path):
    """Test search by description."""
    print("\n" + "="*70)
    print("TEST 6: Search")
    print("="*70)
    
    archive = RuleArchive(db_path)
    
    # Search for "gaps"
    results = archive.search_by_description("gaps")
    
    print(f"\n  Search 'gaps': {len(results)} results")
    for rule in results:
        print(f"    - {rule.description}")
    
    assert len(results) >= 2, "Should find multiple gap-related rules"
    
    # Search for "right"
    results2 = archive.search_by_description("right")
    print(f"\n  Search 'right': {len(results2)} results")
    for rule in results2:
        print(f"    - {rule.description}")
    
    archive.close()
    
    print("\n[OK] Search test passed!")


def test_statistics(db_path):
    """Test archive statistics."""
    print("\n" + "="*70)
    print("TEST 7: Statistics")
    print("="*70)
    
    archive = RuleArchive(db_path)
    
    stats = archive.get_statistics()
    
    print(f"\n  Archive Statistics:")
    print(f"    Total rules: {stats['total_rules']}")
    print(f"    Average confidence: {stats['avg_confidence']:.3f}")
    print(f"    Transferable: {stats['transferable']}")
    
    print(f"\n  By Status:")
    for status, count in stats['by_status'].items():
        print(f"    {status}: {count}")
    
    print(f"\n  By Environment:")
    for env, count in stats['by_environment'].items():
        print(f"    {env}: {count}")
    
    assert stats['total_rules'] >= 4, "Should have multiple rules"
    
    archive.close()
    
    print("\n[OK] Statistics test passed!")


def main():
    # Run tests in sequence
    db_path = test_store_and_retrieve()
    test_persistence(db_path)
    test_confidence_decay(db_path)
    test_cross_game_retrieval(db_path)
    test_staleness_queries(db_path)
    test_search(db_path)
    test_statistics(db_path)
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print("\n" + "="*70)
    print("[OK] ALL ARCHIVE TESTS PASSED!")
    print("="*70)


if __name__ == '__main__':
    main()
