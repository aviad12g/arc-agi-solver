#!/usr/bin/env python3
"""Analyze portfolio performance metrics for budget tuning.

Reads portfolio results JSONL and provides insights about:
- Which lanes win most often
- Per-lane performance statistics
- Budget optimization recommendations
- Success patterns and bottlenecks
"""

import argparse
import json
import statistics
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """Load results from JSON or JSONL file."""
    results = []

    try:
        # Try to load as single JSON first
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Check if it's a single result or a list
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict):
            # Check if it has the structure of our portfolio comparison
            if 'portfolio_enabled' in data and 'portfolio_disabled' in data:
                # Convert to result format
                for task in data['portfolio_enabled']['tasks']:
                    task_copy = task.copy()
                    task_copy['search_stats'] = {'portfolio_metrics': {'winner': 'portfolio'}}
                    results.append(task_copy)
                for task in data['portfolio_disabled']['tasks']:
                    task_copy = task.copy()
                    task_copy['search_stats'] = {'portfolio_metrics': {'winner': 'astar'}}
                    results.append(task_copy)
            else:
                results = [data]
    except json.JSONDecodeError:
        # Try as JSONL
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))

    return results


def analyze_lane_performance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze which lanes win and their performance characteristics."""

    lane_wins = Counter()
    lane_times = defaultdict(list)
    lane_nodes = defaultdict(list)
    lane_success_rates = defaultdict(list)

    total_tasks = len(results)

    for result in results:
        if 'success' not in result:
            continue

        success = result['success']
        search_stats = result.get('search_stats', {})

        # Check for portfolio metrics
        portfolio_metrics = search_stats.get('portfolio_metrics', {})

        if portfolio_metrics:
            # Portfolio was used - analyze winning lane
            winner = portfolio_metrics.get('winner', 'unknown')
            lane_wins[winner] += 1

            if 'lane_times' in portfolio_metrics:
                for lane, time_taken in portfolio_metrics['lane_times'].items():
                    lane_times[lane].append(time_taken)
                    # Success rate per lane (if lane completed successfully)
                    if time_taken < 30.0:  # Reasonable timeout
                        lane_success_rates[lane].append(1 if success else 0)

            # Record nodes expanded for winning lane
            if 'lane_nodes' in portfolio_metrics and winner in portfolio_metrics['lane_nodes']:
                lane_nodes[winner].append(portfolio_metrics['lane_nodes'][winner])

        else:
            # Direct A* - count as 'astar' lane
            lane_wins['astar'] += 1
            solve_time = result.get('computation_time', 0)
            nodes_expanded = result.get('nodes_expanded', 0)

            lane_times['astar'].append(solve_time)
            lane_nodes['astar'].append(nodes_expanded)
            lane_success_rates['astar'].append(1 if success else 0)

    # Calculate statistics
    analysis = {
        'total_tasks': total_tasks,
        'lane_wins': dict(lane_wins),
        'lane_statistics': {}
    }

    for lane in set(list(lane_times.keys()) + list(lane_nodes.keys())):
        times = lane_times[lane]
        nodes = lane_nodes[lane]
        successes = lane_success_rates[lane]

        if times:
            analysis['lane_statistics'][lane] = {
                'win_rate': lane_wins[lane] / total_tasks if total_tasks > 0 else 0,
                'avg_time': statistics.mean(times) if times else 0,
                'median_time': statistics.median(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'avg_nodes': statistics.mean(nodes) if nodes else 0,
                'median_nodes': statistics.median(nodes) if nodes else 0,
                'success_rate': sum(successes) / len(successes) if successes else 0,
                'sample_count': len(times)
            }

    return analysis


def generate_budget_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate budget tuning recommendations based on analysis."""

    recommendations = []

    lane_stats = analysis.get('lane_statistics', {})
    total_tasks = analysis.get('total_tasks', 0)

    if not lane_stats:
        return ["No lane statistics available for analysis"]

    # Find most effective lanes
    effective_lanes = sorted(
        [(lane, stats) for lane, stats in lane_stats.items()],
        key=lambda x: (
            x[1]['win_rate'] * x[1]['success_rate'],  # Combined effectiveness
            -x[1]['median_time']  # Prefer faster lanes
        ),
        reverse=True
    )

    # Budget allocation recommendations
    recommendations.append("🎯 BUDGET TUNING RECOMMENDATIONS:")
    recommendations.append("")

    # Top performing lanes
    recommendations.append("🏆 TOP PERFORMING LANES:")
    for i, (lane, stats) in enumerate(effective_lanes[:3], 1):
        win_pct = stats['win_rate'] * 100
        success_pct = stats['success_rate'] * 100
        med_time = stats['median_time']
        recommendations.append(f"  {i}. {lane}: {win_pct:.1f}% wins, {success_pct:.1f}% success, {med_time:.3f}s median")

    recommendations.append("")

    # Time allocation suggestions
    recommendations.append("⏱️  BUDGET ALLOCATION SUGGESTIONS:")

    # Allocate more time to high-win-rate lanes
    total_win_rate = sum(stats['win_rate'] for stats in lane_stats.values())

    for lane, stats in effective_lanes:
        if total_win_rate > 0:
            suggested_budget = (stats['win_rate'] / total_win_rate) * 2.0  # Scale to ~2s total
            recommendations.append(f"  {lane}: {suggested_budget:.2f}s (based on {stats['win_rate']*100:.1f}% win rate)")

    recommendations.append("")

    # Efficiency analysis
    recommendations.append("⚡ EFFICIENCY ANALYSIS:")

    # Find lanes with high success but low time
    efficient_lanes = [(lane, stats) for lane, stats in lane_stats.items()
                      if stats['success_rate'] > 0.8 and stats['median_time'] < 1.0]

    if efficient_lanes:
        recommendations.append("  Fast & reliable lanes:")
        for lane, stats in efficient_lanes[:3]:
            recommendations.append(f"    {lane}: {stats['median_time']:.3f}s median, {stats['success_rate']*100:.1f}% success")

    # Find problematic lanes
    problematic_lanes = [(lane, stats) for lane, stats in lane_stats.items()
                        if stats['success_rate'] < 0.5 or stats['median_time'] > 5.0]

    if problematic_lanes:
        recommendations.append("  ⚠️  Potentially problematic lanes:")
        for lane, stats in problematic_lanes:
            issues = []
            if stats['success_rate'] < 0.5:
                issues.append(".1f")
            if stats['median_time'] > 5.0:
                issues.append(".1f")
            recommendations.append(f"    {lane}: {', '.join(issues)}")

    return recommendations


def print_analysis_summary(analysis: Dict[str, Any]):
    """Print a formatted summary of the analysis."""

    print("📊 PORTFOLIO PERFORMANCE ANALYSIS")
    print("=" * 50)

    total_tasks = analysis.get('total_tasks', 0)
    lane_wins = analysis.get('lane_wins', {})
    lane_stats = analysis.get('lane_statistics', {})

    print(f"Total tasks analyzed: {total_tasks}")
    print()

    if lane_wins:
        print("🎯 LANE WIN DISTRIBUTION:")
        sorted_wins = sorted(lane_wins.items(), key=lambda x: x[1], reverse=True)
        for lane, wins in sorted_wins:
            pct = (wins / total_tasks * 100) if total_tasks > 0 else 0
            print(f"  {lane}: {wins} wins ({pct:.1f}%)")
        print()

    if lane_stats:
        print("📈 LANE PERFORMANCE STATISTICS:")
        print("┌─────────────┬─────────┬──────────┬──────────┬──────────┬─────────┐")
        print("│ Lane        │ Win %   │ Success% │ Med Time │ Med Nodes│ Samples │")
        print("├─────────────┼─────────┼──────────┼──────────┼──────────┼─────────┤")

        for lane, stats in sorted(lane_stats.items(),
                                 key=lambda x: x[1]['win_rate'], reverse=True):
            win_pct = stats['win_rate'] * 100
            success_pct = stats['success_rate'] * 100
            med_time = stats['median_time']
            med_nodes = stats['median_nodes']
            samples = stats['sample_count']

            print(f"│ {lane:<11} │ {win_pct:>7.1f} │ {success_pct:>8.1f} │ {med_time:>8.3f} │ {med_nodes:>8.0f} │ {samples:>7} │")

        print("└─────────────┴─────────┴──────────┴──────────┴──────────┴─────────┘")
        print()

    # Print recommendations
    recommendations = generate_budget_recommendations(analysis)
    for rec in recommendations:
        print(rec)


def main():
    parser = argparse.ArgumentParser(description="Analyze portfolio performance metrics")
    parser.add_argument("--input", "-i", required=True,
                       help="Input JSONL file with portfolio results")
    parser.add_argument("--output", "-o",
                       help="Output file for detailed analysis (optional)")

    args = parser.parse_args()

    # Load and analyze results
    print(f"Loading results from {args.input}...")
    results = load_results(args.input)

    if not results:
        print("No results found in input file")
        return 1

    print(f"Analyzing {len(results)} results...")
    analysis = analyze_lane_performance(results)

    # Print summary
    print_analysis_summary(analysis)

    # Save detailed analysis if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Detailed analysis saved to {args.output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
