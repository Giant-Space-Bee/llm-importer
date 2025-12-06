#!/usr/bin/env python3
"""
Demo script for Stage 8 Deduplicator testing.

Usage:
    python demo_dedup.py              # All facts
    python demo_dedup.py --limit 100  # First 100 facts
    python demo_dedup.py --limit 150  # First 150 facts
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.aggregator import aggregate, group_by_category, AggregatedFact
from src.deduplicator import (
    deduplicate,
    deduplicate_category,
    deduplicate_merge_sort,
    dedup_batch,
    DeduplicatedFact,
    DEFAULT_MAX_BATCH,
)
from src.providers import LocalProvider


# Monkey-patch for verbose LLM call tracking
_llm_calls = []


def load_checkpoint(path: str) -> list:
    """Load verified facts from checkpoint."""
    with open(path) as f:
        data = json.load(f)
    return data.get("verified_facts", [])


def print_category_stats(title: str, by_category: dict):
    """Print category breakdown."""
    print(f"\n{title}:")
    total = 0
    for cat in sorted(by_category.keys()):
        count = len(by_category[cat])
        total += count
        print(f"  {cat}: {count}")
    print(f"  TOTAL: {total}")


def verbose_dedup_batch(facts, provider, batch_num=None):
    """Wrapper around dedup_batch with timing and logging."""
    global _llm_calls

    batch_label = f"Batch {batch_num}" if batch_num else "Batch"
    fact_type = "AggregatedFact" if facts and isinstance(facts[0], AggregatedFact) else "DeduplicatedFact"

    print(f"    [{batch_label}] Sending {len(facts)} {fact_type}s to LLM...")

    start = time.time()
    result = dedup_batch(facts, provider)
    elapsed = time.time() - start

    reduction = len(facts) - len(result)
    reduction_pct = (reduction / len(facts) * 100) if facts else 0

    call_info = {
        "batch": batch_num,
        "input": len(facts),
        "output": len(result),
        "reduction": reduction,
        "reduction_pct": reduction_pct,
        "time": elapsed,
    }
    _llm_calls.append(call_info)

    print(f"    [{batch_label}] Result: {len(facts)} -> {len(result)} ({reduction} removed, {reduction_pct:.1f}%) in {elapsed:.1f}s")

    return result


def verbose_deduplicate_category(facts, category, provider, max_batch):
    """Verbose version of deduplicate_category."""
    if not facts:
        return []

    print(f"\n  Category: {category} ({len(facts)} facts)")

    # Sort by timestamp (oldest first)
    sorted_facts = sorted(facts, key=lambda f: f.fact.source_timestamp)

    # Show timestamp range
    oldest_ts = sorted_facts[0].fact.source_timestamp
    newest_ts = sorted_facts[-1].fact.source_timestamp
    oldest_date = datetime.fromtimestamp(oldest_ts).strftime("%Y-%m-%d")
    newest_date = datetime.fromtimestamp(newest_ts).strftime("%Y-%m-%d")
    print(f"    Timestamp range: {oldest_date} to {newest_date}")

    # Single batch case
    if len(sorted_facts) <= max_batch:
        print(f"    Single batch (fits in {max_batch})")
        return verbose_dedup_batch(sorted_facts, provider, batch_num=1)

    # Multiple batches - split and process
    num_batches = (len(sorted_facts) + max_batch - 1) // max_batch
    print(f"    Splitting into {num_batches} batches of max {max_batch}")

    batches = []
    for i in range(0, len(sorted_facts), max_batch):
        batches.append(sorted_facts[i:i + max_batch])

    # Dedup each batch
    batch_results = []
    for i, batch in enumerate(batches, 1):
        result = verbose_dedup_batch(batch, provider, batch_num=i)
        batch_results.append(result)

    # Merge-sort pairwise
    print(f"    Merge-sort phase ({len(batch_results)} results to merge)...")
    merge_round = 1
    while len(batch_results) > 1:
        print(f"      Merge round {merge_round}:")
        new_results = []
        for i in range(0, len(batch_results), 2):
            if i + 1 < len(batch_results):
                combined = batch_results[i] + batch_results[i + 1]
                print(f"        Merging results {i+1} + {i+2}: {len(batch_results[i])} + {len(batch_results[i+1])} = {len(combined)}")
                if len(combined) <= max_batch:
                    merged = verbose_dedup_batch(combined, provider, batch_num=f"merge-{merge_round}")
                else:
                    # Recursive case - shouldn't happen often
                    print(f"        Combined too large ({len(combined)}), recursive merge...")
                    merged = verbose_deduplicate_merge_sort(combined, provider, max_batch, depth=1)
                new_results.append(merged)
            else:
                print(f"        Carrying forward result {i+1}: {len(batch_results[i])} facts")
                new_results.append(batch_results[i])
        batch_results = new_results
        merge_round += 1

    final_count = len(batch_results[0]) if batch_results else 0
    print(f"    Category {category} complete: {len(facts)} -> {final_count}")

    return batch_results[0] if batch_results else []


def verbose_deduplicate_merge_sort(facts, provider, max_batch, depth=0):
    """Verbose version of cross-category merge-sort."""
    indent = "      " + "  " * depth

    if not facts:
        return []

    if len(facts) <= max_batch:
        print(f"{indent}Base case: {len(facts)} facts fit in batch")
        return verbose_dedup_batch(facts, provider, batch_num=f"cross-d{depth}")

    # Split in half
    mid = len(facts) // 2
    left = facts[:mid]
    right = facts[mid:]

    print(f"{indent}Split: {len(facts)} -> {len(left)} + {len(right)}")

    # Recursive dedup
    print(f"{indent}Left half:")
    left_deduped = verbose_deduplicate_merge_sort(left, provider, max_batch, depth + 1)

    print(f"{indent}Right half:")
    right_deduped = verbose_deduplicate_merge_sort(right, provider, max_batch, depth + 1)

    # Merge
    combined = left_deduped + right_deduped
    print(f"{indent}Merging: {len(left_deduped)} + {len(right_deduped)} = {len(combined)}")

    if len(combined) <= max_batch:
        return verbose_dedup_batch(combined, provider, batch_num=f"cross-merge-d{depth}")

    # Sliding window for pathological cases
    print(f"{indent}Combined still large ({len(combined)}), sliding window...")
    result = combined
    window_round = 1
    while len(result) > max_batch:
        to_process = result[:max_batch]
        remaining = result[max_batch:]
        deduped = verbose_dedup_batch(to_process, provider, batch_num=f"window-{window_round}")
        result = deduped + remaining

        if len(result) >= len(combined):
            print(f"{indent}No further reduction possible")
            break
        combined = result
        window_round += 1

    return result


def verbose_deduplicate(facts, provider, max_batch):
    """Full verbose deduplication with both phases."""
    global _llm_calls
    _llm_calls = []  # Reset

    if not facts:
        return []

    print("\n" + "=" * 60)
    print("PHASE 1: Within-Category Deduplication")
    print("=" * 60)

    # Group by category
    by_category = group_by_category(facts)
    print(f"\nProcessing {len(by_category)} categories...")

    phase1_start = time.time()
    phase1_results = []

    for category in sorted(by_category.keys()):
        category_facts = by_category[category]
        deduped = verbose_deduplicate_category(category_facts, category, provider, max_batch)
        phase1_results.extend(deduped)

    phase1_elapsed = time.time() - phase1_start

    print(f"\n--- Phase 1 Summary ---")
    print(f"Input:  {len(facts)} aggregated facts")
    print(f"Output: {len(phase1_results)} after within-category dedup")
    print(f"Time:   {phase1_elapsed:.1f}s")

    # Phase 2: Cross-category
    print("\n" + "=" * 60)
    print("PHASE 2: Cross-Category Merge-Sort")
    print("=" * 60)

    phase2_start = time.time()

    if len(phase1_results) <= max_batch:
        print(f"\nAll {len(phase1_results)} facts fit in single batch")
        final = verbose_dedup_batch(phase1_results, provider, batch_num="final-cross")
    else:
        print(f"\n{len(phase1_results)} facts require merge-sort (max_batch={max_batch})")
        final = verbose_deduplicate_merge_sort(phase1_results, provider, max_batch)

    phase2_elapsed = time.time() - phase2_start

    print(f"\n--- Phase 2 Summary ---")
    print(f"Input:  {len(phase1_results)} facts from Phase 1")
    print(f"Output: {len(final)} final deduplicated facts")
    print(f"Time:   {phase2_elapsed:.1f}s")

    return final


def main():
    global _llm_calls

    parser = argparse.ArgumentParser(description="Test Stage 8 Deduplicator")
    parser.add_argument("--limit", type=int, help="Limit facts to test with")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/checkpoint_ea8a3f619a38.json",
        help="Checkpoint file to load",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Max facts per LLM batch (default: 50)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("STAGE 8 DEDUPLICATOR - ENGINEERING TEST")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Batch size: {args.batch_size}")

    # Load facts
    print(f"\n--- LOADING DATA ---")
    print(f"Checkpoint: {args.checkpoint}")
    verified_facts = load_checkpoint(args.checkpoint)
    print(f"Loaded {len(verified_facts)} verified facts")

    # Apply limit if specified
    if args.limit:
        verified_facts = verified_facts[: args.limit]
        print(f"Limited to {len(verified_facts)} facts for testing")

    # Show sample facts
    print(f"\nSample input facts:")
    for f in verified_facts[:3]:
        print(f"  [{f['category']}] {f['fact'][:60]}...")

    # Aggregate
    print("\n--- AGGREGATION ---")
    agg_start = time.time()
    aggregated = aggregate(verified_facts)
    agg_elapsed = time.time() - agg_start

    print(f"Verified facts: {len(verified_facts)}")
    print(f"After aggregation: {len(aggregated)} unique facts")
    print(f"Duplicates removed: {len(verified_facts) - len(aggregated)}")
    print(f"Time: {agg_elapsed:.3f}s")

    # Show category breakdown
    by_cat = group_by_category(aggregated)
    print_category_stats("Input by category", by_cat)

    # Show frequency distribution
    freqs = [f.frequency for f in aggregated]
    freq_1 = sum(1 for f in freqs if f == 1)
    freq_2 = sum(1 for f in freqs if f == 2)
    freq_3plus = sum(1 for f in freqs if f >= 3)
    print(f"\nFrequency distribution:")
    print(f"  Appeared once: {freq_1}")
    print(f"  Appeared twice: {freq_2}")
    print(f"  Appeared 3+ times: {freq_3plus}")

    # Initialize provider
    print("\n--- LLM PROVIDER ---")
    print("Connecting to LM Studio (localhost:1234)...")
    provider = LocalProvider()

    # Test connection
    try:
        import httpx
        resp = httpx.get("http://127.0.0.1:1234/v1/models", timeout=5)
        models = resp.json().get("data", [])
        print(f"Connected! Available models: {len(models)}")
        if models:
            print(f"  Active: {models[0].get('id', 'unknown')}")
    except Exception as e:
        print(f"Warning: Could not verify connection: {e}")

    # Run deduplication
    total_start = time.time()
    deduped = verbose_deduplicate(aggregated, provider, max_batch=args.batch_size)
    total_elapsed = time.time() - total_start

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"\nOverall metrics:")
    print(f"  Verified facts:     {len(verified_facts)}")
    print(f"  After aggregation:  {len(aggregated)}")
    print(f"  After dedup:        {len(deduped)}")
    overall_reduction = (1 - len(deduped) / len(verified_facts)) * 100 if verified_facts else 0
    dedup_reduction = (1 - len(deduped) / len(aggregated)) * 100 if aggregated else 0
    print(f"  Overall reduction:  {overall_reduction:.1f}% (from verified)")
    print(f"  Dedup reduction:    {dedup_reduction:.1f}% (from aggregated)")
    print(f"  Total time:         {total_elapsed:.1f}s")

    # LLM call stats
    print(f"\nLLM call statistics:")
    print(f"  Total calls:        {len(_llm_calls)}")
    total_input = sum(c['input'] for c in _llm_calls)
    total_output = sum(c['output'] for c in _llm_calls)
    total_llm_time = sum(c['time'] for c in _llm_calls)
    print(f"  Total facts processed: {total_input}")
    print(f"  Total facts output:    {total_output}")
    print(f"  Total LLM time:        {total_llm_time:.1f}s")
    print(f"  Avg time per call:     {total_llm_time / len(_llm_calls):.1f}s" if _llm_calls else "  N/A")
    print(f"  Avg facts per call:    {total_input / len(_llm_calls):.1f}" if _llm_calls else "  N/A")

    # Per-call breakdown
    print(f"\nPer-call breakdown:")
    for i, call in enumerate(_llm_calls, 1):
        print(f"  {i:2d}. {call['input']:3d} -> {call['output']:3d} ({call['reduction']:+3d}, {call['reduction_pct']:5.1f}%) in {call['time']:5.1f}s")

    # Category breakdown of output
    dedup_by_cat = {}
    for f in deduped:
        dedup_by_cat.setdefault(f.category, []).append(f)
    print_category_stats("\nOutput by category", dedup_by_cat)

    # Category reduction comparison
    print(f"\nCategory reduction comparison:")
    print(f"  {'Category':<15} {'Input':>6} {'Output':>6} {'Reduced':>8} {'%':>6}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for cat in sorted(by_cat.keys()):
        inp = len(by_cat.get(cat, []))
        out = len(dedup_by_cat.get(cat, []))
        red = inp - out
        pct = (red / inp * 100) if inp else 0
        print(f"  {cat:<15} {inp:>6} {out:>6} {red:>8} {pct:>5.1f}%")

    # Sample output facts
    print("\n--- SAMPLE OUTPUT FACTS ---")
    for cat in sorted(dedup_by_cat.keys()):
        print(f"\n{cat}:")
        for f in dedup_by_cat[cat][:3]:
            print(f"  - {f.fact[:80]}{'...' if len(f.fact) > 80 else ''}")
        if len(dedup_by_cat[cat]) > 3:
            print(f"  ... and {len(dedup_by_cat[cat]) - 3} more")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
