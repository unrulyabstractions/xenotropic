"""
Simple example demonstrating the xenotechnics framework.

This example shows how to:
1. Define structures and systems
2. Compute cores and orientations
3. Detect homogenization
4. Apply xeno-reproduction
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/unrulyabstractions/work/xeno-dynamics')

import xenotechnics as xeno


def main():
    print("=" * 80)
    print("Xenotechnics: Structure-aware Diversity Pursuit")
    print("=" * 80)
    print()

    # ===================================================================
    # 1. Create some example trajectories
    # ===================================================================
    print("1. Creating example trajectories...")
    print()

    trajectories = [
        xeno.create_trajectory("hello world"),
        xeno.create_trajectory("hello there"),
        xeno.create_trajectory("hi"),
        xeno.create_trajectory("greetings everyone"),
        xeno.create_trajectory("hey"),
    ]

    for i, traj in enumerate(trajectories):
        print(f"   Trajectory {i}: {traj}")
    print()

    # ===================================================================
    # 2. Define structures
    # ===================================================================
    print("2. Defining structures (types of organization)...")
    print()

    structures = [
        xeno.LengthStructure(min_length=5, max_length=15, name="moderate_length"),
        xeno.EntropyStructure(name="token_diversity"),
        xeno.PatternStructure(pattern=r"[aeiou]", count_mode=True, max_count=5, name="vowels"),
        xeno.RepetitionStructure(window_size=2, name="low_repetition"),
    ]

    for struct in structures:
        print(f"   {struct.name}: {struct.description}")
    print()

    # ===================================================================
    # 3. Create a system (collection of structures)
    # ===================================================================
    print("3. Creating system from structures...")
    print()

    system = xeno.System(structures)
    print(f"   System: {system}")
    print()

    # ===================================================================
    # 4. Compute compliance for each trajectory
    # ===================================================================
    print("4. Computing system compliance for each trajectory...")
    print()

    for i, traj in enumerate(trajectories):
        compliance = system.compliance(traj)
        print(f"   Trajectory {i}: {compliance}")
        for j, (name, score) in enumerate(zip(system.structure_names(), compliance)):
            print(f"      - {name}: {score:.3f}")
        print()

    # ===================================================================
    # 5. Simulate a probability distribution
    # ===================================================================
    print("5. Simulating probability distributions...")
    print()

    # Baseline: uniform distribution
    baseline_probs = np.ones(len(trajectories)) / len(trajectories)
    print(f"   Baseline (uniform): {baseline_probs}")

    # Homogenized: concentrated on one trajectory
    homogenized_probs = np.array([0.7, 0.15, 0.1, 0.03, 0.02])
    print(f"   Homogenized: {homogenized_probs}")
    print()

    # ===================================================================
    # 6. Compute cores and homogenization metrics
    # ===================================================================
    print("6. Computing system cores and homogenization metrics...")
    print()

    # Baseline core
    baseline_core = xeno.system_core(system, trajectories, baseline_probs)
    print(f"   Baseline core: {baseline_core}")

    # Homogenized core
    homogenized_core = xeno.system_core(system, trajectories, homogenized_probs)
    print(f"   Homogenized core: {homogenized_core}")
    print()

    # Metrics
    baseline_metrics = xeno.compute_homogenization_metrics(
        system, trajectories, baseline_probs
    )
    print("   Baseline metrics:")
    print(f"      E[∂_n] = {baseline_metrics.expected_deviance:.4f}")
    print(f"      Var[∂_n] = {baseline_metrics.deviance_variance:.4f}")
    print(f"      H(⟨Λ_n⟩) = {baseline_metrics.core_entropy:.4f}")
    print(f"      Homogenization score = {baseline_metrics.homogenization_score():.4f}")
    print()

    homogenized_metrics = xeno.compute_homogenization_metrics(
        system, trajectories, homogenized_probs
    )
    print("   Homogenized metrics:")
    print(f"      E[∂_n] = {homogenized_metrics.expected_deviance:.4f}")
    print(f"      Var[∂_n] = {homogenized_metrics.deviance_variance:.4f}")
    print(f"      H(⟨Λ_n⟩) = {homogenized_metrics.core_entropy:.4f}")
    print(f"      Homogenization score = {homogenized_metrics.homogenization_score():.4f}")
    print(f"      Is homogenized? {homogenized_metrics.is_homogenized()}")
    print()

    # ===================================================================
    # 7. Compute orientations and deviances
    # ===================================================================
    print("7. Computing orientations and deviances...")
    print()

    for i, traj in enumerate(trajectories):
        orient = xeno.orientation(traj, system, baseline_core)
        dev = xeno.deviance(orient)
        print(f"   Trajectory {i}:")
        print(f"      Orientation: {orient}")
        print(f"      Deviance: {dev:.4f}")
    print()

    # ===================================================================
    # 8. Xeno-reproduction: score an intervention
    # ===================================================================
    print("8. Scoring xeno-reproduction intervention...")
    print()

    # Compare baseline vs diverse intervention
    # Let's create a more diverse distribution
    diverse_probs = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    diverse_core = xeno.system_core(system, trajectories, diverse_probs)

    intervention_scores = xeno.score_intervention(
        system=system,
        baseline_core=baseline_core,
        baseline_trajectories=trajectories,
        baseline_probs=baseline_probs,
        intervention_core=diverse_core,
        intervention_trajectories=trajectories,
        intervention_probs=diverse_probs,
        lambda_d=1.0,
        lambda_f=0.5,
        lambda_c=0.0,
    )

    print(f"   Diversity score: {intervention_scores.diversity_score:.4f}")
    print(f"   Fairness score: {intervention_scores.fairness_score:.4f}")
    print(f"   Constraint score: {intervention_scores.constraint_score:.4f}")
    print(f"   Total intervention score: {intervention_scores.intervention_score():.4f}")
    print()

    # ===================================================================
    # 9. Compare baseline vs homogenized
    # ===================================================================
    print("9. Comparing baseline vs homogenized distributions...")
    print()

    comparison = xeno.compare_homogenization(baseline_metrics, homogenized_metrics)
    for key, value in comparison.items():
        print(f"   {key}: {value:.4f}")
    print()

    # ===================================================================
    # 10. Diagnose mode collapse
    # ===================================================================
    print("10. Diagnosing potential mode collapse...")
    print()

    diagnosis = xeno.diagnose_mode_collapse(
        homogenized_metrics,
        structure_names=system.structure_names()
    )
    print(f"   Is homogenized? {diagnosis['is_homogenized']}")
    print(f"   Dominant structures: {diagnosis.get('dominant_structures', [])}")
    if diagnosis['warnings']:
        print("   Warnings:")
        for warning in diagnosis['warnings']:
            print(f"      - {warning}")
    print()

    print("=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
