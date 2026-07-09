"""
Sample generator using factorial design.

Produces samples for all combinations of:
  chain_length × conflict_count × noise_count
"""

import itertools
import random
from typing import Iterator

from src.core.types import (
    Fact, Question, Sample, SampleMetadata,
    RelationType, ConflictType, ConflictPosition, NoiseType,
    ExperimentConfig,
)
from src.generation.structure import ChainBuilder, GroundTruthComputer
from src.generation.conflict import ConflictInjector
from src.generation.renderer import NaturalLanguageRenderer


# Noise entity names (lowercase to distinguish from chain entities)
_NOISE_ENTITY_POOL = [
    "X", "Y", "Z", "W", "V", "U", "T", "S", "R", "Q",
    "P", "O", "N", "M", "L", "K", "J", "I", "H", "G",
]


class SampleGenerator:
    """
    Generates experimental samples using factorial design.

    Usage:
        config = ExperimentConfig.from_yaml("config/experiment.yaml")
        gen = SampleGenerator(config)
        samples = list(gen.generate())
        gen.save(samples, "data/samples/")
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.chain_builder = ChainBuilder(seed=config.seed)
        self.conflict_injector = ConflictInjector()
        self.gt_computer = GroundTruthComputer()
        self.renderer = NaturalLanguageRenderer(seed=config.seed)

        # Master RNG for reproducibility
        self._master_rng = random.Random(config.seed)

        # Sub-RNGs seeded from master (ensures independent streams)
        self._shuffle_rng = random.Random(self._master_rng.randint(0, 2**31 - 1))
        self._noise_rng = random.Random(self._master_rng.randint(0, 2**31 - 1))
        self._sample_rng = random.Random(self._master_rng.randint(0, 2**31 - 1))

    @property
    def total_conditions(self) -> int:
        """Total number of parameter combinations."""
        return (
            len(self.config.chain_lengths)
            * len(self.config.conflict_counts)
            * len(self.config.noise_counts)
        )

    @property
    def total_samples(self) -> int:
        """Total number of samples to generate."""
        return self.total_conditions * self.config.samples_per_condition

    def generate(self) -> Iterator[Sample]:
        """
        Generate all samples according to the factorial design.

        Yields:
            Sample objects, each with rendered natural text.
        """
        entity_offset = 0
        noise_offset = 0
        sample_index = 0
        neg_id_counter = 0

        for chain_len, conflict_cnt, noise_cnt in itertools.product(
            self.config.chain_lengths,
            self.config.conflict_counts,
            self.config.noise_counts,
        ):
            for rep in range(self.config.samples_per_condition):
                # Deterministic seed for this sample
                sample_seed = self._sample_rng.randint(0, 2**31 - 1)
                sample_rng = random.Random(sample_seed)

                # 1. Build chain
                chain_facts, chain_entities = self.chain_builder.build_chain(
                    chain_length=chain_len,
                    entity_offset=entity_offset,
                )

                # 2. Inject conflicts (negation facts)
                if conflict_cnt > 0:
                    negation_facts = self.conflict_injector.inject_transitivity_break(
                        chain_facts=chain_facts,
                        chain_entities=chain_entities,
                        conflict_count=conflict_cnt,
                        position=self.config.fixed_conflict_position,
                        start_id=neg_id_counter,
                    )
                    neg_id_counter += conflict_cnt
                else:
                    negation_facts = []

                # 3. Generate noise facts
                noise_facts = self._generate_noise(
                    count=noise_cnt,
                    chain_entities=chain_entities,
                    noise_rng=sample_rng,
                    noise_offset=noise_offset,
                )
                noise_offset += noise_cnt

                # 4. Compute ground truth
                question = self.chain_builder.make_question(chain_entities)
                gt = self.gt_computer.compute(
                    chain_facts=chain_facts,
                    chain_entities=chain_entities,
                    negation_facts=negation_facts,
                    noise_facts=noise_facts,
                    question=question,
                )

                # Skip unresolvable samples (tie)
                if not gt.is_resolvable:
                    # Still advance offsets and counters to keep determinism
                    entity_offset += len(chain_entities)
                    sample_index += 1
                    continue

                # 5. Render natural language
                all_facts = chain_facts + negation_facts + noise_facts
                # Populate natural_text on all facts
                for f in all_facts:
                    f.natural_text = self.renderer.render_fact(f)
                question.natural_text = self.renderer.render_question(question)

                # 6. Compute MIS
                mis = self.conflict_injector.compute_mis(
                    chain_length=chain_len,
                    conflict_count=conflict_cnt,
                    position=self.config.fixed_conflict_position,
                )

                # 7. Build metadata
                metadata = SampleMetadata(
                    chain_length=chain_len,
                    total_facts=len(all_facts),
                    conflict_type=self.config.fixed_conflict_type,
                    conflict_count=conflict_cnt,
                    conflict_positions=(
                        [self.config.fixed_conflict_position] * conflict_cnt
                    ),
                    mis_size=mis,
                    noise_count=noise_cnt,
                    noise_type=self.config.fixed_noise_type,
                    seed=sample_seed,
                    entity_count=len(chain_entities) + noise_cnt * 2,  # each noise fact has 2 entities
                    supporting_fact_count=gt.supporting_count,
                    opposing_fact_count=gt.opposing_count,
                    gt_vote_ratio=gt.vote_ratio,
                )

                # 8. Build sample
                sample = Sample(
                    id=f"s{sample_index:04d}",
                    facts=all_facts,
                    question=question,
                    ground_truth=gt.answer,  # type: ignore (guaranteed resolvable)
                    ground_truth_rationale=gt.rationale,
                    metadata=metadata,
                )

                yield sample

                entity_offset += len(chain_entities)
                sample_index += 1

    def _generate_noise(
        self,
        count: int,
        chain_entities: list[str],
        noise_rng: random.Random,
        noise_offset: int,
    ) -> list[Fact]:
        """
        Generate irrelevant noise facts.

        Noise facts involve entities NOT in the chain,
        so they cannot affect the transitive reasoning.

        Args:
            count: Number of noise facts to generate.
            chain_entities: Current chain entities (to avoid overlap).
            noise_rng: RNG for this sample.
            noise_offset: Starting index for noise entity selection.

        Returns:
            List of noise Facts.
        """
        if count == 0:
            return []

        # Build a pool of noise entities excluding chain entities
        chain_set = set(chain_entities)
        available = [e for e in _NOISE_ENTITY_POOL if e not in chain_set]

        noise_facts = []
        for i in range(count):
            # Pick two distinct entities from the noise pool
            if len(available) < 2:
                # If we run out, generate synthetic names
                subj = f"E{noise_offset + i*2}"
                obj = f"E{noise_offset + i*2 + 1}"
            else:
                subj, obj = noise_rng.sample(available, 2)

            fid = f"noise{noise_offset + i}"
            noise_facts.append(Fact(
                id=fid,
                subject=subj,
                predicate=RelationType.MEMBER_OF,
                obj=obj,
                is_noise=True,
            ))

        return noise_facts

    def generate_all(self) -> list[Sample]:
        """Generate all samples as a list. Convenience wrapper."""
        return list(self.generate())

    def save(self, samples: list[Sample], output_dir: str = "data/samples") -> str:
        """Save samples to a JSONL file."""
        import json
        import os
        from datetime import datetime

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"samples_{timestamp}.jsonl")

        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "id": sample.id,
                    "facts": [
                        {
                            "id": fact.id,
                            "subject": fact.subject,
                            "predicate": fact.predicate.value,
                            "object": fact.obj,
                            "is_noise": fact.is_noise,
                            "natural_text": fact.natural_text,
                        }
                        for fact in sample.facts
                    ],
                    "question": {
                        "subject": sample.question.subject,
                        "predicate": sample.question.predicate.value,
                        "object": sample.question.obj,
                        "natural_text": sample.question.natural_text,
                    },
                    "ground_truth": sample.ground_truth,
                    "ground_truth_rationale": sample.ground_truth_rationale,
                    "metadata": sample.metadata.to_dict(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved {len(samples)} samples to {path}")
        return path

    def print_summary(self, samples: list[Sample]) -> None:
        """Print a summary of generated samples."""
        from collections import Counter

        print("=" * 60)
        print("Sample Generation Summary")
        print("=" * 60)
        print(f"  Total samples:      {len(samples)}")

        # Per condition breakdown
        cond_counts = Counter()
        for s in samples:
            key = (s.metadata.chain_length, s.metadata.conflict_count, s.metadata.noise_count)
            cond_counts[key] += 1

        print(f"  Unique conditions:  {len(cond_counts)}")
        print(f"  Per condition:      ~{len(samples) // len(cond_counts)} samples")
        print()

        # GT distribution
        gt_true = sum(1 for s in samples if s.ground_truth is True)
        gt_false = sum(1 for s in samples if s.ground_truth is False)
        print(f"  GT=True:  {gt_true} ({gt_true/len(samples)*100:.1f}%)")
        print(f"  GT=False: {gt_false} ({gt_false/len(samples)*100:.1f}%)")
        print()

        # Parameter value distributions
        for param in ["chain_length", "conflict_count", "noise_count"]:
            vals = Counter(getattr(s.metadata, param) for s in samples)
            print(f"  {param}: {dict(sorted(vals.items()))}")
        print("=" * 60)
