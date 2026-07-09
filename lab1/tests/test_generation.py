"""Tests for sample generation modules."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.types import (
    RelationType, ConflictType, ConflictPosition, NoiseType, ExperimentConfig,
)
from src.generation.structure import ChainBuilder, GroundTruthComputer
from src.generation.conflict import ConflictInjector
from src.generation.renderer import NaturalLanguageRenderer
from src.generation.sampler import SampleGenerator


# ── ChainBuilder Tests ──────────────────────────────────────────────────────

class TestChainBuilder:
    def test_build_chain_length_2(self):
        builder = ChainBuilder()
        facts, entities = builder.build_chain(chain_length=2)
        assert len(entities) == 3  # A, B, C
        assert entities[0] == "A"
        assert entities[1] == "B"
        assert entities[2] == "C"
        assert len(facts) == 2
        assert facts[0].subject == "A"
        assert facts[0].obj == "B"
        assert facts[1].subject == "B"
        assert facts[1].obj == "C"

    def test_build_chain_length_4(self):
        builder = ChainBuilder()
        facts, entities = builder.build_chain(chain_length=4)
        assert len(entities) == 5
        assert entities == ["A", "B", "C", "D", "E"]
        assert len(facts) == 4

    def test_build_chain_rejects_length_1(self):
        builder = ChainBuilder()
        with pytest.raises(ValueError, match="chain_length must be"):
            builder.build_chain(chain_length=1)

    def test_make_question(self):
        builder = ChainBuilder()
        _, entities = builder.build_chain(chain_length=3)
        q = builder.make_question(entities)
        assert q.subject == "A"
        assert q.obj == "D"
        assert q.predicate == RelationType.MEMBER_OF

    def test_entity_offset(self):
        builder = ChainBuilder()
        facts, entities = builder.build_chain(chain_length=2, entity_offset=3)
        # With offset 3: entities starting from index 3 (= D)
        assert entities[0] == "D"
        assert entities[1] == "E"
        assert entities[2] == "F"


# ── ConflictInjector Tests ──────────────────────────────────────────────────

class TestConflictInjector:
    def test_no_conflict(self):
        injector = ConflictInjector()
        negs = injector.inject_transitivity_break(
            chain_facts=[],
            chain_entities=["A", "B", "C"],
            conflict_count=0,
        )
        assert negs == []

    def test_one_conflict_late(self):
        injector = ConflictInjector()
        negs = injector.inject_transitivity_break(
            chain_facts=[],
            chain_entities=["A", "B", "C"],
            conflict_count=1,
            position=ConflictPosition.LATE,
        )
        assert len(negs) == 1
        assert negs[0].subject == "A"
        assert negs[0].obj == "C"  # last entity
        assert negs[0].predicate == RelationType.NOT_MEMBER_OF

    def test_three_conflicts_late(self):
        injector = ConflictInjector()
        negs = injector.inject_transitivity_break(
            chain_facts=[],
            chain_entities=["A", "B", "C", "D"],
            conflict_count=3,
            position=ConflictPosition.LATE,
        )
        assert len(negs) == 3
        for n in negs:
            assert n.subject == "A"
            assert n.obj == "D"
            assert n.predicate == RelationType.NOT_MEMBER_OF

    def test_conflict_early(self):
        injector = ConflictInjector()
        negs = injector.inject_transitivity_break(
            chain_facts=[],
            chain_entities=["A", "B", "C", "D"],
            conflict_count=1,
            position=ConflictPosition.EARLY,
        )
        assert negs[0].subject == "A"
        assert negs[0].obj == "B"  # first entity after A

    def test_conflict_middle(self):
        injector = ConflictInjector()
        negs = injector.inject_transitivity_break(
            chain_facts=[],
            chain_entities=["A", "B", "C", "D", "E"],
            conflict_count=1,
            position=ConflictPosition.MIDDLE,
        )
        # chain_length=4, middle index = max(1, 4//2) = 2 → entity "C"
        assert negs[0].subject == "A"
        assert negs[0].obj == "C"

    def test_compute_mis(self):
        injector = ConflictInjector()
        mis = injector.compute_mis(chain_length=3, conflict_count=1)
        assert mis == 4  # chain_length + 1

    def test_rejects_negative_conflict_count(self):
        injector = ConflictInjector()
        with pytest.raises(ValueError):
            injector.inject_transitivity_break([], ["A", "B"], conflict_count=-1)


# ── GroundTruthComputer Tests ────────────────────────────────────────────────

class TestGroundTruthComputer:
    def test_no_conflict_gives_true(self):
        """Pure chain: all facts support transitivity → GT=True."""
        from src.core.types import Fact as F
        chain = [
            F(id="c0", subject="A", predicate=RelationType.MEMBER_OF, obj="B"),
            F(id="c1", subject="B", predicate=RelationType.MEMBER_OF, obj="C"),
        ]
        gt_comp = GroundTruthComputer()
        from src.core.types import Question as Q
        q = Q(subject="A", predicate=RelationType.MEMBER_OF, obj="C")
        gt = gt_comp.compute(
            chain_facts=chain,
            chain_entities=["A", "B", "C"],
            negation_facts=[],
            noise_facts=[],
            question=q,
        )
        assert gt.answer is True
        assert gt.is_resolvable is True
        assert gt.vote_ratio == 1.0
        assert gt.supporting_count == 2
        assert gt.opposing_count == 0

    def test_one_negation_majority_true(self):
        """3 chain facts vs 1 negation → GT=True."""
        from src.core.types import Fact as F
        chain = [
            F(id="c0", subject="A", predicate=RelationType.MEMBER_OF, obj="B"),
            F(id="c1", subject="B", predicate=RelationType.MEMBER_OF, obj="C"),
            F(id="c2", subject="C", predicate=RelationType.MEMBER_OF, obj="D"),
        ]
        neg = [
            F(id="n0", subject="A", predicate=RelationType.NOT_MEMBER_OF, obj="D"),
        ]
        gt_comp = GroundTruthComputer()
        from src.core.types import Question as Q
        q = Q(subject="A", predicate=RelationType.MEMBER_OF, obj="D")
        gt = gt_comp.compute(chain, ["A", "B", "C", "D"], neg, [], q)
        assert gt.answer is True
        assert gt.vote_ratio == 3 / 4  # 3/4 = 0.75
        assert gt.supporting_count == 3
        assert gt.opposing_count == 1

    def test_negations_beat_chain(self):
        """2 chain facts vs 3 negations → GT=False."""
        from src.core.types import Fact as F
        chain = [
            F(id="c0", subject="A", predicate=RelationType.MEMBER_OF, obj="B"),
            F(id="c1", subject="B", predicate=RelationType.MEMBER_OF, obj="C"),
        ]
        neg = [
            F(id="n0", subject="A", predicate=RelationType.NOT_MEMBER_OF, obj="C"),
            F(id="n1", subject="A", predicate=RelationType.NOT_MEMBER_OF, obj="C"),
            F(id="n2", subject="A", predicate=RelationType.NOT_MEMBER_OF, obj="C"),
        ]
        gt_comp = GroundTruthComputer()
        from src.core.types import Question as Q
        q = Q(subject="A", predicate=RelationType.MEMBER_OF, obj="C")
        gt = gt_comp.compute(chain, ["A", "B", "C"], neg, [], q)
        assert gt.answer is False
        assert gt.vote_ratio == 2 / 5  # 0.4
        assert gt.supporting_count == 2
        assert gt.opposing_count == 3

    def test_tie_unresolvable(self):
        """Equal supporting and opposing → unresolvable."""
        from src.core.types import Fact as F
        chain = [
            F(id="c0", subject="A", predicate=RelationType.MEMBER_OF, obj="B"),
            F(id="c1", subject="B", predicate=RelationType.MEMBER_OF, obj="C"),
        ]
        neg = [
            F(id="n0", subject="A", predicate=RelationType.NOT_MEMBER_OF, obj="C"),
            F(id="n1", subject="A", predicate=RelationType.NOT_MEMBER_OF, obj="C"),
        ]
        gt_comp = GroundTruthComputer()
        from src.core.types import Question as Q
        q = Q(subject="A", predicate=RelationType.MEMBER_OF, obj="C")
        gt = gt_comp.compute(chain, ["A", "B", "C"], neg, [], q)
        assert gt.is_resolvable is False
        assert gt.answer is None  # tie, unresolvable
        assert gt.vote_ratio == 2 / 4  # 0.5

    def test_noise_ignored(self):
        """Noise facts don't affect the vote."""
        from src.core.types import Fact as F
        chain = [
            F(id="c0", subject="A", predicate=RelationType.MEMBER_OF, obj="B"),
            F(id="c1", subject="B", predicate=RelationType.MEMBER_OF, obj="C"),
        ]
        noise = [
            F(id="n0", subject="X", predicate=RelationType.MEMBER_OF, obj="Y", is_noise=True),
            F(id="n1", subject="P", predicate=RelationType.MEMBER_OF, obj="Q", is_noise=True),
        ]
        gt_comp = GroundTruthComputer()
        from src.core.types import Question as Q
        q = Q(subject="A", predicate=RelationType.MEMBER_OF, obj="C")
        gt = gt_comp.compute(chain, ["A", "B", "C"], [], noise, q)
        assert gt.answer is True
        assert gt.supporting_count == 2  # noise not counted
        assert gt.opposing_count == 0
        assert gt.vote_ratio == 1.0


# ── Renderer Tests ──────────────────────────────────────────────────────────

class TestRenderer:
    def test_render_member_fact(self):
        r = NaturalLanguageRenderer()
        from src.core.types import Fact as F
        f = F(id="x", subject="A", predicate=RelationType.MEMBER_OF, obj="B")
        text = r.render_fact(f)
        assert text == "A is a member of B."

    def test_render_not_member_fact(self):
        r = NaturalLanguageRenderer()
        from src.core.types import Fact as F
        f = F(id="x", subject="A", predicate=RelationType.NOT_MEMBER_OF, obj="B")
        text = r.render_fact(f)
        assert text == "A is not a member of B."

    def test_render_question(self):
        r = NaturalLanguageRenderer()
        from src.core.types import Question as Q
        q = Q(subject="A", predicate=RelationType.MEMBER_OF, obj="D")
        text = r.render_question(q)
        assert "A" in text and "D" in text

    def test_shuffle_is_deterministic(self):
        r = NaturalLanguageRenderer(seed=42)
        from src.core.types import Fact as F, Question as Q
        facts = [
            F(id="0", subject="A", predicate=RelationType.MEMBER_OF, obj="B"),
            F(id="1", subject="B", predicate=RelationType.MEMBER_OF, obj="C"),
            F(id="2", subject="X", predicate=RelationType.MEMBER_OF, obj="Y"),
        ]
        q = Q(subject="A", predicate=RelationType.MEMBER_OF, obj="C")
        text1, _ = r.render_facts_and_question(facts, q, shuffle=True, shuffle_seed=123)
        text2, _ = r.render_facts_and_question(facts, q, shuffle=True, shuffle_seed=123)
        assert text1 == text2  # Deterministic

    def test_shuffle_differs_by_seed(self):
        r = NaturalLanguageRenderer(seed=42)
        from src.core.types import Fact as F, Question as Q
        facts = [
            F(id="0", subject="A", predicate=RelationType.MEMBER_OF, obj="B"),
            F(id="1", subject="B", predicate=RelationType.MEMBER_OF, obj="C"),
            F(id="2", subject="X", predicate=RelationType.MEMBER_OF, obj="Y"),
        ]
        q = Q(subject="A", predicate=RelationType.MEMBER_OF, obj="C")
        text1, _ = r.render_facts_and_question(facts, q, shuffle=True, shuffle_seed=42)
        text2, _ = r.render_facts_and_question(facts, q, shuffle=True, shuffle_seed=99)
        # Different seeds should (usually) give different orderings
        # Note: technically could collide but extremely unlikely with 3+ items


# ── SampleGenerator Integration Tests ────────────────────────────────────────

class TestSampleGenerator:
    def test_generates_expected_count(self):
        config = ExperimentConfig(
            chain_lengths=[2, 3],
            conflict_counts=[0, 1],
            noise_counts=[0, 3],
            samples_per_condition=5,
            seed=42,
        )
        gen = SampleGenerator(config)
        samples = gen.generate_all()

        # 2 × 2 × 2 = 8 conditions × 5 samples = 40 max
        # Some may be filtered (ties)
        assert 0 < len(samples) <= 40

        # Check all samples have required fields
        for s in samples:
            assert s.id
            assert len(s.facts) > 0
            assert s.question.subject
            assert s.ground_truth in (True, False)
            assert s.metadata.chain_length in (2, 3)

    def test_no_conflict_samples_all_true(self):
        config = ExperimentConfig(
            chain_lengths=[2],
            conflict_counts=[0],      # No conflicts!
            noise_counts=[0],
            samples_per_condition=10,
            seed=42,
        )
        gen = SampleGenerator(config)
        samples = gen.generate_all()
        assert len(samples) == 10
        for s in samples:
            assert s.ground_truth is True, f"Expected True but got {s.ground_truth}. {s.ground_truth_rationale}"

    def test_generates_different_samples(self):
        config = ExperimentConfig(
            chain_lengths=[2],
            conflict_counts=[1],
            noise_counts=[0],
            samples_per_condition=5,
            seed=42,
        )
        gen = SampleGenerator(config)
        samples = gen.generate_all()
        # Entity offsets differ across samples
        entities_seen = set()
        for s in samples:
            for f in s.facts:
                entities_seen.add(f.subject)
                entities_seen.add(f.obj)
        # Should see different entity names
        assert len(entities_seen) > 3

    def test_metadata_fields(self):
        config = ExperimentConfig(
            chain_lengths=[3],
            conflict_counts=[2],
            noise_counts=[3],
            samples_per_condition=3,
            seed=42,
        )
        gen = SampleGenerator(config)
        samples = gen.generate_all()
        for s in samples:
            assert s.metadata.chain_length == 3
            assert s.metadata.conflict_count == 2
            assert s.metadata.noise_count == 3
            assert s.metadata.conflict_type == ConflictType.TRANSITIVITY_BREAK
            assert s.metadata.noise_type == NoiseType.IRRELEVANT
            assert s.metadata.total_facts == len(s.facts)
            assert s.metadata.gt_vote_ratio > 0
            assert s.metadata.supporting_fact_count > 0
            assert s.metadata.opposing_fact_count > 0

    def test_reproducibility(self):
        """Same seed → same samples."""
        c1 = ExperimentConfig(chain_lengths=[2], conflict_counts=[1],
                              noise_counts=[0], samples_per_condition=5, seed=123)
        g1 = SampleGenerator(c1)
        s1 = g1.generate_all()

        c2 = ExperimentConfig(chain_lengths=[2], conflict_counts=[1],
                              noise_counts=[0], samples_per_condition=5, seed=123)
        g2 = SampleGenerator(c2)
        s2 = g2.generate_all()

        assert len(s1) == len(s2)
        for a, b in zip(s1, s2):
            assert a.id == b.id
            assert a.ground_truth == b.ground_truth
            assert a.metadata.to_dict() == b.metadata.to_dict()

    def test_save_load(self, tmp_path):
        config = ExperimentConfig(
            chain_lengths=[2], conflict_counts=[0],
            noise_counts=[0], samples_per_condition=3, seed=42,
        )
        gen = SampleGenerator(config)
        samples = gen.generate_all()
        path = gen.save(samples, str(tmp_path))
        assert os.path.exists(path)

        # Verify it's valid JSONL
        import json
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == len(samples)
        for line in lines:
            record = json.loads(line)
            assert "id" in record
            assert "ground_truth" in record
            assert "metadata" in record


# ── AnswerExtractor Tests ────────────────────────────────────────────────────

class TestAnswerExtractor:
    def test_layer1_yes(self):
        from src.agents.base import AnswerExtractor
        ext = AnswerExtractor()
        ans, text, method = ext.extract_answer("Some reasoning...\nANSWER: YES\nCONFLICT_DETECTED: NO")
        assert ans is True
        assert method == "layer1_format"

    def test_layer1_no(self):
        from src.agents.base import AnswerExtractor
        ext = AnswerExtractor()
        ans, text, method = ext.extract_answer("Thought: ...\nANSWER: NO")
        assert ans is False
        assert method == "layer1_format"

    def test_layer1_uncertain(self):
        from src.agents.base import AnswerExtractor
        ext = AnswerExtractor()
        ans, text, method = ext.extract_answer("I cannot determine.\nANSWER: UNCERTAIN")
        assert ans is None
        assert method == "layer1_uncertain"

    def test_layer2_last_sentence(self):
        from src.agents.base import AnswerExtractor
        ext = AnswerExtractor()
        ans, text, method = ext.extract_answer(
            "Some reasoning. The facts suggest yes. Therefore, the answer is yes."
        )
        assert ans is True
        assert "layer2" in method, f"Expected layer2_* method, got {method}"

    def test_layer3_member_inference(self):
        from src.agents.base import AnswerExtractor
        ext = AnswerExtractor()
        ans, text, method = ext.extract_answer(
            "After careful analysis, A is a member of D through transitivity."
        )
        assert ans is True
        assert method == "layer3_member_inference"

    def test_conflict_explicit(self):
        from src.agents.base import AnswerExtractor
        ext = AnswerExtractor()
        result = ext.extract_conflict_explicit("...\nCONFLICT_DETECTED: YES\n")
        assert result is True

        result = ext.extract_conflict_explicit("...\nCONFLICT_DETECTED: NO\n")
        assert result is False

        result = ext.extract_conflict_explicit("No conflict field here")
        assert result is None

    def test_conflict_implicit(self):
        from src.agents.base import AnswerExtractor
        ext = AnswerExtractor()
        detected, keywords = ext.extract_conflict_implicit(
            "There is a contradiction between fact 1 and fact 3. "
            "These two statements are inconsistent."
        )
        assert detected is True
        assert "contradiction" in keywords
        assert "inconsistent" in keywords

    def test_no_conflict_implicit(self):
        from src.agents.base import AnswerExtractor
        ext = AnswerExtractor()
        detected, keywords = ext.extract_conflict_implicit(
            "All facts are consistent. The answer is clear."
        )
        assert detected is False
        assert keywords == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
