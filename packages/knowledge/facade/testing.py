"""Testing capability mixin for RepositoryFacade.

Owns test-to-code intelligence: discovered test cases, tests-for-symbol and
symbols-for-test relationships, test gap analysis, and test impact analysis.
Implementation detail, not exposed to agents.
"""

from typing import List, Dict, Any

from sqlalchemy import select

from packages.knowledge.models import TestCaseModel, TestToCodeRelationModel


class TestCapabilities:
    """Test-to-code intelligence for RepositoryFacade."""

    async def get_test_cases(self, repo_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Return discovered test cases."""
        stmt = select(TestCaseModel).where(TestCaseModel.repo_id == repo_id).limit(limit)
        res = await self.session.execute(stmt)
        rows = res.scalars().all()
        return [
            {
                "test_name": r.test_name,
                "file_path": r.file_path,
                "test_type": r.test_type,
                "framework": r.framework,
            }
            for r in rows
        ]

    async def get_tests_for_symbol(self, repo_id: str, symbol_name: str) -> List[Dict[str, Any]]:
        """Return tests related to a production symbol."""
        result = []
        stmt = select(TestToCodeRelationModel).where(
            TestToCodeRelationModel.repo_id == repo_id,
            TestToCodeRelationModel.target_symbol == symbol_name,
        )
        res = await self.session.execute(stmt)
        for r in res.scalars().all():
            result.append(
                {
                    "test_name": r.test_name,
                    "test_file": r.test_file,
                    "relation_type": r.relation_type,
                    "confidence": r.confidence,
                }
            )

        if not result:
            fuzzy = select(TestToCodeRelationModel).where(TestToCodeRelationModel.repo_id == repo_id)
            for r in (await self.session.execute(fuzzy)).scalars().all():
                if symbol_name.lower() in r.target_symbol.lower() or symbol_name.lower() in r.test_name.lower():
                    result.append(
                        {
                            "test_name": r.test_name,
                            "test_file": r.test_file,
                            "relation_type": r.relation_type,
                            "confidence": r.confidence,
                        }
                    )

        seen = set()
        unique = []
        for t in result:
            key = (t["test_name"], t["test_file"])
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    async def get_symbols_for_test(self, repo_id: str, test_name: str) -> List[Dict[str, Any]]:
        """Return production symbols related to a test."""
        stmt = select(TestToCodeRelationModel).where(
            TestToCodeRelationModel.repo_id == repo_id,
            TestToCodeRelationModel.test_name == test_name,
        )
        res = await self.session.execute(stmt)
        return [
            {
                "target_symbol": r.target_symbol,
                "target_file": r.target_file,
                "relation_type": r.relation_type,
                "confidence": r.confidence,
            }
            for r in res.scalars().all()
        ]

    async def get_test_gap_analysis(self, repo_id: str, symbol: str) -> Dict[str, Any]:
        """Determine whether a changed symbol lacks related tests."""
        tests = await self.get_tests_for_symbol(repo_id, symbol)
        if tests:
            return {
                "target": symbol,
                "has_tests": True,
                "test_count": len(tests),
                "message": f"{len(tests)} related tests identified by static analysis",
                "confidence": "HIGH",
            }
        return {
            "target": symbol,
            "has_tests": False,
            "test_count": 0,
            "message": "No relevant tests were identified by the current static analysis.",
            "confidence": "MEDIUM",
        }

    async def analyze_test_impact(self, repo_id: str, symbol: str) -> Dict[str, Any]:
        """Return directly related, indirectly related, and unaffected tests."""
        direct = await self.get_tests_for_symbol(repo_id, symbol)
        direct_names = set(t["test_name"] for t in direct)

        dependents = await self.get_dependents(repo_id, symbol)
        indirect = []
        for dep in dependents:
            dep_name = dep.get("source_symbol", "")
            if dep_name:
                tests_for_dep = await self.get_tests_for_symbol(repo_id, dep_name)
                for t in tests_for_dep:
                    if t["test_name"] not in direct_names:
                        indirect.append(t)

        seen = set()
        unique_indirect = []
        for t in indirect:
            key = (t["test_name"], t["test_file"])
            if key not in seen:
                seen.add(key)
                unique_indirect.append(t)

        return {
            "target": symbol,
            "directly_related_tests": direct,
            "indirectly_related_tests": unique_indirect,
            "likely_unaffected_tests": [],
        }