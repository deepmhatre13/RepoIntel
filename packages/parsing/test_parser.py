import os
import re
from dataclasses import dataclass, field
from typing import List, Any
from packages.parsing.decoding import decode_file_bytes

@dataclass
class DiscoveredTestCase:
    file_path: str
    test_name: str
    imported_modules: List[str] = field(default_factory=list)
    assert_targets: List[str] = field(default_factory=list)

@dataclass
class TestToCodeMapping:
    test_name: str
    test_file: str
    target_symbol: str
    target_file: str
    relation_type: str = "IMPORTS" # IMPORTS, CALLS, FIXTURES, ASSERTS
    confidence: str = "MEDIUM"

def discover_tests_in_repo(repo_path: str, filtered_files: List[str]) -> List[DiscoveredTestCase]:
    test_cases = []
    test_patterns = [r"test_.*\.py$", r".*_test\.py$", r".*\.test\.(js|ts|jsx|tsx)$", r".*\.spec\.(js|ts|jsx|tsx)$"]
    
    for rel_path in filtered_files:
        is_test_file = any(re.search(pat, rel_path, re.IGNORECASE) for pat in test_patterns) or "tests/" in rel_path.replace("\\", "/").lower() or "test/" in rel_path.replace("\\", "/").lower()
        if not is_test_file:
            continue

        full_path = os.path.join(repo_path, rel_path)
        if not os.path.isfile(full_path):
            continue

        try:
            with open(full_path, "rb") as f:
                raw_bytes = f.read()
            dec_res = decode_file_bytes(raw_bytes, rel_path)
            if not dec_res.success:
                continue
            content = dec_res.content
        except Exception:
            continue

        # Extract test functions
        func_matches = re.findall(r"def (test_\w+)\s*\(", content)
        if not func_matches:
            func_matches = re.findall(r"(?:it|test)\s*\(\s*['\"]([^'\"]+)['\"]", content)

        # Extract imports
        import_matches = re.findall(r"(?:from|import)\s+([\w\.]+)", content)
        
        # Extract assert targets
        assert_matches = re.findall(r"assert\s+([\w\.]+)", content)

        for tname in func_matches:
            test_cases.append(DiscoveredTestCase(
                file_path=rel_path,
                test_name=tname,
                imported_modules=import_matches,
                assert_targets=assert_matches
            ))

    return test_cases

def map_tests_to_production_symbols(test_cases: List[DiscoveredTestCase], prod_symbols: List[Any]) -> List[TestToCodeMapping]:
    mappings = []
    symbol_dict = {getattr(s, 'symbol_name', getattr(s, 'name', '')): s for s in prod_symbols}


    for tc in test_cases:
        # Match test name substring (e.g. test_authenticate_user -> authenticate_user)
        clean_tname = tc.test_name.lower().replace("test_", "").replace("_test", "")
        for sname, s in symbol_dict.items():
            if sname.lower() == clean_tname:
                mappings.append(TestToCodeMapping(
                    test_name=tc.test_name,
                    test_file=tc.file_path,
                    target_symbol=sname,
                    target_file=s.file_path,
                    relation_type="CALLS",
                    confidence="HIGH"
                ))
            elif sname in tc.assert_targets:
                mappings.append(TestToCodeMapping(
                    test_name=tc.test_name,
                    test_file=tc.file_path,
                    target_symbol=sname,
                    target_file=s.file_path,
                    relation_type="ASSERTS",
                    confidence="HIGH"
                ))
            elif any(s.file_path.replace(".py", "").replace("/", ".").endswith(imp) for imp in tc.imported_modules):
                mappings.append(TestToCodeMapping(
                    test_name=tc.test_name,
                    test_file=tc.file_path,
                    target_symbol=sname,
                    target_file=s.file_path,
                    relation_type="IMPORTS",
                    confidence="MEDIUM"
                ))

    return mappings
