"""Phase 23: AST-Aware Git Diff Intelligence.

Parses git diffs, maps changed line ranges to symbols using AST line numbers,
classifies changes, and detects API/dependency changes.
"""
import subprocess
import re
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from packages.parsing.python_ast import parse_python_file, SymbolNode


@dataclass
class DiffHunk:
    """A single hunk from a unified diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    added_lines: List[Tuple[int, str]] = field(default_factory=list)   # (new_line_no, content)
    removed_lines: List[Tuple[int, str]] = field(default_factory=list) # (old_line_no, content)
    context_lines: List[str] = field(default_factory=list)
    raw_text: str = ""

    @property
    def old_line_range(self) -> Tuple[int, int]:
        return (self.old_start, self.old_start + max(0, self.old_count - 1)) if self.old_count > 0 else (self.old_start, self.old_start)

    @property
    def new_line_range(self) -> Tuple[int, int]:
        return (self.new_start, self.new_start + max(0, self.new_count - 1)) if self.new_count > 0 else (self.new_start, self.new_start)


@dataclass
class FileDiff:
    """A file-level diff from a git commit."""
    old_path: str
    new_path: str
    change_type: str  # ADDED, REMOVED, MODIFIED, RENAMED, MOVED
    hunks: List[DiffHunk] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0


class DiffParser:
    """Parses git unified diff output into structured FileDiff objects."""

    HUNK_HEADER_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
    DIFF_FILE_RE = re.compile(r'^diff --git a/(.+?) b/(.+?)$')

    def parse_diff(self, diff_text: str) -> List[FileDiff]:
        """Parse the full diff text into FileDiff objects."""
        if not diff_text.strip():
            return []

        lines = diff_text.splitlines(keepends=False)
        file_diffs: List[FileDiff] = []
        current_file: Optional[FileDiff] = None
        current_hunk: Optional[DiffHunk] = None
        i = 0

        while i < len(lines):
            line = lines[i]

            # File header
            file_match = self.DIFF_FILE_RE.match(line)
            if file_match:
                if current_file and current_hunk:
                    current_file.hunks.append(current_hunk)
                if current_file:
                    file_diffs.append(current_file)

                old_path = file_match.group(1)
                new_path = file_match.group(2)
                change_type = "MODIFIED"
                if line.endswith(" /dev/null"):
                    change_type = "ADDED"
                current_file = FileDiff(old_path=old_path, new_path=new_path, change_type=change_type)
                current_hunk = None
                i += 1
                continue

            # Handle deleted file
            if line.startswith("deleted file"):
                if current_file:
                    current_file.change_type = "REMOVED"
                i += 1
                continue

            # Handle new file
            if line.startswith("new file"):
                if current_file:
                    current_file.change_type = "ADDED"
                i += 1
                continue

            # Hunk header
            hunk_match = self.HUNK_HEADER_RE.match(line)
            if hunk_match:
                if current_hunk and current_file:
                    current_file.hunks.append(current_hunk)

                old_start = int(hunk_match.group(1))
                old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
                new_start = int(hunk_match.group(3))
                new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1
                current_hunk = DiffHunk(
                    old_start=old_start, old_count=old_count,
                    new_start=new_start, new_count=new_count
                )
                i += 1
                continue

            # Hunk body lines
            if current_hunk is not None:
                if line.startswith("+"):
                    content = line[1:]
                    new_line_no = current_hunk.new_start + current_hunk.new_count - current_hunk.new_count + len([l for l in current_hunk.added_lines]) + len([ctx for ctx in current_hunk.context_lines])
                    # Recalculate line numbers properly
                    current_hunk.added_lines.append((0, content))
                    current_hunk.additions = 0
                    current_hunk.context_lines.append("")
                elif line.startswith("-"):
                    content = line[1:]
                    current_hunk.removed_lines.append((0, content))
                elif line.startswith(" "):
                    current_hunk.context_lines.append(line[1:])
                elif line.startswith("\\"):
                    pass  # "No newline at end of file"
            i += 1

        # Close last hunk and file
        if current_hunk and current_file:
            current_file.hunks.append(current_hunk)
        if current_file:
            file_diffs.append(current_file)

        # Post-process: assign correct line numbers and counts
        for fd in file_diffs:
            fd.additions = sum(len(h.added_lines) for h in fd.hunks)
            fd.deletions = sum(len(h.removed_lines) for h in fd.hunks)
            # Fix line numbers in hunks
            for hunk in fd.hunks:
                self._assign_line_numbers(hunk)

        return file_diffs

    def _assign_line_numbers(self, hunk: DiffHunk):
        """Assign proper old/new line numbers to added/removed lines in a hunk."""
        old_line = hunk.old_start if hunk.old_count > 0 else hunk.old_start
        new_line = hunk.new_start if hunk.new_count > 0 else hunk.new_start
        old_idx = 0
        new_idx = 0
        old_lines = list(hunk.removed_lines)
        new_lines = list(hunk.added_lines)
        ctx = hunk.context_lines

        # Rebuild from raw - we need to re-parse the hunk body
        # Simpler approach: re-calculate from raw_text if available
        # Actually, let me just use the old/new_start counters properly
        # We need the original hunk body lines
        # Since context/added/removed are already split, let me fix line numbers:
        hunk.added_lines = []
        hunk.removed_lines = []
        hunk.context_lines = []
        old_l = hunk.old_start
        new_l = hunk.new_start
        raw_lines = hunk.raw_text.splitlines() if hunk.raw_text else []

        for rl in raw_lines:
            if rl.startswith("+"):
                hunk.added_lines.append((new_l, rl[1:]))
                new_l += 1
            elif rl.startswith("-"):
                hunk.removed_lines.append((old_l, rl[1:]))
                old_l += 1
            elif rl.startswith(" "):
                hunk.context_lines.append(rl[1:])
                old_l += 1
                new_l += 1
            elif rl.startswith("\\"):
                pass


class SymbolChangeMapper:
    """Maps diff hunks to AST symbols based on line numbers."""

    def find_affected_symbols(
        self,
        file_path: str,
        file_content: str,
        hunks: List[DiffHunk],
        language: str = "Python"
    ) -> List[Dict[str, Any]]:
        """Find symbols whose line ranges overlap with changed regions."""
        if language != "Python":
            return []

        result = parse_python_file(file_path, file_content)
        if not result.parse_succeeded:
            return []

        affected = []
        for hunk in hunks:
            old_range = (hunk.old_start, hunk.old_start + max(0, hunk.old_count - 1))
            new_range = (hunk.new_start, hunk.new_start + max(0, hunk.new_count - 1))

            for sym in result.symbols:
                sym_range = (sym.line_start, sym.line_end)
                # Check overlap with new range (most relevant for symbol location)
                if self._ranges_overlap(sym_range, new_range):
                    affected.append({
                        "symbol_name": sym.symbol_name,
                        "symbol_type": sym.symbol_type,
                        "line_start": sym.line_start,
                        "line_end": sym.line_end,
                        "signature": sym.signature,
                        "parent_class": sym.parent_class,
                        "hunk_old_range": old_range,
                        "hunk_new_range": new_range,
                        "overlap": True
                    })

        return affected

    def _ranges_overlap(self, r1: Tuple[int, int], r2: Tuple[int, int]) -> bool:
        if r1[1] == 0 and r2[1] == 0:
            return r1[0] == r2[0]
        return not (r1[1] < r2[0] or r2[1] < r1[0])


class ChangeClassifier:
    """Classifies symbol-level changes deterministically."""

    SIGNATURE_KEYWORDS = ["def ", "async def ", "class "]

    def classify(
        self,
        sym_name: str,
        sym_line_start: int,
        sym_line_end: int,
        hunks: List[DiffHunk]
    ) -> str:
        """Classify the type of change for a symbol based on diff hunks."""
        for hunk in hunks:
            new_range = (hunk.new_start, hunk.new_start + max(0, hunk.new_count - 1))
            old_range = (hunk.old_start, hunk.old_start + max(0, hunk.old_count - 1))

            # Check if the symbol's signature line is in the changed region
            for new_line_no, content in hunk.added_lines:
                if self._line_in_range(new_line_no, sym_line_start, sym_line_end):
                    # Check if this line looks like the definition
                    stripped = content.lstrip()
                    if any(kw in stripped for kw in self.SIGNATURE_KEYWORDS):
                        return "SIGNATURE_CHANGE"

            # If any line in the symbol's range was removed/added, it's a BODY_CHANGE
            for rem_line_no, _ in hunk.removed_lines:
                if sym_line_start <= rem_line_no <= sym_line_end:
                    return "BODY_CHANGE"
            for add_line_no, _ in hunk.added_lines:
                if sym_line_start <= add_line_no <= sym_line_end:
                    return "BODY_CHANGE"

        return "BODY_CHANGE"

    def _line_in_range(self, line_no: int, start: int, end: int) -> bool:
        return start <= line_no <= end


class APIChangeDetector:
    """Detects changes to public API signatures."""

    def detect(
        self,
        old_content: str,
        new_content: str,
        changed_symbols: List[Dict[str, Any]],
        repo_id: str,
        commit_sha: str
    ) -> List[Dict[str, Any]]:
        """Detect API changes by comparing signatures before/after."""
        changes = []

        for sym in changed_symbols:
            sig = sym.get("signature", "")
            # Only consider public symbols (not starting with _)
            sym_name = sym.get("symbol_name", "")
            if sym_name.startswith("_"):
                continue

            # Check if the diff includes the symbol definition line
            is_api_change = False
            for hunk in []:  # We'll pass hunks separately
                pass

            # Heuristic: if signature changed from diff, it's an API change
            # We check if the symbol's signature line is in the diff
            changes.append({
                "api_symbol": sym_name,
                "before_signature": sym.get("signature", ""),
                "after_signature": sym.get("signature", ""),
                "commit_sha": commit_sha,
                "change_type": "MODIFIED",
                "confidence": "HIGH" if "def " in sig else "MEDIUM",
                "evidence": f"Symbol '{sym_name}' definition in changed hunk at lines {sym.get('line_start')}-{sym.get('line_end')}"
            })

        return changes

    def detect_from_diff(self, file_path: str, file_content_before: str, file_content_after: str, changed_symbols: List[Dict[str, Any]], commit_sha: str) -> List[Dict[str, Any]]:
        """Detect API changes by comparing before/after signatures."""
        changes = []
        before_symbols = parse_python_file(file_path, file_content_before).symbols if file_content_before else []
        after_symbols = parse_python_file(file_path, file_content_after).symbols if file_content_after else []

        before_map = {s.symbol_name: s for s in before_symbols}
        after_map = {s.symbol_name: s for s in after_symbols}

        # Find symbols that appear in both but with different signatures
        for name in set(before_map.keys()) | set(after_map.keys()):
            if name.startswith("_"):
                continue
            b = before_map.get(name)
            a = after_map.get(name)
            if b and a and b.signature != a.signature:
                changes.append({
                    "api_symbol": name,
                    "before_signature": b.signature,
                    "after_signature": a.signature,
                    "commit_sha": commit_sha,
                    "change_type": "SIGNATURE_CHANGE",
                    "confidence": "HIGH",
                    "evidence": f"Signature changed from '{b.signature}' to '{a.signature}' in {file_path}"
                })
            elif b and not a:
                changes.append({
                    "api_symbol": name,
                    "before_signature": b.signature,
                    "after_signature": None,
                    "commit_sha": commit_sha,
                    "change_type": "REMOVED",
                    "confidence": "HIGH",
                    "evidence": f"API symbol '{name}' was removed from {file_path}"
                })
            elif a and not b:
                changes.append({
                    "api_symbol": name,
                    "before_signature": None,
                    "after_signature": a.signature,
                    "commit_sha": commit_sha,
                    "change_type": "ADDED",
                    "confidence": "HIGH",
                    "evidence": f"API symbol '{name}' was added in {file_path}"
                })

        return changes


class DependencyChangeDetector:
    """Detects dependency changes in manifest files."""

    MANIFEST_FILES = {"package.json", "requirements.txt", "requirements-dev.txt", "pyproject.toml", "go.mod", "Cargo.toml", "pom.xml", "build.gradle", "Gemfile", "composer.json"}

    def detect(self, file_path: str, file_before: str, file_after: str, ecosystem: str) -> List[Dict[str, Any]]:
        """Detect dependency additions/removals/version changes."""
        from packages.parsing.manifest_builder import extract_dependencies_from_manifest

        before_deps = extract_dependencies_from_manifest(file_path, file_before) if file_before else []
        after_deps = extract_dependencies_from_manifest(file_path, file_after) if file_after else []

        before_map = {d.package_name: d for d in before_deps}
        after_map = {d.package_name: d for d in after_deps}

        changes = []
        for pkg in set(before_map.keys()) | set(after_map.keys()):
            b = before_map.get(pkg)
            a = after_map.get(pkg)
            if b and not a:
                changes.append({
                    "package_name": pkg,
                    "ecosystem": ecosystem,
                    "change_type": "REMOVED",
                    "old_version": b.version_spec,
                    "new_version": None,
                    "confidence": "HIGH",
                    "evidence": f"Dependency '{pkg}' removed from {file_path}"
                })
            elif a and not b:
                changes.append({
                    "package_name": pkg,
                    "ecosystem": ecosystem,
                    "change_type": "ADDED",
                    "old_version": None,
                    "new_version": a.version_spec,
                    "confidence": "HIGH",
                    "evidence": f"Dependency '{pkg}' added to {file_path}"
                })
            elif b and a and b.version_spec != a.version_spec:
                changes.append({
                    "package_name": pkg,
                    "ecosystem": ecosystem,
                    "change_type": "VERSION_CHANGED",
                    "old_version": b.version_spec,
                    "new_version": a.version_spec,
                    "confidence": "HIGH",
                    "evidence": f"Dependency '{pkg}' version changed from '{b.version_spec}' to '{a.version_spec}' in {file_path}"
                })

        return changes


def get_git_file_content(repo_path: str, commit_sha: str, file_path: str) -> Optional[str]:
    """Get file content at a specific commit using git show."""
    try:
        cmd = ["git", "show", f"{commit_sha}:{file_path}"]
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, check=True)
        return result.stdout
    except Exception:
        return None


def get_git_diff(repo_path: str, parent_sha: str, commit_sha: str) -> str:
    """Get the diff between parent and commit."""
    try:
        cmd = ["git", "diff", "--no-color", "--no-prefix", f"{parent_sha}..{commit_sha}"]
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, check=True)
        return result.stdout
    except Exception:
        return ""


class DiffAnalyzer:
    """Main entry point for AST-aware git diff analysis (Phase 23)."""

    def __init__(self, max_diff_size: int = 100000):
        self.max_diff_size = max_diff_size
        self.parser = DiffParser()
        self.symbol_mapper = SymbolChangeMapper()
        self.classifier = ChangeClassifier()
        self.api_detector = APIChangeDetector()
        self.dep_detector = DependencyChangeDetector()

    def analyze_commit(
        self,
        repo_path: str,
        commit_sha: str,
        parent_sha: str,
        parsed_symbols: List[SymbolNode]
    ) -> Dict[str, Any]:
        """Analyze a commit's diff and produce structured CommitAnalysis data."""
        diff_text = get_git_diff(repo_path, parent_sha, commit_sha)
        if len(diff_text) > self.max_diff_size:
            diff_text = diff_text[:self.max_diff_size]

        file_diffs = self.parser.parse_diff(diff_text)

        changed_files = []
        changed_symbols = []
        api_changes = []
        dependency_changes = []
        configuration_changes = []
        risk_signals = []

        # Group symbols by file
        symbols_by_file: Dict[str, List[SymbolNode]] = {}
        for s in parsed_symbols:
            symbols_by_file.setdefault(s.file_path, []).append(s)

        for fd in file_diffs:
            changed_files.append({
                "file_path": fd.new_path if fd.new_path != "/dev/null" else fd.old_path,
                "old_path": fd.old_path,
                "new_path": fd.new_path,
                "change_type": fd.change_type,
                "additions": fd.additions,
                "deletions": fd.deletions
            })

            # Fix old/new paths for diff format
            actual_path = fd.new_path if fd.change_type != "REMOVED" else fd.old_path
            if actual_path in ("/dev/null", ""):
                continue

            # Get file content at parent and commit for API change detection
            content_before = get_git_file_content(repo_path, parent_sha, actual_path)
            content_after = get_git_file_content(repo_path, commit_sha, actual_path)

            # Map hunks to symbols
            file_symbols = symbols_by_file.get(actual_path, [])
            sym_results = self.symbol_mapper.find_affected_symbols(
                actual_path, content_after or "", fd.hunks,
                language="Python" if actual_path.endswith(".py") else "Unknown"
            )

            for sym in sym_results:
                change_type = self.classifier.classify(
                    sym["symbol_name"], sym["line_start"], sym["line_end"], fd.hunks
                )
                changed_symbols.append({
                    "file_path": actual_path,
                    "symbol_name": sym["symbol_name"],
                    "symbol_type": sym["symbol_type"],
                    "change_category": change_type,
                    "line_start": sym["line_start"],
                    "line_end": sym["line_end"],
                    "signature": sym["signature"],
                    "confidence": "HIGH"
                })

                if change_type == "SIGNATURE_CHANGE":
                    risk_signals.append(f"API signature change in {sym['symbol_name']} ({actual_path})")

            # API change detection
            if content_before and content_after and actual_path.endswith(".py"):
                api_changes.extend(self.api_detector.detect_from_diff(
                    actual_path, content_before, content_after, sym_results, commit_sha
                ))

            # Dependency change detection
            basename = actual_path.split("/")[-1]
            if basename in DependencyChangeDetector.MANIFEST_FILES:
                ecosystem = "npm" if basename == "package.json" else "pypi" if "requirements" in basename or basename == "pyproject.toml" else "go" if basename == "go.mod" else "rust" if basename == "Cargo.toml" else "unknown"
                dep_changes = self.dep_detector.detect(actual_path, content_before or "", content_after or "", ecosystem)
                dependency_changes.extend(dep_changes)

            # Configuration change detection
            config_exts = (".yaml", ".yml", ".json", ".toml", ".ini", ".env", ".cfg")
            if actual_path.lower().endswith(config_exts) or basename in ("Dockerfile", "docker-compose.yml"):
                if fd.additions > 0 or fd.deletions > 0:
                    configuration_changes.append({
                        "file_path": actual_path,
                        "change_type": fd.change_type,
                        "additions": fd.additions,
                        "deletions": fd.deletions,
                        "evidence": f"Configuration file {actual_path} modified in commit {commit_sha[:8]}"
                    })

        return {
            "changed_files": changed_files,
            "changed_symbols": changed_symbols,
            "api_changes": api_changes,
            "dependency_changes": dependency_changes,
            "configuration_changes": configuration_changes,
            "risk_signals": risk_signals
        }
