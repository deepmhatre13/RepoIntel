from packages.parsing.python_ast import parse_python_file
from packages.parsing.js_ts_ast import parse_js_ts_file
from packages.parsing.manifest_builder import extract_dependencies_from_manifest

def test_python_parsing():
    code = """
import os

class Machine:
    def start(self):
        print("Starting")

def run():
    m = Machine()
    m.start()
"""
    res = parse_python_file("test.py", code)
    assert res.parse_succeeded
    assert len(res.symbols) == 3  # Machine, Machine.start, run
    assert any(s.symbol_name == "Machine" for s in res.symbols)

def test_js_parsing():
    code = """
import { helper } from './utils';

export class Runner {
    run() {
        console.log("running");
    }
}

export function calc(x) {
    return x * 2;
}
"""
    res = parse_js_ts_file("app.js", code)
    assert res.parse_succeeded
    assert len(res.symbols) >= 2
    assert len(res.edges) >= 1

def test_manifest_extraction():
    pkg_json = '{"dependencies": {"express": "^4.17.1"}, "devDependencies": {"jest": "^27.0.0"}}'
    deps = extract_dependencies_from_manifest("package.json", pkg_json)
    assert len(deps) == 2
    assert any(d.package_name == "express" for d in deps)
