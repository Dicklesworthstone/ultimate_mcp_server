import importlib.util
import sys

import pytest

pytest_bdd = pytest.importorskip("pytest_bdd")
from pytest_bdd import scenarios, given, when, then, parsers  # type: ignore


scenarios("sanity.feature")


def _load_module():
    path = "/home/ec2-user/git/ultimate_mcp_server/test_sanity.py"
    spec = importlib.util.spec_from_file_location("umcp_test_sanity", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@given(parsers.parse('an MCP URL "{url}"'))
def mcp_url(url):
    return url


@when("I derive the API base")
def derive_api_base(mcp_url):
    mod = _load_module()
    return mod._derive_api_base_from_mcp_url(mcp_url)


@then(parsers.parse('the API base should be "{expected}"'))
def api_base_should_be(derive_api_base, expected):
    assert derive_api_base == expected


@given("a non-serializable object")
def non_serializable():
    class X:
        def __init__(self):
            self.data = {"a": 1}
            self.blob = b"bytes"

    return X()


@when("I pretty print the object")
def when_pretty(non_serializable):
    mod = _load_module()
    return mod.pretty(non_serializable)


@then(parsers.parse('the pretty output contains "{text}"'))
def then_contains(when_pretty, text):
    assert text in when_pretty
