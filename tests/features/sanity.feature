Feature: Sanity script core behaviors
  Ensures key helper functions are robust and behave consistently.

  Scenario: Deriving API base from MCP URL
    Given an MCP URL "http://localhost:8013/mcp"
    When I derive the API base
    Then the API base should be "http://localhost:8013/api"

  Scenario: Pretty-printing non-serializable objects
    Given a non-serializable object
    When I pretty print the object
    Then the pretty output contains "bytes"
