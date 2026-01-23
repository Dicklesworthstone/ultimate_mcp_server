"""Entry point for running the Ultimate MCP Server CLI as a module."""

if __name__ == "__main__":
    from ultimate_mcp_server.cli.typer_cli import app

    app()
