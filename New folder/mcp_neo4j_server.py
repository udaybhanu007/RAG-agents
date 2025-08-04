import subprocess
import os

class MCPNeo4jServer:
    def __init__(self,
                 command="uvx",
                 args=None,
                 env=None):
        if args is None:
            args = ["mcp-neo4j-cypher@0.2.3", "--transport", "stdio"]
        if env is None:
            env = {
                "NEO4J_URI": "neo4j://localhost:7687",
                "NEO4J_USERNAME": "neo4j",
                "NEO4J_PASSWORD": "password",
                "NEO4J_DATABASE": "neo4j"
            }
        self.command = command
        self.args = args
        self.env = env
        self.process = None  # Store the process object

    def run(self):
        # Merge the current environment with the custom one
        env = os.environ.copy()
        env.update(self.env)
        self.process = subprocess.Popen(
            [self.command] + self.args,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return self.process

    def status(self):
        if self.process is None:
            return "Not started"
        if self.process.poll() is None:
            return f"Running (PID: {self.process.pid})"
        else:
            return f"Stopped (return code: {self.process.returncode})"

# Example usage:
if __name__ == "__main__":
    server = MCPNeo4jServer()
    proc = server.run()
    print("MCP Neo4j server started with PID:", proc.pid)
    print("Status:", server.status())
