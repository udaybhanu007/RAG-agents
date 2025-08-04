import subprocess

class MCPNeo4jClient:
    def __init__(self, process: subprocess.Popen):
        self.process = process

    def send(self, message: str):
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Server process is not running.")
        self.process.stdin.write(message + '\n')
        self.process.stdin.flush()

    def receive(self, multiline=False):
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Server process is not running.")
        if multiline:
            # Read until EOF or timeout or empty line
            lines = []
            while True:
                line = self.process.stdout.readline()
                if not line or line.strip() == '':
                    break
                lines.append(line.strip())
            return '\n'.join(lines)
        else:
            return self.process.stdout.readline().strip()

    def receive_stderr(self):
        if self.process is None or self.process.stderr is None:
            raise RuntimeError("Server process is not running.")
        return self.process.stderr.read().strip()

# Example usage:
if __name__ == "__main__":
    from mcp_neo4j_server import MCPNeo4jServer
    server = MCPNeo4jServer()
    proc = server.run()
    client = MCPNeo4jClient(proc)
    # Send a Cypher query to list the names of databases
    query = '{"query": "SHOW DATABASES YIELD name RETURN name"}'
    client.send(query)
    print("Response:", client.receive(multiline=True))
    # Print any server errors
    if proc.poll() is not None:
        print("Server stderr:", client.receive_stderr())
