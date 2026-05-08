from src.idu_mcp.dependencies.tool_deps.base_tool_dep import BaseDep


class ServerDeps(BaseDep):
    def __init__(self, workers: int):
        super().__init__()
        self.workers = workers
