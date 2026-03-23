from src.idu_mcp.tools_services.geometry_tools import GeometryTools

from .base_tool_dep import BaseDep


class GeomToolsDeps(BaseDep):
    """
    Class for managing urban api tools
    Attributes:
        self.geometry_tools (GeometryTools): GeometryTools
    """

    def __init__(self):
        """
        Constructor for GeomToolsDeps class
        """

        super().__init__()
        self.geometry_tools: GeometryTools = GeometryTools()
