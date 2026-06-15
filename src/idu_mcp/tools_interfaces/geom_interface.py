from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from fastmcp.exceptions import ToolError
from geojson_pydantic import FeatureCollection

from src.idu_mcp.dependencies.dependencies import get_geom_tools
from src.idu_mcp.tools_descriptions import geometry_validation_messages as messages
from src.idu_mcp.tools_services.geometry_tools import GeometryTools
from src.idu_mcp.tools_services.geometry_validator import GeometryToolValidator

geometry_mcp = FastMCP("GEOMETRY MCP")


@geometry_mcp.tool(
    name="CreateBuffers",
    title="Создать буферы слоёв",
    description="""Генерирует геометрические буферы вокруг входных объектов.
    Буферы создаются только для тех слоёв, которые могут использоваться в дальнейшем для наложения ограничений
    
    Входные параметры:
    Параметр | Тип	| Обязателен | Описание
    buffer_info |	dict[str, int |	Literal["round", "flat", "square"] | str] | ✅ | Словарь, где ключ - имя слоя, а значение - параметры буфера (buffer_size, buffer_type, title).
    objects | dict[str, FeatureCollection] | ✅ | Исходные слои объектов (ключ - имя слоя, значение - FeatureCollection в GeoJSON), вокруг которых строятся буферы. Ключи должны совпадать с ключами из buffer_info.

    Выходные данные:
    
    Тип | Описание
    dict[str, FeatureCollection] | Словарь, где ключ - имя слоя (совпадает с ключом из objects), 
    а значение - FeatureCollection (GeoJSON) с геометрией буферов для соответствующего слоя. 
    Координаты возвращаются в той же CRS, что и у входных данных.
    
    Пример вызова:
    
    {
      "buffer_info": {
        "жилая застройка": {"buffer_size": 150, "buffer_type": "round", "title": "Ограничение от промышленных объектов в радиусе 150 метров"},
        "промышленная зона": { "buffer_size": 300, "buffer_type": "square", "title": "Ограничение от водных объектов в радиусе 300 метров"}
      },
      "objects": {
        "жилая застройка": "FeatureCollection",
        "промышленная зона": "FeatureCollection"
      }
    }
    
    Ожидаемый результат:
    
    {
      "жилая застройка": { "type": "FeatureCollection", "features": [ … буферные геометрии … ] },
      "промышленная зона": { "type": "FeatureCollection", "features": [ … буферные геометрии … ] }
    }
    """,
    annotations={"title": "GET buffers for layers", "readOnlyHint": True},
    meta={"author": "LeonDeTur"},
    tags={"geometry", "buffers"},
)
async def create_buffers(
    buffer_info: dict,
    objects: dict,
    geom_tools: GeometryTools = Depends(get_geom_tools),
) -> dict[str, FeatureCollection]:
    """
    Create buffers for layers.
    Args:
        buffer_info (dict[str, int | Literal["round", "flat", "square"] | str]): Buffer info, containing buffer type and buffer size.
        objects (dict): Source object layers as dict[layer_name, FeatureCollection] in GeoJSON, around which
            the buffers are built. Keys must match the keys of buffer_info.
        geom_tools (GeometryTools): GeometryTools instance.
    Returns:
        dict[str, FeatureCollection]: layer of objects which restricts which objects.
    """

    GeometryToolValidator.validate_buffers(buffer_info, objects)
    try:
        return await geom_tools.async_generate_geometry_buffers(buffer_info, objects)
    except Exception as e:
        raise ToolError(messages.BUFFERS_RUNTIME_ERROR.format(error=e)) from e


@geometry_mcp.tool(
    name="CreateRestrictions",
    title="Создать пространственные ограничения",
    description="""Создаёт геометрические «ограничения» (restrictions) для объектов, 
    находящихся в зоне влияния «генераторов» (objects that can impose restrictions).
    
    Для каждого объекта‑приёмника (restricted object) добавляется информация о том, 
    какие ограничения применимы к нему, и возвращается два слоя в формате GeoJSON:

    restricted_objects – объекты с полями restriction_name и restriction_description.
    generated_restrictions – (внутренний слой, совпадает с первым) – используется для дальнейшего анализа/визуализации.
    
    Параметр | Тип | Обязательно | Описание
    generators | list[str] | ✅ | Список типов объектов, которые генерируют ограничения (должны совпадать с ключами из layers).
    objects	list[str] | ✅ | Список типов объектов, которые можут быть ограничены (приёмники).
    restrictions | dict[str, dict[str, str	list[str]]] | ✅ | Словарь, в котором ключ – имя ограничения (обычно это тип объекта‑генератора). Значение – вложенный словарь с метаданными ограничения: "title" – короткое название (строка). "description" – подробное описание (строка). "to" – список типов объектов, к которым это ограничение применимо (список строк).
    layers | dict[str, FeatureCollection] | ✅ | Слои объектов и буферов (ключ – имя слоя, значение – FeatureCollection в GeoJSON), на основе которых строятся ограничения. Ключи должны совпадать с именами из generators и objects.
    Пример одного ограничения:
    {
      "title": "Школа",
      "description": "No new construction",
      "to": ["дом"]
    }
    
    Пример входных данных:
    
    {
        "generators": ["школа"],
        "objects": ["дом"],
        "restrictions": {
            "школа": {"title": "Зоны школ",
          "description": "Запрещено возведение объектов вокруг школ",
          "to": ["дом"]
        }
      },
      "layers": {
        "школа": "FeatureCollection",
        "дом": "FeatureCollection"
      }
    }
    """,
    tags={"geometry", "restrictions"},
)
async def create_restrictions(
    generators: list[str],
    objects: list[str],
    restrictions: dict[str, dict[str, str | list[str]]],
    layers: dict,
    geom_tools: GeometryTools = Depends(get_geom_tools),
) -> dict[str, FeatureCollection]:
    """
    Function forms layers by provided restrictions.
    Args:
        generators (list[str]): list of restriction generators names.
        objects (list[str]): list of all needed objects.
        restrictions (dict[str, dict[str, str | list[str]]]): info with restriction rules.
        layers (dict): Object and buffer layers as dict[layer_name, FeatureCollection] in GeoJSON, used to
            build the restrictions. Keys must match the names from generators and objects.
        geom_tools (GeometryTools): GeometryTools instance.
    Returns:
        dict[str, dict]: tuple of layers where firs FeatureCollection is restricted objects layer
        and second FeatureCollection is generators layer.
    """

    GeometryToolValidator.validate_restrictions(
        generators, objects, restrictions, layers
    )
    try:
        return await geom_tools.async_create_restrictions(
            layers, generators, objects, restrictions
        )
    except Exception as e:
        raise ToolError(messages.RESTRICTIONS_RUNTIME_ERROR.format(error=e)) from e
