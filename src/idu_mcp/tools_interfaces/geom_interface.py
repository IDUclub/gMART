from typing import Literal

from fastmcp import FastMCP, Context
from fastmcp.dependencies import Depends
from fastmcp.server.dependencies import CurrentContext
from geojson_pydantic import FeatureCollection

from src.idu_mcp.dependencies.dependencies import get_restrictions_context
from src.idu_mcp.contexts.geom_contexts.create_restrictions_context import RestrictionsContext
from src.idu_mcp.contexts.geom_contexts.create_buffers_context import BufferContext
from src.idu_mcp.dependencies.dependencies import get_buffers_context, get_geom_tools
from src.idu_mcp.tools_services.geometry_tools import GeometryTools

geometry_mcp = FastMCP("GEOMETRY MCP")


@geometry_mcp.tool(
    title="Создать буферы слоёв",
    description="""Генерирует геометрические буферы вокруг входных объектов.
    Буферы создаются только для тех слоёв, которые могут использоваться в дальнейшем для наложения ограничений
    
    Входные параметры:
    Параметр | Тип	| Обязателен | Описание
    buffer_info |	dict[str, int |	Literal["round", "flat", "square"]]
    
    Выходные данные:
    
    Тип | Описание
    dict[str, FeatureCollection] | Словарь, где ключ - имя слоя (совпадает с ключом из objects), 
    а значение - FeatureCollection (GeoJSON) с геометрией буферов для соответствующего слоя. 
    Координаты возвращаются в той же CRS, что и у входных данных.
    
    Пример вызова:
    
    {
      "buffer_info": {
        "жилая застройка": { "buffer_size": 150, "buffer_type": "round" },
        "промышленная зона": { "buffer_size": 300, "buffer_type": "square" }
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
    ctx: Context = CurrentContext(),
    geom_tools: GeometryTools = Depends(get_geom_tools),
) -> dict[str, FeatureCollection]:
    """
    Create buffers for layers.
    Args:
        buffer_info (dict[str, int | Literal["round", "flat", "square"]]): Buffer info, containing buffer type and buffer size.
        context (BufferContext): Context for mcp tool call.
        geom_tools (GeometryTools): GeometryTools instance.
    Returns:
        dict[str, FeatureCollection]: layer of objects which restricts which objects.
    """
    objects = ctx.request_context.meta.objects
    return await geom_tools.async_generate_geometry_buffers(
        buffer_info, objects
    )


@geometry_mcp.tool(
    title="Создать ограничения на основе правил",
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
    Пример одного ограничения:
    {
      "title": "Школа",
      "description": "No new construction",
      "to": ["house", "apartment"]
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
      }
    }
    """,
    tags={"geometry", "restrictions"},
)
async def create_restrictions(
    generators: list[str],
    objects: list[str],
    restrictions: dict[str, dict[str, str | list[str]]],
    ctx: Context = CurrentContext(),
    geom_tools: GeometryTools = Depends(get_geom_tools),
) -> tuple[dict, dict]:

    layers = ctx.request_context.meta.layers
    return await geom_tools.async_create_restrictions(
        layers, generators, objects, restrictions
    )
