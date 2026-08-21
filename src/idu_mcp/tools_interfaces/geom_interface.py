import asyncio
import json
from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from fastmcp.exceptions import ToolError
from geojson_pydantic import FeatureCollection

from src.idu_mcp.dependencies.dependencies import (
    get_compliance_geometry_tools,
    get_geom_tools,
)
from src.idu_mcp.tools_descriptions import geometry_validation_messages as messages
from src.idu_mcp.tools_services.compliance_geometry import ComplianceGeometryTools
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

    restricted_objects – объекты с исходными атрибутами, составным object_ref,
    полями restriction_name/restriction_description и массивом restriction_evidence,
    объясняющим каждое пересечение и источник правила.
    generated_restrictions – буферные геометрии источников с исходными атрибутами
    и нормативным происхождением – используются для анализа и визуализации.
    
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
    restrictions: dict[
        str,
        dict[str, str | list[str] | dict[str, Any] | None],
    ],
    layers: dict,
    geom_tools: GeometryTools = Depends(get_geom_tools),
) -> dict[str, FeatureCollection]:
    """
    Function forms layers by provided restrictions.
    Args:
        generators (list[str]): list of restriction generators names.
        objects (list[str]): list of all needed objects.
        restrictions: restriction rules. Alongside title/description/to, a rule may carry
            origin, a nullable restriction_id and structured NormGraph provenance.
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


async def _run_compliance_operation(
    method,
    *,
    timeout_seconds: int = 120,
    max_payload_bytes: int = 64 * 1024 * 1024,
    **kwargs,
) -> dict[str, Any]:
    def invoke():
        payload_size = len(
            json.dumps(kwargs.get("layers") or {}, ensure_ascii=False).encode("utf-8")
        )
        if payload_size > max_payload_bytes:
            raise ValueError(
                f"Payload exceeds the {max_payload_bytes} byte operation limit"
            )
        return method(**kwargs)

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(invoke), timeout=timeout_seconds
        )
    except TimeoutError as exc:
        raise ToolError("Операция проверки превысила лимит времени") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolError(f"Некорректные данные проверки: {exc}") from exc
    except Exception as exc:
        raise ToolError(f"Ошибка геопространственной проверки: {exc}") from exc


@geometry_mcp.tool(
    name="CheckDistanceFromSource",
    title="Проверить расстояние или готовую зону",
    description=(
        "Детерминированно проверяет полный набор целевых объектов относительно исходной "
        "или буферной геометрии. Возвращает отдельные корзины violated/passed/unchecked, "
        "coverage и evidence по object_ref."
    ),
    tags={"geometry", "compliance"},
)
async def check_distance_from_source(
    source_layer: str,
    targets: list[str],
    geometry_mode: Literal["buffered", "source_geometry"],
    predicate: Literal["intersects", "within", "contains"],
    violation_when: Literal["matched", "not_matched"],
    result_mode: Literal["violated", "passed", "both"],
    restriction_id: str,
    layers: dict,
    template_version: int = 1,
    distance_m: float | None = None,
    provenance: dict[str, Any] | None = None,
    input_revision: str | None = None,
    tools: ComplianceGeometryTools = Depends(get_compliance_geometry_tools),
) -> dict[str, Any]:
    return await _run_compliance_operation(
        tools.distance_from_source,
        source_layer=source_layer,
        targets=targets,
        geometry_mode=geometry_mode,
        predicate=predicate,
        violation_when=violation_when,
        result_mode=result_mode,
        restriction_id=restriction_id,
        layers=layers,
        template_version=template_version,
        distance_m=distance_m,
        provenance=provenance,
        input_revision=input_revision,
    )


@geometry_mcp.tool(
    name="CheckDistanceTable",
    title="Проверить индивидуальные расстояния",
    description=(
        "Строит индивидуальный буфер источника по разрешённому атрибуту и таблице "
        "непересекающихся диапазонов; неизвестные значения остаются непроверенными."
    ),
    tags={"geometry", "compliance"},
)
async def check_distance_table(
    source_layer: str,
    targets: list[str],
    attribute_field: str,
    bands: list[dict[str, Any]],
    predicate: Literal["intersects", "within", "contains"],
    violation_when: Literal["matched", "not_matched"],
    result_mode: Literal["violated", "passed", "both"],
    restriction_id: str,
    layers: dict,
    template_version: int = 1,
    provenance: dict[str, Any] | None = None,
    input_revision: str | None = None,
    tools: ComplianceGeometryTools = Depends(get_compliance_geometry_tools),
) -> dict[str, Any]:
    return await _run_compliance_operation(
        tools.distance_table,
        source_layer=source_layer,
        targets=targets,
        attribute_field=attribute_field,
        bands=bands,
        predicate=predicate,
        violation_when=violation_when,
        result_mode=result_mode,
        restriction_id=restriction_id,
        layers=layers,
        template_version=template_version,
        provenance=provenance,
        input_revision=input_revision,
    )


@geometry_mcp.tool(
    name="CheckPresenceWithin",
    title="Проверить наличие соседей",
    description=(
        "Выполняет полный spatial left/anti join и возвращает каждый применимый объект "
        "ровно один раз, включая доказательство отсутствия соседа. Минимальное число "
        "соседей проверяется отдельно для каждого обязательного слоя."
    ),
    tags={"geometry", "compliance"},
)
async def check_presence_within(
    objects_layer: str,
    required_neighbor_layers: list[str],
    distance_m: float,
    minimum_neighbors: int,
    result_mode: Literal["violated", "passed", "both"],
    restriction_id: str,
    layers: dict,
    template_version: int = 1,
    provenance: dict[str, Any] | None = None,
    input_revision: str | None = None,
    tools: ComplianceGeometryTools = Depends(get_compliance_geometry_tools),
) -> dict[str, Any]:
    return await _run_compliance_operation(
        tools.presence_within,
        objects_layer=objects_layer,
        required_neighbor_layers=required_neighbor_layers,
        distance_m=distance_m,
        minimum_neighbors=minimum_neighbors,
        result_mode=result_mode,
        restriction_id=restriction_id,
        layers=layers,
        template_version=template_version,
        provenance=provenance,
        input_revision=input_revision,
    )


@geometry_mcp.tool(
    name="CheckZonalAttributeThreshold",
    title="Сравнить атрибут с зональным порогом",
    description=(
        "Сопоставляет объекты зонам и проверяет атрибут по константе или разрешённому "
        "атрибуту зоны с политикой strictest_threshold."
    ),
    tags={"geometry", "compliance"},
)
async def check_zonal_attribute_threshold(
    objects_layer: str,
    zones_layer: str,
    object_attribute: str,
    operator: Literal["<", "<=", ">", ">=", "=="],
    join_predicate: Literal["intersects", "within", "contains"],
    result_mode: Literal["violated", "passed", "both"],
    restriction_id: str,
    layers: dict,
    constant_threshold: float | None = None,
    zone_threshold_attribute: str | None = None,
    template_version: int = 1,
    provenance: dict[str, Any] | None = None,
    input_revision: str | None = None,
    tools: ComplianceGeometryTools = Depends(get_compliance_geometry_tools),
) -> dict[str, Any]:
    return await _run_compliance_operation(
        tools.zonal_attribute_threshold,
        objects_layer=objects_layer,
        zones_layer=zones_layer,
        object_attribute=object_attribute,
        operator=operator,
        constant_threshold=constant_threshold,
        zone_threshold_attribute=zone_threshold_attribute,
        join_predicate=join_predicate,
        result_mode=result_mode,
        restriction_id=restriction_id,
        layers=layers,
        template_version=template_version,
        provenance=provenance,
        input_revision=input_revision,
    )


@geometry_mcp.tool(
    name="CheckZonalRatio",
    title="Проверить долю площади в зоне",
    description=(
        "В локальной метрической CRS обрезает числитель границей зоны, объединяет "
        "перекрытия и сравнивает процент с нормативным порогом."
    ),
    tags={"geometry", "compliance"},
)
async def check_zonal_ratio(
    zones_layer: str,
    numerator_layer: str,
    operator: Literal["<", "<=", ">", ">=", "=="],
    threshold: float,
    result_mode: Literal["violated", "passed", "both"],
    invalid_geometry_policy: Literal["repair", "reject"],
    restriction_id: str,
    layers: dict,
    template_version: int = 1,
    provenance: dict[str, Any] | None = None,
    input_revision: str | None = None,
    tools: ComplianceGeometryTools = Depends(get_compliance_geometry_tools),
) -> dict[str, Any]:
    return await _run_compliance_operation(
        tools.zonal_ratio,
        timeout_seconds=180,
        zones_layer=zones_layer,
        numerator_layer=numerator_layer,
        operator=operator,
        threshold=threshold,
        result_mode=result_mode,
        invalid_geometry_policy=invalid_geometry_policy,
        restriction_id=restriction_id,
        layers=layers,
        template_version=template_version,
        provenance=provenance,
        input_revision=input_revision,
    )
