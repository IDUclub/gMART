GET_SERVICES_BY_GEOMETRY = """Получить сервисы сценария в точной области поиска.

Область — объединение всех геометрий слоёв zones_layers из layers в EPSG:4326.
buffer_m добавляет буфер в метрах в локальной метрической CRS. Геометрия не
упрощается: подходят произвольные полигоны, в том числе будущие изохроны.
services_names — канонические имена типов Urban API в единственном числе.
На каждый тип выполняется один POST /scenarios/{scenario_id}/services/by_geometry/geojson;
типы обрабатываются параллельно (COMPLIANCE_BY_GEOMETRY_CONCURRENCY, по умолчанию 8).
Повторы сетевых сбоев и reset by peer ограничены, с экспоненциальной задержкой.

intersection_type: intersects — объект пересекает область, within — весь объект
внутри области. Проверяется исходная геометрия, без центроидов и bounding box.
При 413 или 422 из-за размера область делится на компоненты, затем на квадранты
без потери геометрии (COMPLIANCE_BY_GEOMETRY_MAX_SPLIT_DEPTH, по умолчанию 6).
Части запрашиваются с intersects; после слияния применяется точный исходный
предикат. Дубликаты удаляются по service_id. При исчерпании глубины — явная ошибка.
physical_object_type_ids зарезервирован: services/by_geometry не поддерживает этот фильтр.

Результат: {layer_name: FeatureCollection}, исходные геометрии WGS84 и свойства
сервисов (service_id, service_type_id/name, service_type, name, capacity,
object_geometry_id, object_ref и остальные поля Urban API).
meta: complete, truncated, revision, calls_made (включая повторы и отказы),
mode (single/split), component_count, part_count (успешные части по всем типам),
intersection_type, per_type_counts (число сервисов по ID типа после дедупликации).
Пустая область возвращает пустой слой без POST; неизвестный тип — ошибка.
"""

GET_PHYSICAL_OBJECTS_BY_GEOMETRY = """Получить физические объекты сценария в точной области поиска.

physical_objects_names — канонические имена типов Urban API в единственном числе.
Область — объединение геометрий zones_layers из layers в EPSG:4326, с buffer_m
в метрах в локальной метрической CRS, без упрощения геометрии.
Один POST /scenarios/{scenario_id}/physical_objects/by_geometry/geojson на тип
с physical_object_type_id и intersection_type (intersects или within).
Типы выполняются параллельно под общим с сервисами семафором:
COMPLIANCE_BY_GEOMETRY_CONCURRENCY (8). При 413/размерном 422 — деление на
компоненты, затем квадранты до COMPLIANCE_BY_GEOMETRY_MAX_SPLIT_DEPTH (6).
Части запрашиваются с intersects; после слияния применяется точный исходный
предикат ко всей области. Дубликаты удаляются по паре physical_object_id,
object_geometry_id: все геометрии одного объекта сохраняются.

Результат: {layer_name: FeatureCollection}, исходные Polygon/MultiPolygon,
LineString/Point в WGS84. Свойства КОМПАКТНЫЕ: physical_object_id,
physical_object_type_id, physical_object_type_name, name, object_geometry_id,
is_scenario_physical_object, is_scenario_geometry, is_locked. Нет building,
вложенных properties, адреса и прочих атрибутов полного GetPhysicalObjects.
Добавляются physical_object_type={physical_object_type_id, name} и object_ref.
Для норм с атрибутами используйте GetPhysicalObjects(centers_only=false).
meta: complete, truncated, revision (physical_objects_by_geometry), calls_made
(все HTTP-попытки), mode (single/split), component_count, part_count,
intersection_type, per_type_counts (уникальные пары объект/геометрия по типам).
Пустая область — пустой слой без POST; неизвестный тип — ошибка.
Повторы транзиентных сбоев ограничены; исчерпание глубины — явная ошибка.
"""

NORM_LAYER = """ Дополнительно возвращает norm_layer — один FeatureCollection нормы:
feature_kind=zone|violation|object; зоны имеют status/status_ru/status_reason,
нарушения — измерение, порог, дефицит, explanation_ru, violation_geometry_mode
и zone_ref; связанные объекты — object_ref, object_name, type и zone_refs.
Общие поля каждой Feature: norm_id, norm_status/norm_status_ru/norm_status_reason,
check_type_en/ru, norm_reference, document_name, clause_number, norm_text.
meta.group_properties перечисляет общие поля, meta.group_by=norm_id.
GeometryCollection не используется; пересечение смешанных типов разбивается
на отдельные Feature. Статус нормы: violated при нарушенной зоне, иначе partial
при непроверенных объектах, иначе passed. Причина violated/partial обязательна.
norm_layer содержит все результаты независимо от фильтра result_mode;
существующие корзины, coverage, summary и evidence сохранены.
"""
