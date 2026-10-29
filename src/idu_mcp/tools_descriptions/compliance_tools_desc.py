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
physical_object_type_ids зарезервирован: Urban API пока не поддерживает этот фильтр.

Результат: {layer_name: FeatureCollection}, исходные геометрии WGS84 и свойства
сервисов (service_id, service_type_id/name, service_type, name, capacity,
object_geometry_id, object_ref и остальные поля Urban API).
meta: complete, truncated, revision, calls_made (включая повторы и отказы),
mode (single/split), component_count, part_count (успешные части по всем типам),
intersection_type, per_type_counts (число сервисов по ID типа после дедупликации).
Пустая область возвращает пустой слой без POST; неизвестный тип — ошибка.
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
