"""User-facing (Russian) error messages for geometry tool input validation.

Kept separate from both the tool interface and the validation logic so that the
wording can be reviewed and edited in one place without touching code.
"""

# CreateBuffers
MISSING_BUFFER_LAYERS = (
    "Не удалось построить буферы: слои {missing} отсутствуют в 'objects'. "
    "Переданные слои objects: {objects}; запрошены буферы для: {requested}."
)
MISSING_BUFFER_FIELDS = (
    "Не удалось построить буферы: в 'buffer_info' для слоя '{layer}' "
    "отсутствуют обязательные поля {fields}."
)
EMPTY_BUFFER_LAYERS = (
    "Не удалось построить буферы: слои {empty} переданы, но не содержат "
    "ни одного объекта с геометрией — на данной территории объекты этих типов "
    "отсутствуют (пустой ответ от GetServices / GetPhysicalObjects). "
    "Постройте буферы только для непустых слоёв."
)
BUFFERS_RUNTIME_ERROR = (
    "Ошибка при выполнении геометрических операций geopandas для буферов: {error}"
)

# CreateRestrictions
MISSING_RESTRICTION_LAYERS = (
    "Не удалось построить ограничения: среди переданных слоёв {layers} "
    "нет генераторов {generators} или целевых объектов {objects} — "
    "проверьте соответствие имён слоёв."
)
MISSING_RESTRICTION_FIELDS = (
    "Не удалось построить ограничения: в ограничении '{name}' "
    "отсутствуют обязательные поля {fields}."
)
EMPTY_RESTRICTION_LAYERS = (
    "Не удалось построить ограничения: слои {empty} переданы, но не содержат "
    "ни одного объекта с геометрией — на данной территории объекты этих типов "
    "отсутствуют (пустой ответ от GetServices / GetPhysicalObjects). "
    "Без геометрии генераторов и целевых объектов ограничения построить нельзя."
)
RESTRICTIONS_RUNTIME_ERROR = (
    "Ошибка при выполнении геометрических операций geopandas для ограничений: {error}"
)
