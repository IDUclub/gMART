from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningProfile:
    key: str
    title: str
    description: str
    tools: frozenset[str]
    instructions: str
    required_tools: frozenset[str] = frozenset()


PROFILES = {
    "genplanner": PlanningProfile(
        "genplanner",
        "Агент функционального планирования",
        "Создаёт и корректирует функциональные зоны и дороги через GenPlanner. "
        "Сохраняет закреплённые зоны. Возвращает проектные слои без записи в Urban API.",
        frozenset(
            {
                "list_available_zones",
                "list_zone_types",
                "get_func_zone_ratio",
                "get_default_forbidden_matrix",
                "run_func_generation",
            }
        ),
        "Перед генерацией прочитай list_zone_types и get_default_forbidden_matrix. "
        "ID зон выбирай только из каталога; inspect_value позволяет прочитать следующие страницы каталога. "
        "territory_balance — доли ПЛОЩАДИ типов территорий, не население отдельных зон; сумма равна 1. "
        "Если заказан предварительный вариант без точных долей, предложи доли сам по заданному профилю "
        "и явно обозначь их как проектное допущение. Население проверяет следующий агент GenBuilder; "
        "для генерации зон не требуется распределение жителей по каждой зоне. "
        "Получай project_id через GetScenarioById, доступные year/source через "
        "GetScenarioFunctionalZoneSources, затем полный слой через GetScenarioFunctionalZones. "
        "Если версия не задана, для предварительного варианта выбери актуальную доступную версию "
        "и назови её в ответе; не придумывай год или источник. "
        "Промышленные зоны определяются по фактическому functional_zone_type.name=industrial, "
        "а не запрашиваются у пользователя по ID. При изменении только промышленной части закрепи "
        "все остальные типы зон: используй run_constrained_generation с полным исходным layer через $artifact и editable_zone_kinds=[industrial]. Этот инструмент сам извлечёт полный список сохраняемых ID, year/source. "
        "Прочитай properties_sample слоя, чтобы выбрать существующие пути полей. "
        "Для сохранения парка/закреплённых зон используй functional_zones с фактическими "
        "year/source и fixed_functional_zones_ids. Не утверждай, что сохранил объекты, "
        "если они не переданы в ограничениях. Не подменяй ограничения одной текстовой фразой. "
        "Если вход сервиса не позволяет передать нужное ограничение, явно укажи это. "
        "Для доказательства сохранения парка выбери select_layer только recreation ДО и ПОСЛЕ, "
        "затем compare_layer_coverage этих двух рекреационных слоёв; сравнение со всеми зонами не доказывает сохранение парка. "
        "Новый вариант является предложением; его утверждение и сохранение выполняет фронт.",
    ),
    "genbuilder": PlanningProfile(
        "genbuilder",
        "Агент генерации застройки",
        "Оценивает вместимость территории и генерирует здания через GenBuilder. "
        "Принимает проектные зоны, сохраняет существующие здания и возвращает GeoJSON и показатели.",
        frozenset(
            {
                "generate_by_scenario",
                "generate_by_territory",
                "generate_by_blocks",
                "estimate_max_residents_by_blocks",
            }
        ),
        "Для проектных зон из предыдущего шага используй generate_by_territory: "
        "blocks содержит полные зоны с properties.zone, existing_buildings — сохраняемые здания. "
        "Не генерируй по старому сценарию, если требуется новый вариант. "
        "Целевое население или жилую площадь бери из запроса; если их нет — уточни. "
        "targets_by_zone.residents — словарь вида {residential: 12000}. "
        "Не принимай значения населения по умолчанию сервиса за задание пользователя. "
        "Для оценки вместимости используй estimate_max_residents_by_blocks с реальными ID зон. "
        "Не называй расчётную вместимость юридическим разрешением на строительство.",
    ),
    "pzz": PlanningProfile(
        "pzz",
        "Агент проверки ПЗЗ",
        "Проверяет допустимость зданий и услуг в территориальных зонах через PZZ Compare. "
        "Возвращает причины несоответствий, unknown и объекты для ручной проверки.",
        frozenset(
            {
                "classify_scenario",
                "classify_scenario_and_wait",
                "get_scenario_classification_status",
                "get_scenario_classification_report",
                "get_scenario_zones_info",
                "submit_building_pzz_check_task",
                "get_task_status",
                "get_task_result",
                "get_task_report",
            }
        ),
        "Проверяй именно запрошенную версию объектов. Для проектных слоёв используй "
        "upload_layer и submit_building_pzz_check_task. Для сохранённого сценария допустим "
        "classify_scenario_and_wait. Не считай queued/running/timed_out готовым результатом. "
        "После создания задачи запроси статус и отчёт. action=confirm или suggest_upload "
        "требует уточнения пользователя; не придумывай confirmed_zone_map. "
        "Различай функциональное зонирование и юридические территориальные зоны ПЗЗ. "
        "Встроенный приблизительный шаблон — только предварительная оценка. "
        "unknown, no_mapping и объекты вне зон не являются соответствием.",
    ),
}


VARIANT_PROVISION = PlanningProfile(
    "provision",
    "Агент обеспеченности проектного варианта",
    "Рассчитывает обеспеченность по проектным зданиям и предложенным услугам без сохранения сценария.",
    frozenset({"CalculateServicesProvision", "CalculateVariantServicesProvision"}),
    "Для исходного сценария используй CalculateServicesProvision. Для нового варианта обязательно "
    "CalculateVariantServicesProvision с полным generated_buildings и/или additional_services через $artifact. "
    "Существующие здания и услуги сохраняются из Urban автоматически. target_population — ОБЩЕЕ население "
    "сценария с существующими жителями, не только новое население. Укажи это допущение в ответе. "
    "ID услуг получи через GetScenarioServiceTypes, as_layer=true. Для сравнения сохрани исходный и новый "
    "результат, сопоставь дефицит, спрос, вместимость и доступность. Новый объект можно предложить через "
    "propose_service только на выбранной реальной площадке с обоснованной вместимостью. "
    "Не выдавай точку размещения за проект здания. Нельзя подменять вариант расчётом старого сценария.",
    required_tools=frozenset({"CalculateVariantServicesProvision"}),
)
