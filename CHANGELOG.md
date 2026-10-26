## v0.28.2 (2026-10-26)

[fix/ICII-205-critic-latency](https://github.com/IDUclub/gMART/pull/252) (#252)

- fix(agents): ускорить проверку ответов критиком агента документов

## v0.28.1 (2026-10-26)

[chore/prod-default-llm-model](https://github.com/IDUclub/gMART/pull/251) (#251)

- fix(build_and_deploy_all): передавать DEFAULT_LLM_MODEL в прод-контейнеры

## v0.28.0 (2026-10-26)

[feat/ICII-205-img-processing](https://github.com/IDUclub/gMART/pull/250) (#250)

- feat(agents): показывать рисунки из документов в ответах агента документов

## v0.27.0 (2026-10-23)

[feat/ICII-151-doc-comparasion](https://github.com/IDUclub/gMART/pull/248) (#248)

- feat(agents): сравнивать документы и редакции документа в агенте документов

## v0.26.2 (2026-10-23)

[fix/ICII-208-territory-descendants](https://github.com/IDUclub/gMART/pull/247) (#247)

- fix(agents): не считать действующими документы территорий ниже найденных по границе файлов

## v0.26.1 (2026-10-23)

[fix/provision-effects-all-services-dev](https://github.com/IDUclub/gMART/pull/246) (#246)

- fix(agents): сравнивать эффекты обеспеченности по всем сервисам сценария
- fix(agents): описывать сравнение эффектов как сравнение выбранного сценария с базовым
- fix(agents): не сравнивать базовый сценарий проекта с самим собой

## v0.26.0 (2026-10-22)

[feat/ICII-208-do-scope](https://github.com/IDUclub/gMART/pull/243) (#243)

- feat(agents): загружать слои MapInfo, MIF/MID, GML, KML и 7z с классами ФГИС ТП

## v0.25.2 (2026-10-22)

[fix/sensitive-data-leaks](https://github.com/IDUclub/gMART/pull/244) (#244)

- fix: не раскрывать токены в ответах 500 и логах, 401 вместо 500 для несервисного токена в idu_mcp

## v0.25.1 (2026-10-20)

[fix/ICII-190-docker-log-rotation](https://github.com/IDUclub/gMART/pull/242) (#242)

- fix(deploy): ограничить docker-логи контейнеров 100 МБ
- fix(logging): раз в день ротировать файловые логи по размеру, не больше 100 МБ

## v0.25.0 (2026-10-19)

[feat/ICII-189-user-files-crs](https://github.com/IDUclub/gMART/pull/241) (#241)

- feat(agents): принимать файлы пользователя в МСК-47 без системы координат
- feat(agents): хранить слои загрузок в файловом кэше, а не в Redis
- feat(agents): распознавать EPSG:3857 без системы координат, остальные файлы возвращать
- feat(deploy): хранить кэш слоёв загрузок в томе, чтобы загрузки переживали деплой
- fix(tests): задавать сервисную авторизацию по умолчанию в тестах

## v0.24.1 (2026-10-19)

[fix/ICII-209-project-from-user-file](https://github.com/IDUclub/gMART/pull/240) (#240)

- fix(agents): создавать проект из файлов от имени пользователя

## v0.24.0 (2026-10-19)

[feat/ICII-209-project-from-user-file](https://github.com/IDUclub/gMART/pull/239) (#239)

- feat(agents): создавать проект и сценарий в Urban API из файлов пользователя

## v0.23.1 (2026-10-17)

[fix/ICII-208-user-file-compilance](https://github.com/IDUclub/gMART/pull/238) (#238)

- fix(agents): не задавать лишних вопросов при проверке файлов и не принимать объекты за функциональные зоны

## v0.23.0 (2026-10-17)

[feat/ICII-208-user-file-compilance](https://github.com/IDUclub/gMART/pull/237) (#237)

- fix: запускать тесты и локальные сервисы на Windows
- feat(agents): проверять файлы пользователя на соответствие нормам

## v0.22.1 (2026-10-15)

[perf/documents-quality](https://github.com/IDUclub/gMART/pull/236) (#236)

- feat(evaluation): добавить набор из 120 вопросов с эталонными ответами по корпусу DVD и автоматическую оценку качества ответов агента документов
- perf(agents): планировать поиск с коротким рассуждением и ограничить длину плана, при упоре в лимит искать по самому вопросу
- fix(agents): писать в ссылке на источник номер пункта, статьи, приложения или таблицы вместо названия раздела, совпадающего с текстом фрагмента
- feat(evaluation): показывать в оценке набора число сгенерированных токенов и вызовов LLM на ответ, не зависящее от загрузки общего сервера модели
- fix(evaluation): принимать в A-03 и D-01 равнозначные формулировки фактов «не должен превышать» и «≤ 15»
- perf(agents): проверять ответ агента документов коротким рассуждением, полным — только ответы с выводом об отсутствии сведений во фрагментах
- fix(agents): перепроверять ответ агента документов полным рассуждением, если короткая проверка не подтвердила ни одной строки
- fix(evaluation): засчитывать отказ с обратным порядком слов «сведений о … нет»
- chore(deploy): передавать в контейнер агентов настройки планировщика и полной проверки критика агента документов

## v0.22.0 (2026-10-12)

[feat/grounded-plus-knowledge](https://github.com/IDUclub/gMART/pull/235) (#235)

- feat(agents): отвечать сначала по фрагментам, затем общими знаниями модели под пометкой «Без опоры на загруженные документы» и не отсылать к внешним источникам
- feat(agents): ссылаться в ответе на документ и пункт вместо номера фрагмента, писать блок без опоры на документы как предположения и ограничить длину проверки критиком

## v0.21.3 (2026-10-08)

[ref/fasten-rag](https://github.com/IDUclub/gMART/pull/234) (#234)

- feat(agents): замерять этапы, вызовы LLM и решения пайплайна агента документов и сравнивать версии бенчмарком задержки
- feat(agents): проверять критиком простые ответы агента документов с низким усилием рассуждения, а рискованные — с полным
- feat(agents): искать пункт с явно названным документом без вызова планировщика LLM и передавать запрос планировщику, если такой пункт не найден
- feat(agents): создавать чат агента документов параллельно с ответом под временным названием и переименовывать его после ответа
- chore(agents): заменить точные ссылки бенчмарка задержки документами из общей базы dev-контура
- feat(agents): проверять критиком ответ по уже подготовленным источникам без повторного сокращения контекста, а при нехватке окна — по процитированным
- feat(agents): начинать смысловой поиск агента документов с узкой выборки и расширять её, только если критику не хватает источников
- feat(agents): держать одну MCP-сессию на запрос к документам, переиспользовать HTTP-соединения и кэшировать метаданные модели и подсчёт токенов
- fix(agents): сообщать о созданном чате до ошибки запроса к документам
- chore(agents): передавать в контейнер переключатели ускорения агента документов и описать их в настройках
- fix: changed instructions for rag service

## v0.21.2 (2026-10-02)

[fix/version-status-token](https://github.com/IDUclub/gMART/pull/233) (#233)

- ci: ставить итоговый статус version токеном VERSION_STATUS_TOKEN, чтобы мердж запускал выкатку на dev

## v0.21.1 (2026-10-02)

[chore/versioning-policy](https://github.com/IDUclub/gMART/pull/232) (#232)

- ci: поднимать версию при каждом мердже в dev, релиз в main только ставит тег
- ci: поднимать версию в ветке PR по включению auto-merge

## v0.21.0 (2026-09-25)

### Feat

- **agents**: растягивать сроки пайплайнов, пока сервер LLM отвечает медленнее обычного (#228)

### Fix

- **agents**: проверять все объекты сценария без ограничений на их число (#227)

## v0.20.0 (2026-09-25)

### Feat

- **agents**: показывать ограничения на территории сценария и применять только действующие на ней нормы (#225)
- **agents**: сужать проверку соответствия темой и документом с выбором документа в диалоге (#221)
- (#213)

### Fix

- **agents**: дополнять фрагменты документов, оборванные на полуслове (#224)
- **agents**: ограничивать выход модели долей от входа и исправлять ответ точечными правками критика (#223)
- **agents**: находить тему по слоям планов и не запускать полную проверку по номеру без списка (#222)
- **agents**: показывать объекты без нарушений и не сохранять пустые слои проверки соответствия (#220)
- **agents**: не выводить системные идентификаторы в ответах агентов (#219)
- **agents**: не пропускать шаги оркестратора после сбоя одного из них (#217)
- **agents**: переходить к текстовому поиску норм и считать несколько типов в scenario-data (#216)
- **agents**: ускорить document-QA и исправить ошибки прогона на dev (#215)

## v0.19.5 (2026-09-22)

### Fix

- **agents**: исправить сбои обработки документов и обеспеченности (#211)

## v0.19.4 (2026-09-21)

### Fix

- **agents**: сократить выдаваемые слои и их атрибуты (#209)
- cite compliance documents and clauses instead of norm IDs (#208)
- respect configured Urban API roots behind load balancers (#206)
- deduplicate compliance checks and request full geometry (#203)

## v0.19.3 (2026-09-16)

### Fix

- (#200)
- allow reading server configuration without a password (#198)
- **agents**: avoid redundant document retries and keep failure details in logs (#196)

## v0.19.2 (2026-09-16)

### Fix

- **scenario-data**: summarise indicator answers instead of listing rows
- persist only compliance tool calls and text (#193)

## v0.19.1 (2026-09-15)

### Fix

- preserve document QA context and accept grounded concise answers (#190)
- resolve exact document references and compact source context (#187)

## v0.19.0 (2026-09-14)

### Feat

- **scenario-data**: show Russian tables without ids and summarize comparisons
- retrieve document structure and reduce large contexts in parallel (#171)

### Fix

- preserve document clarifications and recover interrupted streams (#183)
- resolve and rank document clause clarifications (#182)
- preserve document choices across RAG clarification turns (#181)
- validate DVD retrieval plan modes (#180)
- adapt RAG answer budgets and resume truncated drafts (#177)
- ground document answers and detect deployed model context (#175)
- budget RAG summaries and reject incomplete evidence (#174)
- align scoped RAG parameters with DVD MCP contract (#173)

## v0.18.1 (2026-09-08)

### Fix

- harden scenario data harness results

## v0.18.0 (2026-08-21)

### Feat

- **documents**: manage user documents in test UI (#153)

### Fix

- ci-cd skip changes in tests action (#156)
- preserve named type mapping values (#152)

## v0.17.4 (2026-08-19)

### Fix

- pass user context outside m2m transports (#149) (#150)
- pass user context outside m2m transports (#149)

## v0.17.3 (2026-08-19)

## v0.17.2 (2026-08-19)

### Fix

- updated think parametres (#144) (#145)
- updated think parametres (#144)

## v0.17.1 (2026-08-19)

### Fix

- **ui**: generate ids without secure-context crypto.randomUUID (#142) (#143)
- **ui**: generate ids without secure-context crypto.randomUUID (#142)

## v0.17.0 (2026-08-18)

### Feat

- **auth**: use service tokens for internal calls (#138)

### Fix

- **restrictions**: keep NormGraph out of geometry pipeline (#136)

## v0.16.0 (2026-08-17)

### Feat

- **llm**: resolve the default model from the connected provider

### Fix

- **llm**: document Harmony's real reasoning_effort set and cover it with tests

## v0.15.0 (2026-08-17)

### Feat

- **ui**: add MCP diagnostics console (#128)
- **scenario-data**: add optional-context A2A agent (#127)

### Refactor

- **frontend**: simplify workspace UI (#129)

## v0.14.1 (2026-07-14)

### Fix

- **norms_rag**: recursively normalize NormGraph MCP results (#120)
- **chat_storage**: tolerate new ChatStorage response fields (#119)

## v0.14.0 (2026-07-13)

### Feat

- **orchestrator**: add orchestrator agent — single entry point routing requests across all agents (#113)
- refactored ui (#112)
- **norms_rag**: add NormGraph graph-RAG agent for normative restrictions (#111)
- **version**: - fix rag-response parsing - upgraded version to 0.13.5
- **version**: - updated app-version - upgraded version to 0.13.4
- **version**: - updated app-version - upgraded version to 0.13.4
- **ci**: (#101) (#102)
- **ci**: (#101)
- **toml**: - updated toml dependencies
- **toml**: - updated toml dependencies
- **toml**: - updated toml redis dependencies
- **toml**: - updated toml version to 0.12.0
- **dvd_rag_service**: (#82)
- **pyproj.toml**: - updated project deps
- **config-protection**: - added config-protection via password from secrets
- **pipeline**: - added system password
- **pipeline**: - added pipeline logs
- **pipeline**: - added agents env for pipeline - changed version to 0.10.3
- **version**: - added project_id update for new chat - upgraded version to 0.10.2
- **version**: - added project_id update for new chat - upgraded version to 0.10.2
- **pipeline**: - added agents env for pipeline - changed version to 0.10.3
- **pipeline**: - added agents env for pipeline - changed version to 0.10.3
- **version**: - added project_id update for new chat - upgraded version to 0.10.2
- **system**: (#72)
- **geom_interface**: - version 0.10.0 - moved feature collections to layers and objects arguments - added logs - added simple tool call error handling (will be revised in the future) - changed restriction pipeline extraction
- **provision-agent**: - added provision agent - upgraded version to 0.6.0

### Fix

- **chat_history**: (#105)
- **a2a_controller**: (#95)
- **a2a_service**: (#86)
- **lock**: - updated lock file
- **build_and_deploy_all**: - added DVD_MCP_SERVER to build_and_deploy_all.yaml pipeline
- **sse_executor**: (#80)
- **app_config_response**: - added app configuration response
- **provision_plan_builder**: - fixed scenario and project understanding
- **json_api_handler**: - fixed exception handling
- **a2a**: - a2a adoption
- **a2a**: - a2a adoption
- **build_and_deploy.yaml**: - removed unnecessary ssl disable in actions
- **merge**: - after-merge fixes from main

### Refactor

- **provision_service**: (#63)
- **provision_service**: - removed project_id from required params for provision service requests
- **meta-params**: - meta params moved to tools args

## v0.4.2 (2026-05-14)

### Feat

- **docs**: - updated front-end docs
- **pipeline-storage**: - added pipeline storage - added request_id for restriction pipeline - added bearer token expiration handling during pipeline - upgraded version to 0.5.0
- **restrictions**: - added strict plan matching - chat history context - catalog clarification
- **docs**: - added ai generated readme
- **chat_history**: - added chat history saving - updated docker files
- **compose**: - compose in progress
- **restriction_parser_service**: - added message part saving on generation
- **all-files**: - added chat storage client methods - updated dto models - added title generation - chat saving in progress
- **chat_storage_client**: - created chut storage client - added api client structure for api clients - chat storage api client methods in-progress
- **a2a**: - added a2a support - upgraded toml - upgraded version to 0.3.0
- **restriction_parser_service**: (#40)
- **restriction_parser_service**: - AI-refactored (CODEX) pipeline to planning system - Agents version upgraded to 0.2.0 - gMART version upgraded to 0.2.0
- **gMART 0.1.8**: - added retry generation on pipeline and response generation on error with instruction - added original user response to reformulated user query
- **agent-logs**: - added logging to agents app
- **gMart**: - upgraded gMART version to 0.1.3
- **agents: 0.1.3**: - upgraded agents version to 0.1.3
- **agents**: - added optional parameter temperature to llm request - added user request explanation to restriction agent
- **logs**: - added tiny log
- **agents: 0.1.1**: (#23)
- **agents: 0.1.1**: - added error handling within sse executor.
- **0.1.0**: - first version of gMART services
- **agents**: (#21)
- **agents**: - added values system to ollama requests - fixed sse iteration ending
- **agents: 0.0.7, idu_mcp: 0.0.8**: - agents: added context formation and final response generation - idu_mcp: made all output layers in wgs84 (4326)
- **agents: 0.0.6**: - made restriction formation optional
- **agents: 0.0.5**: - added tools params self-explain for model choice after tool call
- **idu_mcp: 0.0.7**: - added object name, buffer size and restriction title for buffers created objects
- **agents: 0.0.4, idu_mcp: 0.0.6**: - test-ready service with min working pipeline
- **pipline in progress**: - restructions debugging
- **pipline in progress**: - main pipline mostly finished
- **pipline in progress**: - main pipline in progess
- **restriction_parser_service**: - pipeline in progress
- **0.0.1**: test api interface

### Fix

- **pipeline-fix**: - second-try
- **pipeline-fix**: - first try to fix pipeline
- **pipeline-fix**: - second-try
- **sse_executors**: - fixed sse execution error
- **idu_mcp_auth**: - fixed bearer extraction from mcp
- **response_context**: - fixed final generation - upgraded gMART version to 0.1.4
- **example**: - fixed url prefixes
- **exception_handling**: - fixed type handling again
- **exception_handling**: - fixed type handling
- **idu_mcp: 0.0.5, agents: 0.0.3**: - fixed prompts response - fixed agents prompts parsing
- **idu_mcp: 0.0.4, agents: 0.0.2**: - added prompts service to fastmcp - refactored agents services
- **idu_mcp: 0.0.3**: fixed geometry_instruments calls
- **0.0.2**: fixed instrument calls
- **idu_mcp: 0.0.4, agents: 0.0.2**: - added prompts service to fastmcp - refactored agents services
- **idu_mcp: 0.0.3**: fixed geometry_instruments calls
- **0.0.2**: fixed instrument calls
