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
