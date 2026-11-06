"""Urban API calls that create the project, its scenario and the objects."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from src.agents.common.exceptions.api_exceptions import DownstreamServiceError
from src.agents.common.exceptions.base_exceptions import (
    AgentsBaseException,
    AgentsNotFound,
    AgentsUnauthorizedException,
)
from src.agents.services.compilance.compliance_uploads import UploadedLayer, UserFiles
from src.agents.services.project_files.project_creation import (
    READINESS_RETRY_DELAYS,
    ProjectCreator,
    build_items,
    creation_summary,
    describe_error,
    failed_objects,
)
from src.agents.services.project_files.project_types import CreationSpec

BORDER = {
    "type": "Polygon",
    "coordinates": [
        [[37.0, 55.0], [37.01, 55.0], [37.01, 55.01], [37.0, 55.01], [37.0, 55.0]]
    ],
}


def square(x: float, y: float, size: float = 0.001) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
        ],
    }


SPECS = {
    "type=buildings|building=residential": CreationSpec(
        "object", 4, "Жилой дом", building=True
    ),
    "type=playground": CreationSpec(
        "object", 7, "Площадка", service_type_id=70, service_name="Детская площадка"
    ),
    "type=block|blockType=residential": CreationSpec("zone", 1, "residential"),
}


def project_files() -> UserFiles:
    features = [
        {
            "type": "Feature",
            "id": "h1",
            "geometry": square(37.001, 55.001),
            "properties": {
                "type": "buildings",
                "building": "residential",
                "levels": "9",
                "levelHeights": "3,3,3,3,3,3,3,3,3",
                "building:total_area": 5000.123,
                "livingSpace": "4000",
                "population": "80",
            },
        },
        {
            "type": "Feature",
            "geometry": square(37.003, 55.003),
            "properties": {"type": "playground", "capacity": "30"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [37.005, 55.005]},
            "properties": {"type": "block", "blockType": "residential"},
        },
        {
            "type": "Feature",
            "geometry": square(37.004, 55.004),
            "properties": {"type": "calm_rest"},
        },
    ]
    return UserFiles(
        upload_ids=("u",),
        layers=[
            UploadedLayer(
                name="Проект",
                role="project",
                feature_collection={"type": "FeatureCollection", "features": features},
            ),
            UploadedLayer(
                name="Граница",
                role="border",
                feature_collection={
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "geometry": BORDER, "properties": {}}
                    ],
                },
            ),
        ],
    )


def test_items_carry_building_service_and_source():
    zones, objects, rejected = build_items(project_files(), SPECS)

    assert zones == []
    assert rejected == {"некорректная геометрия": 1}  # a block given as a point
    house, playground = objects
    assert house.key == "Проект/0" and house.source_id == "h1"
    assert house.building == {
        "floors": 9,
        "building_area_modeled": 5000.12,
        "properties": {"height": 27.0, "living_area": 4000.0, "population": 80.0},
    }
    assert house.properties["source"] == "gMART"
    assert house.properties["source_layer"] == "Проект"
    assert playground.capacity == 30
    assert playground.label == "type=playground"


class FakeUrban:
    def __init__(self, fail_keys=(), building_id=None, service_404=0):
        self.calls: list[tuple] = []
        self.fail_keys = set(fail_keys)
        self.building_id = building_id
        self.service_404 = service_404
        self.next_id = 100
        self.existing = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": square(37.002, 55.002),
                    "properties": {
                        "physical_object_id": 1,
                        "physical_object_type": {"physical_object_type_id": 4},
                        "is_scenario_physical_object": False,
                    },
                },
                {
                    "type": "Feature",
                    "geometry": square(37.002, 55.002),
                    "properties": {
                        "physical_object_id": 1,
                        "physical_object_type": {"physical_object_type_id": 4},
                    },
                },
                {
                    "type": "Feature",
                    "geometry": square(37.007, 55.007),
                    "properties": {
                        "physical_object_id": 4,
                        "physical_object_type": {"physical_object_type_id": 4},
                        "is_locked": True,
                    },
                },
                {
                    "type": "Feature",
                    "geometry": square(37.006, 55.006),
                    "properties": {
                        "physical_object_id": 2,
                        "physical_object_type": {"physical_object_type_id": 30},
                    },
                },
                {
                    "type": "Feature",
                    "geometry": square(38.0, 56.0),
                    "properties": {
                        "physical_object_id": 3,
                        "physical_object_type": {"physical_object_type_id": 4},
                    },
                },
            ],
        }

    async def create_project(self, token, project, user_id=None):
        self.calls.append(("create_project", project["name"], user_id))
        return {"project_id": 10, "base_scenario": {"id": 11, "name": "base"}}

    async def copy_scenario(self, token, scenario_id, scenario):
        self.calls.append(
            ("copy_scenario", scenario_id, scenario["functional_zone_type_id"])
        )
        return {"scenario_id": 12}

    async def get_project(self, token, project_id):
        assert project_id == 10
        return {"project_id": project_id}

    async def get_project_territory(self, token, project_id):
        assert project_id == 10
        return {"geometry": BORDER}

    async def get_project_scenarios(self, token, project_id):
        assert project_id == 10
        return [{"scenario_id": 11, "is_based": True}, {"scenario_id": 12}]

    async def get_scenario_physical_objects(self, token, scenario_id):
        assert scenario_id == 12
        return {
            **self.existing,
            "features": self.existing["features"]
            + [
                {"properties": {"physical_object_id": object_id}}
                for object_id in range(101, self.next_id + 1)
            ],
        }

    async def delete_scenario_physical_object(
        self, token, scenario_id, physical_object_id, is_scenario_object
    ):
        self.calls.append(("delete", physical_object_id, is_scenario_object))

    async def delete_scenario_functional_zones(self, token, scenario_id):
        self.calls.append(("delete_zones", scenario_id))

    async def add_scenario_functional_zones(self, token, scenario_id, zones):
        self.calls.append(("zones", len(zones)))
        return []

    async def add_scenario_physical_object(self, token, scenario_id, payload):
        name = payload["properties"]["source_id"]
        self.calls.append(("object", name, payload["physical_object_type_id"]))
        if name in self.fail_keys:
            raise DownstreamServiceError(
                service="urban",
                downstream_status=422,
                message="bad",
                error_input={"detail": "invalid geometry"},
            )
        self.next_id += 1
        return {
            "physical_object": {
                "physical_object_id": self.next_id,
                "building": {"id": self.building_id} if self.building_id else None,
            },
            "object_geometry": {"object_geometry_id": self.next_id + 1000},
        }

    async def add_scenario_building(self, token, scenario_id, building):
        self.calls.append(
            ("building", building["physical_object_id"], building["floors"])
        )

    async def patch_scenario_building(self, token, scenario_id, building_id, building):
        self.calls.append(("patch_building", building_id, building["floors"]))

    async def add_scenario_service(self, token, scenario_id, service):
        if self.service_404:
            self.service_404 -= 1
            raise AgentsNotFound("not yet")
        self.calls.append(("service", service["service_type_id"], service["capacity"]))


def run_state(policy="replace_same") -> dict:
    return {
        "owner": "alice",
        "project_name": "Сад",
        "scenario_name": "Сценарий",
        "region_id": 3,
        "region_name": "Москва",
        "profile_id": 13,
        "profile_name": "Многоэтажная",
        "existing_policy": policy,
        "boundary": BORDER,
        "files": ["Проект"],
        "year": 2026,
    }


async def collect(creator, run, zones, objects, token="t", owner="alice"):
    saves = []

    async def save():
        saves.append(dict(run))

    events = [
        event
        async for event in creator.create(token, run, zones, objects, save, owner=owner)
    ]
    return events, saves


async def no_sleep(_):
    return None


async def test_legacy_run_persists_request_owner_before_creation():
    urban = FakeUrban()
    run = run_state("keep")
    run.pop("owner")

    _, saves = await collect(
        ProjectCreator(urban), run, [], [], token="unreadable", owner="alice"
    )

    assert saves[0]["owner"] == "alice"
    assert "project_id" not in saves[0]
    assert urban.calls[0] == ("create_project", "Сад", "alice")


async def test_resumed_run_keeps_owner_when_request_token_changes():
    class Urban(FakeUrban):
        unavailable = True

        async def create_project(self, token, project, user_id=None):
            if self.unavailable:
                raise AgentsNotFound("later")
            return await super().create_project(token, project, user_id=user_id)

    urban = Urban()
    run = run_state("keep")
    run.pop("owner")
    creator = ProjectCreator(urban)
    with pytest.raises(AgentsNotFound):
        await collect(creator, run, [], [], owner="alice")
    assert run["owner"] == "alice"

    urban.unavailable = False
    await collect(creator, run, [], [], token="refreshed-token", owner="alice")

    assert urban.calls[0] == ("create_project", "Сад", "alice")
    assert run["owner"] == "alice"


@pytest.mark.parametrize("resumed", [False, True])
async def test_foreign_owner_rejected_without_urban_calls_or_state_changes(resumed):
    urban = AsyncMock()
    save = AsyncMock()
    run = run_state("keep")
    if resumed:
        run.update(project_id=10, base_scenario_id=11, scenario_id=12)
    before = deepcopy(run)

    with pytest.raises(AgentsUnauthorizedException):
        async for _ in ProjectCreator(urban).create(
            "token", run, [], [], save, owner="bob"
        ):
            pass

    assert not urban.mock_calls
    save.assert_not_awaited()
    assert run == before


@pytest.mark.parametrize("response_owner", [None, "alice"])
async def test_matching_or_absent_response_owner_allows_creation(response_owner):
    class Urban(FakeUrban):
        async def create_project(self, token, project, user_id=None):
            response = await super().create_project(token, project, user_id=user_id)
            if response_owner is not None:
                response["user_id"] = response_owner
            return response

        async def get_project(self, token, project_id):
            response = await super().get_project(token, project_id)
            if response_owner is not None:
                response["user_id"] = response_owner
            return response

    run = run_state("keep")
    events, _ = await collect(ProjectCreator(Urban()), run, [], [])

    assert run["ready"]["done"]
    assert events[-1]["type"] == "project_ready"


@pytest.mark.parametrize("source", ["post", "get", "resume"])
async def test_wrong_owner_stops_before_scenario_or_object_writes(source):
    class Urban(FakeUrban):
        async def create_project(self, token, project, user_id=None):
            response = await super().create_project(token, project, user_id=user_id)
            if source == "post":
                response["user_id"] = "bob"
            return response

        async def get_project(self, token, project_id):
            response = await super().get_project(token, project_id)
            return {**response, "user_id": "bob"}

    urban = Urban()
    run = run_state()
    if source == "resume":
        run.update(project_id=10, base_scenario_id=11, scenario_id=12)
    _, objects, _ = build_items(project_files(), SPECS)

    with pytest.raises(AgentsBaseException, match="проект другого владельца"):
        await collect(ProjectCreator(urban, sleep=no_sleep), run, [], objects)

    assert all(call[0] == "create_project" for call in urban.calls)
    assert run["project_id"] == 10  # keep the id so a retry cannot create a duplicate
    assert "cleanup" not in run and "objects" not in run


async def test_wrong_owner_at_final_confirmation_is_not_ready():
    class Urban(FakeUrban):
        reads = 0

        async def get_project(self, token, project_id):
            self.reads += 1
            response = await super().get_project(token, project_id)
            return {**response, "user_id": "alice" if self.reads == 1 else "bob"}

    run = run_state("keep")
    events, _ = await collect(ProjectCreator(Urban()), run, [], [])

    assert not run["ready"]["done"]
    assert "проект другого владельца" in run["ready"]["error"]
    assert not any(event["type"] == "project_ready" for event in events)


async def test_full_creation_replaces_same_types_inside_the_border():
    urban = FakeUrban(service_404=2)
    run = run_state()
    _, objects, _ = build_items(project_files(), SPECS)

    events, saves = await collect(
        ProjectCreator(urban, sleep=no_sleep), run, [], objects
    )

    names = [call[0] for call in urban.calls]
    assert urban.calls[0] == ("create_project", "Сад", "alice")
    assert ("copy_scenario", 11, 13) in urban.calls
    # Only the house of the same type inside the border, once; the locked one is
    # kept and counted.
    assert [call for call in urban.calls if call[0] == "delete"] == [
        ("delete", 1, False)
    ]
    assert run["cleanup"]["locked"] == 1
    assert creation_summary(run, [], objects)["locked_existing"] == 1
    assert "delete_zones" not in names
    assert ("building", 101, 9) in urban.calls
    assert ("service", 70, 30) in urban.calls
    assert [event["type"] for event in events][:3] == [
        "status",
        "project_created",
        "status",
    ]
    assert run["project_id"] == 10 and run["scenario_id"] == 12
    assert saves and failed_objects(objects, run) == []


async def test_existing_building_is_patched_and_keep_deletes_nothing():
    urban = FakeUrban(building_id=555)
    run = run_state("keep")
    _, objects, _ = build_items(project_files(), SPECS)

    await collect(ProjectCreator(urban), run, [], objects)

    names = [call[0] for call in urban.calls]
    assert "delete" not in names and "delete_zones" not in names
    assert ("patch_building", 555, 9) in urban.calls


async def test_delete_all_removes_every_object_and_zone_inside():
    urban = FakeUrban()
    run = run_state("delete_all")
    zone_specs = {**SPECS}
    user_files = project_files()
    user_files.layers[0].features[2]["geometry"] = square(37.005, 55.005)
    zones, objects, _ = build_items(user_files, zone_specs)

    await collect(ProjectCreator(urban), run, zones, objects)

    assert [call[1] for call in urban.calls if call[0] == "delete"] == [1, 2]
    assert ("delete_zones", 12) in urban.calls
    assert ("zones", 1) in urban.calls


async def test_retry_creates_only_what_failed():
    urban = FakeUrban(fail_keys={"h1"})
    run = run_state("keep")
    zones, objects, _ = build_items(project_files(), SPECS)
    creator = ProjectCreator(urban)

    await collect(creator, run, zones, objects)
    failures = failed_objects(objects, run)
    assert [(item.key, error) for item, error in failures] == [
        ("Проект/0", '422: {"detail": "invalid geometry"}')
    ]
    summary = creation_summary(run, zones, objects)
    assert summary["objects"]["saved"] == 1 and summary["failed_total"] == 1

    urban.fail_keys.clear()
    urban.calls.clear()
    await collect(creator, run, zones, objects)

    assert [call[0] for call in urban.calls] == ["object", "building"]
    assert creation_summary(run, zones, objects)["failed_total"] == 0


def test_error_text_names_the_status():
    error = DownstreamServiceError(
        service="urban", downstream_status=409, message="m", error_input="exists"
    )

    assert describe_error(error) == "409: exists"
    assert describe_error(ValueError("boom")) == "ValueError: boom"


async def test_base_scenario_found_when_the_response_has_none():
    class Urban(FakeUrban):
        async def create_project(self, token, project, user_id=None):
            return {"project_id": 10}

        async def get_project_scenarios(self, token, project_id):
            return [
                {"scenario_id": 20, "is_based": False},
                {"scenario_id": 21, "is_based": True},
                {"scenario_id": 12},
            ]

    urban = Urban()
    run = run_state("keep")

    await collect(ProjectCreator(urban), run, [], [])

    assert run["base_scenario_id"] == 21
    assert ("copy_scenario", 21, 13) in urban.calls


class DelayedUrban(FakeUrban):
    def __init__(
        self,
        *,
        project_errors=(),
        scenario_waits=0,
        object_waits=0,
        base_waits=0,
        territory_waits=0,
    ):
        super().__init__()
        self.project_errors = list(project_errors)
        self.scenario_waits = scenario_waits
        self.object_waits = object_waits
        self.base_waits = base_waits
        self.territory_waits = territory_waits
        self.copied = False
        self.project_reads = 0

    async def get_project(self, token, project_id):
        assert token == "t"
        self.project_reads += 1
        if self.project_errors:
            raise self.project_errors.pop(0)
        return await super().get_project(token, project_id)

    async def copy_scenario(self, token, scenario_id, scenario):
        assert not self.project_errors
        self.copied = True
        return await super().copy_scenario(token, scenario_id, scenario)

    async def get_project_scenarios(self, token, project_id):
        assert token == "t"
        if not self.copied and self.base_waits:
            self.base_waits -= 1
            return []
        if self.copied and self.scenario_waits:
            self.scenario_waits -= 1
            return [{"scenario_id": 11, "is_based": True}]
        return await super().get_project_scenarios(token, project_id)

    async def get_project_territory(self, token, project_id):
        assert token == "t"
        if self.territory_waits:
            self.territory_waits -= 1
            return {"geometry": None}
        return await super().get_project_territory(token, project_id)

    async def get_scenario_physical_objects(self, token, scenario_id):
        assert token == "t"
        assert not self.scenario_waits  # cleanup/uploads must wait for visibility
        if self.object_waits:
            self.object_waits -= 1
            return {"features": []}
        return await super().get_scenario_physical_objects(token, scenario_id)


async def test_project_reads_retry_404_with_exponential_delays():
    urban = DelayedUrban(project_errors=[AgentsNotFound("later")] * 2)
    delays = []

    async def sleep(delay):
        delays.append(delay)

    run = run_state("keep")
    await collect(ProjectCreator(urban, sleep=sleep), run, [], [])
    assert delays == [0.5, 1.0]
    assert run["ready"]["done"]
    assert sum(call[0] == "copy_scenario" for call in urban.calls) == 1


async def test_scenario_and_objects_wait_then_ready_is_persisted():
    urban = DelayedUrban(scenario_waits=2, object_waits=2)
    delays = []

    async def sleep(delay):
        delays.append(delay)

    _, objects, _ = build_items(project_files(), SPECS)
    run = run_state("keep")
    creator = ProjectCreator(urban, sleep=sleep)
    events, saves = await collect(creator, run, [], objects)
    assert delays == [0.5, 1.0, 0.5, 1.0]
    assert events[-1] == {
        "type": "project_ready",
        "content": {
            "project_id": 10,
            "scenario_id": 12,
            "base_scenario_id": 11,
            "objects_confirmed": 2,
        },
    }
    assert run["ready"]["attempts"] == 3
    assert saves[-1]["ready"]["checked_at"]
    assert creation_summary(run, [], objects)["ready"] is True
    reads = urban.project_reads
    await collect(creator, run, [], objects)
    assert urban.project_reads == reads + 1  # recheck ownership before resuming


async def test_never_ready_is_bounded_and_retry_only_rechecks():
    urban = DelayedUrban(territory_waits=100)
    delays = []

    async def sleep(delay):
        delays.append(delay)

    run = run_state("keep")
    creator = ProjectCreator(urban, sleep=sleep)
    events, _ = await collect(creator, run, [], [])
    assert delays == list(READINESS_RETRY_DELAYS)
    assert run["ready"]["attempts"] == len(READINESS_RETRY_DELAYS) + 1
    assert not any(event["type"] == "project_ready" for event in events)
    summary = creation_summary(run, [], [])
    assert summary["ready"] is False
    assert "Геометрия" in summary["ready_error"]
    urban.territory_waits = 0
    urban.calls.clear()
    events, _ = await collect(creator, run, [], [])
    assert urban.calls == []
    assert run["ready"]["done"]
    assert events[-1]["type"] == "project_ready"


@pytest.mark.parametrize("status", [401, 403, 422])
async def test_non_transient_read_errors_do_not_retry(status):
    error = DownstreamServiceError("urban", status, "access denied")
    urban = DelayedUrban(project_errors=[error])
    delays = []

    async def sleep(delay):
        delays.append(delay)

    with pytest.raises(DownstreamServiceError):
        await collect(ProjectCreator(urban, sleep=sleep), run_state("keep"), [], [])
    assert delays == []
    assert not urban.copied


@pytest.mark.parametrize("status", [None, 502, 503])
async def test_transient_read_errors_retry(status):
    urban = DelayedUrban(
        project_errors=[DownstreamServiceError("urban", status, "later")]
    )
    delays = []

    async def sleep(delay):
        delays.append(delay)

    await collect(ProjectCreator(urban, sleep=sleep), run_state("keep"), [], [])
    assert delays == [0.5]


async def test_base_scenario_resolution_waits_for_complete_data():
    class Urban(DelayedUrban):
        async def create_project(self, token, project, user_id=None):
            return {"project_id": 10}

    urban = Urban(base_waits=2)
    delays = []

    async def sleep(delay):
        delays.append(delay)

    run = run_state("keep")
    await collect(ProjectCreator(urban, sleep=sleep), run, [], [])
    assert delays == [0.5, 1.0]
    assert run["base_scenario_id"] == 11


@pytest.mark.parametrize("status", [401, 403])
async def test_final_confirmation_denied_is_saved_without_retries(status):
    class Urban(DelayedUrban):
        async def get_project_territory(self, token, project_id):
            raise DownstreamServiceError("urban", status, "access denied")

    delays = []

    async def sleep(delay):
        delays.append(delay)

    run = run_state("keep")
    events, saves = await collect(ProjectCreator(Urban(), sleep=sleep), run, [], [])
    assert delays == []
    assert run["ready"]["attempts"] == 1
    assert saves[-1]["ready"]["done"] is False
    assert creation_summary(run, [], [])["ready_error"] == f"{status}: access denied"
    assert not any(event["type"] == "project_ready" for event in events)


async def test_missing_objects_never_confirmed():
    urban = DelayedUrban(object_waits=100)
    delays = []

    async def sleep(delay):
        delays.append(delay)

    _, objects, _ = build_items(project_files(), SPECS)
    run = run_state("keep")
    events, _ = await collect(ProjectCreator(urban, sleep=sleep), run, [], objects)
    assert delays == list(READINESS_RETRY_DELAYS)
    assert creation_summary(run, [], objects)["ready"] is False
    assert "Объекты сценария" in run["ready"]["error"]
    assert not any(event["type"] == "project_ready" for event in events)
