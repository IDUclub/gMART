import pytest

from tests.integration.local_stack.scenario_contract import verify_provision_scope


def event(service=22, territory=58):
    return {
        "type": "step_event",
        "content": {
            "agent": "provision",
            "event": {
                "type": "error",
                "content": {
                    "message": str(
                        {
                            "input": {
                                "service_type_id": service,
                                "request_ter_id": territory,
                            },
                            "detail": {"code": "missing_service_normative"},
                        }
                    )
                },
            },
        },
    }


def test_expected_calculation_blocker_is_accepted():
    verify_provision_scope([event()])


@pytest.mark.parametrize(
    "events", [[], [event(1)], [event(22, 99)], [event(), event(1)]]
)
def test_wrong_or_broadened_calculation_is_not_accepted(events):
    with pytest.raises(AssertionError):
        verify_provision_scope(events)
