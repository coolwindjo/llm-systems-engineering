from __future__ import annotations

from services.schemas import ConfigurationDraft


def test_configuration_draft_normalizes_hex_id_and_labels() -> None:
    extraction = ConfigurationDraft(
        ServiceName="WheelSpeedBroadcast",
        ID="0x120",
        Class="signal",
        Frequency="10ms",
        PlayType="periodic",
    )

    assert extraction.ID == 0x120
    assert extraction.Class == "Event"
    assert extraction.Frequency == "10 ms"
    assert extraction.PlayType == "Cyclic"


def test_configuration_draft_unresolved_fields_reports_missing_values() -> None:
    extraction = ConfigurationDraft(
        ServiceName="TorqueCommandSync",
        ID=None,
        Class="Unknown",
        Frequency=None,
        PlayType=None,
    )

    assert extraction.unresolved_fields() == ["ID", "Class", "Frequency", "PlayType"]
