import pytest

from app.services.rasa_model import RasaModelError, normalize_class_names, normalize_rasa_label


def test_normalizes_latest_checkpoint_class_names_to_api_labels():
    assert normalize_class_names(
        [
            "adhbuta",
            "Bhayanak",
            "Bhibhatsa",
            "Hasya",
            "Karuna",
            "Raudra",
            "Shanta",
            "Shringara",
            "Veer",
        ]
    ) == [
        "adhbuta",
        "bhayanak",
        "bhibhatsa",
        "hasya",
        "karuna",
        "raudra",
        "shanta",
        "shringara",
        "veer",
    ]


def test_normalizes_notebook_label_aliases():
    assert normalize_rasa_label("Adhbut") == "adhbuta"
    assert normalize_rasa_label("Bhayank") == "bhayanak"
    assert normalize_rasa_label("Sringara") == "shringara"


def test_rejects_invalid_checkpoint_class_names():
    with pytest.raises(RasaModelError, match="Unsupported rasa label"):
        normalize_class_names(
            [
                "adhbuta",
                "Bhayanak",
                "Bhibhatsa",
                "Hasya",
                "Karuna",
                "Raudra",
                "Shanta",
                "Shringara",
                "Unknown",
            ]
        )


def test_rejects_wrong_checkpoint_class_count():
    with pytest.raises(RasaModelError, match="defines 0 classes"):
        normalize_class_names([])
