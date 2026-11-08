from app.services.rasa_bhava_mapper import map_rasa_to_bhava


def test_maps_model_spellings_to_supported_bhavas():
    assert map_rasa_to_bhava("karuna") == "shoka"
    assert map_rasa_to_bhava("raudra") == "krodha"
    assert map_rasa_to_bhava("bhibhatsa") == "shama"

