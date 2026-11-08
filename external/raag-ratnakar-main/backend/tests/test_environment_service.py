from app.services.environment_service import calculate_prahara


def test_calculate_prahara_keeps_sunrise_and_sunset_on_same_local_date():
    context = calculate_prahara(28.6139, 77.2090, "Asia/Kolkata")
    assert context.sunrise.date() == context.sunset.date()
