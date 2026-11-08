from app.services.raga_recommender import RagaRecommender
from app.services.raga_repository import Raga


class FakeRepository:
    def __init__(self):
        self.ragas = [
            Raga(
                raga_name="Primary Raga",
                primary_sthayi_bhava="Shama",
                secondary_sthayi_bhava=[],
                prahara=[1],
                ritu="vasanta",
                youtube_links={
                    "vocal": ["v1", "v2", "v3"],
                    "instrumental": ["i1", "i2", "i3"],
                },
            ),
            Raga(
                raga_name="Secondary Raga",
                primary_sthayi_bhava="Rati",
                secondary_sthayi_bhava=["Shama"],
                prahara=[3],
                ritu="grishma",
                youtube_links={
                    "vocal": ["v4", "v5", "v6"],
                    "instrumental": ["i4", "i5", "i6"],
                },
            ),
            Raga(
                raga_name="Traditional Krodha Raga",
                primary_sthayi_bhava="Krodha",
                secondary_sthayi_bhava=[],
                prahara=[3],
                ritu="grishma",
                youtube_links={
                    "vocal": ["k1", "k2", "k3"],
                    "instrumental": ["k4"],
                },
            ),
            Raga(
                raga_name="Traditional Utsaha Raga",
                primary_sthayi_bhava="Utsaha",
                secondary_sthayi_bhava=[],
                prahara=[3],
                ritu="grishma",
                youtube_links={
                    "vocal": ["u1", "u2", "u3"],
                    "instrumental": ["u4"],
                },
            ),
        ]

    def all(self):
        return self.ragas

    def get_by_name(self, name):
        return next((raga for raga in self.ragas if raga.raga_name == name), None)


def test_primary_bhava_matches_sort_before_secondary_matches():
    recommender = RagaRecommender(FakeRepository())
    results = recommender.recommend(
        mode="therapeutic",
        detected_bhava="shama",
        current_prahara=3,
        current_ritu="grishma",
        style="both",
    )

    assert results[0].raga_name == "Primary Raga"
    assert results[0].bhava_match_type == "primary"
    assert results[1].raga_name == "Secondary Raga"
    assert results[1].bhava_match_type == "secondary"
    assert results[0].bhava_score == 1.0
    assert results[1].bhava_score == 0.6
    assert len(results) == 2
    assert len(results[0].youtube_links["vocal"]) == 3
    assert len(results[0].youtube_links["instrumental"]) == 1


def test_traditional_mode_returns_best_raga_per_target_with_two_songs_each():
    recommender = RagaRecommender(FakeRepository())
    results = recommender.recommend(
        mode="traditional",
        detected_bhava="krodha",
        current_prahara=3,
        current_ritu="grishma",
        style="both",
    )

    assert [item.raga_name for item in results] == [
        "Traditional Utsaha Raga",
        "Traditional Krodha Raga",
        "Primary Raga",
    ]
    assert all(
        len(item.youtube_links["vocal"]) + len(item.youtube_links["instrumental"]) <= 2
        for item in results
    )
