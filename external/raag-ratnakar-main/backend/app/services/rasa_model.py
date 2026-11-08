from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from app.config import settings

DEFAULT_CLASS_NAMES = [
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

RASA_LABEL_ALIASES = {
    "adhbut": "adhbuta",
    "adhbuta": "adhbuta",
    "adbhuta": "adhbuta",
    "bhayank": "bhayanak",
    "bhayanak": "bhayanak",
    "bhibhatsa": "bhibhatsa",
    "bibhatsa": "bhibhatsa",
    "hasya": "hasya",
    "karuna": "karuna",
    "raudra": "raudra",
    "shanta": "shanta",
    "shringara": "shringara",
    "sringara": "shringara",
    "veer": "veer",
    "vira": "veer",
}


class RasaModelError(RuntimeError):
    """Raised when model dependencies or weights are unavailable."""


@dataclass(frozen=True)
class RasaPrediction:
    rasa: str
    confidence: float
    probabilities: dict[str, float]


def normalize_rasa_label(label: str) -> str:
    normalized = label.strip().lower()
    normalized = normalized.split("(", maxsplit=1)[0].strip()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = normalized.removesuffix("_rasa")
    compact = normalized.replace("_", "")
    canonical = RASA_LABEL_ALIASES.get(compact)
    if canonical is None:
        raise RasaModelError(f"Unsupported rasa label in checkpoint: {label!r}")
    return canonical


def normalize_class_names(class_names: Any) -> list[str]:
    if not isinstance(class_names, (list, tuple)):
        raise RasaModelError("Checkpoint class_names must be a list or tuple.")

    normalized = [normalize_rasa_label(str(name)) for name in class_names]
    if len(normalized) != len(DEFAULT_CLASS_NAMES):
        raise RasaModelError(
            f"Checkpoint defines {len(normalized)} classes, expected {len(DEFAULT_CLASS_NAMES)}."
        )
    if len(set(normalized)) != len(normalized):
        raise RasaModelError("Checkpoint class_names contain duplicate rasa labels after normalization.")
    return normalized


def _checkpoint_output_size(state_dict: Mapping[str, Any]) -> int | None:
    bias = state_dict.get("classifier.4.bias")
    if bias is not None and hasattr(bias, "shape") and len(bias.shape) >= 1:
        return int(bias.shape[0])

    weight = state_dict.get("classifier.4.weight")
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
        return int(weight.shape[0])

    return None


class RasaPredictor:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or settings.model_path
        self._loaded = False
        self._model = None
        self._device = None
        self._transform = None
        self._torch = None
        self._class_names = list(DEFAULT_CLASS_NAMES)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.model_path.exists():
            raise RasaModelError(f"Model file not found: {self.model_path}")

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
        except ImportError as exc:
            raise RasaModelError(
                "PyTorch, torchvision, and Pillow are required for image prediction. "
                "Install backend requirements before using /api/recommend/image."
            ) from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.efficientnet_b2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1408, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 9),
        )

        try:
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location=device)

        state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
        if state_dict is None:
            raise RasaModelError("Checkpoint does not contain model weights.")

        output_size = _checkpoint_output_size(state_dict)
        if output_size is not None and output_size != len(DEFAULT_CLASS_NAMES):
            raise RasaModelError(
                f"Checkpoint classifier outputs {output_size} classes, expected {len(DEFAULT_CLASS_NAMES)}."
            )

        class_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
        self._class_names = normalize_class_names(
            DEFAULT_CLASS_NAMES if class_names is None else class_names
        )

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self._torch = torch
        self._device = device
        self._model = model
        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._loaded = True

    def predict(self, image_file: BinaryIO) -> RasaPrediction:
        self._ensure_loaded()
        try:
            from PIL import Image
        except ImportError as exc:
            raise RasaModelError("Pillow is required for image prediction.") from exc

        image_file.seek(0)
        image = Image.open(image_file).convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self._device)

        with self._torch.no_grad():
            output = self._model(tensor)
            probs = self._torch.softmax(output, dim=1)[0].detach().cpu()

        pred_idx = int(self._torch.argmax(probs).item())
        probabilities = {
            self._class_names[index]: float(probs[index].item())
            for index in range(len(self._class_names))
        }
        return RasaPrediction(
            rasa=self._class_names[pred_idx],
            confidence=float(probs[pred_idx].item()),
            probabilities=probabilities,
        )
