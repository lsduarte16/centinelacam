"""COCO class catalog for YOLOv8 (subset commonly useful for edge missions)."""

# id -> Spanish label
COCO_CLASSES: dict[int, str] = {
    0: "persona",
    1: "bicicleta",
    2: "auto",
    3: "moto",
    5: "bus",
    7: "camion",
    14: "pajaro",
    15: "gato",
    16: "perro",
    17: "caballo",
    18: "oveja",
    19: "vaca",
    24: "mochila",
    25: "paraguas",
    26: "cartera",
    28: "maleta",
    39: "botella",
    41: "taza",
    56: "silla",
    57: "sofa",
    58: "maceta",
    60: "mesa",
    62: "tv",
    63: "laptop",
    64: "mouse",
    66: "teclado",
    67: "celular",
    73: "libro",
    74: "reloj",
    75: "jarron",
    76: "tijeras",
    77: "oso_de_peluche",
    78: "secador",
}

# Grouped for UI
COCO_GROUPS: dict[str, list[int]] = {
    "Personas": [0],
    "Vehiculos": [1, 2, 3, 5, 7],
    "Animales": [14, 15, 16, 17, 18, 19],
    "Objetos": [24, 25, 26, 28, 39, 41, 56, 57, 58, 60, 62, 63, 64, 66, 67, 73, 74, 75, 76, 77, 78],
}


def class_catalog() -> list[dict]:
    """Flat catalog for API/UI."""
    items = []
    for group, ids in COCO_GROUPS.items():
        for cid in ids:
            items.append({"id": cid, "name": COCO_CLASSES[cid], "group": group})
    return items
