from deepface import DeepFace
import hashlib

SIZE = 256

memory = [None] * SIZE
img = "reference.jpg"


def extract_facial_features_from_img(img):
    return DeepFace.represent(
        img_path=img, model_name="VGG-Face", detector_backend="retinaface"
    )


def modulo_hashing(facial_features, size=SIZE):
    return facial_features % size


def insert(facial_features, memory):
    facial_features_int = int(facial_features, 16)
    index = modulo_hashing(facial_features_int)

    while memory[index] is not None:
        index = (index + 1) % SIZE

    memory[index] = facial_features


facial_features = extract_facial_features_from_img(img)
hashed_img = hashlib.sha256(str(facial_features).encode("utf-8")).hexdigest()

insert(hashed_img, memory)

print(memory)
