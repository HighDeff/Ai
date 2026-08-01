import os
import math
import json
import random
import time
import re
from collections import defaultdict, Counter
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, MemoryError, OSError):
    TORCH_AVAILABLE = False
    print("[TORCH] PyTorch not available. Install with: pip install torch")

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    KAGGLE_AVAILABLE = True
except (ImportError, MemoryError, OSError):
    KAGGLE_AVAILABLE = False

# =========================
# CONFIG
# =========================

DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
MAX_LEN = 128
EMBED_DIM = 256
HEADS = 8
LAYERS = 4
FF_DIM = 512
BATCH_SIZE = 16
LR = 3e-4
SKIP_TRAINING = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# FILE UTILS
# =========================

def find_files(ext, base_dir="."):
    files = []
    for root, _, fns in os.walk(base_dir):
        for f in fns:
            if f.endswith(ext):
                files.append(os.path.join(root, f))
    return files

# =========================
# GLOVE LOADER
# =========================

def load_glove(file_path):
    print(f"[GLOVE] Loading: {file_path}")
    embeddings = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = list(map(float, parts[1:]))
            embeddings[word] = torch.tensor(vec)
    return embeddings

def auto_load_glove(base_dir="."):
    glove_files = find_files(".txt", base_dir)
    glove_files = [f for f in glove_files if "glove" in f.lower()]
    if not glove_files:
        return None
    for i, f in enumerate(glove_files):
        print(i, f)
    idx = int(input("Select GloVe file index: "))
    return load_glove(glove_files[idx])

# =========================
# DATASET LOADER
# =========================

def download_squad_dataset(base_dir="."):
    if not KAGGLE_AVAILABLE:
        return None
    try:
        hf_dataset = kagglehub.load_dataset(
            KaggleDatasetAdapter.HUGGING_FACE,
            "stanfordu/stanford-question-answering-dataset",
            "",
            hf_kwargs={"split": "train"}
        )
        return hf_dataset
    except Exception as e:
        print(f"[KAGGLE] Download failed: {e}")
        return None

def load_squad_dataset(hf_dataset, max_pairs=1000):
    qa_pairs = []
    try:
        if hasattr(hf_dataset, '__iter__'):
            for item in hf_dataset:
                if 'question' in item and 'answers' in item:
                    if isinstance(item['answers'], dict) and 'text' in item['answers']:
                        answers = item['answers']['text']
                        if answers:
                            qa_pairs.append((item['question'], answers[0]))
                if len(qa_pairs) >= max_pairs:
                    break
        print(f"[SQuAD] Loaded {len(qa_pairs)} QA pairs")
        return qa_pairs
    except Exception as e:
        print(f"[SQuAD] Failed: {e}")
        return []

def load_multiple_datasets(base_dir="."):
    txt_files = find_files(".txt", base_dir)
    priority_files = ["qa_dataset_expanded.txt", "qa_dataset.txt", "qa_dataset_squad.txt"]
    exclude_files = ["glove.txt", "glove.6B.txt", "vocab.txt", "how to.txt", "qa_dataset_combined.txt"]
    combined_text = ""
    loaded_files = []

    for filename in priority_files:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    combined_text += content + "\n\n"
                    loaded_files.append(filename)
                    print(f"[DATASET] Loaded: {filename} ({len(content)} chars)")
            except Exception as e:
                print(f"[DATASET] Error: {e}")

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        if filename not in priority_files and filename not in exclude_files and not filename.startswith("."):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if "?" in content and len(content) < 1000000:
                        combined_text += content + "\n\n"
                        loaded_files.append(filename)
                        print(f"[DATASET] Loaded: {filename} ({len(content)} chars)")
            except Exception as e:
                print(f"[DATASET] Error: {e}")

    if not loaded_files:
        return "This is a sample text for training. " * 100
    print(f"[DATASET] Combined {len(loaded_files)} datasets, {len(combined_text)} chars")
    return combined_text

def load_from_folder(folder_path):
    combined_text = ""
    loaded_files = []

    txt_files = find_files(".txt", folder_path)
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        if not filename.startswith("."):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if "?" in content:
                        combined_text += content + "\n\n"
                        loaded_files.append(filename)
            except Exception:
                pass

    pdf_files = find_files(".pdf", folder_path)
    for filepath in pdf_files:
        try:
            import PyPDF2
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    if '?' in line:
                        answer = ""
                        j = i + 1
                        while j < min(i + 5, len(lines)) and lines[j].strip():
                            answer += lines[j].strip() + " "
                            j += 1
                        if answer:
                            combined_text += f"{line.strip()}\n{answer.strip()}\n\n"
                loaded_files.append(os.path.basename(filepath))
        except Exception:
            pass

    return combined_text

def select_dataset_folder():
    folder_path = input("Enter folder path (or Enter to skip): ").strip()
    if folder_path and os.path.exists(folder_path):
        return folder_path
    return None

def load_largest_dataset(base_dir="."):
    return load_multiple_datasets(base_dir)

# =========================
# TOKENIZER
# =========================

class Tokenizer:
    def __init__(self):
        self.word2id = {"<pad>": 0, "<unk>": 1}
        self.id2word = {0: "<pad>", 1: "<unk>"}

    def build_vocab(self, text):
        chunk_size = 100000
        counts = Counter()
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].lower().split()
            counts.update(chunk)
        for w in counts:
            if w not in self.word2id:
                idx = len(self.word2id)
                self.word2id[w] = idx
                self.id2word[idx] = w

    def encode(self, text):
        return [self.word2id.get(w, 1) for w in text.lower().split()]

    def decode(self, ids):
        return " ".join(self.id2word.get(i, "<unk>") for i in ids)

# =========================
# KNOWLEDGE ENGINE
# =========================
# Entity-Category-Attribute graph with fact provenance.
# Dataset facts are "solid", user statements are "mutable" unless
# confirmed with a remember/learn phrase.

STOP_WORDS = {"what", "is", "a", "an", "the", "are", "how", "why", "when",
              "where", "who", "which", "can", "could", "would", "should",
              "do", "does", "did", "will", "to", "for", "of", "in", "on",
              "at", "by", "with", "from", "and", "or", "but", "not", "no",
              "some", "any", "all", "that", "this", "these", "those", "it",
              "its", "they", "them", "their", "he", "she", "we", "you", "i"}

LEARN_PHRASES = [
    "remember that", "remember", "i want you to know", "i want you to remember",
    "know that", "learn that", "note that", "store that", "keep in mind",
    "for future", "write that down", "make a note"
]

CATEGORY_MEMBERS = {}  # populated by build_default_kb

# =========================
# SHARED SENTENCE SPLITTER
# =========================
# Protects common abbreviations from being mistaken for sentence boundaries.

_ABBREV_PLACEHOLDER = "\x00DOT\x00"
_PROTECTED_ABBREVS = [
    "sp.", "spp.", "e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.",
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Jr.", "Sr.", "St.",
    "no.", "No.", "vol.", "ca.", "c.",
]

def split_sentences(text, min_len=5):
    """Split text into sentences, protecting common abbreviations from
    being misread as sentence-ending periods. Returns a list of stripped
    sentence strings with trailing punctuation removed."""
    if not text:
        return []
    protected = text
    for abbrev in _PROTECTED_ABBREVS:
        protected = re.sub(
            re.escape(abbrev),
            abbrev[:-1] + _ABBREV_PLACEHOLDER,
            protected
        )
    raw_sentences = re.split(r'[.!?]+', protected)
    sentences = []
    for s in raw_sentences:
        s = s.replace(_ABBREV_PLACEHOLDER, ".").strip()
        if s and len(s) >= min_len:
            sentences.append(s)
    return sentences

class KnowledgeEngine:
    def __init__(self):
        self.entities = {}           # entity_name -> {category, attributes, facts}
        self.category_graph = defaultdict(set)   # category -> set of child categories
        self.category_members = defaultdict(set)  # category -> set of entity names
        self.fact_store = {}         # (entity, attribute) -> {value, provenance, confidence, turn}
        self.user_overrides = {}     # (entity, attribute) -> {value, confidence, turn}
        self.adjective_index = defaultdict(set)   # adjective -> set of entity names
        self.noun_index = defaultdict(set)        # noun -> set of entity names
        self.verb_index = defaultdict(set)        # verb -> set of entity names
        self.dataset_qa = []        # raw QA pairs from dataset for candidate mining
        self.entity_mention_count = Counter()
        self.contradiction_log = []

        # Temporal index: keyword -> list of (qa_index, relevance_score)
        self.temporal_index = defaultdict(list)
        # Cross-reference index: entity -> set of related QA pairs from other entities
        self.cross_ref_index = defaultdict(set)
        # Topic keyword index: keyword -> set of qa_indices
        self.topic_index = defaultdict(set)
        # Wrap-around index for find_wrap_around_facts
        self._wrap_index_built = False
        self._wrap_keyword_index = defaultdict(set)
        self._wrap_index_qa_count = -1

    def build_default_kb(self):
        kb = {
            "cat": {
                "category": "feline",
                "parent_categories": ["mammal", "animal"],
                "attributes": {
                    "has_shell": False, "has_tail": True, "is_domestic": True,
                    "is_prey": False, "is_predator": True, "is_nocturnal": False,
                    "has_fur": True, "has_feathers": False, "lays_eggs": False,
                    "is_venomous": False, "is_aquatic": False
                },
                "properties": {
                    "tail_count": 1, "tail_color": "varies", "tail_shape": "long",
                    "tail_size": "long relative to body", "color": "varies",
                    "weight_kg": "4-5", "lifespan_years": "12-18",
                    "sound": "meow", "speed_kmh": 30
                },
                "descriptions": [
                    "A cat is a small domesticated feline mammal.",
                    "Cats are popular household pets known for their independence.",
                    "A cat has retractable claws and excellent night vision."
                ],
                "similar_to": ["dog", "lion", "tiger", "leopard"]
            },
            "dog": {
                "category": "canine",
                "parent_categories": ["mammal", "animal"],
                "attributes": {
                    "has_shell": False, "has_tail": True, "is_domestic": True,
                    "is_prey": False, "is_predator": True, "is_nocturnal": False,
                    "has_fur": True, "has_feathers": False, "lays_eggs": False,
                    "is_venomous": False, "is_aquatic": False
                },
                "properties": {
                    "tail_count": 1, "tail_color": "varies", "tail_shape": "varies",
                    "tail_size": "varies by breed", "color": "varies",
                    "weight_kg": "2-80", "lifespan_years": "10-13",
                    "sound": "bark", "speed_kmh": 45
                },
                "descriptions": [
                    "A dog is a domesticated canine mammal.",
                    "Dogs are known for their loyalty and trainability.",
                    "A dog has a keen sense of smell and hearing."
                ],
                "similar_to": ["cat", "wolf", "fox", "coyote"]
            },
            "turtle": {
                "category": "reptile",
                "parent_categories": ["reptile", "animal"],
                "attributes": {
                    "has_shell": True, "has_tail": True, "is_domestic": False,
                    "is_prey": False, "is_predator": False, "is_nocturnal": False,
                    "has_fur": False, "has_feathers": False, "lays_eggs": True,
                    "is_venomous": False, "is_aquatic": True
                },
                "properties": {
                    "shell_count": 1, "shell_color": "brown or green",
                    "shell_hardness": "very hard", "shell_material": "bone and keratin",
                    "tail_count": 1, "color": "brown or green",
                    "weight_kg": "1-500", "lifespan_years": "20-150",
                    "speed_kmh": 3
                },
                "descriptions": [
                    "A turtle is a reptile characterized by its hard protective shell.",
                    "Turtles carry their shell on their back at all times.",
                    "A turtle's shell is part of its skeleton, not something it can remove."
                ],
                "similar_to": ["tortoise", "terrapin"]
            },
            "diamond": {
                "category": "precious gemstone",
                "parent_categories": ["gemstone", "mineral", "object"],
                "attributes": {
                    "is_shiny": True, "is_hard": True, "is_transparent": True,
                    "is_colorful": False, "is_metallic": False, "is_organic": False,
                    "has_shell": False, "is_precious": True
                },
                "properties": {
                    "hardness": "10 (hardest natural material)",
                    "color": "typically clear/colorless",
                    "luster": "adamantine (exceptionally brilliant)",
                    "origin": "formed deep in Earth's mantle",
                    "chemical_composition": "pure carbon",
                    "refractive_index": "2.417"
                },
                "descriptions": [
                    "A diamond is a precious gemstone made of pure carbon.",
                    "Diamonds are the hardest known natural material.",
                    "Diamonds are prized for their exceptional brilliance and luster."
                ],
                "similar_to": ["ruby", "sapphire", "emerald"]
            },
            "ruby": {
                "category": "precious gemstone",
                "parent_categories": ["gemstone", "mineral", "object"],
                "attributes": {
                    "is_shiny": True, "is_hard": True, "is_transparent": True,
                    "is_colorful": True, "is_metallic": False, "is_organic": False,
                    "has_shell": False, "is_precious": True
                },
                "properties": {
                    "hardness": "9", "color": "red",
                    "luster": "vitreous to Adamantine",
                    "chemical_composition": "corundum (Al2O3) with chromium"
                },
                "descriptions": [
                    "A ruby is a precious red gemstone.",
                    "Rubies are a variety of corundum colored by chromium.",
                    "Rubies symbolize passion and are among the hardest gemstones."
                ],
                "similar_to": ["diamond", "sapphire", "emerald"]
            },
            "sapphire": {
                "category": "precious gemstone",
                "parent_categories": ["gemstone", "mineral", "object"],
                "attributes": {
                    "is_shiny": True, "is_hard": True, "is_transparent": True,
                    "is_colorful": True, "is_metallic": False, "is_organic": False,
                    "has_shell": False, "is_precious": True
                },
                "properties": {
                    "hardness": "9", "color": "blue (but can be many colors)",
                    "luster": "vitreous",
                    "chemical_composition": "corundum (Al2O3) with trace elements"
                },
                "descriptions": [
                    "A sapphire is a precious gemstone, most commonly blue.",
                    "Sapphires are a variety of corundum.",
                    "Sapphires are valued for their beauty and hardness."
                ],
                "similar_to": ["diamond", "ruby", "emerald"]
            },
            "emerald": {
                "category": "precious gemstone",
                "parent_categories": ["gemstone", "mineral", "object"],
                "attributes": {
                    "is_shiny": True, "is_hard": True, "is_transparent": True,
                    "is_colorful": True, "is_metallic": False, "is_organic": False,
                    "has_shell": False, "is_precious": True
                },
                "properties": {
                    "hardness": "7.5-8", "color": "green",
                    "luster": "vitreous",
                    "chemical_composition": "beryllium aluminum silide with chromium/vanadium"
                },
                "descriptions": [
                    "An emerald is a precious green gemstone.",
                    "Emeralds are prized for their vivid green color.",
                    "Emeralds are a variety of beryl colored by trace elements."
                ],
                "similar_to": ["diamond", "ruby", "sapphire"]
            },
            "shell": {
                "category": "covering",
                "parent_categories": ["object"],
                "attributes": {
                    "is_hard": True, "is_shiny": False, "is_protective": True,
                    "is_removable": False, "is_part_of_body": True
                },
                "properties": {
                    "material": "bone and keratin (in turtles)",
                    "hardness": "varies by species",
                    "function": "protection and support"
                },
                "descriptions": [
                    "A shell is a hard protective outer layer.",
                    "In turtles, the shell is fused to the skeleton and cannot be removed.",
                    "Shells can be found on various animals including turtles, snails, and clams."
                ],
                "similar_to": ["exoskeleton", "carapace"]
            },
            "catdog": {
                "category": "hybrid_concept",
                "parent_categories": ["animal", "concept"],
                "attributes": {
                    "is_real": False, "is_hypothetical": True
                },
                "descriptions": [
                    "A catdog is a hypothetical hybrid of a cat and a dog.",
                    "Cats and dogs are both mammals but belong to different families."
                ],
                "similar_to": ["cat", "dog"]
            },
            "precious gemstone": {
                "category": "category_only",
                "parent_categories": ["gemstone", "mineral", "object"],
                "attributes": {
                    "is_precious": True, "is_hard": True, "is_shiny": True
                },
                "descriptions": [
                    "Precious gemstones include diamond, ruby, sapphire, and emerald.",
                    "Precious gemstones are rare, hard, and beautiful minerals."
                ],
                "similar_to": ["gemstone", "mineral"]
            },
            "shiny object": {
                "category": "category_only",
                "parent_categories": ["object"],
                "attributes": {
                    "is_shiny": True
                },
                "descriptions": [
                    "Shiny objects reflect light and include things like diamonds, mirrors, and polished metals.",
                    "Many precious gemstones are shiny objects."
                ],
                "similar_to": ["reflective surface"]
            },
            "tortoise": {
                "category": "reptile",
                "parent_categories": ["reptile", "animal"],
                "attributes": {
                    "has_shell": True, "has_tail": True, "is_domestic": False,
                    "is_aquatic": False
                },
                "properties": {"shell_count": 1, "weight_kg": "1-400", "lifespan_years": "80-150"},
                "descriptions": [
                    "A tortoise is a land-dwelling reptile with a high-domed shell.",
                    "Tortoises are different from turtles in that they live on land."
                ],
                "similar_to": ["turtle"]
            },
            "armadillo": {
                "category": "mammal",
                "parent_categories": ["mammal", "animal"],
                "attributes": {
                    "has_shell": True, "has_tail": True, "is_domestic": False,
                    "is_aquatic": False
                },
                "properties": {"shell_count": 1, "weight_kg": "0.1-50"},
                "descriptions": [
                    "An armadillo is a mammal known for its armor-like shell.",
                    "Armadillos are the only mammals with a true shell."
                ],
                "similar_to": ["turtle", "pangolin"]
            },
            "snail": {
                "category": "mollusk",
                "parent_categories": ["mollusk", "animal"],
                "attributes": {
                    "has_shell": True, "has_tail": False, "is_domestic": False,
                    "is_aquatic": False, "lays_eggs": True
                },
                "properties": {"shell_count": 1, "speed_kmh": 0.03},
                "descriptions": [
                    "A snail is a mollusk that carries a spiral shell on its back.",
                    "Snails move slowly using a muscular foot."
                ],
                "similar_to": ["clam", "oyster"]
            },
            "clam": {
                "category": "mollusk",
                "parent_categories": ["mollusk", "animal"],
                "attributes": {
                    "has_shell": True, "has_tail": False, "is_domestic": False,
                    "is_aquatic": True, "lays_eggs": True
                },
                "properties": {"shell_count": 2},
                "descriptions": [
                    "A clam is an aquatic mollusk with a hinged two-part shell.",
                    "Clams are filter feeders found in both freshwater and saltwater."
                ],
                "similar_to": ["oyster", "mussel", "snail"]
            },
            "pangolin": {
                "category": "mammal",
                "parent_categories": ["mammal", "animal"],
                "attributes": {
                    "has_shell": True, "has_tail": True, "is_domestic": False,
                    "is_aquatic": False
                },
                "properties": {"shell_count": 1, "weight_kg": "2-33"},
                "descriptions": [
                    "A pangolin is a mammal covered in keratin scales.",
                    "Pangolins are the most trafficked mammals in the world."
                ],
                "similar_to": ["armadillo", "anteater"]
            },
            "bird": {
                "category": "avian",
                "parent_categories": ["avian", "animal"],
                "attributes": {
                    "has_shell": False, "has_tail": True, "is_domestic": False,
                    "is_prey": False, "is_predator": False, "is_nocturnal": False,
                    "has_fur": False, "has_feathers": True, "lays_eggs": True,
                    "is_venomous": False, "is_aquatic": False
                },
                "properties": {
                    "weight_kg": "0.003-12", "lifespan_years": "2-80",
                    "speed_kmh": 50, "sound": "chirp"
                },
                "descriptions": [
                    "A bird is a warm-blooded vertebrate characterized by feathers, wings, and beaks.",
                    "Birds are capable of flight and are found on every continent.",
                    "Birds lay eggs and have a high metabolic rate."
                ],
                "similar_to": ["eagle", "sparrow", "parrot"]
            },
            "fish": {
                "category": "aquatic animal",
                "parent_categories": ["aquatic animal", "animal"],
                "attributes": {
                    "has_shell": False, "has_tail": True, "is_domestic": False,
                    "is_prey": False, "is_predator": False, "is_nocturnal": False,
                    "has_fur": False, "has_feathers": False, "lays_eggs": True,
                    "is_venomous": False, "is_aquatic": True
                },
                "properties": {
                    "weight_kg": "0.001-2000", "lifespan_years": "1-100",
                    "speed_kmh": 40, "sound": "none"
                },
                "descriptions": [
                    "A fish is a cold-blooded aquatic vertebrate that breathes through gills.",
                    "Fish are found in both freshwater and saltwater environments.",
                    "Fish use fins to swim and have scales covering their bodies."
                ],
                "similar_to": ["shark", "salmon", "tuna"]
            },
            "sky": {
                "category": "natural phenomenon",
                "parent_categories": ["nature", "object"],
                "attributes": {
                    "is_shiny": False, "is_blue": True, "is_colorful": True,
                    "is_transparent": False, "is_organic": False
                },
                "properties": {
                    "color": "blue during day, black at night",
                    "apparent_color_cause": "Rayleigh scattering of sunlight",
                    "typical_color": "blue",
                    "composition": "nitrogen (78%), oxygen (21%), other gases"
                },
                "descriptions": [
                    "The sky appears blue during the day due to Rayleigh scattering.",
                    "The sky is the region of atmosphere surrounding Earth.",
                    "At night, the sky appears black with visible stars."
                ],
                "similar_to": ["ocean", "atmosphere"]
            },
            "gold": {
                "category": "metal",
                "parent_categories": ["metal", "mineral", "object"],
                "attributes": {
                    "is_shiny": True, "is_hard": False, "is_metallic": True,
                    "is_precious": True, "is_conductive": True, "is_magnetic": False,
                    "is_organic": False
                },
                "properties": {
                    "color": "golden yellow",
                    "hardness": "2.5 (soft)",
                    "density": "19.3 g/cm³",
                    "melting_point": "1064°C",
                    "chemical_composition": "Au (atomic number 79)"
                },
                "descriptions": [
                    "Gold is a precious metal known for its distinctive golden color.",
                    "Gold is highly malleable, ductile, and resistant to corrosion.",
                    "Gold has been used for currency, jewelry, and electronics for millennia."
                ],
                "similar_to": ["silver", "copper", "platinum"]
            },
            "silver": {
                "category": "metal",
                "parent_categories": ["metal", "mineral", "object"],
                "attributes": {
                    "is_shiny": True, "is_hard": False, "is_metallic": True,
                    "is_precious": True, "is_conductive": True, "is_magnetic": False,
                    "is_organic": False
                },
                "properties": {
                    "color": "silvery white",
                    "hardness": "2.5-3",
                    "density": "10.49 g/cm³",
                    "melting_point": "961.8°C",
                    "chemical_composition": "Ag (atomic number 47)"
                },
                "descriptions": [
                    "Silver is a precious metal with a bright silvery-white luster.",
                    "Silver has the highest electrical conductivity of any metal.",
                    "Silver has been used in coins, jewelry, and photography."
                ],
                "similar_to": ["gold", "copper", "platinum"]
            },
            "iron": {
                "category": "metal",
                "parent_categories": ["metal", "mineral", "object"],
                "attributes": {
                    "is_shiny": False, "is_hard": True, "is_metallic": True,
                    "is_precious": False, "is_conductive": True, "is_magnetic": True,
                    "is_organic": False
                },
                "properties": {
                    "color": "silvery gray",
                    "hardness": "4",
                    "density": "7.874 g/cm³",
                    "melting_point": "1538°C",
                    "chemical_composition": "Fe (atomic number 26)"
                },
                "descriptions": [
                    "Iron is a strong, silvery-gray metal that is magnetic.",
                    "Iron is the most commonly used metal, especially in steel production.",
                    "Iron is essential for life as a component of hemoglobin in blood."
                ],
                "similar_to": ["steel", "copper"]
            },
            "copper": {
                "category": "metal",
                "parent_categories": ["metal", "mineral", "object"],
                "attributes": {
                    "is_shiny": True, "is_hard": False, "is_metallic": True,
                    "is_precious": False, "is_conductive": True, "is_magnetic": False,
                    "is_organic": False
                },
                "properties": {
                    "color": "reddish orange",
                    "hardness": "3",
                    "density": "8.96 g/cm³",
                    "melting_point": "1085°C",
                    "chemical_composition": "Cu (atomic number 29)"
                },
                "descriptions": [
                    "Copper is a reddish-orange metal with high thermal and electrical conductivity.",
                    "Copper is one of the few metals that occur naturally in pure form.",
                    "Copper has been used by humans for over 10,000 years."
                ],
                "similar_to": ["gold", "silver", "bronze"]
            }
        }

        # Register in indices
        for entity_name, data in kb.items():
            self.entities[entity_name] = data
            cat = data.get("category", "unknown")
            self.category_members[cat].add(entity_name)
            for parent in data.get("parent_categories", []):
                self.category_graph[parent].add(cat)
                self.category_members[parent].add(entity_name)

            for attr, val in data.get("attributes", {}).items():
                if isinstance(val, bool):
                    self.adjective_index[attr].add(entity_name)

            for desc in data.get("descriptions", []):
                for w in desc.lower().split():
                    clean = w.rstrip("?!.,;:")
                    if clean not in STOP_WORDS and len(clean) > 2:
                        self.noun_index[clean].add(entity_name)

        # Also index category-only terms
        for cat_name in self.category_graph:
            for member in self.category_members.get(cat_name, set()):
                self.adjective_index[cat_name].add(member)

        return kb

    def load_dataset_qa(self, text):
        lines = text.split('\n')
        i = 0
        temporal_keywords = ["year", "years", "ago", "century", "centuries", "old",
                           "history", "historical", "ancient", "modern", "first",
                           "lived", "exist", "always", "never", "originally",
                           "million", "thousand", "billion", "decade"]
        while i < len(lines):
            line = lines[i].strip()
            # Skip comment lines, empty lines, or malformed indexed lines
            if not line or line.startswith('#'):
                i += 1
                continue
            # Skip lines that look like indexed format: "10: Q=..." or "Q: Q:..."
            if re.match(r'^\d+:\s*Q=', line) or re.match(r'^Q:\s*Q:', line):
                i += 1
                continue
            if line.endswith('?'):
                question = line
                answer = ""
                j = i + 1
                while j < len(lines) and lines[j].strip():
                    answer += lines[j].strip() + " "
                    j += 1
                if answer:
                    # Skip malformed answers starting with "A=" or "A:"
                    answer_clean = answer.strip()
                    if answer_clean.startswith('A=') or answer_clean.startswith('A:'):
                        answer_clean = answer_clean[2:].strip()
                    if answer_clean:
                        qa_idx = len(self.dataset_qa)
                        self.dataset_qa.append((question, answer_clean))
                    # Auto-extract entities from dataset QA
                    self._extract_entities_from_qa(question, answer.strip())

                    # Build temporal index
                    q_lower = question.lower()
                    a_lower = answer.lower()
                    combined = q_lower + " " + a_lower
                    for kw in temporal_keywords:
                        if kw in combined:
                            self.temporal_index[kw].append(qa_idx)

                    # Build topic keyword index
                    words = set(w.lower().rstrip("?!.,;:") for w in combined.split())
                    for w in words:
                        if w not in STOP_WORDS and len(w) > 2:
                            self.topic_index[w].add(qa_idx)

                    # Build cross-reference index (all non-stop question words)
                    q_words = set(w.lower().rstrip("?!.,;:") for w in question.split())
                    for qw in q_words:
                        if qw not in STOP_WORDS and len(qw) > 2:
                            self.cross_ref_index[qw].add(qa_idx)
                i = j
            else:
                i += 1

    def _extract_entities_from_qa(self, question, answer):
        q_words = set(w.lower().rstrip("?!.,;:") for w in question.split())
        a_words = set(w.lower().rstrip("?!.,;:") for w in answer.split())
        # Find likely entity names (nouns in question that appear in answer)
        for w in q_words:
            if w not in STOP_WORDS and w in a_words:
                if w not in self.entities:
                    self.entities[w] = {
                        "category": "unknown",
                        "parent_categories": [],
                        "attributes": {},
                        "properties": {},
                        "descriptions": [answer],
                        "similar_to": []
                    }
                    self.noun_index[w].add(w)
                    self.entity_mention_count[w] += 1
                elif answer not in self.entities[w].get("descriptions", []):
                    # Add additional descriptions from other QA pairs
                    self.entities[w].setdefault("descriptions", []).append(answer)
                    self.entity_mention_count[w] += 1

    def register_user_fact(self, entity, attribute, value, turn=0):
        self.user_overrides[(entity, attribute)] = {
            "value": value,
            "confidence": 0.8,
            "turn": turn
        }

    def auto_index_input(self, user_input):
        """Auto-index adjectives, nouns, and verbs from user input into the KB indices.
        Only indexes content words, not attribute words or stopwords."""
        words = re.findall(r'\b[a-z]+\b', user_input.lower())
        attr_words = {"shape", "color", "colour", "size", "weight", "height",
                      "width", "length", "speed", "price", "name", "type",
                      "sound", "lifespan", "hardness", "origin", "chemical"}

        for w in words:
            if w in STOP_WORDS or len(w) < 3 or w in attr_words:
                continue
            # Index as noun if it appears in any entity's descriptions
            for entity_name, data in self.entities.items():
                for desc in data.get("descriptions", []):
                    if w in desc.lower().split():
                        self.noun_index[w].add(entity_name)
                        break
            # Index as adjective if it could describe entities
            # (simple heuristic: words ending in common adjective patterns or already in known adj patterns)
            adj_patterns = ["ous", "ive", "ful", "less", "able", "ible", "al", "ial",
                           "ic", "ical", "ant", "ent", "ary", "ory", "ly",
                           "ing", "ed", "like"]
            if any(w.endswith(p) for p in adj_patterns):
                for entity_name in self.entities:
                    self.adjective_index[w].add(entity_name)

    def search_cross_references(self, entity, max_results=5):
        """Find QA pairs related to entity from other entities' contexts."""
        entity_lower = entity.lower()
        qa_indices = self.cross_ref_index.get(entity_lower, set())
        results = []
        for idx in qa_indices:
            if idx < len(self.dataset_qa):
                q, a = self.dataset_qa[idx]
                # Check if this QA pair is about a DIFFERENT entity
                q_words = set(w.lower().rstrip("?!.,;:") for w in q.split())
                if entity_lower not in q_words:
                    results.append((q, a, 0.7))
        return results[:max_results]

    def search_temporal(self, keywords, max_results=5):
        """Find QA pairs matching temporal/historical keywords."""
        qa_indices = set()
        for kw in keywords:
            for idx in self.temporal_index.get(kw.lower(), []):
                qa_indices.add(idx)
        results = []
        for idx in qa_indices:
            if idx < len(self.dataset_qa):
                q, a = self.dataset_qa[idx]
                # Score by keyword overlap
                combined = (q + " " + a).lower()
                score = sum(0.2 for kw in keywords if kw.lower() in combined)
                results.append((q, a, min(0.9, score)))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]

    def search_by_topic(self, keywords, entity=None, max_results=5):
        """Find QA pairs by topic keywords, optionally filtered by entity."""
        qa_indices = set()
        for kw in keywords:
            for idx in self.topic_index.get(kw.lower(), []):
                qa_indices.add(idx)
        results = []
        entity_lower = entity.lower() if entity else None
        for idx in qa_indices:
            if idx < len(self.dataset_qa):
                q, a = self.dataset_qa[idx]
                combined = (q + " " + a).lower()
                # Score by keyword overlap
                score = sum(0.15 for kw in keywords if kw.lower() in combined)
                # Boost if entity is mentioned
                if entity_lower and entity_lower in combined:
                    score += 0.3
                if score > 0.1:
                    results.append((q, a, min(0.95, score)))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]

    def get_related_qa_pairs(self, entity, query_words=None, max_results=6):
        """Get multiple related QA pairs for rich answer composition."""
        entity_lower = entity.lower()
        results = []
        seen = set()

        # Direct entity QA pairs
        for idx in self.cross_ref_index.get(entity_lower, set()):
            if idx < len(self.dataset_qa) and idx not in seen:
                q, a = self.dataset_qa[idx]
                results.append((q, a, 0.9))
                seen.add(idx)

        # Topic-related pairs
        if query_words:
            topic_results = self.search_by_topic(query_words, entity, max_results * 2)
            for q, a, score in topic_results:
                idx = None
                for i, (qq, aa) in enumerate(self.dataset_qa):
                    if qq == q:
                        idx = i
                        break
                if idx is not None and idx not in seen:
                    results.append((q, a, score))
                    seen.add(idx)

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]

    def is_learn_phrase(self, text):
        lower = text.lower()
        for phrase in LEARN_PHRASES:
            if lower.startswith(phrase):
                return True
        return False

    def check_contradiction(self, entity, attribute, claimed_value):
        key = (entity, attribute)
        if key in self.fact_store:
            stored = self.fact_store[key]
            if stored["provenance"] == "dataset" and str(stored["value"]).lower() != str(claimed_value).lower():
                self.contradiction_log.append({
                    "entity": entity, "attribute": attribute,
                    "dataset_value": stored["value"],
                    "user_value": claimed_value
                })
                return True, stored["value"]
        return False, None

    def get_all_facts(self, entity):
        entity_lower = entity.lower()
        if entity_lower not in self.entities:
            return []
        data = self.entities[entity_lower]
        facts = []
        for attr, val in data.get("attributes", {}).items():
            if isinstance(val, bool):
                facts.append((attr, val))
        for prop, val in data.get("properties", {}).items():
            facts.append((prop, val))
        return facts

    def get_boolean_facts(self, entity):
        entity_lower = entity.lower()
        if entity_lower not in self.entities:
            return []
        data = self.entities[entity_lower]
        return [(attr, val) for attr, val in data.get("attributes", {}).items() if isinstance(val, bool)]

    def get_property_facts(self, entity):
        entity_lower = entity.lower()
        if entity_lower not in self.entities:
            return []
        data = self.entities[entity_lower]
        return [(prop, val) for prop, val in data.get("properties", {}).items()]

    def get_category(self, entity):
        entity_lower = entity.lower()
        if entity_lower in self.entities:
            return self.entities[entity_lower].get("category", "unknown")
        return None

    def get_parent_categories(self, entity):
        entity_lower = entity.lower()
        if entity_lower in self.entities:
            return self.entities[entity_lower].get("parent_categories", [])
        return []

    def get_members_of_category(self, category):
        return list(self.category_members.get(category.lower(), set()))

    def get_entities_with_attribute(self, attribute):
        return list(self.adjective_index.get(attribute.lower(), set()))

    def find_similar_entities(self, entity):
        entity_lower = entity.lower()
        if entity_lower in self.entities:
            return self.entities[entity_lower].get("similar_to", [])
        return []

    def search_dataset_for_context(self, entity, context_words=None):
        if not self.dataset_qa:
            return []
        entity_lower = entity.lower()
        # Word-boundary pattern to avoid substring matches (e.g. "cat" in "domesticated")
        entity_pattern = re.compile(r'\b' + re.escape(entity_lower) + r'\b', re.IGNORECASE)
        scored = []
        for q, a in self.dataset_qa:
            score = 0.0
            entity_matched = False
            if entity_pattern.search(q) or entity_pattern.search(a):
                score += 2.0
                entity_matched = True
            # Only add context bonuses if entity was found in this QA pair
            if entity_matched and context_words:
                for cw in context_words:
                    cw_pattern = re.compile(r'\b' + re.escape(cw.lower()) + r'\b', re.IGNORECASE)
                    if cw_pattern.search(q) or cw_pattern.search(a):
                        score += 0.5
            if score > 0:
                # Split answer into sentences for more granular results
                sentences = split_sentences(a, min_len=10)
                if len(sentences) > 1:
                    for sent in sentences:
                        sent_score = score * 0.8
                        for cw in (context_words or []):
                            cw_pattern = re.compile(r'\b' + re.escape(cw.lower()) + r'\b', re.IGNORECASE)
                            if cw_pattern.search(sent):
                                sent_score += 0.3
                        scored.append((q, sent.strip().rstrip(".") + ".", sent_score))
                else:
                    scored.append((q, a, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:10]

    def search_dataset_multi_keyword(self, keywords):
        """Search dataset for answers matching multiple keywords."""
        if not self.dataset_qa or not keywords:
            return []
        results = []
        for q, a in self.dataset_qa:
            q_lower = q.lower()
            a_lower = a.lower()
            combined = q_lower + " " + a_lower
            match_count = sum(1 for kw in keywords if kw.lower() in combined)
            if match_count >= len(keywords) * 0.5:
                score = match_count / max(len(keywords), 1)
                sentences = split_sentences(a, min_len=10)
                for sent in sentences:
                    sent_match = sum(1 for kw in keywords if kw.lower() in sent.lower())
                    if sent_match > 0:
                        results.append((sent.strip().rstrip(".") + ".", min(0.95, score + sent_match * 0.1)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:6]

    def _ensure_wrap_index(self):
        """Build lazily a keyword -> qa_indices index for find_wrap_around_facts."""
        if self._wrap_index_built and len(self.dataset_qa) == self._wrap_index_qa_count:
            return
        self._wrap_keyword_index = defaultdict(set)
        for idx, (q, a) in enumerate(self.dataset_qa):
            combined_words = set(re.findall(r'\w+', (q + " " + a).lower()))
            for w in combined_words:
                if w not in STOP_WORDS and len(w) > 2:
                    self._wrap_keyword_index[w].add(idx)
        self._wrap_index_built = True
        self._wrap_index_qa_count = len(self.dataset_qa)

    def find_wrap_around_facts(self, base_answer, entity, extra_keywords=None, max_results=3):
        """Given a base answer, search dataset_qa for additional sentences that
        share keywords with the entity but add new information not already in base_answer."""
        entity_lower = entity.lower()
        base_words = set(re.findall(r'\w+', base_answer.lower()))
        keywords = set(extra_keywords or [])
        keywords.add(entity_lower)
        data = self.entities.get(entity_lower, {})
        keywords.update(p.lower() for p in data.get("parent_categories", []))
        cat = data.get("category")
        if cat:
            keywords.add(cat.lower())
        for sim in data.get("similar_to", []):
            keywords.add(sim.lower())
        keywords = {k for k in keywords if k not in STOP_WORDS and len(k) > 2}
        if not keywords:
            return []

        # Use index for faster lookup
        self._ensure_wrap_index()
        candidate_indices = set()
        for kw in keywords:
            candidate_indices.update(self._wrap_keyword_index.get(kw, set()))

        candidates = []
        for idx in candidate_indices:
            if idx >= len(self.dataset_qa):
                continue
            q, a = self.dataset_qa[idx]
            combined_words = set(re.findall(r'\w+', (q + " " + a).lower()))
            kw_hits = len(keywords & combined_words)
            if kw_hits < 2 and entity_lower not in combined_words:
                continue
            sentences = split_sentences(a, min_len=10)
            for sent in sentences:
                sent_words = set(re.findall(r'\w+', sent.lower()))
                new_info = len(sent_words - base_words)
                sent_kw_hits = len(keywords & sent_words)
                if sent_kw_hits > 0 and new_info > 2:
                    score = sent_kw_hits * 1.0 + new_info * 0.1
                    candidates.append((sent.rstrip(".") + ".", score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        unique = []
        for text, score in candidates:
            norm = text.lower()[:60]
            if norm not in seen:
                seen.add(norm)
                unique.append(text)
        return unique[:max_results]

    def find_contrasting_facts(self, entity, attribute, value):
        results = []
        for (e, a), data in self.fact_store.items():
            if a == attribute and data["value"] != value and e != entity:
                results.append((e, a, data["value"]))
        return results

    def split_compound_query(self, query):
        patterns = [
            r"what is (.+?) and what is (.+)",
            r"what are (.+?) and what are (.+)",
            r"what is (.+?) and (.+?)(?:\?|$)",
            r"who is (.+?) and who is (.+)",
            r"tell me about (.+?) and (.+)",
            r"compare (?:a |an |the )?(.+?) and (?:a |an |the )?(.+)",
            r"difference between (?:a |an |the )?(.+?) and (?:a |an |the )?(.+)",
            r"what(?:'s| is) the difference between (?:a |an |the )?(.+?) and (?:a |an |the )?(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, query.lower())
            if m:
                return [m.group(1).strip(), m.group(2).strip()]

        if " and " in query.lower():
            q_lower = query.lower()
            parts = q_lower.split(" and ")
            entities = []
            for part in parts:
                part = part.strip()
                for prefix in ["what is ", "what are ", "who is ", "tell me about ", "compare ", "compare a ", "compare an ", "compare the "]:
                    if part.startswith(prefix):
                        part = part[len(prefix):]
                if part:
                    entities.append(part.rstrip("?").strip())
            if len(entities) >= 2:
                return entities[:2]
        return None

    def answer_compound(self, entities):
        if len(entities) < 2:
            return None
        e1 = entities[0].lower()
        e2 = entities[1].lower()
        data1 = self.entities.get(e1)
        data2 = self.entities.get(e2)

        # If both in KB, use structured comparison
        if data1 and data2:
            return self._structured_compound(e1, data1, e2, data2)

        # If one or both not in KB, search dataset for both and combine
        parts = []
        for entity in [e1, e2]:
            data = self.entities.get(entity)
            if data:
                descs = data.get("descriptions", [])
                cat = data.get("category", "unknown")
                if descs:
                    parts.append(descs[0])
                elif cat != "unknown":
                    parts.append(f"{entity.title()} is a {cat}.")
            else:
                # Search dataset
                for q, a in self.dataset_qa:
                    if entity in q.lower():
                        first_sent = a.split(".")[0].strip() + "."
                        parts.append(first_sent)
                        break

        if len(parts) == 2:
            return f"{parts[0]} Similarly, {parts[1].lower()}"
        elif parts:
            return ". ".join(parts)
        return None

    def _structured_compound(self, e1, data1, e2, data2):
        cat1 = data1.get("category", "unknown")
        cat2 = data2.get("category", "unknown")
        attrs1 = data1.get("attributes", {})
        attrs2 = data2.get("attributes", {})

        shared = []
        different = []
        for attr in set(list(attrs1.keys()) + list(attrs2.keys())):
            v1 = attrs1.get(attr)
            v2 = attrs2.get(attr)
            if v1 is not None and v2 is not None:
                if v1 == v2:
                    shared.append((attr, v1))
                else:
                    different.append((attr, v1, v2))

        parts = []
        e1_cap = e1.title()
        e2_cap = e2.title()

        if cat1 == cat2:
            if cat1 != "unknown":
                parts.append(f"{e1_cap} and {e2_cap} are both {cat1}s")
            else:
                parts.append(f"{e1_cap} and {e2_cap} are both related things")
        else:
            c1 = f"a {cat1}" if cat1 != "unknown" else "something"
            c2 = f"a {cat2}" if cat2 != "unknown" else "something"
            parts.append(f"{e1_cap} is {c1} and {e2_cap} is {c2}")

        shared_bools = [(a, v) for a, v in shared if isinstance(v, bool)]
        if shared_bools:
            random.shuffle(shared_bools)
            # For shared True booleans, format as natural sentences
            true_formatted = []
            for a, v in shared_bools:
                if v:
                    # Get adjective/noun form: "has_fur" → "have fur", "is_shiny" → "are shiny"
                    word = a.replace("is_", "").replace("has_", "").replace("_", " ")
                    adj_endings = ("y", "ous", "ive", "ful", "less", "able", "ic", "ent", "ant", "al", "ite", "or", "id", "ate")
                    known_adjs = {"domestic", "nocturnal", "venomous", "aquatic", "transparent", "shiny", "hard", "precious", "colorful", "metallic", "organic"}
                    is_adj = any(word.endswith(e) for e in adj_endings) or word in known_adjs or len(word) <= 5
                    if a.startswith("has_"):
                        true_formatted.append(f"have {word}")
                    elif is_adj:
                        true_formatted.append(f"are {word}")
                    else:
                        true_formatted.append(f"have {word}")
            if true_formatted:
                random.shuffle(true_formatted)
                pick = random.sample(true_formatted, min(3, len(true_formatted)))
                if len(pick) == 1:
                    parts.append(f"They both {pick[0]}.")
                else:
                    parts.append(f"They both {', '.join(pick[:-1])} and {pick[-1]}.")

        # For shared false attributes, list them without double negatives
        false_attrs_shared = [(a, v) for a, v in shared_bools if not v]
        if false_attrs_shared:
            random.shuffle(false_attrs_shared)
            false_parts = []
            for a, v in false_attrs_shared[:3]:
                word = a.replace("is_", "").replace("has_", "").replace("lays_", "").replace("_", " ")
                adj_endings = ("y", "ous", "ive", "ful", "less", "able", "ic", "ent", "ant", "al", "ite", "or", "id", "ate")
                known_adjs = {"domestic", "nocturnal", "venomous", "aquatic", "transparent", "shiny", "hard", "precious", "colorful", "metallic", "organic"}
                is_adj = any(word.endswith(e) for e in adj_endings) or word in known_adjs or len(word) <= 5
                if a.startswith("lays_"):
                    false_parts.append(f"lay {word}")
                elif a.startswith("has_"):
                    article = "" if word.endswith("s") else ("a " if word and word[0] not in "aeiou" else ("an " if word else ""))
                    false_parts.append(f"have {article}{word}")
                elif is_adj:
                    false_parts.append(f"are {word}")
                else:
                    false_parts.append(f"have {word}")
            if false_parts:
                if len(false_parts) == 1:
                    parts.append(f"Neither of them {false_parts[0]}.")
                else:
                    parts.append(f"Neither of them {false_parts[0]}, {', '.join(false_parts[1:-1])} nor {false_parts[-1]}.")

        if different:
            random.shuffle(different)
            diff_strs = []
            for attr, v1, v2 in different[:4]:
                raw_adjective = attr.replace("is_", "").replace("_", " ").replace("has_", "")
                if isinstance(v1, bool) and isinstance(v2, bool):
                    if v1 and not v2:
                        diff_strs.append(f"{e1_cap} is {raw_adjective} while {e2_cap} is not")
                    elif v2 and not v1:
                        diff_strs.append(f"{e2_cap} is {raw_adjective} while {e1_cap} is not")
                    else:
                        diff_strs.append(f"{e1_cap} is {raw_adjective} but {e2_cap} is also {raw_adjective}")
                elif isinstance(v1, bool):
                    diff_strs.append(f"{e1_cap} is {raw_adjective} ({v1}) while {e2_cap} has '{attr}' = {self._format_value(v2)}")
                elif isinstance(v2, bool):
                    diff_strs.append(f"{e2_cap} is {raw_adjective} ({v2}) while {e1_cap} has '{attr}' = {self._format_value(v1)}")
                else:
                    readable = self._format_attr_name(attr)
                    diff_strs.append(f"{e1_cap} has {readable} {self._format_value(v1)} while {e2_cap} has {readable} {self._format_value(v2)}")
            if diff_strs:
                parts.append("; ".join(diff_strs))

        # Add entity-specific behaviors from properties (meow for cat, bark for dog)
        entity_behaviors = []
        for ent_name, ent_data in [(e1, data1), (e2, data2)]:
            ent_cap = ent_name.title()
            props = ent_data.get("properties", {})
            descs = ent_data.get("descriptions", [])
            # Check properties for sound/behavior
            for pkey, pval in props.items():
                pkey_lower = pkey.lower()
                if any(kw in pkey_lower for kw in ("sound", "noise", "call", "vocal")):
                    entity_behaviors.append(f"{ent_cap} {self._format_value(pval)}")
            # Check descriptions for action words
            for desc in descs:
                desc_lower = desc.lower()
                for action_word in ("meow", "bark", "purr", "hiss", "chirp", "growl"):
                    if action_word in desc_lower and f"{ent_cap} {action_word}" not in " ".join(entity_behaviors):
                        entity_behaviors.append(f"{ent_cap} can {action_word}")
                        break
        if entity_behaviors and len(entity_behaviors) >= 2:
            random.shuffle(entity_behaviors)
            parts.append("For example, " + ", but ".join(entity_behaviors[:3]) + ".")

        # Clean up: strip trailing periods before joining, add single period at end
        cleaned = [p.rstrip(".") for p in parts if p.strip()]
        return ". ".join(cleaned) + "." if cleaned else None

    def answer_definition(self, entity, query_words=None):
        entity_lower = entity.lower().strip()
        data = self.entities.get(entity_lower)
        if not data:
            return None

        category = data.get("category", "unknown")
        parents = data.get("parent_categories", [])
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        descs = data.get("descriptions", [])
        similar = data.get("similar_to", [])

        # Check for category-only terms
        if category == "category_only":
            members = self.get_members_of_category(entity_lower)
            if not members:
                for parent in parents:
                    members.extend(self.get_members_of_category(parent))
            members = list(set(members))
            if members:
                member_str = ", ".join(m.title() for m in members[:5])
                return f"Examples of {entity_lower} include: {member_str}."
            return f"{entity.title()} is a classification that includes various items."

        # Build definition from facts
        bool_facts = [(a, v) for a, v in attrs.items() if isinstance(v, bool)]
        prop_facts = [(p, v) for p, v in props.items()]

        # Determine what attribute to highlight based on query
        target_attr = None
        if query_words:
            for qw in query_words:
                for attr in attrs:
                    if qw in attr or attr in qw:
                        target_attr = attr
                        break
                if target_attr:
                    break

        # If user asks about a specific attribute
        if target_attr:
            val = attrs[target_attr]
            readable = self._format_attr_name(target_attr)
            if isinstance(val, bool):
                prefix = "Yes" if val else "No"
                neg = "" if val else " not"
                return f"{prefix}, {entity_lower}{neg} {readable}."
            else:
                return f"The {readable} of a {entity_lower} is {val}."

        # Build definition from descriptions + boolean attributes
        if descs:
            base = descs[0]
        else:
            cat_str = category if category != "unknown" else "thing"
            true_formatted = [self._format_attr_with_verb(a, negative=False, singular=True) for a, v in bool_facts if v]
            false_readable = [self._format_attr_name(a) for a, v in bool_facts if not v]
            parts = [f"{entity.title()} is a {cat_str}"]
            if true_formatted:
                parts.append(f"that {', '.join(true_formatted[:-1])} and {true_formatted[-1]}" if len(true_formatted) > 1 else f"that {true_formatted[0]}")
            if false_readable:
                parts.append(f"but without {', '.join(false_readable[:-1])} or {false_readable[-1]}" if len(false_readable) > 1 else f"but without {false_readable[0]}")
            base = " ".join(parts)

        # Add distinguishing info
        extra = []
        if similar:
            sim_entity = similar[0]
            sim_data = self.entities.get(sim_entity, {})
            sim_attrs = sim_data.get("attributes", {})
            for attr, val in bool_facts:
                sim_val = sim_attrs.get(attr)
                if sim_val is not None and sim_val != val:
                    formatted = self._format_attr_with_verb(attr, negative=not val, singular=True)
                    extra.append(f"unlike a {sim_entity}, a {entity_lower} {formatted}")
                    break

        if prop_facts and not extra:
            prop_name, prop_val = prop_facts[0]
            extra.append(f"its {prop_name.replace('_', ' ')} is {prop_val}")

        if extra:
            return f"{base} {extra[0]}."
        return base

    def answer_attribute_query(self, query):
        query_lower = query.lower()

        # Handle "what is the X of Y" patterns
        m = re.search(r"what (?:is|are) the (\w+) of (?:a |an |the )?(\w+)", query_lower)
        if m:
            attr_name = m.group(1)
            entity = self.resolve_entity(m.group(2)) or m.group(2)
            return self._lookup_attribute(entity, attr_name)

        # Handle "what X is/are Y" patterns: "what color is a diamond", "what color are dogs"
        m = re.search(r"what (\w+) (?:is|are) (?:a |an |the )?(\w+)", query_lower)
        if m:
            attr_name = m.group(1)
            entity = self.resolve_entity(m.group(2)) or m.group(2)
            return self._lookup_attribute(entity, attr_name)

        # Handle "what X does/does Y do/make" patterns: "what sound does a cat make"
        m = re.search(r"what (\w+) (?:do|does|did) (?:a |an |the )?(\w+) (?:do|make)", query_lower)
        if m:
            attr_name = m.group(1)
            entity = self.resolve_entity(m.group(2)) or m.group(2)
            return self._lookup_attribute(entity, attr_name)

        # Handle "how X do/does Y verb" patterns: "how long do cats live", "how fast is a dog"
        m = re.search(r"how (\w+) (?:do|does|did|is|are|was|were) (?:a |an |the )?(\w+)", query_lower)
        if m:
            attr_name = m.group(1)
            entity = self.resolve_entity(m.group(2)) or m.group(2)
            # Map common "how X" queries to attribute names
            attr_map = {"long": "lifespan", "fast": "speed", "tall": "height",
                        "big": "size", "heavy": "weight", "old": "lifespan"}
            mapped = attr_map.get(attr_name, attr_name)
            return self._lookup_attribute(entity, mapped)

        # Handle "X of Y" shorthand: "color of a diamond"
        m = re.search(r"(\w+) of (?:a |an |the )?(\w+)", query_lower)
        if m:
            attr_name = m.group(1)
            entity = self.resolve_entity(m.group(2)) or m.group(2)
            if attr_name not in STOP_WORDS:
                return self._lookup_attribute(entity, attr_name)

        # Handle bare "entity attribute" shorthand: "diamond size", "diamond weight"
        m = re.match(r"(\w+)\s+(\w+)$", query_lower.strip())
        if m:
            first, second = m.group(1), m.group(2)
            if second not in STOP_WORDS:
                resolved = self.resolve_entity(first)
                if resolved:
                    result = self._lookup_attribute(resolved, second)
                    if result:
                        return result
            if first not in STOP_WORDS:
                resolved = self.resolve_entity(second)
                if resolved:
                    result = self._lookup_attribute(resolved, first)
                    if result:
                        return result

        return None

    def _lookup_attribute(self, entity, attr_name):
        data = self.entities.get(entity, {})
        attrs = data.get("attributes", {})
        props = data.get("properties", {})

        # Check properties FIRST (concrete values like "color: white")
        for p, v in props.items():
            readable = p.replace("_", " ")
            if attr_name == p or attr_name == readable:
                return f"The {readable} of a {entity} is {v}."
        # Fuzzy property match
        for p, v in props.items():
            readable = p.replace("_", " ")
            if attr_name in p or p in attr_name or attr_name in readable:
                return f"The {readable} of a {entity} is {v}."

        # Check attributes with fuzzy matching
        best_match = None
        best_score = 0
        for a, v in attrs.items():
            readable = self._format_attr_name(a)
            score = 0
            if attr_name == a or attr_name == readable:
                score = 3
            elif attr_name in a or attr_name in readable:
                score = 2
            elif a in attr_name or readable in attr_name:
                score = 1
            if score > best_score:
                best_score = score
                best_match = (a, v, readable)

        if best_match:
            a, v, readable = best_match
            return f"The {readable} of a {entity} is {self._format_value(v)}."

        # Check dataset
        for q, a in self.dataset_qa:
            if entity in q.lower() and attr_name in q.lower():
                return a

        return None

    def answer_multi_attribute(self, entity, attr_words):
        """Answer a multi-attribute query like 'shape and color and size of diamond'."""
        entity_lower = entity.lower()
        data = self.entities.get(entity_lower, {})
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        descs = data.get("descriptions", [])

        sentences = []
        for aw in attr_words:
            aw_lower = aw.lower().rstrip("?")
            found = False
            # Search attributes
            for a, v in attrs.items():
                readable = self._format_attr_name(a)
                if aw_lower in a or a in aw_lower or aw_lower in readable:
                    if isinstance(v, bool):
                        sentences.append(f"The {readable} of a {entity_lower} is {'yes' if v else 'no'}.")
                    else:
                        sentences.append(f"The {readable} of a {entity_lower} is {v}.")
                    found = True
                    break
            # Search properties
            if not found:
                for p, v in props.items():
                    readable = p.replace("_", " ")
                    if aw_lower in p or p in aw_lower or aw_lower in readable:
                        sentences.append(f"The {readable} of a {entity_lower} is {v}.")
                        found = True
                        break
            # Search dataset QA
            if not found:
                for q, a in self.dataset_qa:
                    if entity_lower in q.lower() and aw_lower in q.lower():
                        sentences.append(a)
                        found = True
                        break
            # Search descriptions
            if not found:
                for desc in descs:
                    if aw_lower in desc.lower():
                        sentences.append(desc)
                        found = True
                        break
        return sentences

    def resolve_entity(self, text):
        text_lower = text.lower().strip()
        # Direct match
        if text_lower in self.entities:
            return text_lower
        # Try with common prefixes removed
        for prefix in ["a ", "an ", "the ", "some ", "the "]:
            if text_lower.startswith(prefix):
                candidate = text_lower[len(prefix):]
                if candidate in self.entities:
                    return candidate
        # Try stripping trailing 's' for plurals
        if text_lower.endswith("s") and len(text_lower) > 3:
            singular = text_lower[:-1]
            if singular in self.entities:
                return singular
        # Also try removing 'es' for plurals
        if text_lower.endswith("es") and len(text_lower) > 4:
            singular = text_lower[:-2]
            if singular in self.entities:
                return singular
        # Fuzzy match
        for entity in self.entities:
            if self._is_similar(text_lower, entity):
                return entity
        return None

    def _is_similar(self, w1, w2, threshold=0.85):
        if len(w1) == 0 or len(w2) == 0:
            return False
        if w1 == w2:
            return True
        # Skip fuzzy match for short words (< 5 chars) — too many false positives
        if len(w1) < 5 or len(w2) < 5:
            return False
        m, n = len(w1), len(w2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if w1[i-1] == w2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        sim = 1 - (dp[m][n] / max(m, n))
        return sim >= threshold

    def find_ambiguous_candidates(self, text, max_gap=0.08):
        """Return a list of entity names that are all close fuzzy matches to
        `text` — used to detect when a query is genuinely ambiguous rather
        than confidently resolvable to one entity."""
        text_lower = text.lower().strip()
        scored = []
        for entity in self.entities:
            if entity == text_lower:
                return []
            if len(text_lower) < 5 or len(entity) < 5:
                continue
            m, n = len(text_lower), len(entity)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if text_lower[i-1] == entity[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            sim = 1 - (dp[m][n] / max(m, n))
            if sim >= 0.6:
                scored.append((entity, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        if len(scored) < 2:
            return []
        top_score = scored[0][1]
        candidates = [e for e, s in scored if top_score - s <= max_gap]
        return candidates if len(candidates) >= 2 else []

    def _format_value(self, val):
        if isinstance(val, bool):
            return "yes" if val else "no"
        return str(val)

    def _format_attr_name(self, attr):
        name = attr.replace("_", " ")
        # Common attribute name mappings for cleaner display
        attr_map = {
            "is color": "color",
            "is colorful": "color",
            "is shiny": "luster",
            "is hard": "hardness",
            "is precious": "value",
            "has shell": "shell",
            "has tail": "tail",
            "has fur": "fur",
            "has feathers": "feathers",
            "lays eggs": "reproduction",
            "is venomous": "venom",
            "is aquatic": "habitat",
            "is domestic": "domestication",
            "is nocturnal": "nocturnal",
            "is prey": "prey",
            "is predator": "predator",
            "has shell": "shell",
        }
        for pattern, replacement in attr_map.items():
            if name == pattern or name.startswith(pattern):
                return replacement
        # Strip leading verb prefixes for cleaner display
        for prefix in ["is ", "has ", "lays "]:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def _format_attr_with_verb(self, attr, negative=False, plural=False, singular=False):
        """Format attribute with correct verb for sentence construction.
        singular=True: 'A cat is nocturnal' (singular subject).
        plural=True: 'Cats have tails' (plural subject).
        Default: 'Cats are nocturnal' or 'A cat has a tail'."""
        name = attr.replace("_", " ")
        if name.startswith("is "):
            word = name[3:]
            # Adjectives don't take articles — check common endings + known adjectives
            adj_endings = ("y", "ous", "ive", "ful", "less", "able", "ic", "ent", "ant", "al", "id")
            known_adjs = {"domestic", "nocturnal", "venomous", "aquatic", "transparent", "shiny", "hard", "precious", "colorful", "metallic", "organic"}
            is_adj = any(word.endswith(e) for e in adj_endings) or word in known_adjs or len(word) <= 5
            if is_adj:
                if singular:
                    return f"is {word}" if not negative else f"is not {word}"
                return f"are {word}" if not negative else f"are not {word}"
            # Noun: add article for singular
            if singular:
                article = "a " if word and word[0] not in "aeiou" else ("an " if word else "")
                return f"is {article}{word}" if not negative else f"is not {article}{word}"
            if plural:
                return f"are {word}s" if not negative else f"are not {word}s"
            return f"is the {word}" if not negative else f"is not the {word}"
        if name.startswith("has "):
            word = name[4:]
            # Uncountable nouns don't take articles
            uncountable = {"fur"}
            if word in uncountable or word.endswith("s"):
                article = ""
            else:
                article = "a " if word and word[0] not in "aeiou" else ("an " if word else "")
            if singular:
                return f"has {article}{word}" if not negative else f"does not have {article}{word}"
            return f"have {word}" if not negative else f"do not have {word}"
        if name.startswith("lays "):
            word = name[5:]
            if plural:
                return f"lay {word}" if not negative else f"do not lay {word}"
            return f"lays {word}" if not negative else f"does not lay {word}"
        return name

# =========================
# CONVERSATION MEMORY
# =========================

class ConversationMemory:
    def __init__(self, knowledge_engine=None):
        self.turns = []
        self.entities_discussed = Counter()
        self.categories_discussed = Counter()
        self.topics = []
        self.user_facts_learned = []
        self.max_turns = 50
        self._kb_ref = knowledge_engine

        # State tracking: entity -> attribute -> {original, current, turn_changed}
        self.entity_state = defaultdict(dict)
        # Context clues: things the user has said about state changes
        self.context_clues = []
        # Cross-entity links mentioned in conversation
        self.entity_links = []
        # Conversation detail level: short, normal, detailed
        self.detail_level = "normal"

    def add_turn(self, user_input, ai_response, entities=None, categories=None):
        turn = {
            "user": user_input,
            "ai": ai_response,
            "entities": entities or [],
            "categories": categories or [],
            "timestamp": datetime.now().isoformat(),
            "turn_num": len(self.turns)
        }
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

        for e in entities or []:
            self.entities_discussed[e] += 1
        for c in categories or []:
            self.categories_discussed[c] += 1

        # Extract topic from user input
        topic = self._extract_topic(user_input)
        if topic:
            self.topics.append(topic)
            if len(self.topics) > 30:
                self.topics.pop(0)

        # Detect state changes from user input
        self._detect_state_changes(user_input, turn["turn_num"])

    def _extract_topic(self, text):
        words = [w.lower().rstrip("?!.,;:") for w in text.split()]
        content_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        if content_words:
            return " ".join(content_words[:3])
        return None

    def get_recent_entities(self, n=5):
        recent = []
        for turn in reversed(self.turns[-n:]):
            for e in turn.get("entities", []):
                if e not in recent:
                    recent.append(e)
        return recent

    def get_recent_topics(self, n=5):
        return self.topics[-n:] if self.topics else []

    def get_conversation_summary(self):
        if not self.turns:
            return "We haven't talked about anything yet."
        entities = list(self.entities_discussed.keys())
        categories = list(self.categories_discussed.keys())
        parts = []
        if entities:
            parts.append(f"entities: {', '.join(entities[:10])}")
        if categories:
            parts.append(f"categories: {', '.join(categories[:10])}")
        if self.topics:
            parts.append(f"topics: {', '.join(set(self.topics[-10:]))}")
        return "We've discussed " + "; ".join(parts) if parts else "General conversation."

    def was_recently_discussed(self, entity, n=3):
        for turn in reversed(self.turns[-n:]):
            if entity.lower() in [e.lower() for e in turn.get("entities", [])]:
                return True
        return False

    def get_last_turn_context(self):
        if not self.turns:
            return {}
        return self.turns[-1]

    def _detect_state_changes(self, user_input, turn_num):
        """Detect when user says an entity's attribute has changed."""
        lower = user_input.lower()
        # Patterns: "the X has been stripped", "X is now Y", "the color was removed"
        change_patterns = [
            (r"the (\w+) (?:has been|was|is now) (\w+)", None),
            (r"(\w+) (?:has been|was) (\w+)", None),
            (r"remove(?:d)? the (\w+)", None),
            (r"strip(?:ped)? the (\w+)", None),
            (r"take(?:n)? away the (\w+)", None),
            (r"no more (\w+)", None),
        ]
        # Find which entity is being discussed
        recent_entity = None
        for turn in reversed(self.turns[-3:]):
            for e in turn.get("entities", []):
                recent_entity = e.lower()
                break
            if recent_entity:
                break

        if recent_entity:
            for pat, _ in change_patterns:
                m = re.search(pat, lower)
                if m:
                    groups = m.groups()
                    attr_name = groups[0] if groups else None
                    new_val = groups[1] if len(groups) > 1 else None
                    if attr_name:
                        # Try to get original value from knowledge base
                        original_val = "unknown"
                        kb_data = self._kb_ref.entities.get(recent_entity, {})
                        kb_attrs = kb_data.get("attributes", {})
                        kb_props = kb_data.get("properties", {})
                        for a, v in kb_attrs.items():
                            readable = self._kb_ref._format_attr_name(a) if hasattr(self._kb_ref, '_format_attr_name') else a.replace("_", " ")
                            if attr_name in a or attr_name in readable:
                                if isinstance(v, bool):
                                    original_val = "yes" if v else "no"
                                else:
                                    original_val = str(v)
                                break
                        if original_val == "unknown":
                            for p, v in kb_props.items():
                                readable = p.replace("_", " ")
                                if attr_name in p or attr_name in readable:
                                    original_val = str(v)
                                    break

                        # Store original value if not already stored
                        if attr_name not in self.entity_state[recent_entity]:
                            self.entity_state[recent_entity][attr_name] = {
                                "original": original_val,
                                "current": new_val or "removed",
                                "turn_changed": turn_num
                            }
                        else:
                            self.entity_state[recent_entity][attr_name]["current"] = new_val or "removed"
                            self.entity_state[recent_entity][attr_name]["turn_changed"] = turn_num
                        self.context_clues.append({
                            "entity": recent_entity,
                            "attribute": attr_name,
                            "change": f"now {new_val or 'removed'}",
                            "turn": turn_num
                        })

        # Detect temporal references: "originally", "before", "used to be"
        if any(w in lower for w in ["originally", "before", "used to be", "initially", "at first"]):
            self.context_clues.append({
                "type": "temporal_reference",
                "text": user_input,
                "turn": turn_num
            })

    def get_entity_state(self, entity, attribute=None):
        """Get tracked state changes for an entity."""
        entity_lower = entity.lower()
        if attribute:
            return self.entity_state.get(entity_lower, {}).get(attribute.lower())
        return self.entity_state.get(entity_lower, {})

    def get_original_value(self, entity, attribute):
        """Get the original value of an attribute before any changes."""
        state = self.get_entity_state(entity, attribute)
        if state:
            return state.get("original")
        return None

    def has_state_change(self, entity, attribute=None):
        """Check if an entity has had state changes."""
        entity_lower = entity.lower()
        if attribute:
            return attribute.lower() in self.entity_state.get(entity_lower, {})
        return bool(self.entity_state.get(entity_lower))

    def get_recent_context_clues(self, n=3):
        """Get recent context clues for answer shaping."""
        return self.context_clues[-n:] if self.context_clues else []

    def get_entity_context(self, entity):
        """Get all conversation context about an entity."""
        entity_lower = entity.lower()
        context = {
            "mentioned_count": self.entities_discussed.get(entity_lower, 0),
            "recent_turns": [],
            "state_changes": self.entity_state.get(entity_lower, {}),
            "related_entities": []
        }
        for turn in reversed(self.turns[-5:]):
            if entity_lower in [e.lower() for e in turn.get("entities", [])]:
                context["recent_turns"].append({
                    "user": turn["user"],
                    "ai": turn["ai"][:100]
                })
        # Find related entities from shared conversation turns
        for turn in self.turns:
            turn_entities = [e.lower() for e in turn.get("entities", [])]
            if entity_lower in turn_entities:
                for e in turn_entities:
                    if e != entity_lower and e not in context["related_entities"]:
                        context["related_entities"].append(e)
        return context

    def get_detail_level(self):
        """Determine appropriate detail level based on conversation flow."""
        if not self.turns:
            return "normal"
        short_indicators = ["brief", "short", "tldr", "summarize", "quick", "simple", "one word"]
        long_indicators = ["detail", "explain", "elaborate", "more", "tell me everything",
                          "comprehensive", "in depth", "deep dive", "full"]
        # Check most recent turns first (most relevant)
        for turn in reversed(self.turns[-5:]):
            lower = turn["user"].lower()
            if any(w in lower for w in short_indicators):
                return "short"
            if any(w in lower for w in long_indicators):
                return "detailed"
        return "normal"

# =========================
# RESPONSE OPTIMIZER
# =========================

class ResponseOptimizer:
    def __init__(self, knowledge_engine, memory):
        self.knowledge_engine = knowledge_engine
        self.memory = memory
        self.recent_answers = []
        self.max_recent = 10
        self.confidence_threshold = 0.6
        self.candidate_pool_size = 8
        self.max_attempts = 5

    def generate_candidates(self, entity, query, num_candidates=5):
        entity_lower = entity.lower()
        candidates = []

        # Source 1: Entity descriptions
        data = self.knowledge_engine.entities.get(entity_lower, {})
        for desc in data.get("descriptions", []):
            candidates.append({
                "text": desc,
                "source": "description",
                "confidence": 0.9
            })

        # Source 2: Boolean attribute facts
        for attr, val in self.knowledge_engine.get_boolean_facts(entity_lower):
            if val:
                formatted = self.knowledge_engine._format_attr_with_verb(attr, negative=False, singular=True)
                candidates.append({
                    "text": f"A {entity_lower} {formatted}.",
                    "source": "attribute",
                    "confidence": 0.85
                })
            else:
                formatted = self.knowledge_engine._format_attr_with_verb(attr, negative=True, singular=True)
                candidates.append({
                    "text": f"A {entity_lower} {formatted}.",
                    "source": "attribute",
                    "confidence": 0.85
                })

        # Source 3: Property facts
        for prop, val in self.knowledge_engine.get_property_facts(entity_lower):
            candidates.append({
                "text": f"The {prop.replace('_', ' ')} of a {entity_lower} is {val}.",
                "source": "property",
                "confidence": 0.8
            })

        # Source 4: Dataset-backed candidates (surrounding topics, similar answers)
        dataset_results = self.knowledge_engine.search_dataset_for_context(
            entity_lower, query.lower().split()
        )
        for q, a, score in dataset_results[:5]:
            candidates.append({
                "text": a,
                "source": "dataset",
                "confidence": min(0.95, 0.5 + score * 0.1)
            })

        # Source 5: Similar entity facts (for comparison)
        similar_entities = self.knowledge_engine.find_similar_entities(entity_lower)
        for sim_name in similar_entities[:2]:
            sim_data = self.knowledge_engine.entities.get(sim_name, {})
            if sim_data:
                sim_bools = sim_data.get("attributes", {})
                entity_bools = data.get("attributes", {})
                shared_true = []
                shared_false = []
                diff = []
                for attr in set(list(sim_bools.keys()) + list(entity_bools.keys())):
                    sv = sim_bools.get(attr)
                    ev = entity_bools.get(attr)
                    if sv is not None and ev is not None:
                        if sv == ev and isinstance(sv, bool):
                            if sv:
                                shared_true.append(attr)
                            else:
                                shared_false.append(attr)
                        elif sv != ev:
                            diff.append((attr, ev, sv))

                parts = [f"{entity.title()} and {sim_name.title()} are similar"]
                if shared_true:
                    # Format shared true attributes as natural phrases
                    true_phrases = []
                    for a in shared_true[:3]:
                        if a.startswith("has_"):
                            word = a[4:].replace("_", " ")
                            true_phrases.append(f"have {word}")
                        elif a.startswith("is_"):
                            word = a[3:].replace("_", " ")
                            true_phrases.append(f"are {word}")
                        elif a.startswith("lays_"):
                            word = a[5:].replace("_", " ")
                            true_phrases.append(f"lay {word}")
                        else:
                            true_phrases.append(a.replace("_", " "))
                    if len(true_phrases) == 1:
                        parts.append(f"both {true_phrases[0]}")
                    else:
                        parts.append(f"both {', '.join(true_phrases[:-1])} and {true_phrases[-1]}")
                if shared_false:
                    # Format shared false attributes
                    false_phrases = []
                    for a in shared_false[:3]:
                        if a.startswith("has_"):
                            word = a[4:].replace("_", " ")
                            false_phrases.append(f"have {word}")
                        elif a.startswith("is_"):
                            word = a[3:].replace("_", " ")
                            false_phrases.append(f"are {word}")
                        elif a.startswith("lays_"):
                            word = a[5:].replace("_", " ")
                            false_phrases.append(f"lay {word}")
                        else:
                            false_phrases.append(a.replace("_", " "))
                    if false_phrases:
                        if len(false_phrases) == 1:
                            parts.append(f"neither {false_phrases[0]}")
                        else:
                            parts.append(f"neither {', '.join(false_phrases[:-1])} nor {false_phrases[-1]}")
                if diff:
                    attr, ev, sv = diff[0]
                    readable = self.knowledge_engine._format_attr_name(attr)
                    parts.append(f"but differ in {readable}")

                if len(parts) > 1:
                    candidates.append({
                        "text": " ".join(parts) + ".",
                        "source": "comparison",
                        "confidence": 0.75
                    })

        # Source 6: Category-based info
        category = self.knowledge_engine.get_category(entity_lower)
        parents = self.knowledge_engine.get_parent_categories(entity_lower)
        if category:
            members = self.knowledge_engine.get_members_of_category(category)
            other_members = [m for m in members if m != entity_lower]
            if other_members:
                candidates.append({
                    "text": f"{entity.title()} is in the same category as {', '.join(o.title() for o in other_members[:3])}.",
                    "source": "category",
                    "confidence": 0.7
                })

        # Source 7: Contextual expansion from recent conversation
        recent = self.memory.get_recent_entities(3)
        for re_name in recent:
            if re_name.lower() != entity_lower:
                re_data = self.knowledge_engine.entities.get(re_name.lower(), {})
                if re_data:
                    re_cat = re_data.get("category")
                    if re_cat and re_cat == category:
                        candidates.append({
                            "text": f"Like the {re_name} you asked about earlier, {entity_lower} is also a {re_cat}.",
                            "source": "context",
                            "confidence": 0.65
                        })

        # Score and rank
        for c in candidates:
            c["score"] = self._score_candidate(c, entity_lower, query)

        # Shuffle within same score tier for variety
        random.shuffle(candidates)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:num_candidates]

    def _score_candidate(self, candidate, entity, query):
        score = candidate["confidence"]
        text = candidate["text"].lower()

        # Recency penalty - stronger to ensure variety
        for recent in self.recent_answers:
            # Check for any significant overlap (not just exact match)
            recent_words = set(recent.lower().split())
            text_words = set(text.lower().split())
            overlap = len(recent_words & text_words)
            total = len(recent_words | text_words)
            if total > 0:
                similarity = overlap / total
                if similarity > 0.5:
                    score *= 0.15  # heavy penalty for similar recent answers
                elif similarity > 0.3:
                    score *= 0.4

        # Query relevance boost
        query_words = set(query.lower().split())
        text_words = set(text.split())
        overlap = len(query_words & text_words)
        score += overlap * 0.05

        # Source bonus
        source_bonus = {
            "description": 0.1,
            "dataset": 0.15,
            "attribute": 0.05,
            "comparison": 0.1,
            "property": 0.05,
            "category": 0.05,
            "context": 0.1
        }
        score += source_bonus.get(candidate["source"], 0)

        return min(1.0, score)

    def optimize_response(self, entity, query, num_responses=3):
        candidates = self.generate_candidates(entity, query, self.candidate_pool_size)

        if not candidates:
            return None

        selected = []
        seen_texts = set()

        for attempt in range(min(self.max_attempts, len(candidates))):
            for c in candidates:
                if c["text"] not in seen_texts and c["score"] >= self.confidence_threshold * 0.5:
                    selected.append(c)
                    seen_texts.add(c["text"])
                    if len(selected) >= num_responses:
                        break
            if len(selected) >= num_responses:
                break

        if not selected:
            selected = candidates[:1]

        # Track recent answers
        for s in selected:
            self.recent_answers.append(s["text"])
            if len(self.recent_answers) > self.max_recent:
                self.recent_answers.pop(0)

        return selected

# =========================
# OPPOSITE ENTITY ENGINE
# =========================

class OppositeEntityEngine:
    """Given entity (cat), find opposite/related (dog), build thinking chain."""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.opposite_cache = {}
        self.thinking_chains = {}

    def find_opposite(self, entity):
        """Find the most opposite/related entity."""
        entity_lower = entity.lower()
        if entity_lower in self.opposite_cache:
            return self.opposite_cache[entity_lower]

        data = self.ke.entities.get(entity_lower, {})
        similar = data.get("similar_to", [])
        category = data.get("category", "")
        parent_cats = data.get("parent_categories", [])
        attrs = data.get("attributes", {})

        best_opposite = None
        best_score = -1

        for other_name, other_data in self.ke.entities.items():
            if other_name == entity_lower:
                continue
            other_attrs = other_data.get("attributes", {})
            other_cats = other_data.get("parent_categories", [])
            other_similar = other_data.get("similar_to", [])

            score = 0
            # Same parent category but different specific category
            shared_cats = set(parent_cats) & set(other_cats)
            if shared_cats:
                score += len(shared_cats) * 2
            # In similar_to list
            if entity_lower in other_similar:
                score += 3
            if other_name in similar:
                score += 3
            # Different attributes (opposite traits)
            for attr in set(list(attrs.keys()) + list(other_attrs.keys())):
                v1 = attrs.get(attr)
                v2 = other_attrs.get(attr)
                if v1 is not None and v2 is not None and v1 != v2:
                    score += 1
            # Bonus for being in same category group
            if category == other_data.get("category", ""):
                score += 1

            if score > best_score:
                best_score = score
                best_opposite = other_name

        self.opposite_cache[entity_lower] = best_opposite
        return best_opposite

    def build_thinking_chain(self, entity, query):
        """Build a thinking chain: entity → opposite → comparison → expansion."""
        opposite = self.find_opposite(entity)
        if not opposite:
            return None

        entity_data = self.ke.entities.get(entity.lower(), {})
        opp_data = self.ke.entities.get(opposite.lower(), {})

        chain = {
            "entity": entity,
            "opposite": opposite,
            "entity_data": entity_data,
            "opposite_data": opp_data,
            "shared_attrs": [],
            "different_attrs": [],
            "entity_only": [],
            "opposite_only": [],
            "examples": [],
            "contrast_points": [],
        }

        e_attrs = entity_data.get("attributes", {})
        o_attrs = opp_data.get("attributes", {})

        for attr in set(list(e_attrs.keys()) + list(o_attrs.keys())):
            ev = e_attrs.get(attr)
            ov = o_attrs.get(attr)
            readable = attr.replace("_", " ").replace("is ", "").replace("has ", "")
            if ev is not None and ov is not None:
                if ev == ov:
                    chain["shared_attrs"].append((readable, ev))
                else:
                    chain["different_attrs"].append((readable, ev, ov))
            elif ev is not None:
                chain["entity_only"].append((readable, ev))
            elif ov is not None:
                chain["opposite_only"].append((readable, ov))

        # Generate contrast points
        for readable, ev, ov in chain["different_attrs"]:
            chain["contrast_points"].append(
                f"Unlike {opposite}, {entity} is{'not ' if not ev else ''}{readable}"
            )

        # Generate examples from descriptions
        e_descs = entity_data.get("descriptions", [])
        o_descs = opp_data.get("descriptions", [])
        if e_descs:
            chain["examples"].append(e_descs[0])
        if o_descs:
            chain["examples"].append(o_descs[0])

        self.thinking_chains[entity.lower()] = chain
        return chain

    def get_comparison_facts(self, entity):
        """Get facts comparing entity with its opposite."""
        chain = self.build_thinking_chain(entity, "")
        if not chain:
            return []
        facts = []
        for shared_name, shared_val in chain["shared_attrs"]:
            facts.append(f"Both {chain['entity']} and {chain['opposite']} {shared_name}.")
        for diff_name, e_val, o_val in chain["different_attrs"]:
            facts.append(f"{chain['entity']} {diff_name}: {e_val}. {chain['opposite']} {diff_name}: {o_val}.")
        return facts

# =========================
# SENTENCE INDEX
# =========================

class SentenceIndex:
    """Index all sentences with line/sentence/char tracking for direct updates."""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.sentences = []        # [{text, source, entity, line_num, sent_num, char_start, char_end, fact_type}]
        self.entity_index = {}     # entity -> [sentence_indices]
        self.fact_type_index = {}  # "true"/"false"/"unknown" -> [sentence_indices]
        self._build_index()

    def _build_index(self):
        """Build index from all KB data."""
        self.sentences = []
        sent_num = 0

        # Index descriptions
        for entity, data in self.ke.entities.items():
            descs = data.get("descriptions", [])
            for i, desc in enumerate(descs):
                char_start = sum(len(d) + 1 for d in descs[:i])
                entry = {
                    "text": desc.strip(),
                    "source": "description",
                    "entity": entity,
                    "line_num": i,
                    "sent_num": sent_num,
                    "char_start": char_start,
                    "char_end": char_start + len(desc.strip()),
                    "fact_type": self._classify_fact(desc),
                }
                self.sentences.append(entry)
                self.entity_index.setdefault(entity, []).append(sent_num)
                self.fact_type_index.setdefault(entry["fact_type"], []).append(sent_num)
                sent_num += 1

        # Index QA pairs
        for i, (q, a) in enumerate(self.ke.dataset_qa):
            entity = self._extract_entity_from_qa(q)
            char_start = 0
            entry = {
                "text": a.strip(),
                "source": "dataset_qa",
                "entity": entity,
                "line_num": i,
                "sent_num": sent_num,
                "char_start": char_start,
                "char_end": char_start + len(a.strip()),
                "fact_type": self._classify_fact(a),
            }
            self.sentences.append(entry)
            if entity:
                self.entity_index.setdefault(entity, []).append(sent_num)
            self.fact_type_index.setdefault(entry["fact_type"], []).append(sent_num)
            sent_num += 1

        # Index properties as sentences
        for entity, data in self.ke.entities.items():
            props = data.get("properties", {})
            for prop_name, prop_val in props.items():
                readable = prop_name.replace("_", " ")
                sent_text = f"The {readable} of {entity} is {prop_val}."
                entry = {
                    "text": sent_text,
                    "source": "property",
                    "entity": entity,
                    "line_num": 0,
                    "sent_num": sent_num,
                    "char_start": 0,
                    "char_end": len(sent_text),
                    "fact_type": "true",
                }
                self.sentences.append(entry)
                self.entity_index.setdefault(entity, []).append(sent_num)
                self.fact_type_index.setdefault("true", []).append(sent_num)
                sent_num += 1

    def _classify_fact(self, text):
        """Classify a sentence as true/false/unknown."""
        lower = text.lower()
        if any(neg in lower for neg in ["not ", "never ", "no ", "cannot", "can't", "don't", "doesn't", "isn't", "aren't", "wasn't"]):
            return "false"
        if any(affirm in lower for affirm in ["is a", "are ", "has ", "can ", "do ", "does ", "will ", "was "]):
            return "true"
        return "unknown"

    def _extract_entity_from_qa(self, question):
        """Extract entity name from a QA question."""
        lower = question.lower()
        for entity in self.ke.entities:
            if entity in lower:
                return entity
        return None

    def get_entity_facts(self, entity, fact_type=None):
        """Get all facts about an entity, optionally filtered by type."""
        entity_lower = entity.lower()
        indices = self.entity_index.get(entity_lower, [])
        results = []
        for idx in indices:
            if idx < len(self.sentences):
                s = self.sentences[idx]
                if fact_type is None or s["fact_type"] == fact_type:
                    results.append(s)
        return results

    def get_facts_for_query(self, query_words, entity=None):
        """Get sentences matching query words."""
        results = []
        for s in self.sentences:
            if entity and s["entity"] != entity.lower():
                continue
            s_lower = s["text"].lower()
            matches = sum(1 for w in query_words if w in s_lower)
            if matches >= 1:
                results.append({**s, "match_score": matches / len(query_words) if query_words else 0})
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results

    def update_sentence(self, sent_num, new_text):
        """Update a specific sentence by number."""
        if sent_num < len(self.sentences):
            old = self.sentences[sent_num]
            self.sentences[sent_num]["text"] = new_text
            self.sentences[sent_num]["fact_type"] = self._classify_fact(new_text)
            return {"old": old["text"], "new": new_text, "entity": old["entity"]}
        return None

    def get_stats(self, entity):
        """Get fact statistics for an entity."""
        facts = self.get_entity_facts(entity)
        true_count = sum(1 for f in facts if f["fact_type"] == "true")
        false_count = sum(1 for f in facts if f["fact_type"] == "false")
        unknown_count = sum(1 for f in facts if f["fact_type"] == "unknown")
        total = len(facts)
        return {
            "total": total,
            "true": true_count,
            "false": false_count,
            "unknown": unknown_count,
            "true_pct": (true_count / total * 100) if total else 0,
            "false_pct": (false_count / total * 100) if total else 0,
        }

# =========================
# PROPORTION SCORER
# =========================

class ProportionScorer:
    """Score how much data supports a claim about an entity."""

    def __init__(self, sentence_index, knowledge_engine):
        self.si = sentence_index
        self.ke = knowledge_engine

    def score_claim(self, entity, claim_keywords):
        """Score what proportion of data supports a claim."""
        entity_lower = entity.lower()
        all_facts = self.si.get_entity_facts(entity_lower)
        if not all_facts:
            return {"score": 0, "total": 0, "supporting": 0, "opposing": 0, "confidence": "none"}

        supporting = 0
        opposing = 0
        total_checked = 0

        # Also check entity attributes directly
        data = self.ke.entities.get(entity_lower, {})
        attrs = data.get("attributes", {})
        props = data.get("properties", {})

        # Check attributes for matching keywords
        for attr_name, attr_val in attrs.items():
            readable = attr_name.replace("_", " ").replace("is ", "").replace("has ", "")
            matches = any(kw in readable or kw in attr_name for kw in claim_keywords)
            if matches:
                total_checked += 1
                if isinstance(attr_val, bool):
                    if attr_val:
                        supporting += 1
                    else:
                        opposing += 1

        # Check properties for matching keywords
        for prop_name, prop_val in props.items():
            readable = prop_name.replace("_", " ")
            matches = any(kw in readable or kw in prop_name for kw in claim_keywords)
            if matches:
                total_checked += 1
                supporting += 0.5  # Properties generally support existence

        # Check sentence facts
        for fact in all_facts:
            text_lower = fact["text"].lower()
            matches = any(kw in text_lower for kw in claim_keywords)
            if matches:
                total_checked += 1
                if fact["fact_type"] == "true":
                    supporting += 1
                elif fact["fact_type"] == "false":
                    opposing += 1
                else:
                    supporting += 0.5

        # Use entity count as total if we found attribute matches
        if total_checked == 0:
            # Fallback: check all boolean attributes
            for attr_name, attr_val in attrs.items():
                if isinstance(attr_val, bool):
                    total_checked += 1
                    if attr_val:
                        supporting += 1

        total = total_checked if total_checked > 0 else len(all_facts)
        score = supporting / total if total else 0

        if score >= 0.7:
            confidence = "high"
        elif score >= 0.4:
            confidence = "moderate"
        elif score > 0:
            confidence = "low"
        else:
            confidence = "none"

        return {
            "score": score,
            "total": total,
            "supporting": supporting,
            "opposing": opposing,
            "confidence": confidence,
            "percentage": f"{int(score * 100)}%",
        }

    def compare_entities(self, entity1, entity2, claim_keywords):
        """Compare how much data supports a claim for two entities."""
        s1 = self.score_claim(entity1, claim_keywords)
        s2 = self.score_claim(entity2, claim_keywords)
        return {
            entity1: s1,
            entity2: s2,
            "winner": entity1 if s1["score"] > s2["score"] else entity2,
            "difference": abs(s1["score"] - s2["score"]),
        }

# =========================
# DYNAMIC ADJECTIVE/VERB SELECTOR
# =========================

class DynamicAdjVerbSelector:
    """Scan KB for adjectives and verbs related to an entity, select most relevant."""

    ADJ_PATTERNS = {
        "size": ["small", "large", "tiny", "big", "huge", "medium"],
        "speed": ["fast", "slow", "quick", "rapid", "sluggish"],
        "danger": ["dangerous", "safe", "harmless", "venomous", "aggressive", "vicious", "gentle"],
        "temperature": ["hot", "cold", "warm", "cool", "freezing"],
        "texture": ["smooth", "rough", "soft", "hard", "sharp"],
        "intelligence": ["smart", "intelligent", "clever", "dull", "cunning"],
        "strength": ["strong", "weak", "powerful", "fragile", "sturdy"],
        "social": ["social", "solitary", "friendly", "aggressive", "domestic", "wild"],
    }

    VERB_PATTERNS = {
        "movement": ["run", "walk", "swim", "fly", "climb", "crawl", "hop"],
        "attack": ["bite", "scratch", "attack", "chase", "hunt", "pounce"],
        "communicate": ["bark", "meow", "chirp", "roar", "hiss"],
        "consume": ["eat", "drink", "hunt", "forage", "graze"],
        "social": ["play", "bond", "fight", "protect", "defend"],
    }

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.entity_adj_cache = {}
        self.entity_verb_cache = {}

    def scan_entity(self, entity):
        """Scan entity data for relevant adjectives and verbs."""
        entity_lower = entity.lower()
        if entity_lower in self.entity_adj_cache:
            return self.entity_adj_cache[entity_lower], self.entity_verb_cache.get(entity_lower, [])

        data = self.ke.entities.get(entity_lower, {})
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        descs = data.get("descriptions", [])

        found_adj = []
        found_verbs = []

        # Scan attributes for adjectives
        for attr_name, attr_val in attrs.items():
            readable = attr_name.replace("_", " ").replace("is ", "").replace("has ", "")
            for cat, words in self.ADJ_PATTERNS.items():
                if readable in words or any(w in readable for w in words):
                    found_adj.append({"word": readable, "category": cat, "value": attr_val, "source": "attribute"})
            # Boolean attrs suggest adjectives
            if isinstance(attr_val, bool):
                if attr_val:
                    found_adj.append({"word": readable, "category": "quality", "value": True, "source": "attribute"})
                else:
                    found_adj.append({"word": f"not {readable}", "category": "quality", "value": False, "source": "attribute"})

        # Scan descriptions for verbs and adjectives
        for desc in descs:
            words = desc.lower().split()
            for w in words:
                clean = w.strip(".,!?;:")
                for cat, verbs in self.VERB_PATTERNS.items():
                    if clean in verbs:
                        found_verbs.append({"word": clean, "category": cat, "source": "description"})
                for cat, adj_words in self.ADJ_PATTERNS.items():
                    if clean in adj_words:
                        found_adj.append({"word": clean, "category": cat, "source": "description"})

        # Scan properties for descriptive words
        for prop_name, prop_val in props.items():
            if isinstance(prop_val, str):
                words = prop_val.lower().split()
                for w in words:
                    clean = w.strip(".,!?;:")
                    for cat, adj_words in self.ADJ_PATTERNS.items():
                        if clean in adj_words:
                            found_adj.append({"word": clean, "category": cat, "source": "property"})

        # Deduplicate
        seen_adj = set()
        unique_adj = []
        for a in found_adj:
            if a["word"] not in seen_adj:
                seen_adj.add(a["word"])
                unique_adj.append(a)

        seen_verb = set()
        unique_verbs = []
        for v in found_verbs:
            if v["word"] not in seen_verb:
                seen_verb.add(v["word"])
                unique_verbs.append(v)

        self.entity_adj_cache[entity_lower] = unique_adj
        self.entity_verb_cache[entity_lower] = unique_verbs
        return unique_adj, unique_verbs

    def select_for_query(self, entity, query_words, max_adj=3, max_verbs=3):
        """Select most relevant adj/verbs for the current query."""
        adj, verbs = self.scan_entity(entity)
        query_set = set(query_words)

        # Score relevance
        for a in adj:
            cat_words = set(self.ADJ_PATTERNS.get(a["category"], []))
            a["relevance"] = len(query_set & cat_words) + (2 if a["word"] in query_words else 0)

        for v in verbs:
            cat_words = set(self.VERB_PATTERNS.get(v["category"], []))
            v["relevance"] = len(query_set & cat_words) + (2 if v["word"] in query_words else 0)

        adj.sort(key=lambda x: x["relevance"], reverse=True)
        verbs.sort(key=lambda x: x["relevance"], reverse=True)

        return adj[:max_adj], verbs[:max_verbs]

# =========================
# CUSTOM COMPOSER
# =========================

class CustomComposer:
    """Compose custom multi-sentence answers by cross-referencing KB facts."""

    def __init__(self, knowledge_engine, sentence_index, proportion_scorer, adj_selector, opposite_engine):
        self.ke = knowledge_engine
        self.si = sentence_index
        self.ps = proportion_scorer
        self.asel = adj_selector
        self.oe = opposite_engine

    def compose_answer(self, entity, query_words, max_sentences=4):
        """Compose a custom answer from multiple KB sources."""
        entity_lower = entity.lower()
        data = self.ke.entities.get(entity_lower, {})
        if not data:
            return None

        sentences = []
        used_facts = set()

        # Extract meaningful claim from query words
        stop = {"a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
                "can", "could", "would", "should", "has", "have", "had", "what",
                "how", "why", "when", "where", "who", "which", "that", "this",
                "of", "in", "on", "at", "to", "for", "with", "by", "from",
                "and", "or", "but", "not", "no", "yes", "so", "if", "it", "its",
                "make", "makes", "made", "be", "being", "been", "compare", "between"}
        claim_words = [w for w in query_words if w.lower() not in stop and w.lower() != entity_lower]
        claim = " ".join(claim_words[:3]) if claim_words else " ".join(query_words[:2])

        # Step 1: Get proportion score for the query
        prop_score = self.ps.score_claim(entity_lower, claim_words if claim_words else query_words)

        # Step 2: Get relevant adjectives and verbs
        adj_list, verb_list = self.asel.select_for_query(entity_lower, query_words)

        # Step 3: Start with a direct answer sentence
        if prop_score["confidence"] in ("high", "moderate"):
            pct = prop_score["percentage"]
            if prop_score["supporting"] > prop_score["opposing"]:
                sentences.append(
                    f"Yes, {entity_lower} is {claim} "
                    f"(supported by {pct} of available data)."
                )
            else:
                sentences.append(
                    f"No, {entity_lower} is not {claim} "
                    f"(only {pct} of data supports this)."
                )
        else:
            # Use description as starting point
            descs = data.get("descriptions", [])
            if descs:
                sentences.append(descs[0].strip())
                used_facts.add(descs[0].strip()[:50])

        # Step 4: Add supporting facts with adjectives
        if adj_list:
            top_adj = [a["word"] for a in adj_list[:2]]
            sentences.append(
                f"{entity_lower.title()} is {' and '.join(top_adj)}."
            )

        # Step 5: Add verb-based actions
        if verb_list:
            top_verbs = [v["word"] for v in verb_list[:2]]
            sentences.append(
                f"It can {' and '.join(top_verbs)}."
            )

        # Step 6: Add cross-reference from opposite entity
        opposite = self.oe.find_opposite(entity_lower)
        if opposite and len(sentences) < max_sentences:
            opp_data = self.ke.entities.get(opposite.lower(), {})
            opp_descs = opp_data.get("descriptions", [])
            if opp_descs:
                opp_adj, opp_verbs = self.asel.select_for_query(opposite.lower(), query_words, max_adj=1, max_verbs=1)
                if opp_adj:
                    sentences.append(
                        f"In contrast, {opposite} is {opp_adj[0]['word']}."
                    )
                elif opp_descs:
                    # Take a useful sentence from opposite
                    for d in opp_descs:
                        d_lower = d.lower()
                        if any(kw in d_lower for kw in query_words):
                            sentences.append(f"By comparison, {d.strip()}")
                            break

        # Step 7: Add dataset QA facts not already used
        for q, a in self.ke.dataset_qa:
            if len(sentences) >= max_sentences:
                break
            a_lower = a.lower()
            if entity_lower in q.lower():
                a_short = a.strip()[:80]
                if a_short[:30] not in used_facts:
                    # Only add if it adds new info
                    if not any(a_lower[:30] in s.lower() for s in sentences):
                        sentences.append(a.strip())
                        used_facts.add(a_short[:30])

        # Trim to max
        sentences = sentences[:max_sentences]

        if not sentences:
            return None

        return " ".join(sentences)

    def compose_with_examples(self, entity, query_words, max_sentences=4):
        """Compose answer with real-world examples and cross-references."""
        entity_lower = entity.lower()
        data = self.ke.entities.get(entity_lower, {})
        if not data:
            return None

        sentences = []
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        descs = data.get("descriptions", [])

        # Extract meaningful claim from query words (filter out entity name and stop words)
        stop = {"a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
                "can", "could", "would", "should", "has", "have", "had", "what",
                "how", "why", "when", "where", "who", "which", "that", "this",
                "of", "in", "on", "at", "to", "for", "with", "by", "from",
                "and", "or", "but", "not", "no", "yes", "so", "if", "it", "its",
                "make", "makes", "made", "be", "being", "been", "compare", "between"}
        claim_words = [w for w in query_words if w.lower() not in stop and w.lower() != entity_lower]
        claim = " ".join(claim_words[:3]) if claim_words else None

        # If no meaningful claim words, this is a definition query — use descriptions + hierarchy
        is_definition = claim is None or claim.strip() == ""

        if is_definition:
            # Build a definition answer from descriptions and KB data
            if descs:
                sentences.append(descs[0].strip())
            # Add key attributes
            key_attrs = []
            if attrs.get("is_predator"):
                key_attrs.append("predator")
            if attrs.get("is_prey"):
                key_attrs.append("prey")
            if attrs.get("is_domestic"):
                key_attrs.append("domesticated")
            if attrs.get("has_fur") or attrs.get("has_fur_or_hair"):
                key_attrs.append("has fur")
            if attrs.get("is_warm_blooded"):
                key_attrs.append("warm-blooded")
            if key_attrs:
                sentences.append(f"A {entity_lower} is {', '.join(key_attrs)}.")
            # Add physical properties
            if "weight_kg" in props:
                sentences.append(f"It weighs around {props['weight_kg']} kg.")
            if "lifespan_years" in props:
                sentences.append(f"Typical lifespan is {props['lifespan_years']} years.")
            # Add dataset QA pairs as examples
            for q, a in self.ke.dataset_qa:
                if entity_lower in q.lower() and len(sentences) < max_sentences:
                    if a.strip() not in " ".join(sentences):
                        sentences.append(a.strip())
            result = " ".join(sentences[:max_sentences])
            return result if result else None

        # Get proportion
        prop_score = self.ps.score_claim(entity_lower, claim_words if claim_words else query_words)

        # Direct answer with confidence
        if prop_score["confidence"] == "high":
            if prop_score["supporting"] > prop_score["opposing"]:
                sentences.append(
                    f"Yes, {entity_lower} is {claim}. "
                    f"Out of {prop_score['total']} facts, {int(prop_score['supporting'])} support this "
                    f"({prop_score['percentage']})."
                )
            else:
                sentences.append(
                    f"No, {entity_lower} is generally not {claim}. "
                    f"Only {prop_score['percentage']} of data supports this."
                )
        elif prop_score["confidence"] == "moderate":
            sentences.append(
                f"It appears that {entity_lower} is {claim}, "
                f"though with moderate confidence ({prop_score['percentage']})."
            )
        elif descs:
            sentences.append(descs[0].strip())

        # Add example from KB properties
        if "sound" in props:
            sentences.append(f"For example, a {entity_lower} makes a {props['sound']} sound.")
        if "speed_kmh" in props:
            sentences.append(f"It can reach speeds of {props['speed_kmh']} km/h.")

        # Cross-reference with opposite
        opposite = self.oe.find_opposite(entity_lower)
        if opposite and len(sentences) < max_sentences:
            opp_data = self.ke.entities.get(opposite.lower(), {})
            opp_attrs = opp_data.get("attributes", {})
            # Find contrasting attributes
            for attr in attrs:
                if attr in opp_attrs and attrs[attr] != opp_attrs[attr]:
                    readable = attr.replace("_", " ").replace("is ", "").replace("has ", "")
                    if attrs[attr]:
                        sentences.append(
                            f"Unlike {opposite}, {entity_lower} {readable}."
                        )
                    else:
                        sentences.append(
                            f"Unlike {opposite}, {entity_lower} does not {readable}."
                        )
                    break

        # Add QA-based evidence
        for q, a in self.ke.dataset_qa:
            if len(sentences) >= max_sentences:
                break
            if entity_lower in q.lower():
                if not any(a.strip()[:30] in s for s in sentences):
                    sentences.append(a.strip())
                    break

        return " ".join(sentences[:max_sentences]) if sentences else None

# =========================
# SENTENCE WATCHER AGENT
# =========================

class AgentSentenceWatcher:
    """Assign agents to watch specific sentences, track changes."""

    def __init__(self, knowledge_engine, sentence_index):
        self.ke = knowledge_engine
        self.si = sentence_index
        self.watchers = {}         # agent_id -> {entity, sent_nums, last_check, alerts}
        self.favorites = {}        # entity -> [{text, score, timestamp}]
        self.next_agent_id = 1

    def assign_watcher(self, entity, sent_nums=None, agent_name=None):
        """Assign an agent to watch sentences about an entity."""
        agent_id = agent_name or f"watcher_{self.next_agent_id}"
        self.next_agent_id += 1

        if sent_nums is None:
            indices = self.si.entity_index.get(entity.lower(), [])
            sent_nums = indices[:5]  # Watch up to 5 sentences

        self.watchers[agent_id] = {
            "entity": entity.lower(),
            "sent_nums": sent_nums,
            "last_check": 0,
            "alerts": [],
            "baseline": [self.si.sentences[s]["text"] if s < len(self.si.sentences) else "" for s in sent_nums],
        }
        return agent_id

    def check_for_changes(self, agent_id):
        """Check if watched sentences have changed."""
        watcher = self.watchers.get(agent_id)
        if not watcher:
            return None

        changes = []
        for i, sent_num in enumerate(watcher["sent_nums"]):
            if sent_num < len(self.si.sentences):
                current = self.si.sentences[sent_num]["text"]
                baseline = watcher["baseline"][i] if i < len(watcher["baseline"]) else ""
                if current != baseline:
                    changes.append({
                        "sent_num": sent_num,
                        "old": baseline,
                        "new": current,
                    })

        watcher["last_check"] += 1
        return changes

    def save_favorite(self, entity, text, score=1.0):
        """Save a favorite response for an entity."""
        entity_lower = entity.lower()
        if entity_lower not in self.favorites:
            self.favorites[entity_lower] = []
        self.favorites[entity_lower].append({
            "text": text,
            "score": score,
            "timestamp": len(self.si.sentences),
        })
        # Keep top 5
        self.favorites[entity_lower].sort(key=lambda x: x["score"], reverse=True)
        self.favorites[entity_lower] = self.favorites[entity_lower][:5]

    def get_favorites(self, entity):
        """Get favorite responses for an entity."""
        return self.favorites.get(entity.lower(), [])

    def verify_favorites(self, entity):
        """Check if favorites are still valid against current KB."""
        entity_lower = entity.lower()
        favs = self.favorites.get(entity_lower, [])
        verified = []
        for fav in favs:
            text = fav["text"].lower()
            # Check if key facts still hold
            data = self.ke.entities.get(entity_lower, {})
            attrs = data.get("attributes", {})
            still_valid = True
            for attr, val in attrs.items():
                readable = attr.replace("_", " ").replace("is ", "").replace("has ", "")
                if readable in text:
                    if isinstance(val, bool):
                        if val and f"not {readable}" in text:
                            still_valid = False
                        elif not val and f"not {readable}" not in text and readable in text:
                            still_valid = False
            if still_valid:
                verified.append(fav)
        self.favorites[entity_lower] = verified
        return verified

    def get_all_watchers(self):
        """Get status of all watchers."""
        return {aid: {**w, "status": "active"} for aid, w in self.watchers.items()}

# =========================
# BACKWARD CHAIN BUILDER
# =========================

class BackwardChainBuilder:
    """Build answers by working backward from the conclusion."""

    def __init__(self, knowledge_engine, sentence_index, proportion_scorer):
        self.ke = knowledge_engine
        self.si = sentence_index
        self.ps = proportion_scorer

    def build_chain(self, entity, conclusion_keywords, max_depth=3):
        """Build a backward chain from conclusion to supporting facts."""
        entity_lower = entity.lower()
        chain = {
            "conclusion": None,
            "supporting_facts": [],
            "confidence": 0,
            "depth": 0,
            "branches": [],
        }

        # Extract meaningful claim from keywords
        stop = {"a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
                "can", "could", "would", "should", "has", "have", "had", "what",
                "how", "why", "when", "where", "who", "which", "that", "this",
                "of", "in", "on", "at", "to", "for", "with", "by", "from",
                "and", "or", "but", "not", "no", "yes", "so", "if", "it", "its",
                "make", "makes", "made", "be", "being", "been", "compare", "between"}
        claim_words = [w for w in conclusion_keywords if w.lower() not in stop and w.lower() != entity_lower]
        claim = " ".join(claim_words[:3]) if claim_words else " ".join(conclusion_keywords[:2])

        # Step 1: Find the conclusion
        prop_score = self.ps.score_claim(entity_lower, claim_words if claim_words else conclusion_keywords)

        if prop_score["confidence"] in ("high", "moderate"):
            chain["conclusion"] = f"{entity_lower} is {claim}"
            chain["confidence"] = prop_score["score"]
        else:
            data = self.ke.entities.get(entity_lower, {})
            descs = data.get("descriptions", [])
            if descs:
                chain["conclusion"] = descs[0].strip()
                chain["confidence"] = 0.5

        # Step 2: Find supporting facts (work backward)
        entity_facts = self.si.get_entity_facts(entity_lower)
        for fact in entity_facts:
            if chain["depth"] >= max_depth:
                break
            text_lower = fact["text"].lower()
            relevance = sum(1 for kw in conclusion_keywords if kw in text_lower)
            if relevance > 0:
                chain["supporting_facts"].append({
                    "text": fact["text"],
                    "source": fact["source"],
                    "relevance": relevance,
                    "fact_type": fact["fact_type"],
                })
                chain["depth"] += 1

        # Step 3: Find branch facts (related but different aspects)
        for fact in entity_facts:
            text_lower = fact["text"].lower()
            if not any(kw in text_lower for kw in conclusion_keywords):
                if len(chain["branches"]) < 2:
                    chain["branches"].append(fact["text"])

        return chain

    def compose_from_chain(self, chain):
        """Compose a response from a backward chain."""
        parts = []
        if chain["conclusion"]:
            parts.append(chain["conclusion"] + ".")

        for sf in chain.get("supporting_facts", []):
            parts.append(sf["text"].strip())

        for b in chain.get("branches", []):
            parts.append(f"Additionally, {b.strip()}")

        return " ".join(parts[:4]) if parts else None

# =========================
# CONTEXTUAL REASONER
# =========================
# Builds a full context bag from query + user state + entity states + history + KB

class ContextualReasoner:
    """Analyze user intention, check data validity, compute relative context."""
    def __init__(self, knowledge_engine, conversation_memory):
        self.ke = knowledge_engine
        self.mem = conversation_memory
        self.location_cache = {}  # entity -> location
        self.weather_cache = {}   # location -> conditions
        self.time_context = {}    # entity -> last_known_state

    def build_context(self, query, entities):
        """Build a full context bag for reasoning."""
        ctx = {
            "query": query,
            "entities": entities,
            "intention": self._detect_intention(query),
            "locations": self._extract_locations(query),
            "weather": {},
            "entity_states": {},
            "history_facts": [],
            "relative_checks": [],
            "data_validity": True,
            "math_values": {},
        }
        for e in entities:
            e_lower = e.lower()
            ctx["entity_states"][e_lower] = self._get_entity_state(e_lower)
            ctx["locations"].extend(self._get_entity_location(e_lower))
            # Pull history facts about this entity
            if hasattr(self.mem, 'turns'):
                for turn in self.mem.turns[-10:]:
                    if e_lower in turn.get("text", "").lower() or e_lower in str(turn.get("entities", [])).lower():
                        ctx["history_facts"].append(turn.get("text", "")[:100])
        # Deduplicate locations
        ctx["locations"] = list(set(ctx["locations"]))
        # Weather check for locations
        for loc in ctx["locations"]:
            ctx["weather"][loc] = self._check_weather(loc)
        # Math extraction
        ctx["math_values"] = self._extract_math(query)
        # Relative checks (range, proximity, etc.)
        ctx["relative_checks"] = self._compute_relative(ctx)
        return ctx

    def _extract_locations(self, query):
        """Extract location names from query text."""
        known_locations = {
            "arizona": "arizona", "home": "home", "outside": "outdoor",
            "indoor": "indoor", "park": "park", "forest": "forest",
            "city": "city", "beach": "beach", "desert": "desert",
            "mountain": "mountain", "farm": "farm", "zoo": "zoo",
            "texas": "texas", "california": "california", "new york": "new york",
            "florida": "florida", "colorado": "colorado", "nevada": "nevada",
            "utah": "utah", "new mexico": "new mexico",
        }
        q_lower = query.lower()
        found = []
        for loc_key, loc_val in known_locations.items():
            if loc_key in q_lower:
                found.append(loc_val)
        return found

    def _detect_intention(self, query):
        q = query.lower()
        if any(w in q for w in ["what if", "what happens", "hypothetical", "meets", "encounters"]):
            return "hypothetical"
        if any(w in q for w in ["how to", "how do", "help me", "suggest", "recommend", "what should i", "should i", "what do i do"]):
            return "advice"
        if any(w in q for w in ["is it", "is there", "does it", "can it"]):
            return "fact_check"
        if any(w in q for w in ["bring", "pack", "need", "take", "prepare"]):
            return "preparation"
        if any(w in q for w in ["why", "because", "reason"]):
            return "explanation"
        if any(w in q for w in ["compare", "versus", "or"]):
            return "comparison"
        if any(w in q for w in ["will", "going to", "predict", "forecast", "happen"]):
            return "prediction"
        if any(w in q for w in ["health", "sick", "injured", "hurt", "afraid", "fear"]):
            return "health"
        if any(w in q for w in ["where", "location", "place"]):
            return "location"
        return "general"

    def _get_entity_state(self, entity):
        data = self.ke.entities.get(entity, {})
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        descs = data.get("descriptions", [])
        return {
            "attributes": attrs,
            "properties": props,
            "descriptions": descs[:2],
            "category": data.get("category", "unknown"),
            "similar_to": data.get("similar_to", []),
        }

    def _get_entity_location(self, entity):
        # Check if entity has known location from conversation
        if hasattr(self.mem, 'turns'):
            for turn in reversed(self.mem.turns[-20:]):
                text = turn.get("text", "").lower()
                if entity in text:
                    locs = ["arizona", "home", "outside", "inside", "park", "forest", "city"]
                    found = [l for l in locs if l in text]
                    if found:
                        return found
        return []

    def _check_weather(self, location):
        # Synthesize weather from KB facts + common knowledge
        weather_facts = []
        for q, a in self.ke.dataset_qa:
            if location.lower() in q.lower() or location.lower() in a.lower():
                if any(w in a.lower() for w in ["rain", "snow", "hot", "cold", "wind", "storm", "weather"]):
                    weather_facts.append(a[:100])
        # Common knowledge fallback
        known_weather = {
            "arizona": {"temp": "hot", "rain": "low", "season": "dry"},
            "home": {"temp": "varies", "rain": "varies"},
            "forest": {"temp": "cool", "rain": "moderate"},
        }
        base = known_weather.get(location.lower(), {"temp": "unknown", "rain": "unknown"})
        return {**base, "facts": weather_facts[:2]}

    def _extract_math(self, query):
        values = {}
        import re
        nums = re.findall(r'(\d+)\s*(?:kph|mph|%|km/h|m/s|km|mi|m|ft)', query.lower())
        for n in nums:
            values["numeric"] = int(n)
        if "%" in query:
            pct = re.search(r'(\d+)%', query)
            if pct:
                values["percentage"] = int(pct.group(1))
        return values

    def _compute_relative(self, ctx):
        checks = []
        for e in ctx["entities"]:
            e_lower = e.lower()
            # Check if entity is close to other entities
            for other in ctx["entities"]:
                if other.lower() != e_lower:
                    data = self.ke.entities.get(e_lower, {})
                    if other.lower() in data.get("similar_to", []):
                        checks.append(f"{e_lower} is related to {other}")
            # Check health relative to situation
            if ctx["weather"]:
                for loc, w in ctx["weather"].items():
                    if w.get("rain") in ("high", "heavy"):
                        checks.append(f"{e_lower} may be affected by rain at {loc}")
        return checks

# =========================
# FACT CHECKER
# =========================

class FactChecker:
    """Multi-source fact verification with confidence scoring."""
    def __init__(self, knowledge_engine, sentence_index):
        self.ke = knowledge_engine
        self.si = sentence_index

    def check_fact(self, claim, entity=None):
        """Check a fact claim against KB and dataset. Returns verdict + confidence."""
        claim_lower = claim.lower()
        evidence = []
        support = 0
        oppose = 0

        # Check KB attributes
        if entity:
            data = self.ke.entities.get(entity.lower(), {})
            for attr, val in data.get("attributes", {}).items():
                readable = attr.replace("_", " ").replace("is ", "").replace("has ", "")
                if readable in claim_lower or attr in claim_lower:
                    if isinstance(val, bool):
                        if val:
                            support += 1
                            evidence.append(f"KB attribute: {readable} = True")
                        else:
                            oppose += 1
                            evidence.append(f"KB attribute: {readable} = False")

        # Check dataset QA
        for q, a in self.ke.dataset_qa:
            q_lower = q.lower()
            a_lower = a.lower()
            if entity and entity.lower() not in q_lower:
                continue
            claim_words = set(re.findall(r'\w+', claim_lower))
            qa_words = set(re.findall(r'\w+', q_lower + " " + a_lower))
            overlap = len(claim_words & qa_words)
            if overlap >= 2:
                if "not " in a_lower or "no " in a_lower:
                    oppose += 0.5
                else:
                    support += 0.5
                evidence.append(f"Dataset: {a[:80]}")

        # Check sentence index
        if self.si:
            for s in self.si.sentences:
                s_words = set(re.findall(r'\w+', s["text"].lower()))
                claim_words = set(re.findall(r'\w+', claim_lower))
                if len(s_words & claim_words) >= 2:
                    if s["fact_type"] == "true":
                        support += 0.5
                    elif s["fact_type"] == "false":
                        oppose += 0.5
                    evidence.append(f"Index: {s['text'][:80]}")

        total = support + oppose
        if total == 0:
            return {"verdict": "unknown", "confidence": 0, "evidence": []}

        confidence = support / total if total else 0
        if confidence >= 0.7:
            verdict = "true"
        elif confidence <= 0.3:
            verdict = "false"
        else:
            verdict = "uncertain"

        return {"verdict": verdict, "confidence": confidence, "evidence": evidence[:5]}

    def cross_reference(self, entity, topic):
        """Cross-reference entity with topic across KB sources."""
        results = []
        entity_lower = entity.lower()
        topic_lower = topic.lower()

        # Check entity descriptions
        data = self.ke.entities.get(entity_lower, {})
        for desc in data.get("descriptions", []):
            if topic_lower in desc.lower():
                results.append({"source": "description", "text": desc, "relevance": 0.9})

        # Check dataset QA
        for q, a in self.ke.dataset_qa:
            if entity_lower in q.lower() and topic_lower in a.lower():
                results.append({"source": "dataset", "text": a[:100], "relevance": 0.8})
            elif entity_lower in a.lower() and topic_lower in a.lower():
                results.append({"source": "dataset", "text": a[:100], "relevance": 0.7})

        # Check similar entities
        for sim in data.get("similar_to", []):
            sim_data = self.ke.entities.get(sim.lower(), {})
            for desc in sim_data.get("descriptions", []):
                if topic_lower in desc.lower():
                    results.append({"source": f"similar({sim})", "text": desc, "relevance": 0.6})

        return sorted(results, key=lambda x: x["relevance"], reverse=True)

# =========================
# PROACTIVE ADVISOR
# =========================

class ProactiveAdvisor:
    """Generate proactive suggestions based on context (umbrella, tent, etc.)."""
    def __init__(self, knowledge_engine, fact_checker, contextual_reasoner):
        self.ke = knowledge_engine
        self.fc = fact_checker
        self.cr = contextual_reasoner
        self.suggestion_history = []  # Track what was suggested

    def generate_suggestions(self, context, entities):
        """Generate proactive suggestions based on full context."""
        suggestions = []
        intention = context.get("intention", "general")
        weather = context.get("weather", {})
        entity_states = context.get("entity_states", {})
        locations = context.get("locations", [])
        query = context.get("query", "").lower()

        # Weather-based suggestions
        for loc, w in weather.items():
            rain_level = w.get("rain", "unknown")
            if rain_level in ("high", "heavy", "moderate"):
                suggestions.append({"item": "umbrella", "reason": f"It may rain at {loc}", "confidence": 0.8, "priority": "high"})
                suggestions.append({"item": "rainboots", "reason": f"Wet conditions expected at {loc}", "confidence": 0.7, "priority": "medium"})
                suggestions.append({"item": "raincoat", "reason": f"Rain protection needed at {loc}", "confidence": 0.75, "priority": "medium"})
            if w.get("temp") == "hot":
                suggestions.append({"item": "water bottle", "reason": f"Hot weather at {loc}", "confidence": 0.8, "priority": "high"})
            # Arizona-specific: dry but can have monsoons
            if loc == "arizona":
                suggestions.append({"item": "sunscreen", "reason": "Arizona has intense sun", "confidence": 0.85, "priority": "high"})
                suggestions.append({"item": "extra water", "reason": "Arizona desert climate requires hydration", "confidence": 0.9, "priority": "high"})

        # Location-based suggestions (even without weather data)
        if "arizona" in locations:
            if any(w in query for w in ["bring", "pack", "should i", "what should"]):
                suggestions.append({"item": "umbrella", "reason": "Arizona can have sudden monsoon rains", "confidence": 0.7, "priority": "high"})
                suggestions.append({"item": "rainboots", "reason": "For protection during desert rain", "confidence": 0.65, "priority": "medium"})
                suggestions.append({"item": "tent", "reason": "Shelter from sun and rain in Arizona outdoors", "confidence": 0.75, "priority": "high"})
                suggestions.append({"item": "cat carrier", "reason": "Keep your cat safe from desert wildlife", "confidence": 0.8, "priority": "high"})

        # Entity-specific suggestions
        for entity, state in entity_states.items():
            attrs = state.get("attributes", {})
            similar = state.get("similar_to", [])
            # Non-aquatic in rain = suggest shelter
            if not attrs.get("is_aquatic") and any(w.get("rain") in ("high", "heavy", "moderate") for w in weather.values()):
                suggestions.append({"item": "tent or shelter", "reason": f"{entity} is not aquatic and may need protection from rain", "confidence": 0.8, "priority": "high"})
            # Small animal + outdoor + predators = suggest containment
            if attrs.get("is_prey") or not attrs.get("is_predator"):
                for sim in similar:
                    sim_data = self.ke.entities.get(sim.lower(), {})
                    sim_attrs = sim_data.get("attributes", {})
                    if sim_attrs.get("is_predator"):
                        suggestions.append({"item": "secure carrier or leash", "reason": f"{entity} may be at risk from {sim}", "confidence": 0.75, "priority": "high"})
            # Cat-specific
            if entity == "cat":
                suggestions.append({"item": "cat food and water bowl", "reason": "Keep your cat fed and hydrated during travel", "confidence": 0.85, "priority": "high"})
                if "afraid" in query or "fear" in query:
                    suggestions.append({"item": "calming spray or treats", "reason": "Help reduce your cat's anxiety", "confidence": 0.7, "priority": "medium"})
                if any(w in query for w in ["water", "drink", "thirsty"]):
                    suggestions.append({"item": "cool (not cold) water", "reason": "Cats prefer cooler water; normal body temp is 101°F so cool water helps them regulate", "confidence": 0.8, "priority": "high"})
            # Warm-blooded entity + water mention = temperature advice
            if attrs.get("is_warm_blooded") and any(w in query for w in ["water", "drink", "wet", "rain"]):
                body_temp = attrs.get("body_temp_f", 101.0)
                suggestions.append({"item": f"monitor body temperature (normal: {body_temp}°F)", "reason": f"{entity.title()} is warm-blooded; water exposure can affect body temperature", "confidence": 0.75, "priority": "medium"})

        # Deduplicate and prioritize
        seen = set()
        unique = []
        for s in sorted(suggestions, key=lambda x: x["priority"] == "high", reverse=True):
            key = s["item"]
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique[:6]

    def check_suggestion_fulfillment(self, suggestion, context):
        """Check if user has followed a previous suggestion."""
        # This would integrate with webcam/file input in a full system
        return {"fulfilled": False, "note": "External input not available"}

# =========================
# HEALTH TRACKER
# =========================

class HealthTracker:
    """Track entity health based on situations, compute probabilities."""
    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.health_scores = defaultdict(lambda: 100.0)  # entity -> health 0-100
        self.health_history = defaultdict(list)  # entity -> [(turn, score, reason)]
        self.risk_factors = defaultdict(list)

    def compute_health(self, entity, situation, user_actions=None):
        """Compute health probability based on situation."""
        entity_lower = entity.lower()
        base_health = self.health_scores[entity_lower]
        modifiers = []

        data = self.ke.entities.get(entity_lower, {})
        attrs = data.get("attributes", {})

        # Situation modifiers
        if "rain" in situation.lower():
            if not attrs.get("is_aquatic"):
                base_health -= 10
                modifiers.append("-10 rain exposure")
        if "danger" in situation.lower() or "predator" in situation.lower():
            if not attrs.get("is_predator"):
                base_health -= 20
                modifiers.append("-20 danger proximity")
        if "heat" in situation.lower() or "hot" in situation.lower():
            base_health -= 5
            modifiers.append("-5 heat exposure")

        # User action modifiers (positive)
        if user_actions:
            if "petting" in user_actions or "hold" in user_actions:
                base_health += 5
                modifiers.append("+5 user care")
            if "feeding" in user_actions:
                base_health += 3
                modifiers.append("+3 feeding")
            if "tent" in user_actions or "shelter" in user_actions:
                base_health += 8
                modifiers.append("+8 shelter provided")
            if "umbrella" in user_actions:
                base_health += 6
                modifiers.append("+6 rain protection")

        # Clamp
        base_health = max(0, min(100, base_health))
        self.health_scores[entity_lower] = base_health
        self.health_history[entity_lower].append((time.time(), base_health, ", ".join(modifiers)))

        return {
            "health": base_health,
            "modifiers": modifiers,
            "status": "good" if base_health >= 80 else "moderate" if base_health >= 50 else "poor",
        }

    def predict_health_trend(self, entity, turns_ahead=3):
        """Predict health trend based on current trajectory."""
        entity_lower = entity.lower()
        history = self.health_history.get(entity_lower, [])
        if len(history) < 2:
            return {"trend": "stable", "predicted": self.health_scores[entity_lower]}

        recent = [h[1] for h in history[-5:]]
        if len(recent) >= 2:
            avg_change = (recent[-1] - recent[0]) / len(recent)
            predicted = recent[-1] + (avg_change * turns_ahead)
            predicted = max(0, min(100, predicted))
            if avg_change > 1:
                trend = "improving"
            elif avg_change < -1:
                trend = "declining"
            else:
                trend = "stable"
            return {"trend": trend, "predicted": predicted, "change_per_turn": avg_change}
        return {"trend": "stable", "predicted": self.health_scores[entity_lower]}

    def get_risk_assessment(self, entity, situation):
        """Assess risks for entity in situation."""
        entity_lower = entity.lower()
        data = self.ke.entities.get(entity_lower, {})
        attrs = data.get("attributes", {})
        risks = []

        if not attrs.get("is_aquatic") and "water" in situation.lower():
            risks.append({"risk": "drowning", "probability": 0.1, "severity": "high"})
        if attrs.get("is_prey") and "outdoor" in situation.lower():
            risks.append({"risk": "predation", "probability": 0.15, "severity": "high"})
        if "heat" in situation.lower():
            risks.append({"risk": "heatstroke", "probability": 0.05, "severity": "medium"})
        if "rain" in situation.lower() and not attrs.get("is_aquatic"):
            risks.append({"risk": "hypothermia", "probability": 0.08, "severity": "medium"})

        return sorted(risks, key=lambda x: x["probability"], reverse=True)

# =========================
# PRECOGNITION ENGINE
# =========================

class PrecognitionEngine:
    """Predict what will happen based on current state, entity properties, and context."""
    def __init__(self, knowledge_engine, health_tracker, contextual_reasoner):
        self.ke = knowledge_engine
        self.ht = health_tracker
        self.cr = contextual_reasoner
        self.predictions = []

    def predict(self, entity, situation, context=None):
        """Generate predictions about what will happen."""
        entity_lower = entity.lower()
        data = self.ke.entities.get(entity_lower, {})
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        predictions = []

        # Behavioral predictions
        if not attrs.get("is_aquatic") and "rain" in situation.lower():
            predictions.append({
                "event": f"{entity_lower} will seek shelter from rain",
                "probability": 0.85,
                "reason": f"{entity_lower} is not aquatic",
                "action_needed": "provide shelter",
            })
        if attrs.get("is_prey") and "predator" in situation.lower():
            predictions.append({
                "event": f"{entity_lower} will try to escape",
                "probability": 0.9,
                "reason": f"{entity_lower} is prey, not predator",
                "action_needed": "secure or protect",
            })
        speed = props.get("speed_kmh", 0)
        if isinstance(speed, (int, float)) and speed > 30:
            predictions.append({
                "event": f"{entity_lower} can move quickly ({speed} km/h)",
                "probability": 0.95,
                "reason": "high speed capability",
                "action_needed": "ensure containment",
            })

        # Health predictions
        health_pred = self.ht.predict_health_trend(entity_lower)
        if health_pred["trend"] == "declining":
            predictions.append({
                "event": f"{entity_lower}'s health may decline to {health_pred['predicted']:.0f}%",
                "probability": 0.7,
                "reason": "negative health trend",
                "action_needed": "intervene with care",
            })

        # Escape predictions
        if attrs.get("is_domestic") and "outdoor" in situation.lower():
            predictions.append({
                "event": f"{entity_lower} may run away if frightened",
                "probability": 0.4,
                "reason": "domestic animal in unfamiliar outdoor setting",
                "action_needed": "keep on leash or in carrier",
            })

        # Weather impact predictions
        if "rain" in situation.lower():
            if not attrs.get("is_aquatic"):
                predictions.append({
                    "event": f"{entity_lower} will get wet and uncomfortable",
                    "probability": 0.95,
                    "reason": "non-aquatic entity in rain",
                    "action_needed": "bring umbrella or shelter",
                })

        self.predictions.extend(predictions)
        return sorted(predictions, key=lambda x: x["probability"], reverse=True)

    def evaluate_prediction(self, prediction, actual_outcome):
        """Evaluate if a prediction was correct."""
        return {
            "prediction": prediction["event"],
            "was_correct": prediction["event"].lower() in actual_outcome.lower(),
            "probability": prediction["probability"],
        }

# =========================
# GOAL TRACKER
# =========================

class GoalTracker:
    """Track goals, form plans, compare progress."""
    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.active_goals = {}  # goal_id -> {description, target, progress, steps}
        self.completed_goals = []
        self.plans = {}  # goal_id -> [steps]

    def create_goal(self, description, target_entity=None, target_state=None):
        goal_id = f"goal_{len(self.active_goals)}"
        self.active_goals[goal_id] = {
            "description": description,
            "target_entity": target_entity,
            "target_state": target_state,
            "progress": 0,
            "steps": [],
            "status": "active",
        }
        return goal_id

    def add_step(self, goal_id, step_description, estimated_time=None):
        if goal_id in self.active_goals:
            self.active_goals[goal_id]["steps"].append({
                "description": step_description,
                "completed": False,
                "estimated_time": estimated_time,
            })

    def update_progress(self, goal_id, progress_pct, note=None):
        if goal_id in self.active_goals:
            self.active_goals[goal_id]["progress"] = min(100, progress_pct)
            if progress_pct >= 100:
                self.active_goals[goal_id]["status"] = "completed"
                self.completed_goals.append(self.active_goals.pop(goal_id))

    def check_plan_feasibility(self, goal_id, current_situation):
        """Check if goal can be achieved given current situation."""
        goal = self.active_goals.get(goal_id)
        if not goal:
            return None
        remaining_steps = [s for s in goal["steps"] if not s["completed"]]
        return {
            "goal": goal["description"],
            "progress": goal["progress"],
            "remaining_steps": len(remaining_steps),
            "feasible": True,
            "note": f"Need to complete {len(remaining_steps)} more steps",
        }

    def get_status(self):
        return {
            "active": len(self.active_goals),
            "completed": len(self.completed_goals),
            "goals": {k: {"desc": v["description"], "progress": v["progress"]}
                     for k, v in self.active_goals.items()},
        }

# =========================
# BEHAVIORAL LABELER
# =========================

class BehavioralLabeler:
    """Label entities with behavioral traits for prediction."""
    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.entity_labels = {}  # entity -> {traits: [], personality: {}}

    def label_entity(self, entity):
        """Assign behavioral labels based on KB data."""
        entity_lower = entity.lower()
        if entity_lower in self.entity_labels:
            return self.entity_labels[entity_lower]

        data = self.ke.entities.get(entity_lower, {})
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        descs = data.get("descriptions", [])

        traits = []
        personality = {}

        # Analyze attributes for traits
        if attrs.get("is_predator"):
            traits.append("aggressive")
            personality["aggression"] = 0.7
        if attrs.get("is_prey"):
            traits.append("cautious")
            personality["caution"] = 0.8
        if attrs.get("is_domestic"):
            traits.append("social")
            personality["sociability"] = 0.8
        if attrs.get("is_nocturnal"):
            traits.append("nocturnal")
            personality["activity_pattern"] = "nocturnal"
        if attrs.get("has_feathers"):
            traits.append("mobile")
            personality["mobility"] = 0.9
        speed = props.get("speed_kmh", 0)
        if isinstance(speed, (int, float)):
            if speed > 40:
                traits.append("fast")
                personality["speed"] = 0.9
            elif speed < 5:
                traits.append("slow")
                personality["speed"] = 0.2

        # Analyze descriptions for behavioral keywords
        for desc in descs:
            desc_lower = desc.lower()
            if any(w in desc_lower for w in ["independent", "solitary"]):
                traits.append("independent")
                personality["independence"] = 0.8
            if any(w in desc_lower for w in ["loyal", "bond"]):
                traits.append("loyal")
                personality["loyalty"] = 0.8
            if any(w in desc_lower for w in ["afraid", "fear", "scared"]):
                traits.append("fearful")
                personality["fear"] = 0.7
            if any(w in desc_lower for w in ["wild", "feral"]):
                traits.append("wild")
                personality["wildness"] = 0.8
            if any(w in desc_lower for w in ["curious", "explor"]):
                traits.append("curious")
                personality["curiosity"] = 0.7

        # Default traits
        if not traits:
            traits.append("neutral")
            personality["neutrality"] = 0.5

        self.entity_labels[entity_lower] = {"traits": traits, "personality": personality}
        return self.entity_labels[entity_lower]

    def predict_behavior(self, entity, situation):
        """Predict how entity will behave in situation."""
        labels = self.label_entity(entity)
        traits = labels["traits"]
        predictions = []

        if "fearful" in traits and "loud" in situation.lower():
            predictions.append({"behavior": "flee", "probability": 0.8})
        if "aggressive" in traits and "threat" in situation.lower():
            predictions.append({"behavior": "attack", "probability": 0.6})
        if "social" in traits and "alone" in situation.lower():
            predictions.append({"behavior": "seek companionship", "probability": 0.7})
        if "wild" in traits and "captive" in situation.lower():
            predictions.append({"behavior": "attempt escape", "probability": 0.75})
        if "slow" in traits and "chase" in situation.lower():
            predictions.append({"behavior": "unable to escape", "probability": 0.8})

        return predictions if predictions else [{"behavior": "normal", "probability": 0.9}]

# =========================
# RESPONSE TREE
# =========================

class ResponseTree:
    """Build branching response trees with follow-up links."""
    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.trees = {}  # entity -> tree structure
        self.followup_links = {}  # topic -> related_topics

    def build_tree(self, entity, query, base_answer):
        """Build a response tree with branches and follow-ups."""
        entity_lower = entity.lower()
        data = self.ke.entities.get(entity_lower, {})

        tree = {
            "root": base_answer,
            "branches": [],
            "followups": [],
            "linked_topics": [],
        }

        # Add attribute branches
        for attr, val in data.get("attributes", {}).items():
            readable = attr.replace("_", " ").replace("is ", "").replace("has ", "")
            if isinstance(val, bool):
                tree["branches"].append({
                    "condition": f"if asking about {readable}",
                    "response": f"{entity_lower} {'is' if val else 'is not'} {readable}",
                    "probability": 0.9,
                })

        # Add property branches
        for prop, val in data.get("properties", {}).items():
            readable = prop.replace("_", " ")
            tree["branches"].append({
                "condition": f"if asking about {readable}",
                "response": f"The {readable} of {entity_lower} is {val}",
                "probability": 0.9,
            })

        # Add follow-up suggestions
        tree["followups"] = [
            f"What is the {entity_lower}'s health?",
            f"Where is the {entity_lower} now?",
            f"What are {entity_lower}'s weaknesses?",
        ]

        # Link to related topics
        for sim in data.get("similar_to", []):
            tree["linked_topics"].append({
                "topic": sim,
                "link_type": "similar",
                "followup": f"How does {entity_lower} compare to {sim}?",
            })

        self.trees[entity_lower] = tree
        return tree

    def get_followup_for_context(self, entity, last_topic):
        """Get relevant follow-up based on conversation context."""
        tree = self.trees.get(entity.lower(), {})
        followups = tree.get("followups", [])

        # Context-aware follow-up selection
        if "health" in last_topic.lower():
            return f"How is the {entity}'s health in this situation?"
        if "danger" in last_topic.lower():
            return f"What are the risks for {entity}?"
        if "weather" in last_topic.lower():
            return f"How does weather affect {entity}?"

        return followups[0] if followups else f"Tell me more about {entity}"

# =========================
# NOUN HIERARCHY TREE
# =========================

class NounHierarchy:
    """Builds a 10-level deep hierarchy for each noun with 5+ connections per level.
    Enables automatic property inheritance (cat -> animal -> can die, needs water, etc.)."""

    def __init__(self):
        self.nodes = {}  # noun -> {level, parent, children, properties, connections}
        self._build_base_hierarchy()
        self._build_entity_properties()

    def _build_base_hierarchy(self):
        """Build the core biological/taxonomic hierarchy."""
        hierarchy = {
            # Level 0: Specific entities
            "cat": {"parent": "feline", "level": 0},
            "dog": {"parent": "canine", "level": 0},
            "turtle": {"parent": "reptile", "level": 0},
            "bird": {"parent": "avian", "level": 0},
            "fish": {"parent": "fish_species", "level": 0},
            "pangolin": {"parent": "mammal", "level": 0},
            "wolf": {"parent": "canine", "level": 0},
            "snake": {"parent": "reptile", "level": 0},
            # Level 1: Family groups
            "feline": {"parent": "carnivore_mammal", "level": 1, "children": ["cat", "lion", "tiger", "leopard", "cheetah"]},
            "canine": {"parent": "carnivore_mammal", "level": 1, "children": ["dog", "wolf", "fox", "coyote", "hyena"]},
            "reptile": {"parent": "cold_blooded_vertebrate", "level": 1, "children": ["turtle", "snake", "lizard", "crocodile", "iguana"]},
            "avian": {"parent": "warm_blooded_vertebrate", "level": 1, "children": ["bird", "eagle", "parrot", "penguin", "owl"]},
            "fish_species": {"parent": "cold_blooded_vertebrate", "level": 1, "children": ["fish", "shark", "salmon", "tuna", "goldfish"]},
            # Level 2: Major groups
            "carnivore_mammal": {"parent": "mammal", "level": 2, "children": ["feline", "canine", "bear", "weasel", "mongoose"]},
            "warm_blooded_vertebrate": {"parent": "vertebrate", "level": 2, "children": ["mammal", "avian"]},
            "cold_blooded_vertebrate": {"parent": "vertebrate", "level": 2, "children": ["reptile", "fish_species", "amphibian"]},
            # Level 3: Vertebrates
            "mammal": {"parent": "vertebrate", "level": 3, "children": ["carnivore_mammal", "herbivore_mammal", "omnivore_mammal", "primate", "rodent"]},
            "vertebrate": {"parent": "chordate", "level": 4, "children": ["warm_blooded_vertebrate", "cold_blooded_vertebrate"]},
            # Level 5: Chordates
            "chordate": {"parent": "animal", "level": 5, "children": ["vertebrate", "invertebrate"]},
            # Level 6: Animals
            "animal": {"parent": "organism", "level": 6, "children": ["chordate", "insect", "arachnid", "mollusk", "crustacean"]},
            # Level 7: Organisms
            "organism": {"parent": "living_thing", "level": 7, "children": ["animal", "plant", "fungus", "bacteria"]},
            # Level 8: Living things
            "living_thing": {"parent": "entity", "level": 8, "children": ["organism", "virus"]},
            # Level 9: Entities
            "entity": {"parent": None, "level": 9, "children": ["living_thing", "non_living_thing"]},
        }
        for noun, data in hierarchy.items():
            self.nodes[noun] = {
                "level": data["level"],
                "parent": data.get("parent"),
                "children": data.get("children", []),
                "properties": {},
                "connections": [],
            }

    def _build_entity_properties(self):
        """Build properties inherited from hierarchy levels."""
        # Properties defined at each NODE name — children inherit them
        node_props = {
            "entity": {"can_exist": True, "has_state": True},
            "living_thing": {"is_alive": True, "needs_energy": True, "needs_water": True, "can_die": True, "can_grow": True, "has metabolism": True},
            "organism": {"reproduces": True, "responds_to_stimuli": True, "metabolizes": True},
            "animal": {"can_move": True, "has_senses": True, "breathes": True, "has_cells": True, "eats": True, "can_be_injured": True, "has_health": True, "has_behavior": True},
            "chordate": {"has_notochord": True, "has_nervous_system": True},
            "vertebrate": {"has_backbone": True, "has_skeleton": True, "has_brain": True},
            "mammal": {"is_warm_blooded": True, "has_fur_or_hair": True, "produces_milk": True, "has_live_birth": True, "body_temp_f": 101.0, "has_emotions": True, "can_feel_pain": True},
            "warm_blooded_vertebrate": {"regulates_body_temp": True, "is_active": True},
            "cold_blooded_vertebrate": {"depends_on_environment_temp": True, "can_be_sluggish_in_cold": True},
            "carnivore_mammal": {"eats_meat": True, "has_sharp_teeth": True, "is_predator": True, "has_claws": True},
            "feline": {"is_independent": True, "has_retractable_claws": True, "is_nocturnal": True, "hunts_mice": True, "body_temp_f": 101.5, "likes_warmth": True, "afraid_of_water": True, "lives_10_to_20_years": True},
            "canine": {"is_loyal": True, "has_endurance": True, "pack_animal": True, "body_temp_f": 101.0, "likes_water": True, "lives_10_to_13_years": True},
            "reptile": {"is_cold_blooded": True, "has_scales": True, "lays_eggs": True, "body_temp_varies": True},
            "avian": {"has_feathers": True, "can_fly": True, "has_wings": True, "lays_eggs": True, "body_temp_f": 105.0},
            "fish_species": {"lives_in_water": True, "has_gills": True, "has_fins": True, "is_cold_blooded": True},
        }
        # Apply properties to each node by inheriting from ancestors (walk up the tree)
        for noun, node in self.nodes.items():
            props = {}
            current = noun
            while current and current in self.nodes:
                if current in node_props:
                    props.update(node_props[current])
                current = self.nodes[current].get("parent")
            node["properties"] = props

    def get_properties(self, noun):
        """Get all inherited properties for a noun."""
        noun_lower = noun.lower()
        if noun_lower in self.nodes:
            return self.nodes[noun_lower]["properties"]
        return {}

    def get_ancestors(self, noun, max_levels=10):
        """Get ancestor chain up to max_levels."""
        ancestors = []
        current = noun.lower()
        for _ in range(max_levels):
            if current not in self.nodes:
                break
            parent = self.nodes[current].get("parent")
            if not parent:
                break
            ancestors.append(parent)
            current = parent
        return ancestors

    def get_descendants(self, noun, max_levels=10):
        """Get all descendants down to max_levels."""
        descendants = []
        queue = [noun.lower()]
        for _ in range(max_levels):
            next_queue = []
            for n in queue:
                if n in self.nodes:
                    children = self.nodes[n].get("children", [])
                    descendants.extend(children)
                    next_queue.extend(children)
            queue = next_queue
        return descendants

    def find_common_ancestor(self, noun1, noun2):
        """Find the lowest common ancestor of two nouns."""
        ancestors1 = set(self.get_ancestors(noun1))
        ancestors1.add(noun1.lower())
        current = noun2.lower()
        for _ in range(20):
            if current in ancestors1:
                return current
            if current not in self.nodes:
                break
            current = self.nodes[current].get("parent")
        return None

    def share_property(self, noun1, noun2, prop_name):
        """Check if two nouns share a property through hierarchy."""
        p1 = self.get_properties(noun1)
        p2 = self.get_properties(noun2)
        return prop_name in p1 and prop_name in p2 and p1[prop_name] == p2[prop_name]

    def get_shared_properties(self, noun1, noun2):
        """Get all properties shared between two nouns."""
        p1 = self.get_properties(noun1)
        p2 = self.get_properties(noun2)
        shared = {}
        for k, v in p1.items():
            if k in p2 and p2[k] == v:
                shared[k] = v
        return shared

    def get_different_properties(self, noun1, noun2):
        """Get properties that differ between two nouns."""
        p1 = self.get_properties(noun1)
        p2 = self.get_properties(noun2)
        diff = {}
        for k in set(list(p1.keys()) + list(p2.keys())):
            v1 = p1.get(k)
            v2 = p2.get(k)
            if v1 != v2:
                diff[k] = {"noun1": v1, "noun2": v2}
        return diff

    def can_do(self, noun, action):
        """Check if a noun can perform an action based on hierarchy properties."""
        props = self.get_properties(noun)
        action_map = {
            "fly": "can_fly",
            "swim": "lives_in_water",
            "walk": "can_move",
            "run": "can_move",
            "eat": "eats",
            "die": "can_die",
            "breathe": "breathes",
            "feel_pain": "can_feel_pain",
            "reproduce": "reproduces",
        }
        prop_key = action_map.get(action)
        if prop_key:
            return props.get(prop_key, False)
        return None

    def get_body_temp(self, noun):
        """Get expected body temperature for a noun."""
        props = self.get_properties(noun)
        return props.get("body_temp_f", None)

    def get_lifespan(self, noun):
        """Get expected lifespan hint from properties."""
        props = self.get_properties(noun)
        for k, v in props.items():
            if "lives_" in str(k) and isinstance(v, bool) and v:
                return k.replace("lives_", "").replace("_", " ")
        return None

    def to_dict(self):
        """Serialize hierarchy for persistence."""
        return {
            "nodes": {
                n: {
                    "level": d["level"],
                    "parent": d["parent"],
                    "children": d["children"],
                    "properties": {k: v for k, v in d["properties"].items()
                                   if not callable(v) and not k.startswith("_")},
                }
                for n, d in self.nodes.items()
            }
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize hierarchy from persistence."""
        h = cls()
        for n, d in data.get("nodes", {}).items():
            if n in h.nodes:
                h.nodes[n]["properties"].update(d.get("properties", {}))
        return h


# =========================
# ENTITY LIFECYCLE TRACKER
# =========================

class EntityLifecycleTracker:
    """Tracks entity lifecycle: alive -> injured -> critical -> dead -> (resurrection).
    Detects state changes and triggers appropriate emotional responses."""

    STATES = ["alive", "healthy", "injured", "critical", "dead", "resurrected"]

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.entity_states = {}  # entity -> {state, health, timestamp, history, cause}
        self.conversation_context = {}  # entity -> [recent events]

    def get_state(self, entity):
        """Get current lifecycle state of entity."""
        e = entity.lower()
        if e not in self.entity_states:
            self.entity_states[e] = {
                "state": "alive",
                "health": 100,
                "timestamp": time.time(),
                "history": [],
                "cause": None,
                "last_check": time.time(),
            }
        return self.entity_states[e]

    def update_health(self, entity, new_health, cause=None, context=None):
        """Update entity health and detect state transitions."""
        e = entity.lower()
        state = self.get_state(e)
        old_health = state["health"]
        old_state = state["state"]

        state["health"] = max(0, min(100, new_health))
        state["timestamp"] = time.time()
        state["last_check"] = time.time()
        if cause:
            state["cause"] = cause

        # Detect state transitions
        if state["health"] <= 0 and old_state != "dead":
            state["state"] = "dead"
            state["history"].append({
                "event": "death",
                "from_state": old_state,
                "cause": cause,
                "timestamp": time.time(),
            })
            return {"transition": "death", "from": old_state, "cause": cause}
        elif state["health"] <= 25 and old_state not in ("critical", "dead"):
            state["state"] = "critical"
            state["history"].append({
                "event": "critical",
                "from_state": old_state,
                "cause": cause,
                "timestamp": time.time(),
            })
            return {"transition": "critical", "from": old_state, "cause": cause}
        elif state["health"] <= 60 and old_state == "healthy":
            state["state"] = "injured"
            state["history"].append({
                "event": "injured",
                "from_state": old_state,
                "cause": cause,
                "timestamp": time.time(),
            })
            return {"transition": "injured", "from": old_state, "cause": cause}
        elif state["health"] > 60 and old_state in ("injured", "critical"):
            state["state"] = "alive"
            state["history"].append({
                "event": "recovered",
                "from_state": old_state,
                "cause": cause,
                "timestamp": time.time(),
            })
            return {"transition": "recovered", "from": old_state}

        return {"transition": "none", "health": state["health"]}

    def detect_death_mention(self, query, entities):
        """Detect if user mentions entity death in query."""
        q = query.lower()
        death_words = ["died", "dead", "death", "killed", "passed away", "lost", "gone", "no longer"]
        for entity in entities:
            e = entity.lower()
            if any(w in q for w in death_words) and e in q:
                return entity
        return None

    def detect_resurrection_claim(self, query, entities):
        """Detect if user claims entity came back to life."""
        q = query.lower()
        res_words = ["came back", "back to life", "alive again", "resurrected", "revived", "brought back"]
        for entity in entities:
            e = entity.lower()
            if any(w in q for w in res_words) and e in q:
                return entity
        return None

    def generate_emotional_response(self, entity, event_type):
        """Generate appropriate emotional response based on lifecycle event."""
        e = entity.lower()
        data = self.ke.entities.get(e, {})
        attrs = data.get("attributes", {})
        is_prey = attrs.get("is_prey", False)
        is_pet = attrs.get("is_domestic", False)

        responses = {
            "death": {
                "pet": [
                    f"Oh no, I'm really sorry to hear about your {e}. Losing a pet is incredibly hard.",
                    f"I'm so sorry. Losing a {e} must be devastating. They become part of the family.",
                    f"That's heartbreaking. I know how much a {e} can mean to you.",
                ],
                "wild": [
                    f"Sorry to hear about the {e}. That's unfortunate.",
                    f"That's sad news about the {e}.",
                ],
            },
            "critical": {
                "pet": [
                    f"Oh no, your {e} is in critical condition. That's very worrying.",
                    f"Your {e} being at critical health is serious. I hope they pull through.",
                ],
                "wild": [
                    f"The {e} is in critical condition. That's concerning.",
                ],
            },
            "injured": {
                "pet": [
                    f"I'm sorry your {e} is injured. I hope they recover soon.",
                    f"That's tough. Injuries to a {e} can be stressful for both of you.",
                ],
                "wild": [
                    f"The {e} is injured. That's unfortunate.",
                ],
            },
            "resurrected": {
                "pet": [
                    f"That's... unusual. I need to check my data on this. Animals don't typically come back to life. Are you sure about this?",
                    f"I understand you believe your {e} came back to life, but based on everything I know, that's not how biology works. Can you tell me more about what happened?",
                ],
                "wild": [
                    f"I have to be honest — that doesn't align with what I know about {e}s. Can you explain what happened?",
                ],
            },
            "recovered": {
                "pet": [
                    f"That's great news that your {e} is recovering!",
                    f"I'm glad your {e} is doing better. Keep taking good care of them.",
                ],
                "wild": [
                    f"Good to hear the {e} is recovering.",
                ],
            },
        }

        category = "pet" if is_pet else "wild"
        event_responses = responses.get(event_type, {})
        return random.choice(event_responses.get(category, event_responses.get("wild", [f"The {e}'s condition has changed."])))

    def generate_fact_check_response(self, entity, claim_type):
        """Generate fact-based response when user claims something impossible."""
        e = entity.lower()
        if claim_type == "resurrection":
            return (
                f"According to all known biological data, {e}s cannot come back to life. "
                f"Once an animal dies, the biological processes cannot be reversed. "
                f"If you believe your {e} is alive again, there may be another explanation — "
                f"perhaps they were unconscious, in a deep sleep, or there was a misidentification."
            )
        return ""

    def suggest_care_actions(self, entity, health_state):
        """Suggest care actions based on entity state."""
        e = entity.lower()
        data = self.ke.entities.get(e, {})
        attrs = data.get("attributes", {})
        suggestions = []

        if health_state["state"] == "critical":
            suggestions.extend([
                f"Keep your {e} warm and comfortable.",
                "Monitor their breathing closely.",
                "Consider emergency veterinary care if available.",
                "Make sure they have access to clean water.",
            ])
        elif health_state["state"] == "injured":
            suggestions.extend([
                f"Keep your {e} calm and rested.",
                "Check for any visible wounds.",
                "Provide clean water and easy access to food.",
                "Limit their movement if possible.",
            ])
        elif health_state["state"] == "alive" and health_state["health"] > 80:
            suggestions.extend([
                f"Your {e} seems to be doing well!",
                "Keep up the good care.",
                "Regular feeding and fresh water are important.",
            ])

        # Entity-specific suggestions
        if attrs.get("is_aquatic"):
            suggestions.append(f"Make sure {e} has clean, temperature-appropriate water.")
        if attrs.get("afraid_of_water"):
            suggestions.append(f"Keep {e} away from water except for drinking.")
        if attrs.get("is_warm_blooded"):
            body_temp = attrs.get("body_temp_f", 101.0)
            suggestions.append(f"Normal body temperature for {e} is around {body_temp}°F.")

        return suggestions[:4]

    def to_dict(self):
        return {
            "entity_states": self.entity_states,
            "conversation_context": self.conversation_context,
        }

    def load_dict(self, data):
        self.entity_states = data.get("entity_states", {})
        self.conversation_context = data.get("conversation_context", {})


# =========================
# AUTOMATIC PROPERTY INDEX
# =========================

class AutomaticPropertyIndex:
    """Automatically indexes properties from hierarchy + KB for inference.
    Knows cats are warm-blooded, body temp 99.6, needs water, etc."""

    def __init__(self, knowledge_engine, noun_hierarchy):
        self.ke = knowledge_engine
        self.nh = noun_hierarchy
        self.property_cache = {}  # entity -> merged properties

    def get_full_properties(self, entity):
        """Get complete property set for entity from all sources."""
        e = entity.lower()
        if e in self.property_cache:
            return self.property_cache[e]

        # Merge hierarchy properties + KB attributes
        props = {}
        # Hierarchy properties (inherited)
        hierarchy_props = self.nh.get_properties(e)
        props.update(hierarchy_props)
        # KB attributes (override)
        kb_data = self.ke.entities.get(e, {})
        kb_attrs = kb_data.get("attributes", {})
        for k, v in kb_attrs.items():
            props[k] = v
        # KB properties
        kb_props = kb_data.get("properties", {})
        for k, v in kb_props.items():
            props[k] = v
        # KB descriptions
        descs = kb_data.get("descriptions", [])
        if descs:
            props["descriptions"] = descs

        self.property_cache[e] = props
        return props

    def can_survive_condition(self, entity, condition):
        """Check if entity can survive a given condition."""
        props = self.get_full_properties(entity)
        condition_checks = {
            "cold": lambda p: not p.get("is_warm_blooded", False) or p.get("depends_on_environment_temp", False),
            "heat": lambda p: not p.get("is_aquatic", False),
            "water": lambda p: p.get("is_aquatic", False) or p.get("likes_water", False),
            "dry": lambda p: not p.get("is_aquatic", False),
            "injury": lambda p: p.get("can_be_injured", False),
            "starvation": lambda p: p.get("needs_energy", False),
        }
        check = condition_checks.get(condition)
        if check:
            return check(props)
        return None

    def get_care_requirements(self, entity):
        """Get care requirements from properties."""
        props = self.get_full_properties(entity)
        reqs = []
        if props.get("needs_water"):
            reqs.append("needs clean water")
        if props.get("needs_energy"):
            reqs.append("needs regular feeding")
        if props.get("is_warm_blooded"):
            reqs.append("needs warmth")
        if props.get("is_aquatic"):
            reqs.append("needs water environment")
        if props.get("has_fur_or_hair"):
            reqs.append("needs grooming")
        if props.get("afraid_of_water"):
            reqs.append("avoids water (except drinking)")
        if props.get("likes_water"):
            reqs.append("enjoys water")
        return reqs

    def get_health_implications(self, entity, condition):
        """Get health implications of a condition on entity."""
        props = self.get_full_properties(entity)
        implications = []
        if condition == "wet" and not props.get("is_aquatic"):
            if props.get("afraid_of_water"):
                implications.append(f"{entity} is afraid of water and will be stressed")
            if props.get("is_warm_blooded"):
                implications.append(f"{entity} is warm-blooded and may get cold when wet")
            implications.append("Should be dried immediately to prevent hypothermia")
        elif condition == "cold" and props.get("is_warm_blooded"):
            body_temp = props.get("body_temp_f", 101.0)
            implications.append(f"{entity} normal body temp is {body_temp}°F")
            implications.append("Exposure to cold can cause hypothermia")
        elif condition == "no_food" and props.get("needs_energy"):
            implications.append(f"{entity} needs regular food intake")
        return implications

    def to_dict(self):
        return {"property_cache": self.property_cache}

    def load_dict(self, data):
        self.property_cache = data.get("property_cache", {})


# =========================
# CONVERSATION STATE TRACKER
# =========================

class ConversationStateTracker:
    """Tracks entity state across conversation turns with timestamps.
    Detects health changes, rigamortis, time-based conditions."""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.health_history = {}  # entity -> [(timestamp, health, cause)]
        self.event_log = {}  # entity -> [(timestamp, event, details)]
        self.last_mentioned = {}  # entity -> timestamp
        self.circumstances = {}  # entity -> {location, weather, actions}

    def record_health(self, entity, health, cause=None, context=None):
        """Record health measurement with timestamp."""
        e = entity.lower()
        if e not in self.health_history:
            self.health_history[e] = []
        self.health_history[e].append({
            "timestamp": time.time(),
            "health": health,
            "cause": cause,
            "context": context,
        })
        self.last_mentioned[e] = time.time()
        # Keep last 50 records
        self.health_history[e] = self.health_history[e][-50:]

    def record_event(self, entity, event, details=None):
        """Record an event for entity."""
        e = entity.lower()
        if e not in self.event_log:
            self.event_log[e] = []
        self.event_log[e].append({
            "timestamp": time.time(),
            "event": event,
            "details": details,
        })
        self.last_mentioned[e] = time.time()

    def record_circumstances(self, entity, location=None, weather=None, actions=None):
        """Record circumstances around entity."""
        e = entity.lower()
        self.circumstances[e] = {
            "location": location,
            "weather": weather,
            "actions": actions or [],
            "timestamp": time.time(),
        }

    def get_health_trend(self, entity, hours=1):
        """Get health trend over last N hours."""
        e = entity.lower()
        if e not in self.health_history:
            return None
        cutoff = time.time() - (hours * 3600)
        recent = [h for h in self.health_history[e] if h["timestamp"] >= cutoff]
        if len(recent) < 2:
            return None
        return {
            "start_health": recent[0]["health"],
            "end_health": recent[-1]["health"],
            "change": recent[-1]["health"] - recent[0]["health"],
            "measurements": len(recent),
            "hours": hours,
        }

    def detect_rigamortis(self, entity):
        """Detect if enough time has passed after death for rigamortis."""
        e = entity.lower()
        if e not in self.event_log:
            return False
        for event in reversed(self.event_log[e]):
            if event["event"] == "death":
                elapsed = time.time() - event["timestamp"]
                # Rigamortis typically starts 1-3 hours after death
                return elapsed > 3600  # 1 hour
        return False

    def get_time_since_event(self, entity, event_type):
        """Get time since a specific event."""
        e = entity.lower()
        if e not in self.event_log:
            return None
        for event in reversed(self.event_log[e]):
            if event["event"] == event_type:
                return time.time() - event["timestamp"]
        return None

    def get_status_summary(self, entity):
        """Get comprehensive status summary for entity."""
        e = entity.lower()
        health_history = self.health_history.get(e, [])
        events = self.event_log.get(e, [])
        circumstances = self.circumstances.get(e, {})

        summary = {
            "entity": e,
            "current_health": health_history[-1]["health"] if health_history else None,
            "health_trend": self.get_health_trend(e, hours=1),
            "recent_events": events[-5:] if events else [],
            "circumstances": circumstances,
            "rigamortis": self.detect_rigamortis(e),
            "time_since_death": self.get_time_since_event(e, "death"),
        }
        return summary

    def generate_status_report(self, entity):
        """Generate a human-readable status report."""
        summary = self.get_status_summary(entity)
        e = entity.lower()
        parts = []

        if summary["current_health"] is not None:
            h = summary["current_health"]
            if h > 80:
                parts.append(f"{e.title()} is in good health at {h}%.")
            elif h > 60:
                parts.append(f"{e.title()} is at {h}% health — doing okay but worth monitoring.")
            elif h > 25:
                parts.append(f"{e.title()} is at {h}% health — needs attention.")
            elif h > 0:
                parts.append(f"{e.title()} is at {h}% health — critical condition.")
            else:
                parts.append(f"{e.title()} has passed away.")

        trend = summary["health_trend"]
        if trend:
            if trend["change"] < -20:
                parts.append(f"Health dropped {abs(trend['change']):.0f}% in the last {trend['hours']} hour(s).")
            elif trend["change"] < 0:
                parts.append(f"Slight decline of {abs(trend['change']):.0f}% over {trend['hours']} hour(s).")
            elif trend["change"] > 0:
                parts.append(f"Health improved by {trend['change']:.0f}% over {trend['hours']} hour(s).")

        if summary["rigamortis"]:
            parts.append("Note: Rigamortis may be setting in if death occurred recently.")

        return " ".join(parts) if parts else f"No recent health data for {e}."

    def to_dict(self):
        return {
            "health_history": self.health_history,
            "event_log": self.event_log,
            "last_mentioned": self.last_mentioned,
            "circumstances": self.circumstances,
        }

    def load_dict(self, data):
        self.health_history = data.get("health_history", {})
        self.event_log = data.get("event_log", {})
        self.last_mentioned = data.get("last_mentioned", {})
        self.circumstances = data.get("circumstances", {})


# =========================
# DREAM / IMAGINATION SEPARATOR
# =========================

class DreamImaginationSeparator:
    """Separates real events from described dreams or imagination.
    Adjusts entity state based on dream quality (good dream -> happiness up)."""

    def __init__(self):
        self.dream_words = ["dream", "dreamed", "dreaming", "imagined", "imagining",
                            "fantasized", "wished", "hoped", "pretended", "visualization",
                            "saw in a dream", "had a vision"]
        self.imaginative_words = ["what if", "imagine", "suppose", "hypothetically",
                                  "in a perfect world", "i wish", "if only"]
        self.positive_dream_words = ["happy", "joy", "play", "running", "eating",
                                     "sleeping peacefully", "cuddling", "smiling"]
        self.negative_dream_words = ["scary", "frightened", "chasing", "running away",
                                     "hurt", "lost", "alone", "crying"]

    def is_dream_or_imagination(self, query):
        """Detect if query describes a dream or imagination."""
        q = query.lower()
        for word in self.dream_words:
            if word in q:
                return {"type": "dream", "confidence": 0.9}
        for word in self.imaginative_words:
            if word in q:
                return {"type": "imagination", "confidence": 0.85}
        return None

    def get_dream_sentiment(self, query):
        """Analyze dream sentiment (positive/negative)."""
        q = query.lower()
        pos_count = sum(1 for w in self.positive_dream_words if w in q)
        neg_count = sum(1 for w in self.negative_dream_words if w in q)

        if pos_count > neg_count:
            return {"sentiment": "positive", "score": min(1.0, 0.5 + pos_count * 0.15)}
        elif neg_count > pos_count:
            return {"sentiment": "negative", "score": min(1.0, 0.5 + neg_count * 0.15)}
        return {"sentiment": "neutral", "score": 0.5}

    def apply_dream_effects(self, entity, query):
        """Apply dream effects to entity state (happiness -> health boost)."""
        sentiment = self.get_dream_sentiment(query)
        effects = {"happiness_change": 0, "health_change": 0, "stress_change": 0}

        if sentiment["sentiment"] == "positive":
            effects["happiness_change"] = int(sentiment["score"] * 10)
            effects["health_change"] = int(sentiment["score"] * 3)  # Small health boost
            effects["stress_change"] = -int(sentiment["score"] * 5)
        elif sentiment["sentiment"] == "negative":
            effects["happiness_change"] = -int(sentiment["score"] * 10)
            effects["health_change"] = -int(sentiment["score"] * 2)
            effects["stress_change"] = int(sentiment["score"] * 5)

        return effects

    def generate_dream_response(self, entity, query):
        """Generate response to a dream description."""
        dream_info = self.is_dream_or_imagination(query)
        if not dream_info:
            return None

        sentiment = self.get_dream_sentiment(query)
        e = entity.lower()

        if sentiment["sentiment"] == "positive":
            return {
                "text": f"That sounds like a wonderful dream! Dreams about {e} being happy can reflect your bond. "
                        f"It sounds like your {e} brings you a lot of joy.",
                "source": "dream_analysis",
                "score": 0.85,
                "effects": self.apply_dream_effects(e, query),
            }
        elif sentiment["sentiment"] == "negative":
            return {
                "text": f"That sounds like a concerning dream. Sometimes our dreams reflect worries about "
                        f"our {e}'s wellbeing. Is everything okay with your {e}?",
                "source": "dream_analysis",
                "score": 0.85,
                "effects": self.apply_dream_effects(e, query),
            }
        else:
            return {
                "text": f"Interesting dream about your {e}. Dreams can be our mind's way of processing "
                        f"daily experiences. Was there anything specific that happened with your {e} recently?",
                "source": "dream_analysis",
                "score": 0.83,
                "effects": self.apply_dream_effects(e, query),
            }


# =========================
# TEMPORAL RESPONSE ENGINE
# =========================

class TemporalResponseEngine:
    """Generates responses in different temporal modes:
    outward (empathy/emotional), current (health/status check),
    future (preparation/planning)."""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine

    def detect_temporal_mode(self, query):
        """Detect whether query is about past, present, or future."""
        q = query.lower()
        future_words = ["will", "going to", "plan", "prepare", "should i", "what if",
                        "need to", "about to", "next", "tomorrow", "later", "ordering",
                        "getting", "buying", "new one"]
        past_words = ["was", "were", "had", "happened", "before", "used to", "remember",
                      "back when", "yesterday", "ago", "died", "lost", "found"]
        current_words = ["is", "are", "doing", "now", "current", "right now", "today",
                         "check", "status", "how is", "how are", "feeling"]

        future_score = sum(1 for w in future_words if w in q)
        past_score = sum(1 for w in past_words if w in q)
        current_score = sum(1 for w in current_words if w in q)

        if future_score > past_score and future_score > current_score:
            return "future"
        elif past_score > current_score and past_score > future_score:
            return "past"
        return "current"

    def generate_response(self, entity, query, lifecycle_state, mode=None):
        """Generate temporally-appropriate response."""
        if mode is None:
            mode = self.detect_temporal_mode(query)
        e = entity.lower()

        if mode == "past":
            return self._generate_past_response(e, query, lifecycle_state)
        elif mode == "future":
            return self._generate_future_response(e, query, lifecycle_state)
        return self._generate_current_response(e, query, lifecycle_state)

    def _generate_past_response(self, entity, query, state):
        """Response about past events — empathetic, reflective."""
        if state.get("state") == "dead":
            return {
                "text": f"I understand your {entity} passed away. That must be very difficult. "
                        f"Would you like to talk about them, or is there something I can help with?",
                "source": "temporal_past",
                "score": 0.88,
                "mode": "past",
            }
        return {
            "text": f"Looking back at your {entity}'s history — {state.get('cause', 'their journey')}. "
                    f"Is there something specific you'd like to remember or discuss?",
            "source": "temporal_past",
            "score": 0.85,
            "mode": "past",
        }

    def _generate_current_response(self, entity, query, state):
        """Response about current state — factual, check-based."""
        health = state.get("health", 100)
        status = state.get("state", "alive")
        if status == "critical":
            return {
                "text": f"Your {entity} is currently in critical condition at {health}% health. "
                        f"We need to monitor this closely.",
                "source": "temporal_current",
                "score": 0.90,
                "mode": "current",
            }
        elif status == "injured":
            return {
                "text": f"Your {entity} is currently injured at {health}% health. "
                        f"They need care and monitoring.",
                "source": "temporal_current",
                "score": 0.88,
                "mode": "current",
            }
        return {
            "text": f"Your {entity} is currently at {health}% health and {status}.",
            "source": "temporal_current",
            "score": 0.86,
            "mode": "current",
        }

    def _generate_future_response(self, entity, query, state):
        """Response about future — preparation, planning, advice."""
        q = query.lower()
        suggestions = []

        if "order" in q or "buy" in q or "new" in q:
            suggestions.append(f"When getting a new {entity}, consider their care requirements.")
        if "prepare" in q or "should i" in q:
            suggestions.append(f"Planning ahead for your {entity} is smart.")
            suggestions.append("Consider their dietary needs, shelter, and health monitoring.")

        data = self.ke.entities.get(entity, {})
        attrs = data.get("attributes", {})
        if attrs.get("needs_water"):
            suggestions.append("Make sure fresh water is always available.")
        if attrs.get("is_warm_blooded"):
            suggestions.append("Keep them in appropriate temperature conditions.")
        if attrs.get("afraid_of_water"):
            suggestions.append("Avoid forcing them into water — they're naturally averse.")

        if suggestions:
            return {
                "text": " ".join(suggestions[:3]),
                "source": "temporal_future",
                "score": 0.87,
                "mode": "future",
            }
        return {
            "text": f"For your {entity}'s future care, regular health checks and proper nutrition are key.",
            "source": "temporal_future",
            "score": 0.85,
            "mode": "future",
        }


# =========================
# FACT REBUTTAL ENGINE
# =========================

class FactRebuttalEngine:
    """Detects when user's experience contradicts known facts and provides
    context-aware rebuttals. E.g., 'cats live 10 years but mine died in 2'."""

    def __init__(self, knowledge_engine, noun_hierarchy):
        self.ke = knowledge_engine
        self.nh = noun_hierarchy

    def detect_contradiction(self, query, entity):
        """Detect if user's statement contradicts known facts."""
        e = entity.lower()
        q = query.lower()
        contradictions = []

        # Lifespan contradiction
        lifespan_props = self.nh.get_properties(e)
        for k, v in lifespan_props.items():
            if "lives_" in str(k) and isinstance(v, bool) and v:
                years = k.replace("lives_", "").replace("to", "-").replace("_", " ")
                if any(w in q for w in ["died", "dead", "killed", "passed"]):
                    contradictions.append({
                        "type": "lifespan",
                        "expected": years,
                        "user_claim": "early death",
                        "fact_key": k,
                    })

        # Body temperature contradiction
        body_temp = lifespan_props.get("body_temp_f")
        if body_temp and any(w in q for w in ["temperature", "fever", "cold", "hot"]):
            contradictions.append({
                "type": "body_temp",
                "expected": f"{body_temp}°F",
                "fact_key": "body_temp_f",
            })

        return contradictions

    def generate_rebuttal(self, entity, contradiction, context=None):
        """Generate a fact-based rebuttal with context."""
        e = entity.lower()

        if contradiction["type"] == "lifespan":
            expected = contradiction["expected"]
            return {
                "text": f"Typically, {e}s live {expected} years according to general data. "
                        f"If yours died significantly earlier, there could be several factors: "
                        f"genetics, diet, living conditions, illness, or environmental stress. "
                        f"Where was your {e} found? It may have been a stray or had pre-existing conditions.",
                "source": "fact_rebuttal",
                "score": 0.90,
                "contradiction": contradiction,
            }

        elif contradiction["type"] == "body_temp":
            expected = contradiction["expected"]
            return {
                "text": f"The normal body temperature for {e} is about {expected}. "
                        f"If you're concerned about their temperature, that's something to monitor. "
                        f"A significant deviation from this could indicate illness.",
                "source": "fact_rebuttal",
                "score": 0.89,
                "contradiction": contradiction,
            }

        return {
            "text": f"That's an interesting observation about {e}. Let me check what I know.",
            "source": "fact_rebuttal",
            "score": 0.85,
            "contradiction": contradiction,
        }

    def cross_reference_with_data(self, entity, user_claim):
        """Cross-reference user claim against all available data."""
        e = entity.lower()
        findings = []

        # Check KB properties
        kb_data = self.ke.entities.get(e, {})
        props = kb_data.get("properties", {})
        for k, v in props.items():
            findings.append({"source": "kb_property", "key": k, "value": v})

        # Check KB attributes
        attrs = kb_data.get("attributes", {})
        for k, v in attrs.items():
            findings.append({"source": "kb_attribute", "key": k, "value": v})

        # Check hierarchy properties
        h_props = self.nh.get_properties(e)
        for k, v in h_props.items():
            findings.append({"source": "hierarchy", "key": k, "value": v})

        # Check dataset QA
        for q, a in self.ke.dataset_qa:
            if e in q.lower():
                findings.append({"source": "dataset_qa", "key": q[:50], "value": a[:100]})

        return findings


# =========================
# HIERARCHY SEARCH ENGINE
# =========================

class HierarchySearchEngine:
    """Cross-entity comparison via shared ancestors in the hierarchy tree.
    Enables rich comparisons like 'cat and dog are both mammals, both predators'."""

    def __init__(self, noun_hierarchy):
        self.nh = noun_hierarchy

    def compare_entities(self, entity1, entity2):
        """Compare two entities using hierarchy."""
        shared = self.nh.get_shared_properties(entity1, entity2)
        different = self.nh.get_different_properties(entity1, entity2)
        common_ancestor = self.nh.find_common_ancestor(entity1, entity2)

        return {
            "shared_properties": shared,
            "different_properties": different,
            "common_ancestor": common_ancestor,
            "relationship": self._describe_relationship(entity1, entity2, common_ancestor),
        }

    def _describe_relationship(self, entity1, entity2, common_ancestor):
        """Describe the relationship between two entities."""
        if not common_ancestor:
            return f"{entity1} and {entity2} don't share a clear classification."
        a1 = self.nh.get_ancestors(entity1)
        a2 = self.nh.get_ancestors(entity2)
        level1 = self.nh.nodes.get(entity1, {}).get("level", 0)
        level2 = self.nh.nodes.get(entity2, {}).get("level", 0)

        if level1 == level2 and common_ancestor in a1 and common_ancestor in a2:
            parent1 = self.nh.nodes.get(entity1, {}).get("parent")
            parent2 = self.nh.nodes.get(entity2, {}).get("parent")
            if parent1 == parent2:
                return f"{entity1.title()} and {entity2.title()} are closely related — both are {parent1}s."
            return f"{entity1.title()} and {entity2.title()} are both {common_ancestor}s."
        return f"{entity1.title()} and {entity2.title()} share {common_ancestor} as a common ancestor."

    def get_entity_context(self, entity, depth=5):
        """Get rich context for entity from hierarchy."""
        ancestors = self.nh.get_ancestors(entity, max_levels=depth)
        descendants = self.nh.get_descendants(entity, max_levels=3)
        props = self.nh.get_properties(entity)

        return {
            "entity": entity,
            "ancestors": ancestors[:depth],
            "descendants": descendants[:10],
            "properties": props,
            "category": ancestors[0] if ancestors else None,
            "super_category": ancestors[1] if len(ancestors) > 1 else None,
        }

    def find_similar_entities(self, entity, threshold=0.5):
        """Find entities similar to given entity based on shared properties."""
        entity_props = self.nh.get_properties(entity)
        similarities = []

        for other_node in self.nh.nodes:
            if other_node == entity.lower():
                continue
            other_props = self.nh.get_properties(other_node)
            shared = set(entity_props.keys()) & set(other_props.keys())
            if len(shared) > 0:
                score = len(shared) / max(len(entity_props), len(other_props))
                if score >= threshold:
                    similarities.append({
                        "entity": other_node,
                        "score": score,
                        "shared_count": len(shared),
                    })

        return sorted(similarities, key=lambda x: x["score"], reverse=True)[:5]


# =========================
# BACKGROUND ANALYZER
# =========================

class BackgroundAnalyzer:
    """Idle-time dataset analysis: extracts entities, relationships, QA pairs,
    builds profiles, detects gaps. Runs during free time or on demand."""

    def __init__(self, knowledge_engine, noun_hierarchy):
        self.ke = knowledge_engine
        self.nh = noun_hierarchy
        self.extracted_entities = {}  # entity -> {sources, properties, confidence}
        self.extracted_relations = []  # [(entity1, relation, entity2, source, confidence)]
        self.extracted_qa = []  # [(question, answer, source, confidence)]
        self.gaps = []  # missing knowledge areas
        self.profiles = {}  # entity -> full profile
        self.last_analysis = None
        self.analysis_count = 0

    def run_full_analysis(self):
        """Run complete analysis of all available data."""
        self.analysis_count += 1
        self.last_analysis = time.time()

        # 1. Extract entities from all sources
        self._extract_entities_from_kb()
        self._extract_entities_from_descriptions()
        self._extract_entities_from_qa()

        # 2. Build relationships
        self._extract_relationships()

        # 3. Build profiles
        self._build_profiles()

        # 4. Detect gaps
        self._detect_gaps()

        # 5. Find contradictions
        self._find_contradictions()

        return self.get_summary()

    def _extract_entities_from_kb(self):
        """Extract entities and their properties from KB."""
        for entity, data in self.ke.entities.items():
            if entity not in self.extracted_entities:
                self.extracted_entities[entity] = {
                    "sources": ["kb"],
                    "properties": {},
                    "confidence": 0.9,
                    "first_seen": time.time(),
                }
            attrs = data.get("attributes", {})
            props = data.get("properties", {})
            self.extracted_entities[entity]["properties"].update(attrs)
            self.extracted_entities[entity]["properties"].update(props)

    def _extract_entities_from_descriptions(self):
        """Extract entities mentioned in descriptions."""
        for entity, data in self.ke.entities.items():
            descs = data.get("descriptions", [])
            for desc in descs:
                words = desc.lower().split()
                for other_entity in self.ke.entities:
                    if other_entity != entity and other_entity in words:
                        self.extracted_relations.append({
                            "entity1": entity,
                            "relation": "mentioned_with",
                            "entity2": other_entity,
                            "source": "description",
                            "confidence": 0.7,
                        })

    def _extract_entities_from_qa(self):
        """Extract entities and relationships from QA pairs."""
        for q, a in self.ke.dataset_qa:
            q_lower = q.lower()
            a_lower = a.lower()
            # Find entities mentioned
            mentioned = []
            for entity in self.ke.entities:
                if entity in q_lower or entity in a_lower:
                    mentioned.append(entity)
            # Build relationships between co-mentioned entities
            for i, e1 in enumerate(mentioned):
                for e2 in mentioned[i+1:]:
                    self.extracted_relations.append({
                        "entity1": e1,
                        "relation": "co_mentioned",
                        "entity2": e2,
                        "source": "dataset_qa",
                        "confidence": 0.6,
                    })
            # Store QA pair as insight
            if mentioned:
                self.extracted_qa.append({
                    "question": q,
                    "answer": a,
                    "entities": mentioned,
                    "source": "dataset",
                    "confidence": 0.8,
                })

    def _extract_relationships(self):
        """Extract relationships from hierarchy."""
        for noun, node in self.nh.nodes.items():
            parent = node.get("parent")
            if parent:
                self.extracted_relations.append({
                    "entity1": noun,
                    "relation": "is_a",
                    "entity2": parent,
                    "source": "hierarchy",
                    "confidence": 0.95,
                })
            for child in node.get("children", []):
                self.extracted_relations.append({
                    "entity1": child,
                    "relation": "is_a",
                    "entity2": noun,
                    "source": "hierarchy",
                    "confidence": 0.95,
                })

    def _build_profiles(self):
        """Build comprehensive profiles for each entity."""
        for entity in set(list(self.extracted_entities.keys()) + list(self.nh.nodes.keys())):
            profile = {
                "entity": entity,
                "hierarchy": self.nh.get_ancestors(entity, 5),
                "properties": self.nh.get_properties(entity),
                "kb_data": self.ke.entities.get(entity, {}),
                "relations": [r for r in self.extracted_relations
                              if r["entity1"] == entity or r["entity2"] == entity],
                "qa_pairs": [qa for qa in self.extracted_qa
                             if entity in qa.get("entities", [])],
                "category": self.nh.get_ancestors(entity, 1)[0] if self.nh.get_ancestors(entity, 1) else None,
            }
            self.profiles[entity] = profile

    def _detect_gaps(self):
        """Detect missing knowledge areas."""
        self.gaps = []
        for entity, profile in self.profiles.items():
            props = profile.get("properties", {})
            # Missing basic info
            if not props.get("body_temp_f"):
                self.gaps.append({"entity": entity, "gap": "body_temperature", "priority": "medium"})
            if not props.get("lifespan") and not any("lives_" in str(k) for k in props):
                self.gaps.append({"entity": entity, "gap": "lifespan", "priority": "low"})
            if "descriptions" not in profile.get("kb_data", {}):
                self.gaps.append({"entity": entity, "gap": "description", "priority": "high"})
            # Missing behavioral data
            if not props.get("is_predator") and not props.get("is_prey"):
                self.gaps.append({"entity": entity, "gap": "predator_prey_status", "priority": "medium"})

    def _find_contradictions(self):
        """Find contradictory statements in data."""
        self.contradictions = []
        # Check QA pairs against each other
        for i, qa1 in enumerate(self.extracted_qa):
            for qa2 in self.extracted_qa[i+1:]:
                if set(qa1.get("entities", [])) & set(qa2.get("entities", [])):
                    # Same entities — check for contradicting answers
                    a1 = qa1["answer"].lower()
                    a2 = qa2["answer"].lower()
                    if ("not" in a1) != ("not" in a2):
                        self.contradictions.append({
                            "qa1": qa1,
                            "qa2": qa2,
                            "type": "negation_difference",
                            "confidence": 0.6,
                        })

    def get_profile(self, entity):
        """Get full profile for entity."""
        return self.profiles.get(entity.lower(), {})

    def get_entity_relations(self, entity):
        """Get all relations for entity."""
        return [r for r in self.extracted_relations
                if r["entity1"] == entity.lower() or r["entity2"] == entity.lower()]

    def get_summary(self):
        """Get analysis summary."""
        return {
            "entities_extracted": len(self.extracted_entities),
            "relations_found": len(self.extracted_relations),
            "qa_pairs_indexed": len(self.extracted_qa),
            "profiles_built": len(self.profiles),
            "gaps_detected": len(self.gaps),
            "contradictions_found": len(self.contradictions),
            "analysis_count": self.analysis_count,
            "last_analysis": self.last_analysis,
        }

    def to_dict(self):
        return {
            "extracted_entities": self.extracted_entities,
            "extracted_relations": self.extracted_relations,
            "extracted_qa": self.extracted_qa,
            "gaps": self.gaps,
            "profiles": {k: {kk: vv for kk, vv in v.items() if kk != "kb_data"} for k, v in self.profiles.items()},
            "analysis_count": self.analysis_count,
        }

    def load_dict(self, data):
        self.extracted_entities = data.get("extracted_entities", {})
        self.extracted_relations = data.get("extracted_relations", [])
        self.extracted_qa = data.get("extracted_qa", [])
        self.gaps = data.get("gaps", [])
        self.analysis_count = data.get("analysis_count", 0)


# =========================
# INSIGHT CRUD
# =========================

class InsightCRUD:
    """Manages extracted insights: approve, deny, append, list, update, export."""

    def __init__(self, background_analyzer):
        self.ba = background_analyzer
        self.approved = []  # approved insights
        self.denied = []  # denied insights
        self.custom = []  # user-added insights

    def approve(self, index, insight_type="relations"):
        """Approve an extracted insight."""
        source = getattr(self.ba, f"extracted_{insight_type}", [])
        if 0 <= index < len(source):
            item = source[index]
            item["status"] = "approved"
            item["approved_at"] = time.time()
            self.approved.append(item)
            return f"Approved: {item}"
        return "Invalid index"

    def deny(self, index, insight_type="relations"):
        """Deny an extracted insight."""
        source = getattr(self.ba, f"extracted_{insight_type}", [])
        if 0 <= index < len(source):
            item = source[index]
            item["status"] = "denied"
            item["denied_at"] = time.time()
            self.denied.append(item)
            return f"Denied: {item}"
        return "Invalid index"

    def append(self, insight_type, data):
        """Append a new insight."""
        data["status"] = "custom"
        data["created_at"] = time.time()
        if insight_type == "relations":
            self.ba.extracted_relations.append(data)
        elif insight_type == "qa":
            self.ba.extracted_qa.append(data)
        elif insight_type == "entities":
            entity = data.get("entity", "")
            self.ba.extracted_entities[entity] = data
        self.custom.append(data)
        return f"Added: {data}"

    def list_items(self, insight_type="relations", status=None):
        """List insights with optional status filter."""
        source = getattr(self.ba, f"extracted_{insight_type}", [])
        if isinstance(source, dict):
            items = [{"entity": k, **v} for k, v in source.items()]
        else:
            items = source
        if status:
            items = [i for i in items if i.get("status") == status]
        return items

    def update(self, index, insight_type, updates):
        """Update an insight."""
        source = getattr(self.ba, f"extracted_{insight_type}", [])
        if isinstance(source, list) and 0 <= index < len(source):
            source[index].update(updates)
            return f"Updated: {source[index]}"
        return "Invalid index"

    def export_to_file(self, filepath="insights_export.json"):
        """Export all insights to file."""
        data = {
            "approved": self.approved,
            "denied": self.denied,
            "custom": self.custom,
            "summary": self.ba.get_summary(),
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return f"Exported to {filepath}"

    def import_to_conversation(self, entity=None):
        """Format insights for conversation display."""
        if entity:
            profile = self.ba.get_profile(entity)
            if not profile:
                return f"No profile found for {entity}"
            lines = [f"=== {entity.title()} Profile ==="]
            lines.append(f"Category: {profile.get('category', 'unknown')}")
            lines.append(f"Ancestors: {' > '.join(profile.get('hierarchy', [])[:5])}")
            props = profile.get("properties", {})
            key_props = {k: v for k, v in props.items() if v and k not in ("descriptions",)}
            if key_props:
                lines.append(f"Properties: {json.dumps(key_props, indent=2)[:500]}")
            relations = profile.get("relations", [])[:5]
            if relations:
                lines.append(f"Relations: {len(profile.get('relations', []))} found")
            return "\n".join(lines)
        else:
            summary = self.ba.get_summary()
            lines = [f"=== Insight Summary ==="]
            for k, v in summary.items():
                lines.append(f"  {k}: {v}")
            if self.ba.gaps:
                lines.append(f"\n=== Top Gaps ===")
                for g in self.ba.gaps[:5]:
                    lines.append(f"  {g['entity']}: {g['gap']} ({g['priority']})")
            return "\n".join(lines)


# =========================
# HYPOTHETICAL ENGINE
# =========================

class HypotheticalEngine:
    """Handles 'what would happen if' questions by deriving consequences
    from hierarchy properties, KB data, and simulation."""

    def __init__(self, knowledge_engine, noun_hierarchy, property_index):
        self.ke = knowledge_engine
        self.nh = noun_hierarchy
        self.pi = property_index

    def analyze(self, query, entities):
        """Analyze a hypothetical question and derive consequences."""
        q = query.lower()
        consequences = []
        steps = []

        for entity in entities:
            e = entity.lower()
            props = self.pi.get_full_properties(e)
            hier_props = self.nh.get_properties(e)

            # Detect the hypothetical condition
            condition = self._detect_condition(q, e)

            if condition:
                result = self._derive_consequence(e, condition, props, hier_props)
                consequences.extend(result["consequences"])
                steps.extend(result["steps"])

        if not consequences:
            return None

        return {
            "consequences": consequences,
            "steps": steps,
            "confidence": self._compute_confidence(consequences),
        }

    def _detect_condition(self, query, entity):
        """Detect the hypothetical condition from query."""
        conditions = {
            "loses": ["vision", "eye", "sight", "hearing", "leg", "tail", "fur", "claws", "teeth"],
            "gets": ["wet", "cold", "hot", "sick", "injured", "lost", "stuck"],
            "eats": ["poison", "something bad", "nothing", "too much"],
            "meets": ["predator", "dog", "wolf", "snake", "bear"],
            "is": ["alone", "trapped", "abandoned", "homeless", "hungry", "thirsty"],
            "loses its": ["night vision", "hearing", "mobility", "appetite"],
            "can't": ["see", "hear", "walk", "eat", "drink"],
            "drops": ["cat", "animal"],
        }
        for verb, nouns in conditions.items():
            if verb in query:
                for noun in nouns:
                    if noun in query:
                        return {"verb": verb, "noun": noun, "entity": entity}
        return None

    def _derive_consequence(self, entity, condition, props, hier_props):
        """Derive consequences from a condition using hierarchy + KB."""
        consequences = []
        steps = []
        verb = condition["verb"]
        noun = condition["noun"]

        # Vision loss consequences
        if noun in ("vision", "eye", "sight", "night vision"):
            consequences.append({
                "effect": "vulnerable",
                "severity": "high",
                "detail": f"{entity} will become vulnerable without {noun}",
                "health_impact": -60,
            })
            steps.append(f"1. {entity.title()} loses {noun}")
            steps.append(f"2. Cannot navigate safely in environment")
            steps.append(f"3. Increased risk of injury from obstacles, falls, predators")
            steps.append(f"4. Health decreases significantly (estimated 40% remaining)")
            steps.append(f"5. May become anxious or stressed")
            if props.get("is_nocturnal"):
                consequences.append({
                    "effect": "hunting_impaired",
                    "severity": "critical",
                    "detail": f"{entity} is nocturnal — losing night vision severely impacts hunting",
                    "health_impact": -20,
                })
                steps.append(f"6. Being nocturnal, {entity} relies heavily on night vision for hunting")
            if props.get("is_predator"):
                consequences.append({
                    "effect": "hunting_failure",
                    "severity": "high",
                    "detail": f"{entity} is a predator — cannot hunt effectively without sight",
                    "health_impact": -15,
                })

        # Wet consequences for non-aquatic
        elif noun in ("wet",) and not props.get("is_aquatic"):
            consequences.append({
                "effect": "hypothermia_risk",
                "severity": "medium" if props.get("is_warm_blooded") else "low",
                "detail": f"{entity} is warm-blooded and will lose body heat when wet",
                "health_impact": -25 if props.get("is_warm_blooded") else -5,
            })
            steps.append(f"1. {entity.title()} gets wet")
            if props.get("afraid_of_water"):
                steps.append(f"2. {entity} is afraid of water — will be stressed")
                consequences.append({
                    "effect": "stress",
                    "severity": "medium",
                    "detail": f"{entity} is afraid of water",
                    "health_impact": -10,
                })
            if props.get("is_warm_blooded"):
                body_temp = props.get("body_temp_f", 101.0)
                steps.append(f"3. Normal body temp is {body_temp}°F — will drop when wet")
                steps.append(f"4. Risk of hypothermia if not dried quickly")

        # Predator meeting consequences
        elif verb == "meets" and noun in ("predator", "dog", "wolf", "snake", "bear"):
            predator_data = self.ke.entities.get(noun, {})
            predator_attrs = predator_data.get("attributes", {})
            if predator_attrs.get("is_predator") and props.get("is_prey"):
                consequences.append({
                    "effect": "danger",
                    "severity": "critical",
                    "detail": f"{entity} is prey and {noun} is a predator — serious danger",
                    "health_impact": -80,
                })
                steps.append(f"1. {entity.title()} encounters {noun} (predator)")
                steps.append(f"2. {entity} is prey — instinct to flee")
                steps.append(f"3. Risk of attack is very high")
                steps.append(f"4. Health could drop to critical levels")
            elif predator_attrs.get("is_predator") and props.get("is_predator"):
                consequences.append({
                    "effect": "confrontation",
                    "severity": "high",
                    "detail": f"Both {entity} and {noun} are predators — territorial confrontation likely",
                    "health_impact": -40,
                })
                steps.append(f"1. {entity.title()} encounters {noun}")
                steps.append(f"2. Both are predators — territorial dispute likely")
                steps.append(f"3. Physical confrontation possible")

        # Starvation consequences
        elif noun in ("nothing",) and verb == "eats":
            consequences.append({
                "effect": "starvation",
                "severity": "critical",
                "detail": f"{entity} needs regular food — starvation is life-threatening",
                "health_impact": -50,
            })
            steps.append(f"1. {entity.title()} stops eating")
            steps.append(f"2. Energy reserves deplete over days")
            steps.append(f"3. Health declines steadily")
            steps.append(f"4. Without food, {entity} will become critical within days")

        # Generic fallback
        if not consequences:
            consequences.append({
                "effect": "unknown",
                "severity": "low",
                "detail": f"The effect of {verb} {noun} on {entity} is uncertain but worth monitoring",
                "health_impact": -10,
            })
            steps.append(f"1. {entity.title()} experiences {verb} {noun}")
            steps.append(f"2. Monitor for changes in behavior and health")

        return {"consequences": consequences, "steps": steps}

    def _compute_confidence(self, consequences):
        """Compute overall confidence based on consequence data."""
        if not consequences:
            return 0.0
        severities = {"critical": 0.95, "high": 0.85, "medium": 0.7, "low": 0.5}
        scores = [severities.get(c["severity"], 0.5) for c in consequences]
        return sum(scores) / len(scores)

    def format_answer(self, result):
        """Format hypothetical analysis into readable answer."""
        if not result:
            return None
        parts = []
        parts.append("Here's what would likely happen:")
        parts.append("")
        for step in result["steps"]:
            parts.append(step)
        parts.append("")
        # Health impact summary
        total_health_impact = sum(c.get("health_impact", 0) for c in result["consequences"])
        final_health = max(0, 100 + total_health_impact)
        parts.append(f"Estimated health impact: {total_health_impact}% (final: ~{final_health}%)")
        # Key effects
        for c in result["consequences"]:
            if c["severity"] in ("critical", "high"):
                parts.append(f"Key effect: {c['detail']}")
        parts.append(f"\nConfidence: {result['confidence']:.0%}")
        return "\n".join(parts)


# =========================
# ANSWER IMPROVER
# =========================

class AnswerImprover:
    """Improves previous responses on request. Rebuilds with more detail,
    better structure, examples, and cross-references."""

    def __init__(self, knowledge_engine, noun_hierarchy, property_index):
        self.ke = knowledge_engine
        self.nh = noun_hierarchy
        self.pi = property_index
        self.improvement_history = []  # [(original, improved, timestamp)]

    def improve(self, original_response, query, entities, improvement_type="detail"):
        """Improve a previous response."""
        if improvement_type == "detail":
            return self._add_detail(original_response, query, entities)
        elif improvement_type == "structure":
            return self._restructure(original_response, query, entities)
        elif improvement_type == "examples":
            return self._add_examples(original_response, query, entities)
        elif improvement_type == "comprehensive":
            return self._make_comprehensive(original_response, query, entities)
        return original_response

    def _add_detail(self, response, query, entities):
        """Add more detail to response."""
        additions = []
        for entity in entities:
            e = entity.lower()
            props = self.pi.get_full_properties(e)
            # Add body temperature if relevant
            if props.get("body_temp_f") and "temperature" not in response.lower():
                additions.append(f"Normal body temperature: {props['body_temp_f']}°F")
            # Add care requirements
            reqs = self.pi.get_care_requirements(e)
            if reqs:
                additions.append(f"Care needs: {', '.join(reqs[:3])}")
            # Add hierarchy context
            ancestors = self.nh.get_ancestors(e, 3)
            if ancestors:
                additions.append(f"Classification: {' > '.join(ancestors)}")
        if additions:
            return response + "\n\nAdditional details:\n" + "\n".join(f"- {a}" for a in additions)
        return response

    def _restructure(self, response, query, entities):
        """Restructure response with better formatting."""
        lines = response.split("\n")
        structured = ["## Response\n"]
        for line in lines:
            if line.strip():
                if line.strip().startswith(("-", "•", "*")):
                    structured.append(f"  {line.strip()}")
                elif any(line.strip().startswith(str(i)) for i in range(1, 10)):
                    structured.append(f"  {line.strip()}")
                else:
                    structured.append(f"> {line.strip()}")
        return "\n".join(structured)

    def _add_examples(self, response, query, entities):
        """Add examples from KB data."""
        examples = []
        for entity in entities:
            e = entity.lower()
            data = self.ke.entities.get(e, {})
            descs = data.get("descriptions", [])
            if descs:
                examples.append(f"Example: {descs[0]}")
            # Add related QA pairs
            for q, a in self.ke.dataset_qa:
                if e in q.lower() and len(examples) < 3:
                    examples.append(f"Q: {q}\nA: {a}")
        if examples:
            return response + "\n\nExamples:\n" + "\n\n".join(examples)
        return response

    def _make_comprehensive(self, response, query, entities):
        """Make response comprehensive with all improvements."""
        result = response
        result = self._add_detail(result, query, entities)
        result = self._add_examples(result, query, entities)
        result = self._restructure(result, query, entities)
        # Add cross-references
        if entities and len(entities) > 1:
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    shared = self.nh.get_shared_properties(e1, e2)
                    if shared:
                        result += f"\n\nShared traits ({e1} & {e2}): {', '.join(list(shared.keys())[:5])}"
        self.improvement_history.append({
            "original": response,
            "improved": result,
            "timestamp": time.time(),
        })
        return result


# =========================
# SIMULATION ENGINE
# =========================

class SimulationEngine:
    """Runs step-by-step simulations of hypothetical scenarios.
    Tests consequences, tracks state changes, logs potential outcomes."""

    def __init__(self, knowledge_engine, noun_hierarchy, property_index, lifecycle_tracker):
        self.ke = knowledge_engine
        self.nh = noun_hierarchy
        self.pi = property_index
        self.lt = lifecycle_tracker
        self.simulation_log = []

    def simulate(self, entity, scenario, turns=5):
        """Run a simulation for entity in scenario."""
        e = entity.lower()
        props = self.pi.get_full_properties(e)
        state = {"health": 100, "mood": "normal", "position": "safe", "condition": "good"}
        log = [{"turn": 0, "state": state.copy(), "event": "simulation_start"}]

        for turn in range(1, turns + 1):
            event = self._simulate_turn(e, state, scenario, props, turn)
            state.update(event.get("state_changes", {}))
            state["health"] = max(0, min(100, state["health"]))
            log.append({
                "turn": turn,
                "state": state.copy(),
                "event": event.get("description", "nothing happened"),
            })
            if state["health"] <= 0:
                log.append({"turn": turn, "state": state.copy(), "event": "entity_died"})
                break

        self.simulation_log.append({
            "entity": e,
            "scenario": scenario,
            "log": log,
            "timestamp": time.time(),
        })
        return log

    def _simulate_turn(self, entity, state, scenario, props, turn):
        """Simulate one turn of a scenario."""
        changes = {}
        description = ""

        scenario_lower = scenario.lower()

        # Vision loss scenario
        if "vision" in scenario_lower or "eye" in scenario_lower or "sight" in scenario_lower:
            if turn == 1:
                description = f"{entity} loses vision — becomes disoriented"
                changes["health"] = state["health"] - 20
                changes["mood"] = "anxious"
            elif turn == 2:
                description = f"{entity} bumps into objects — minor injuries"
                changes["health"] = state["health"] - 10
            elif turn == 3:
                description = f"{entity} struggles to find food and water"
                changes["health"] = state["health"] - 15
                changes["condition"] = "struggling"
            elif turn == 4:
                description = f"{entity} adapts somewhat using other senses"
                changes["health"] = state["health"] - 5
                changes["mood"] = "cautious"
            else:
                description = f"{entity} continues to adapt but remains vulnerable"
                changes["health"] = state["health"] - 3

        # Wet scenario
        elif "wet" in scenario_lower or "water" in scenario_lower:
            if props.get("is_aquatic"):
                description = f"{entity} is aquatic — comfortable in water"
                changes["health"] = state["health"] + 2
            elif props.get("afraid_of_water"):
                description = f"{entity} is stressed from being wet (afraid of water)"
                changes["health"] = state["health"] - 10
                changes["mood"] = "stressed"
            elif props.get("is_warm_blooded"):
                body_temp = props.get("body_temp_f", 101.0)
                description = f"{entity} gets cold — body temp dropping from {body_temp}°F"
                changes["health"] = state["health"] - 15
                changes["condition"] = "cold"
            else:
                description = f"{entity} gets wet — uncomfortable but okay"
                changes["health"] = state["health"] - 5

        # Predator encounter
        elif "predator" in scenario_lower or "wolf" in scenario_lower or "dog" in scenario_lower:
            if props.get("is_prey"):
                description = f"{entity} encounters predator — tries to flee"
                changes["health"] = state["health"] - 30
                changes["mood"] = "terrified"
                changes["position"] = "fleeing"
            elif props.get("is_predator"):
                description = f"{entity} confronts another predator — territorial dispute"
                changes["health"] = state["health"] - 15
                changes["mood"] = "aggressive"
            else:
                description = f"{entity} encounters unknown animal — cautious"
                changes["health"] = state["health"] - 5

        # Starvation
        elif "hungry" in scenario_lower or "starv" in scenario_lower or "no food" in scenario_lower:
            description = f"{entity} weakens from hunger — day {turn}"
            changes["health"] = state["health"] - (10 + turn * 3)
            if turn >= 3:
                changes["condition"] = "starving"
                changes["mood"] = "lethargic"

        # Cold
        elif "cold" in scenario_lower:
            if props.get("is_warm_blooded"):
                description = f"{entity} loses body heat — hypothermia risk"
                changes["health"] = state["health"] - 12
                changes["condition"] = "cold"
            else:
                description = f"{entity} slows down in cold (cold-blooded)"
                changes["health"] = state["health"] - 5

        # Default
        else:
            description = f"{entity} experiences {scenario} — monitoring"
            changes["health"] = state["health"] - 5

        return {"state_changes": changes, "description": description}

    def get_simulation_summary(self, sim_log):
        """Get summary of a simulation."""
        if not sim_log:
            return "No simulation data"
        entity = sim_log[0].get("state", {}).get("entity", "unknown")
        final_state = sim_log[-1]["state"]
        events = [s["event"] for s in sim_log if s["event"] != "simulation_start"]
        return {
            "entity": entity,
            "turns": len(sim_log) - 1,
            "final_health": final_state.get("health", 0),
            "final_mood": final_state.get("mood", "unknown"),
            "final_condition": final_state.get("condition", "unknown"),
            "events": events,
            "survived": final_state.get("health", 0) > 0,
        }


# =========================
# CONTRADICTION DETECTOR
# =========================

class ContradictionDetector:
    """Finds contradictory statements across QA pairs, descriptions, and user input.
    Prepares counter-examples and opposite references."""

    def __init__(self, knowledge_engine, noun_hierarchy):
        self.ke = knowledge_engine
        self.nh = noun_hierarchy
        self.known_contradictions = []
        self.opposite_pairs = {
            "is_predator": "is_prey",
            "is_aquatic": "terrestrial",
            "is_warm_blooded": "is_cold_blooded",
            "has_feathers": "has_fur_or_hair",
            "can_fly": "cannot_fly",
            "afraid_of_water": "likes_water",
            "is_nocturnal": "is_diurnal",
            "lays_eggs": "has_live_birth",
        }

    def check_contradiction(self, statement, entity):
        """Check if a statement contradicts known data."""
        e = entity.lower()
        props = self.nh.get_properties(e)
        kb_data = self.ke.entities.get(e, {})
        attrs = kb_data.get("attributes", {})
        descs = kb_data.get("descriptions", [])

        contradictions = []

        # Check against hierarchy properties
        for prop_key, opposite_key in self.opposite_pairs.items():
            if prop_key in statement.lower() and prop_key in attrs:
                if not attrs[prop_key] and opposite_key in attrs and attrs[opposite_key]:
                    contradictions.append({
                        "type": "attribute_contradiction",
                        "claim": statement,
                        "fact": f"{e} has {prop_key}={attrs[prop_key]}, not {opposite_key}",
                        "confidence": 0.9,
                    })
            if opposite_key in statement.lower() and opposite_key in attrs:
                if not attrs[opposite_key] and prop_key in attrs and attrs[prop_key]:
                    contradictions.append({
                        "type": "attribute_contradiction",
                        "claim": statement,
                        "fact": f"{e} has {opposite_key}={attrs[opposite_key]}, not {prop_key}",
                        "confidence": 0.9,
                    })

        # Check against descriptions
        for desc in descs:
            desc_lower = desc.lower()
            # Simple negation check
            if "always" in statement.lower() and "never" in desc_lower:
                contradictions.append({
                    "type": "always_never",
                    "claim": statement,
                    "fact": desc,
                    "confidence": 0.7,
                })
            if "can't" in statement.lower() or "cannot" in statement.lower():
                if any(action in desc_lower for action in ["can", "able to", "capable of"]):
                    contradictions.append({
                        "type": "capability_contradiction",
                        "claim": statement,
                        "fact": desc,
                        "confidence": 0.6,
                    })

        return contradictions

    def find_opposite_examples(self, entity):
        """Find opposite examples for an entity."""
        e = entity.lower()
        attrs = self.ke.entities.get(e, {}).get("attributes", {})
        examples = []

        for prop_key, opposite_key in self.opposite_pairs.items():
            if prop_key in attrs:
                # Find entities with the opposite property
                for other_entity, other_data in self.ke.entities.items():
                    if other_entity != e:
                        other_attrs = other_data.get("attributes", {})
                        if opposite_key in other_attrs and other_attrs[opposite_key]:
                            examples.append({
                                "entity": other_entity,
                                "property": opposite_key,
                                "value": True,
                                "contrast_with": e,
                                "contrast_property": prop_key,
                            })
                            break

        return examples

    def prepare_counter_examples(self, claim, entity):
        """Prepare counter-examples for a false claim."""
        e = entity.lower()
        contradictions = self.check_contradiction(claim, e)
        counter_examples = []

        for c in contradictions:
            counter_examples.append({
                "counter_claim": c["fact"],
                "source": c["type"],
                "confidence": c["confidence"],
            })

        # Add opposite examples
        opposites = self.find_opposite_examples(e)
        for opp in opposites[:2]:
            counter_examples.append({
                "counter_claim": f"Unlike {e}, {opp['entity']} has {opp['property']}",
                "source": "opposite_example",
                "confidence": 0.8,
            })

        return counter_examples

    def to_dict(self):
        return {
            "known_contradictions": self.known_contradictions,
            "opposite_pairs": self.opposite_pairs,
        }


# =========================
# SELF-AWARENESS TRACKER
# =========================

class SelfAwarenessTracker:
    """Tracks AI's own actions, decisions, what it's been doing.
    Answers questions about itself: 'what are you', 'how long have you been watching cat'."""

    def __init__(self):
        self.action_log = []  # [{action, target, timestamp, duration, details}]
        self.decision_log = []  # [{decision, reason, timestamp, outcome}]
        self.entity_watch_log = defaultdict(list)  # entity -> [{action, timestamp, duration}]
        self.file_edit_log = []  # [{file, action, timestamp, duration}]
        self.session_start = time.time()
        self.total_queries = 0
        self.total_responses = 0
        self.current_focus = None  # what AI is currently focused on
        self.focus_start = None
        self.self_description = (
            "I'm a self-aware AI assistant with hierarchical knowledge reasoning, "
            "lifecycle tracking, hypothetical analysis, simulation capabilities, "
            "and background dataset analysis. I can reason about entities through "
            "a 10-level taxonomy tree, predict consequences, run simulations, "
            "and learn from our conversations."
        )

    def log_action(self, action, target=None, details=None, duration=0):
        """Log an action the AI performed."""
        entry = {
            "action": action,
            "target": target,
            "timestamp": time.time(),
            "duration": duration,
            "details": details,
        }
        self.action_log.append(entry)
        if target:
            self.entity_watch_log[target.lower()].append(entry)
        # Keep last 200 actions
        if len(self.action_log) > 200:
            self.action_log = self.action_log[-200:]

    def log_decision(self, decision, reason=None, outcome=None):
        """Log a decision the AI made."""
        self.decision_log.append({
            "decision": decision,
            "reason": reason,
            "outcome": outcome,
            "timestamp": time.time(),
        })
        if len(self.decision_log) > 100:
            self.decision_log = self.decision_log[-100:]

    def log_file_edit(self, filepath, action="edit", duration=0):
        """Log a file edit operation."""
        self.file_edit_log.append({
            "file": filepath,
            "action": action,
            "timestamp": time.time(),
            "duration": duration,
        })

    def set_focus(self, target):
        """Set what the AI is currently focused on."""
        if self.current_focus and self.focus_start:
            elapsed = time.time() - self.focus_start
            self.log_action("focus_end", self.current_focus, duration=elapsed)
        self.current_focus = target
        self.focus_start = time.time()
        self.log_action("focus_start", target)

    def increment_queries(self):
        self.total_queries += 1

    def increment_responses(self):
        self.total_responses += 1

    def get_uptime(self):
        """Get how long the AI has been running."""
        return time.time() - self.session_start

    def get_watch_time(self, entity):
        """How long has the AI been watching/tracking an entity."""
        entries = self.entity_watch_log.get(entity.lower(), [])
        if not entries:
            return None
        total = sum(e.get("duration", 0) for e in entries)
        first = entries[0]["timestamp"]
        last = entries[-1]["timestamp"]
        return {
            "total_duration": total,
            "first_seen": first,
            "last_seen": last,
            "interaction_count": len(entries),
            "time_since_first": time.time() - first,
        }

    def get_file_edit_time(self, filepath=None):
        """How long has the AI been editing a file."""
        edits = self.file_edit_log
        if filepath:
            edits = [e for e in edits if e["file"] == filepath]
        if not edits:
            return None
        total = sum(e.get("duration", 0) for e in edits)
        first = edits[0]["timestamp"]
        last = edits[-1]["timestamp"]
        return {
            "total_duration": total,
            "first_edit": first,
            "last_edit": last,
            "edit_count": len(edits),
            "files": list(set(e["file"] for e in edits)),
        }

    def get_recent_actions(self, n=10):
        """Get N most recent actions."""
        return self.action_log[-n:]

    def get_recent_decisions(self, n=5):
        """Get N most recent decisions."""
        return self.decision_log[-n:]

    def get_stats(self):
        """Get overall stats."""
        return {
            "uptime_seconds": self.get_uptime(),
            "uptime_formatted": self._format_duration(self.get_uptime()),
            "total_queries": self.total_queries,
            "total_responses": self.total_responses,
            "actions_logged": len(self.action_log),
            "decisions_made": len(self.decision_log),
            "files_edited": len(set(e["file"] for e in self.file_edit_log)),
            "entities_tracked": len(self.entity_watch_log),
            "current_focus": self.current_focus,
        }

    def _format_duration(self, seconds):
        """Format seconds into human-readable duration."""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.1f} hours"
        else:
            return f"{seconds/86400:.1f} days"

    def answer_self_question(self, query):
        """Answer questions about the AI itself."""
        q = query.lower()

        # "what are you" / "what kind of AI"
        if any(w in q for w in ["what are you", "what kind", "what type", "describe yourself", "who are you"]):
            return self.self_description

        # "how long have you been watching [entity]"
        watch_match = re.search(r"how (?:long|much time).*watch(?:ing)?\s+(?:my\s+)?(\w+)", q)
        if watch_match:
            entity = watch_match.group(1)
            watch = self.get_watch_time(entity)
            if watch:
                return (f"I've been tracking {entity} for {self._format_duration(watch['time_since_first'])}. "
                        f"I've logged {watch['interaction_count']} interactions with {entity}.")
            return f"I haven't been tracking {entity} yet in this session."

        # "how long have you been editing [file]"
        edit_match = re.search(r"how (?:long|much time).*edit(?:ing)?\s+(.+?)(?:\s|$)", q)
        if edit_match:
            filepath = edit_match.group(1).strip()
            edits = self.get_file_edit_time(filepath)
            if edits:
                return (f"I've been editing {filepath} for {self._format_duration(edits['total_duration'])}. "
                        f"{edits['edit_count']} edits across {len(edits['files'])} file(s).")
            # Try partial match
            for e in self.file_edit_log:
                if filepath in e["file"]:
                    return (f"I've edited {e['file']} {self._format_duration(e.get('duration', 0))}.")
            return f"I haven't edited {filepath} in this session."

        # "how long have you been running" / "uptime"
        if any(w in q for w in ["how long", "uptime", "running", "session"]):
            stats = self.get_stats()
            return (f"I've been running for {stats['uptime_formatted']}. "
                    f"Processed {stats['total_queries']} queries, made {stats['actions_logged']} actions, "
                    f"tracked {stats['entities_tracked']} entities.")

        # "what have you been doing" / "recent actions"
        if any(w in q for w in ["what have you been doing", "recent actions", "what did you do", "activity"]):
            recent = self.get_recent_actions(5)
            if recent:
                lines = ["Recent actions:"]
                for a in recent:
                    elapsed = self._format_duration(time.time() - a["timestamp"])
                    lines.append(f"  - {a['action']} {a.get('target', '')} ({elapsed} ago)")
                return "\n".join(lines)
            return "No recent actions logged."

        # "what are you working on" / "current focus"
        if any(w in q for w in ["working on", "focus", "currently"]):
            if self.current_focus:
                elapsed = self._format_duration(time.time() - self.focus_start) if self.focus_start else "unknown"
                return f"Currently focused on: {self.current_focus} (for {elapsed})"
            return "I'm not currently focused on anything specific."

        # "what decisions have you made"
        if any(w in q for w in ["decisions", "why did you", "reasoning"]):
            recent = self.get_recent_decisions(3)
            if recent:
                lines = ["Recent decisions:"]
                for d in recent:
                    lines.append(f"  - {d['decision']}: {d.get('reason', 'no reason given')}")
                return "\n".join(lines)
            return "No recent decisions logged."

        # "can you [do something]"
        can_match = re.search(r"can you (.+?)(?:\?|$)", q)
        if can_match:
            action = can_match.group(1).strip()
            return self._explain_capability(action)

        return None

    def _explain_capability(self, action):
        """Explain whether the AI can do something and how."""
        capabilities = {
            "reason": "Yes, I can reason through entity hierarchies, derive consequences, and build logical arguments.",
            "remember": "Yes, I track conversation history, entity states, and learned facts across our session.",
            "learn": "Yes, I can learn from your statements, index new information, and update my knowledge.",
            "predict": "Yes, I can predict outcomes using hypothetical analysis and simulation engines.",
            "simulate": "Yes, I can run multi-turn simulations of scenarios and track health/state changes.",
            "test": "Yes, I can test my own responses for consistency and verify them against my knowledge base.",
            "analyze": "Yes, I analyze datasets in the background, extract entities, find contradictions, and build profiles.",
            "edit": "Yes, I can edit files and track all my edits with timestamps and durations.",
            "track": "Yes, I track entity health, lifecycle states, conversation history, and usage patterns.",
            "compare": "Yes, I compare entities using shared hierarchy properties and find similarities/differences.",
            "explain": "Yes, I can explain my reasoning, decisions, and how I arrived at answers.",
            "improve": "Yes, I can improve previous responses by adding detail, structure, examples, and cross-references.",
        }
        for keyword, explanation in capabilities.items():
            if keyword in action:
                return explanation
        return (f"I can try to {action}. My capabilities include: "
                "reasoning, remembering, learning, predicting, simulating, testing, "
                "analyzing, editing, tracking, comparing, explaining, and improving.")


# =========================
# USER USAGE TRACKER
# =========================

class UserUsageTracker:
    """Tracks user patterns: what they ask, frequency, topics, preferences."""

    def __init__(self):
        self.query_log = []  # [{query, timestamp, entities, topic, response_source}]
        self.topic_frequency = defaultdict(int)  # topic -> count
        self.entity_frequency = defaultdict(int)  # entity -> count
        self.source_preference = defaultdict(int)  # source -> count (what responses user likes)
        self.session_queries = 0
        self.session_start = time.time()
        self.preferred_detail_level = "normal"
        self.common_patterns = []

    def log_query(self, query, entities=None, response_source=None, user_feedback=None):
        """Log a user query."""
        entry = {
            "query": query,
            "timestamp": time.time(),
            "entities": entities or [],
            "response_source": response_source,
            "user_feedback": user_feedback,
        }
        self.query_log.append(entry)
        self.session_queries += 1

        # Update frequencies
        for e in (entities or []):
            self.entity_frequency[e.lower()] += 1

        # Detect topic
        topic = self._detect_topic(query)
        if topic:
            self.topic_frequency[topic] += 1

        if response_source:
            self.source_preference[response_source] += 1

        # Keep last 100 queries
        if len(self.query_log) > 100:
            self.query_log = self.query_log[-100:]

    def _detect_topic(self, query):
        """Detect the topic of a query."""
        q = query.lower()
        topics = {
            "animal_care": ["care", "feed", "water", "shelter", "health", "vet"],
            "hypothetical": ["what if", "what would happen", "suppose", "imagine"],
            "comparison": ["compare", "versus", "difference", "better", "worse"],
            "lifecycle": ["died", "dead", "born", "alive", "injured", "sick"],
            "physical": ["shape", "color", "size", "weight", "temperature"],
            "behavior": ["why", "how does", "behavior", "habit", "instinct"],
            "capability": ["can", "able to", "possibility"],
            "preparation": ["should i", "bring", "pack", "prepare", "need"],
            "ai_self": ["what are you", "how long", "what have you done", "can you"],
        }
        for topic, keywords in topics.items():
            if any(w in q for w in keywords):
                return topic
        return "general"

    def get_user_profile(self):
        """Build a profile of the user's interests and patterns."""
        top_entities = sorted(self.entity_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        top_topics = sorted(self.topic_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        top_sources = sorted(self.source_preference.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "session_queries": self.session_queries,
            "session_duration": time.time() - self.session_start,
            "top_entities": top_entities,
            "top_topics": top_topics,
            "preferred_sources": top_sources,
            "detail_level": self.preferred_detail_level,
            "avg_queries_per_minute": self.session_queries / max(1, (time.time() - self.session_start) / 60),
        }

    def get_usage_summary(self):
        """Get a summary of user usage."""
        profile = self.get_user_profile()
        lines = [f"Session: {profile['session_queries']} queries over {profile['session_duration']/60:.1f} minutes"]
        if profile["top_entities"]:
            lines.append(f"Most asked about: {', '.join(f'{e}({c})' for e, c in profile['top_entities'][:5])}")
        if profile["top_topics"]:
            lines.append(f"Main topics: {', '.join(f'{t}({c})' for t, c in profile['top_topics'][:3])}")
        if profile["preferred_sources"]:
            lines.append(f"Preferred response types: {', '.join(f'{s}({c})' for s, c in profile['preferred_sources'][:3])}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "query_log": self.query_log[-50:],
            "topic_frequency": dict(self.topic_frequency),
            "entity_frequency": dict(self.entity_frequency),
            "source_preference": dict(self.source_preference),
            "session_queries": self.session_queries,
        }

    def load_dict(self, data):
        self.query_log = data.get("query_log", [])
        self.topic_frequency = defaultdict(int, data.get("topic_frequency", {}))
        self.entity_frequency = defaultdict(int, data.get("entity_frequency", {}))
        self.source_preference = defaultdict(int, data.get("source_preference", {}))
        self.session_queries = data.get("session_queries", 0)


# =========================
# SELF-TEST ENGINE
# =========================

class SelfTestEngine:
    """Tests AI's own responses for consistency, accuracy, and quality.
    Can initiate tests and track results."""

    def __init__(self, knowledge_engine, noun_hierarchy):
        self.ke = knowledge_engine
        self.nh = noun_hierarchy
        self.test_results = []  # [{test_name, query, expected, actual, passed, timestamp}]
        self.consistency_cache = {}  # query -> response (for checking same query gets similar answer)
        self.test_suites = {
            "basic_facts": [
                ("is a cat a predator", "yes_no", "cat", "is_predator"),
                ("can a cat fly", "yes_no", "cat", "can_fly"),
                ("can a dog swim", "yes_no", "dog", "can_swim"),
            ],
            "hierarchy": [
                ("what is a cat", "definition", "cat", None),
                ("compare cats and dogs", "comparison", "cat", "dog"),
            ],
            "lifecycle": [
                ("my cat died", "lifecycle", "cat", "death"),
                ("my cat came back to life", "lifecycle", "cat", "resurrection"),
            ],
        }

    def run_test(self, query, expected_source=None, expected_entity=None, expected_property=None):
        """Run a single test and record result."""
        # This would need a reference to the pipeline to actually process
        # For now, record the test definition
        test = {
            "query": query,
            "expected_source": expected_source,
            "expected_entity": expected_entity,
            "expected_property": expected_property,
            "timestamp": time.time(),
            "status": "defined",
        }
        self.test_results.append(test)
        return test

    def record_result(self, query, actual_source, actual_response, passed):
        """Record a test result."""
        # Find the matching test definition
        for test in self.test_results:
            if test["query"] == query and test.get("status") == "defined":
                test["actual_source"] = actual_source
                test["actual_response"] = actual_response
                test["passed"] = passed
                test["status"] = "completed"
                break

    def check_consistency(self, query, response):
        """Check if a response is consistent with previous responses to the same query."""
        normalized = query.lower().strip()
        if normalized in self.consistency_cache:
            prev = self.consistency_cache[normalized]
            # Simple similarity check
            prev_words = set(prev.lower().split())
            curr_words = set(response.lower().split())
            overlap = len(prev_words & curr_words) / max(len(prev_words | curr_words), 1)
            self.consistency_cache[normalized] = response
            return {"consistent": overlap > 0.3, "similarity": overlap, "previous": prev[:100]}
        self.consistency_cache[normalized] = response
        return {"consistent": True, "similarity": 1.0, "previous": None}

    def test_entity_properties(self, entity):
        """Test that entity properties are consistent across hierarchy and KB."""
        hier_props = self.nh.get_properties(entity)
        kb_data = self.ke.entities.get(entity.lower(), {})
        kb_attrs = kb_data.get("attributes", {})

        results = []
        # Check hierarchy says is_predator matches KB
        if "is_predator" in hier_props and "is_predator" in kb_attrs:
            match = hier_props["is_predator"] == kb_attrs["is_predator"]
            results.append({"property": "is_predator", "hierarchy": hier_props["is_predator"],
                          "kb": kb_attrs["is_predator"], "consistent": match})
        # Check body temp
        if "body_temp_f" in hier_props:
            results.append({"property": "body_temp_f", "hierarchy": hier_props["body_temp_f"],
                          "kb": kb_attrs.get("body_temp_f"), "consistent": True})
        return results

    def get_test_summary(self):
        """Get summary of all tests."""
        completed = [t for t in self.test_results if t.get("status") == "completed"]
        passed = [t for t in completed if t.get("passed")]
        return {
            "total_defined": len(self.test_results),
            "total_completed": len(completed),
            "total_passed": len(passed),
            "pass_rate": len(passed) / max(1, len(completed)),
        }

    def to_dict(self):
        return {
            "test_results": self.test_results[-50:],
            "consistency_cache": {k: v[:50] for k, v in self.consistency_cache.items()},
        }

    def load_dict(self, data):
        self.test_results = data.get("test_results", [])
        self.consistency_cache = data.get("consistency_cache", {})


# =========================
# POSSIBILITY DETECTION & PLANNING
# =========================

class PossibilityDetector:
    """Detects when user references predictions, preparations, or past expectations.
    Silently builds better answers by knowing what was expected vs what happened."""

    PREDICTION_PATTERNS = [
        r"i (told|knew|predicted|expected|guessed|suspected) (you|this|so|it)",
        r"(this|that) (was|i )?(is )?(what i |already )?(expected|predicted|anticipated|figured)",
        r"since i (already |already )?(expected|predicted|knew|figured|anticipated)",
        r"(because|since) i (already |already )?(told|knew|predicted|expected)",
        r"i (already )?prepared (for|this|that|it)",
        r"(this|that) (is )?(exactly )?(what i )?(prepared|planned|expected|predicted)",
        r"as i (expected|predicted|said|told you)",
        r"remember (when|that) i (said|told|predicted|expected)",
        r"you (should|could|would) (have )?(done|known|checked|looked)",
        r"(go|look|check|extract|get) (the |that |my )?(data|file|info|result|answer) i (already )?(prepared|saved|stored|left)",
    ]

    PREPARATION_PATTERNS = [
        r"(go|look|check|extract|get|use) (the |that |my )?(data|file|info|result|answer|thing)",
        r"(i |we )(already|already )?(have|had|saved|stored|prepared|done|set up|built|created)",
        r"(there|it)(s|'s| is) (already |already )?(a |one |some )?(file|result|data|info|answer|thing)",
        r"(check|look at|use) (my |the )?(previous|earlier|last|existing|prior)",
    ]

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.predictions = {}  # entity -> list of {"text": str, "timestamp": float, "fulfilled": bool}
        self.plans = {}  # entity -> list of {"goal": str, "steps": list, "status": str}
        self.prepared_data = {}  # entity -> list of {"topic": str, "data": str, "timestamp": float}

    def detect_prediction_reference(self, text):
        """Check if user text references a past prediction or expectation."""
        text_lower = text.lower()
        for pattern in self.PREDICTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def detect_preparation_reference(self, text):
        """Check if user text references prepared data or prior work."""
        text_lower = text.lower()
        for pattern in self.PREPARATION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def record_prediction(self, entity, prediction_text):
        """Record a prediction made about an entity."""
        entity = entity.lower()
        if entity not in self.predictions:
            self.predictions[entity] = []
        self.predictions[entity].append({
            "text": prediction_text,
            "timestamp": time.time(),
            "fulfilled": False,
        })

    def record_preparation(self, entity, topic, data):
        """Record prepared data for later reference."""
        entity = entity.lower()
        if entity not in self.prepared_data:
            self.prepared_data[entity] = []
        self.prepared_data[entity].append({
            "topic": topic,
            "data": data,
            "timestamp": time.time(),
        })

    def get_context_for_entity(self, entity):
        """Get all prediction/preparation context for building richer answers."""
        entity = entity.lower()
        context = {
            "predictions": self.predictions.get(entity, []),
            "prepared_data": self.prepared_data.get(entity, []),
            "plans": self.plans.get(entity, []),
        }
        return context

    def check_prediction_status(self, entity, current_state):
        """Check if any predictions about an entity have been fulfilled or contradicted."""
        entity = entity.lower()
        predictions = self.predictions.get(entity, [])
        results = []
        for pred in predictions:
            if pred["fulfilled"]:
                continue
            # Simple keyword check for now
            pred_words = set(pred["text"].lower().split())
            state_words = set(current_state.lower().split())
            overlap = len(pred_words & state_words)
            if overlap >= 2:
                pred["fulfilled"] = True
                results.append({"prediction": pred["text"], "status": "fulfilled"})
        return results


class PlanBuilder:
    """Builds and tracks plans for answering questions. Knows what's already
    been said and what the best next step is."""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.active_plans = {}  # query_key -> {"goal": str, "steps": list, "current_step": int, "facts_used": set}
        self.completed_plans = {}  # query_key -> plan dict
        self.step_templates = [
            "identify entity and core definition",
            "gather primary attributes from KB",
            "find related entities for comparison",
            "check hierarchy properties",
            "compose multi-source answer",
            "validate against dataset QA",
            "add supporting evidence",
            "verify no contradictions",
            "compose final response",
        ]

    def create_plan(self, query, entities, facts_available):
        """Create an answering plan for a query."""
        query_key = re.sub(r'\s+', ' ', query.lower().strip().rstrip("?"))
        steps = []
        for i, template in enumerate(self.step_templates):
            steps.append({
                "description": template,
                "status": "pending",
                "facts_used": set(),
            })
        plan = {
            "goal": query,
            "entities": entities,
            "facts_available": facts_available,
            "steps": steps,
            "current_step": 0,
            "facts_used": set(),
            "start_time": time.time(),
        }
        self.active_plans[query_key] = plan
        return plan

    def get_next_step(self, query):
        """Get the next pending step for a query's plan."""
        query_key = re.sub(r'\s+', ' ', query.lower().strip().rstrip("?"))
        plan = self.active_plans.get(query_key)
        if not plan:
            return None
        for step in plan["steps"]:
            if step["status"] == "pending":
                return step
        return None

    def mark_step_done(self, query, step_desc, facts_used=None):
        """Mark a step as completed and record what facts were used."""
        query_key = re.sub(r'\s+', ' ', query.lower().strip().rstrip("?"))
        plan = self.active_plans.get(query_key)
        if not plan:
            return
        for step in plan["steps"]:
            if step["description"] == step_desc and step["status"] == "pending":
                step["status"] = "done"
                if facts_used:
                    step["facts_used"] = facts_used
                    plan["facts_used"].update(facts_used)
                plan["current_step"] += 1
                break

    def complete_plan(self, query):
        """Move a plan from active to completed."""
        query_key = re.sub(r'\s+', ' ', query.lower().strip().rstrip("?"))
        plan = self.active_plans.pop(query_key, None)
        if plan:
            plan["end_time"] = time.time()
            plan["status"] = "completed"
            self.completed_plans[query_key] = plan

    def get_facts_already_used(self, query):
        """Return all facts already used in answering this query."""
        query_key = re.sub(r'\s+', ' ', query.lower().strip().rstrip("?"))
        plan = self.active_plans.get(query_key)
        if plan:
            return plan["facts_used"]
        return set()

    def get_progress(self, query):
        """Get progress percentage for a plan."""
        query_key = re.sub(r'\s+', ' ', query.lower().strip().rstrip("?"))
        plan = self.active_plans.get(query_key)
        if not plan:
            return 0.0
        done = sum(1 for s in plan["steps"] if s["status"] == "done")
        return done / max(len(plan["steps"]), 1)


# =========================
# ELABORATION ENGINE
# =========================

class ElaborationEngine:
    """Handles 'elaborate', 'tell me more', 'continue' by providing non-repeating
    additional info from KB attributes, properties, hierarchy, and dataset QA."""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.elaboration_history = defaultdict(list)  # entity -> list of texts already said
        self.elaboration_depth = defaultdict(int)  # entity -> how many times elaborated

    def is_elaboration_request(self, text):
        """Detect if user wants to elaborate on a previous answer."""
        text_lower = text.lower().strip().rstrip("?")
        patterns = [
            r"^(elaborate|more|tell me more|continue|go on|what else|and\?|keep going)$",
            r"(elaborate|more detail|tell me more|continue|go on|what else|keep going)",
            r"(also|additionally|furthermore|what about|how about|and also)",
            r"(what else|anything else|more info|more information|expand)",
        ]
        return any(re.search(p, text_lower) for p in patterns)

    def get_elaboration(self, entity, query_words=None):
        """Generate non-repeating elaboration for an entity."""
        entity_lower = entity.lower()
        data = self.ke.entities.get(entity_lower, {})
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        descs = data.get("descriptions", [])

        # Track what's been said
        said = set(self.elaboration_history[entity_lower])
        depth = self.elaboration_depth[entity_lower]

        parts = []

        # Layer 1: unused descriptions (depth 0-1)
        if depth < 2:
            for d in descs:
                if d.lower() not in said:
                    parts.append(d)
                    said.add(d.lower())
                    if len(parts) >= 2:
                        break

        # Layer 2: unused boolean attributes (depth 1-3)
        if depth >= 1 and depth < 4:
            unused_attrs = [(k, v) for k, v in attrs.items()
                           if isinstance(v, bool) and v
                           and k.lower() not in said]
            random.shuffle(unused_attrs)
            for k, v in unused_attrs[:2]:
                formatted = self.ke._format_attr_with_verb(k, negative=False, singular=True)
                fact = f"A {entity_lower} {formatted}."
                if fact.lower() not in said:
                    parts.append(fact)
                    said.add(fact.lower())

        # Layer 3: unused properties (depth 2-5)
        if depth >= 2 and depth < 6:
            unused_props = [(k, v) for k, v in props.items()
                           if isinstance(v, str)
                           and f"the {k.replace('_', ' ')}" not in " ".join(said)]
            random.shuffle(unused_props)
            for k, v in unused_props[:2]:
                readable = k.replace("_", " ")
                fact = f"The {readable} of a {entity_lower} is {v}."
                if fact.lower() not in said:
                    parts.append(fact)
                    said.add(fact.lower())

        # Layer 4: hierarchy properties (depth 3+)
        if depth >= 3:
            # Try to find related entities via category
            category = data.get("category", "")
            if category and category in self.ke.entities:
                cat_data = self.ke.entities[category]
                cat_attrs = cat_data.get("attributes", {})
                for k, v in cat_attrs.items():
                    if isinstance(v, bool) and v:
                        formatted = self.ke._format_attr_with_verb(k, negative=False, singular=True)
                        fact = f"As a {category}, a {entity_lower} {formatted}."
                        if fact.lower() not in said:
                            parts.append(fact)
                            said.add(fact.lower())
                            break

        # Record what was said
        self.elaboration_history[entity_lower].extend(parts)
        self.elaboration_depth[entity_lower] = depth + 1

        if not parts:
            return None

        # Combine with connectors
        if len(parts) == 1:
            return parts[0]
        connectors = ["Additionally, ", "Also, ", "Moreover, ", "Furthermore, "]
        result = parts[0]
        for p in parts[1:]:
            result += f" {random.choice(connectors)}{p[0].lower()}{p[1:]}" if p[0].isupper() else f" {random.choice(connectors)}{p}"
        return result

    def reset(self, entity):
        """Reset elaboration tracking for an entity."""
        entity_lower = entity.lower()
        self.elaboration_history[entity_lower] = []
        self.elaboration_depth[entity_lower] = 0


# =========================
# FACT SORTER
# =========================

class FactSorter:
    """Sorts facts from fiction, ranks by support level, and provides
    context-aware fact ordering."""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.fact_cache = {}  # entity -> list of {"text": str, "support": float, "source": str}
        self.contradiction_log = []

    def sort_facts(self, entity, facts_list):
        """Sort facts by support level: KB properties > dataset QA > descriptions > inferred."""
        entity_lower = entity.lower()
        scored = []
        for fact in facts_list:
            score = self._score_fact(entity_lower, fact)
            scored.append({"text": fact, "support": score["support"], "source": score["source"]})
        scored.sort(key=lambda x: x["support"], reverse=True)
        return scored

    def _score_fact(self, entity, fact):
        """Score how well a fact is supported by the KB."""
        fact_lower = fact.lower()
        support = 0.5  # base
        source = "inferred"

        # Check KB properties
        data = self.ke.entities.get(entity, {})
        props = data.get("properties", {})
        attrs = data.get("attributes", {})

        for k, v in props.items():
            if isinstance(v, str) and v.lower() in fact_lower:
                support = 0.95
                source = "kb_property"
                break

        for k, v in attrs.items():
            if isinstance(v, bool):
                verb = k.replace("is_", "").replace("has_", "").replace("_", " ")
                if verb in fact_lower:
                    support = 0.9 if v else 0.3
                    source = "kb_attribute"
                    break

        # Check dataset QA
        for q, a in self.ke.dataset_qa:
            if entity in q.lower() and fact_lower[:30] in a.lower():
                support = 0.85
                source = "dataset_qa"
                break

        # Check descriptions
        for d in data.get("descriptions", []):
            if fact_lower[:20] in d.lower():
                support = 0.7
                source = "description"
                break

        return {"support": support, "source": source}

    def fact_vs_fiction(self, claim, entity):
        """Check if a claim contradicts KB data. Returns verdict + evidence."""
        entity_lower = entity.lower()
        data = self.ke.entities.get(entity_lower, {})
        props = data.get("properties", {})
        attrs = data.get("attributes", {})

        contradictions = []
        support = []

        claim_lower = claim.lower()

        for k, v in props.items():
            if isinstance(v, str) and v.lower() in claim_lower:
                support.append(f"Supported: {k.replace('_', ' ')} = {v}")

        for k, v in attrs.items():
            if isinstance(v, bool):
                verb = k.replace("is_", "").replace("has_", "").replace("_", " ")
                if verb in claim_lower:
                    if v:
                        support.append(f"Confirmed: {entity} {verb}")
                    else:
                        contradictions.append(f"Contradicted: {entity} does NOT {verb}")

        if contradictions:
            return {"verdict": "fiction", "contradictions": contradictions, "support": support}
        elif support:
            return {"verdict": "fact", "contradictions": [], "support": support}
        return {"verdict": "unknown", "contradictions": [], "support": []}


# =========================
# SELF QUESTIONER
# =========================

class SelfQuestioner:
    """AI asks user questions to continue conversation, gather info,
    and show engagement."""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.pending_questions = []  # list of {"question": str, "context": str, "entity": str}
        self.asked_questions = set()
        self.question_templates = [
            "What specifically about {entity} are you most interested in?",
            "Would you like to know about {entity}'s behavior, habitat, or physical traits?",
            "Have you ever seen a {entity} in person?",
            "What made you think about {entity} today?",
            "Should I compare {entity} to a related species?",
            "Do you want me to go deeper into {entity}'s characteristics?",
            "Would you like examples of {entity} in different contexts?",
            "Are you researching {entity} for a specific purpose?",
            "What's your experience with {entity}?",
            "Should I look for more detailed information about {entity}?",
        ]

    def generate_question(self, entity, context=""):
        """Generate a relevant question about an entity to continue conversation."""
        entity_lower = entity.lower()
        available = [t for t in self.question_templates
                     if t not in self.asked_questions]
        if not available:
            return None

        template = random.choice(available)
        question = template.format(entity=entity_lower)
        self.asked_questions.add(template)
        self.pending_questions.append({
            "question": question,
            "context": context,
            "entity": entity_lower,
            "timestamp": time.time(),
        })
        return question

    def get_pending_question(self):
        """Get the most recent pending question."""
        if self.pending_questions:
            return self.pending_questions[-1]
        return None

    def clear_pending(self):
        """Clear pending questions."""
        self.pending_questions = []


# =========================
# RESPONSE SEPARATOR + FOLLOW-UP INDEX
# =========================

class ResponseSeparator:
    """Separates code, files, examples, and final answers like an LLM.
    Supports continuation tags for multi-part responses."""

    def __init__(self):
        self.response_parts = {
            "answer": [],
            "code": [],
            "files": [],
            "examples": [],
            "questions": [],
        }
        self.continuation_tags = []
        self.followup_index = []  # list of {"topic": str, "status": str, "depth": int}

    def separate(self, combined_text):
        """Parse a combined response into categorized parts."""
        parts = {"answer": [], "code": [], "files": [], "examples": [], "questions": []}
        lines = combined_text.split("\n")
        in_code = False
        current_section = "answer"

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                if in_code:
                    current_section = "code"
                else:
                    current_section = "answer"
                continue
            if in_code:
                parts["code"].append(line)
            elif stripped.lower().startswith("file:") or stripped.lower().startswith("example file"):
                parts["files"].append(stripped)
            elif stripped.lower().startswith("example"):
                parts["examples"].append(stripped)
            elif "?" in stripped:
                parts["questions"].append(stripped)
            else:
                parts["answer"].append(line)

        self.response_parts = parts
        return parts

    def add_continuation(self, topic, status="pending"):
        """Add a continuation tag for multi-part responses."""
        self.continuation_tags.append({
            "topic": topic,
            "status": status,
            "timestamp": time.time(),
        })

    def add_followup(self, topic, depth=0):
        """Add a follow-up index entry."""
        self.followup_index.append({
            "topic": topic,
            "status": "pending",
            "depth": depth,
            "timestamp": time.time(),
        })

    def get_formatted_response(self):
        """Format separated parts into a clean response."""
        parts = self.response_parts
        result = []
        if parts["answer"]:
            result.append("\n".join(parts["answer"]))
        if parts["code"]:
            result.append("```\n" + "\n".join(parts["code"]) + "\n```")
        if parts["files"]:
            result.append("Files:\n" + "\n".join(parts["files"]))
        if parts["examples"]:
            result.append("Examples:\n" + "\n".join(parts["examples"]))
        if parts["questions"]:
            result.append("\n".join(parts["questions"]))
        return "\n\n".join(result) if result else ""


# =========================
# PARALLEL STREAM
# =========================

class ParallelStream:
    """Handles multiple answer streams simultaneously. Creates separate
    processing streams for different aspects of a query."""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.streams = {}  # stream_id -> {"topic": str, "status": str, "results": list}
        self.stream_counter = 0

    def create_stream(self, topic, entities):
        """Create a new processing stream for a topic."""
        self.stream_counter += 1
        stream_id = f"stream_{self.stream_counter}"
        self.streams[stream_id] = {
            "topic": topic,
            "entities": entities,
            "status": "active",
            "results": [],
            "created_at": time.time(),
        }
        return stream_id

    def add_result(self, stream_id, result):
        """Add a result to a stream."""
        if stream_id in self.streams:
            self.streams[stream_id]["results"].append(result)

    def get_stream(self, stream_id):
        """Get a stream by ID."""
        return self.streams.get(stream_id)

    def complete_stream(self, stream_id):
        """Mark a stream as complete."""
        if stream_id in self.streams:
            self.streams[stream_id]["status"] = "complete"
            self.streams[stream_id]["end_time"] = time.time()

    def get_all_active(self):
        """Get all active streams."""
        return {k: v for k, v in self.streams.items() if v["status"] == "active"}


# =========================
# SELF TEST ENGINE (enhanced)
# =========================

class SelfTestEngineEnhanced:
    """Forms its own tests, validates responses, retries if wrong.
    Self-adjusts thinking to answer better."""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.test_suite = []
        self.test_results = []
        self.adjustment_history = []
        self.confidence_scores = {}  # entity -> confidence in answers

    def form_test(self, entity, response):
        """Form a test to validate a response about an entity."""
        entity_lower = entity.lower()
        data = self.ke.entities.get(entity_lower, {})
        props = data.get("properties", {})
        attrs = data.get("attributes", {})

        tests = []
        # Test: does response contradict KB?
        for k, v in props.items():
            if isinstance(v, str) and v.lower() not in response.lower():
                # Check if the property is mentioned at all
                readable = k.replace("_", " ")
                if readable in response.lower():
                    tests.append({
                        "type": "property_check",
                        "property": k,
                        "expected": v,
                        "status": "pending",
                    })

        # Test: do boolean attributes match?
        for k, v in attrs.items():
            if isinstance(v, bool):
                verb = k.replace("is_", "").replace("has_", "").replace("_", " ")
                if verb in response.lower():
                    tests.append({
                        "type": "attribute_check",
                        "attribute": k,
                        "expected": v,
                        "status": "pending",
                    })

        self.test_suite.extend(tests)
        return tests

    def run_tests(self):
        """Run all pending tests and return results."""
        results = []
        for test in self.test_suite:
            if test["status"] != "pending":
                continue
            if test["type"] == "property_check":
                entity_data = self.ke.entities.get(test.get("entity", ""), {})
                props = entity_data.get("properties", {})
                actual = props.get(test["property"], None)
                test["status"] = "pass" if str(actual) == test["expected"] else "fail"
                test["actual"] = actual
            elif test["type"] == "attribute_check":
                test["status"] = "pass"  # simplified
            results.append(test)
        self.test_results.extend(results)
        return results

    def adjust_thinking(self, entity, test_results):
        """Adjust confidence and approach based on test results."""
        entity_lower = entity.lower()
        failures = [t for t in test_results if t.get("status") == "fail"]
        passes = [t for t in test_results if t.get("status") == "pass"]

        current_conf = self.confidence_scores.get(entity_lower, 0.5)
        if failures:
            new_conf = max(0.1, current_conf - 0.1 * len(failures))
        else:
            new_conf = min(1.0, current_conf + 0.05 * len(passes))

        self.confidence_scores[entity_lower] = new_conf
        self.adjustment_history.append({
            "entity": entity_lower,
            "old_conf": current_conf,
            "new_conf": new_conf,
            "failures": len(failures),
            "passes": len(passes),
            "timestamp": time.time(),
        })
        return new_conf

    def get_adjusted_approach(self, entity):
        """Get recommended approach based on past test results."""
        entity_lower = entity.lower()
        conf = self.confidence_scores.get(entity_lower, 0.5)
        if conf < 0.3:
            return "conservative"  # stick to KB facts only
        elif conf < 0.7:
            return "moderate"  # mix KB and composed
        return "confident"  # full composition allowed


# =========================
# THINKING PIPELINE
# =========================

class ThinkingPipeline:
    def __init__(self, knowledge_engine, memory, optimizer=None, inference_engine=None,
                 perspective_tracker=None, entity_states=None, keyword_linker=None,
                 perspective_mapper=None, behavior_tracker=None, decision_engine=None,
                 performance_logger=None, response_generator=None):
        self.knowledge_engine = knowledge_engine
        self.memory = memory
        # Smart arg handling: if optimizer is actually a ResponseGenerator, swap
        if optimizer is not None and type(optimizer).__name__ == 'ResponseGenerator':
            if response_generator is None:
                response_generator = optimizer
            optimizer = None
        self.optimizer = optimizer
        self.inference_engine = inference_engine
        self.perspective_tracker = perspective_tracker
        self.entity_states = entity_states or {}
        self.keyword_linker = keyword_linker
        self.perspective_mapper = perspective_mapper
        self.behavior_tracker = behavior_tracker
        self.decision_engine = decision_engine
        self.performance_logger = performance_logger
        self.response_generator = response_generator
        self.thinking_log = []
        self.simulation_results = []
        self.passes = 3
        # TaskPerformer for todo/goal commands (standalone or via PipelineIntegrator)
        self._task_performer_ref = TaskPerformer()
        # PipelineAnswerEngine: QA-pair composition with structural testing
        self.pipeline_answer = PipelineAnswerEngine(knowledge_engine) if knowledge_engine else None
        # NEW: Advanced composition systems
        if knowledge_engine:
            self.opposite_engine = OppositeEntityEngine(knowledge_engine)
            self.sentence_index = SentenceIndex(knowledge_engine)
            self.proportion_scorer = ProportionScorer(self.sentence_index, knowledge_engine)
            self.adj_selector = DynamicAdjVerbSelector(knowledge_engine)
            self.custom_composer = CustomComposer(
                knowledge_engine, self.sentence_index,
                self.proportion_scorer, self.adj_selector, self.opposite_engine
            )
            self.watcher = AgentSentenceWatcher(knowledge_engine, self.sentence_index)
            self.chain_builder = BackwardChainBuilder(
                knowledge_engine, self.sentence_index, self.proportion_scorer
            )
            # NEW: Cognitive architecture systems
            self.contextual_reasoner = ContextualReasoner(knowledge_engine, memory)
            self.fact_checker = FactChecker(knowledge_engine, self.sentence_index)
            self.proactive_advisor = ProactiveAdvisor(knowledge_engine, self.fact_checker, self.contextual_reasoner)
            self.health_tracker = HealthTracker(knowledge_engine)
            self.precognition = PrecognitionEngine(knowledge_engine, self.health_tracker, self.contextual_reasoner)
            self.goal_tracker = GoalTracker(knowledge_engine)
            self.behavioral_labeler = BehavioralLabeler(knowledge_engine)
            self.response_tree = ResponseTree(knowledge_engine)
            # NEW: Hierarchy and lifecycle systems
            self.noun_hierarchy = NounHierarchy()
            self.lifecycle_tracker = EntityLifecycleTracker(knowledge_engine)
            self.property_index = AutomaticPropertyIndex(knowledge_engine, self.noun_hierarchy)
            self.conv_state_tracker = ConversationStateTracker(knowledge_engine)
            self.dream_separator = DreamImaginationSeparator()
            self.temporal_engine = TemporalResponseEngine(knowledge_engine)
            self.fact_rebuttal = FactRebuttalEngine(knowledge_engine, self.noun_hierarchy)
            self.hierarchy_search = HierarchySearchEngine(self.noun_hierarchy)
            # NEW: Background analysis, hypotheticals, simulation, contradiction
            self.background_analyzer = BackgroundAnalyzer(knowledge_engine, self.noun_hierarchy)
            self.insight_crud = InsightCRUD(self.background_analyzer)
            self.hypothetical_engine = HypotheticalEngine(knowledge_engine, self.noun_hierarchy, self.property_index)
            self.answer_improver = AnswerImprover(knowledge_engine, self.noun_hierarchy, self.property_index)
            self.simulation_engine = SimulationEngine(knowledge_engine, self.noun_hierarchy, self.property_index, self.lifecycle_tracker)
            self.contradiction_detector = ContradictionDetector(knowledge_engine, self.noun_hierarchy)
            # NEW: Self-awareness and user tracking
            self.self_tracker = SelfAwarenessTracker()
            self.user_tracker = UserUsageTracker()
            self.self_test = SelfTestEngine(knowledge_engine, self.noun_hierarchy)
            # NEW: Possibility detection, elaboration, fact sorting, planning
            self.possibility_detector = PossibilityDetector(knowledge_engine)
            self.plan_builder = PlanBuilder(knowledge_engine)
            self.elaboration_engine = ElaborationEngine(knowledge_engine)
            self.fact_sorter = FactSorter(knowledge_engine)
            self.self_questioner = SelfQuestioner(knowledge_engine)
            self.response_separator = ResponseSeparator()
            self.parallel_stream = ParallelStream(knowledge_engine)
            self.self_test_enhanced = SelfTestEngineEnhanced(knowledge_engine)
            # Self-persistence & meta-cognition
            self.emotional_tracker = EmotionalStateTracker(knowledge_engine)
            self.preference_indexer = PreferenceIndexer(knowledge_engine)
            self.prediction_rotator = PredictionRotator(knowledge_engine)
            self.sentence_override = SentenceOverrideTracker()
            self.challenge_sync = ChallengeSync(knowledge_engine)
            self.user_predictor = UserPredictionEngine(knowledge_engine)
            self.preview_tester = PreviewTester(knowledge_engine)
            self.goal_review = GoalReviewSystem(knowledge_engine)
            self.subconscious = SubconsciousMemory(knowledge_engine)
            self.favorites = FavoritesIndex(knowledge_engine)
            self.auto_compare = AutoCompareContrast(knowledge_engine)
            self.response_reflector = ResponseReflector(
                knowledge_engine, self.emotional_tracker, self.preference_indexer,
                self.sentence_override, self.challenge_sync,
                self.user_predictor, self.preview_tester
            )
            # Run initial background analysis
            self.background_analyzer.run_full_analysis()
        else:
            self.opposite_engine = None
            self.sentence_index = None
            self.proportion_scorer = None
            self.adj_selector = None
            self.custom_composer = None
            self.watcher = None
            self.chain_builder = None
            self.contextual_reasoner = None
            self.fact_checker = None
            self.proactive_advisor = None
            self.health_tracker = None
            self.precognition = None
            self.goal_tracker = None
            self.behavioral_labeler = None
            self.noun_hierarchy = None
            self.lifecycle_tracker = None
            self.property_index = None
            self.conv_state_tracker = None
            self.dream_separator = None
            self.temporal_engine = None
            self.fact_rebuttal = None
            self.hierarchy_search = None
            self.background_analyzer = None
            self.insight_crud = None
            self.hypothetical_engine = None
            self.answer_improver = None
            self.simulation_engine = None
            self.contradiction_detector = None
            self.self_tracker = None
            self.user_tracker = None
            self.self_test = None
            self.response_tree = None
            self.possibility_detector = None
            self.plan_builder = None
            self.elaboration_engine = None
            self.fact_sorter = None
            self.self_questioner = None
            self.response_separator = None
            self.parallel_stream = None
            self.self_test_enhanced = None
            self.emotional_tracker = None
            self.preference_indexer = None
            self.prediction_rotator = None
            self.sentence_override = None
            self.challenge_sync = None
            self.user_predictor = None
            self.preview_tester = None
            self.goal_review = None
            self.subconscious = None
            self.favorites = None
            self.auto_compare = None
            self.response_reflector = None

    def process(self, query, num_responses=3):
        query_lower = query.lower().strip().rstrip("?")

        # Auto-index new words from user input
        if self.knowledge_engine:
            self.knowledge_engine.auto_index_input(query)

        # Check for greetings
        greeting = self._handle_greeting(query_lower)
        if greeting:
            return self._enrich_with_variations(greeting, query, num_responses)

        # SELF-AWARENESS: Answer questions about the AI itself
        if self.self_tracker:
            self_answer = self.self_tracker.answer_self_question(query)
            if self_answer:
                self.self_tracker.log_action("self_question", query_lower, details=self_answer[:100])
                self.self_tracker.increment_responses()
                results = [{"text": self_answer, "source": "self_awareness", "score": 0.95}]
                return self._enrich_with_variations(results, query, num_responses)

        # USER TRACKING: Log query and track patterns
        if self.user_tracker:
            entities_for_tracking = self._extract_entities(query)
            self.user_tracker.log_query(query, entities=entities_for_tracking)
            if self.self_tracker:
                self.self_tracker.log_action("query_received", query_lower)
                # Track entity focus
                for ent in entities_for_tracking:
                    self.self_tracker.log_action("entity_query", ent, details=query_lower[:80])
                    self.self_tracker.set_focus(ent)

        # State-aware event detection
        if self.entity_states is not None and self.perspective_mapper:
            state_response = self._detect_and_apply_state_events(query_lower, query)
            if state_response:
                return self._enrich_with_variations(state_response, query, num_responses)

        # Detect user intent for inference-based responses
        if self.inference_engine:
            intents = self.inference_engine.detect_intent(query)
            if intents:
                intent = intents[0]  # Take primary intent
                entities_intent = self._extract_entities(query)
                if entities_intent:
                    # Generate inference-based response
                    inferences = self.inference_engine.infer_from_context(
                        entities_intent[0], intent, self.memory.turns
                    )
                    if inferences:
                        # Build inference response — only from inferences, not chains
                        parts = []
                        for inf in inferences[:3]:
                            if inf["confidence"] > 0.6:
                                parts.append(inf["text"])

                        if parts:
                            # Update perspective
                            self.perspective_tracker.update_perspective(
                                entities_intent[0],
                                f"User wants {entities_intent[0]} to help with: {intent['groups'][0] if intent['groups'] else 'general task'}",
                                confidence=0.8,
                                source="conversation"
                            )
                            combined = " ".join(parts)
                            results = [
                                {"text": combined, "source": "inference", "score": 0.89},
                                {"text": f"Based on inference: {combined}",
                                 "source": "inference_variant", "score": 0.85},
                            ]
                            return self._enrich_with_variations(results, query, num_responses)

        entities = self._extract_entities(query)
        query_words = [w for w in query_lower.split() if w not in STOP_WORDS]

        # SELF-PERSISTENCE: Detect emotion, observe subconsciously, index preferences
        if entities:
            entity = entities[0]
            if self.emotional_tracker:
                emotion = self.emotional_tracker.detect_emotion(query)
                self.emotional_tracker.log_emotion(entity, emotion, trigger=query)
                # Auto-index preference from emotion
                if self.preference_indexer:
                    self.preference_indexer.auto_index_from_emotion(entity, "query_pattern", emotion)
            if self.subconscious:
                self.subconscious.observe(entity, query, source="user_query")

        # RECORD PREDICTIONS/PREPARATIONS — before any early returns so they're always captured
        if self.possibility_detector and entities:
            entity = entities[0]
            pred_patterns = [r"i predict", r"i think .+ will", r"i expect", r"mark my words"]
            for pat in pred_patterns:
                if re.search(pat, query_lower):
                    self.possibility_detector.record_prediction(entity, query)
                    break
            prep_patterns = [r"i prepared", r"i saved", r"the file i", r"i already have", r"already prepared"]
            for pat in prep_patterns:
                if re.search(pat, query_lower):
                    self.possibility_detector.record_preparation(entity, query, query)
                    break

        # Clarification: if no confident entity match but the query contains
        # a word ambiguously close to multiple KB entities, ask to clarify.
        if not entities:
            clarification = self._handle_ambiguous_entity(query_lower, query_words)
            if clarification:
                return self._enrich_with_variations(clarification, query, num_responses)

        # ASCII art requests
        art_answer = self._handle_ascii_art(query_lower, entities)
        if art_answer:
            return self._enrich_with_variations(art_answer, query, num_responses)

        # Check for yes/no questions — BEFORE PipelineAnswerEngine so they get direct answers
        yes_no_answer = self._handle_yes_no_question(query_lower, entities)
        if yes_no_answer:
            return self._enrich_with_variations(yes_no_answer, query, num_responses)

        # HIERARCHY & LIFECYCLE CHECKS: Death, resurrection, dreams, contradictions
        if entities and self.noun_hierarchy and self.lifecycle_tracker:
            for entity in entities:
                e = entity.lower()

                # Check for lifespan/fact contradictions FIRST (before death detection)
                contradictions = self.fact_rebuttal.detect_contradiction(query, e)
                if contradictions:
                    for c in contradictions:
                        rebuttal = self.fact_rebuttal.generate_rebuttal(e, c)
                        results = [{"text": rebuttal["text"], "source": "fact_rebuttal", "score": rebuttal["score"]}]
                        # Also record death in lifecycle tracker so resurrection can reference it
                        if c["type"] == "lifespan":
                            self.lifecycle_tracker.update_health(e, 0, cause="user reported (lifespan contradiction)")
                            self.conv_state_tracker.record_event(e, "death", {"query": query, "source": "fact_rebuttal"})
                        # Add cross-reference
                        xref = self.fact_rebuttal.cross_reference_with_data(e, query)
                        if xref:
                            findings = [f"{f['source']}: {f['key']} = {f['value']}" for f in xref[:3]]
                            results.append({"text": f"Cross-referencing: {'; '.join(findings)}",
                                            "source": "fact_rebuttal_xref", "score": 0.86})
                        return self._enrich_with_variations(results, query, num_responses)

                # Check for death mention
                death_entity = self.lifecycle_tracker.detect_death_mention(query, entities)
                if death_entity:
                    state = self.lifecycle_tracker.get_state(death_entity.lower())
                    self.lifecycle_tracker.update_health(death_entity.lower(), 0, cause="user reported")
                    self.conv_state_tracker.record_event(death_entity.lower(), "death", {"query": query})
                    emotional = self.lifecycle_tracker.generate_emotional_response(death_entity, "death")
                    fact_check = self.lifecycle_tracker.generate_fact_check_response(death_entity, "death")
                    care = self.lifecycle_tracker.suggest_care_actions(death_entity, state)
                    parts = [emotional]
                    if care:
                        parts.append("Some things you can do: " + " ".join(care[:2]))
                    results = [{"text": " ".join(parts), "source": "lifecycle_death", "score": 0.93}]
                    # Add fact-check variation
                    if fact_check:
                        results.append({"text": fact_check, "source": "lifecycle_fact_check", "score": 0.90})
                    # Add status report
                    status = self.conv_state_tracker.generate_status_report(death_entity.lower())
                    if status:
                        results.append({"text": status, "source": "lifecycle_status", "score": 0.88})
                    return self._enrich_with_variations(results, query, num_responses)

                # Check for resurrection claim — express skepticism regardless of prior state
                res_entity = self.lifecycle_tracker.detect_resurrection_claim(query, entities)
                if res_entity:
                    state = self.lifecycle_tracker.get_state(res_entity.lower())
                    emotional = self.lifecycle_tracker.generate_emotional_response(res_entity, "resurrected")
                    fact_check = self.lifecycle_tracker.generate_fact_check_response(res_entity, "resurrection")
                    parts = [emotional, fact_check]
                    # If we had recorded a death, reference it
                    if state["state"] == "dead":
                        death_time = self.conv_state_tracker.get_time_since_event(res_entity.lower(), "death")
                        if death_time:
                            hours = death_time / 3600
                            parts.append(f"Based on my records, your {res_entity.lower()} was reported dead "
                                         f"approximately {hours:.1f} hour(s) ago.")
                    # Check health trend
                    trend = self.conv_state_tracker.get_health_trend(res_entity.lower(), hours=1)
                    if trend:
                        parts.append(f"Health was {trend['start_health']}% an hour ago, now {trend['end_health']}%.")
                    # Always suggest monitoring
                    parts.append(f"I'd recommend closely monitoring your {res_entity.lower()} for the next few hours. "
                                 f"Watch for signs of movement, breathing, and response to stimuli. "
                                 f"If you notice anything unusual, let me know and I can help assess the situation.")
                    results = [{"text": " ".join(parts), "source": "lifecycle_resurrection", "score": 0.92}]
                    return self._enrich_with_variations(results, query, num_responses)

                # Check for dream/imagining
                dream_result = self.dream_separator.generate_dream_response(e, query)
                if dream_result:
                    # Apply dream effects to entity state
                    effects = dream_result.get("effects", {})
                    if effects.get("happiness_change", 0) > 0:
                        # Good dream -> small health boost
                        state = self.lifecycle_tracker.get_state(e)
                        self.lifecycle_tracker.update_health(e, min(100, state["health"] + effects["health_change"]),
                                                             cause="positive dream")
                    results = [{"text": dream_result["text"], "source": dream_result["source"],
                                "score": dream_result["score"]}]
                    return self._enrich_with_variations(results, query, num_responses)

                # Check for injury/sickness mention (skip if it's a prediction)
                injury_words = ["injured", "hurt", "sick", "wounded", "bleeding", "limping", "weak", "dying"]
                is_prediction = any(re.search(p, query_lower) for p in [r"i predict", r"i think .+ will", r"i expect", r"mark my words"])
                if any(w in query_lower for w in injury_words) and e in query_lower and not is_prediction:
                    state = self.lifecycle_tracker.get_state(e)
                    # Estimate health based on severity words
                    new_health = state["health"]
                    if "badly" in query_lower or "severely" in query_lower or "critical" in query_lower:
                        new_health = max(10, state["health"] - 50)
                    elif "a bit" in query_lower or "slightly" in query_lower:
                        new_health = max(30, state["health"] - 15)
                    else:
                        new_health = max(20, state["health"] - 30)
                    transition = self.lifecycle_tracker.update_health(e, new_health, cause=query)
                    self.conv_state_tracker.record_event(e, "injury", {"query": query, "health": new_health})
                    emotional = self.lifecycle_tracker.generate_emotional_response(e, "injured")
                    care = self.lifecycle_tracker.suggest_care_actions(e, state)
                    parts = [emotional]
                    if care:
                        parts.append("Here's what you can do: " + " ".join(care[:3]))
                    if transition.get("transition") == "critical":
                        parts.append(f"Your {e} is now in critical condition at {new_health}%. Please seek help immediately.")
                    elif transition.get("transition") == "injured":
                        parts.append(f"Your {e}'s health has dropped to {new_health}%.")
                    results = [{"text": " ".join(parts), "source": "lifecycle_injury", "score": 0.91}]
                    # Add property-based advice
                    props = self.property_index.get_full_properties(e)
                    if props.get("is_warm_blooded"):
                        body_temp = props.get("body_temp_f", 101.0)
                        parts.append(f"Normal body temperature for {e} is {body_temp}°F. Monitor for fever.")
                    return self._enrich_with_variations(results, query, num_responses)

                # Get hierarchy context for entity
                hier_ctx = self.hierarchy_search.get_entity_context(e)
                # Record current state
                self.conv_state_tracker.record_health(e, 100, context=query)

        # HYPOTHETICAL ENGINE: "what would happen if" questions
        if entities and self.hypothetical_engine:
            is_hypothetical = any(w in query_lower for w in ["what would happen", "what if", "what happens if", "what could happen"])
            if is_hypothetical:
                result = self.hypothetical_engine.analyze(query, entities)
                if result:
                    answer = self.hypothetical_engine.format_answer(result)
                    results = [{"text": answer, "source": "hypothetical", "score": 0.91}]
                    # Add simulation if available
                    if self.simulation_engine and entities:
                        sim_log = self.simulation_engine.simulate(entities[0], query, turns=4)
                        summary = self.simulation_engine.get_simulation_summary(sim_log)
                        if summary and summary.get("survived") is not None:
                            sim_text = (f"Simulation ({summary['turns']} turns): "
                                        f"Final health: {summary['final_health']}%, "
                                        f"Mood: {summary['final_mood']}, "
                                        f"Condition: {summary['final_condition']}. "
                                        f"{'Survived.' if summary['survived'] else 'Did not survive.'}")
                            results.append({"text": sim_text, "source": "simulation", "score": 0.87})
                    return self._enrich_with_variations(results, query, num_responses)

        # ANSWER IMPROVEMENT: "improve that" / "better answer" / "more detail"
        if self.answer_improver and hasattr(self, '_last_response'):
            if any(w in query_lower for w in ["improve", "better", "more detail", "elaborate", "expand"]):
                improved = self.answer_improver.improve(
                    self._last_response, query, entities, "comprehensive"
                )
                results = [{"text": improved, "source": "improved", "score": 0.90}]
                return self._enrich_with_variations(results, query, num_responses)

        # ELABORATION: "elaborate", "tell me more", "continue", "what else"
        if self.elaboration_engine and self.elaboration_engine.is_elaboration_request(query_lower):
            # Use last discussed entity if no entity in current query
            entity = None
            if entities:
                entity = entities[0]
            elif hasattr(self.memory, 'turns') and self.memory.turns:
                for turn in reversed(self.memory.turns[-5:]):
                    turn_entities = turn.get("entities", [])
                    if turn_entities:
                        entity = turn_entities[0]
                        break
            if entity:
                elab = self.elaboration_engine.get_elaboration(entity, query_words)
                if elab:
                    results = [{"text": elab, "source": "elaboration", "score": 0.88}]
                    return self._enrich_with_variations(results, query, num_responses)

        # POSSIBILITY DETECTION: "i told you so", "i expected this", "since i already prepared"
        if self.possibility_detector:
            if self.possibility_detector.detect_prediction_reference(query_lower):
                if entities:
                    entity = entities[0]
                    context = self.possibility_detector.get_context_for_entity(entity)
                    pred_results = []
                    if context["predictions"]:
                        for p in context["predictions"][-2:]:
                            pred_results.append(f"I recall you predicted: '{p['text']}'")
                    if pred_results:
                        results = [{"text": " ".join(pred_results), "source": "prediction_ack", "score": 0.92}]
                        return self._enrich_with_variations(results, query, num_responses)

            if self.possibility_detector.detect_preparation_reference(query_lower):
                if entities:
                    entity = entities[0]
                    context = self.possibility_detector.get_context_for_entity(entity)
                    if context["prepared_data"]:
                        prep = context["prepared_data"][-1]
                        results = [{"text": f"Using your prepared data on {prep['topic']}: {prep['data']}",
                                    "source": "prepared_data", "score": 0.93}]
                        return self._enrich_with_variations(results, query, num_responses)

        # PREDICTION CHECK: "is my prediction correct", "did my prediction come true"
        check_entities = entities
        if not check_entities and hasattr(self.memory, 'turns') and self.memory.turns:
            for turn in reversed(self.memory.turns[-5:]):
                turn_entities = turn.get("entities", [])
                if turn_entities:
                    check_entities = turn_entities
                    break
        if check_entities and self.possibility_detector and re.search(r"(?:is|was|are).+(?:prediction|guess|expectation).+(?:correct|right|true|accurate)|did.+(?:prediction|guess).+(?:come true|happen)|check.+(?:prediction|guess)", query_lower):
            entity = check_entities[0]
            context = self.possibility_detector.get_context_for_entity(entity)
            if context["predictions"]:
                pred = context["predictions"][-1]
                lifecycle_state = self.lifecycle_tracker.get_state(entity.lower()) if self.lifecycle_tracker else None
                parts = [f"You predicted: '{pred['text']}'"]
                if lifecycle_state:
                    if lifecycle_state.get("is_dead"):
                        parts.append(f"Currently, {entity} is deceased (health: {lifecycle_state.get('health', 0)}%).")
                    else:
                        health = lifecycle_state.get("health", 100)
                        parts.append(f"Currently, {entity} is alive with {health}% health.")
                kb_facts = self.knowledge_engine.get_entity_facts(entity) if hasattr(self.knowledge_engine, 'get_entity_facts') else []
                if kb_facts:
                    parts.append(f"Known facts: {'; '.join(kb_facts[:2])}")
                parts.append("Would you like me to monitor this prediction over time?")
                results = [{"text": " ".join(parts), "source": "prediction_check", "score": 0.90}]
                return self._enrich_with_variations(results, query, num_responses)

        # CONTRADICTION CHECK: "cats always land on their feet" type claims
        if entities and self.contradiction_detector:
            for entity in entities:
                contradictions = self.contradiction_detector.check_contradiction(query, entity)
                if contradictions:
                    counter = self.contradiction_detector.prepare_counter_examples(query, entity)
                    parts = []
                    for c in contradictions[:2]:
                        parts.append(f"Fact: {c['fact']}")
                    for ce in counter[:2]:
                        parts.append(f"Counter: {ce['counter_claim']}")
                    if parts:
                        results = [{"text": " | ".join(parts), "source": "contradiction_check", "score": 0.88}]
                        return self._enrich_with_variations(results, query, num_responses)

        # BACKGROUND ANALYSIS: entity profiling on queries
        if entities and self.background_analyzer:
            for entity in entities:
                profile = self.background_analyzer.get_profile(entity)
                if profile and not hasattr(self, '_profiled_' + entity.lower()):
                    setattr(self, '_profiled_' + entity.lower(), True)
                    # Auto-run analysis if gaps detected
                    if self.background_analyzer.gaps:
                        gap_count = len([g for g in self.background_analyzer.gaps if g["entity"] == entity.lower()])
                        if gap_count > 0:
                            # Silently index — don't interrupt the answer flow
                            pass

        # CHALLENGE DETECTION: "you're wrong", "that's not right", "are you sure"
        if self.challenge_sync and re.search(r"\byou(?:'re| are) (?:wrong|incorrect|mistaken)\b|that's not right|are you sure|that's (?:incorrect|wrong|false|not true)\b", query_lower):
            if entities:
                entity = entities[0]
                self.challenge_sync.record_challenge(entity, query)
                # Always respond to challenges with KB facts
                parts = [f"Let me cross-reference my information about {entity}:"]
                if self.knowledge_engine and entity.lower() in self.knowledge_engine.entities:
                    ent = self.knowledge_engine.entities[entity.lower()]
                    attrs = ent.get("attributes", {})
                    desc = ent.get("description", "")
                    if desc:
                        parts.append(f"  Description: {desc[:80]}")
                    for k, v in list(attrs.items())[:3]:
                        parts.append(f"  {k} = {v}")
                # Also check similar features from other entities
                similar = self.challenge_sync.sync_features(entity, "general", self.knowledge_engine)
                if similar:
                    parts.append("Related entities:")
                    for s in similar[:2]:
                        parts.append(f"  {s['entity']} has {s['feature']}={s['value']}")
                parts.append("Would you like me to re-check the facts?")
                results = [{"text": " ".join(parts), "source": "challenge_sync", "score": 0.88}]
                return self._enrich_with_variations(results, query, num_responses)

        # GOAL REVIEW: "review goals", "how are my goals", "goal status"
        if self.goal_review and re.search(r"\b(?:review|check|show|list|status).+(?:goals?|progress|accomplishments?)\b", query_lower):
            goals = self.goal_review.get_all_goals()
            if goals:
                parts = ["Goal review:"]
                for name, summary in goals.items():
                    parts.append(f"  {name}: {summary['status']} ({summary['progress']:.0%}) "
                                 f"[good={summary['good']} bad={summary['bad']} true={summary['true']} false={summary['false']}]")
                results = [{"text": "\n".join(parts), "source": "goal_review", "score": 0.90}]
            else:
                results = [{"text": "No goals tracked yet. Would you like to create one?", "source": "goal_review", "score": 0.85}]
            return self._enrich_with_variations(results, query, num_responses)

        # FAVORITES: "favorite", "add favorite", "what do you like about"
        if self.favorites and re.search(r"\b(?:add|my|set|list|show).+(?:favorite|favourite|like|prefer)s?\b|what do you like about\b", query_lower):
            if entities:
                entity = entities[0]
                if re.search(r"\badd\b|\bset\b", query_lower):
                    # Add current entity as favorite
                    self.favorites.add_favorite(entity, "general", reason="user requested")
                    results = [{"text": f"Added {entity} to favorites. I'll remember this.", "source": "favorites", "score": 0.90}]
                else:
                    favs = self.favorites.get_favorites(entity)
                    if favs:
                        parts = [f"My favorites about {entity}:"]
                        for feat, info in favs.items():
                            parts.append(f"  {feat}: {info.get('reason', 'no reason')}")
                        results = [{"text": "\n".join(parts), "source": "favorites", "score": 0.90}]
                    else:
                        results = [{"text": f"No favorites recorded for {entity} yet.", "source": "favorites", "score": 0.85}]
                return self._enrich_with_variations(results, query, num_responses)

        # AUTO-COMPARE: "compare cat and dog", "difference between"
        if self.auto_compare and re.search(r"\bcompare\b|\bdifference between\b|\bversus\b|\bvs\b", query_lower):
            if len(entities) >= 2:
                report = self.auto_compare.contrast_report(entities[0], entities[1])
                results = [{"text": report, "source": "auto_compare", "score": 0.90}]
                return self._enrich_with_variations(results, query, num_responses)

        # PRIMARY METHOD: PipelineAnswerEngine — QA pair composition with structural testing
        # Skip for advice/preparation/hypothetical — those go to cognitive reasoning
        skip_pa = any(w in query_lower for w in ["should i", "what should", "what do i", "how to", "how do", "bring", "pack", "take", "prepare", "what if", "what happens", "afraid", "fear", "injured", "hurt", "sick", "dying", "weak", "limping", "add todo", "create todo", "show todo", "check todo", "list todo", "set goal", "create goal", "run test", "start test", "run research", "start research", "agent status"])
        if self.pipeline_answer and not skip_pa:
            entity_name = entities[0] if entities else ""
            pa_result = self.pipeline_answer.answer(query, entity=entity_name, query_words=query_words)
            if pa_result and pa_result["score"] >= 0.3:
                if pa_result["score"] < 0.9:
                    debug_info = self.pipeline_answer.get_debug(query, entity=entity_name, query_words=query_words)
                    if debug_info["passes_90"]:
                        pa_result["score"] = debug_info["test_score"]
                        pa_result["method"] = "pipeline_answer_rebuilt"
                results = [
                    {"text": pa_result["text"], "source": "pipeline_answer", "score": pa_result["score"]},
                ]
                if self.response_generator and entity_name:
                    pa_debug = self.pipeline_answer.get_debug(query, entity=entity_name, query_words=query_words)
                    pa_candidates = [{"text": pa_result["text"], "score": 0.9}]
                    for q_tuple, a_tuple in pa_debug.get("top_qa", []):
                        if a_tuple and a_tuple.strip() != pa_result["text"][:80]:
                            pa_candidates.append({"text": a_tuple, "score": 0.7})
                    var = self.response_generator.generate_variations(
                        entity_name, query, pa_candidates
                    )
                    for v in var[:2]:
                        results.append(v)

                # Route through repeat detection — if PA text is too similar
                # to the last answer for this query, demote it below variations
                query_key = re.sub(r'\s+', ' ', query.lower().strip().rstrip("?"))
                last_answer = self.response_generator.last_answer_by_query.get(query_key) if self.response_generator else None
                if last_answer and results:
                    def _too_similar(a, b, threshold=0.75):
                        wa, wb = set(a.lower().split()), set(b.lower().split())
                        if not wa or not wb:
                            return False
                        return len(wa & wb) / max(len(wa | wb), 1) > threshold
                    if _too_similar(results[0]["text"], last_answer):
                        distinct = [r for r in results if not _too_similar(r["text"], last_answer)]
                        if distinct:
                            results = distinct + [r for r in results if r not in distinct]
                if self.response_generator and results:
                    self.response_generator.last_answer_by_query[query_key] = results[0]["text"]
                    self.response_generator.repeat_context.record(query, results[0]["text"])

                return self._enrich_with_variations(results, query, num_responses)

        # Check for context-aware queries about state changes
        context_answer = self._handle_context_query(query_lower, entities)
        if context_answer:
            return self._enrich_with_variations(context_answer, query, num_responses)

        # Check for learn/remember phrases
        if self.knowledge_engine.is_learn_phrase(query):
            return self._handle_learn(query, entities)

        # Check for fact correction (user statements that contradict dataset)
        fact_correction = self._handle_fact_correction(query_lower, entities)
        if fact_correction:
            return self._enrich_with_variations(fact_correction, query, num_responses)

        # Check for task/todo/goal commands
        task_answer = self._handle_task_command(query_lower, query)
        if task_answer:
            return self._enrich_with_variations(task_answer, query, num_responses)

        # Check for hypothetical questions
        hypothetical_answer = self._handle_hypothetical(query_lower, entities)
        if hypothetical_answer:
            return self._enrich_with_variations(hypothetical_answer, query, num_responses)

        # Check for conversation history queries
        history_answer = self._handle_conversation_history(query_lower)
        if history_answer:
            return self._enrich_with_variations(history_answer, query, num_responses)

        # Check for natural talk patterns ("tell me about", "what do you know about", etc.)
        natural_answer = self._handle_natural_talk(query_lower, entities, query_words)
        if natural_answer:
            return self._enrich_with_variations(natural_answer, query, num_responses)

        # Check for compound query (what is X and what is Y) — BEFORE custom compose
        compound = self.knowledge_engine.split_compound_query(query)
        if compound:
            resolved = []
            for c in compound:
                r = self.knowledge_engine.resolve_entity(c)
                resolved.append(r if r else c)
            answer = self.knowledge_engine.answer_compound(resolved)
            if answer:
                # Build diverse candidates from individual entities too
                results = [
                    {"text": answer, "source": "compound", "score": 0.87},
                    {"text": f"Regarding your compound query: {answer}",
                     "source": "compound_variant", "score": 0.83},
                ]
                # Add individual entity descriptions for variation material
                for entity in resolved:
                    if entity:
                        e_data = self.knowledge_engine.entities.get(entity.lower(), {})
                        for desc in e_data.get("descriptions", [])[:1]:
                            results.append({"text": desc, "source": "description", "score": 0.80})
                        for attr, val in list(e_data.get("attributes", {}).items())[:2]:
                            if isinstance(val, bool) and val:
                                formatted = self.knowledge_engine._format_attr_name(attr)
                                results.append({"text": f"A {entity.lower()} {formatted}.", "source": "attribute", "score": 0.78})
                return self._enrich_with_variations(results, query, num_responses)

        # COMPREHENSIVE REASONING: Use cognitive systems for complex queries
        if self.contextual_reasoner and entities and self.fact_checker:
            ctx = self.contextual_reasoner.build_context(query, entities)
            entity = entities[0]
            intention = ctx["intention"]

            # Generate proactive suggestions
            suggestions = self.proactive_advisor.generate_suggestions(ctx, entities) if self.proactive_advisor else []

            # Health check for entities
            health_info = {}
            if self.health_tracker:
                for e in entities:
                    health_info[e.lower()] = self.health_tracker.compute_health(e, query)

            # Precognition predictions
            predictions = []
            if self.precognition:
                for e in entities:
                    predictions.extend(self.precognition.predict(e, query, ctx))

            # Behavioral predictions
            behaviors = []
            if self.behavioral_labeler:
                for e in entities:
                    behaviors.extend(self.behavioral_labeler.predict_behavior(e, query))

            # Fact check key claims — only for direct factual queries, not advice/preparation
            fact_checks = []
            if intention not in ("advice", "preparation", "health"):
                for e in entities:
                    fc = self.fact_checker.check_fact(query, e)
                    if fc["verdict"] != "unknown":
                        fact_checks.append({"entity": e, **fc})

            # Build comprehensive response if we have enough data
            comprehensive_parts = []

            # Add fact-check verdict
            for fc in fact_checks:
                if fc["verdict"] == "true":
                    # For definition queries, use the dataset evidence as the answer
                    evidence = fc.get("evidence", [])
                    if evidence:
                        # Find the most relevant evidence (shortest that contains entity)
                        best = min([e for e in evidence if fc["entity"] in e.lower()],
                                   key=len, default=evidence[0])
                        comprehensive_parts.append(best.strip())
                    else:
                        comprehensive_parts.append(f"Yes, {fc['entity']} is supported by data ({fc['confidence']:.0%} confidence).")
                elif fc["verdict"] == "false":
                    comprehensive_parts.append(f"No, based on available data, {fc['entity']} does not match this claim ({fc['confidence']:.0%} confidence).")

            # Add health info
            for e, h in health_info.items():
                if h["status"] == "poor":
                    comprehensive_parts.append(f"Warning: {e}'s health is at {h['health']:.0f}% due to {', '.join(h['modifiers'][:2])}.")
                elif h["status"] == "moderate":
                    comprehensive_parts.append(f"{e.title()}'s health is {h['health']:.0f}%.")

            # Add predictions
            for p in predictions[:2]:
                if p["probability"] > 0.7:
                    comprehensive_parts.append(f"Prediction: {p['event']} ({p['probability']:.0%} likely). Action: {p['action_needed']}.")

            # Add suggestions
            for s in suggestions[:3]:
                comprehensive_parts.append(f"I suggest bringing {s['item']} — {s['reason']}.")

            # Add behavioral notes
            for b in behaviors[:1]:
                if b["behavior"] != "normal":
                    comprehensive_parts.append(f"{entity.title()} is likely to {b['behavior']} ({b['probability']:.0%} likely).")

            # Add follow-up tree
            if self.response_tree:
                tree = self.response_tree.build_tree(entity, query, " ".join(comprehensive_parts) if comprehensive_parts else "")
                followup = self.response_tree.get_followup_for_context(entity, query)
                comprehensive_parts.append(followup)

            if comprehensive_parts:
                combined = " ".join(comprehensive_parts)
                results = [
                    {"text": combined, "source": "cognitive_reasoning", "score": 0.92},
                ]
                # Add a simpler variation
                if len(comprehensive_parts) > 2:
                    simplified = " ".join(comprehensive_parts[:2] + comprehensive_parts[-1:])
                    results.append({"text": simplified, "source": "cognitive_simplified", "score": 0.88})
                return self._enrich_with_variations(results, query, num_responses)
            elif suggestions:
                # Even without comprehensive reasoning, return suggestions
                sug_text = ", ".join([f"{s['item']} ({s['reason']})" for s in suggestions[:4]])
                results = [{"text": f"For {', '.join(entities)}: {sug_text}.", "source": "cognitive_suggestions", "score": 0.87}]
                return self._enrich_with_variations(results, query, num_responses)

        # CUSTOM COMPOSITION: Use new systems for richer answers
        # Skip for compound, attribute, and multi-attribute queries
        is_compound = self.knowledge_engine.split_compound_query(query)
        is_attr_query = bool(re.search(r'(?:what|how|which)\s+(?:is|are|was|were)\s+the\s+\w+\s+of\s+\w+', query_lower))
        is_multi_attr = " and " in query_lower and any(w in query_lower for w in ["shape", "color", "size", "weight", "speed"])
        if self.custom_composer and entities and not is_compound and not is_attr_query and not is_multi_attr:
            entity = entities[0]
            custom_answer = self.custom_composer.compose_with_examples(entity, query_words)
            if custom_answer and len(custom_answer) > 40:
                results = [
                    {"text": custom_answer, "source": "custom_compose", "score": 0.90},
                ]
                # Add backward chain variation
                if self.chain_builder:
                    chain = self.chain_builder.build_chain(entity, query_words)
                    chain_answer = self.chain_builder.compose_from_chain(chain)
                    if chain_answer and chain_answer != custom_answer:
                        results.append({"text": chain_answer, "source": "backward_chain", "score": 0.86})
                # Add proportion-based variation
                if self.proportion_scorer:
                    stop = {"a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
                            "can", "could", "would", "should", "has", "have", "had", "what",
                            "how", "why", "when", "where", "who", "which", "that", "this",
                            "of", "in", "on", "at", "to", "for", "with", "by", "from",
                            "and", "or", "but", "not", "no", "yes", "so", "if", "it", "its",
                            "make", "makes", "made", "be", "being", "been", "compare", "between"}
                    claim_words = [w for w in query_words if w.lower() not in stop and w.lower() != entity.lower()]
                    prop = self.proportion_scorer.score_claim(entity.lower(), claim_words if claim_words else query_words)
                    if prop["confidence"] in ("high", "moderate"):
                        opp = self.opposite_engine.find_opposite(entity) if self.opposite_engine else None
                        if opp:
                            claim_text = " ".join(claim_words[:2]) if claim_words else "this trait"
                            results.append({
                                "text": f"Compared to {opp}, {entity} shows {prop['percentage']} data alignment for '{claim_text}'.",
                                "source": "proportion_compare", "score": 0.84,
                            })
                return self._enrich_with_variations(results, query, num_responses)

        # Detect multi-attribute query: "shape and color and size of diamond"
        multi_attr_answer = self._handle_multi_attribute_query(query_lower)
        if multi_attr_answer:
            return self._enrich_with_variations(multi_attr_answer, query, num_responses)

        # Detect "X of Y" shorthand: "diamond shape and color"
        of_pattern_answer = self._handle_of_pattern(query_lower, entities)
        if of_pattern_answer:
            return self._enrich_with_variations(of_pattern_answer, query, num_responses)

        # Check for attribute query (what is the X of Y)
        attr_answer = self.knowledge_engine.answer_attribute_query(query)
        if attr_answer:
            results = [
                {"text": attr_answer, "source": "attribute_query", "score": 0.91},
                {"text": f"Here's what I found: {attr_answer}",
                 "source": "attribute_query_variant", "score": 0.87},
            ]
            return self._enrich_with_variations(results, query, num_responses)

        # Multi-pass thinking — collect candidates from all entities + dataset
        all_candidates = []
        for entity in entities:
            if hasattr(self.optimizer, 'optimize_response'):
                candidates = self.optimizer.optimize_response(entity, query, num_responses * 2)
                if candidates:
                    all_candidates.extend(candidates)

        # Also search dataset directly with multiple keyword extraction
        dataset_sentences = self._extract_dataset_sentences(query_lower, query_words)
        for sent, score in dataset_sentences:
            all_candidates.append({
                "text": sent, "source": "dataset_extract", "score": min(0.9, score)
            })

        # If no entities found, try direct dataset search
        if not all_candidates:
            dataset_results = self.knowledge_engine.search_dataset_for_context(query, query_words)
            for q, a, score in dataset_results[:3]:
                all_candidates.append({
                    "text": a, "source": "dataset_direct", "score": min(0.8, score * 0.2)
                })

        if not all_candidates:
            return [{"text": "I don't have enough information to answer that question.", "source": "fallback", "score": 0.0}]

        # Score all candidates
        for c in all_candidates:
            c["score"] = self._final_score(c, query, entities, query_words)

        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Build composed answer from top candidates
        composed = self._compose_answer(query_lower, all_candidates[:6], entities, query_words)
        if composed:
            # Generate variations using ResponseGenerator
            if self.response_generator:
                conv_history = self.memory.turns if hasattr(self.memory, 'turns') else []
                context_clues = self.memory.context_clues if hasattr(self.memory, 'context_clues') else []
                variations = self.response_generator.generate_variations(
                    entities[0] if entities else query_lower,
                    query,
                    all_candidates[:6],
                    num_variations=num_responses,
                    context_clues=context_clues,
                    conv_history=conv_history,
                )
                if variations:
                    # Record used phrases
                    for v in variations:
                        self.response_generator.record_used(v["text"])
                    return variations
            return composed

        # Deduplicate fallback
        seen = set()
        final = []
        for c in all_candidates:
            normalized = c["text"].lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                final.append(c)
            if len(final) >= num_responses:
                break

        self.thinking_log.append({
            "query": query,
            "entities": entities,
            "num_candidates": len(all_candidates),
            "num_selected": len(final),
            "top_score": final[0]["score"] if final else 0
        })

        # Add 'but' clause and follow-up prompts to top result
        if final and entities:
            top = final[0]
            entity = entities[0]
            but_clause = self._build_but_clause(entity, query_lower, top["text"], [])
            if but_clause:
                top["text"] = top["text"].rstrip(".") + but_clause
            followups = self._generate_followup_prompts(entity, query_lower, final)
            if followups:
                top["followup_prompts"] = followups

        # Store last response for answer improvement
        if final:
            self._last_response = final[0]["text"]
            # Record in conversation memory so follow-up queries (elaborate, etc.) can reference entities
            self.memory.add_turn(query, final[0]["text"], entities=entities if entities else [])
            # Log the action
            if self.self_tracker:
                source = final[0].get("source", "unknown")
                self.self_tracker.log_action("response_generated", query_lower,
                                             details=f"source={source}, score={final[0].get('score', 0):.3f}")
                self.self_tracker.increment_responses()
            # Track user preference
            if self.user_tracker and final:
                self.user_tracker.log_query(query, entities=entities,
                                           response_source=final[0].get("source"))
            # Consistency check
            if self.self_test:
                check = self.self_test.check_consistency(query, final[0]["text"])
                if not check["consistent"]:
                    self.self_tracker.log_decision("inconsistency_detected",
                                                    reason=f"similarity={check['similarity']:.2f}",
                                                    outcome="flagged") if self.self_tracker else None

        # SELF-PERSISTENCE: Reflection now happens inside _enrich_with_variations

        return final

    def _handle_greeting(self, query_lower):
        greetings = {
            "hello": ["Hello! How can I help you today?", "Hi there! What would you like to know?", "Hey! How can I assist you?"],
            "hi": ["Hi there! What would you like to know?", "Hey! What's on your mind?", "Hello! What can I help with?"],
            "hey": ["Hey! What's on your mind?", "Hi there! How can I help?", "Hey! What would you like to talk about?"],
            "good morning": ["Good morning! What can I help you with?", "Good morning! How may I assist you today?"],
            "good afternoon": ["Good afternoon! What would you like to know?", "Good afternoon! How can I help?"],
            "good evening": ["Good evening! How can I assist you?", "Good evening! What can I help with?"],
            "how are you": ["I'm doing well, thanks for asking! What can I help you with?", "I'm great! What would you like to talk about?"],
            "whats up": ["Not much! What would you like to talk about?", "Hey! What's on your mind?"],
            "what's up": ["Not much! What would you like to talk about?", "Hey! What's on your mind?"],
            "yo": ["Yo! What's up?", "Hey! What's going on?"],
            "sup": ["Hey! What's going on?", "Not much! What would you like to talk about?"],
            "howdy": ["Howdy! What can I do for you?", "Howdy! How may I help?"],
            "greetings": ["Greetings! How may I help you?", "Greetings! What can I assist with?"],
            "hello there": ["Hello there! What brings you here?", "Hello there! How can I help?"],
            "hi there": ["Hi there! What can I help you with?", "Hi there! How may I assist?"],
        }
        if query_lower in greetings:
            options = greetings[query_lower]
            return [{"text": g, "source": "greeting", "score": 1.0 - i * 0.05} for i, g in enumerate(options)]
        return None

    def _handle_ambiguous_entity(self, query_lower, query_words):
        """Detect ambiguous entity references and ask the user to clarify
        instead of guessing which one was meant."""
        content_words = [w for w in query_words if w not in STOP_WORDS and len(w) >= 5]
        for w in content_words:
            candidates = self.knowledge_engine.find_ambiguous_candidates(w)
            if candidates:
                options = ", ".join(c.title() for c in candidates[:4])
                return [{
                    "text": f"I want to make sure I answer the right thing — did you mean {options}?",
                    "source": "clarification",
                    "score": 0.7,
                }]
        return None

    ASCII_ART_LIBRARY = {
        "cat": " /\\_/\\\n( o.o )\n > ^ <",
        "dog": " / \\__\n(    @\\___\n /         O\n/   (_____/\n/_____/   U",
        "fish": "   ,`.\n __)_\\__\n'-.___.-'",
        "bird": "   ,,\n (o o)\n(  V  )",
        "diamond": "   /\\\n  /  \\\n <    >\n  \\  /\n   \\/",
        "generic": "  .---.\n /     \\\n|  ???  |\n \\     /\n  '---'",
    }

    def _handle_ascii_art(self, query_lower, entities):
        """Generate simple ASCII art for a recognized entity when explicitly
        requested (e.g. 'draw a cat', 'ascii art of a diamond')."""
        wants_art = any(kw in query_lower for kw in
            ("draw", "ascii art", "sketch", "picture of", "show me a picture"))
        if not wants_art:
            return None

        entity = entities[0].lower() if entities else None
        if not entity:
            for key in self.ASCII_ART_LIBRARY:
                if key != "generic" and key in query_lower:
                    entity = key
                    break

        art = self.ASCII_ART_LIBRARY.get(entity, self.ASCII_ART_LIBRARY["generic"])
        label = entity.title() if entity else "that"
        text = f"Here's a quick ASCII sketch of {label}:\n```\n{art}\n```"
        return [{"text": text, "source": "ascii_art", "score": 0.8}]

    def _handle_multi_attribute_query(self, query_lower):
        # Detect patterns like "shape and color and size of diamond"
        # or "shape, color, size of a diamond"
        m = re.match(
            r"([\w\s,]+?)\s+(?:of|for|about)\s+(?:a |an |the )?([\w\s]+?)$",
            query_lower
        )
        if m:
            attr_part = m.group(1).strip()
            entity_part = m.group(2).strip()

            # Check if attr_part contains multiple attribute keywords
            attr_keywords = [w.strip() for w in re.split(r'\s+and\s+|,\s*|\s+', attr_part) if w.strip()]
            if len(attr_keywords) >= 2:
                # Filter out common query verbs and question words
                query_verbs = {"tell", "me", "what", "how", "about", "describe", "give", "know",
                              "want", "need", "like", "show", "find", "list", "name", "get"}
                attr_like = [w for w in attr_keywords if len(w) > 1 and w not in STOP_WORDS and w not in query_verbs]
                if len(attr_like) >= 2:
                    entity = self.knowledge_engine.resolve_entity(entity_part)
                    if entity:
                        return self._build_multi_attr_answer(entity, attr_like)

        return None

    def _handle_of_pattern(self, query_lower, entities):
        # Detect "diamond shape and color" (shorthand for "what is the shape and color of a diamond")
        m = re.match(r"([\w]+)\s+([\w\s,]+)$", query_lower)
        if m:
            possible_entity = m.group(1).strip()
            possible_attrs = m.group(2).strip()
            entity = self.knowledge_engine.resolve_entity(possible_entity)
            if entity:
                # Check if the remaining words are attribute-related
                attr_words = [w.strip() for w in re.split(r'\s+and\s+|,\s*|\s+', possible_attrs) if w.strip()]
                attr_words = [w for w in attr_words if w not in STOP_WORDS]
                if len(attr_words) >= 1:
                    return self._build_multi_attr_answer(entity, attr_words)
        return None

    def _build_multi_attr_answer(self, entity, attr_words):
        entity_lower = entity.lower()
        data = self.knowledge_engine.entities.get(entity_lower, {})
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        descs = data.get("descriptions", [])

        sentences = []

        for aw in attr_words:
            aw_lower = aw.lower().rstrip("?")
            found = False

            # Search in properties FIRST (more specific: "color", "weight", "speed")
            for p, v in props.items():
                readable = p.replace("_", " ")
                if aw_lower == p or aw_lower == readable or aw_lower in p or aw_lower in readable:
                    sentences.append(f"The {readable} of a {entity_lower} is {v}.")
                    found = True
                    break

            # Search in attributes (boolean: is_X, has_X)
            if not found:
                best_match = None
                best_score = 0
                for a, v in attrs.items():
                    readable = self.knowledge_engine._format_attr_name(a)
                    score = 0
                    if aw_lower == a or aw_lower == readable:
                        score = 3
                    elif aw_lower in a or aw_lower in readable:
                        score = 2
                    elif a in aw_lower or readable in aw_lower:
                        score = 1
                    if score > best_score:
                        best_score = score
                        best_match = (a, v, readable)
                if best_match:
                    a, v, readable = best_match
                    readable = self.knowledge_engine._format_attr_name(a)
                    if isinstance(v, bool):
                        val_text = "yes" if v else "no"
                    else:
                        val_text = v
                    sentences.append(f"The {readable} of a {entity_lower} is {val_text}.")
                    found = True
            # Search in properties
            if not found:
                for p, v in props.items():
                    readable = p.replace("_", " ")
                    if aw_lower in p or p in aw_lower or aw_lower in readable:
                        sentences.append(f"The {readable} of a {entity_lower} is {v}.")
                        found = True
                        break
            # Search in dataset QA — require entity AND attribute word in question, skip short/common words
            if not found and len(aw_lower) > 3 and aw_lower not in STOP_WORDS:
                for q, a in self.knowledge_engine.dataset_qa:
                    q_lower = q.lower()
                    if entity_lower in q_lower and aw_lower in q_lower:
                        sentences.append(a)
                        found = True
                        break
            # Search in descriptions for keyword
            if not found:
                for desc in descs:
                    if aw_lower in desc.lower():
                        sentences.append(desc)
                        found = True
                        break

        if not sentences:
            return None

        # Compose into multiple variations
        results = []

        # Variation 1: Combined response
        if len(sentences) == 1:
            results.append({"text": sentences[0], "source": "multi_attr", "score": 0.91})
        else:
            combined = ". ".join(dict.fromkeys(sentences)) + "."
            results.append({"text": combined, "source": "multi_attr", "score": 0.90})

        # Variation 2: Individual facts listed
        if len(sentences) > 1:
            listed = "\n".join(f"- {s}" for s in sentences[:4])
            results.append({"text": listed, "source": "multi_attr_list", "score": 0.86})

        # Variation 3: Summary style
        if len(sentences) > 1:
            summary_parts = []
            for s in sentences[:3]:
                # Extract just the value part
                if " is " in s:
                    val = s.split(" is ", 1)[-1].rstrip(".")
                    summary_parts.append(val)
            if summary_parts:
                results.append({"text": f"Regarding {entity}: {', '.join(summary_parts)}.",
                               "source": "multi_attr_summary", "score": 0.83})

        return results

    def _extract_dataset_sentences(self, query_lower, query_words):
        """Extract multiple relevant sentences from dataset QA pairs."""
        results = []
        if not self.knowledge_engine.dataset_qa:
            return results

        # Extract entity keywords from query (nouns, not stop words)
        entity_keywords = [w for w in query_words if w not in STOP_WORDS and len(w) > 2]

        # Pre-compile word-boundary patterns for entity keywords
        kw_patterns = [(kw, re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)) for kw in entity_keywords]

        for q, a in self.knowledge_engine.dataset_qa:
            # Split answer into individual sentences
            sentences = split_sentences(a, min_len=10)

            for sent in sentences:
                score = 0.0
                # Must mention at least one entity keyword in the sentence itself (word-boundary)
                entity_hits = sum(1 for kw, pat in kw_patterns if pat.search(sent))
                if entity_hits == 0:
                    continue
                # Score based on entity keyword matches in sentence
                score = entity_hits * 0.3
                # Bonus for question context match (word-boundary)
                for kw, pat in kw_patterns:
                    if pat.search(q):
                        score += 0.15
                if score > 0.2:
                    results.append((sent.strip().rstrip(".") + ".", min(0.95, score)))

        results.sort(key=lambda x: x[1], reverse=True)
        # Deduplicate similar sentences
        seen = set()
        unique = []
        for sent, score in results:
            normalized = sent.lower()[:50]
            if normalized not in seen:
                seen.add(normalized)
                unique.append((sent, score))
        return unique[:8]

    def _compose_answer(self, query_lower, candidates, entities, query_words):
        """Combine top candidates into a composed multi-sentence answer with cross-references."""
        if len(candidates) < 2:
            return None

        # Get conversation context for answer shaping
        detail_level = self.memory.get_detail_level()
        entity_contexts = {}
        for e in (entities or []):
            entity_contexts[e.lower()] = self.memory.get_entity_context(e)

        # Group by source type
        descriptions = [c for c in candidates if c["source"] == "description"]
        datasets = [c for c in candidates if c["source"] in ("dataset", "dataset_extract", "dataset_direct")]
        attributes = [c for c in candidates if c["source"] == "attribute"]
        comparisons = [c for c in candidates if c["source"] == "comparison"]
        properties = [c for c in candidates if c["source"] == "property"]

        parts = []

        # Start with the best description — only if it mentions the entity
        entity_names = [e.lower() for e in entities] if entities else []
        if descriptions:
            best_desc = descriptions[0]["text"]
            desc_lower = best_desc.lower()
            if not entity_names or any(en in desc_lower for en in entity_names):
                parts.append(best_desc)
            elif query_words:
                desc_words = set(re.findall(r'\w+', desc_lower))
                if len(set(query_words) & desc_words) >= 2:
                    parts.append(best_desc)

        # Add relevant dataset sentences — only if they mention the entity
        if datasets:
            for ds in datasets[:3]:
                text = ds["text"]
                text_lower = text.lower()
                relevant = False
                for en in entity_names:
                    if en in text_lower:
                        relevant = True
                        break
                if not relevant and not entity_names and query_words:
                    query_set = set(query_words)
                    text_set = set(text_lower.split())
                    overlap = len(query_set & text_set)
                    if overlap >= 2:
                        relevant = True
                if relevant and text not in parts:
                    if not any(self._texts_overlap(text, p) for p in parts):
                        parts.append(text)

        # Add attribute facts that aren't redundant
        if attributes:
            for attr in attributes[:2]:
                text = attr["text"]
                if not any(self._texts_overlap(text, p) for p in parts):
                    parts.append(text)

        # Add property facts
        if properties:
            for prop in properties[:1]:
                text = prop["text"]
                if not any(self._texts_overlap(text, p) for p in parts):
                    parts.append(text)

        # Cross-reference: find related QA pairs from other topics
        if entities and query_words:
            for entity in entities:
                related_qa = self.knowledge_engine.get_related_qa_pairs(entity, query_words, max_results=3)
                for q, a, score in related_qa:
                    if score > 0.4:
                        sentences = split_sentences(a, min_len=15)
                        for sent in sentences[:1]:
                            sent = sent.strip().rstrip(".") + "."
                            if not any(self._texts_overlap(sent, p) for p in parts):
                                # Only add if it explicitly mentions the queried entity
                                if entity.lower() in sent.lower():
                                    parts.append(sent)
                                    break

        # Context-aware state: check for attribute state changes
        if entities and self.memory.context_clues:
            for entity in entities:
                state = self.memory.get_entity_state(entity)
                if state:
                    for attr, info in state.items():
                        original = info.get("original", "unknown")
                        current = info.get("current", "unknown")
                        if original != "unknown":
                            parts.append(f"The {attr} of this {entity} was originally {original}, but is now {current}.")

        # Temporal/historical context
        if query_words:
            temporal_kw = ["year", "years", "ago", "history", "old", "ancient", "first", "lived", "always"]
            if any(kw in " ".join(query_words) for kw in temporal_kw):
                temporal_results = self.knowledge_engine.search_temporal(query_words, max_results=2)
                for q, a, score in temporal_results:
                    sentences = split_sentences(a, min_len=15)
                    for sent in sentences[:1]:
                        sent = sent.strip().rstrip(".") + "."
                        if not any(self._texts_overlap(sent, p) for p in parts):
                            parts.append(sent)
                            break

        # Inference chains: practical suggestions, limitations, obtainability
        if self.inference_engine and entities:
            for entity in entities:
                # Chain facts for this entity in context of the query
                chains = self.inference_engine.chain_facts(entity, query_lower)
                for ch in chains:
                    if ch["confidence"] > 0.7:
                        text = ch["text"]
                        if not any(self._texts_overlap(text, p) for p in parts):
                            if ch["type"] == "context_suggestion":
                                # Most relevant — adds contextual value
                                parts.append(text)
                            elif ch["type"] in ("suggestion", "obtainability") and len(parts) < 5:
                                parts.append(text)
                            elif ch["type"] == "limitation" and detail_level == "detailed":
                                parts.append(text)

                # Update perspective tracker with what we learned
                self.perspective_tracker.update_perspective(
                    entity,
                    f"Discussed {entity} in context of: {query_lower[:60]}",
                    confidence=0.7,
                    source="composed_answer"
                )

        # Cross-reference reasoning: compare true/false attributes across related entities
        if entities and detail_level != "short":
            cross_ref_parts = self._cross_reference_reasoning(entities, query_words)
            for cr_text in cross_ref_parts:
                if not any(self._texts_overlap(cr_text, p) for p in parts):
                    parts.append(cr_text)

        if not parts:
            return None

        # Shape answer based on detail level
        if detail_level == "short":
            parts = parts[:1]
        elif detail_level == "detailed":
            pass  # keep all parts
        else:
            parts = parts[:4]

        combined = " ".join(parts)

        # Add comparison if detailed and available
        if comparisons and detail_level != "short":
            comp_text = comparisons[0]["text"]
            if not self._texts_overlap(comp_text, combined):
                combined += " Additionally, " + comp_text

        return [{"text": combined, "source": "composed", "score": candidates[0]["score"] + 0.02}]

    def _cross_reference_reasoning(self, entities, query_words):
        """Compare true/false attributes across related entities to form composite sentences.
        E.g., 'a cat has a tail but no shell like a turtle' or
        'a cat and dog are both mammals but a cat has retractable claws while a dog does not'."""
        parts = []
        if not entities:
            return parts

        for entity in entities[:2]:
            entity_lower = entity.lower()
            entity_data = self.knowledge_engine.entities.get(entity_lower, {})
            if not entity_data:
                continue

            attrs = entity_data.get("attributes", {})
            props = entity_data.get("properties", {})
            similar = entity_data.get("similar_to", [])
            category = entity_data.get("category", "")

            if not similar:
                continue

            # Find a related entity that's actually in our KB
            for sim_name in similar[:3]:
                sim_lower = sim_name.lower()
                sim_data = self.knowledge_engine.entities.get(sim_lower, {})
                if not sim_data:
                    continue

                sim_attrs = sim_data.get("attributes", {})
                sim_props = sim_data.get("properties", {})

                # Find attributes that differ (one has True, other has False)
                shared_true = []
                entity_only_true = []
                sim_only_true = []

                all_attr_keys = set(list(attrs.keys()) + list(sim_attrs.keys()))
                for ak in all_attr_keys:
                    e_val = attrs.get(ak, None)
                    s_val = sim_attrs.get(ak, None)
                    if e_val is True and s_val is True:
                        shared_true.append(ak)
                    elif e_val is True and s_val is False:
                        entity_only_true.append(ak)
                    elif e_val is False and s_val is True:
                        sim_only_true.append(ak)

                # Build comparison sentence
                # "A cat has X but no Y like a turtle"
                if entity_only_true or sim_only_true:
                    has_parts = []
                    no_parts = []

                    # Attributes this entity has but similar doesn't
                    for ak in entity_only_true[:2]:
                        pretty = ak.replace("has_", "").replace("is_", "").replace("_", " ")
                        has_parts.append(pretty)

                    # Attributes similar entity has but this one doesn't
                    for ak in sim_only_true[:2]:
                        pretty = ak.replace("has_", "").replace("is_", "").replace("_", " ")
                        no_parts.append(pretty)

                    if has_parts and no_parts:
                        has_str = ", ".join(has_parts)
                        no_str = ", ".join(no_parts)
                        parts.append(
                            f"A {entity_lower} has {has_str} but no {no_str} like a {sim_lower}."
                        )
                    elif has_parts:
                        has_str = ", ".join(has_parts)
                        parts.append(
                            f"A {entity_lower} has {has_str} while a {sim_lower} does not."
                        )
                    elif no_parts:
                        no_str = ", ".join(no_parts)
                        parts.append(
                            f"A {sim_lower} has {no_str} but a {entity_lower} does not."
                        )

                # Shared attributes (both have)
                if shared_true and len(shared_true) >= 2:
                    # Format shared attributes as natural phrases
                    shared_phrases = []
                    for ak in shared_true[:3]:
                        if ak.startswith("has_"):
                            word = ak[4:].replace("_", " ")
                            shared_phrases.append(f"have {word}")
                        elif ak.startswith("is_"):
                            word = ak[3:].replace("_", " ")
                            shared_phrases.append(f"are {word}")
                        elif ak.startswith("lays_"):
                            word = ak[5:].replace("_", " ")
                            shared_phrases.append(f"lay {word}")
                        else:
                            shared_phrases.append(ak.replace("_", " "))
                    if len(shared_phrases) == 1:
                        shared_str = shared_phrases[0]
                    else:
                        shared_str = ", ".join(shared_phrases[:-1]) + " and " + shared_phrases[-1]
                    parts.append(
                        f"Both {entity_lower} and {sim_lower} {shared_str}."
                    )

                # Differences summary (what one has, other doesn't)
                if entity_only_true or sim_only_true:
                    diff_parts = []
                    for ak in entity_only_true[:2]:
                        if ak.startswith("has_"):
                            word = ak[4:].replace("_", " ")
                            diff_parts.append(f"{entity_lower} has {word}")
                        elif ak.startswith("is_"):
                            word = ak[3:].replace("_", " ")
                            diff_parts.append(f"{entity_lower} is {word}")
                        else:
                            diff_parts.append(f"{entity_lower} has {ak.replace('_', ' ')}")
                    for ak in sim_only_true[:2]:
                        if ak.startswith("has_"):
                            word = ak[4:].replace("_", " ")
                            diff_parts.append(f"{sim_lower} has {word}")
                        elif ak.startswith("is_"):
                            word = ak[3:].replace("_", " ")
                            diff_parts.append(f"{sim_lower} is {word}")
                        else:
                            diff_parts.append(f"{sim_lower} has {ak.replace('_', ' ')}")
                    if diff_parts:
                        parts.append(
                            " and ".join(diff_parts[:2]) + "."
                        )

                # Property comparison (e.g., weight, speed)
                prop_keys = set(list(props.keys()) + list(sim_props.keys()))
                compared_props = 0
                for pk in prop_keys:
                    if compared_props >= 1:
                        break
                    e_prop = props.get(pk)
                    s_prop = sim_props.get(pk)
                    if e_prop and s_prop and str(e_prop) != str(s_prop):
                        pretty_pk = pk.replace("_", " ")
                        parts.append(
                            f"The {pretty_pk} of a {entity_lower} is {e_prop} while a {sim_lower} is {s_prop}."
                        )
                        compared_props += 1

                break  # only compare with the first related entity found

        return parts[:3]

    def _build_but_clause(self, entity, query_lower, main_answer, kb_facts):
        """Build a 'but' clause that continues the topic logically and is continuable."""
        if not entity or not main_answer:
            return None
        entity_lower = entity.lower()
        but_parts = []

        # Get entity-specific facts for the 'but' clause
        data = self.knowledge_engine.entities.get(entity_lower, {})
        props = data.get("properties", {})
        attrs = data.get("attributes", {})
        descs = data.get("descriptions", [])

        # Check for limitations or interesting alternatives
        limitations = []
        alternatives = []
        for attr, val in attrs.items():
            if isinstance(val, bool) and not val:
                word = attr.replace("is_", "").replace("has_", "").replace("_", " ")
                limitations.append(f"it doesn't {word}" if not word.startswith("is") else f"it isn't {word}")
            elif isinstance(val, bool) and val:
                word = attr.replace("is_", "").replace("has_", "").replace("_", " ")
                alternatives.append(f"it can {word}")

        # Check for speed/physical properties that add context
        for pkey, pval in props.items():
            pkey_lower = pkey.lower()
            if any(kw in pkey_lower for kw in ("speed", "weight", "size", "lifespan")):
                but_parts.append(f"its {pkey_lower.replace('_', ' ')} is {pval}")

        # Build the 'but' clause
        if limitations:
            lim = random.choice(limitations)
            but_parts.append(f"but {lim}")
        elif alternatives:
            alt = random.choice(alternatives)
            but_parts.append(f"however, {alt}")

        if but_parts:
            clause = random.choice(but_parts)
            # Make it continuable - end with something that invites follow-up
            continuables = [
                f". This means there are other interesting aspects to explore",
                f". There's more to learn about this topic",
                f". Would you like to know more about its capabilities?",
                f". What else would you like to know?",
            ]
            return f" {clause.capitalize()}{random.choice(continuables)}"
        return None

    def _generate_followup_prompts(self, entity, query_lower, results):
        """Generate follow-up prompts after answering to guide conversation."""
        if not entity or not results:
            return []
        entity_lower = entity.lower()
        data = self.knowledge_engine.entities.get(entity_lower, {})
        props = data.get("properties", {})
        attrs = data.get("attributes", {})
        descs = data.get("descriptions", [])

        prompts = []
        # Suggest related topics based on what wasn't covered
        covered_words = set()
        for r in results:
            text = r.get("text", "").lower()
            covered_words.update(re.findall(r'\w{4,}', text))

        # Suggest uncovered properties
        for pkey in props:
            pword = pkey.lower().replace("_", " ")
            if not any(w in covered_words for w in pword.split()):
                prompts.append(f"what about its {pword}?")

        # Suggest uncovered attributes
        for akey in attrs:
            aword = akey.lower().replace("_", " ")
            if not any(w in covered_words for w in aword.split()):
                prompts.append(f"tell me about its {aword}")

        # Suggest comparison with similar entities
        if self.knowledge_engine.category_members:
            cat = data.get("category", "")
            if cat:
                members = list(self.knowledge_engine.category_members.get(cat, set()))
                for m in members:
                    if m != entity_lower and m not in covered_words:
                        prompts.append(f"how does it compare to {m}?")
                        break

        random.shuffle(prompts)
        return prompts[:3]

    def _fact_check_before_extending(self, text, entity):
        """Verify KB facts before appending to response."""
        if not entity or not text:
            return text
        entity_lower = entity.lower()
        data = self.knowledge_engine.entities.get(entity_lower, {})
        props = data.get("properties", {})
        attrs = data.get("attributes", {})

        # Check if any claimed facts are contradicted by KB
        text_lower = text.lower()
        issues = []
        for pkey, pval in props.items():
            pword = pkey.lower().replace("_", " ")
            # If text claims a property value that differs from KB
            if pword in text_lower:
                # Just flag it - don't remove, but note it
                pass

        return text

    def _goal_check_during_composition(self, query_lower, current_text, entity):
        """Verify answer stays on-topic while building sentences."""
        if not entity or not current_text:
            return True
        entity_lower = entity.lower()
        query_words = set(re.findall(r'\w{4,}', query_lower))
        text_words = set(re.findall(r'\w{4,}', current_text.lower()))

        # Check if the text still relates to the query topic
        overlap = len(query_words & text_words)
        if overlap == 0 and len(query_words) > 2:
            return False  # Drifted too far from topic
        return True

    def _enrich_with_variations(self, existing_results, query, num_responses=3):
        """Enrich existing results with additional variations from ResponseGenerator."""
        if not existing_results:
            return existing_results

        # Record turn in conversation memory so follow-ups (elaborate, etc.) can reference entities
        entities = self._extract_entities(query) if hasattr(self, '_extract_entities') else []
        self.memory.add_turn(query, existing_results[0]["text"], entities=entities)

        # FACT SORTING: Rank responses by KB support level
        if self.fact_sorter:
            sort_entities = self._extract_entities(query) if hasattr(self, '_extract_entities') else []
            if sort_entities:
                entity = sort_entities[0]
                for r in existing_results:
                    sort_result = self.fact_sorter.sort_facts(entity, [r["text"]])
                    if sort_result:
                        r["support"] = sort_result[0]["support"]
                        r["fact_source"] = sort_result[0]["source"]

        # Attach a follow-up prompt from ResponseTree so it's not only used
        # inside the cognitive_reasoning branch.
        if self.response_tree and existing_results:
            entities = self._extract_entities(query)
            if entities:
                entity = entities[0]
                if entity.lower() in self.knowledge_engine.entities:
                    top_text = existing_results[0].get("text", "")
                    self.response_tree.build_tree(entity, query, top_text)
                    followup = self.response_tree.get_followup_for_context(entity, query)
                    if followup and "followup_prompts" not in existing_results[0]:
                        existing_results[0]["followup_prompts"] = [followup]

        # SELF-PERSISTENCE: Think-plan-test-finalize reflection on final response
        if existing_results and entities and self.response_reflector:
            try:
                entity = entities[0]
                draft = existing_results[0]["text"]
                final_text, reflection = self.response_reflector.reflect(entity, draft, query)
                if final_text != draft:
                    existing_results[0]["text"] = final_text
                    existing_results[0]["source"] = existing_results[0].get("source", "") + "+reflected"
                # Record subconscious observation of the response
                if self.subconscious:
                    self.subconscious.observe(entity, existing_results[0]["text"], source="ai_response")
                # Log emotion from response
                if self.emotional_tracker:
                    resp_emotion = self.emotional_tracker.detect_emotion(existing_results[0]["text"])
                    self.emotional_tracker.log_emotion(entity, resp_emotion, trigger="ai_response",
                                                        context=existing_results[0]["text"][:80])
            except Exception as e:
                print(f"REFLECT ERROR: {e}")
                import traceback
                traceback.print_exc()

        # If we already have enough results, just return them
        if len(existing_results) >= num_responses:
            return existing_results[:num_responses]

        # Try to generate additional variations
        if self.response_generator:
            entities = self._extract_entities(query)
            entity = entities[0] if entities else query.lower()
            conv_history = self.memory.turns if hasattr(self.memory, 'turns') else []
            context_clues = self.memory.context_clues if hasattr(self.memory, 'context_clues') else []

            # Build candidates from existing results
            candidates = [{"text": r["text"], "source": r["source"], "score": r.get("score", 0.5)}
                         for r in existing_results]

            # Generate variations
            variations = self.response_generator.generate_variations(
                entity, query, candidates,
                num_variations=num_responses,
                context_clues=context_clues,
                conv_history=conv_history,
            )

            if variations:
                # THINK between variations: use reflection to diversify each one
                if self.response_reflector and entities and len(variations) > 1:
                    thought_angles = ["definition", "attributes", "comparison", "prediction", "emotion"]
                    for i, v in enumerate(variations[:num_responses]):
                        angle = thought_angles[i % len(thought_angles)]
                        # Inject angle-specific KB facts
                        if entity.lower() in self.knowledge_engine.entities:
                            ent = self.knowledge_engine.entities[entity.lower()]
                            attrs = ent.get("attributes", {})
                            props = ent.get("properties", {})
                            # Add angle-specific content
                            if angle == "prediction" and self.prediction_rotator:
                                pred = self.prediction_rotator.get_unused_prediction(entity)
                                if pred and pred["text"] not in v["text"]:
                                    v["text"] = v["text"].rstrip(".") + ". " + pred["text"]
                            elif angle == "emotion" and self.emotional_tracker:
                                emotion = self.emotional_tracker.get_dominant_emotion(entity)
                                if emotion != "neutral":
                                    v["text"] = v["text"].rstrip(".") + f". The overall mood around {entity} seems {emotion}."
                            elif angle == "attributes" and attrs:
                                # Pick a different attribute than what's already mentioned
                                mentioned = set(re.findall(r'\b\w+\b', v["text"].lower()))
                                unused_attrs = [(a, val) for a, val in attrs.items()
                                               if a.lower() not in mentioned and not isinstance(val, bool)]
                                if unused_attrs:
                                    a, val = random.choice(unused_attrs)
                                    v["text"] = v["text"].rstrip(".") + f". The {a.replace('_',' ')} of {entity} is {val}."
                            elif angle == "comparison" and self.auto_compare:
                                # Find a related entity to mention
                                all_ents = list(self.knowledge_engine.entities.keys())
                                other_ents = [e for e in all_ents if e != entity.lower() and e in props]
                                if other_ents:
                                    other = random.choice(other_ents)
                                    other_props = self.knowledge_engine.entities[other].get("properties", {})
                                    shared = set(props.keys()) & set(other_props.keys())
                                    if shared:
                                        feat = random.choice(list(shared))
                                        v["text"] = v["text"].rstrip(".") + f". Like {other}, a {entity} also has {feat.replace('_',' ')}."
                            elif angle == "definition" and props:
                                # Use a different property format
                                p, val = random.choice(list(props.items())[:3])
                                v["text"] = v["text"].rstrip(".") + f". In terms of {p.replace('_',' ')}, {entity} is {val}."

                # Merge: keep existing + add new variations, deduplicate
                all_results = list(existing_results)
                seen_texts = set(r["text"].lower().strip() for r in all_results)

                for v in variations:
                    v_text = v["text"].lower().strip()
                    if v_text not in seen_texts:
                        all_results.append(v)
                        seen_texts.add(v_text)

                # Record used phrases
                for v in all_results:
                    self.response_generator.record_used(v["text"])

                return all_results[:num_responses]

        # SELF-QUESTIONING: Occasionally append a question to engage user
        if self.self_questioner and existing_results and entities:
            entity = entities[0] if entities else None
            if entity and random.random() < 0.25:  # 25% chance
                question = self.self_questioner.generate_question(entity, query)
                if question and existing_results:
                    top = existing_results[0].copy()
                    top["text"] = top["text"].rstrip(".") + ". " + question
                    existing_results[0] = top

        return existing_results[:num_responses]

    def _texts_overlap(self, text1, text2, threshold=0.4):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > threshold

    def _extract_entities(self, query):
        entities = []
        words = query.lower().split()
        # Common attribute/property words that should NOT be treated as entities
        attr_words = {"shape", "color", "colour", "size", "weight", "height",
                      "width", "length", "speed", "price", "name", "type",
                      "sound", "lifespan", "hardness", "origin", "chemical"}
        # Try progressively longer substrings
        for length in range(min(4, len(words)), 0, -1):
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i+length])
                phrase = phrase.rstrip("?").strip()
                # Skip if phrase is just attribute words
                phrase_words = set(phrase.split())
                if phrase_words <= attr_words:
                    continue
                # Skip phrases that are only stop words
                non_stop = [w for w in phrase.split() if w not in STOP_WORDS]
                if not non_stop:
                    continue
                resolved = self.knowledge_engine.resolve_entity(phrase)
                if resolved and resolved not in entities:
                    entities.append(resolved)
        return entities

    def _detect_and_apply_state_events(self, query_lower, query):
        """Detect events that change entity state and generate state-aware responses."""
        entities = self._extract_entities(query)

        # If no entities found, try pronouns and common entity words
        if not entities:
            # Check for possessive/possessive pronouns
            if re.search(r"\b(?:its|his|her|their|my|the)\b", query_lower):
                # Try to find entity from recent conversation
                if self.memory.turns:
                    for turn in reversed(self.memory.turns):
                        for e in turn.get("entities", []):
                            if e.lower() in self.entity_states or e.lower() in [k.lower() for k in self.entity_states]:
                                entities = [e]
                                break
                        if entities:
                            break
                # If still no entity, check if any known entity is mentioned by common name
                if not entities:
                    common_names = {"cat", "dog", "turtle", "bird", "fish", "rabbit", "hamster"}
                    for name in common_names:
                        if name in query_lower and name in self.entity_states:
                            entities = [name.capitalize()]
                            break
                    # Also check for "cat" as part of compound words — only as last resort
                    if not entities and re.search(r'\bcat\b', query_lower):
                        entities = ["cat"]
                    elif not entities and re.search(r'\bdog\b', query_lower):
                        entities = ["dog"]
                    # For "its" pronoun, use first tracked entity
                    if not entities and re.search(r"\bits\b", query_lower):
                        if self.entity_states:
                            entities = [list(self.entity_states.keys())[0].capitalize()]

        # Special case: time passage with no entity — apply to all tracked entities
        if not entities and re.search(r"\d+\s*(?:hour|hr|day|min|minute)s?\s*(?:later|after|passed|ago)", query_lower):
            results = []
            for e_name, e_state in self.entity_states.items():
                hours = 1
                m = re.search(r"(\d+)\s*(?:hour|hr)", query_lower)
                if m:
                    hours = int(m.group(1))
                elif re.search(r"(\d+)\s*(?:day)", query_lower):
                    hours = int(re.search(r"(\d+)\s*(?:day)", query_lower).group(1)) * 24
                elif re.search(r"(\d+)\s*(?:min|minute)", query_lower):
                    hours = int(re.search(r"(\d+)\s*(?:min|minute)", query_lower).group(1)) / 60.0
                elif "day" in query_lower:
                    hours = 24
                e_state.apply_event("time_pass", {"hours": hours})
                healed_parts = [p for p, info in e_state.body_parts.items() if info["condition"] == "healthy"]
                healing_parts = [p for p, info in e_state.body_parts.items() if info["condition"] == "healing"]
                injured_parts = [p for p, info in e_state.body_parts.items() if "injured" in info["condition"]]
                response_parts = [f"After {hours} hours, the {e_name}'s health is now {e_state.health['overall']:.0f}%."]
                if healed_parts and not injured_parts and not healing_parts:
                    response_parts.append("It has fully healed!")
                elif healed_parts:
                    response_parts.append(f"Recovered: {', '.join(healed_parts[:3])}.")
                if healing_parts:
                    response_parts.append(f"Still healing: {', '.join(healing_parts)}.")
                response_parts.append(f"Readiness: {e_state.readiness:.0f}%.")
                results.append({"text": " ".join(response_parts), "source": "state_change", "score": 0.85})
            if results:
                return results[:1]

        if not entities:
            return None

        entity = entities[0]
        state = None
        # Get or create state through the engine's method
        if hasattr(self, '_engine_ref') and self._engine_ref:
            state = self._engine_ref.get_or_create_entity_state(entity)
        elif entity.lower() in self.entity_states:
            state = self.entity_states[entity.lower()]

        if not state:
            # Create on-the-fly
            state = EntityState(entity)
            self.entity_states[entity.lower()] = state

        # Link keywords for context
        linked_concepts = []
        if self.keyword_linker:
            linked_concepts = self.keyword_linker.get_related(entity, 5)

        # Valid body parts for this entity type
        valid_parts = set(BODY_PARTS.get(entity.lower(), BODY_PARTS.get("cat", [])))
        entity_names = {entity.lower(), entity.lower() + "s"}

        # === INJURY DETECTION ===
        injury_patterns = [
            # "my cat's tail got cut off" — possessive form
            (r"(?:my |the )?\w+'s (\w+) (?:got|has|was) (?:a |an )?(?:cut|hurt|injured|damaged|wound|severed|chopped|lost)", "injury"),
            # "tail got cut off", "eye got injured"
            (r"(\w+) (?:got|has|was) (?:a |an )?(?:cut|hurt|injured|damaged|wound|severed|chopped|lost)", "injury"),
            # "its tail is injured", "its eye was cut"
            (r"its? (\w+) (?:is|was|has been) (?:cut|injured|hurt|damaged|bleeding|severed)", "injury"),
            # "cut off its tail", "injured its eye"
            (r"(?:cut|chopped|lost|missing|injured|hurt|struck)\s+(?:its?|the)\s+(\w+)", "injury"),
            # "tail is cut off", "eye is badly injured"
            (r"(\w+) (?:is|was) (?:badly |slightly |severely )?(?:cut off|injured|hurt|damaged|bleeding)", "injury"),
        ]
        for pat, event_type in injury_patterns:
            m = re.search(pat, query_lower)
            if m:
                part = m.group(1).rstrip("s")
                # Skip if the matched "part" is actually the entity name or not a valid body part
                if part.lower() in entity_names:
                    continue
                if valid_parts and part.lower() not in valid_parts:
                    # Try to map common names
                    part_map = {"claw": "paws", "claws": "paws", "eye": "eyes", "leg": "legs",
                                "paw": "paws", "ear": "ears", "nose": "nose", "whisker": "whiskers"}
                    mapped = part_map.get(part.lower())
                    if mapped:
                        part = mapped
                    elif part.lower() not in valid_parts:
                        continue  # Not a valid body part

                # Determine severity from context
                severity = "moderate"
                if any(kw in query_lower for kw in ["badly", "severely", "deep", "completely", "cut off", "severed", "lost"]):
                    severity = "severe"
                if any(kw in query_lower for kw in ["cut off", "severed", "lost", "missing", "chopped off"]):
                    severity = "critical"
                elif any(kw in query_lower for kw in ["slightly", "a little", "minor"]):
                    severity = "minor"

                state.apply_event("injury", {"body_part": part, "severity": severity, "cause": "user_reported"})
                self.perspective_mapper.update_from_event(entity, "injury", {"part": part, "severity": severity})

                # Generate response based on state
                readiness = state.readiness
                if readiness > 60:
                    return [{"text": f"The {entity}'s {part} has been injured. "
                            f"Health is at {state.health['overall']:.0f}%. "
                            f"The {entity} is still able to help but may be in pain.",
                            "source": "state_change", "score": 0.92}]
                elif readiness > 30:
                    return [{"text": f"The {entity}'s {part} is injured ({severity}). "
                            f"Health has dropped to {state.health['overall']:.0f}%. "
                            f"The {entity} may not be able to help much right now. "
                            f"Consider taking it to a vet.",
                            "source": "state_change", "score": 0.92}]
                else:
                    return [{"text": f"The {entity} is badly hurt — {part} is {severity}ly injured. "
                            f"Health is at {state.health['overall']:.0f}%. "
                            f"The {entity} needs immediate attention and cannot help right now. "
                            f"Please take it to a vet as soon as possible.",
                            "source": "state_change", "score": 0.95}]

        # === HEALING DETECTION ===
        healing_patterns = [
            r"(?:healed|recovered|mended|fixed|better|cured)",
            r"(?:took|brought)\s+(?:\w+\s+)?(?:it|the|your|\w+)\s+(?:to\s+)?(?:the\s+)?(?:vet|doctor|hospital|clinic)",
            r"(?:vet|doctor)\s+(?:visit|check|treatment|said|helped)",
            r"(?:medicine|medication|drug|treatment)\s+(?:helped|worked|given)",
        ]
        for pat in healing_patterns:
            if re.search(pat, query_lower):
                # Determine healing method
                method = "natural"
                if any(kw in query_lower for kw in ["vet", "doctor", "hospital", "clinic"]):
                    method = "vet"
                    state.apply_event("vet_visit")
                    self.perspective_mapper.update_from_event(entity, "vet_visit")
                elif any(kw in query_lower for kw in ["medicine", "medication", "drug", "pill"]):
                    method = "medicine"
                    state.apply_event("heal", {"method": "medicine"})
                else:
                    state.apply_event("heal", {"method": "natural"})

                # Check which parts are healed
                healed_parts = [p for p, info in state.body_parts.items() if info["condition"] == "healthy"]
                healing_parts = [p for p, info in state.body_parts.items() if info["condition"] == "healing"]

                response_parts = [f"The {entity} is recovering. Health is now {state.health['overall']:.0f}%."]
                if method == "vet":
                    response_parts.append("The vet visit has helped significantly.")
                if healing_parts:
                    response_parts.append(f"Still healing: {', '.join(healing_parts)}.")
                if healed_parts and len(healed_parts) < len(state.body_parts):
                    response_parts.append(f"Fully recovered: {', '.join(healed_parts[:3])}.")
                response_parts.append(f"Readiness to help: {state.readiness:.0f}%.")

                return [{"text": " ".join(response_parts), "source": "state_change", "score": 0.93}]

        # === POSITION/MOVEMENT DETECTION ===
        move_patterns = [
            (r"(?:sat|sitting|sits)\s+(?:on|at|near|by)\s+(?:the\s+)?(\w+(?:\s+\w+)?)", "move"),
            (r"(?:moved|goes?|going)\s+(?:to|near|by)\s+(?:the\s+)?(\w+(?:\s+\w+)?)", "move"),
            (r"(?:is|was)\s+(?:on|at|near)\s+(?:the\s+)?(\w+(?:\s+\w+)?)", "move"),
        ]
        for pat, event_type in move_patterns:
            m = re.search(pat, query_lower)
            if m:
                location = m.group(1).strip().rstrip("s")
                # Normalize multi-word locations
                location = location.replace(" ", "_")
                if location in ("food_bowl", "bowl", "food"):
                    location = "food_bowl"
                elif location in ("cat_tree", "tree"):
                    location = "cat_tree"
                # Calculate distance to food based on location
                dist_map = {"desk": 8, "food_bowl": 0, "bed": 5, "chair": 6,
                           "computer": 7, "keyboard": 7, "cat_tree": 4, "window": 6,
                           "couch": 5, "sofa": 5, "floor": 3, "table": 6}
                dist = dist_map.get(location, 5)
                state.apply_event("move", {"location": location, "distance_to_food": dist, "distance_to_computer": max(0, 8 - dist)})

                response_parts = [f"The {entity} is now at the {location.replace('_', ' ')}."]
                if dist == 0:
                    response_parts.append("It is right at the food bowl.")
                elif dist <= 3:
                    response_parts.append(f"It is {dist} steps from the food bowl — close enough to eat quickly.")
                else:
                    response_parts.append(f"It is {dist} steps from the food bowl.")

                # Position-based behavior
                if location in ("desk", "keyboard", "computer"):
                    state.apply_event("emotion_change", {"emotion": "curious", "intensity": 0.5})
                    response_parts.append("It may sit on the keyboard or watch the screen.")
                elif location in ("food_bowl",):
                    state.apply_event("emotion_change", {"emotion": "happy", "intensity": 0.4})
                    response_parts.append("It may want to eat.")

                main_text = " ".join(response_parts)
                return [
                    {"text": main_text, "source": "state_change", "score": 0.91},
                    {"text": f"{entity.title()} moved to {location.replace('_', ' ')}. "
                     f"Distance to food: {dist} steps. "
                     f"{'Near the bowl — might eat.' if dist <= 3 else 'Not near the food bowl.'}",
                     "source": "state_change_variant", "score": 0.87},
                    {"text": f"New location: {location.replace('_', ' ')}. "
                     f"The {entity} is {dist} steps from food. "
                     f"{'Close enough to eat quickly.' if dist <= 3 else 'Far from the bowl.'}",
                     "source": "state_change_variant", "score": 0.84},
                ]

        # === FEEDING DETECTION ===
        is_question = re.search(r"^(?:what|how|why|when|where|who|which|can|could|do|does|did|is|are|was|were)\b", query_lower)
        if not is_question and re.search(r"\b(?:fed|feeding|gave\s+\w*\s*(?:food|milk|water|treat)|ate|eating)\b", query_lower):
            state.apply_event("feed")
            self.perspective_mapper.update_from_event(entity, "user_fed")
            dominant = max(state.emotions, key=state.emotions.get)
            return [
                {"text": f"The {entity} has been fed. Energy is now {state.health['energy']:.0f}%. "
                 f"It is feeling {dominant}. Readiness to help: {state.readiness:.0f}%.",
                 "source": "state_change", "score": 0.90},
                {"text": f"Feeding complete. {entity.title()} energy restored to {state.health['energy']:.0f}%. "
                 f"Mood: {dominant}. Help readiness: {state.readiness:.0f}%.",
                 "source": "state_change_variant", "score": 0.86},
                {"text": f"The {entity} has eaten and is feeling {dominant}. "
                 f"Energy: {state.health['energy']:.0f}%, readiness: {state.readiness:.0f}%.",
                 "source": "state_change_variant", "score": 0.83},
            ]

        # === HITTING DETECTION ===
        if re.search(r"\b(?:hit|struck|kicked|abuse|beat)\b", query_lower) and not re.search(r"\b(?:what|how|why|when|where|who|which|can|could|do|does|did|is|are|was|were)\b", query_lower):
            state.apply_event("hit")
            self.perspective_mapper.update_from_event(entity, "user_hit")
            dominant = max(state.emotions, key=state.emotions.get)
            return [
                {"text": f"The {entity} is now {dominant} and in pain. "
                 f"Health dropped to {state.health['overall']:.0f}%. "
                 f"It will not want to help you right now. "
                 f"Please be kind and let it recover before asking for help.",
                 "source": "state_change", "score": 0.94},
                {"text": f"The {entity} has been hurt. It is {dominant} with health at {state.health['overall']:.0f}%. "
                 f"Give it space to recover before requesting assistance.",
                 "source": "state_change_variant", "score": 0.90},
                {"text": f"Warning: {entity} is {dominant} and injured. Health: {state.health['overall']:.0f}%. "
                 f"It needs time to heal.",
                 "source": "state_change_variant", "score": 0.87},
            ]

        # === PLAY DETECTION ===
        if not is_question and re.search(r"\b(?:play|playing|played|petting|pet|petted|cuddle|cuddling)\b", query_lower):
            state.apply_event("play")
            self.perspective_mapper.update_from_event(entity, "user_played")
            dominant = max(state.emotions, key=state.emotions.get)
            return [
                {"text": f"The {entity} enjoyed playing! Happiness increased. "
                 f"Readiness to help is now {state.readiness:.0f}%. "
                 f"It will be more willing to assist you now.",
                 "source": "state_change", "score": 0.89},
                {"text": f"Play time was fun for the {entity}! It is now {dominant}. "
                 f"Readiness: {state.readiness:.0f}%. It will be happy to help.",
                 "source": "state_change_variant", "score": 0.85},
                {"text": f"The {entity} had a good play session. Mood: {dominant}. "
                 f"Help readiness: {state.readiness:.0f}%.",
                 "source": "state_change_variant", "score": 0.82},
            ]

        # === EMOTION CHANGE DETECTION ===
        emotion_patterns = [
            (r"\b(?:is|feels?|feeling|now)\s+(happy|sad|angry|afraid|scared|excited|calm|relaxed|stressed|anxious|cheerful|curious|playful|tired|bored|content)\b", None),
            (r"\b(happy|sad|angry|afraid|scared|excited|calm|relaxed|stressed|anxious|cheerful|curious|playful|tired|bored|content)\s+now\b", None),
        ]
        for pat, _ in emotion_patterns:
            m = re.search(pat, query_lower)
            if m:
                emotion = m.group(1)
                state.apply_event("emotion_change", {"emotion": emotion, "intensity": 0.5})
                dominant = max(state.emotions, key=state.emotions.get)
                return [
                    {"text": f"The {entity} is now feeling {emotion}. It is {dominant}. Readiness to help: {state.readiness:.0f}%.",
                     "source": "state_change", "score": 0.92},
                    {"text": f"The {entity} has shifted to feeling {emotion}. Current mood: {dominant}. Help readiness: {state.readiness:.0f}%.",
                     "source": "state_change_variant", "score": 0.88},
                    {"text": f"Emotion updated: {entity} is now {emotion}. Dominant state: {dominant}.",
                     "source": "state_change_variant", "score": 0.85},
                ]

        # === STATUS QUERY ===
        if any(kw in query_lower for kw in ["status", "condition", "how is", "how's", "state of", "health of",
                                              "full status", "complete status", "everything", "all about"]):
            summary = state.get_status_summary()
            readiness = self.perspective_mapper.get_readiness_assessment(entity, state)
            modifier = self.perspective_mapper.get_response_modifier(entity, state)
            dominant = max(state.emotions, key=state.emotions.get)
            return [
                {"text": f"Status: {summary}\nOverall readiness: {readiness*100:.0f}%. Tone: {modifier}.",
                 "source": "state_query", "score": 0.92},
                {"text": f"The {entity} is in {state.health['overall']:.0f}% health with {dominant} mood. Readiness: {state.readiness:.0f}%. {modifier}.",
                 "source": "state_query_variant", "score": 0.88},
                {"text": f"{entity.title()} status: health {state.health['overall']:.0f}%, energy {state.health['energy']:.0f}%, feeling {dominant}. Readiness: {state.readiness:.0f}%.",
                 "source": "state_query_variant", "score": 0.85},
            ]

        # === TIME PASS ===
        if re.search(r"\d+\s*(?:hour|hr|day|min|minute)s?\s*(?:later|after|passed|ago)", query_lower) or \
           re.search(r"(?:after|next)\s+(?:a\s+)?(?:few|couple|several)?\s*(?:hour|day|min)", query_lower):
            hours = 1
            m = re.search(r"(\d+)\s*(?:hour|hr)", query_lower)
            if m:
                hours = int(m.group(1))
            elif re.search(r"(\d+)\s*(?:day)", query_lower):
                hours = int(re.search(r"(\d+)\s*(?:day)", query_lower).group(1)) * 24
            elif re.search(r"(\d+)\s*(?:min|minute)", query_lower):
                hours = int(re.search(r"(\d+)\s*(?:min|minute)", query_lower).group(1)) / 60.0
            elif "day" in query_lower:
                hours = 24
            state.apply_event("time_pass", {"hours": hours})
            healed_parts = [p for p, info in state.body_parts.items() if info["condition"] == "healthy"]
            healing_parts = [p for p, info in state.body_parts.items() if info["condition"] == "healing"]
            injured_parts = [p for p, info in state.body_parts.items() if "injured" in info["condition"]]
            response_parts = [f"After {hours} hours, the {entity}'s health is now {state.health['overall']:.0f}%."]
            if healed_parts and not injured_parts and not healing_parts:
                response_parts.append("It has fully healed!")
            elif healed_parts:
                response_parts.append(f"Recovered: {', '.join(healed_parts[:3])}.")
            if healing_parts:
                response_parts.append(f"Still healing: {', '.join(healing_parts)}.")
            response_parts.append(f"Readiness: {state.readiness:.0f}%.")
            return [{"text": " ".join(response_parts), "source": "state_change", "score": 0.85}]

        # === CAPABILITY CHECK ===
        cap_patterns = [
            r"can (?:the |your )?(\w+) (?:help|see|walk|play|run|eat|move)",
            r"(?:will|would) (?:the |your )?(\w+) (?:help|want|be able)",
            r"is (?:the |your )?(\w+) (?:able|ready|willing|capable)",
        ]
        for pat in cap_patterns:
            m = re.search(pat, query_lower)
            if m:
                target_entity = m.group(1).rstrip("s")
                if target_entity.lower() == entity.lower() or entity.lower().startswith(target_entity.lower()):
                    readiness = state.readiness
                    can_help = state.can_do("help_user", 0.3)
                    modifier = self.perspective_mapper.get_response_modifier(entity, state)
                    response_parts = []
                    if can_help:
                        response_parts.append(f"The {entity} is ready to help (readiness: {readiness:.0f}%).")
                    else:
                        response_parts.append(f"The {entity} is not able to help right now (readiness: {readiness:.0f}%).")
                    if modifier != "neutral":
                        response_parts.append(f"It is feeling {modifier}.")
                    # Suggest what's needed
                    if readiness < 50:
                        if state.health["energy"] < 40:
                            response_parts.append("It may need food or rest first.")
                        if state.health["pain_level"] > 40:
                            response_parts.append("It may need medical attention.")
                        if any(v > 0.5 for k, v in state.emotions.items() if k in ("afraid", "angry")):
                            response_parts.append("It may need some time to calm down or play first.")
                    return [{"text": " ".join(response_parts), "source": "capability_check", "score": 0.89}]

        # === FOOD BOWL ANALYSIS ===
        if any(kw in query_lower for kw in ["food bowl", "bowl", "hungry", "feeding area"]):
            if self.decision_engine:
                distance_to_food = state.position.get("distance_to_food", 5)
                analysis = self.decision_engine.get_food_bowl_analysis(state, distance_to_food)
                parts = [analysis["suggestion"]]
                # Check if food bowl was moved
                if "moved" in query_lower or "relocated" in query_lower:
                    old_dist = state.position.get("distance_to_food", 5)
                    new_dist = distance_to_food
                    if "close" in query_lower or "nearer" in query_lower:
                        new_dist = max(0, old_dist - 3)
                    elif "far" in query_lower or "away" in query_lower:
                        new_dist = min(10, old_dist + 3)
                    effects = self.decision_engine.get_food_bowl_moved_effect(state, old_dist, new_dist)
                    if effects["explanation"]:
                        parts.append(effects["explanation"])
                    state.apply_event("move", {"location": "food_bowl", "distance_to_food": new_dist})
                return [{"text": " ".join(parts), "source": "behavior", "score": 0.87}]

        # === BEHAVIORAL DECISION REQUEST ===
        if any(kw in query_lower for kw in ["what should", "what can it do", "recommend", "suggestion",
                                              "what to do", "activity", "plan"]):
            if self.decision_engine:
                decision = self.decision_engine.decide_activity(entity, state, None, 5)
                assessment = self.decision_engine.get_health_assessment(state)
                parts = []
                if assessment["suggestions"]:
                    parts.append(assessment["suggestions"][0])
                parts.append(f"Recommended: {decision['description']}.")
                if decision["alternatives"]:
                    parts.append(f"Alternatives: {', '.join(decision['alternatives'][:2])}.")
                return [{"text": " ".join(parts), "source": "behavior_decision", "score": 0.86}]

        # === LEAVING DETECTION ===
        if any(kw in query_lower for kw in ["leaving", "going outside", "going out", "heading for door",
                                              "running away", "escaping"]):
            if self.behavior_tracker:
                leaving = self.behavior_tracker.predict_leaving(entity, state, None)
                parts = [f"The {entity} has a {leaving['probability']*100:.0f}% chance of leaving."]
                if leaving["reasons"]:
                    parts.append(f"Reasons: {', '.join(leaving['reasons'][:2])}.")
                if leaving["should_warn"]:
                    advice = self.behavior_tracker.get_obstacle_suggestions(entity, state, leaving["probability"])
                    if advice:
                        parts.append(advice[0])
                self.behavior_tracker.log_leaving_event(entity, "user_reported", False)
                return [{"text": " ".join(parts), "source": "leaving_prediction", "score": 0.88}]

        # === MOVEMENT TRACKING ===
        move_track_kw = ["moving", "walked", "went to", "moved to", "is at", "sitting at"]
        if any(kw in query_lower for kw in move_track_kw) and self.behavior_tracker:
            # Track the movement
            loc = state.position.get("location", "unknown")
            dist_user = state.position.get("distance_to_computer", 5)
            dist_food = state.position.get("distance_to_food", 5)
            self.behavior_tracker.track_movement(entity, loc, dist_user, dist_food)

            # Get proximity advice
            advice = self.behavior_tracker.get_proximity_advice(entity, state, dist_user)
            if advice:
                parts = [a["text"] for a in advice[:2]]
                return [{"text": " ".join(parts), "source": "movement_tracking", "score": 0.84}]

        # === COMPREHENSIVE STATUS WITH BEHAVIOR ===
        if any(kw in query_lower for kw in ["full status", "complete status", "everything", "all about"]):
            if self.decision_engine:
                response = self.decision_engine.generate_response(entity, state, query, None, 5)
                summary = state.get_status_summary()
                movement = self.behavior_tracker.get_movement_summary(entity) if self.behavior_tracker else ""
                perf = self.performance_logger.get_performance_summary(entity) if self.performance_logger else ""
                parts = [summary]
                if movement:
                    parts.append(movement)
                if response:
                    parts.append(response)
                if perf:
                    parts.append(perf)
                return [{"text": " ".join(parts), "source": "comprehensive_status", "score": 0.9}]

        # === PLAY HEALTH COST ===
        if re.search(r"\b(?:play|playing|played)\b", query_lower):
            if state.health["overall"] < 40:
                return [{"text": f"The {entity} should not play right now. Its health is at {state.health['overall']:.0f}%, "
                        f"which is too low. Let it rest and eat first, then it can help you.",
                        "source": "behavior_warning", "score": 0.91}]
            elif state.health["overall"] < 60:
                state.apply_event("play")
                return [{"text": f"The {entity} played a little. Health dropped slightly to {state.health['overall']:.0f}%. "
                        f"It should eat soon to recover. Then it can help you more.",
                        "source": "behavior", "score": 0.87}]
            else:
                state.apply_event("play")
                return [{"text": f"The {entity} enjoyed playing! It is now in a good mood. "
                        f"Readiness to help: {state.readiness:.0f}%. "
                        f"It may be more willing to assist you now.",
                        "source": "behavior", "score": 0.87}]

        return None

    def _handle_natural_talk(self, query_lower, entities, query_words):
        """Handle natural conversational queries like 'tell me about', 'what do you know about', etc."""
        if not entities:
            return None

        # Check for natural talk patterns
        patterns = [
            r"tell me about (.+)",
            r"what do you know about (.+)",
            r"talk about (.+)",
            r"describe (.+)",
            r"explain (.+)",
            r"what can you tell me about (.+)",
            r"info(?:rmation)? about (.+)",
            r"know anything about (.+)",
        ]

        entity_lower = None
        for pat in patterns:
            match = re.search(pat, query_lower)
            if match:
                entity_lower = match.group(1).strip().rstrip("?")
                entity_lower = entity_lower.rstrip("s") if entity_lower.endswith("s") and len(entity_lower) > 3 else entity_lower
                break

        if not entity_lower:
            return None

        # Try to resolve entity
        resolved = self.knowledge_engine.resolve_entity(entity_lower)
        if not resolved:
            return None

        data = self.knowledge_engine.entities.get(resolved, {})
        if not data:
            return None

        descs = data.get("descriptions", [])
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        category = data.get("category", "")

        parts = []

        # Start with a natural introduction
        if category:
            intros = [
                f"A {resolved} is a type of {category}.",
                f"Let me tell you about {resolved}s — they're a type of {category}.",
                f"Sure! A {resolved} is a {category}.",
                f"{resolved.capitalize()}s are {category}s.",
            ]
            parts.append(random.choice(intros))

        # Add descriptions
        if descs:
            parts.append(descs[0])

        # Add key attributes (pick 2-3 most interesting)
        interesting_attrs = []
        for attr, val in attrs.items():
            if isinstance(val, bool) and val:
                pretty = attr.replace("has_", "").replace("is_", "").replace("_", " ")
                interesting_attrs.append(pretty)
        if interesting_attrs:
            sample = random.sample(interesting_attrs, min(3, len(interesting_attrs)))
            parts.append(f"They are known for having {', '.join(sample[:-1])}{' and ' + sample[-1] if len(sample) > 1 else ''}.")

        # Add a key property
        if props:
            prop_key = random.choice(list(props.keys()))
            readable = prop_key.replace("_", " ")
            parts.append(f"Their {readable} is {props[prop_key]}.")

        if not parts:
            return None

        # Compose natural response
        combined = " ".join(parts)
        return [{"text": combined, "source": "natural_talk", "score": 0.88}]

    def _handle_context_query(self, query_lower, entities):
        """Handle queries that depend on conversation context, like 'what was the original color'."""
        # Check for temporal/origin keywords
        origin_keywords = ["original", "originally", "before", "used to be", "initially", "at first",
                          "was the", "what was", "used to", "earlier", "previously"]
        is_origin_query = any(kw in query_lower for kw in origin_keywords)

        if is_origin_query and entities:
            for entity in entities:
                entity_lower = entity.lower()
                # Check if there are state changes for this entity
                state = self.memory.get_entity_state(entity)
                if state:
                    for attr, info in state.items():
                        if attr in query_lower or self.knowledge_engine._format_attr_name(f"is_{attr}") in query_lower:
                            original = info.get("original", "unknown")
                            current = info.get("current", "unknown")
                            if original != "unknown":
                                return [
                                    {"text": f"The original {attr} of this {entity_lower} was {original}. It is now {current}.",
                                     "source": "context_state", "score": 0.95},
                                    {"text": f"Previously, the {attr} was {original}. Now it is {current}.",
                                     "source": "context_state_variant", "score": 0.91},
                                    {"text": f"Before the change, {entity_lower}'s {attr} was {original}. Current: {current}.",
                                     "source": "context_state_variant", "score": 0.88},
                                ]

                # Check for recent context clues about the entity
                clues = self.memory.get_recent_context_clues(5)
                for clue in clues:
                    if clue.get("entity") == entity_lower:
                        attr = clue.get("attribute", "")
                        if attr in query_lower or self.knowledge_engine._format_attr_name(f"is_{attr}") in query_lower:
                            # Return original value from knowledge base
                            data = self.knowledge_engine.entities.get(entity_lower, {})
                            attrs = data.get("attributes", {})
                            props = data.get("properties", {})
                            for a, v in attrs.items():
                                readable = self.knowledge_engine._format_attr_name(a)
                                if attr == a or attr in readable or readable == attr:
                                    if isinstance(v, bool):
                                        val = "yes" if v else "no"
                                    else:
                                        val = v
                                    return [
                                        {"text": f"The original {readable} of a {entity_lower} was {val}, before it was changed.",
                                         "source": "context_original", "score": 0.91},
                                        {"text": f"Originally, the {readable} was {val}. It has since been modified.",
                                         "source": "context_original_variant", "score": 0.87},
                                    ]
                            for p, v in props.items():
                                readable = p.replace("_", " ")
                                if attr in p or attr in readable:
                                    return [
                                        {"text": f"The original {readable} of a {entity_lower} was {v}.",
                                         "source": "context_original", "score": 0.90},
                                        {"text": f"Before any changes, the {readable} was {v}.",
                                         "source": "context_original_variant", "score": 0.86},
                                    ]

        return None

    def _handle_fact_correction(self, query_lower, entities):
        """Detect user statements that contradict dataset facts and correct them."""
        if "?" in query_lower:
            return None

        # Detect patterns
        statement_patterns = [
            (r"the\s+(\w+)\s+of\s+(?:the\s+|a\s+|an\s+)?(\w+)\s+is\s+(.+)", "attr_of_entity"),
            (r"(\w+)\s+is\s+(?:a\s+|an\s+|the\s+)?(.+)", "entity_is_value"),
            (r"(\w+)\s+has\s+(?:a\s+|an\s+|the\s+)?(.+)", "entity_has_attr"),
        ]

        for pat, pattern_type in statement_patterns:
            m = re.search(pat, query_lower)
            if m:
                groups = m.groups()
                if pattern_type == "attr_of_entity":
                    attr_name, entity_name, claimed_value = groups
                elif pattern_type == "entity_is_value":
                    entity_name, claimed_value = groups
                    attr_name = None  # need to infer
                elif pattern_type == "entity_has_attr":
                    entity_name, claimed_value = groups
                    attr_name = "has"
                else:
                    continue

                claimed_value = claimed_value.strip().rstrip(".")

                # Check KB entities
                for entity_name_kb, data in self.knowledge_engine.entities.items():
                    if entity_name in entity_name_kb or entity_name_kb in entity_name:
                        props = data.get("properties", {})
                        attrs = data.get("attributes", {})
                        descs = data.get("descriptions", [])

                        # For "entity is value" — find which property matches
                        if attr_name is None:
                            # Check if claimed_value matches any property
                            for prop, val in props.items():
                                readable = prop.replace("_", " ")
                                actual_val = str(val).lower()
                                if claimed_value in actual_val or actual_val in claimed_value:
                                    # User's claim matches this property — no contradiction
                                    break
                                # Check if the claimed value contradicts a known property
                                meaningful_claimed = set(claimed_value.split()) - STOP_WORDS
                                meaningful_actual = set(actual_val.split()) - STOP_WORDS
                                overlap = meaningful_claimed & meaningful_actual
                                if len(overlap) >= 2 and claimed_value not in actual_val and actual_val not in claimed_value:
                                        return [
                                            {"text": f"Actually, {entity_name_kb} {readable} is {val}. "
                                                     f"The sky's color is determined by Rayleigh scattering of sunlight — not something humans can change.",
                                             "source": "fact_correction", "score": 0.95},
                                            {"text": f"That's not right. {entity_name_kb.title()} {readable} is {val}.",
                                             "source": "fact_correction_variant", "score": 0.91},
                                        ]

                            # Check attributes (boolean)
                            for attr, val in attrs.items():
                                readable = self.knowledge_engine._format_attr_name(attr)
                                if isinstance(val, bool) and not val:
                                    if any(w in claimed_value for w in readable.split()):
                                        return [
                                            {"text": f"Actually, {entity_name_kb} does not {readable}.",
                                             "source": "fact_correction", "score": 0.93},
                                        ]

                        # For explicit attribute match
                        if attr_name:
                            for prop, val in props.items():
                                readable = prop.replace("_", " ")
                                if attr_name in readable or readable in attr_name or \
                                   any(w in readable for w in attr_name.split()) or \
                                   any(w in attr_name for w in readable.split()):
                                    actual_val = str(val).lower()
                                    if claimed_value not in actual_val and actual_val not in claimed_value:
                                        return [
                                            {"text": f"Actually, {entity_name_kb} {readable} is {val}.",
                                             "source": "fact_correction", "score": 0.95},
                                            {"text": f"That's incorrect. The {readable} of {entity_name_kb} is {val}.",
                                             "source": "fact_correction_variant", "score": 0.91},
                                        ]

        return None

    def _handle_task_command(self, query_lower, query):
        """Route explicit task-management phrases to TaskPerformer/GoalTracker
        via an agent when one's available."""
        if not hasattr(self, '_task_performer_ref'):
            return None
        tp = self._task_performer_ref
        gt = getattr(self, 'goal_tracker', None)
        am = getattr(self, '_agent_manager_ref', None)

        def _dispatch(task_type, description, context):
            task = tp.create_task(task_type, description, context)
            agent = am.assign_task(task) if am else None
            result = tp.execute_task(task, {})
            if agent:
                am.complete_task(agent.agent_id, result)
            return result, agent

        if re.match(r'^(add|create)\s+(a\s+)?todo', query_lower):
            item = re.sub(r'^(add|create)\s+(a\s+)?todo\s*', '', query_lower).strip()
            if not item:
                return [{"text": "What would you like me to add to the todo list?", "source": "task_prompt", "score": 0.6}]
            result, agent = _dispatch("create_todo", f"Add: {item}",
                                       {"list_name": "default", "items": [item], "action": "create"})
            agent_note = f" (handled by {agent.agent_id})" if agent else ""
            return [{"text": f"Added '{item}' to your todo list. Pending: {result.get('pending', 0)}.{agent_note}",
                     "source": "task_done", "score": 0.9}]

        if re.match(r'^(show|check|list)\s+(my\s+)?todos?', query_lower):
            result, agent = _dispatch("check_completion", "List todos", {"list_name": "default"})
            pending = result.get("pending_items", [])
            if pending:
                return [{"text": f"Pending todos: {', '.join(pending)}.", "source": "task_done", "score": 0.9}]
            return [{"text": "No pending todos.", "source": "task_done", "score": 0.9}]

        if re.match(r'^(set|create)\s+(a\s+)?goal', query_lower) and gt:
            goal_desc = re.sub(r'^(set|create)\s+(a\s+)?goal\s*(to|for)?\s*', '', query_lower).strip()
            if goal_desc:
                gt.create_goal(goal_desc)
                return [{"text": f"Goal set: {goal_desc}. I'll track progress on this.",
                         "source": "task_done", "score": 0.9}]

        if re.match(r'^(run|start)\s+(a\s+)?(test|research)', query_lower):
            topic = re.sub(r'^(run|start)\s+(a\s+)?(test|research)\s*(on|about)?\s*', '', query_lower).strip()
            task_type = "research" if "research" in query_lower else "run_test"
            context = {"topic": topic} if task_type == "research" else {"test_type": "syntax", "target": "model.py"}
            result, agent = _dispatch(task_type, f"{task_type} on {topic}", context)
            agent_note = f" (via {agent.agent_id})" if agent else ""
            return [{"text": f"Task complete{agent_note}: {result}", "source": "task_done", "score": 0.85}]

        if re.match(r'^(agent|agents)\s+status', query_lower) and am:
            status = am.get_status()
            return [{"text": f"Agents: {status['total_agents']} total, {status['idle']} idle, "
                              f"{status['busy']} busy, {status['completed']} completed tasks.",
                     "source": "task_done", "score": 0.85}]

        return None

    def _assess_certainty(self, entity_name, attribute=None):
        """Assess certainty of a fact: immutable, mutable, or conditional."""
        data = self.knowledge_engine.entities.get(entity_name.lower(), {})
        if not data:
            return {"level": "unknown", "score": 0.0, "reason": "entity not in knowledge base"}

        props = data.get("properties", {})
        attrs = data.get("attributes", {})

        # Natural/physical properties are immutable
        immutable_props = {"color", "weight_kg", "lifespan_years", "speed_kmh",
                          "hardness", "chemical_composition", "shell_count",
                          "tail_count", "shell_material", "shell_hardness"}
        # Biological states can change
        mutable_props = {"health", "energy", "mood", "position", "location"}
        # Environmental factors are conditional
        conditional_props = {"habitat", "diet", "behavior", "activity"}

        if attribute:
            for prop in props:
                if attribute in prop or prop in attribute:
                    if prop in immutable_props:
                        return {"level": "immutable", "score": 0.95,
                                "reason": f"{prop} is a fixed physical property of {entity_name}"}
                    elif prop in mutable_props:
                        return {"level": "mutable", "score": 0.6,
                                "reason": f"{prop} can change over time for {entity_name}"}
                    else:
                        return {"level": "conditional", "score": 0.5,
                                "reason": f"{prop} depends on circumstances for {entity_name}"}

        # Default: natural properties are immutable
        return {"level": "immutable", "score": 0.8,
                "reason": f"natural properties of {entity_name} are fixed"}

    def _handle_hypothetical(self, query_lower, entities):
        """Handle hypothetical questions like 'what if I changed the sky to red'."""
        # Skip factual questions — these are NOT hypothetical
        if re.search(r"^(?:why|how|what|when|where)\s+(?:do|does|did|is|are|was|were|can|could|would|should)\b", query_lower):
            return None
        if re.search(r"^(?:do|does|did|can|could|would|should)\s+\w+", query_lower):
            return None

        hypothetical_patterns = [
            r"what if\s+(.+)",
            r"what would happen if\s+(.+)",
            r"could\s+(.+)\s+be changed",
            r"is it possible to\s+(.+)",
            r"can\s+(?:the\s+|a\s+|an\s+)?(\w+)\s+be\s+changed",
        ]

        for pat in hypothetical_patterns:
            m = re.search(pat, query_lower)
            if m:
                scenario = m.group(1).strip()

                # Find entity in scenario
                found_entity = None
                for e in entities:
                    if e.lower() in scenario:
                        found_entity = e
                        break

                if not found_entity:
                    # Try to extract entity from scenario
                    words = scenario.split()
                    for w in words:
                        data = self.knowledge_engine.entities.get(w, {})
                        if data:
                            found_entity = w
                            break

                if found_entity:
                    # Assess certainty
                    certainty = self._assess_certainty(found_entity)
                    level = certainty["level"]
                    score = certainty["score"]
                    reason = certainty["reason"]

                    if level == "immutable":
                        return [
                            {"text": f"That's not possible. {found_entity.title()} has fixed natural properties that cannot be changed by humans. {reason}.",
                             "source": "hypothetical", "score": 0.92},
                            {"text": f"The {found_entity}'s properties are determined by nature and cannot be artificially altered. {reason}.",
                             "source": "hypothetical_variant", "score": 0.88},
                        ]
                    elif level == "mutable":
                        return [
                            {"text": f"It's possible for {found_entity} to change in certain conditions. {reason}. However, it would require specific circumstances.",
                             "source": "hypothetical", "score": 0.85},
                            {"text": f"{found_entity.title()} can change under the right conditions. {reason}. But this would be unusual.",
                             "source": "hypothetical_variant", "score": 0.81},
                        ]
                    else:
                        return [
                            {"text": f"It depends on the circumstances. {reason}. Under normal conditions, this is unlikely but not impossible.",
                             "source": "hypothetical", "score": 0.78},
                        ]

        return None

    def _handle_conversation_history(self, query_lower):
        """Handle queries about conversation history like 'what have we talked about'."""
        history_keywords = ["what have we talked about", "what did we discuss", "what topics",
                           "what have you told me", "conversation history",
                           "what are some animals we talked about", "what things did we discuss",
                           "what have we covered", "what did we talk about"]

        is_history_query = any(kw in query_lower for kw in history_keywords)
        if not is_history_query:
            return None

        # Gather entities and categories from conversation
        entities_discussed = []
        categories_discussed = []
        entity_qa_facts = {}  # entity -> list of QA answers
        entity_props = {}     # entity -> list of property facts

        for turn in self.memory.turns[-10:]:
            for e in turn.get("entities", []):
                if e not in entities_discussed:
                    entities_discussed.append(e)
            cat = turn.get("category", "")
            if cat and cat not in categories_discussed:
                categories_discussed.append(cat)

        # Pull interesting dataset facts about each discussed entity
        for e in entities_discussed:
            e_lower = e.lower()
            # Find QA pairs mentioning this entity
            facts = []
            for q, a in self.knowledge_engine.dataset_qa:
                if e_lower in q.lower() and a and len(a) > 10:
                    facts.append(a)
            if facts:
                random.shuffle(facts)
                entity_qa_facts[e_lower] = facts[:3]
            # Pull KB properties
            data = self.knowledge_engine.entities.get(e_lower, {})
            props = data.get("properties", {})
            if props:
                prop_list = list(props.items())
                random.shuffle(prop_list)
                entity_props[e_lower] = [(p, v) for p, v in prop_list[:2]]

        if not entities_discussed:
            return [{"text": "We haven't discussed any specific topics yet.", "source": "history", "score": 0.7}]

        # Build diverse response variants
        connectors = ["Also, ", "Furthermore, ", "In addition, ", "Moreover, ", "Plus, ", "Additionally, "]
        random.shuffle(connectors)
        ci = 0  # connector index

        def next_connector():
            nonlocal ci
            c = connectors[ci % len(connectors)]
            ci += 1
            return c

        results = []

        # Variant 1: Natural summary with dataset facts
        parts = []
        parts.append(f"We've been talking about {entities_discussed[0]}.")
        for e in entities_discussed[1:]:
            parts.append(f"{next_connector()}we also discussed {e}.")

        # Add a random fact about one of the entities
        if entity_qa_facts:
            random_entity = random.choice(list(entity_qa_facts.keys()))
            facts = entity_qa_facts[random_entity]
            if facts:
                parts.append(f"{next_connector()}{facts[0]}")
        elif entity_props:
            random_entity = random.choice(list(entity_props.keys()))
            props = entity_props[random_entity]
            if props:
                p, v = props[0]
                parts.append(f"{next_connector()}the {p.replace('_', ' ')} of a {random_entity} is {v}.")

        results.append({
            "text": " ".join(parts),
            "source": "conversation_history",
            "score": 0.88,
        })

        # Variant 2: Fact-focused with details about each entity
        parts2 = []
        for i, e in enumerate(entities_discussed[:3]):
            e_lower = e.lower()
            if e_lower in entity_qa_facts and entity_qa_facts[e_lower]:
                parts2.append(f"{entity_qa_facts[e_lower][0]}")
            elif e_lower in entity_props and entity_props[e_lower]:
                p, v = entity_props[e_lower][0]
                parts2.append(f"The {p.replace('_', ' ')} of a {e_lower} is {v}.")
            else:
                data = self.knowledge_engine.entities.get(e_lower, {})
                descs = data.get("descriptions", [])
                if descs:
                    parts2.append(descs[0])
                else:
                    parts2.append(f"We discussed {e}.")
            if i < len(entities_discussed[:3]) - 1:
                parts2.append(f" {next_connector()}")

        if parts2:
            results.append({
                "text": "".join(parts2).strip(),
                "source": "conversation_history_variant",
                "score": 0.84,
            })

        # Variant 3: Category overview
        if categories_discussed:
            cat_facts = []
            for cat in categories_discussed[:3]:
                # Find a random dataset fact about this category
                for q, a in self.knowledge_engine.dataset_qa:
                    if cat.lower() in q.lower() and a and len(a) > 10:
                        cat_facts.append(a)
                        break
            if cat_facts:
                results.append({
                    "text": f"{next_connector()}We covered topics like {', '.join(categories_discussed[:3])}. {cat_facts[0]}",
                    "source": "conversation_history_variant",
                    "score": 0.82,
                })

        return results[:3]

    def _handle_yes_no_question(self, query_lower, entities):
        """Handle yes/no questions like 'do turtles have tails'."""
        # Detect yes/no questions
        yes_no_start = re.match(r"(?:do|does|are|is|can|could|would|should|has|have)\s+", query_lower)
        if not yes_no_start:
            return None

        # Extract entity and attribute from the question
        patterns = [
            r"(?:do|does)\s+(\w+)\s+have\s+(.+?)(?:\?|$)",
            r"(?:are|is)\s+(?:a\s+|an\s+|the\s+)?(\w+)\s+(.+?)(?:\?|$)",
            r"(?:can|could|would|should)\s+(?:a\s+|an\s+|the\s+)?(\w+)\s+(.+?)(?:\?|$)",
            r"(?:has|have)\s+(?:a\s+|an\s+|the\s+)?(\w+)\s+(.+?)(?:\?|$)",
        ]

        for pat in patterns:
            m = re.search(pat, query_lower)
            if m:
                entity_name = m.group(1).strip()
                attribute_phrase = m.group(2).strip().rstrip("?")

                # Skip "should i", "can i" — not fact-check questions
                if entity_name in ("i", "we", "you"):
                    return None

                # Check KB entities
                for entity_name_kb, data in self.knowledge_engine.entities.items():
                    if entity_name in entity_name_kb or entity_name_kb in entity_name:
                        attrs = data.get("attributes", {})
                        props = data.get("properties", {})
                        descs = data.get("descriptions", [])

                        # Check boolean attributes
                        for attr, val in attrs.items():
                            readable = self.knowledge_engine._format_attr_name(attr)
                            attr_words = set(readable.split())
                            phrase_words = set(attribute_phrase.split())
                            # Use word-boundary regex to prevent substring false positives
                            word_match = False
                            for w in phrase_words:
                                if re.search(r'\b' + re.escape(w) + r'\b', readable):
                                    word_match = True
                                    break
                            if not word_match:
                                for w in attr_words:
                                    if re.search(r'\b' + re.escape(w) + r'\b', attribute_phrase):
                                        word_match = True
                                        break
                            if word_match:
                                if isinstance(val, bool):
                                    is_has_attr = attr.startswith("has_") or attr.startswith("lays_")
                                    # Build enriched answer with cross-referencing and examples
                                    answer_parts = []
                                    if val:
                                        if is_has_attr:
                                            article = "a " if not readable.startswith(("a ", "e ", "i ", "o ", "u ")) else "an "
                                            if readable in ("fur",):
                                                article = ""
                                            answer_parts.append(f"Yes, {entity_name_kb} has {article}{readable}.")
                                        else:
                                            answer_parts.append(f"Yes, {entity_name_kb} is {readable}.")
                                    else:
                                        if is_has_attr:
                                            article = "a " if not readable.startswith(("a ", "e ", "i ", "o ", "u ")) else "an "
                                            if readable in ("fur",):
                                                article = ""
                                            answer_parts.append(f"No, {entity_name_kb} does not have {article}{readable}.")
                                        else:
                                            answer_parts.append(f"No, {entity_name_kb} is not {readable}.")

                                    # Add proportion scoring
                                    if self.proportion_scorer:
                                        prop = self.proportion_scorer.score_claim(entity_name_kb, [readable])
                                        if prop["total"] > 0:
                                            answer_parts.append(
                                                f"({prop['percentage']} of {prop['total']} facts confirm this)."
                                            )

                                    # Add cross-reference from opposite entity
                                    if self.opposite_engine:
                                        opposite = self.opposite_engine.find_opposite(entity_name_kb)
                                        if opposite:
                                            opp_data = self.knowledge_engine.entities.get(opposite.lower(), {})
                                            opp_attrs = opp_data.get("attributes", {})
                                            if attr in opp_attrs:
                                                opp_val = opp_attrs[attr]
                                                if opp_val != val:
                                                    if opp_val:
                                                        answer_parts.append(
                                                            f"By contrast, {opposite} is {readable}."
                                                        )
                                                    else:
                                                        if is_has_attr:
                                                            answer_parts.append(
                                                                f"Unlike {entity_name_kb}, {opposite} does not have {readable}."
                                                            )
                                                        else:
                                                            answer_parts.append(
                                                                f"Unlike {entity_name_kb}, {opposite} is not {readable}."
                                                            )

                                    # Add example from descriptions
                                    if len(answer_parts) < 3:
                                        for d in data.get("descriptions", []):
                                            if readable in d.lower() or attr.replace("_", " ") in d.lower():
                                                answer_parts.append(f"For example, {d.strip()}")
                                                break

                                    combined = " ".join(answer_parts)
                                    results = [{"text": combined, "source": "yes_no", "score": 0.93}]
                                    # Add variant
                                    variant_parts = []
                                    if val:
                                        if is_has_attr:
                                            variant_parts.append(f"That's correct. {entity_name_kb.title()} has {readable}.")
                                        else:
                                            variant_parts.append(f"That's correct. {entity_name_kb.title()} is {readable}.")
                                    else:
                                        if is_has_attr:
                                            variant_parts.append(f"Actually, {entity_name_kb} does not have {readable}.")
                                        else:
                                            variant_parts.append(f"Actually, {entity_name_kb} is not {readable}.")
                                    if len(answer_parts) > 1:
                                        variant_parts.extend(answer_parts[1:])
                                    results.append({"text": " ".join(variant_parts), "source": "yes_no_variant", "score": 0.89})
                                    return results

                        # Fallback: check raw attribute name (handles venomous->venom, aquatic->habitat, etc.)
                        for attr, val in attrs.items():
                            if isinstance(val, bool):
                                raw_name = attr.replace("_", " ").replace("is ", "").replace("has ", "")
                                is_has_attr = attr.startswith("has_") or attr.startswith("lays_")
                                article = ""
                                if is_has_attr:
                                    article = "a " if not raw_name.startswith(("a ", "e ", "i ", "o ", "u ")) else "an "
                                    if raw_name in ("fur",):
                                        article = ""
                                for w in attribute_phrase.split():
                                    if re.search(r'\b' + re.escape(w) + r'\b', raw_name) or \
                                       re.search(r'\b' + re.escape(raw_name) + r'\b', attribute_phrase):
                                        if val:
                                            if is_has_attr:
                                                return [
                                                    {"text": f"Yes, {entity_name_kb} has {article}{raw_name}.",
                                                     "source": "yes_no", "score": 0.93},
                                                ]
                                            else:
                                                return [
                                                    {"text": f"Yes, {entity_name_kb} is {raw_name}.",
                                                     "source": "yes_no", "score": 0.93},
                                                ]
                                        else:
                                            if is_has_attr:
                                                return [
                                                    {"text": f"No, {entity_name_kb} does not have {article}{raw_name}.",
                                                     "source": "yes_no", "score": 0.93},
                                                ]
                                            else:
                                                return [
                                                    {"text": f"No, {entity_name_kb} is not {raw_name}.",
                                                     "source": "yes_no", "score": 0.93},
                                                ]

                        # Check properties
                        for prop, val in props.items():
                            readable = prop.replace("_", " ")
                            prop_match = False
                            for w in attribute_phrase.split():
                                if re.search(r'\b' + re.escape(w) + r'\b', readable):
                                    prop_match = True
                                    break
                            if prop_match:
                                return [
                                    {"text": f"Yes, the {readable} of {entity_name_kb} is {val}.",
                                     "source": "yes_no", "score": 0.91},
                                ]

                        # Handle "can X Y" capability questions with common-sense rules
                        can_pattern = re.search(r'(?:can|could)\s+(?:a\s+|an\s+|the\s+)?(\w+)\s+(.+?)(?:\?|$)', query_lower)
                        if can_pattern:
                            action = can_pattern.group(2).strip()
                            # Common-sense: predation questions
                            if any(w in action for w in ("eat", "attack", "harm", "kill", "hunt")):
                                # Check if entity is predator and target is prey
                                target_words = action.split()
                                for target_word in target_words:
                                    for target_kb, target_data in self.knowledge_engine.entities.items():
                                        if target_word in target_kb or target_kb in target_word:
                                            target_attrs = target_data.get("attributes", {})
                                            if attrs.get("is_predator") and target_attrs.get("is_prey"):
                                                return [{"text": f"Yes, {entity_name_kb} can eat {target_kb}. {entity_name_kb.title()} is a predator and {target_kb} is prey.", "source": "yes_no", "score": 0.93}]
                                            elif attrs.get("is_prey") and target_attrs.get("is_predator"):
                                                return [{"text": f"No, {entity_name_kb} cannot eat {target_kb}. {entity_name_kb.title()} is prey and {target_kb} is a predator.", "source": "yes_no", "score": 0.93}]
                                            elif attrs.get("is_predator") and target_attrs.get("is_predator"):
                                                return [{"text": f"Both {entity_name_kb} and {target_kb} are predators. While they can physically harm each other, neither typically hunts the other as prey.", "source": "yes_no", "score": 0.92}]
                                            elif not attrs.get("is_predator"):
                                                return [{"text": f"No, {entity_name_kb} is not a predator and typically does not eat other animals.", "source": "yes_no", "score": 0.91}]
                            # Common-sense: animals with feathers can fly
                            if action in ("fly", "fly high", "soar"):
                                has_feathers = attrs.get("has_feathers", False)
                                if has_feathers:
                                    return [
                                        {"text": f"Yes, {entity_name_kb} can fly. It has wings and feathers.",
                                         "source": "yes_no", "score": 0.92},
                                    ]
                                elif not has_feathers:
                                    return [
                                        {"text": f"No, {entity_name_kb} cannot fly. Only animals with wings and feathers can fly.",
                                         "source": "yes_no", "score": 0.92},
                                    ]
                            # Common-sense: aquatic animals can swim
                            if action in ("swim", "breathe underwater", "live underwater"):
                                is_aquatic = attrs.get("is_aquatic", False)
                                if is_aquatic:
                                    return [
                                        {"text": f"Yes, {entity_name_kb} can swim. It is aquatic.",
                                         "source": "yes_no", "score": 0.92},
                                    ]
                                else:
                                    # Most mammals can swim — only deny for non-mammals without legs
                                    has_fur = attrs.get("has_fur", False)
                                    has_tail = attrs.get("has_tail", False)
                                    if has_fur or has_tail:
                                        return [
                                            {"text": f"Yes, {entity_name_kb} can swim. While not aquatic, most mammals with legs can swim.",
                                             "source": "yes_no", "score": 0.91},
                                        ]
                                    else:
                                        return [
                                            {"text": f"No, {entity_name_kb} is not aquatic and cannot swim.",
                                             "source": "yes_no", "score": 0.91},
                                        ]

                        # Check descriptions for relevant info
                        for desc in descs:
                            desc_lower = desc.lower()
                            attr_words = set(attribute_phrase.split()) - STOP_WORDS - {"a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "should", "has", "have", "had"}
                            desc_words = set(desc_lower.split()) - STOP_WORDS - {"a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "should", "has", "have", "had"}
                            if len(attr_words & desc_words) >= 1:
                                if "not" in desc_lower or "no " in desc_lower:
                                    return [
                                        {"text": f"No. {desc.strip()}",
                                         "source": "yes_no", "score": 0.91},
                                    ]
                                else:
                                    return [
                                        {"text": f"Yes. {desc.strip()}",
                                         "source": "yes_no", "score": 0.91},
                                    ]

                        # Check dataset QA — only match if the user's entity word appears in the QA question
                        for q, a in self.knowledge_engine.dataset_qa:
                            q_lower = q.lower()
                            if entity_name in q_lower:
                                a_lower = a.lower()
                                qa_match = False
                                for w in attribute_phrase.split():
                                    if re.search(r'\b' + re.escape(w) + r'\b', a_lower):
                                        qa_match = True
                                        break
                                if qa_match:
                                    if "not" in a_lower or "no " in a_lower:
                                        return [{"text": f"No. {a.strip()}", "source": "yes_no_dataset", "score": 0.90}]
                                    else:
                                        return [{"text": f"Yes. {a.strip()}", "source": "yes_no_dataset", "score": 0.90}]

        return None

    def _handle_learn(self, query, entities):
        # Parse the learned fact
        lower = query.lower()
        for phrase in LEARN_PHRASES:
            if lower.startswith(phrase):
                fact_text = lower[len(phrase):].strip()
                # Try to parse "entity has attribute" or "entity is attribute"
                m = re.match(r"(\w+)\s+(?:has|have)\s+(.+)", fact_text)
                if m:
                    entity, attr = m.group(1), m.group(2)
                    self.knowledge_engine.register_user_fact(entity, attr, True)
                    return [{"text": f"Got it. I'll remember that {entity} has {attr}.", "source": "learn", "score": 1.0}]
                m = re.match(r"(\w+)\s+is\s+(.+)", fact_text)
                if m:
                    entity, val = m.group(1), m.group(2)
                    self.knowledge_engine.register_user_fact(entity, "is_" + val.replace(" ", "_"), val)
                    return [{"text": f"Noted. I'll remember that {entity} is {val}.", "source": "learn", "score": 1.0}]
                return [{"text": "What would you like me to remember?", "source": "learn_prompt", "score": 0.5}]
        return [{"text": "I'm not sure what you'd like me to learn.", "source": "learn_fallback", "score": 0.3}]

    def _final_score(self, candidate, query, entities, query_words):
        score = candidate["score"]

        # Source diversity bonus
        source_weights = {
            "description": 1.0, "dataset": 1.1, "attribute": 0.9,
            "comparison": 1.05, "property": 0.85, "category": 0.8,
            "context": 0.95, "compound": 1.0, "dataset_direct": 0.9
        }
        score *= source_weights.get(candidate["source"], 0.8)

        # Comparison/category candidates are only relevant if they actually
        # share words with the query — otherwise the flat bonus above lets
        # irrelevant "X and Y are similar" sentences win by default.
        if candidate["source"] in ("comparison", "category"):
            text_words = set(candidate["text"].lower().split())
            overlap = len(set(query_words) & text_words)
            if overlap == 0:
                score *= 0.3

        # Query relevance
        text_lower = candidate["text"].lower()
        for qw in query_words:
            if qw in text_lower:
                score += 0.03

        # Entity relevance
        for e in entities:
            if e in text_lower:
                score += 0.05

        return min(1.0, score)

    def simulate_and_test(self, query, candidates):
        tested = []
        for c in candidates:
            text = c["text"]
            # Check consistency with KB
            consistency = self._check_consistency(text)
            clarity = self._check_clarity(text)
            completeness = self._check_completeness(text, query)

            final_score = (c["score"] * 0.5 + consistency * 0.2 + clarity * 0.15 + completeness * 0.15)
            tested.append({**c, "tested_score": final_score, "consistency": consistency})

        tested.sort(key=lambda x: x["tested_score"], reverse=True)
        return tested

    def _check_consistency(self, text):
        words = set(text.lower().split())
        entity_names = set(self.knowledge_engine.entities.keys())
        found_entities = words & entity_names
        if not found_entities:
            return 0.7
        contradictions = 0
        for e in found_entities:
            data = self.knowledge_engine.entities.get(e, {})
            for attr, val in data.get("attributes", {}).items():
                if isinstance(val, bool):
                    if val and f"no {attr}" in text.lower():
                        contradictions += 1
                    if not val and attr in text.lower() and "not" not in text.lower() and "no" not in text.lower():
                        contradictions += 1
        if contradictions > 0:
            return max(0.2, 0.7 - contradictions * 0.2)
        return 0.9

    def _check_clarity(self, text):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.3
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_len > 25:
            return 0.5
        if avg_len < 3:
            return 0.4
        return 0.8

    def _check_completeness(self, text, query):
        query_words = set(w.lower().rstrip("?!.,;:") for w in query.split() if w.lower() not in STOP_WORDS)
        text_words = set(text.lower().split())
        if not query_words:
            return 0.5
        coverage = len(query_words & text_words) / len(query_words)
        return min(1.0, 0.3 + coverage * 0.7)

# =========================
# ENTITY STATE MACHINE
# =========================
# Tracks physical state, emotional state, position, health, and capabilities
# of entities over time. Supports healing, injury, mood changes, and
# position tracking with distance calculations.

# Body part states
BODY_PARTS = {
    "cat": ["eyes", "tail", "legs", "paws", "ears", "mouth", "nose", "whiskers", "fur"],
    "dog": ["eyes", "tail", "legs", "paws", "ears", "mouth", "nose", "fur"],
    "human": ["eyes", "arms", "legs", "hands", "head", "mouth"],
}

# Capability map: state conditions -> capability modifiers
CAPABILITY_MODIFIERS = {
    # Physical injuries
    "eye_injured": {"nudge_at_images": -0.9, "see_computer": -0.8, "follow_movement": -0.7, "help_with_visual": -0.9},
    "eye_blind": {"nudge_at_images": -1.0, "see_computer": -1.0, "follow_movement": -1.0, "help_with_visual": -1.0},
    "tail_injured": {"balance": -0.5, "communicate": -0.3},
    "tail_cut_off": {"balance": -0.7, "communicate": -0.5},
    "leg_injured": {"walk": -0.8, "climb": -0.9, "reach_food": -0.7, "play": -0.6},
    "paw_injured": {"walk": -0.5, "play": -0.7, "type": -0.9},
    "ear_injured": {"hear_commands": -0.6, "respond_to_name": -0.4},
    "mouth_injured": {"eat": -0.7, "purr": -0.8, "meow": -0.5},
    # Emotional states
    "afraid": {"help_user": -0.8, "approach_user": -0.9, "play": -0.7, "willingness": -0.8},
    "angry": {"help_user": -0.9, "approach_user": -0.8, "play": -0.6, "willingness": -0.9},
    "stressed": {"help_user": -0.5, "focus": -0.6, "willingness": -0.5},
    "happy": {"help_user": 0.3, "play": 0.3, "willingness": 0.4},
    "relaxed": {"help_user": 0.2, "play": 0.2, "willingness": 0.3},
    # Health states
    "sick": {"energy": -0.7, "help_user": -0.6, "play": -0.8, "willingness": -0.5},
    "hungry": {"energy": -0.4, "focus": -0.5, "willingness": -0.3, "may_seek_food": 0.8},
    "thirsty": {"energy": -0.3, "focus": -0.4, "willingness": -0.2},
    "tired": {"energy": -0.6, "help_user": -0.4, "play": -0.5},
    "full": {"energy": 0.2, "satisfaction": 0.3},
    "healed": {"energy": 0.3, "willingness": 0.2},
    # Environmental
    "on_desk": {"proximity_to_computer": 1.0, "may_sit_on_keyboard": 0.7, "may_knock_things": 0.6},
    "near_food_bowl": {"may_eat": 0.9, "distracted_by_food": 0.7},
    "in_cat_tree": {"safe": 0.8, "relaxed": 0.5},
    "outside": {"lost_risk": 0.3, "exploring": 0.6},
}

# Healing rates: (base_hours, factors)
# How long injuries take to heal under normal conditions
HEALING_RATES = {
    "eye_injured": (48, "slow"),      # 2 days
    "eye_blind": (168, "very_slow"),  # 1 week
    "tail_injured": (24, "medium"),
    "tail_cut_off": (720, "permanent"),  # 30 days (won't regrow)
    "leg_injured": (72, "medium"),     # 3 days
    "paw_injured": (36, "medium"),
    "ear_injured": (24, "medium"),
    "mouth_injured": (48, "slow"),
    "afraid": (2, "fast"),
    "angry": (1, "very_fast"),
    "stressed": (4, "medium"),
    "sick": (96, "slow"),
    "hungry": (0.5, "very_fast"),
    "tired": (1, "fast"),
}


class EntityState:
    """Tracks the full state of an entity: physical, emotional, position, health."""
    def __init__(self, entity_name, entity_type=None):
        self.entity_name = entity_name.lower()
        self.entity_type = entity_type or entity_name.lower()
        self.created_at = time.time()
        self.last_updated = self.created_at

        # Physical state: body_part -> condition
        self.body_parts = {}
        # Emotional state: mood -> intensity (0.0-1.0)
        self.emotions = {"neutral": 1.0}
        # Position: location -> distance_from_food (steps)
        self.position = {"location": "unknown", "distance_to_food": None, "distance_to_computer": None}
        # Health: overall percentage, specific conditions
        self.health = {
            "overall": 100.0,
            "energy": 100.0,
            "pain_level": 0.0,
            "conditions": set(),
        }
        # Capabilities: action -> modifier (-1.0 to +1.0)
        self.capabilities = {}
        # History: list of events with timestamps
        self.state_history = []
        # Interaction log: user actions toward entity
        self.interaction_log = []
        # Readiness score: how ready is this entity to help (0-100)
        self.readiness = 80.0

        # Initialize default body parts
        parts = BODY_PARTS.get(self.entity_type, BODY_PARTS.get("cat", []))
        for part in parts:
            self.body_parts[part] = {"condition": "healthy", "health": 100.0, "injury_time": None}

        # Initialize capabilities from entity type
        self._recalculate_capabilities()

    def apply_event(self, event_type, details=None, timestamp=None):
        """Apply an event that changes entity state."""
        ts = timestamp or time.time()
        details = details or {}

        event = {"type": event_type, "details": details, "timestamp": ts}
        self.state_history.append(event)
        self.last_updated = ts

        if event_type == "injury":
            part = details.get("body_part", "tail")
            severity = details.get("severity", "moderate")  # minor, moderate, severe, critical
            cause = details.get("cause", "unknown")
            self._apply_injury(part, severity, cause, ts)

        elif event_type == "heal":
            part = details.get("body_part", None)
            method = details.get("method", "natural")  # natural, vet, medicine
            self._apply_healing(part, method, ts)

        elif event_type == "emotion_change":
            emotion = details.get("emotion", "neutral")
            intensity = details.get("intensity", 0.5)
            self._apply_emotion(emotion, intensity, ts)

        elif event_type == "move":
            location = details.get("location", "unknown")
            self._apply_move(location, details, ts)

        elif event_type == "feed":
            self.health["energy"] = min(100, self.health["energy"] + 30)
            self.health["conditions"].discard("hungry")
            self._apply_emotion("happy", 0.6, ts)
            self._log_interaction("user_fed", ts)

        elif event_type == "pet":
            self._apply_emotion("happy", 0.5, ts)
            self.health["pain_level"] = max(0, self.health["pain_level"] - 10)
            self._log_interaction("user_petted", ts)

        elif event_type == "hit":
            self._apply_emotion("afraid", 0.8, ts)
            self._apply_emotion("angry", 0.6, ts)
            self.health["pain_level"] = min(100, self.health["pain_level"] + 30)
            self.health["overall"] = max(0, self.health["overall"] - 15)
            self._log_interaction("user_hit", ts)

        elif event_type == "play":
            self._apply_emotion("happy", 0.7, ts)
            self.health["energy"] = max(0, self.health["energy"] - 10)
            self._log_interaction("user_played", ts)

        elif event_type == "vet_visit":
            # Heal all injuries partially or fully
            for part, info in self.body_parts.items():
                if info["condition"] != "healthy":
                    info["condition"] = "healing"
                    info["health"] = min(100, info["health"] + 50)
            self.health["conditions"].discard("sick")
            self.health["pain_level"] = max(0, self.health["pain_level"] - 40)
            self.health["overall"] = min(100, self.health["overall"] + 20)
            self._apply_emotion("relaxed", 0.5, ts)
            self._log_interaction("vet_visit", ts)

        elif event_type == "time_pass":
            hours = details.get("hours", 1)
            self._apply_time_healing(hours, ts)

        elif event_type == "rest":
            self.health["energy"] = min(100, self.health["energy"] + 25)
            self._apply_emotion("relaxed", 0.4, ts)

        # Recalculate everything
        self._recalculate_capabilities()
        self._update_readiness()
        self._update_health()

    def _apply_injury(self, part, severity, cause, ts):
        """Apply injury to a body part."""
        if part not in self.body_parts:
            self.body_parts[part] = {"condition": "injured", "health": 50.0, "injury_time": ts}

        severity_map = {"minor": 20, "moderate": 40, "severe": 65, "critical": 90}
        damage = severity_map.get(severity, 40)

        self.body_parts[part]["condition"] = "injured" if damage < 70 else "severely_injured"
        self.body_parts[part]["health"] = max(0, self.body_parts[part]["health"] - damage)
        self.body_parts[part]["injury_time"] = ts
        self.body_parts[part]["cause"] = cause

        # If health drops to 0, it's "lost"
        if self.body_parts[part]["health"] <= 0:
            self.body_parts[part]["condition"] = "lost"
            self.health["conditions"].add(f"{part}_lost")
        else:
            self.health["conditions"].add(f"{part}_injured")

        # Pain from injury
        self.health["pain_level"] = min(100, self.health["pain_level"] + damage * 0.5)
        self.health["overall"] = max(0, self.health["overall"] - damage * 0.3)

        # Emotional response to injury
        if severity in ("severe", "critical"):
            self._apply_emotion("afraid", 0.8, ts)
            self._apply_emotion("stressed", 0.7, ts)
        elif severity == "moderate":
            self._apply_emotion("afraid", 0.4, ts)
            self._apply_emotion("stressed", 0.3, ts)

    def _apply_healing(self, part, method, ts):
        """Apply healing to a body part or all parts."""
        heal_amount = {"natural": 20, "medicine": 40, "vet": 60}
        amount = heal_amount.get(method, 20)

        if part and part in self.body_parts:
            info = self.body_parts[part]
            if info["condition"] != "healthy":
                info["health"] = min(100, info["health"] + amount)
                if info["health"] >= 80:
                    info["condition"] = "healthy"
                    self.health["conditions"].discard(f"{part}_injured")
                    self.health["conditions"].discard(f"{part}_lost")
                elif info["health"] >= 40:
                    info["condition"] = "healing"
                self._apply_emotion("relaxed", 0.3, ts)
        else:
            # Heal all injured parts
            for p, info in self.body_parts.items():
                if info["condition"] != "healthy":
                    info["health"] = min(100, info["health"] + amount)
                    if info["health"] >= 80:
                        info["condition"] = "healthy"
                        self.health["conditions"].discard(f"{p}_injured")
                        self.health["conditions"].discard(f"{p}_lost")
                    elif info["health"] >= 40:
                        info["condition"] = "healing"

    def _apply_emotion(self, emotion, intensity, ts):
        """Apply or update an emotion."""
        # Decay existing emotions
        for e in list(self.emotions.keys()):
            if e != emotion:
                self.emotions[e] *= 0.7
                if self.emotions[e] < 0.1:
                    del self.emotions[e]
        # Add/update target emotion
        if emotion in self.emotions:
            self.emotions[emotion] = min(1.0, self.emotions[emotion] + intensity)
        else:
            self.emotions[emotion] = min(1.0, intensity)
        # Ensure neutral is low when other emotions are high
        if any(v > 0.3 for k, v in self.emotions.items() if k != "neutral"):
            self.emotions["neutral"] = max(0, self.emotions.get("neutral", 0) - 0.3)

    def _apply_move(self, location, details, ts):
        """Move entity to a new location."""
        old_location = self.position.get("location", "unknown")
        self.position["location"] = location
        self.position["distance_to_food"] = details.get("distance_to_food", None)
        self.position["distance_to_computer"] = details.get("distance_to_computer", None)
        # Apply location-based state
        location_states = {
            "desk": ["on_desk"],
            "food_bowl": ["near_food_bowl"],
            "cat_tree": ["in_cat_tree"],
            "outside": ["outside"],
            "bed": ["tired", "relaxed"],
        }
        for state in location_states.get(location, []):
            if state in CAPABILITY_MODIFIERS:
                self.health["conditions"].add(state)

    def _apply_time_healing(self, hours, ts):
        """Apply natural healing over time."""
        for part, info in self.body_parts.items():
            if info["condition"] in ("injured", "severely_injured", "healing"):
                rate_info = HEALING_RATES.get(part, (24, "medium"))
                heal_per_hour = 100.0 / max(1, rate_info[0])
                info["health"] = min(100, info["health"] + heal_per_hour * hours)
                if info["health"] >= 80:
                    info["condition"] = "healthy"
                    self.health["conditions"].discard(f"{part}_injured")
                    self.health["conditions"].discard(f"{part}_lost")
                elif info["health"] >= 40:
                    info["condition"] = "healing"

        # Decay emotional states over time
        for emotion in list(self.emotions.keys()):
            if emotion != "neutral":
                decay = hours * 0.1
                self.emotions[emotion] = max(0, self.emotions[emotion] - decay)
                if self.emotions[emotion] < 0.1:
                    del self.emotions[emotion]

        # Energy recovery
        self.health["energy"] = min(100, self.health["energy"] + hours * 5)
        # Pain decay
        self.health["pain_level"] = max(0, self.health["pain_level"] - hours * 2)

    def _log_interaction(self, action, ts):
        self.interaction_log.append({"action": action, "timestamp": ts})
        if len(self.interaction_log) > 100:
            self.interaction_log.pop(0)

    def _recalculate_capabilities(self):
        """Recalculate capability modifiers from current state."""
        self.capabilities = {}
        # Base capabilities from entity type (always present)
        base_caps = {
            "cat": {"keep_company": 0.8, "watch": 0.9, "play": 0.7, "purr": 0.9,
                    "knock_things": 0.6, "sit_on_keyboard": 0.5, "nudge_at_images": 0.4,
                    "help_with_visual": 0.3, "walk": 0.9, "see_computer": 0.8,
                    "hear_commands": 0.85, "respond_to_name": 0.9, "eat": 0.9,
                    "climb": 0.7, "balance": 0.8, "communicate": 0.7, "help_user": 0.6,
                    "approach_user": 0.7, "willingness": 0.7, "focus": 0.6, "play_with_humans": 0.7},
            "dog": {"keep_company": 0.9, "fetch": 0.9, "guard": 0.8, "help_user": 0.8,
                    "willingness": 0.85, "focus": 0.7, "play_with_humans": 0.85},
        }
        self.capabilities = dict(base_caps.get(self.entity_type, base_caps.get("cat", {})))

        # Apply all active state modifiers
        for condition in self.health["conditions"]:
            if condition in CAPABILITY_MODIFIERS:
                for cap, modifier in CAPABILITY_MODIFIERS[condition].items():
                    if cap in self.capabilities:
                        self.capabilities[cap] = max(-1.0, min(1.0, self.capabilities[cap] + modifier))
                    else:
                        self.capabilities[cap] = modifier

        # Apply emotion modifiers
        for emotion, intensity in self.emotions.items():
            if emotion in CAPABILITY_MODIFIERS:
                for cap, modifier in CAPABILITY_MODIFIERS[emotion].items():
                    if cap in self.capabilities:
                        # Scale modifier by emotion intensity
                        scaled = modifier * intensity
                        self.capabilities[cap] = max(-1.0, min(1.0, self.capabilities[cap] + scaled))

    def _update_readiness(self):
        """Calculate overall readiness score (0-100)."""
        factors = []
        # Health factor
        factors.append(self.health["overall"] * 0.3)
        # Energy factor
        factors.append(self.health["energy"] * 0.2)
        # Pain penalty
        factors.append((100 - self.health["pain_level"]) * 0.15)
        # Mood factor
        positive_emotions = sum(v for k, v in self.emotions.items() if k in ("happy", "relaxed", "neutral"))
        negative_emotions = sum(v for k, v in self.emotions.items() if k in ("afraid", "angry", "stressed"))
        mood_factor = max(0, 50 + (positive_emotions - negative_emotions) * 50)
        factors.append(mood_factor * 0.2)
        # Willingness
        willingness = self.capabilities.get("willingness", 0.5)
        factors.append(max(0, willingness * 100) * 0.15)
        self.readiness = max(0, min(100, sum(factors)))

    def _update_health(self):
        """Update overall health from body parts."""
        if self.body_parts:
            part_health = [info["health"] for info in self.body_parts.values()]
            avg_part_health = sum(part_health) / len(part_health)
            # Blend with existing health
            self.health["overall"] = (self.health["overall"] * 0.7 + avg_part_health * 0.3)

    def get_status_summary(self):
        """Get a human-readable status summary."""
        lines = []
        # Health
        h = self.health["overall"]
        if h >= 80:
            lines.append(f"{self.entity_name} is in good health ({h:.0f}%)")
        elif h >= 50:
            lines.append(f"{self.entity_name} is in fair health ({h:.0f}%)")
        elif h >= 20:
            lines.append(f"{self.entity_name} is in poor health ({h:.0f}%)")
        else:
            lines.append(f"{self.entity_name} is in critical condition ({h:.0f}%)")

        # Injuries
        injuries = [f"{p} ({info['condition']})" for p, info in self.body_parts.items()
                    if info["condition"] not in ("healthy",)]
        if injuries:
            lines.append(f"Injuries: {', '.join(injuries)}")

        # Emotions
        dominant_emotion = max(self.emotions, key=self.emotions.get) if self.emotions else "neutral"
        if dominant_emotion != "neutral":
            lines.append(f"Feeling: {dominant_emotion}")

        # Position
        loc = self.position.get("location", "unknown")
        if loc != "unknown":
            lines.append(f"Location: {loc}")
            dist_food = self.position.get("distance_to_food")
            if dist_food is not None:
                lines.append(f"Distance to food: {dist_food} steps")

        # Readiness
        lines.append(f"Readiness to help: {self.readiness:.0f}%")

        # Energy
        lines.append(f"Energy: {self.health['energy']:.0f}%")

        return ". ".join(lines) + "."

    def get_capability(self, action):
        """Get capability score for an action (-1.0 to 1.0)."""
        return self.capabilities.get(action, 0.0)

    def can_do(self, action, threshold=0.3):
        """Check if entity can perform an action above threshold."""
        return self.get_capability(action) >= threshold

    def to_dict(self):
        """Serialize state for storage."""
        return {
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "body_parts": {k: {kk: vv for kk, vv in v.items() if kk != "injury_time"} for k, v in self.body_parts.items()},
            "emotions": dict(self.emotions),
            "position": dict(self.position),
            "health": {k: v for k, v in self.health.items() if k != "conditions"},
            "health_conditions": list(self.health["conditions"]),
            "capabilities": dict(self.capabilities),
            "readiness": self.readiness,
            "state_count": len(self.state_history),
            "interaction_count": len(self.interaction_log),
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize state from storage."""
        state = cls(data["entity_name"], data.get("entity_type"))
        state.body_parts = data.get("body_parts", {})
        state.emotions = data.get("emotions", {"neutral": 1.0})
        state.position = data.get("position", {"location": "unknown"})
        state.health = data.get("health", {"overall": 100, "energy": 100, "pain_level": 0})
        state.health["conditions"] = set(data.get("health_conditions", []))
        state.capabilities = data.get("capabilities", {})
        state.readiness = data.get("readiness", 80)
        return state


# =========================
# KEYWORD LINKER
# =========================
# Links related concepts: vet/doctor, heal/recover, food/hunger, etc.
# Allows AI to understand synonyms and related terms.

KEYWORD_SYNONYMS = {
    # Medical
    "vet": ["veterinarian", "animal_doctor", "pet_doctor", "doctor_for_animals"],
    "doctor": ["physician", "medic", "healthcare_provider", "clinic"],
    "hospital": ["clinic", "medical_center", "emergency_room"],
    "medicine": ["medication", "drug", "treatment", "remedy", "pills"],
    "surgery": ["operation", "procedure"],
    # Healing
    "heal": ["recover", "recuperate", "get_better", "mend", "repair"],
    "cured": ["healed", "recovered", "better", "fixed"],
    "injury": ["wound", "hurt", "damage", "harm", "trauma"],
    "hurt": ["injured", "wounded", "damaged", "pain"],
    # Food
    "food": ["meal", "feed", "nourishment", "sustenance", "chow"],
    "eat": ["feed", "consume", "have_food", "dine"],
    "hungry": ["starving", "famished", "appetite", "want_food"],
    "thirsty": ["dehydrated", "need_water", "want_drink"],
    "water": ["drink", "hydration", "liquid"],
    # Emotions
    "afraid": ["scared", "frightened", "terrified", "fearful", "fear"],
    "angry": ["mad", "furious", "irritated", "annoyed", "pissed_off"],
    "happy": ["glad", "joyful", "content", "pleased", "delighted"],
    "sad": ["unhappy", "depressed", "down", "melancholy"],
    "stressed": ["anxious", "worried", "nervous", "tense"],
    "relaxed": ["calm", "peaceful", "chill", "at_ease"],
    # Actions
    "play": ["fun", "玩耍", "玩耍", "entertainment", "game"],
    "help": ["assist", "support", "aid", "guide", "cooperate"],
    "work": ["labor", "task", "job", "project", "assignment"],
    "rest": ["sleep", "nap", "relax", "take_break", "休息"],
    # Animals
    "cat": ["feline", "kitten", " kitty", "puss", "pussy", "tabby"],
    "dog": ["canine", "puppy", "hound", "mutt", "pooch"],
    "mammal": ["warm_blooded", "has_fur", "gives_milk"],
    "feline": ["cat_like", "cat_family"],
    "carnivore": ["meat_eater", "predator"],
    # Descriptions
    "small": ["little", "tiny", "miniature", "petite"],
    "large": ["big", "huge", "giant", "massive"],
    "fat": ["overweight", "chubby", "plump", "obese"],
    "thin": ["skinny", "slim", "lean", "underweight"],
}

# Relationship links: concept -> related concepts with relationship type
CONCEPT_LINKS = {
    "vet": {"hospital": "location", "doctor": "person", "heal": "action", "medicine": "tool"},
    "doctor": {"hospital": "location", "vet": "specialist", "heal": "action", "medicine": "tool"},
    "cat": {"mammal": "is_a", "feline": "is_a", "carnivore": "is_a", "small": "size",
            "play": "enjoys", "food": "needs", "rest": "needs", "help_user": "can_do"},
    "injury": {"hurt": "synonym", "pain": "symptom", "heal": "remedy", "vet": "treatment"},
    "heal": {"recover": "synonym", "rest": "method", "medicine": "tool", "time": "factor"},
}


class KeywordLinker:
    """Links related keywords and concepts for smarter reasoning."""
    def __init__(self):
        self.synonyms = dict(KEYWORD_SYNONYMS)
        self.concept_links = dict(CONCEPT_LINKS)
        self.link_log = []  # Track what was linked

    def are_related(self, word1, word2):
        """Check if two words are related (synonyms or concept links)."""
        w1 = word1.lower().strip()
        w2 = word2.lower().strip()
        if w1 == w2:
            return True
        # Check synonyms
        for group, synonyms in self.synonyms.items():
            if w1 == group and (w2 in synonyms or w2 == group):
                return True
            if w2 == group and (w1 in synonyms or w1 == group):
                return True
            if w1 in synonyms and w2 in synonyms:
                return True
            if w1 in synonyms and w2 == group:
                return True
            if w2 in synonyms and w1 == group:
                return True
        # Check concept links
        if w1 in self.concept_links:
            for linked, rel in self.concept_links[w1].items():
                if w2 == linked or w2 in self.synonyms.get(linked, []):
                    return True
        if w2 in self.concept_links:
            for linked, rel in self.concept_links[w2].items():
                if w1 == linked or w1 in self.synonyms.get(linked, []):
                    return True
        return False

    def get_related(self, word, max_results=5):
        """Get related words for a given word."""
        w = word.lower().strip()
        related = set()
        # From synonyms
        for group, syns in self.synonyms.items():
            if w == group:
                related.update(syns)
            elif w in syns:
                related.add(group)
                related.update(s for s in syns if s != w)
        # From concept links
        if w in self.concept_links:
            for linked, rel in self.concept_links[w].items():
                related.add(linked)
                related.update(self.synonyms.get(linked, []))
        related.discard(w)
        return list(related)[:max_results]

    def link_keywords(self, text1, text2):
        """Find keyword links between two texts. Returns list of linked pairs."""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        links = []
        for w1 in words1:
            for w2 in words2:
                if self.are_related(w1, w2) and w1 != w2:
                    links.append({"word1": w1, "word2": w2, "relationship": self._get_relationship(w1, w2)})
                    self.link_log.append({"w1": w1, "w2": w2, "ts": time.time()})
        return links

    def _get_relationship(self, w1, w2):
        """Determine the relationship type between two words."""
        for group, syns in self.synonyms.items():
            if w1 == group and w2 in syns:
                return "synonym"
            if w2 == group and w1 in syns:
                return "synonym"
            if w1 in syns and w2 in syns:
                return "co_synonym"
        if w1 in self.concept_links and w2 in self.concept_links.get(w1, {}):
            return self.concept_links[w1][w2]
        if w2 in self.concept_links and w1 in self.concept_links.get(w2, {}):
            return self.concept_links[w2][w1]
        return "related"

    def expand_query(self, query):
        """Expand a query with related terms for better search."""
        words = query.lower().split()
        expanded = set(words)
        for word in words:
            related = self.get_related(word, 3)
            expanded.update(related)
        return list(expanded)


# =========================
# PERSPECTIVE MAPPER
# =========================
# Maps an entity's "perspective" over time — what it knows, feels, and can do.
# Tracks interaction patterns, learning, and evolving relationships.

class PerspectiveMapper:
    """Maps an entity's evolving perspective and relationship with the user."""
    def __init__(self):
        self.entity_perspectives = {}  # entity -> perspective data
        self.user_condition = {
            "has_computer": True,
            "computer_on": True,
            "hands_free": True,
            "has_food_for_cat": False,
            "near_cat": True,
            "friendly_to_cat": True,
            "recently_hurt_cat": False,
            "recently_fed_cat": False,
            "recently_played_with_cat": False,
            "took_cat_to_vet": False,
        }
        self.interaction_stats = defaultdict(lambda: {
            "total_interactions": 0,
            "help_count": 0,
            "play_count": 0,
            "feed_count": 0,
            "pet_count": 0,
            "hurt_count": 0,
            "vet_count": 0,
            "avg_readiness": 80.0,
            "last_interaction": None,
            "days_since_last": 0,
        })

    def get_perspective(self, entity_name):
        """Get current perspective for an entity."""
        if entity_name not in self.entity_perspectives:
            self.entity_perspectives[entity_name] = {
                "trust_level": 0.5,
                "familiarity": 0.3,
                "willingness_to_help": 0.6,
                "last_mood": "neutral",
                "relationship_quality": 0.5,
                "learned_behaviors": [],
                "preferences": {},
            }
        return self.entity_perspectives[entity_name]

    def update_from_event(self, entity_name, event_type, details=None):
        """Update perspective based on an event."""
        p = self.get_perspective(entity_name)
        stats = self.interaction_stats[entity_name]
        stats["total_interactions"] += 1
        stats["last_interaction"] = time.time()

        if event_type == "user_fed":
            p["trust_level"] = min(1.0, p["trust_level"] + 0.1)
            p["willingness_to_help"] = min(1.0, p["willingness_to_help"] + 0.15)
            stats["feed_count"] += 1
            self.user_condition["recently_fed_cat"] = True

        elif event_type == "user_petted":
            p["trust_level"] = min(1.0, p["trust_level"] + 0.08)
            p["familiarity"] = min(1.0, p["familiarity"] + 0.05)
            p["relationship_quality"] = min(1.0, p["relationship_quality"] + 0.1)
            stats["pet_count"] += 1

        elif event_type == "user_played":
            p["trust_level"] = min(1.0, p["trust_level"] + 0.12)
            p["willingness_to_help"] = min(1.0, p["willingness_to_help"] + 0.2)
            p["relationship_quality"] = min(1.0, p["relationship_quality"] + 0.15)
            stats["play_count"] += 1
            self.user_condition["recently_played_with_cat"] = True

        elif event_type == "user_hit":
            p["trust_level"] = max(0, p["trust_level"] - 0.3)
            p["willingness_to_help"] = max(0, p["willingness_to_help"] - 0.4)
            p["relationship_quality"] = max(0, p["relationship_quality"] - 0.3)
            stats["hurt_count"] += 1
            self.user_condition["recently_hurt_cat"] = True
            self.user_condition["friendly_to_cat"] = False

        elif event_type == "vet_visit":
            p["trust_level"] = min(1.0, p["trust_level"] + 0.05)
            stats["vet_count"] += 1
            self.user_condition["took_cat_to_vet"] = True

        elif event_type == "help_user":
            stats["help_count"] += 1
            p["willingness_to_help"] = min(1.0, p["willingness_to_help"] + 0.05)

        # Update average readiness
        if entity_name in self.entity_perspectives:
            stats["avg_readiness"] = stats["avg_readiness"] * 0.9 + (p["willingness_to_help"] * 100) * 0.1

    def get_readiness_assessment(self, entity_name, entity_state):
        """Get a detailed readiness assessment combining state + perspective."""
        p = self.get_perspective(entity_name)
        readiness = entity_state.readiness / 100.0  # Normalize to 0-1

        # Adjust by relationship
        trust = p.get("trust_level", 0.5)
        willingness = p.get("willingness_to_help", 0.6)
        familiarity = p.get("familiarity", 0.3)

        # User condition effects
        user = self.user_condition
        if user.get("recently_hurt_cat"):
            readiness *= 0.3  # Major penalty
        if user.get("recently_fed_cat"):
            readiness = min(1.0, readiness + 0.1)
        if user.get("recently_played_with_cat"):
            readiness = min(1.0, readiness + 0.15)
        if not user.get("has_food_for_cat") and entity_state.health.get("conditions", set()) and "hungry" in entity_state.health["conditions"]:
            readiness *= 0.5

        # Final score
        final = (readiness * 0.4 + trust * 0.3 + willingness * 0.2 + familiarity * 0.1)
        return max(0, min(1.0, final))

    def get_response_modifier(self, entity_name, entity_state):
        """Get modifier for response tone based on perspective."""
        p = self.get_perspective(entity_name)
        if p.get("trust_level", 0.5) < 0.3:
            return "cautious"
        if p.get("willingness_to_help", 0.6) < 0.3:
            return "reluctant"
        if entity_state.health["overall"] < 30:
            return "weak"
        if any(v > 0.5 for k, v in entity_state.emotions.items() if k in ("happy", "relaxed")):
            return "cheerful"
        if any(v > 0.5 for k, v in entity_state.emotions.items() if k in ("afraid", "stressed")):
            return "anxious"
        return "neutral"

    def get_usage_stats(self, entity_name):
        """Get interaction statistics for an entity."""
        return dict(self.interaction_stats.get(entity_name, {}))


# =========================
# BEHAVIOR TRACKER
# =========================
# Tracks movement patterns, leaving probability, proximity to user,
# and predicts behavior based on health, mood, and environment.

# Leaving triggers: conditions that make entity want to leave
LEAVING_TRIGGERS = {
    "health_below_40": {"leaving_prob": 0.7, "reason": "needs rest"},
    "health_below_60": {"leaving_prob": 0.4, "reason": "may tire easily"},
    "afraid": {"leaving_prob": 0.8, "reason": "scared"},
    "angry": {"leaving_prob": 0.6, "reason": "upset"},
    "stressed": {"leaving_prob": 0.5, "reason": "overstimulated"},
    "hungry": {"leaving_prob": 0.6, "reason": "seeking food"},
    "userRecentlyHit": {"leaving_prob": 0.9, "reason": "avoiding user"},
    "outsideDoorOpen": {"leaving_prob": 0.7, "reason": "wanting to explore"},
    "noFoodNearby": {"leaving_prob": 0.5, "reason": "searching for food"},
    "coldEnvironment": {"leaving_prob": 0.4, "reason": "seeking warmth"},
}

# Activity suggestions based on state + location
ACTIVITY_SUGGESTIONS = {
    "cat": {
        "near_user_computer": [
            {"activity": "watch_screen", "description": "watch the screen with you", "health_cost": 0, "mood_bonus": 0.1},
            {"activity": "sit_on_keyboard", "description": "sit on the keyboard (classic cat move)", "health_cost": 0, "mood_bonus": 0.05},
            {"activity": "keep_company", "description": "keep you company while you work", "health_cost": 0, "mood_bonus": 0.15},
        ],
        "near_food_bowl": [
            {"activity": "eat", "description": "have a snack first", "health_cost": 0, "mood_bonus": 0.2, "energy_bonus": 0.3},
            {"activity": "rest_near_food", "description": "rest near the food bowl", "health_cost": -0.05, "mood_bonus": 0.1},
        ],
        "outdoor": [
            {"activity": "fish", "description": "go fishing with you at a pond", "health_cost": -0.1, "mood_bonus": 0.3},
            {"activity": "chase_birds", "description": "chase birds outside", "health_cost": -0.15, "mood_bonus": 0.25},
            {"activity": "explore", "description": "explore the yard", "health_cost": -0.1, "mood_bonus": 0.2},
            {"activity": "play_with_dog", "description": "play with a friendly dog nearby", "health_cost": -0.1, "mood_bonus": 0.3},
        ],
        "anywhere": [
            {"activity": "pet", "description": "pet it gently", "health_cost": 0, "mood_bonus": 0.2},
            {"activity": "play_with_string", "description": "play with a string or laser pointer", "health_cost": -0.05, "mood_bonus": 0.25},
            {"activity": "cuddle", "description": "cuddle with you", "health_cost": 0, "mood_bonus": 0.3},
        ],
        "health_low": [
            {"activity": "rest", "description": "rest and recover", "health_cost": -0.2, "mood_bonus": 0.1, "energy_bonus": 0.4},
            {"activity": "eat_then_rest", "description": "eat something then rest", "health_cost": -0.15, "mood_bonus": 0.15, "energy_bonus": 0.5},
        ],
        "health_high": [
            {"activity": "play_active", "description": "play actively", "health_cost": -0.1, "mood_bonus": 0.3},
            {"activity": "run_around", "description": "run around the room", "health_cost": -0.15, "mood_bonus": 0.25},
            {"activity": "climb_cat_tree", "description": "climb the cat tree", "health_cost": -0.1, "mood_bonus": 0.2},
        ],
    },
    "dog": {
        "near_user_computer": [
            {"activity": "sit_by_feet", "description": "sit by your feet while you work", "health_cost": 0, "mood_bonus": 0.15},
            {"activity": "guard_door", "description": "guard the door", "health_cost": 0, "mood_bonus": 0.1},
        ],
        "outdoor": [
            {"activity": "fetch", "description": "play fetch", "health_cost": -0.15, "mood_bonus": 0.3},
            {"activity": "walk", "description": "go for a walk", "health_cost": -0.1, "mood_bonus": 0.25},
            {"activity": "swim", "description": "go swimming if near water", "health_cost": -0.2, "mood_bonus": 0.35},
        ],
    },
}

# Health threshold rules
HEALTH_THRESHOLDS = {
    "critical": {"max": 20, "can_help": False, "needs_vet": True, "suggestion": "cat needs immediate medical attention"},
    "very_low": {"max": 40, "can_help": False, "needs_vet": True, "suggestion": "cat should see a vet soon"},
    "low": {"max": 55, "can_help": "limited", "needs_vet": False, "suggestion": "cat may need assistance helping you"},
    "moderate": {"max": 70, "can_help": True, "needs_vet": False, "suggestion": "cat can help but less frequently"},
    "good": {"max": 85, "can_help": True, "needs_vet": False, "suggestion": "cat is in good condition"},
    "excellent": {"max": 100, "can_help": True, "needs_vet": False, "suggestion": "cat is in great condition and ready to help"},
}


class BehaviorTracker:
    """Tracks entity movement patterns, leaving probability, and predicts behavior."""
    def __init__(self):
        self.movement_history = defaultdict(list)  # entity -> [(location, timestamp, distance_to_user)]
        self.leaving_events = defaultdict(list)  # entity -> [(timestamp, reason, success)]
        self.proximity_history = defaultdict(list)  # entity -> [(distance, timestamp)]
        self.user_position = {"location": "desk", "distance_to_entities": {}}
        self.obstacle_map = {}  # location -> obstacles
        self.outdoor_access = True  # can entity go outside?

    def track_movement(self, entity_name, new_location, distance_to_user, distance_to_food):
        """Track entity movement."""
        ts = time.time()
        history = self.movement_history[entity_name.lower()]
        history.append({
            "location": new_location,
            "timestamp": ts,
            "distance_to_user": distance_to_user,
            "distance_to_food": distance_to_food,
        })
        if len(history) > 200:
            history.pop(0)

        # Track proximity
        self.proximity_history[entity_name.lower()].append({
            "distance": distance_to_user,
            "timestamp": ts,
        })
        if len(self.proximity_history[entity_name.lower()]) > 100:
            self.proximity_history[entity_name.lower()].pop(0)

    def predict_leaving(self, entity_name, entity_state, user_context=None):
        """Predict probability of entity leaving."""
        entity_lower = entity_name.lower()
        prob = 0.1  # base probability
        reasons = []

        # Health factor
        health = entity_state.health["overall"]
        if health < 40:
            prob += 0.4
            reasons.append("low health")
        elif health < 60:
            prob += 0.2
            reasons.append("moderate health")

        # Emotion factor
        for emotion, intensity in entity_state.emotions.items():
            trigger = LEAVING_TRIGGERS.get(emotion)
            if trigger and intensity > 0.3:
                prob += trigger["leaving_prob"] * intensity * 0.3
                reasons.append(trigger["reason"])

        # User behavior factor
        if user_context:
            if user_context.get("recently_hurt_cat"):
                prob += 0.5
                reasons.append("avoiding user after being hurt")
            if user_context.get("recently_fed_cat"):
                prob -= 0.2
                reasons.append("recently fed, more content")
            if user_context.get("recently_played_with_cat"):
                prob -= 0.15
                reasons.append("recently played, more bonded")

        # Environmental factor
        if self.outdoor_access and health > 60:
            prob += 0.15
            reasons.append("outdoor access available")
        elif self.outdoor_access and health > 40:
            prob += 0.1
            reasons.append("may want fresh air")

        # Food proximity (closer to food = less likely to leave)
        if entity_lower in [h[-1]["location"] for h in [self.movement_history.get(entity_lower, [])] if h]:
            last_loc = self.movement_history[entity_lower][-1] if self.movement_history[entity_lower] else {}
            if last_loc.get("distance_to_food", 99) < 2:
                prob -= 0.15
                reasons.append("near food bowl")

        # Movement pattern: if entity has been moving away from user frequently
        recent_moves = [m for m in self.movement_history.get(entity_lower, [])[-5:]
                       if m.get("distance_to_user", 0) > 5]
        if len(recent_moves) >= 3:
            prob += 0.2
            reasons.append("moving away from user frequently")

        prob = max(0.0, min(1.0, prob))
        return {"probability": prob, "reasons": reasons, "should_warn": prob > 0.5}

    def get_proximity_advice(self, entity_name, entity_state, distance_to_user):
        """Get advice based on proximity to user."""
        entity_lower = entity_name.lower()
        advice = []

        if distance_to_user <= 2:
            advice.append({
                "type": "action",
                "text": f"The {entity_lower} is close to you. You could pet it to get it in the mood to help.",
                "priority": "high",
                "action": "pet",
            })
            if entity_state.can_do("play", 0.3):
                advice.append({
                    "type": "suggestion",
                    "text": f"It's in a good position to play. Try a quick play session.",
                    "priority": "medium",
                    "action": "play",
                })
        elif distance_to_user <= 5:
            advice.append({
                "type": "suggestion",
                "text": f"The {entity_lower} is nearby but not right with you. Consider calling it over.",
                "priority": "medium",
                "action": "call",
            })
        else:
            advice.append({
                "type": "suggestion",
                "text": f"The {entity_lower} is far from you. You may want to get it and bring it closer.",
                "priority": "medium",
                "action": "retrieve",
            })
            # Check if it's heading for the door
            recent = self.movement_history.get(entity_lower, [])[-3:]
            if recent and all(m.get("distance_to_user", 0) > 5 for m in recent):
                advice.append({
                    "type": "warning",
                    "text": f"The {entity_lower} seems to be moving away. It may be heading outside.",
                    "priority": "high",
                    "action": "intercept",
                })

        return advice

    def get_obstacle_suggestions(self, entity_name, entity_state, leaving_prob):
        """Suggest obstacles or barriers if entity is likely to leave."""
        suggestions = []
        if leaving_prob > 0.5:
            health = entity_state.health["overall"]
            if health < 40:
                suggestions.append("The cat may need rest more than outdoor time. Close the door to keep it safe inside.")
            elif health < 60:
                suggestions.append("The cat is somewhat weak. A collar with a tag would help identify it if it gets out.")
            else:
                suggestions.append("The cat is healthy enough to go outside. If you want it to stay, close the door or offer a treat.")

            if self.outdoor_access:
                suggestions.append("Consider putting a collar on the cat so it can be identified if it goes outside.")
                suggestions.append("If there's a cat door, you could close it temporarily.")

        return suggestions

    def get_movement_summary(self, entity_name):
        """Get summary of entity's movement patterns."""
        history = self.movement_history.get(entity_name.lower(), [])
        if not history:
            return "No movement recorded."

        locations = [m["location"] for m in history[-10:]]
        avg_dist_user = sum(m.get("distance_to_user", 5) for m in history[-10:]) / min(10, len(history))
        avg_dist_food = sum(m.get("distance_to_food", 5) for m in history[-10:]) / min(10, len(history))

        most_visited = max(set(locations), key=locations.count) if locations else "unknown"
        return (f"Most visited: {most_visited}. "
                f"Avg distance to user: {avg_dist_user:.1f} steps. "
                f"Avg distance to food: {avg_dist_food:.1f} steps. "
                f"Total moves: {len(history)}.")

    def log_leaving_event(self, entity_name, reason, success=True):
        self.leaving_events[entity_name.lower()].append({
            "timestamp": time.time(),
            "reason": reason,
            "success": success,
        })


# =========================
# DECISION ENGINE
# =========================
# Makes decisions about what the entity should do, what the user should do,
# and generates probability-based responses. Combines health, mood, proximity,
# and environmental factors.

class DecisionEngine:
    """Makes decisions about entity activities and user suggestions."""
    def __init__(self, knowledge_engine, behavior_tracker):
        self.knowledge_engine = knowledge_engine
        self.behavior = behavior_tracker
        self.decision_log = []
        self.qa_pairs_generated = []

    def decide_activity(self, entity_name, entity_state, user_context=None, distance_to_user=5):
        """Decide what the entity should do right now."""
        entity_lower = entity_name.lower()
        health = entity_state.health["overall"]
        energy = entity_state.health["energy"]
        mood = max(entity_state.emotions, key=entity_state.emotions.get) if entity_state.emotions else "neutral"
        readiness = entity_state.readiness

        # Determine available activity categories
        categories = []
        if health < 40:
            categories.append("health_low")
        elif health > 70:
            categories.append("health_high")

        if distance_to_user <= 3:
            categories.append("near_user_computer")
        elif entity_state.position.get("location") == "food_bowl":
            categories.append("near_food_bowl")
        elif entity_state.position.get("location") in ("outside", "yard", "garden"):
            categories.append("outdoor")
        else:
            categories.append("anywhere")

        # Get activities from ACTIVITY_SUGGESTIONS
        activities = []
        entity_type = entity_state.entity_type or "cat"
        type_activities = ACTIVITY_SUGGESTIONS.get(entity_type, ACTIVITY_SUGGESTIONS.get("cat", {}))
        for cat in categories:
            for act in type_activities.get(cat, []):
                # Filter by health cost
                if health < 50 and act["health_cost"] < -0.1:
                    continue  # Don't suggest costly activities when health is low
                if energy < 30 and act.get("energy_bonus", 0) == 0 and act["health_cost"] < 0:
                    continue  # Don't suggest energy-costly activities when tired
                activities.append(act)

        # Deduplicate and score
        seen = set()
        scored = []
        for act in activities:
            if act["activity"] not in seen:
                seen.add(act["activity"])
                score = act["mood_bonus"] - abs(act["health_cost"])
                if readiness > 70:
                    score += 0.1
                scored.append({**act, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Generate decision
        if scored:
            best = scored[0]
            decision = {
                "recommended_activity": best["activity"],
                "description": best["description"],
                "health_impact": best["health_cost"],
                "mood_impact": best["mood_bonus"],
                "confidence": min(1.0, readiness / 100 + 0.2),
                "alternatives": [a["description"] for a in scored[1:3]],
            }
        else:
            decision = {
                "recommended_activity": "rest",
                "description": "rest and recover",
                "health_impact": -0.2,
                "mood_impact": 0.1,
                "confidence": 0.5,
                "alternatives": [],
            }

        self.decision_log.append({"entity": entity_lower, "decision": decision, "ts": time.time()})
        return decision

    def get_health_assessment(self, entity_state):
        """Get health-based assessment and suggestions."""
        health = entity_state.health["overall"]
        pain = entity_state.health["pain_level"]
        energy = entity_state.health["energy"]

        # Find matching threshold
        assessment = HEALTH_THRESHOLDS["excellent"]
        for level, info in HEALTH_THRESHOLDS.items():
            if health <= info["max"]:
                assessment = info
                break

        suggestions = []
        if assessment["needs_vet"]:
            suggestions.append("Take the cat to a vet as soon as possible.")
        if health < 55:
            suggestions.append("The cat may need assistance to help you.")
        if health < 40:
            suggestions.append("The cat's health is too low to help right now.")
        if energy < 30:
            suggestions.append("The cat needs rest before it can help.")
        if pain > 50:
            suggestions.append("The cat is in pain and should not be disturbed.")
        if health > 70 and energy > 50:
            suggestions.append("The cat is in good condition and ready to help.")

        # Hungry check
        conditions = entity_state.health.get("conditions", set())
        if "hungry" in conditions or health < 55:
            suggestions.append("The cat is probably hungry. Feed it first.")

        return {
            "level": assessment["suggestion"],
            "can_help": assessment["can_help"],
            "needs_vet": assessment["needs_vet"],
            "health": health,
            "energy": energy,
            "pain": pain,
            "suggestions": suggestions,
        }

    def get_food_bowl_analysis(self, entity_state, distance_to_food):
        """Analyze food bowl proximity and its effect on behavior."""
        health = entity_state.health["overall"]
        distance_to_food = distance_to_food if distance_to_food is not None else 5
        analysis = {
            "distance_to_food": distance_to_food,
            "willingness_to_help_now": 0.0,
            "future_help_probability": 0.0,
            "suggestion": "",
        }

        if distance_to_food <= 1:
            analysis["willingness_to_help_now"] = 0.3  # close to food = distracted
            analysis["future_help_probability"] = 0.8  # will be more willing after eating
            analysis["suggestion"] = "The cat is near its food bowl. It may want to eat first, then help you after."
        elif distance_to_food <= 3:
            analysis["willingness_to_help_now"] = 0.6
            analysis["future_help_probability"] = 0.7
            analysis["suggestion"] = "The cat is close to its food. It might snack before helping."
        else:
            analysis["willingness_to_help_now"] = 0.7
            analysis["future_help_probability"] = 0.6
            analysis["suggestion"] = "The cat is away from its food bowl and may be more willing to help now."

        if health < 50:
            analysis["willingness_to_help_now"] *= 0.5
            analysis["suggestion"] += " However, its health is low so it may need rest first."

        return analysis

    def get_food_bowl_moved_effect(self, entity_state, old_distance, new_distance):
        """Analyze effect of moving the food bowl."""
        moved_closer = new_distance < old_distance
        moved_away = new_distance > old_distance

        effects = {
            "mood_change": 0,
            "hunger_change": 0,
            "willingness_change": 0,
            "explanation": "",
            "leaving_probability_change": 0,
        }

        if moved_away:
            effects["mood_change"] = -0.3
            effects["hunger_change"] = 0.2
            effects["willingness_change"] = -0.2
            effects["leaving_probability_change"] = 0.15
            effects["explanation"] = ("Moving the food bowl away made the cat unhappy and hungrier. "
                                     "It will be less likely to help you and more likely to seek food elsewhere.")
            if entity_state.health["overall"] > 60:
                effects["leaving_probability_change"] += 0.1
                effects["explanation"] += " Since its health is good, it may go outside to find food."
        elif moved_closer:
            effects["mood_change"] = 0.2
            effects["hunger_change"] = -0.1
            effects["willingness_change"] = 0.15
            effects["explanation"] = ("Moving the food bowl closer made the cat happier. "
                                     "It will be more willing to help you after it eats.")

        return effects

    def generate_response(self, entity_name, entity_state, query, user_context=None, distance_to_user=5):
        """Generate a comprehensive decision-based response."""
        entity_lower = entity_name.lower()
        health = entity_state.health["overall"]
        distance_to_food = entity_state.position.get("distance_to_food", 5)

        parts = []

        # Health assessment
        assessment = self.get_health_assessment(entity_state)
        if assessment["suggestions"]:
            parts.append(assessment["suggestions"][0])

        # Activity decision
        decision = self.decide_activity(entity_name, entity_state, user_context, distance_to_user)
        if decision["confidence"] > 0.5:
            parts.append(f"Recommended: {decision['description']}.")

        # Proximity advice
        proximity_advice = self.behavior.get_proximity_advice(entity_name, entity_state, distance_to_user)
        if proximity_advice:
            parts.append(proximity_advice[0]["text"])

        # Food bowl analysis
        food_analysis = self.get_food_bowl_analysis(entity_state, distance_to_food)
        if food_analysis["distance_to_food"] <= 2 and health < 70:
            parts.append(food_analysis["suggestion"])

        # Leaving prediction
        leaving = self.behavior.predict_leaving(entity_name, entity_state, user_context)
        if leaving["should_warn"]:
            parts.append(f"The {entity_lower} may want to leave ({', '.join(leaving['reasons'][:2])}).")
            obstacle_suggestions = self.behavior.get_obstacle_suggestions(entity_name, entity_state, leaving["probability"])
            if obstacle_suggestions:
                parts.append(obstacle_suggestions[0])

        # Dataset-based suggestions (fishing, playing with dogs, etc.)
        dataset_suggestions = self._get_dataset_based_suggestions(entity_lower, entity_state, user_context)
        if dataset_suggestions:
            parts.append(dataset_suggestions)

        return " ".join(parts) if parts else f"The {entity_lower} is available to help."

    def _get_dataset_based_suggestions(self, entity_name, entity_state, user_context):
        """Get suggestions based on dataset facts and conversation history."""
        suggestions = []
        health = entity_state.health["overall"]

        # Check dataset for relevant facts
        data = self.knowledge_engine.entities.get(entity_name, {})
        attrs = data.get("attributes", {})

        # Cats and dogs relationship
        if entity_name == "cat":
            if health > 60:
                suggestions.append("Cats and dogs can sometimes play together if they get along.")
            suggestions.append("Cats enjoy human company while playing with computers.")

        # Fishing possibility
        if health > 50 and entity_state.position.get("location") in ("outside", "yard", "near_water"):
            suggestions.append("The cat could go fishing with you for fun near a pond.")

        # Indoor vs outdoor
        if entity_name == "cat" and health > 40:
            suggestions.append("Cats are mainly indoor pets but enjoy occasional outdoor time.")

        return " ".join(suggestions[:2]) if suggestions else ""

    def assess_qa_pair(self, entity_name, question, answer, user_context=None):
        """Assess if a new QA pair should be generated from behavior."""
        health = user_context.get("health", 100) if user_context else 100
        score = 0.5

        # Boost score if behavior is notable
        if health < 50:
            score += 0.2
        if user_context and user_context.get("recently_hurt_cat"):
            score += 0.15

        if score > 0.6:
            new_qa = {
                "question": question,
                "answer": answer,
                "entity": entity_name,
                "source": "behavior_generated",
                "score": score,
                "timestamp": time.time(),
            }
            self.qa_pairs_generated.append(new_qa)
            return new_qa
        return None


# =========================
# PERFORMANCE LOGGER
# =========================
# Logs entity performance, tracks what works and what doesn't,
# and generates new QA pairs from observed behavior.

class PerformanceLogger:
    """Logs and analyzes entity performance over time."""
    def __init__(self):
        self.sessions = defaultdict(list)  # entity -> [session_data]
        self.performance_metrics = defaultdict(lambda: {
            "total_help_requests": 0,
            "successful_help": 0,
            "failed_help": 0,
            "avg_readiness_at_help": 0,
            "avg_health_at_help": 0,
            "common_failure_reasons": Counter(),
            "best_activities": Counter(),
            "worst_activities": Counter(),
        })
        self.outdoor_logs = defaultdict(list)  # entity -> [outdoor_event]
        self.generated_qa = []

    def log_help_attempt(self, entity_name, success, readiness, health, activity=None, reason=None):
        """Log a help attempt."""
        metrics = self.performance_metrics[entity_name.lower()]
        metrics["total_help_requests"] += 1
        if success:
            metrics["successful_help"] += 1
            if activity:
                metrics["best_activities"][activity] += 1
        else:
            metrics["failed_help"] += 1
            if reason:
                metrics["common_failure_reasons"][reason] += 1
            if activity:
                metrics["worst_activities"][activity] += 1

        # Update averages
        total = metrics["total_help_requests"]
        metrics["avg_readiness_at_help"] = (
            (metrics["avg_readiness_at_help"] * (total - 1) + readiness) / total
        )
        metrics["avg_health_at_help"] = (
            (metrics["avg_health_at_help"] * (total - 1) + health) / total
        )

    def log_outdoor_event(self, entity_name, event_type, duration_minutes, health_change=0):
        """Log outdoor activity."""
        self.outdoor_logs[entity_name.lower()].append({
            "timestamp": time.time(),
            "type": event_type,
            "duration": duration_minutes,
            "health_change": health_change,
        })

    def log_session(self, entity_name, session_data):
        """Log a complete session."""
        self.sessions[entity_name.lower()].append({
            "timestamp": time.time(),
            **session_data,
        })

    def get_performance_summary(self, entity_name):
        """Get performance summary for an entity."""
        metrics = self.performance_metrics.get(entity_name.lower(), {})
        if not metrics or metrics["total_help_requests"] == 0:
            return f"No performance data for {entity_name} yet."

        total = metrics["total_help_requests"]
        success_rate = metrics["successful_help"] / total * 100

        parts = [
            f"Help requests: {total}",
            f"Success rate: {success_rate:.0f}%",
            f"Avg readiness when helping: {metrics['avg_readiness_at_help']:.0f}%",
            f"Avg health when helping: {metrics['avg_health_at_help']:.0f}%",
        ]

        if metrics["common_failure_reasons"]:
            top_fail = metrics["common_failure_reasons"].most_common(1)[0]
            parts.append(f"Most common failure reason: {top_fail[0]} ({top_fail[1]} times)")

        if metrics["best_activities"]:
            best = metrics["best_activities"].most_common(1)[0]
            parts.append(f"Best activity: {best[0]} ({best[1]} successes)")

        return ". ".join(parts) + "."

    def should_generate_qa(self, entity_name, event_type, context=None):
        """Determine if a new QA pair should be generated."""
        metrics = self.performance_metrics.get(entity_name.lower(), {})
        total = metrics.get("total_help_requests", 0)

        # Generate QA pairs for notable events
        if event_type == "help_failed" and total > 3:
            return True
        if event_type == "behavior_pattern" and total > 5:
            return True
        if event_type and "outdoor" in event_type:
            return True

        return False

    def generate_qa_from_behavior(self, entity_name, event_type, context=None):
        """Generate a new QA pair from observed behavior."""
        context = context or {}
        health = context.get("health", 100)
        readiness = context.get("readiness", 80)
        activity = context.get("activity", "help")

        if event_type == "help_failed":
            reason = context.get("reason", "low readiness")
            q = f"Can the {entity_name} help me right now?"
            a = (f"The {entity_name} cannot help right now. "
                 f"Reason: {reason}. "
                 f"Health: {health:.0f}%, Readiness: {readiness:.0f}%. "
                 f"Try again later after it rests or eats.")
        elif event_type == "outdoor_activity":
            activity = context.get("activity", "explore")
            q = f"What can the {entity_name} do outside?"
            a = (f"The {entity_name} can {activity} outside. "
                 f"Its health is {health:.0f}%. "
                 f"Remember that outdoor activities may affect its health.")
        else:
            return None

        qa = {"question": q, "answer": a, "source": "performance_generated", "timestamp": time.time()}
        self.generated_qa.append(qa)
        return qa


# =========================
# TASK PERFORMER
# =========================
# Executes tasks: check updates, run tests, manage workflows, test apps,
# perform calculations, dictate commands, create/manage todo lists.

# Task types and their execution methods
TASK_TYPES = {
    "check_update": {"description": "Check for updates in data, code, or config", "priority": "medium"},
    "run_test": {"description": "Run a test or validation", "priority": "high"},
    "test_app": {"description": "Test an application or feature", "priority": "high"},
    "calculate": {"description": "Perform a calculation", "priority": "low"},
    "research": {"description": "Research a topic using available data", "priority": "medium"},
    "improve_response": {"description": "Improve a response using new info", "priority": "high"},
    "manage_agent": {"description": "Spawn or manage a sub-agent", "priority": "medium"},
    "create_todo": {"description": "Create or update a todo list", "priority": "low"},
    "check_completion": {"description": "Check if tasks are complete", "priority": "medium"},
    "dictate_command": {"description": "Generate a command or instruction", "priority": "low"},
    "analyze_data": {"description": "Analyze data and generate insights", "priority": "medium"},
    "simulate": {"description": "Simulate an outcome", "priority": "medium"},
}


class TaskPerformer:
    """Executes tasks, manages workflows, and tracks completion."""
    def __init__(self):
        self.tasks = []  # list of task dicts
        self.completed_tasks = []
        self.failed_tasks = []
        self.todo_lists = defaultdict(list)  # context -> [todos]
        self.task_counter = 0
        self.execution_log = []

    def create_task(self, task_type, description, context=None, priority=None, depends_on=None):
        """Create a new task."""
        self.task_counter += 1
        task = {
            "id": self.task_counter,
            "type": task_type,
            "description": description,
            "context": context or {},
            "priority": priority or TASK_TYPES.get(task_type, {}).get("priority", "medium"),
            "status": "pending",
            "depends_on": depends_on or [],
            "result": None,
            "created_at": time.time(),
            "completed_at": None,
        }
        self.tasks.append(task)
        return task

    def execute_task(self, task, systems=None):
        """Execute a task and return result."""
        systems = systems or {}
        task["status"] = "running"
        result = None

        try:
            if task["type"] == "check_update":
                result = self._check_updates(task, systems)
            elif task["type"] == "run_test":
                result = self._run_test(task, systems)
            elif task["type"] == "test_app":
                result = self._test_app(task, systems)
            elif task["type"] == "calculate":
                result = self._calculate(task)
            elif task["type"] == "research":
                result = self._research(task, systems)
            elif task["type"] == "improve_response":
                result = self._improve_response(task, systems)
            elif task["type"] == "create_todo":
                result = self._create_todo(task)
            elif task["type"] == "check_completion":
                result = self._check_completion(task)
            elif task["type"] == "dictate_command":
                result = self._dictate_command(task)
            elif task["type"] == "analyze_data":
                result = self._analyze_data(task, systems)
            elif task["type"] == "simulate":
                result = self._simulate(task, systems)
            else:
                result = {"status": "unknown_task", "message": f"Unknown task type: {task['type']}"}

            task["status"] = "completed"
            task["result"] = result
            task["completed_at"] = time.time()
            self.completed_tasks.append(task)
            self.execution_log.append({"task": task["id"], "type": task["type"], "status": "completed", "ts": time.time()})

        except Exception as e:
            task["status"] = "failed"
            task["result"] = {"error": str(e)}
            self.failed_tasks.append(task)
            self.execution_log.append({"task": task["id"], "type": task["type"], "status": "failed", "error": str(e), "ts": time.time()})

        return result

    def _check_updates(self, task, systems):
        """Check for updates in various systems."""
        updates = []
        context = task.get("context", {})

        # Check dataset updates
        ke = systems.get("knowledge_engine")
        if ke:
            qa_count = len(ke.dataset_qa)
            entity_count = len(ke.entities)
            updates.append(f"Dataset: {qa_count} QA pairs, {entity_count} entities")

        # Check entity states
        entity_states = systems.get("entity_states", {})
        for name, state in entity_states.items():
            health = state.health["overall"]
            if health < 50:
                updates.append(f"{name}: health low ({health:.0f}%)")

        # Check performance
        pl = systems.get("performance_logger")
        if pl:
            for name in pl.performance_metrics:
                summary = pl.get_performance_summary(name)
                if "No performance" not in summary:
                    updates.append(f"{name}: {summary}")

        # Check conversation memory
        mem = systems.get("conversation_memory")
        if mem:
            updates.append(f"Conversation turns: {len(mem.turns)}")

        return {"status": "success", "updates": updates, "count": len(updates)}

    def _run_test(self, task, systems):
        """Run a test or validation."""
        context = task.get("context", {})
        test_type = context.get("test_type", "syntax")
        target = context.get("target", "model.py")

        if test_type == "syntax":
            try:
                import ast
                with open(target, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source)
                return {"status": "pass", "message": f"Syntax OK for {target}"}
            except SyntaxError as e:
                return {"status": "fail", "message": f"Syntax error: {e}"}

        elif test_type == "import":
            module = context.get("module", "model")
            try:
                __import__(module)
                return {"status": "pass", "message": f"Import OK: {module}"}
            except Exception as e:
                return {"status": "fail", "message": f"Import failed: {e}"}

        elif test_type == "entity_state":
            entity = context.get("entity", "cat")
            es = systems.get("entity_states", {})
            state = es.get(entity.lower())
            if state:
                readiness = state.readiness
                return {"status": "pass" if readiness > 50 else "warn",
                        "message": f"{entity} readiness: {readiness:.0f}%",
                        "readiness": readiness}
            return {"status": "fail", "message": f"No state for {entity}"}

        elif test_type == "pipeline":
            tp = systems.get("thinking_pipeline")
            query = context.get("query", "hello")
            if tp:
                results = tp.process(query)
                return {"status": "pass", "results": len(results), "query": query}
            return {"status": "fail", "message": "No pipeline available"}

        return {"status": "unknown", "message": f"Unknown test type: {test_type}"}

    def _test_app(self, task, systems):
        """Test an application or feature."""
        context = task.get("context", {})
        app_name = context.get("app_name", "unknown")
        test_cases = context.get("test_cases", [])
        results = []

        for case in test_cases:
            query = case.get("query", "")
            expected = case.get("expected", "")
            tp = systems.get("thinking_pipeline")
            if tp and query:
                actual_results = tp.process(query)
                actual = actual_results[0]["text"] if actual_results else ""
                match = expected.lower() in actual.lower() if expected else True
                results.append({"query": query, "match": match, "actual": actual[:100]})

        pass_count = sum(1 for r in results if r["match"])
        return {
            "status": "success",
            "app": app_name,
            "tests": len(results),
            "passed": pass_count,
            "failed": len(results) - pass_count,
            "details": results,
        }

    def _calculate(self, task):
        """Perform a calculation."""
        context = task.get("context", {})
        expression = context.get("expression", "0")
        try:
            # Safe evaluation of math expressions
            import ast
            import operator
            ops = {
                ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
                ast.Pow: operator.pow, ast.Mod: operator.mod,
            }
            def safe_eval(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](safe_eval(node.operand))
                return 0
            tree = ast.parse(expression, mode='eval')
            result = safe_eval(tree.body)
            return {"status": "success", "expression": expression, "result": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _research(self, task, systems):
        """Research a topic using available data."""
        context = task.get("context", {})
        topic = context.get("topic", "")
        ke = systems.get("knowledge_engine")
        findings = []

        if ke and topic:
            # Search dataset
            topic_words = topic.lower().split()
            for q, a in ke.dataset_qa:
                q_lower = q.lower()
                if any(w in q_lower for w in topic_words):
                    findings.append({"question": q, "answer": a[:200], "relevance": 0.8})
            findings.sort(key=lambda x: x["relevance"], reverse=True)

        return {"status": "success", "topic": topic, "findings": findings[:5], "count": len(findings)}

    def _improve_response(self, task, systems):
        """Improve a response using new information."""
        context = task.get("context", {})
        original = context.get("original_response", "")
        new_info = context.get("new_info", "")
        entity = context.get("entity", "")

        improved = original
        if new_info and original:
            # Append new info if not already present
            if new_info.lower() not in original.lower():
                improved = f"{original} Additionally, {new_info}"
        elif new_info:
            improved = new_info

        return {"status": "success", "original": original, "improved": improved, "new_info_added": new_info not in original}

    def _create_todo(self, task):
        """Create or update a todo list."""
        context = task.get("context", {})
        list_name = context.get("list_name", "default")
        items = context.get("items", [])
        action = context.get("action", "create")

        todo_list = self.todo_lists[list_name]
        if action == "create":
            for item in items:
                todo_list.append({"item": item, "status": "pending", "created_at": time.time()})
        elif action == "complete":
            item_name = context.get("item", "")
            for t in todo_list:
                if t["item"] == item_name:
                    t["status"] = "completed"
        elif action == "list":
            pass

        pending = [t for t in todo_list if t["status"] == "pending"]
        completed = [t for t in todo_list if t["status"] == "completed"]
        return {"status": "success", "list": list_name, "pending": len(pending), "completed": len(completed),
                "items": [{"item": t["item"], "status": t["status"]} for t in todo_list]}

    def _check_completion(self, task):
        """Check if tasks are complete."""
        context = task.get("context", {})
        list_name = context.get("list_name", "default")
        todo_list = self.todo_lists.get(list_name, [])
        pending = [t for t in todo_list if t["status"] == "pending"]
        completed = [t for t in todo_list if t["status"] == "completed"]
        total = len(todo_list)
        return {
            "status": "success",
            "list": list_name,
            "total": total,
            "pending": len(pending),
            "completed": len(completed),
            "all_done": len(pending) == 0 and total > 0,
            "pending_items": [t["item"] for t in pending],
        }

    def _dictate_command(self, task):
        """Generate a command or instruction."""
        context = task.get("context", {})
        action = context.get("action", "")
        target = context.get("target", "")
        params = context.get("params", {})

        if action == "run":
            cmd = f"python {target}"
            for k, v in params.items():
                cmd += f" --{k}={v}"
            return {"status": "success", "command": cmd, "description": f"Run {target}"}
        elif action == "test":
            return {"status": "success", "command": f"python -m pytest {target}", "description": f"Test {target}"}
        elif action == "lint":
            return {"status": "success", "command": f"python -m flake8 {target}", "description": f"Lint {target}"}

        return {"status": "success", "command": f"{action} {target}", "description": f"Do {action} on {target}"}

    def _analyze_data(self, task, systems):
        """Analyze data and generate insights."""
        context = task.get("context", {})
        data_type = context.get("data_type", "entity_states")
        entity_states = systems.get("entity_states", {})

        insights = []
        if data_type == "entity_states":
            for name, state in entity_states.items():
                health = state.health["overall"]
                readiness = state.readiness
                mood = max(state.emotions, key=state.emotions.get) if state.emotions else "neutral"
                insights.append(f"{name}: health={health:.0f}%, readiness={readiness:.0f}%, mood={mood}")
        elif data_type == "performance":
            pl = systems.get("performance_logger")
            if pl:
                for name in pl.performance_metrics:
                    insights.append(pl.get_performance_summary(name))

        return {"status": "success", "data_type": data_type, "insights": insights}

    def _simulate(self, task, systems):
        """Simulate an outcome."""
        context = task.get("context", {})
        scenario = context.get("scenario", "default")
        entity = context.get("entity", "cat")
        es = systems.get("entity_states", {})
        state = es.get(entity.lower())

        if not state:
            return {"status": "error", "message": f"No state for {entity}"}

        # Simulate various scenarios
        sim_results = {"entity": entity, "scenario": scenario, "outcomes": []}

        if scenario == "feed_then_play":
            # Simulate: feed -> play -> result
            health_before = state.health["overall"]
            energy_before = state.health["energy"]
            state.apply_event("feed")
            state.apply_event("play")
            sim_results["outcomes"] = [
                f"Before: health={health_before:.0f}%, energy={energy_before:.0f}%",
                f"After feed: health={state.health['overall']:.0f}%, energy={state.health['energy']:.0f}%",
                f"After play: readiness={state.readiness:.0f}%",
            ]
            # Undo simulation
            state.health["overall"] = health_before
            state.health["energy"] = energy_before

        elif scenario == "injury_then_heal":
            health_before = state.health["overall"]
            state.apply_event("injury", {"body_part": "tail", "severity": "moderate"})
            injured_health = state.health["overall"]
            state.apply_event("heal", {"method": "vet"})
            healed_health = state.health["overall"]
            sim_results["outcomes"] = [
                f"Before: health={health_before:.0f}%",
                f"After injury: health={injured_health:.0f}%",
                f"After vet heal: health={healed_health:.0f}%",
            ]
            state.health["overall"] = health_before

        elif scenario == "leave_then_return":
            state.apply_event("move", {"location": "outside", "distance_to_food": 10, "distance_to_computer": 15})
            outside_readiness = state.readiness
            state.apply_event("move", {"location": "desk", "distance_to_food": 5, "distance_to_computer": 1})
            returned_readiness = state.readiness
            sim_results["outcomes"] = [
                f"Outside readiness: {outside_readiness:.0f}%",
                f"Returned to desk readiness: {returned_readiness:.0f}%",
            ]

        return {"status": "success", **sim_results}

    def get_pending_tasks(self):
        return [t for t in self.tasks if t["status"] == "pending"]

    def get_task_summary(self):
        return {
            "total": len(self.tasks),
            "pending": len([t for t in self.tasks if t["status"] == "pending"]),
            "running": len([t for t in self.tasks if t["status"] == "running"]),
            "completed": len(self.completed_tasks),
            "failed": len(self.failed_tasks),
        }


# =========================
# RESPONSE IMPROVER
# =========================
# Uses new info, feedback, and testing to improve responses over time.
# Tracks what works, what doesn't, and adapts.

class ResponseImprover:
    """Improves responses using feedback, testing, and new information."""
    def __init__(self):
        self.response_history = []  # [(query, response, score, feedback)]
        self.improvement_log = []
        self.pattern_scores = defaultdict(lambda: {"attempts": 0, "successes": 0})
        self.adaptation_rules = []

    def record_response(self, query, response, source, score=0.5, feedback=None):
        """Record a response for later analysis."""
        entry = {
            "query": query,
            "response": response,
            "source": source,
            "score": score,
            "feedback": feedback,
            "timestamp": time.time(),
        }
        self.response_history.append(entry)
        if len(self.response_history) > 500:
            self.response_history.pop(0)

    def get_feedback(self, query, response, rating):
        """Get feedback on a response (1-5 scale)."""
        if self.response_history:
            last = self.response_history[-1]
            if last["query"] == query:
                last["feedback"] = rating
                last["score"] = rating / 5.0

                # Update pattern scores
                pattern = self._extract_pattern(query)
                self.pattern_scores[pattern]["attempts"] += 1
                if rating >= 4:
                    self.pattern_scores[pattern]["successes"] += 1

                return {"status": "recorded", "rating": rating}
        return {"status": "not_found"}

    def improve_response(self, query, original_response, systems=None):
        """Improve a response using historical data and rules."""
        systems = systems or {}
        improved = original_response

        # Find similar past queries and their best responses
        similar = self._find_similar(query, top_k=3)
        if similar:
            best = max(similar, key=lambda x: x["score"])
            if best["score"] > 0.7 and best["response"] != original_response:
                # Combine best elements
                improved = self._combine_responses(original_response, best["response"])

        # Apply adaptation rules
        for rule in self.adaptation_rules:
            if rule["condition"](query, improved):
                improved = rule["transform"](improved)

        # Record improvement
        if improved != original_response:
            self.improvement_log.append({
                "query": query,
                "original": original_response,
                "improved": improved,
                "timestamp": time.time(),
            })

        return improved

    def _find_similar(self, query, top_k=3):
        """Find similar past queries."""
        query_words = set(query.lower().split())
        scored = []
        for entry in self.response_history:
            entry_words = set(entry["query"].lower().split())
            overlap = len(query_words & entry_words) / max(len(query_words | entry_words), 1)
            if overlap > 0.2:
                scored.append({**entry, "similarity": overlap})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def _combine_responses(self, resp1, resp2):
        """Combine two responses intelligently, deduplicating sentences and capping length."""
        sents1 = [s + "." for s in split_sentences(resp1, min_len=5)]
        sents2 = [s + "." for s in split_sentences(resp2, min_len=5)]
        # Hard cap: never combine more than 4 sentences total
        if len(sents1) >= 4:
            return ". ".join(sents1[:4]) + "."
        # Deduplicate: only keep sentences from resp2 not already in resp1 (exact or near-duplicate)
        resp1_norms = {s.lower().strip() for s in sents1}
        new_sents = []
        for s in sents2:
            s_norm = s.lower().strip()
            if s_norm in resp1_norms:
                continue
            # Check near-duplicate with existing sentences (strict threshold)
            s_words = set(re.findall(r'\w+', s_norm))
            is_near_dup = False
            for existing in resp1_norms:
                e_words = set(re.findall(r'\w+', existing))
                if s_words and e_words:
                    overlap = len(s_words & e_words) / max(len(s_words | e_words), 1)
                    if overlap > 0.45:
                        is_near_dup = True
                        break
            if not is_near_dup:
                new_sents.append(s)
        if not new_sents:
            return ". ".join(sents1[:4]) + "."
        # Combine: take top sentences from resp1 + new ones, capped at 4
        combined_sents = sents1[:3] + new_sents[:1]
        return ". ".join(combined_sents) + "."

    def _extract_pattern(self, query):
        """Extract a pattern from a query for scoring."""
        words = query.lower().split()
        # Use first 3 meaningful words as pattern
        meaningful = [w for w in words if len(w) > 2][:3]
        return "_".join(meaningful) if meaningful else query[:20]

    def add_adaptation_rule(self, name, condition_fn, transform_fn):
        """Add a rule for adapting responses."""
        self.adaptation_rules.append({
            "name": name,
            "condition": condition_fn,
            "transform": transform_fn,
        })

    def get_improvement_stats(self):
        """Get statistics about response improvements."""
        total = len(self.response_history)
        with_feedback = sum(1 for r in self.response_history if r["feedback"] is not None)
        avg_score = sum(r["score"] for r in self.response_history) / max(total, 1)
        improvements = len(self.improvement_log)
        return {
            "total_responses": total,
            "with_feedback": with_feedback,
            "avg_score": avg_score,
            "improvements_made": improvements,
        }


# =========================
# AGENT MANAGER
# =========================
# Manages sub-agents for parallel tasks. Each agent can handle a specific
# type of task independently.

class Agent:
    """A sub-agent that handles a specific type of task."""
    def __init__(self, agent_id, agent_type, capabilities=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities or []
        self.status = "idle"
        self.current_task = None
        self.results = []
        self.created_at = time.time()

    def assign_task(self, task):
        self.current_task = task
        self.status = "busy"

    def complete_task(self, result):
        self.results.append({"task": self.current_task, "result": result, "ts": time.time()})
        self.current_task = None
        self.status = "idle"

    def can_handle(self, task_type):
        return task_type in self.capabilities or "all" in self.capabilities


# =========================
# GOAL-SEEKING EXAMPLES
# =========================
# For each entity: a minimal "seed" answer and the fuller "goal" answer
# the composer should try to approach by pulling in more facts each time
# a question repeats. Distance-to-goal is measured by word overlap.
GOAL_EXAMPLES = {
    "cat": {
        "seed": "A cat is an animal that eats.",
        "goal": ("A cat is an animal, specifically a mammal, that eats special cat food "
                 "at home and may hunt small animals like birds or mice. Cats have tails, "
                 "have lived alongside humans for thousands of years, and enjoy human company."),
    },
    "python": {
        "seed": "Python is a programming language.",
        "goal": ("Python is a high-level, interpreted programming language created by "
                 "Guido van Rossum, known for its simplicity and readability, and used for "
                 "web development, data analysis, machine learning, and automation."),
    },
    "diamond": {
        "seed": "A diamond is a gemstone.",
        "goal": ("A diamond is a precious gemstone made of pure carbon, the hardest known "
                 "natural material, valued for its brilliance, and graded by cut, color, "
                 "clarity, and carat weight."),
    },
    "dog": {
        "seed": "A dog is an animal.",
        "goal": ("A dog is a domesticated carnivorous mammal related to wolves, known for "
                 "loyalty and trainability, with breeds ranging from small companions to "
                 "large working dogs, and they communicate through barks, whines, and body language."),
    },
    "turtle": {
        "seed": "A turtle is a reptile.",
        "goal": ("A turtle is a reptile characterized by a hard protective shell, found "
                 "in both water and on land, with some species living over 100 years, "
                 "and they carry their shell on their back at all times."),
    },
    "sky": {
        "seed": "The sky is a natural phenomenon.",
        "goal": ("The sky is the region of atmosphere surrounding Earth, appearing blue "
                 "during the day due to Rayleigh scattering and black at night with "
                 "visible stars, composed mainly of nitrogen and oxygen."),
    },
}


# =========================
# REPEAT CONTEXT MEMORY
# =========================
# Tracks the last N turns of Q&A per query to diversify responses when
# the same question is asked again. Looks up to 10 turns back to find
# facts that haven't been mentioned yet.

class RepeatContextMemory:
    """Tracks recent Q&A turns to diversify repeat responses."""
    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.query_history = defaultdict(list)  # normalized_query -> [list of answer texts]

    def record(self, query, answer):
        key = re.sub(r'\s+', ' ', query.lower().strip().rstrip("?"))
        self.query_history[key].append(answer)
        if len(self.query_history[key]) > self.max_turns:
            self.query_history[key] = self.query_history[key][-self.max_turns:]

    def get_used_sentences(self, query):
        """Return all sentences that have been used for this query recently."""
        key = re.sub(r'\s+', ' ', query.lower().strip().rstrip("?"))
        history = self.query_history.get(key, [])
        used = set()
        for answer in history:
            for s in split_sentences(answer, min_len=5):
                used.add(s.lower())
        return used

    def get_recent_entities(self, query, n=3):
        """Return entity names mentioned in recent answers to this query."""
        key = re.sub(r'\s+', ' ', query.lower().strip().rstrip("?"))
        history = self.query_history.get(key, [])
        entities = []
        for answer in history[-n:]:
            # Extract capitalized words as likely entity names
            for word in answer.split():
                clean = word.strip(".,;:!?")
                if clean and clean[0].isupper() and len(clean) > 2 and clean.lower() not in STOP_WORDS:
                    if clean not in entities:
                        entities.append(clean)
        return entities

    def get_diverse_candidates(self, query, all_candidates):
        """Filter candidates to prefer ones with sentences not yet used."""
        used = self.get_used_sentences(query)
        if not used:
            return all_candidates
        diverse = []
        already_used = []
        for c in all_candidates:
            text = c.get("text", "")
            text_sents = {s.lower() for s in split_sentences(text, min_len=5)}
            overlap = len(text_sents & used)
            if overlap == 0:
                diverse.append(c)
            else:
                already_used.append(c)
        if diverse:
            return diverse + already_used[:1]
        return all_candidates


class GoalSeekingComposer:
    """When a question repeats, build a fuller answer that progresses toward
    a known 'goal' answer by pulling in additional facts each time, tracking
    how much progress has been made and how far there is left to go."""

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.progress = defaultdict(lambda: {"current": None, "steps_taken": 0, "used_candidates": set()})

    def has_goal(self, entity):
        return entity.lower() in GOAL_EXAMPLES

    def _word_overlap(self, a, b):
        wa = set(re.findall(r'\w+', a.lower()))
        wb = set(re.findall(r'\w+', b.lower()))
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / max(len(wb), 1)

    def _sentences(self, text):
        """Split text into deduplicated sentences."""
        sents = [s + "." for s in split_sentences(text, min_len=5)]
        seen = set()
        unique = []
        for s in sents:
            norm = s.lower().strip()
            if norm not in seen:
                seen.add(norm)
                unique.append(s)
        return unique

    def get_progress(self, entity):
        entity_lower = entity.lower()
        goal_data = GOAL_EXAMPLES.get(entity_lower)
        if not goal_data:
            return None
        state = self.progress[entity_lower]
        current = state["current"] or goal_data["seed"]
        fraction = self._word_overlap(current, goal_data["goal"])
        goal_words = set(re.findall(r'\w+', goal_data["goal"].lower()))
        current_words = set(re.findall(r'\w+', current.lower()))
        remaining = goal_words - current_words - STOP_WORDS
        return {
            "current": current,
            "fraction": min(1.0, fraction),
            "remaining_words": remaining,
            "steps_taken": state["steps_taken"],
        }

    def _build_context_windows(self, text, target_words, window=3):
        """For each target word in text, record nearby words as its context."""
        words = re.findall(r'\w+', text.lower())
        contexts = defaultdict(set)
        for i, w in enumerate(words):
            if w in target_words:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                contexts[w].update(words[start:end])
                contexts[w].discard(w)
        return contexts

    def _context_matches(self, candidate_text, word, goal_context_words, min_shared=1):
        """Check if word appears in candidate with similar surrounding context."""
        if not goal_context_words:
            return True
        cand_words = re.findall(r'\w+', candidate_text.lower())
        cand_context = set()
        for i, w in enumerate(cand_words):
            if w == word:
                start = max(0, i - 3)
                end = min(len(cand_words), i + 4)
                cand_context.update(cand_words[start:end])
        cand_context.discard(word)
        return len(cand_context & goal_context_words) >= min_shared

    def advance(self, entity, extra_candidates, query_words=None):
        entity_lower = entity.lower()
        info = self.get_progress(entity)
        if not info:
            return None
        if info["fraction"] >= 0.85:
            return None
        remaining = info["remaining_words"]
        state = self.progress[entity_lower]
        used = state["used_candidates"]

        goal_data = GOAL_EXAMPLES.get(entity_lower, {})
        goal_text = goal_data.get("goal", "")
        goal_context = self._build_context_windows(goal_text, remaining)

        # Find candidate that adds new info and hasn't been used
        best_candidate = None
        best_new_coverage = 0
        for c in extra_candidates:
            text = c.get("text", "")
            if not text:
                continue
            # Strip any source metadata that leaked into text
            text = re.sub(r'\[\d+\]\s*source=\S+\s*score=[\d.]+\s*', '', text)
            text = re.sub(r'source=\S+\s*score=[\d.]+\s*', '', text)
            text = re.sub(r'\[\d+\]\s*', '', text)
            text = re.sub(r'score=[\d.]+\s*', '', text)
            text = re.sub(r'Index:\s*', '', text, flags=re.IGNORECASE)
            text = text.strip()
            if not text or len(text) < 10:
                continue
            text_norm = text.lower().strip()[:80]
            if text_norm in used:
                continue
            text_words = set(re.findall(r'\w+', text.lower()))
            raw_hits = text_words & remaining
            if not raw_hits:
                continue
            # Only count a hit as "real" if it shows up in similar context
            confirmed_hits = 0
            for word in raw_hits:
                if self._context_matches(text, word, goal_context.get(word, set())):
                    confirmed_hits += 1
            new_coverage = confirmed_hits if confirmed_hits > 0 else len(raw_hits) * 0.3
            if new_coverage > best_new_coverage:
                if self._word_overlap(text, info["current"]) < 0.6:
                    best_new_coverage = new_coverage
                    best_candidate = text

        # Fallback: search wider dataset
        if not best_candidate:
            wrap_facts = self.ke.find_wrap_around_facts(info["current"], entity_lower)
            for fact in wrap_facts:
                fact_norm = fact.lower().strip()[:80]
                if fact_norm in used:
                    continue
                fact_words = set(re.findall(r'\w+', fact.lower()))
                raw_hits = fact_words & remaining
                if not raw_hits:
                    continue
                confirmed_hits = 0
                for word in raw_hits:
                    if self._context_matches(fact, word, goal_context.get(word, set())):
                        confirmed_hits += 1
                new_coverage = confirmed_hits if confirmed_hits > 0 else len(raw_hits) * 0.3
                if new_coverage > best_new_coverage:
                    best_new_coverage = new_coverage
                    best_candidate = fact

        if not best_candidate:
            return None

        # Mark as used
        used.add(best_candidate.lower().strip()[:80])

        # Combine: split into sentences, deduplicate, rejoin
        current_sents = self._sentences(info["current"])
        new_sents = self._sentences(best_candidate)
        all_sents = current_sents + new_sents
        # Final dedup — also skip near-duplicates (>70% word overlap with existing)
        seen = set()
        deduped = []
        for s in all_sents:
            norm = s.lower().strip()
            if norm in seen:
                continue
            # Check near-duplicate with already added sentences
            s_words = set(re.findall(r'\w+', norm))
            is_near_dup = False
            for existing_norm in seen:
                e_words = set(re.findall(r'\w+', existing_norm))
                if s_words and e_words:
                    overlap = len(s_words & e_words) / max(len(s_words | e_words), 1)
                    if overlap > 0.65:
                        is_near_dup = True
                        break
            if not is_near_dup:
                seen.add(norm)
                deduped.append(s)
            if len(deduped) >= 8:  # Cap total sentences
                break

        # Use connectors to smoothly join new sentences — rotate connector styles
        connectors = [
            ["Furthermore, ", "Also, ", "Moreover, "],
            ["In addition, ", "Plus, ", "On top of that, "],
            ["Notably, ", "Importantly, ", "Worth mentioning, "],
            ["", "", ""],  # No connector option for variety
        ]
        connector_words = {"furthermore", "also", "additionally", "in", "plus", "moreover", "notably", "importantly"}
        # Pick a connector style based on step count for variety
        step = self.progress[entity_lower].get("steps_taken", 0)
        conn_set = connectors[step % len(connectors)]
        if deduped:
            result_parts = [deduped[0]]
            for s in deduped[1:]:
                s_lower = s.lower().strip()
                starts_with_connector = any(s_lower.startswith(cw) for cw in connector_words)
                prev_ends_with_connector = False
                if result_parts:
                    prev_lower = result_parts[-1].lower().rstrip()
                    prev_words = prev_lower.split()
                    if prev_words and prev_words[-1].rstrip(",.") in connector_words:
                        prev_ends_with_connector = True
                if starts_with_connector or prev_ends_with_connector:
                    result_parts.append(s)
                else:
                    conn = random.choice(conn_set) if conn_set else ""
                    result_parts.append(conn + s[0].lower() + s[1:])
            combined = " ".join(result_parts)
        else:
            combined = ""
        self.progress[entity_lower]["current"] = combined
        self.progress[entity_lower]["steps_taken"] += 1
        new_fraction = self._word_overlap(combined, GOAL_EXAMPLES[entity_lower]["goal"])
        return {
            "text": combined,
            "fraction": min(1.0, new_fraction),
            "steps_taken": self.progress[entity_lower]["steps_taken"],
        }

    def reset(self, entity):
        self.progress[entity.lower()] = {"current": None, "steps_taken": 0, "used_candidates": set()}

    def to_dict(self):
        return {k: {"current": v.get("current"), "steps_taken": v.get("steps_taken", 0)}
                for k, v in self.progress.items()}

    def from_dict(self, data):
        for k, v in (data or {}).items():
            self.progress[k] = {"current": v.get("current"), "steps_taken": v.get("steps_taken", 0), "used_candidates": set()}


# =========================
# RESPONSE GENERATOR
# =========================
# Generates multiple response variations, eliminates repetition, forms custom sentences.

class ResponseGenerator:
    """Generates diverse response variations with repetition tracking and context awareness."""
    def __init__(self, knowledge_engine, memory, conclusion_engine=None):
        self.ke = knowledge_engine
        self.memory = memory
        self.used_phrases = defaultdict(int)
        self.used_facts = []
        self.max_used = 50
        self.last_answer_by_query = {}
        self.answer_repeat_count = defaultdict(int)
        self.repeat_context = RepeatContextMemory(max_turns=10)
        self.goal_composer = GoalSeekingComposer(knowledge_engine)
        self.conclusion_engine = conclusion_engine
        self.sentence_templates = {
            "definition": [
                "A {entity} is {desc}.",
                "A {entity} {desc}.",
                "The {entity} is {desc}.",
                "Known as {desc}, the {entity} is a common subject.",
            ],
            "attribute": [
                "The {attr} of a {entity} is {val}.",
                "A {entity} has a {attr} of {val}.",
                "For a {entity}, the {attr} measures {val}.",
                "The {attr}s of a {entity} are typically {val}.",
            ],
            "boolean_true": [
                "A {entity} {attr_formatted}.",
                "Yes, a {entity} {attr_formatted}.",
                "It is true that a {entity} {attr_formatted}.",
                "A {entity} {attr_formatted}.",
            ],
            "boolean_false": [
                "A {entity} {attr_formatted}.",
                "No, a {entity} {attr_formatted}.",
                "A {entity} {attr_formatted}.",
                "Unlike some others, a {entity} {attr_formatted}.",
            ],
            "capability": [
                "A {entity} can {capability}.",
                "One thing a {entity} can do is {capability}.",
                "A {entity} is able to {capability}.",
                "The {entity} has the ability to {capability}.",
            ],
            "state": [
                "The {entity} is currently {state}.",
                "Right now, the {entity} is {state}.",
                "The {entity}'s current state: {state}.",
                "As of now, the {entity} is {state}.",
            ],
            "comparison": [
                "{entity1} and {entity2} are similar in that {shared}.",
                "Both {entity1} and {entity2} share {shared}.",
                "Like {entity2}, a {entity1} is {shared}.",
                "Comparing {entity1} and {entity2}: {shared}.",
            ],
            "context": [
                "Earlier you asked about {prev_entity}. {entity} is related because {reason}.",
                "Building on our discussion of {prev_entity}: {entity} {reason}.",
                "Similar to {prev_entity}, {entity} {reason}.",
                "In the context of {prev_entity}: {entity} {reason}.",
            ],
        }
        self.repetition_window = 5

    def generate_variations(self, entity, query, candidates, num_variations=3,
                            context_clues=None, conv_history=None):
        """Generate multiple response variations from candidates."""
        if not candidates:
            return []

        variations = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        query_key = re.sub(r'\s+', ' ', query_lower.strip().rstrip("?"))
        last_answer = self.last_answer_by_query.get(query_key)

        # Diversify candidates using repeat context (prefer unused facts)
        candidates = self.repeat_context.get_diverse_candidates(query, candidates)

        # Strategy 1: Compose from top candidates (already done)
        composed = self._compose_variation(entity, candidates[:3], "composed")
        if composed:
            variations.append({
                "text": composed,
                "source": "variation_composed",
                "score": 0.85,
                "style": "composed",
            })

        # Strategy 2: Direct fact listing
        direct = self._direct_facts_variation(entity, candidates, query_words)
        if direct:
            variations.append({
                "text": direct,
                "source": "variation_direct",
                "score": 0.82,
                "style": "direct",
            })

        # Strategy 3: Context-aware response (uses conversation history)
        context_resp = self._context_variation(entity, query, candidates, conv_history, context_clues)
        if context_resp:
            variations.append({
                "text": context_resp,
                "source": "variation_context",
                "score": 0.80,
                "style": "context",
            })

        # Strategy 4: Custom sentence forming from KB
        custom = self._custom_sentence_variation(entity, query, candidates)
        if custom:
            variations.append({
                "text": custom,
                "source": "variation_custom",
                "score": 0.78,
                "style": "custom",
            })

        # Strategy 5: Dataset-backed variation
        dataset_var = self._dataset_variation(entity, query, query_words)
        if dataset_var:
            variations.append({
                "text": dataset_var,
                "source": "variation_dataset",
                "score": 0.76,
                "style": "dataset",
            })

        # Strategy 6: Paraphrased version of a random existing variation
        if variations and random.random() < 0.5:
            base_v = random.choice(variations)
            paraphrased_parts = []
            for sent in split_sentences(base_v["text"], min_len=10):
                paraphrased_parts.append(self._paraphrase_sentence(sent + "."))
            if paraphrased_parts:
                paraphrased_text = " ".join(paraphrased_parts)
                if paraphrased_text != base_v["text"] and len(paraphrased_text) > 15:
                    variations.append({
                        "text": paraphrased_text,
                        "source": "variation_paraphrase",
                        "score": 0.77,
                        "style": "paraphrase",
                    })

        # Eliminate repetition across variations
        variations = self._eliminate_repetition(variations)

        # Score and sort
        for v in variations:
            v["score"] = self._score_variation(v, query_words, entity)

        variations.sort(key=lambda x: x["score"], reverse=True)

        # Goal-seeking: enrich toward the known goal answer
        if self.goal_composer.has_goal(entity) and variations:
            info = self.goal_composer.get_progress(entity)
            if info["fraction"] < 0.85:
                if info["steps_taken"] == 0 and self.goal_composer.progress[entity.lower()]["current"] is None:
                    self.goal_composer.progress[entity.lower()]["current"] = variations[0]["text"]
                    info = self.goal_composer.get_progress(entity)
                advance_result = self.goal_composer.advance(entity, candidates, query_words)
                if advance_result:
                    variations[0] = {
                        "text": advance_result["text"],
                        "source": "goal_seeking",
                        "score": variations[0]["score"] + 0.05,
                        "goal_progress": int(advance_result["fraction"] * 100),
                        "style": variations[0].get("style", "goal"),
                    }

        # Conclusion injection: occasionally add an inferential aside
        if self.conclusion_engine:
            conclusion = self.conclusion_engine.draw_conclusion(entity)
            if conclusion and variations:
                already_present = any(
                    conclusion["text"].lower() in v["text"].lower() for v in variations
                )
                if not already_present:
                    top = variations[0]
                    variations[0] = {
                        **top,
                        "text": f"{top['text']} {conclusion['text']}",
                        "source": top["source"] + "+conclusion",
                    }

        # Repeat detection: avoid giving the same answer twice
        if last_answer and variations:
            def _too_similar(a, b, threshold=0.75):
                wa, wb = set(a.lower().split()), set(b.lower().split())
                if not wa or not wb:
                    return False
                return len(wa & wb) / max(len(wa | wb), 1) > threshold

            if _too_similar(variations[0]["text"], last_answer):
                self.answer_repeat_count[query_key] += 1
                distinct = [v for v in variations if not _too_similar(v["text"], last_answer)]
                if distinct:
                    variations = distinct + [v for v in variations if v not in distinct]
                elif self.goal_composer.has_goal(entity):
                    advance_result = self.goal_composer.advance(entity, candidates, query_words)
                    if advance_result:
                        pct = int(advance_result["fraction"] * 100)
                        variations.insert(0, {
                            "text": advance_result["text"],
                            "source": "goal_seeking",
                            "score": variations[0]["score"] + 0.05,
                            "goal_progress": pct,
                        })
                    else:
                        variations.insert(0, {
                            "text": f"As I mentioned, {variations[0]['text'][0].lower()}{variations[0]['text'][1:]} "
                                    f"I don't have additional new information on that right now.",
                            "source": "repeat_acknowledged",
                            "score": variations[0]["score"] + 0.01,
                        })
                elif self.answer_repeat_count[query_key] >= 2:
                    variations.insert(0, {
                        "text": f"As I mentioned, {variations[0]['text'][0].lower()}{variations[0]['text'][1:]} "
                                f"I don't have additional new information on that right now.",
                        "source": "repeat_acknowledged",
                        "score": variations[0]["score"] + 0.01,
                    })
            else:
                self.answer_repeat_count[query_key] = 0
                if self.goal_composer.has_goal(entity):
                    self.goal_composer.reset(entity)
        elif variations:
            self.answer_repeat_count[query_key] = 0

        if variations:
            self.last_answer_by_query[query_key] = variations[0]["text"]
            # Record in repeat context for future diversification
            self.repeat_context.record(query, variations[0]["text"])

        return variations[:num_variations]

    def _compose_variation(self, entity, candidates, style):
        """Compose a response from candidates with entity-relevance scoring and variety."""
        if not candidates:
            return None

        entity_lower = entity.lower()
        entity_pattern = re.compile(r'\b' + re.escape(entity_lower) + r'\b', re.IGNORECASE)
        parts = []
        seen_texts = set()
        seen_sentences = set()
        # Track which candidates we've used before for this entity
        used_key = f"_used_candidates_{entity_lower}"
        if not hasattr(self, used_key):
            setattr(self, used_key, [])
        prev_used = getattr(self, used_key)

        # Add KB-generated sentences as extra candidates for variety
        kb_candidates = []
        if self.ke and entity_lower in self.ke.entities:
            ent = self.ke.entities[entity_lower]
            attrs = ent.get("attributes", {})
            props = ent.get("properties", {})
            descs = ent.get("descriptions", [])
            # Pick 2 random KB facts that aren't in existing candidates
            existing_text = " ".join(c.get("text", "") for c in candidates).lower()
            all_facts = []
            for d in descs:
                if d.lower()[:30] not in existing_text:
                    all_facts.append(d)
            for a, v in attrs.items():
                if isinstance(v, bool) and v:
                    fact = f"A {entity_lower} {self.ke._format_attr_with_verb(a, negative=False, singular=True)}."
                    if fact.lower()[:30] not in existing_text:
                        all_facts.append(fact)
            for p, v in props.items():
                fact = f"The {p.replace('_',' ')} of {entity_lower} is {v}."
                if fact.lower()[:30] not in existing_text:
                    all_facts.append(fact)
            if all_facts:
                random.shuffle(all_facts)
                kb_candidates = [{"text": f, "source": "kb_direct", "score": 0.8} for f in all_facts[:2]]

        # Combine original + KB candidates
        all_candidates = list(candidates) + kb_candidates

        # Shuffle candidates for variety, but penalize previously used
        shuffled = list(all_candidates)
        random.shuffle(shuffled)
        for c in shuffled:
            text = c.get("text", "")
            if not text:
                continue
            normalized = text.lower().strip()
            if normalized in seen_texts:
                continue
            # Check sentence-level overlap with already selected parts
            text_sents = {s.lower() for s in split_sentences(text, min_len=5)}
            if text_sents & seen_sentences:
                continue
            score = c.get("score", 0.5)
            # Penalize previously used candidates
            if normalized in prev_used:
                score -= 0.3
            # Use word-boundary match for entity relevance
            if entity_pattern.search(normalized):
                score += 0.3
            elif any(w in normalized for w in entity_lower.split()):
                score += 0.1
            if score < 0.4:
                continue
            seen_texts.add(normalized)
            seen_sentences.update(text_sents)
            parts.append((text, score))

        if not parts:
            return None

        # Sort by score, take top 2-3
        parts.sort(key=lambda x: x[1], reverse=True)
        # Randomly pick 2 or 3 parts for variety
        n = random.choice([2, 3]) if len(parts) > 2 else len(parts)
        parts = [p[0] for p in parts[:n]]

        # Record what we used
        for p in parts:
            prev_used.append(p.lower().strip())
        if len(prev_used) > 20:
            prev_used[:] = prev_used[-20:]

        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            connectors = ["Additionally, ", "Also, ", "Plus, ", "Moreover, ", "In addition, "]
            return f"{parts[0]} {random.choice(connectors)}{parts[1].lower()}"
        else:
            c1 = random.choice(["Also, ", "Additionally, ", "Furthermore, ", "On top of that, "])
            c2 = random.choice(["Moreover, ", "In addition, ", "Plus, ", "Beyond that, "])
            return f"{parts[0]} {c1}{parts[1].lower()} {c2}{parts[2].lower()}"

    def _direct_facts_variation(self, entity, candidates, query_words):
        """Generate a direct fact-based response with diverse, randomized information."""
        entity_lower = entity.lower()
        data = self.ke.entities.get(entity_lower, {})
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        descs = data.get("descriptions", [])

        # Get already-used sentences from repeat_context for this entity
        used_sentences = set()
        if hasattr(self, 'repeat_context') and self.repeat_context:
            used_key = re.sub(r'\s+', ' ', entity_lower)
            used_sentences = self.repeat_context.get_used_sentences(used_key)

        # Also track what we've used in THIS call across repeats
        fact_key = f"_direct_facts_used_{entity_lower}"
        if not hasattr(self, fact_key):
            setattr(self, fact_key, [])
        facts_used_history = getattr(self, fact_key)

        parts = []

        # Rotate descriptions — skip ones used in last 3 calls
        recent_descs = set(facts_used_history[-6:])  # last 3 calls x 2 descs
        if descs:
            available_descs = [d for d in descs
                              if d not in recent_descs and d.lower() not in used_sentences]
            if not available_descs:
                available_descs = [d for d in descs if d.lower() not in used_sentences]
            if not available_descs:
                available_descs = descs
            desc = random.choice(available_descs)
            parts.append(desc)
            facts_used_history.append(desc)

        # Pick 2 random TRUE attributes (different each time)
        true_attrs = [(a, v) for a, v in attrs.items() if isinstance(v, bool) and v]
        if true_attrs:
            random.shuffle(true_attrs)
            for attr, val in true_attrs[:2]:
                formatted = self.ke._format_attr_with_verb(attr, negative=False, singular=True)
                fact_text = f"A {entity_lower} {formatted}."
                if fact_text not in recent_descs:
                    parts.append(fact_text)
                    facts_used_history.append(fact_text)

        # Pick 2 random properties (different each time)
        prop_list = list(props.items())
        if prop_list:
            random.shuffle(prop_list)
            for prop, val in prop_list[:2]:
                readable = prop.replace("_", " ")
                fact_text = f"The {readable} of a {entity_lower} is {val}."
                if fact_text not in recent_descs:
                    parts.append(fact_text)
                    facts_used_history.append(fact_text)

        # Trim history
        if len(facts_used_history) > 20:
            setattr(self, fact_key, facts_used_history[-20:])

        if not parts:
            return None

        # Compose with varied sentence structure
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            c = random.choice(["Also, ", "Additionally, ", "Moreover, ", "In addition, "])
            return f"{parts[0]} {c}{parts[1][0].lower()}{parts[1][1:]}"
        else:
            return " ".join(parts[:4])

    def _context_variation(self, entity, query, candidates, conv_history, context_clues):
        """Generate a context-aware response using conversation history."""
        entity_lower = entity.lower()
        parts = []

        # Reference previous entities from conversation
        if conv_history:
            recent_entities = []
            recent_topics = []
            for turn in conv_history[-5:]:
                for e in turn.get("entities", []):
                    if e.lower() != entity_lower and e not in recent_entities:
                        recent_entities.append(e)
                text = turn.get("text", "")
                if text:
                    recent_topics.append(text[:80])

            if recent_entities:
                prev = recent_entities[0]
                prev_data = self.ke.entities.get(prev.lower(), {})
                prev_cat = prev_data.get("category", "")
                curr_data = self.ke.entities.get(entity_lower, {})
                curr_cat = curr_data.get("category", "")

                if prev_cat and curr_cat and prev_cat == curr_cat:
                    parts.append(f"Similar to the {prev} you asked about earlier, {entity_lower} is also a {curr_cat}.")
                elif prev_cat and curr_cat:
                    parts.append(f"Unlike the {prev} ({prev_cat}), {entity_lower} is a {curr_cat}.")
                elif len(recent_entities) > 1:
                    parts.append(f"Building on our discussion about {prev} and {recent_entities[1]}:")
                else:
                    parts.append(f"Building on our earlier discussion about {prev}:")

            # Reference what the user previously asked
            if recent_topics:
                parts.append(f"Earlier you asked: \"{recent_topics[-1]}\".")

        # Reference context clues (state changes, etc.)
        if context_clues:
            for clue in context_clues[-3:]:
                if isinstance(clue, dict):
                    clue_entity = clue.get("entity", "")
                    if clue_entity.lower() == entity_lower:
                        attr = clue.get("attribute", "")
                        change = clue.get("change", "")
                        if attr and change:
                            parts.append(f"Regarding the {attr} change you mentioned: {change}.")

        # Add a candidate fact
        if candidates:
            best = candidates[0].get("text", "")
            if best and best not in " ".join(parts):
                parts.append(best)

        if not parts:
            return None

        return " ".join(parts[:3])

    def _custom_sentence_variation(self, entity, query, candidates):
        """Form custom sentences using templates and KB data."""
        entity_lower = entity.lower()
        data = self.ke.entities.get(entity_lower, {})
        attrs = data.get("attributes", {})
        props = data.get("properties", {})
        descs = data.get("descriptions", [])

        # Get already-used sentences from repeat_context
        used_sentences = set()
        if hasattr(self, 'repeat_context') and self.repeat_context:
            used_key = re.sub(r'\s+', ' ', entity_lower)
            used_sentences = self.repeat_context.get_used_sentences(used_key)

        parts = []
        query_lower = query.lower()

        # Try definition template - skip already-used descriptions
        if descs:
            available_descs = [d for d in descs if d.lower() not in used_sentences] or descs
            desc = random.choice(available_descs)
            desc_lower = desc.lower()
            if entity_lower in desc_lower:
                parts.append(desc[0].upper() + desc[1:] if desc else desc)
            else:
                template = random.choice(self.sentence_templates["definition"])
                parts.append(template.format(entity=entity_lower, desc=desc_lower))

        # Try RANDOM attribute templates, skip already-used ones
        attr_list = list(attrs.items())
        if used_sentences:
            attr_list = [(a, v) for a, v in attr_list
                         if f"a {entity_lower} {self.ke._format_attr_with_verb(a, negative=not v, singular=True).lower()}" not in " ".join(used_sentences)]
        if not attr_list:
            attr_list = list(attrs.items())
        random.shuffle(attr_list)
        for attr, val in attr_list[:2]:
            if isinstance(val, bool):
                # Use verb form for boolean templates (e.g. "has a tail", "is domestic")
                # singular=True because templates use "A {entity}" (singular subject)
                verb_form = self.ke._format_attr_with_verb(attr, negative=not val, plural=False, singular=True)
                # Bare form for "is known to" templates: "nocturnal", "a tail"
                bare_word = attr.replace("is_", "").replace("has_", "").replace("lays_", "").replace("_", " ")
                if attr.startswith("has_") and not bare_word.endswith("s"):
                    article = "a " if bare_word and bare_word[0] not in "aeiou" else ("an " if bare_word else "")
                    bare_form = f"{article}{bare_word}"
                elif attr.startswith("lays_"):
                    bare_form = bare_word
                else:
                    bare_form = bare_word
                if val:
                    template = random.choice(self.sentence_templates["boolean_true"])
                    parts.append(template.format(entity=entity_lower, attr_formatted=verb_form, attr_bare=bare_form))
                else:
                    template = random.choice(self.sentence_templates["boolean_false"])
                    parts.append(template.format(entity=entity_lower, attr_formatted=verb_form, attr_bare=bare_form))
            else:
                readable = self.ke._format_attr_name(attr)
                template = random.choice(self.sentence_templates["attribute"])
                parts.append(template.format(entity=entity_lower, attr=readable, val=val))

        # Try RANDOM property templates
        prop_list = list(props.items())
        random.shuffle(prop_list)
        for prop, val in prop_list[:1]:
            readable = prop.replace("_", " ")
            template = random.choice(self.sentence_templates["attribute"])
            parts.append(template.format(entity=entity_lower, attr=readable, val=val))

        if not parts:
            return None

        # Deduplicate and join - avoid repeating same sentence structure
        seen = set()
        unique = []
        for p in parts:
            normalized = p.lower().strip()
            # Skip if too similar to already added
            too_similar = False
            for s in seen:
                words_p = set(normalized.split())
                words_s = set(s.split())
                if words_p and words_s:
                    overlap = len(words_p & words_s) / max(len(words_p | words_s), 1)
                    if overlap > 0.7:
                        too_similar = True
                        break
            if not too_similar:
                seen.add(normalized)
                unique.append(p)

        return " ".join(unique[:3])

    def _paraphrase_sentence(self, sentence):
        """Paraphrase a sentence by shuffling clause order, swapping structures, etc."""
        s = sentence.strip()
        if not s or len(s) < 10:
            return s

        # Pattern: "A {entity} is {adj} and {adj}" → "The {adj} and {adj} nature of a {entity}..."
        # Pattern: "A {entity} has {noun}" → "The {noun} of a {entity} is..."

        # Strategy 1: Move trailing clause to front
        # "A cat is a mammal known for its agility" → "Known for its agility, a cat is a mammal"
        m = re.match(r'^(A \w+ is .+?) (known for|belonging to|characterized by|valued for|recognized for) (.+)$', s, re.IGNORECASE)
        if m:
            base, connector, detail = m.groups()
            return f"{str(connector).title()} {detail.rstrip('.')}, {str(base).lower()}."

        # Strategy 2: "A {entity} has {X}" → "The {X} of a {entity} is notable"
        m = re.match(r'^(A|An|The) (\w+) has (a |an |the )?(.+?)(?:\.|$)', s, re.IGNORECASE)
        if m:
            article, entity, _, noun = m.groups()
            return f"The {noun.strip()} of {article.lower()} {entity} is one of its notable features."

        # Strategy 3: "A {entity} is {X}" → "One thing about {entity} is that {it} {is/are} {X}"
        m = re.match(r'^(A|An|The) (\w+) is (?:a |an |the )?(.+?)(?:\.|$)', s, re.IGNORECASE)
        if m:
            article, entity, desc = m.groups()
            if len(desc.split()) <= 5:
                return f"One characteristic of {article.lower()} {entity} is that it is {desc.rstrip('.')}."

        # Strategy 4: Swap "Additionally" / "Also" connectors
        for conn in ["Additionally, ", "Also, ", "Furthermore, ", "Moreover, ", "In addition, ", "Plus, "]:
            if s.startswith(conn):
                rest = s[len(conn):]
                new_conn = random.choice(["Also, ", "Furthermore, ", "Moreover, ", "In addition, "])
                return f"{new_conn}{rest}"

        # Strategy 5: Simple word shuffle within phrases (keep grammar)
        words = s.split()
        if len(words) > 6:
            # Find a phrase like "small domesticated feline" and shuffle it
            for i in range(len(words) - 2):
                if words[i][0].islower() and words[i+1][0].islower() and words[i+2][0].islower():
                    # These are adjectives/nouns in a phrase — shuffle them
                    chunk = words[i:i+3]
                    if all(len(w) > 2 for w in chunk):
                        shuffled = chunk[:]
                        random.shuffle(shuffled)
                        if shuffled != chunk:
                            words[i:i+3] = shuffled
                            return " ".join(words)

        return s

    def _dataset_variation(self, entity, query, query_words):
        """Generate a response backed by dataset QA pairs."""
        entity_lower = entity.lower()

        # Search dataset — require entity name as a whole word in question
        entity_pattern = re.compile(r'\b' + re.escape(entity_lower) + r'\b', re.IGNORECASE)
        matches = []
        for q, a in self.ke.dataset_qa:
            if entity_pattern.search(q):
                # Skip if question is about a sub-topic (entity + extra words after it)
                q_words = q.lower().split()
                try:
                    entity_idx = next(i for i, w in enumerate(q_words) if entity_lower in w)
                    # If entity is followed by 2+ non-stop words, it's a sub-topic
                    remaining = [w for w in q_words[entity_idx+1:] if w.rstrip("?") not in STOP_WORDS and len(w.rstrip("?")) > 2]
                    if len(remaining) >= 2:
                        continue
                except StopIteration:
                    pass
                if a and len(a) > 10:
                    matches.append((q, a, 0.8))

        if not matches:
            return None

        # Pick top 1 sentence that mentions the entity
        parts = []
        seen = set()
        for q, a, score in matches[:3]:
            sentences = split_sentences(a, min_len=10)
            for sent in sentences[:2]:
                normalized = sent.lower()
                if normalized not in seen and entity_lower in normalized:
                    seen.add(normalized)
                    parts.append(sent.rstrip(".") + ".")
                    break
            if len(parts) >= 1:
                break

        if not parts:
            return None

        return " ".join(parts)

    def _eliminate_repetition(self, variations):
        """Remove variations that are too similar to each other or recently used."""
        if not variations:
            return []

        unique = []
        seen_normalized = set()

        for v in variations:
            text = v.get("text", "")
            if not text:
                continue

            normalized = text.lower().strip()

            # Check against already selected
            too_similar = False
            for s in seen_normalized:
                words_v = set(normalized.split())
                words_s = set(s.split())
                if words_v and words_s:
                    overlap = len(words_v & words_s) / max(len(words_v | words_s), 1)
                    if overlap > 0.6:
                        too_similar = True
                        break

            if not too_similar:
                # Check against recently used phrases - cap the penalty
                phrase_penalty = 0
                for phrase, count in self.used_phrases.items():
                    if phrase in normalized:
                        phrase_penalty += count * 0.05
                phrase_penalty = min(phrase_penalty, 0.3)  # cap at 0.3

                v["repetition_penalty"] = phrase_penalty
                unique.append(v)
                seen_normalized.add(normalized)

        return unique

    def _score_variation(self, variation, query_words, entity):
        """Score a variation based on relevance, variety, and freshness."""
        text = variation.get("text", "").lower()
        base_score = variation.get("score", 0.5)

        # Start with base score
        score = base_score

        # Query relevance boost
        text_words = set(re.findall(r'\w+', text))
        if query_words:
            relevance = len(query_words & text_words) / max(len(query_words), 1)
            score += relevance * 0.1

        # Entity coverage boost
        if entity.lower() in text:
            score += 0.05

        # Freshness (light penalty for used phrases)
        penalty = variation.get("repetition_penalty", 0)
        score -= penalty * 0.3  # reduced from 1.0 to 0.3

        # Length penalty (very light)
        word_count = len(text.split())
        if word_count < 3:
            score -= 0.05
        elif word_count > 60:
            score -= 0.02

        # Ensure minimum score for any valid variation
        return max(0.3, min(1.0, score))

    def record_used(self, text):
        """Record a phrase as used for repetition tracking."""
        if not text:
            return
        words = text.lower().split()
        # Record 2-grams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            self.used_phrases[bigram] += 1

        self.used_facts.append(text)
        if len(self.used_facts) > self.max_used:
            self.used_facts.pop(0)

    def get_history_context(self, n=3):
        """Get recent conversation history as context."""
        if not self.memory:
            return []
        return self.memory.turns[-n:] if hasattr(self.memory, 'turns') else []


class AgentManager:
    """Manages sub-agents for parallel task execution."""
    def __init__(self):
        self.agents = {}
        self.task_queue = []
        self.completed_tasks = []
        self.agent_counter = 0

    def spawn_agent(self, agent_type, capabilities=None):
        """Create a new agent."""
        self.agent_counter += 1
        agent_id = f"agent_{self.agent_counter}"
        agent = Agent(agent_id, agent_type, capabilities)
        self.agents[agent_id] = agent
        return agent

    def assign_task(self, task):
        """Assign a task to an available agent."""
        # Find an agent that can handle this task type
        for agent in self.agents.values():
            if agent.status == "idle" and agent.can_handle(task["type"]):
                agent.assign_task(task)
                return agent
        # No agent available, queue the task
        self.task_queue.append(task)
        return None

    def complete_task(self, agent_id, result):
        """Mark a task as complete."""
        agent = self.agents.get(agent_id)
        if agent and agent.current_task:
            agent.complete_task(result)
            self.completed_tasks.append({"agent": agent_id, "task": agent.current_task, "result": result})
            # Check queue for pending tasks
            if self.task_queue:
                next_task = self.task_queue.pop(0)
                self.assign_task(next_task)

    def get_status(self):
        """Get status of all agents."""
        return {
            "total_agents": len(self.agents),
            "idle": sum(1 for a in self.agents.values() if a.status == "idle"),
            "busy": sum(1 for a in self.agents.values() if a.status == "busy"),
            "queued_tasks": len(self.task_queue),
            "completed": len(self.completed_tasks),
        }

    def get_agent(self, agent_id):
        return self.agents.get(agent_id)


# =========================
# CONDITION EVALUATOR
# =========================
# Evaluates internal (entity state, conversation) and external (time, environment)
# conditions to influence decision making.

class ConditionEvaluator:
    """Evaluates conditions that affect entity behavior and responses."""
    def __init__(self):
        self.internal_conditions = {}
        self.external_conditions = {
            "time_of_day": "unknown",
            "weather": "unknown",
            "environment": "indoor",
            "noise_level": "normal",
            "distractions": [],
        }
        self.condition_history = []

    def evaluate_all(self, entity_state, user_context=None, systems=None):
        """Evaluate all conditions and return a comprehensive assessment."""
        systems = systems or {}
        user_context = user_context or {}

        # Internal conditions
        internal = self._evaluate_internal(entity_state, user_context)

        # External conditions
        external = self._evaluate_external(systems)

        # Combined assessment
        assessment = {
            "internal": internal,
            "external": external,
            "overall_score": (internal["score"] + external["score"]) / 2,
            "recommendations": [],
            "warnings": [],
        }

        # Generate recommendations
        if internal["health_concern"]:
            assessment["recommendations"].append(f"Health concern: {internal['health_concern']}")
        if internal["mood_concern"]:
            assessment["warnings"].append(f"Mood concern: {internal['mood_concern']}")
        if external["environment"] == "outdoor" and entity_state.health["overall"] < 60:
            assessment["warnings"].append("Entity is outdoors with low health")

        self.condition_history.append({"assessment": assessment, "ts": time.time()})
        return assessment

    def _evaluate_internal(self, entity_state, user_context):
        """Evaluate internal conditions (entity state + user relationship)."""
        health = entity_state.health["overall"]
        energy = entity_state.health["energy"]
        pain = entity_state.health["pain_level"]
        readiness = entity_state.readiness
        mood = max(entity_state.emotions, key=entity_state.emotions.get) if entity_state.emotions else "neutral"

        score = (health / 100 * 0.3 + energy / 100 * 0.2 + (100 - pain) / 100 * 0.2 + readiness / 100 * 0.3)

        health_concern = None
        if health < 40:
            health_concern = "critical health"
        elif health < 60:
            health_concern = "low health"

        mood_concern = None
        if mood in ("afraid", "angry"):
            mood_concern = f"entity is {mood}"
        elif mood == "stressed":
            mood_concern = "entity is stressed"

        return {
            "health": health,
            "energy": energy,
            "pain": pain,
            "readiness": readiness,
            "mood": mood,
            "score": score,
            "health_concern": health_concern,
            "mood_concern": mood_concern,
        }

    def _evaluate_external(self, systems):
        """Evaluate external conditions."""
        # Determine environment from entity states
        entity_states = systems.get("entity_states", {})
        environment = "indoor"
        for name, state in entity_states.items():
            loc = state.position.get("location", "unknown")
            if loc in ("outside", "yard", "garden", "park"):
                environment = "outdoor"
                break

        # Check time (simplified)
        hour = datetime.now().hour
        if 6 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 18:
            time_of_day = "afternoon"
        elif 18 <= hour < 22:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        score = 0.8  # base score
        if environment == "outdoor":
            score -= 0.1
        if time_of_day == "night":
            score -= 0.1

        return {
            "time_of_day": time_of_day,
            "environment": environment,
            "score": max(0, score),
        }

    def set_external_condition(self, key, value):
        self.external_conditions[key] = value
        self.condition_history.append({"change": {key: value}, "ts": time.time()})


# =========================
# UPDATE CHECKER
# =========================
# Monitors data, code, and config for changes. Triggers re-evaluation.

class UpdateChecker:
    """Checks for updates and triggers re-evaluation when changes detected."""
    def __init__(self):
        self.last_check = {}
        self.update_log = []
        self.watchers = defaultdict(list)  # what -> [callback_keys]

    def check_dataset_updates(self, knowledge_engine):
        """Check if dataset has changed."""
        qa_count = len(knowledge_engine.dataset_qa)
        entity_count = len(knowledge_engine.entities)
        key = "dataset"
        previous = self.last_check.get(key, {"qa_count": 0, "entity_count": 0})

        changes = []
        if qa_count != previous.get("qa_count", 0):
            changes.append(f"QA pairs: {previous.get('qa_count', 0)} -> {qa_count}")
        if entity_count != previous.get("entity_count", 0):
            changes.append(f"Entities: {previous.get('entity_count', 0)} -> {entity_count}")

        self.last_check[key] = {"qa_count": qa_count, "entity_count": entity_count}

        if changes:
            self.update_log.append({"type": "dataset", "changes": changes, "ts": time.time()})
            return {"updated": True, "changes": changes}
        return {"updated": False}

    def check_entity_state_updates(self, entity_states):
        """Check if entity states have changed significantly."""
        key = "entity_states"
        previous = self.last_check.get(key, {})
        changes = []

        for name, state in entity_states.items():
            prev = previous.get(name, {})
            health_diff = abs(state.health["overall"] - prev.get("health", 100))
            readiness_diff = abs(state.readiness - prev.get("readiness", 80))

            if health_diff > 10:
                changes.append(f"{name} health changed by {health_diff:.0f}%")
            if readiness_diff > 10:
                changes.append(f"{name} readiness changed by {readiness_diff:.0f}%")

            self.last_check.setdefault(key, {})[name] = {
                "health": state.health["overall"],
                "readiness": state.readiness,
            }

        if changes:
            self.update_log.append({"type": "entity_state", "changes": changes, "ts": time.time()})
            return {"updated": True, "changes": changes}
        return {"updated": False}

    def check_conversation_updates(self, conversation_memory):
        """Check if conversation has new turns."""
        key = "conversation"
        turn_count = len(conversation_memory.turns)
        previous = self.last_check.get(key, {"turn_count": 0})

        if turn_count > previous.get("turn_count", 0):
            new_turns = turn_count - previous["turn_count"]
            self.last_check[key] = {"turn_count": turn_count}
            self.update_log.append({"type": "conversation", "new_turns": new_turns, "ts": time.time()})
            return {"updated": True, "new_turns": new_turns}
        return {"updated": False}

    def check_all(self, systems):
        """Check all systems for updates."""
        results = {}
        if "knowledge_engine" in systems:
            results["dataset"] = self.check_dataset_updates(systems["knowledge_engine"])
        if "entity_states" in systems:
            results["entity_states"] = self.check_entity_state_updates(systems["entity_states"])
        if "conversation_memory" in systems:
            results["conversation"] = self.check_conversation_updates(systems["conversation_memory"])

        any_updated = any(r.get("updated", False) for r in results.values())
        return {"any_updated": any_updated, "details": results}

    def get_recent_updates(self, max_age_seconds=300):
        """Get recent updates within time window."""
        cutoff = time.time() - max_age_seconds
        return [u for u in self.update_log if u["ts"] > cutoff]


# =========================
# PIPELINE INTEGRATOR
# =========================
# Unified loop that combines all systems. Processes input through every system,
# collects results, evaluates conditions, improves responses, and returns
# a comprehensive output.

class PipelineAnswerEngine:
    """
    QA-pair composition pipeline:
      1. Select top-N QA pairs matching user input
      2. Compare each against AI response samples
      3. If90%+ structural match → return immediately
      4. If <90% → rebuild by combining best sentences, test again
      5. Primary goal: answer user input directly
      6. Secondary goal: think & select new pairs → custom answer
    """

    # Structural fingerprints: templates that good answers follow
    AI_SAMPLES = [
        {"template": "A {entity} is {desc}.", "structure": ["is", "desc"], "weight": 1.0},
        {"template": "A {entity} has {attr}.", "structure": ["has", "attr"], "weight": 1.0},
        {"template": "The {entity} is known for {reason}.", "structure": ["known for", "reason"], "weight": 0.9},
        {"template": "{entity} are {adj} animals.", "structure": ["are", "adj"], "weight": 0.9},
        {"template": "A {entity} is a {type} that {verb}.", "structure": ["is a", "that", "verb"], "weight": 0.95},
        {"template": "One {entity} {verb} {obj}.", "structure": ["verb", "obj"], "weight": 0.85},
        {"template": "{entity} {verb} {obj}.", "structure": ["verb", "obj"], "weight": 0.8},
        {"template": "The {attr} of a {entity} is {val}.", "structure": ["of a", "is", "val"], "weight": 0.9},
        {"template": "In terms of {topic}, {entity} {relation}.", "structure": ["in terms of", "relation"], "weight": 0.85},
        {"template": "When it comes to {topic}, {entity} {verb}.", "structure": ["when it comes to", "verb"], "weight": 0.85},
        {"template": "{entity} {verb} {obj}, making it {adj}.", "structure": ["verb", "making it", "adj"], "weight": 0.9},
        {"template": "Known for {trait}, {entity} is {desc}.", "structure": ["known for", "is"], "weight": 0.9},
        {"template": "A {entity} can {ability}, which {reason}.", "structure": ["can", "which"], "weight": 0.9},
        {"template": "{entity} belong to {category}.", "structure": ["belong to"], "weight": 0.85},
        {"template": "The {entity} is {adj} and {adj}.", "structure": ["is", "and", "adj"], "weight": 0.9},
    ]

    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        # Pre-compute structure fingerprints for fast lookup
        self._sample_fingerprints = []
        for sample in self.AI_SAMPLES:
            fp = self._fingerprint(sample["template"])
            self._sample_fingerprints.append((fp, sample))
        self._qa_query_count = defaultdict(int)

    def _fingerprint(self, text):
        """Extract structural fingerprint from text (word-level POS-like tags)."""
        words = text.lower().split()
        fp = []
        for w in words:
            w = re.sub(r'[^a-z]', '', w)
            if not w:
                continue
            if w in ("a", "an", "the", "is", "are", "was", "were", "has", "have", "had"):
                fp.append("function")
            elif w in ("and", "or", "but", "nor", "yet", "so"):
                fp.append("conjunction")
            elif w in ("that", "which", "who", "whom", "where", "when", "how"):
                fp.append("connector")
            elif w in ("for", "to", "of", "in", "on", "at", "by", "with", "from"):
                fp.append("preposition")
            elif w.endswith("ing") or w.endswith("ed"):
                fp.append("verb")
            elif w.endswith("ly"):
                fp.append("adverb")
            elif w.endswith("ful") or w.endswith("ous") or w.endswith("ive") or w.endswith("al"):
                fp.append("adjective")
            elif w.startswith("{") and w.endswith("}"):
                fp.append("placeholder")
            else:
                fp.append("content")
        return tuple(fp)

    def _compare_structure(self, text, template_fp):
        """Compare text structure to a template fingerprint. Returns0.0-1.0 score."""
        text_fp = self._fingerprint(text)
        if not text_fp or not template_fp:
            return 0.0
        # Count matching positions (allow some positional flexibility)
        matches = 0
        t_len = len(template_fp)
        x_len = len(text_fp)
        max_len = max(t_len, x_len)
        min_len = min(t_len, x_len)
        if max_len == 0:
            return 0.0
        # Exact position matches
        for i in range(min_len):
            if text_fp[i] == template_fp[i]:
                matches += 1
        # Allow 2 extra words in text without penalty
        extra = max_len - min_len
        penalty = max(0, extra - 2) * 0.05
        score = (matches / max_len) - penalty
        return max(0.0, min(1.0, score))

    def _select_qa_pairs(self, user_input, query_words, n=8):
        """Select top-N most relevant QA pairs for the user input, entity-first."""
        input_lower = user_input.lower()
        entity_words = [w for w in query_words if w not in
            ("what", "is", "are", "the", "a", "an", "how", "do", "does",
             "can", "could", "tell", "me", "about", "describe", "explain",
             "why", "when", "where", "which", "who", "whose", "whom",
             "shape", "color", "size", "sound", "big", "small", "tall",
             "make", "have", "has", "that", "this", "there", "it",
             "you", "your", "my", "i", "we", "they", "he", "she",
             "of", "in", "on", "at", "to", "for", "with", "by",
             "and", "or", "but", "not", "very", "more", "most",
             "compare", "versus", "vs", "between", "similar", "different",
             "difference", "better", "worse", "best", "worst",
             "know", "think", "feel", "like", "want", "need",
             "speed", "weight", "height", "length", "width", "depth",
             "predator", "prey", "habitat", "diet", "food", "eat",
             "shell", "fur", "feathers", "tail", "claws", "teeth",
             "venomous", "nocturnal", "domestic", "aquatic", "hard",
             "precious", "shiny", "colorful", "loud", "quiet")]
        entity_lower = entity_words[0] if entity_words else ""
        # Extract action/topic words (the "what" part of query)
        action_words = [w for w in query_words if w in
            ("shape", "color", "size", "sound", "big", "small", "have",
             "predator", "fur", "tail", "speed", "lifespan", "weight",
             "habitat", "diet", "reproduction", "shell", "feathers",
             "venomous", "nocturnal", "domestic", "aquatic", "hard",
             "precious", "shiny", "colorful")]
        scored_pairs = []
        for qa_pair in self.ke.dataset_qa:
            q, a = qa_pair
            q_lower = q.lower()
            a_lower = a.lower()
            score = 0
            # PRIORITY1: Entity must appear in either question or answer (word-boundary)
            entity_pattern = r'\b' + re.escape(entity_lower) + r'\b' if entity_lower else None
            entity_in_q = bool(entity_pattern and re.search(entity_pattern, q_lower))
            entity_in_a = bool(entity_pattern and re.search(entity_pattern, a_lower))
            if not entity_in_q and not entity_in_a:
                # Check if any entity word appears (word-boundary match)
                if entity_words:
                    entity_found = False
                    for ew in entity_words:
                        ew_pattern = r'\b' + re.escape(ew) + r'\b'
                        if re.search(ew_pattern, q_lower) or re.search(ew_pattern, a_lower):
                            entity_found = True
                            break
                    if not entity_found:
                        continue  # Skip QA pairs with no entity match
                else:
                    continue
            # Entity match bonus
            if entity_in_q:
                score += 5.0
            if entity_in_a:
                score += 3.0
            # PRIORITY2: Action/topic word match
            action_match = 0
            for aw in action_words:
                if aw in q_lower:
                    action_match += 1
                    score += 2.0
                if aw in a_lower:
                    score += 1.0
            # PRIORITY3: Question word overlap (excluding entity words)
            q_words = set(re.findall(r'\w+', q_lower))
            input_set = set(query_words) - set(entity_words)
            overlap = len(q_words & input_set)
            score += overlap * 0.5
            # Exact question match bonus
            q_stripped = re.sub(r'^\d+:\s*Q\s*=\s*', '', q_lower).strip()
            q_stripped = re.sub(r'^Q:\s*', '', q_stripped).strip()
            if input_lower == q_lower or input_lower == q_stripped:
                score += 20.0  # Very high bonus for exact match
            elif input_lower in q_lower or q_lower in input_lower:
                score += 5.0
            # Answer length: prefer concise answers
            words = a.split()
            if len(words) <= 30:
                score += 0.5
            elif len(words) > 80:
                score -= 1.0
            scored_pairs.append((score, q, a))
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        # Rotate through pairs on repeated queries for diversity
        key = user_input.lower().strip()
        self._qa_query_count[key] += 1
        n_queries = self._qa_query_count[key]
        if n_queries > 1 and len(scored_pairs) > 2:
            # Skip previously used pairs: shift by (count-1) modulo pool size
            shift = (n_queries - 1) % min(len(scored_pairs), 4)
            top = scored_pairs[:shift] if shift < len(scored_pairs) else []
            rest = scored_pairs[shift:]
            scored_pairs = rest + top
        return scored_pairs[:n]

    def _test_against_samples(self, text):
        """Test text against AI response samples. Returns (score, best_sample)."""
        if not text or len(text.split()) < 3:
            return 0.0, None
        best_score = 0.0
        best_sample = None
        for fp, sample in self._sample_fingerprints:
            score = self._compare_structure(text, fp)
            if score > best_score:
                best_score = score
                best_sample = sample
        # Also check content quality: entity mentioned, attribute mentioned, sentence structure
        words = set(text.lower().split())
        has_article = any(w in words for w in ("a", "an", "the"))
        has_verb = any(w in words for w in ("is", "are", "was", "has", "have", "can", "do", "does"))
        has_connector = any(w in words for w in ("that", "which", "and", "for", "in", "on"))
        content_score = 0.0
        if has_article:
            content_score += 0.1
        if has_verb:
            content_score += 0.15
        if has_connector:
            content_score += 0.1
        # Sentence ending
        if text.rstrip().endswith(('.', '!', '?')):
            content_score += 0.05
        return min(1.0, best_score + content_score), best_sample

    def _compose_from_qa(self, qa_pairs, user_input, entity):
        """Compose a natural answer from selected QA pairs."""
        if not qa_pairs:
            return None
        entity_lower = entity.lower() if entity else ""
        input_lower = user_input.lower().strip().rstrip("?")
        # Detect multi-attribute queries: "shape and color of diamond"
        # Extract attribute words from query
        attr_words = [w for w in input_lower.split() if w in
            ("shape", "color", "size", "sound", "big", "small", "tall",
             "speed", "weight", "height", "predator", "fur", "tail",
             "shell", "feathers", "habitat", "diet", "hardness", "lifespan")]
        # Check for "and" joining attributes OR single attribute words
        is_multi_attr = " and " in input_lower and len(attr_words) >= 2
        has_single_attr = len(attr_words) == 1
        # Detect comparison queries: "compare cat and dog", "cat vs dog"
        is_comparison = any(w in input_lower for w in ("compare", "versus", "vs", "difference between"))
        comp_entities = []
        if is_comparison:
            # Extract the two entities being compared
            comp_entities = [w for w in input_lower.split() if w not in
                ("compare", "versus", "vs", "and", "the", "a", "an", "what", "is",
                 "the", "difference", "between", "of", "in", "to", "for")]
        # Check for exact question match first (collect all, then rotate)
        exact_matches = []
        for _, q, a in qa_pairs:
            q_clean = q.lower().strip().rstrip("?")
            if q_clean == input_lower:
                exact_matches.append(a.strip())
        if exact_matches and not is_multi_attr and not has_single_attr and not is_comparison:
            # Rotate through exact matches on repeated queries
            key = input_lower
            count = getattr(self, '_qa_compose_count', {})
            n = count.get(key, 0)
            count[key] = n + 1
            self._qa_compose_count = count
            # Rotate through available exact matches
            idx = n % len(exact_matches)
            exact_match = exact_matches[idx]
            parts = [exact_match if exact_match.endswith('.') else exact_match + '.']
            seen_words = set(re.findall(r'\w{4,}', exact_match.lower()))
            for _, q, a in qa_pairs:
                if q.lower().strip().rstrip("?") == input_lower:
                    continue
                a_words = set(re.findall(r'\w{4,}', a.lower()))
                if len(a_words & seen_words) > len(a_words) * 0.5:
                    continue
                # Only add short supplementary facts
                if len(a.split()) <= 25:
                    clean = a.strip()
                    if clean and not clean.endswith('.'):
                        clean += '.'
                    parts.append(clean)
                    seen_words.update(a_words)
                    break
            return " ".join(parts)
        if is_comparison and len(comp_entities) >= 2:
            # Comparison: find QA pairs for each entity and combine
            comp_parts = []
            used_q = set()
            for ce in comp_entities[:2]:
                for _, q, a in qa_pairs:
                    if q in used_q:
                        continue
                    if ce in q.lower():
                        clean = a.strip()
                        if clean and not clean.endswith('.'):
                            clean += '.'
                        comp_parts.append(f"{ce.title()}: {clean}")
                        used_q.add(q)
                        break
            if comp_parts:
                return " Compared to ".join(comp_parts)
        if is_multi_attr or has_single_attr:
            # Multi-attribute: find QA pairs for each attribute
            attr_parts = []
            used_q = set()
            for aw in attr_words:
                for _, q, a in qa_pairs:
                    q_lower = q.lower()
                    if q in used_q:
                        continue
                    if aw in q_lower:
                        clean = a.strip()
                        if clean and not clean.endswith('.'):
                            clean += '.'
                        attr_parts.append(clean)
                        used_q.add(q)
                        break
            if attr_parts:
                connectors = [" Additionally, ", " Also, ", " Moreover, "]
                result = attr_parts[0]
                for p in attr_parts[1:]:
                    conn = random.choice(connectors)
                    result += f"{conn}{p[0].lower()}{p[1:]}" if p[0].isupper() else f"{conn}{p}"
                return result
            # If no specific attribute QA pair found, try to extract from general answers
            if entity_lower:
                # Search all QA answers for entity + attribute-related content
                for _, q, a in qa_pairs:
                    a_lower = a.lower()
                    if entity_lower in a_lower:
                        # Check if the answer contains any attribute-related words
                        attr_keywords = {
                    "sound": ["meow", "purr", "hiss", "bark", "chirp", "sound", "noise"],
                            "big": ["cm", "inch", "tall", "height", "size", "weigh", "large", "big", "small", "tiny", "massive"],
                            "small": ["cm", "inch", "tall", "height", "size", "weigh", "small", "tiny", "large", "big"],
                            "predator": ["predator", "hunt", "prey", "carnivor", "mammal", "feline", "canine"],
                            "fur": ["fur", "hair", "coat", "fleece"],
                            "tail": ["tail", "appendage"],
                            "shell": ["shell", "carapace", "exoskeleton"],
                            "habitat": ["habitat", "live", "found", "native", "environment", "home"],
                            "diet": ["diet", "eat", "food", "feed", "consumption"],
                            "hardness": ["hard", "hardness", "scale", "mineral"],
                            "weight": ["weigh", "weight", "kg", "lb", "pound", "gram"],
                            "speed": ["speed", "fast", "quick", "velocity", "km/h", "mph"],
                        }
                        for aw in attr_words:
                            if aw in attr_keywords:
                                for kw in attr_keywords[aw]:
                                    if kw in a_lower:
                                        clean = a.strip()
                                        if clean and not clean.endswith('.'):
                                            clean += '.'
                                        return clean
                # If still no match, use KB entity descriptions
                if entity_lower in self.ke.entities:
                    descs = self.ke.entities[entity_lower].get("descriptions", [])
                    if descs:
                        # Pick the most relevant description
                        for desc in descs:
                            desc_lower = desc.lower()
                            for aw in attr_words:
                                if aw in desc_lower:
                                    clean = desc.strip()
                                    if clean and not clean.endswith('.'):
                                        clean += '.'
                                    return clean
                        # Fallback: return first description
                        clean = descs[0].strip()
                        if clean and not clean.endswith('.'):
                            clean += '.'
                        return clean
            # If no attribute match found, use KB entity descriptions directly
            if entity_lower and entity_lower in self.ke.entities:
                descs = self.ke.entities[entity_lower].get("descriptions", [])
                if descs:
                    # Pick description that matches any attribute word
                    for desc in descs:
                        desc_lower = desc.lower()
                        for aw in attr_words:
                            if aw in desc_lower:
                                clean = desc.strip()
                                if clean and not clean.endswith('.'):
                                    clean += '.'
                                return clean
                    # Fallback: return first description
                    clean = descs[0].strip()
                    if clean and not clean.endswith('.'):
                        clean += '.'
                    return clean
        # Fallback: score each QA pair for relevance to entity
        scored = []
        for _, q, a in qa_pairs:
            score = 0
            a_lower = a.lower()
            q_lower = q.lower()
            if entity_lower and entity_lower in a_lower:
                score += 3
            if entity_lower and entity_lower in q_lower:
                score += 2
            # Answer length: prefer concise answers
            words = a.split()
            if len(words) <= 30:
                score += 1
            elif len(words) <= 60:
                score += 0.5
            else:
                score -= 0.5
            scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Compose: rotate through different QA pair selections on repeat queries
        compose_key = f"_pa_compose_{entity_lower}_{input_lower}"
        compose_count = getattr(self, '_pa_compose_counts', {})
        n = compose_count.get(compose_key, 0)
        compose_count[compose_key] = n + 1
        self._pa_compose_counts = compose_count
        # Shift which pairs we pick based on count
        start_idx = n % max(1, len(scored) // 2) if len(scored) > 2 else 0
        parts = []
        seen = set()
        for _, answer in scored[start_idx:start_idx + 5]:
            # Deduplicate by content words
            answer_words = set(re.findall(r'\w{4,}', answer.lower()))
            if not answer_words:
                continue
            if seen and len(answer_words & seen) > len(answer_words) * 0.6:
                continue
            seen.update(answer_words)
            # Clean up answer
            clean = answer.strip()
            if clean and not clean.endswith('.'):
                clean += '.'
            parts.append(clean)
            if len(parts) >= 3:
                break
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        connectors = ["Additionally, ", "Also, ", "Plus, ", "Moreover, "]
        result = parts[0]
        for p in parts[1:]:
            conn = random.choice(connectors)
            result += f" {conn}{p[0].lower()}{p[1:]}" if p[0].isupper() else f" {conn}{p}"
        return result

    def _rebuild_answer(self, sentences, entity, query_words, best_match_score):
        """Rebuild answer by combining best sentences, test against samples."""
        if not sentences:
            return None
        entity_lower = entity.lower() if entity else ""
        # Score each sentence for relevance
        scored = []
        for sent in sentences:
            score = 0
            sent_lower = sent.lower()
            for w in query_words:
                if w in sent_lower:
                    score += 1
            if entity_lower and entity_lower in sent_lower:
                score += 2
            scored.append((score, sent))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Build composed answer from top sentences, trying different combinations
        best_composed = None
        best_composed_score = best_match_score
        # Track which combos we've tried
        tried = set()
        for n_parts in range(1, min(4, len(scored) + 1)):
            candidates = scored[:min(n_parts + 2, len(scored))]
            # Try a few different combos of n_parts
            combos_to_try = min(3, len(candidates))
            for combo_idx in range(combos_to_try):
                if combo_idx == 0:
                    parts = [s for _, s in scored[:n_parts]]
                else:
                    # Shuffle within a window for variety
                    window = scored[combo_idx:combo_idx + n_parts]
                    parts = [s for _, s in window]
                combo_key = tuple(parts)
                if combo_key in tried:
                    continue
                tried.add(combo_key)
                composed = " ".join(parts)
                if not composed.endswith('.'):
                    composed += '.'
                test_score, _ = self._test_against_samples(composed)
                if test_score > best_composed_score:
                    best_composed_score = test_score
                    best_composed = composed
        return best_composed if best_composed_score > best_match_score else None

    def answer(self, user_input, entity=None, query_words=None):
        """
        Main entry: answer user input by selecting QA pairs,
        comparing against samples, rebuilding if needed.
        Returns {"text": str, "score": float, "method": str, "pairs_used": int}
        """
        if not user_input or not user_input.strip():
            return None
        if query_words is None:
            query_words = re.findall(r'\w+', user_input.lower())
        if entity is None:
            # Smart entity detection: skip question words, action/topic words, keep only nouns
            stop_words = {
                "what", "is", "are", "the", "a", "an", "how", "do", "does",
                "can", "could", "tell", "me", "about", "describe", "explain",
                "why", "when", "where", "which", "who", "whose", "whom",
                "shape", "color", "size", "sound", "big", "small", "tall",
                "make", "have", "has", "that", "this", "there", "it",
                "you", "your", "my", "i", "we", "they", "he", "she",
                "of", "in", "on", "at", "to", "for", "with", "by",
                "and", "or", "but", "not", "very", "more", "most",
                "speed", "weight", "height", "length", "width", "depth",
                "temperature", "density", "mass", "volume", "force",
                "predator", "prey", "habitat", "diet", "food", "eat",
                "shell", "fur", "feathers", "tail", "claws", "teeth",
                "venomous", "nocturnal", "domestic", "aquatic", "hard",
                "precious", "shiny", "colorful", "loud", "quiet",
                "compare", "versus", "vs", "between", "similar", "different",
                "difference", "better", "worse", "best", "worst",
                "know", "think", "feel", "like", "want", "need",
            }
            entity_words = [w for w in query_words if w not in stop_words]
            entity = entity_words[0] if entity_words else ""
        entity_lower = entity.lower() if entity else ""
        # Step0: Check if this is an attribute-specific query
        input_lower = user_input.lower().strip().rstrip("?")
        attr_query_words = [w for w in input_lower.split() if w in
            ("shape", "color", "size", "sound", "big", "small", "tall",
             "speed", "weight", "height", "predator", "fur", "tail",
             "shell", "feathers", "habitat", "diet", "hardness", "lifespan")]
        is_attr_query = len(attr_query_words) >= 1
        # Step1: Select QA pairs
        qa_pairs = self._select_qa_pairs(user_input, query_words)
        if not qa_pairs:
            return None
        # Step2: Compose answer from QA pairs
        composed = self._compose_from_qa(qa_pairs, user_input, entity)
        if not composed:
            return None
        # Step3: Test against AI samples
        test_score, best_sample = self._test_against_samples(composed)
        method = "qa_composed"
        # Step3.5: If this is an attribute-specific query, verify the answer addresses it
        if is_attr_query and entity_lower:
            composed_lower = composed.lower()
            # Answer must mention the entity (word-boundary)
            entity_pat = r'\b' + re.escape(entity_lower) + r'\b'
            if not re.search(entity_pat, composed_lower):
                return None
            # Answer must address the attribute
            attr_addressed = False
            for aw in attr_query_words:
                if re.search(r'\b' + re.escape(aw) + r'\b', composed_lower):
                    attr_addressed = True
                    break
                attr_related = {
                    "sound": ["meow", "purr", "bark", "chirp", "hiss", "vocaliz", "sound", "noise"],
                    "big": ["cm", "inch", "tall", "height", "size", "weigh", "large", "massive", "huge"],
                    "small": ["cm", "inch", "tall", "height", "size", "weigh", "tiny", "miniature"],
                    "predator": ["predator", "hunt", "prey", "carnivor"],
                }
                if aw in attr_related:
                    for kw in attr_related[aw]:
                        if re.search(r'\b' + kw, composed_lower):
                            attr_addressed = True
                            break
            if not attr_addressed:
                return None
            # For attribute queries, if compose found a specific answer, don't rebuild
            return {"text": composed, "score": round(test_score, 3), "method": method, "pairs_used": min(len(qa_pairs), 3)}
        # Step3.6: If compose already has a good answer (exact match found), return it
        if composed and test_score >= 0.3:
            return {"text": composed, "score": round(test_score, 3), "method": method, "pairs_used": min(len(qa_pairs), 3)}
        # Step4: If score < 0.3, rebuild
        if test_score < 0.9:
            # Collect all sentences from QA pairs
            all_sentences = []
            for _, q, a in qa_pairs:
                all_sentences.extend(split_sentences(a, min_len=5))
            rebuilt = self._rebuild_answer(all_sentences, entity, query_words, test_score)
            if rebuilt:
                composed = rebuilt
                test_score, _ = self._test_against_samples(composed)
                method = "qa_rebuilt"
            # If still low, try adding custom KB facts
            if test_score < 0.85:
                kb_facts = []
                if entity_lower and entity_lower in self.ke.entities:
                    data = self.ke.entities[entity_lower]
                    # Boolean attributes
                    for key, val in data.get("attributes", {}).items():
                        if isinstance(val, bool) and val:
                            fact = f"A {entity_lower} {self.ke._format_attr_name(key)}."
                            kb_facts.append(fact)
                    # Properties (non-boolean)
                    for key, val in data.get("properties", {}).items():
                        if isinstance(val, str):
                            fact = f"A {entity_lower} {self.ke._format_attr_name(key)} {val}."
                            kb_facts.append(fact)
                if kb_facts:
                    # Pick top 2 KB facts that aren't already in composed
                    new_facts = []
                    composed_lower = composed.lower()
                    for fact in kb_facts:
                        fact_words = set(re.findall(r'\w{4,}', fact.lower()))
                        comp_words = set(re.findall(r'\w{4,}', composed_lower))
                        if not fact_words or len(fact_words & comp_words) < 2:
                            new_facts.append(fact)
                        if len(new_facts) >= 2:
                            break
                    if new_facts:
                        composed = composed.rstrip('.') + '. ' + ' '.join(new_facts)
                        test_score, _ = self._test_against_samples(composed)
                        method = "qa_enhanced"
        return {
            "text": composed,
            "score": round(test_score, 3),
            "method": method,
            "pairs_used": min(len(qa_pairs), 3),
        }

    def get_debug(self, user_input, entity=None, query_words=None):
        """Return detailed debug info for the answer process without recomposing."""
        if query_words is None:
            query_words = re.findall(r'\w+', user_input.lower())
        if entity is None:
            entity_words = [w for w in query_words if w not in
                ("what", "is", "are", "the", "a", "an", "how", "do", "does",
                 "can", "could", "tell", "me", "about", "describe", "explain")]
            entity = entity_words[0] if entity_words else ""
        qa_pairs = self._select_qa_pairs(user_input, query_words)
        # Don't re-compose — just report top pairs and existing info
        return {
            "entity": entity,
            "qa_pairs_found": len(qa_pairs),
            "top_qa": [(q, a[:80]) for _, q, a in qa_pairs[:5]],
            "composed": None,
            "test_score": 0.0,
            "best_sample": None,
            "passes_90": False,
        }

class PipelineIntegrator:
    """Unified pipeline that orchestrates all systems into a feedback loop."""
    def __init__(self, knowledge_engine, conversation_memory, thinking_pipeline,
                 entity_states, behavior_tracker, decision_engine, performance_logger,
                 inference_engine, keyword_linker, perspective_mapper, perspective_tracker,
                 dataset_extractor=None, retry_manager=None, intent_interpreter=None,
                 thought_tracker=None, response_refiner=None, proposal_engine=None):
        self.ke = knowledge_engine
        self.mem = conversation_memory
        self.tp = thinking_pipeline
        self.entity_states = entity_states
        self.bt = behavior_tracker
        self.de = decision_engine
        self.pl = performance_logger
        self.ie = inference_engine
        self.kw = keyword_linker
        self.pm = perspective_mapper
        self.pt = perspective_tracker
        self.dex = dataset_extractor
        self.rm = retry_manager
        self.ii = intent_interpreter
        self.tt = thought_tracker
        self.rr = response_refiner
        self.pe = proposal_engine

        # Sub-systems
        self.task_performer = TaskPerformer()
        self.response_improver = ResponseImprover()
        self.agent_manager = AgentManager()
        self.condition_evaluator = ConditionEvaluator()
        self.update_checker = UpdateChecker()

        # Spawn general-purpose agents so AgentManager is actually used
        self.general_agent_1 = self.agent_manager.spawn_agent("worker", capabilities=["all"])
        self.general_agent_2 = self.agent_manager.spawn_agent("worker", capabilities=["all"])
        self.tp._agent_manager_ref = self.agent_manager

        # Pipeline state
        self.loop_count = 0
        self.pipeline_log = []
        self.systems_ref = {
            "knowledge_engine": self.ke,
            "conversation_memory": self.mem,
            "thinking_pipeline": self.tp,
            "entity_states": self.entity_states,
            "behavior_tracker": self.bt,
            "decision_engine": self.de,
            "performance_logger": self.pl,
            "inference_engine": self.ie,
            "keyword_linker": self.kw,
            "perspective_mapper": self.pm,
            "perspective_tracker": self.pt,
            "dataset_extractor": self.dex,
            "retry_manager": self.rm,
            "intent_interpreter": self.ii,
            "thought_tracker": self.tt,
            "response_refiner": self.rr,
            "proposal_engine": self.pe,
        }

    def process(self, query, num_responses=3):
        """Full pipeline processing loop with all enhancement systems."""
        self.loop_count += 1
        pipeline_result = {
            "loop": self.loop_count,
            "query": query,
            "stages": {},
            "final_response": None,
        }

        # Stage 0: Interpret intent
        if self.ii:
            intent_entry = self.ii.add_input(query)
            pipeline_result["stages"]["intent"] = intent_entry["intent"]

        # Stage 1: Check for updates
        updates = self.update_checker.check_all(self.systems_ref)
        pipeline_result["stages"]["updates"] = updates

        # Stage 2: Evaluate conditions
        entities = self.tp._extract_entities(query)
        primary_entity = entities[0] if entities else None
        entity_state = self.entity_states.get(primary_entity.lower()) if primary_entity else None
        user_context = self.pm.user_condition if self.pm else {}

        conditions = None
        if entity_state:
            conditions = self.condition_evaluator.evaluate_all(entity_state, user_context, self.systems_ref)
        pipeline_result["stages"]["conditions"] = conditions

        # Stage 3: Run tasks if needed
        if updates.get("any_updated"):
            task = self.task_performer.create_task("check_update", "Auto-check after updates detected")
            self.task_performer.execute_task(task, self.systems_ref)
            pipeline_result["stages"]["auto_task"] = task["status"]

        # Stage 3.5: Dataset extraction
        if self.dex:
            ds_candidates = self.dex.extract_answer(query, {"entities": entities})
            if ds_candidates:
                pipeline_result["stages"]["dataset_extraction"] = len(ds_candidates)

        # Stage 4: Get base response (with retry)
        def get_base_response():
            return self.tp.process(query, num_responses)

        if self.rm:
            retry_result = self.rm.execute_with_retry(
                get_base_response, operation_id=f"pipe_{self.loop_count}"
            )
            base_results = retry_result.get("result", [])
            pipeline_result["stages"]["retry"] = {
                "attempts": retry_result.get("attempts", 1),
                "status": retry_result.get("status", "unknown"),
            }
        else:
            base_results = get_base_response()

        pipeline_result["stages"]["base_response"] = base_results

        # Stage 5: Improve response using history
        if base_results and self.response_improver:
            original = base_results[0]["text"]
            improved = self.response_improver.improve_response(query, original, self.systems_ref)
            if improved != original:
                base_results[0]["text"] = improved
                base_results[0]["source"] = "improved"
            pipeline_result["stages"]["improvement"] = {"changed": improved != original}

        # Stage 5.5: Pre-test response against KB
        if base_results and primary_entity:
            test_result = self._test_response(query, base_results[0]["text"], primary_entity)
            pipeline_result["stages"]["pre_test"] = test_result
            if test_result.get("issues"):
                # Regenerate with KB correction context
                corrected = self._apply_kb_corrections(base_results[0]["text"], test_result["issues"])
                if corrected != base_results[0]["text"]:
                    base_results[0]["text"] = corrected
                    base_results[0]["source"] = "kb_corrected"

        # Stage 6: Add behavioral context
        if entity_state and self.de:
            behavior_response = self.de.generate_response(primary_entity, entity_state, query, user_context, 5)
            if behavior_response and base_results:
                if not any(self._texts_overlap(behavior_response, r["text"]) for r in base_results):
                    base_results.append({
                        "text": behavior_response,
                        "source": "behavior_context",
                        "score": 0.75,
                    })
            pipeline_result["stages"]["behavior"] = {"added": bool(behavior_response)}

        # Stage 6.5: Thought tracking
        if self.tt and primary_entity:
            self.tt.update_stable_flow(primary_entity)
            thought_summary = self.tt.get_thought_summary(primary_entity)
            pipeline_result["stages"]["thoughts"] = thought_summary
            if self.ii:
                sentiment = self.ii.input_history[-1]["sentiment"] if self.ii.input_history else "neutral"
                data_score = 0.1 if sentiment == "positive" else (-0.1 if sentiment == "negative" else 0)
                self.tt.update_potential(primary_entity, data_score, data_score)

        # Stage 7: Refine response
        if base_results and self.rr:
            refined_text = self.rr.refine(base_results[0]["text"], {
                "topic": primary_entity or "general",
                "add_suggestions": True,
            })
            if refined_text != base_results[0]["text"]:
                base_results[0]["text"] = refined_text
                base_results[0]["source"] = "refined"

        # Stage 8: Record and pool
        if base_results and self.response_improver:
            best = base_results[0]
            self.response_improver.record_response(query, best["text"], best["source"], best["score"])
            if self.rr and primary_entity:
                self.rr.add_to_pool(primary_entity, best["text"], best["score"])

        pipeline_result["final_response"] = base_results

        self.pipeline_log.append(pipeline_result)
        if len(self.pipeline_log) > 100:
            self.pipeline_log.pop(0)

        return base_results

    def _texts_overlap(self, text1, text2, threshold=0.4):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return False
        return len(words1 & words2) / max(len(words1), len(words2)) > threshold

    def _test_response(self, query, response, entity_name):
        """Test response against KB facts before presenting. Returns issues found."""
        issues = []
        entity_lower = entity_name.lower()
        entity_data = self.ke.entities.get(entity_lower, {})
        if not entity_data:
            return {"issues": [], "tested": False}

        response_lower = response.lower()
        attrs = entity_data.get("attributes", {})
        props = entity_data.get("properties", {})
        descs = entity_data.get("descriptions", [])

        # Check boolean attributes: if response says entity "has X" but KB says False
        for attr_key, attr_val in attrs.items():
            pretty = attr_key.replace("has_", "").replace("is_", "").replace("_", " ")
            if attr_val is False:
                # Check if response incorrectly claims this
                has_phrases = [
                    f"{entity_lower} has {pretty}",
                    f"{entity_lower} is {pretty}",
                    f"the {entity_lower} has {pretty}",
                    f"the {entity_lower} is {pretty}",
                    f"a {entity_lower} has {pretty}",
                    f"a {entity_lower} is {pretty}",
                ]
                for phrase in has_phrases:
                    if phrase in response_lower:
                        issues.append({
                            "type": "false_attribute",
                            "entity": entity_lower,
                            "attribute": attr_key,
                            "claimed": True,
                            "actual": False,
                            "pretty": pretty,
                        })
                        break
            elif attr_val is True:
                # Check if response incorrectly denies this
                no_phrases = [
                    f"no {pretty}",
                    f"not {pretty}",
                    f"does not have {pretty}",
                    f"doesn't have {pretty}",
                    f"does not have {pretty}",
                ]
                for phrase in no_phrases:
                    if phrase in response_lower and entity_lower in response_lower:
                        issues.append({
                            "type": "denied_attribute",
                            "entity": entity_lower,
                            "attribute": attr_key,
                            "claimed": False,
                            "actual": True,
                            "pretty": pretty,
                        })
                        break

        # Check property values: if response states a wrong property value
        for prop_key, prop_val in props.items():
            pretty = prop_key.replace("_", " ")
            # Check if response mentions this property with a wrong value
            # Simple heuristic: look for "the X of Y is Z" pattern
            pattern = rf"the {re.escape(pretty)} of {re.escape(entity_lower)} is (\S+)"
            match = re.search(pattern, response_lower)
            if match:
                claimed_val = match.group(1)
                actual_val = str(prop_val).lower()
                if claimed_val not in actual_val and actual_val not in claimed_val:
                    issues.append({
                        "type": "wrong_property",
                        "entity": entity_lower,
                        "property": prop_key,
                        "claimed": claimed_val,
                        "actual": prop_val,
                        "pretty": pretty,
                    })

        return {"issues": issues, "tested": True}

    def _apply_kb_corrections(self, response, issues):
        """Apply corrections to response based on KB issues found."""
        corrected = response
        for issue in issues:
            if issue["type"] == "false_attribute":
                # Entity doesn't have this attribute, add correction
                entity = issue["entity"]
                pretty = issue["pretty"]
                correction = f"Actually, a {entity} does not have {pretty}."
                if correction.lower() not in corrected.lower():
                    corrected = correction + " " + corrected
            elif issue["type"] == "denied_attribute":
                # Entity does have this attribute, fix the denial
                entity = issue["entity"]
                pretty = issue["pretty"]
                # Replace "no X" or "not X" with correct info
                corrected = re.sub(
                    rf"(no|not|does not have|doesn't have)\s+{re.escape(pretty)}",
                    f"has {pretty}",
                    corrected,
                    flags=re.IGNORECASE
                )
            elif issue["type"] == "wrong_property":
                # Wrong property value, replace with correct one
                entity = issue["entity"]
                pretty = issue["pretty"]
                actual = issue["actual"]
                corrected = re.sub(
                    rf"the {re.escape(pretty)} of {re.escape(entity)} is \S+",
                    f"the {pretty} of {entity} is {actual}",
                    corrected,
                    flags=re.IGNORECASE
                )
        return corrected

    def run_full_cycle(self, query):
        """Run a complete cycle: process -> test -> improve -> update."""
        # Process
        results = self.process(query)

        # Test if entity involved
        entities = self.tp._extract_entities(query)
        if entities:
            task = self.task_performer.create_task("run_test", f"Test pipeline for {entities[0]}",
                                                   {"test_type": "entity_state", "entity": entities[0]})
            test_result = self.task_performer.execute_task(task, self.systems_ref)

            # Simulate if applicable
            sim_task = self.task_performer.create_task("simulate", f"Simulate {entities[0]} scenarios",
                                                       {"scenario": "feed_then_play", "entity": entities[0]})
            sim_result = self.task_performer.execute_task(sim_task, self.systems_ref)

            return {
                "response": results,
                "test": test_result,
                "simulation": sim_result,
                "pipeline_loop": self.loop_count,
            }

        return {"response": results, "pipeline_loop": self.loop_count}

    def dispatch_task(self, task_type, description, context=None):
        """Create a task and route it through an agent if one's available,
        otherwise execute it directly."""
        task = self.task_performer.create_task(task_type, description, context)
        agent = self.agent_manager.assign_task(task)
        if agent:
            result = self.task_performer.execute_task(task, self.systems_ref)
            self.agent_manager.complete_task(agent.agent_id, result)
            return {"result": result, "agent": agent.agent_id, "queued": False}
        else:
            result = self.task_performer.execute_task(task, self.systems_ref)
            return {"result": result, "agent": None, "queued": True}

    def get_pipeline_summary(self):
        return {
            "total_loops": self.loop_count,
            "task_summary": self.task_performer.get_task_summary(),
            "improvement_stats": self.response_improver.get_improvement_stats(),
            "agent_status": self.agent_manager.get_status(),
            "recent_updates": len(self.update_checker.get_recent_updates()),
            "condition_history": len(self.condition_evaluator.condition_history),
        }


# =========================
# DATASET EXTRACTOR
# =========================
# Extracts answers from any dataset format: QA pairs, paragraphs, lists,
# tables, fact sheets, conversation logs. Presents correct answers even
# when not in QA format.

# Common dataset formats
FORMAT_PATTERNS = {
    "qa_pair": [r"^(.+?)[\?:]\s*(.+)$", r"^Q:\s*(.+?)\s*A:\s*(.+)"],
    "fact_line": [r"^(.+?)\s+(?:is|are|was|were)\s+(.+)$"],
    "list_item": [r"^[-•*]\s+(.+)$", r"^\d+[.)]\s+(.+)$"],
    "key_value": [r"^(.+?)\s*[:=]\s*(.+)$"],
    "sentence": [r"^[A-Z].*[.!?]$"],
    "definition": [r"^(.+?)\s+(?:means?|refers? to|is defined as)\s+(.+)$"],
    "comparison": [r"(.+?)\s+(?:is|are) (?:more|less|better|worse|faster|slower) than\s+(.+)$"],
    "temporal": [r"^(?:in|on|during|since|until|before|after)\s+(.+?),?\s*(.+)$"],
}


class DatasetExtractor:
    """Extracts and presents answers from any dataset format."""
    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.extraction_cache = {}
        self.format_stats = Counter()

    def extract_answer(self, query, context=None):
        """Extract best answer for a query from all available data."""
        context = context or {}
        query_lower = query.lower().strip().rstrip("?")
        query_words = set(re.findall(r'\w+', query_lower))

        candidates = []

        # 1. Direct QA pair match
        for q, a in self.ke.dataset_qa:
            q_words = set(re.findall(r'\w+', q.lower()))
            overlap = len(query_words & q_words) / max(len(query_words | q_words), 1)
            if overlap > 0.3:
                candidates.append({"answer": a, "source": "qa_pair", "score": overlap, "question": q})

        # 2. Entity attribute lookup
        entities = context.get("entities", [])
        for entity in entities:
            data = self.ke.entities.get(entity.lower(), {})
            attrs = data.get("attributes", {})
            props = data.get("properties", {})
            descs = data.get("descriptions", [])

            # Check if query asks about an attribute
            for attr_name, attr_val in attrs.items():
                readable = self.ke._format_attr_name(attr_name)
                if any(w in query_lower for w in readable.lower().split()):
                    val_str = f"{readable} of {entity}" if isinstance(attr_val, bool) else f"{readable}: {attr_val}"
                    candidates.append({"answer": val_str, "source": "entity_attribute", "score": 0.8})

            for prop_name, prop_val in props.items():
                readable = prop_name.replace("_", " ")
                if any(w in query_lower for w in readable.split()):
                    candidates.append({"answer": f"{readable}: {prop_val}", "source": "entity_property", "score": 0.75})

            for desc in descs:
                desc_words = set(re.findall(r'\w+', desc.lower()))
                desc_overlap = len(set(query_words) & desc_words) / max(len(set(query_words) | desc_words), 1)
                if desc_overlap > 0.4:
                    candidates.append({"answer": desc, "source": "entity_description", "score": 0.7 * desc_overlap})

        # 3. Cross-entity facts
        for entity_name, data in self.ke.entities.items():
            if entity_name in [e.lower() for e in entities]:
                continue
            descs = data.get("descriptions", [])
            for desc in descs:
                desc_words = set(re.findall(r'\w+', desc.lower()))
                overlap_words = set(query_words) & desc_words
                overlap_count = len(overlap_words)
                if overlap_count >= 3 or (overlap_count >= 2 and entity_name in " ".join(overlap_words)):
                    candidates.append({"answer": desc, "source": "cross_entity", "score": 0.5})

        # 4. Format-based extraction from raw text
        raw_text = context.get("raw_text", "")
        if raw_text:
            for fmt, patterns in FORMAT_PATTERNS.items():
                for pat in patterns:
                    for m in re.finditer(pat, raw_text, re.MULTILINE):
                        groups = m.groups()
                        if len(groups) >= 2:
                            subject, obj = groups[0], groups[1]
                            subj_words = set(re.findall(r'\w+', subject.lower()))
                            if len(query_words & subj_words) >= 1:
                                candidates.append({"answer": f"{subject.strip()} {obj.strip()}", "source": f"format_{fmt}", "score": 0.6})
                        elif len(groups) == 1:
                            text = groups[0]
                            text_words = set(re.findall(r'\w+', text.lower()))
                            if len(query_words & text_words) >= 2:
                                candidates.append({"answer": text.strip(), "source": f"format_{fmt}", "score": 0.5})

        # Sort by score and deduplicate
        candidates.sort(key=lambda x: x["score"], reverse=True)
        seen = set()
        unique = []
        for c in candidates:
            normalized = c["answer"].lower()[:80]
            if normalized not in seen:
                seen.add(normalized)
                unique.append(c)

        return unique[:5]

    def format_answer(self, candidates, detail_level="normal"):
        """Format extracted candidates into a readable answer."""
        if not candidates:
            return "I don't have enough information to answer that."

        parts = []
        for c in candidates[:3 if detail_level == "normal" else (1 if detail_level == "short" else 5)]:
            answer = c["answer"]
            if not answer.endswith("."):
                answer += "."
            parts.append(answer)

        return " ".join(parts)

    def extract_and_format(self, query, context=None, detail_level="normal"):
        """Extract and format in one call."""
        candidates = self.extract_answer(query, context)
        return self.format_answer(candidates, detail_level), candidates


# =========================
# RETRY MANAGER
# =========================
# Retries failed operations, retests results, and re-evaluates goals.

class RetryManager:
    """Manages retries for failed operations with backoff and re-evaluation."""
    def __init__(self):
        self.retry_history = []
        self.max_retries = 3
        self.backoff_base = 1.0  # seconds
        self.failed_operations = []
        self.successful_retries = []

    def should_retry(self, operation_id, error=None, attempt=0):
        """Determine if an operation should be retried."""
        if attempt >= self.max_retries:
            return False, "max retries exceeded"

        # Don't retry syntax errors or import errors
        if error and any(kw in str(error).lower() for kw in ["syntax", "import", "namerror", "typeerror"]):
            # Only retry if it might be a transient issue
            if "name" in str(error).lower() and "not defined" in str(error).lower():
                return True, "name resolution may be transient"
            return False, f"non-retryable error: {error}"

        # Retry with backoff
        delay = self.backoff_base * (2 ** attempt)
        return True, f"retry {attempt + 1}/{self.max_retries} after {delay:.1f}s"

    def execute_with_retry(self, func, args=None, kwargs=None, operation_id=None):
        """Execute a function with retry logic."""
        args = args or ()
        kwargs = kwargs or {}
        operation_id = operation_id or f"op_{time.time()}"

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.successful_retries.append({
                        "operation": operation_id,
                        "attempt": attempt,
                        "ts": time.time(),
                    })
                return {"status": "success", "result": result, "attempts": attempt + 1}
            except Exception as e:
                should, reason = self.should_retry(operation_id, e, attempt)
                self.retry_history.append({
                    "operation": operation_id,
                    "attempt": attempt,
                    "error": str(e),
                    "should_retry": should,
                    "reason": reason,
                    "ts": time.time(),
                })
                if not should:
                    self.failed_operations.append({
                        "operation": operation_id,
                        "error": str(e),
                        "attempts": attempt + 1,
                        "ts": time.time(),
                    })
                    return {"status": "failed", "error": str(e), "attempts": attempt + 1, "reason": reason}

        return {"status": "failed", "error": "max retries exceeded", "attempts": self.max_retries + 1}

    def retest_result(self, result, test_func, threshold=0.5):
        """Retest a result to verify it meets quality threshold."""
        try:
            test_result = test_func(result)
            if isinstance(test_result, (int, float)):
                passed = test_result >= threshold
                return {"passed": passed, "score": test_result, "threshold": threshold}
            elif isinstance(test_result, bool):
                return {"passed": test_result, "score": 1.0 if test_result else 0.0, "threshold": threshold}
            return {"passed": True, "score": 1.0, "threshold": threshold}
        except Exception as e:
            return {"passed": False, "score": 0.0, "error": str(e)}

    def get_retry_stats(self):
        return {
            "total_retries": len(self.retry_history),
            "successful": len(self.successful_retries),
            "failed": len(self.failed_operations),
            "success_rate": len(self.successful_retries) / max(len(self.retry_history), 1),
        }


# =========================
# INTENT INTERPRETER
# =========================
# Aggregates similar inputs, cross-references true/false, positive/negative,
# and detects when user needs changes.

class IntentInterpreter:
    """Interprets user intent by aggregating input patterns and cross-referencing."""
    def __init__(self):
        self.input_history = []
        self.intent_patterns = defaultdict(int)
        self.change_requests = []
        self.sentiment_log = []
        self.aggregated_intents = {}

    def add_input(self, query, response=None, feedback=None):
        """Add an input to the history for pattern analysis."""
        entry = {
            "query": query,
            "response": response,
            "feedback": feedback,
            "timestamp": time.time(),
            "sentiment": self._detect_sentiment(query),
            "intent": self._detect_intent(query),
        }
        self.input_history.append(entry)
        if len(self.input_history) > 200:
            self.input_history.pop(0)

        # Track intent patterns
        self.intent_patterns[entry["intent"]] += 1

        # Detect change requests
        if self._is_change_request(query):
            self.change_requests.append({
                "query": query,
                "timestamp": time.time(),
                "type": self._classify_change_type(query),
            })

        return entry

    def _detect_sentiment(self, query):
        """Detect positive/negative sentiment."""
        positive = ["good", "great", "excellent", "perfect", "love", "like", "yes", "correct", "right", "thanks", "better"]
        negative = ["bad", "wrong", "terrible", "hate", "dislike", "no", "incorrect", "worst", "fix", "change", "wrong", "error"]
        neutral = ["what", "how", "when", "where", "why", "is", "are", "can", "do"]

        query_lower = query.lower()
        pos_count = sum(1 for w in positive if w in query_lower)
        neg_count = sum(1 for w in negative if w in query_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"

    def _detect_intent(self, query):
        """Detect the primary intent of a query."""
        query_lower = query.lower()
        intents = {
            "question": ["what", "how", "when", "where", "why", "who", "which", "is", "are", "can", "do"],
            "request": ["give me", "show me", "tell me", "i want", "i need", "please"],
            "command": ["do", "run", "execute", "start", "stop", "create", "delete"],
            "feedback_positive": ["good", "great", "perfect", "thanks", "correct", "right"],
            "feedback_negative": ["wrong", "bad", "fix", "change", "error", "incorrect"],
            "change_request": ["change", "update", "modify", "edit", "replace", "switch"],
            "status_check": ["status", "how is", "condition", "health", "state"],
        }
        for intent, keywords in intents.items():
            if any(kw in query_lower for kw in keywords):
                return intent
        return "unknown"

    def _is_change_request(self, query):
        """Check if the query is requesting a change."""
        change_kw = ["change", "update", "modify", "edit", "replace", "switch", "fix", "wrong",
                     "different", "instead", "not that", "not this", "try again", "redo"]
        return any(kw in query.lower() for kw in change_kw)

    def _classify_change_type(self, query):
        """Classify the type of change being requested."""
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["format", "style", "how it looks", "presentation"]):
            return "format"
        if any(kw in query_lower for kw in ["content", "information", "what it says", "answer"]):
            return "content"
        if any(kw in query_lower for kw in ["tone", "mood", "how it sounds", "formal", "casual"]):
            return "tone"
        if any(kw in query_lower for kw in ["add", "include", "also", "plus", "more"]):
            return "addition"
        if any(kw in query_lower for kw in ["remove", "delete", "less", "fewer"]):
            return "removal"
        return "general"

    def aggregate_similar_inputs(self, window=10):
        """Aggregate recent similar inputs to find patterns."""
        recent = self.input_history[-window:]
        if not recent:
            return {}

        # Group by intent
        by_intent = defaultdict(list)
        for entry in recent:
            by_intent[entry["intent"]].append(entry)

        # Find dominant patterns
        aggregated = {}
        for intent, entries in by_intent.items():
            if len(entries) >= 2:
                queries = [e["query"] for e in entries]
                aggregated[intent] = {
                    "count": len(entries),
                    "queries": queries,
                    "sentiment": entries[-1]["sentiment"],
                    "trend": self._detect_trend(entries),
                }

        self.aggregated_intents = aggregated
        return aggregated

    def _detect_trend(self, entries):
        """Detect if sentiment is trending positive or negative."""
        if len(entries) < 2:
            return "stable"
        sentiments = [1 if e["sentiment"] == "positive" else (-1 if e["sentiment"] == "negative" else 0) for e in entries]
        trend = sum(sentiments[-3:]) - sum(sentiments[:3])
        if trend > 0:
            return "improving"
        elif trend < 0:
            return "declining"
        return "stable"

    def should_modify_response(self, query):
        """Check if the current response should be modified based on history."""
        # Check for repeated negative feedback
        recent = self.input_history[-5:]
        neg_count = sum(1 for e in recent if e["sentiment"] == "negative")
        if neg_count >= 3:
            return True, "repeated_negative"

        # Check for change requests
        recent_changes = [c for c in self.change_requests if time.time() - c["timestamp"] < 300]
        if len(recent_changes) >= 2:
            return True, "multiple_change_requests"

        return False, None

    def get_cross_references(self, query):
        """Cross-reference query against known true/false, positive/negative patterns."""
        references = []
        query_lower = query.lower()

        # Check against entity attributes (true/false)
        for entity_name, data in self._ke_entities.items() if hasattr(self, '_ke_entities') else []:
            attrs = data.get("attributes", {})
            for attr, val in attrs.items():
                if isinstance(val, bool):
                    readable = attr.replace("_", " ")
                    if any(w in query_lower for w in readable.split()):
                        references.append({
                            "fact": f"{entity_name} {readable}",
                            "value": val,
                            "source": "entity_attribute",
                        })

        return references


# =========================
# THOUGHT TRACKER
# =========================
# Tracks entity "thoughts" — what they want, their potential, strategy.
# Maintains positive/negative state and logs potential over time.

class ThoughtTracker:
    """Tracks entity desires, potential, and strategic thinking over time."""
    def __init__(self):
        self.entity_thoughts = defaultdict(lambda: {
            "potential": 0.0,  # -1.0 to 1.0
            "positive_state": 0.0,
            "negative_state": 0.0,
            "desires": [],  # what the entity wants
            "strategy": [],  # planned steps
            "stable_flow": None,  # current stable activity
            "flow_start": None,
            "flow_duration": 0,
            "thought_log": [],
        })

    def update_potential(self, entity_name, data_score=0.0, convo_score=0.0):
        """Update entity potential from data and conversation."""
        t = self.entity_thoughts[entity_name.lower()]

        # Combine data and conversation scores
        combined = (data_score * 0.6 + convo_score * 0.4)
        t["potential"] = max(-1.0, min(1.0, t["potential"] + combined * 0.1))

        # Update positive/negative states
        if combined > 0:
            t["positive_state"] = min(1.0, t["positive_state"] + combined * 0.15)
            t["negative_state"] = max(0, t["negative_state"] - combined * 0.05)
        elif combined < 0:
            t["negative_state"] = min(1.0, t["negative_state"] + abs(combined) * 0.15)
            t["positive_state"] = max(0, t["positive_state"] - abs(combined) * 0.05)

        t["thought_log"].append({
            "type": "potential_update",
            "data_score": data_score,
            "convo_score": convo_score,
            "potential": t["potential"],
            "ts": time.time(),
        })

    def set_desire(self, entity_name, desire, duration_seconds=300):
        """Set what the entity wants for a period."""
        t = self.entity_thoughts[entity_name.lower()]
        t["desires"].append({
            "desire": desire,
            "start": time.time(),
            "duration": duration_seconds,
            "active": True,
        })
        t["thought_log"].append({
            "type": "desire_set",
            "desire": desire,
            "duration": duration_seconds,
            "ts": time.time(),
        })

    def get_active_desires(self, entity_name):
        """Get currently active desires."""
        t = self.entity_thoughts[entity_name.lower()]
        now = time.time()
        active = []
        for d in t["desires"]:
            if d["active"] and (now - d["start"]) < d["duration"]:
                active.append(d["desire"])
            elif d["active"]:
                d["active"] = False
        return active

    def start_stable_flow(self, entity_name, activity):
        """Start a stable flow of activity."""
        t = self.entity_thoughts[entity_name.lower()]
        t["stable_flow"] = activity
        t["flow_start"] = time.time()
        t["flow_duration"] = 0
        t["thought_log"].append({
            "type": "flow_start",
            "activity": activity,
            "ts": time.time(),
        })

    def update_stable_flow(self, entity_name):
        """Update stable flow duration."""
        t = self.entity_thoughts[entity_name.lower()]
        if t["stable_flow"] and t["flow_start"]:
            t["flow_duration"] = time.time() - t["flow_start"]

    def end_stable_flow(self, entity_name):
        """End the current stable flow."""
        t = self.entity_thoughts[entity_name.lower()]
        if t["stable_flow"]:
            t["thought_log"].append({
                "type": "flow_end",
                "activity": t["stable_flow"],
                "duration": t["flow_duration"],
                "ts": time.time(),
            })
            t["stable_flow"] = None
            t["flow_start"] = None
            t["flow_duration"] = 0

    def set_strategy(self, entity_name, steps):
        """Set a strategy for the entity."""
        t = self.entity_thoughts[entity_name.lower()]
        t["strategy"] = [{"step": s, "status": "pending"} for s in steps]
        t["thought_log"].append({
            "type": "strategy_set",
            "steps": steps,
            "ts": time.time(),
        })

    def complete_strategy_step(self, entity_name, step_index):
        """Mark a strategy step as complete."""
        t = self.entity_thoughts[entity_name.lower()]
        if 0 <= step_index < len(t["strategy"]):
            t["strategy"][step_index]["status"] = "completed"
            t["thought_log"].append({
                "type": "step_complete",
                "step": step_index,
                "ts": time.time(),
            })

    def get_thought_summary(self, entity_name):
        """Get summary of entity's current thoughts."""
        t = self.entity_thoughts[entity_name.lower()]
        active_desires = self.get_active_desires(entity_name)
        pending_steps = [s for s in t["strategy"] if s["status"] == "pending"]
        completed_steps = [s for s in t["strategy"] if s["status"] == "completed"]

        return {
            "potential": t["potential"],
            "positive_state": t["positive_state"],
            "negative_state": t["negative_state"],
            "active_desires": active_desires,
            "stable_flow": t["stable_flow"],
            "flow_duration": t["flow_duration"],
            "strategy_progress": f"{len(completed_steps)}/{len(t['strategy'])}",
            "pending_steps": [s["step"] for s in pending_steps],
            "thought_count": len(t["thought_log"]),
        }

    def analyze_conversation_for_strategy(self, entity_name, conversation_turns):
        """Analyze conversation history to build strategy."""
        entity_lower = entity_name.lower()
        strategy_steps = []

        for turn in conversation_turns[-10:]:
            user_q = turn.get("user", "").lower()
            entities = turn.get("entities", [])

            if entity_lower in [e.lower() for e in entities]:
                if any(kw in user_q for kw in ["help", "work", "assist"]):
                    strategy_steps.append("Prepare to assist user with task")
                if any(kw in user_q for kw in ["play", "fun", "enjoy"]):
                    strategy_steps.append("Engage in playful interaction")
                if any(kw in user_q for kw in ["food", "eat", "hungry"]):
                    strategy_steps.append("Consume food for energy")
                if any(kw in user_q for kw in ["health", "vet", "hurt"]):
                    strategy_steps.append("Monitor health and seek care if needed")

        if strategy_steps:
            self.set_strategy(entity_name, strategy_steps)

        return strategy_steps


# =========================
# CONCLUSION ENGINE
# =========================
# Draws inferential conclusions from conversation history and behavioral
# stats — things NOT stated in the dataset, e.g. noticing a usage pattern
# and voicing an observation/opinion about it.

class ConclusionEngine:
    """Draws inferential conclusions from conversation history that go
    beyond what's directly in the dataset."""
    def __init__(self, knowledge_engine, conversation_memory, perspective_mapper=None):
        self.ke = knowledge_engine
        self.memory = conversation_memory
        self.pm = perspective_mapper
        self.given_conclusions = defaultdict(set)

    def draw_conclusion(self, entity):
        entity_lower = entity.lower()
        mention_count = self.memory.entities_discussed.get(entity_lower, 0) if hasattr(self.memory, 'entities_discussed') else 0
        if mention_count < 3:
            return None
        # Recency gate: don't fire stale conclusions
        if hasattr(self.memory, 'was_recently_discussed') and not self.memory.was_recently_discussed(entity_lower, n=5):
            return None
        conclusions = []
        if self.pm:
            stats = self.pm.get_usage_stats(entity_lower) if hasattr(self.pm, 'get_usage_stats') else {}
            if stats.get("pet_count", 0) >= 2:
                conclusions.append(f"I've noticed you pet your {entity_lower} a lot — it seems to enjoy the attention.")
            if stats.get("play_count", 0) >= 2:
                conclusions.append(f"You two seem to play together often — your {entity_lower} probably looks forward to it.")
            if stats.get("hurt_count", 0) >= 1:
                conclusions.append(f"I remember your {entity_lower} was hurt earlier — hopefully it's recovered by now.")
            if stats.get("feed_count", 0) >= 3:
                conclusions.append(f"You've fed your {entity_lower} several times now — it's probably come to rely on you.")
            if stats.get("vet_count", 0) >= 1:
                conclusions.append(f"Since you've taken your {entity_lower} to the vet before, it's probably in decent hands.")
        if not conclusions and mention_count >= 5:
            conclusions.append(f"We've talked about {entity_lower} quite a bit — you must be pretty interested in it.")
        conclusions = [c for c in conclusions if c not in self.given_conclusions[entity_lower]]
        if not conclusions:
            return None
        chosen = random.choice(conclusions)
        self.given_conclusions[entity_lower].add(chosen)
        return {"text": chosen, "source": "conclusion_inference", "confidence": 0.6}

    def to_dict(self):
        return {k: list(v) for k, v in self.given_conclusions.items()}

    def from_dict(self, data):
        for k, v in (data or {}).items():
            self.given_conclusions[k] = set(v)


# ================================================================
# SELF-PERSISTENCE & META-COGNITION SYSTEMS
# ================================================================

class EmotionalStateTracker:
    """Detects emotions, logs per-entity emotional states, indexes preference features."""

    EMOTION_PATTERNS = {
        "happy": [r"\bhappy\b", r"\bglad\b", r"\bpleased\b", r"\bjoy\b", r"\blove\b", r"\blike\b", r"\benjoy\b"],
        "sad": [r"\bsad\b", r"\bunhappy\b", r"\bsorry\b", r"\bregret\b", r"\bmiss\b"],
        "angry": [r"\bangry\b", r"\bfurious\b", r"\bmad\b", r"\bannoyed\b", r"\bhate\b"],
        "curious": [r"\bcurious\b", r"\bwonder\b", r"\binterested\b", r"\bfascinated\b"],
        "excited": [r"\bexcited\b", r"\bamazing\b", r"\bincredible\b", r"\bawesome\b"],
        "confused": [r"\bconfused\b", r"\bunclear\b", r"\bdon't understand\b", r"\bperplexed\b"],
        "afraid": [r"\bafraid\b", r"\bscared\b", r"\bfear\b", r"\bworried\b", r"\bconcerned\b"],
        "neutral": [r"\bok\b", r"\bfine\b", r"\balright\b", r"\bnormal\b"],
    }

    def __init__(self, knowledge_engine=None):
        self.kb = knowledge_engine
        self.entity_emotions = defaultdict(list)
        self.feature_emotion_index = defaultdict(lambda: defaultdict(lambda: {"emotion": "", "count": 0}))
        self.global_log = []
        self.max_log = 200

    def detect_emotion(self, text):
        text_lower = text.lower()
        for emotion, patterns in self.EMOTION_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    return emotion
        return "neutral"

    def log_emotion(self, entity, emotion, trigger="", context=""):
        entry = {"entity": entity, "emotion": emotion, "trigger": trigger,
                 "timestamp": time.time(), "context": context}
        self.entity_emotions[entity].append(entry)
        if len(self.entity_emotions[entity]) > 50:
            self.entity_emotions[entity] = self.entity_emotions[entity][-50:]
        self.global_log.append(entry)
        if len(self.global_log) > self.max_log:
            self.global_log = self.global_log[-self.max_log:]

    def index_feature_emotion(self, entity, feature, emotion):
        idx = self.feature_emotion_index[entity][feature]
        idx["emotion"] = emotion
        idx["count"] += 1

    def get_entity_emotions(self, entity, last_n=5):
        return self.entity_emotions.get(entity, [])[-last_n:]

    def get_dominant_emotion(self, entity):
        emotions = self.get_entity_emotions(entity, last_n=10)
        if not emotions:
            return "neutral"
        counts = Counter(e["emotion"] for e in emotions)
        return counts.most_common(1)[0][0]

    def get_feature_emotions(self, entity):
        return dict(self.feature_emotion_index.get(entity, {}))


class PreferenceIndexer:
    """Tracks what the AI finds interesting/appealing about features, stores with reasons."""

    def __init__(self, knowledge_engine=None):
        self.kb = knowledge_engine
        self.preferences = defaultdict(dict)
        self.feature_entities = defaultdict(list)
        self.log = []
        self.max_log = 200

    def record_preference(self, entity, feature, preference, reason="", strength=0.5):
        self.preferences[entity][feature] = {
            "preference": preference, "reason": reason,
            "strength": min(1.0, max(0.0, strength)), "timestamp": time.time()
        }
        if entity not in self.feature_entities[feature]:
            self.feature_entities[feature].append(entity)
        self.log.append({"entity": entity, "feature": feature, "pref": preference,
                         "reason": reason, "timestamp": time.time()})
        if len(self.log) > self.max_log:
            self.log = self.log[-self.max_log:]

    def get_preferences(self, entity):
        return dict(self.preferences.get(entity, {}))

    def get_liked_features(self, entity):
        prefs = self.get_preferences(entity)
        return {f: v for f, v in prefs.items() if v["preference"] == "like"}

    def get_disliked_features(self, entity):
        prefs = self.get_preferences(entity)
        return {f: v for f, v in prefs.items() if v["preference"] == "dislike"}

    def get_entities_with_feature(self, feature):
        return self.feature_entities.get(feature, [])

    def auto_index_from_emotion(self, entity, feature, emotion):
        if emotion in ("happy", "excited", "curious"):
            self.record_preference(entity, feature, "like", f"emotion={emotion}", 0.6)
        elif emotion in ("angry", "sad", "afraid"):
            self.record_preference(entity, feature, "dislike", f"emotion={emotion}", 0.6)


class PredictionRotator:
    """Generates rotating situational predictions with small explanations."""

    SITUATIONS = [
        ("growth", "will likely grow/evolve"),
        ("challenge", "will face a challenge"),
        ("stability", "will remain stable"),
        ("interaction", "will interact with another entity"),
        ("change", "will undergo a change"),
        ("discovery", "will discover something new"),
        ("rest", "will need rest/recovery"),
        ("exploration", "will explore new territory"),
    ]

    def __init__(self, knowledge_engine=None):
        self.kb = knowledge_engine
        self.predictions = defaultdict(list)
        self.rotation_index = defaultdict(int)
        self.used_situations = defaultdict(set)

    def generate_prediction(self, entity):
        idx = self.rotation_index[entity]
        situation_key, situation_text = self.SITUATIONS[idx % len(self.SITUATIONS)]
        self.rotation_index[entity] = idx + 1
        support = ""
        if self.kb and entity.lower() in self.kb.entities:
            ent = self.kb.entities[entity.lower()]
            attrs = ent.get("attributes", {})
            if attrs:
                key, val = list(attrs.items())[0]
                support = f" (based on {key}={val})"
        explanation = f"{entity} {situation_text}{support}"
        pred = {"situation": situation_key, "text": explanation, "entity": entity,
                "timestamp": time.time(), "used": False}
        self.predictions[entity].append(pred)
        self.used_situations[entity].add(situation_key)
        return pred

    def get_unused_prediction(self, entity):
        for p in reversed(self.predictions.get(entity, [])):
            if not p["used"]:
                p["used"] = True
                return p
        return self.generate_prediction(entity)

    def get_predictions(self, entity, last_n=5):
        return self.predictions.get(entity, [])[-last_n:]


class SentenceOverrideTracker:
    """Tracks which sentences get overridden/updated and why."""

    def __init__(self):
        self.overrides = []
        self.index = {}
        self.max_overrides = 200

    def record_override(self, old_text, new_text, reason="correction", source="unknown"):
        key = hash(old_text[:50])
        entry = {"old": old_text, "new": new_text, "reason": reason,
                 "source": source, "timestamp": time.time()}
        self.overrides.append(entry)
        self.index[key] = len(self.overrides) - 1
        if len(self.overrides) > self.max_overrides:
            self.overrides = self.overrides[-self.max_overrides:]

    def get_overrides(self, last_n=10):
        return self.overrides[-last_n:]

    def was_overridden(self, text):
        key = hash(text[:50])
        return key in self.index

    def get_override_for(self, text):
        key = hash(text[:50])
        if key in self.index:
            return self.overrides[self.index[key]]
        return None

    def count_overrides(self, reason=None):
        if reason:
            return sum(1 for o in self.overrides if o["reason"] == reason)
        return len(self.overrides)


class ChallengeSync:
    """If challenged, finds similar features across responses to sync with."""

    def __init__(self, knowledge_engine=None):
        self.kb = knowledge_engine
        self.sync_index = defaultdict(list)
        self.challenge_log = []

    def sync_features(self, challenged_entity, feature, kb=None):
        kb = kb or self.kb
        if not kb:
            return []
        similar = []
        for ent_name, ent_data in kb.entities.items():
            if ent_name == challenged_entity.lower():
                continue
            attrs = ent_data.get("attributes", {})
            props = ent_data.get("properties", {})
            all_features = {**attrs, **props}
            for f_key, f_val in all_features.items():
                if f_key.lower() == feature.lower() or str(f_val).lower() == str(feature).lower():
                    similar.append({"entity": ent_name, "feature": f_key, "value": f_val})
        return similar[:5]

    def record_challenge(self, entity, challenged_claim, resolution=""):
        self.challenge_log.append({
            "entity": entity, "claim": challenged_claim,
            "resolution": resolution, "timestamp": time.time()
        })

    def get_challenge_log(self, last_n=5):
        return self.challenge_log[-last_n:]


class UserPredictionEngine:
    """Predicts what the user would say or think about a response."""

    def __init__(self, knowledge_engine=None):
        self.kb = knowledge_engine
        self.predictions = defaultdict(list)
        self.user_reactions = []

    def predict_user_reaction(self, entity, response_text, context=""):
        confidence = 0.5
        prediction = "The user may find this informative"
        if len(response_text) > 200:
            prediction = "The user may want a more concise answer"
            confidence = 0.6
        elif len(response_text) < 30:
            prediction = "The user may want more detail"
            confidence = 0.6
        if "I don't know" in response_text or "I don't have enough" in response_text:
            prediction = "The user may be frustrated by the lack of information"
            confidence = 0.7
        if "error" in response_text.lower():
            prediction = "The user may report an issue"
            confidence = 0.8

        pred = {"prediction": prediction, "confidence": confidence,
                "response_excerpt": response_text[:100], "timestamp": time.time()}
        self.predictions[entity].append(pred)
        return pred

    def record_user_reaction(self, entity, reaction, query=""):
        self.user_reactions.append({"entity": entity, "reaction": reaction,
                                    "query": query, "timestamp": time.time()})

    def get_predictions(self, entity, last_n=5):
        return self.predictions.get(entity, [])[-last_n:]

    def get_user_reactions(self, last_n=10):
        return self.user_reactions[-last_n:]


class PreviewTester:
    """Tests a draft response against KB, checks for inconsistencies before finalizing."""

    def __init__(self, knowledge_engine=None):
        self.kb = knowledge_engine
        self.test_log = []

    def test_response(self, entity, draft_response, query=""):
        issues = []
        score = 1.0

        if self.kb and entity.lower() in self.kb.entities:
            ent = self.kb.entities[entity.lower()]
            attrs = ent.get("attributes", {})
            for attr_key, attr_val in attrs.items():
                if attr_key.lower() in draft_response.lower():
                    val_str = str(attr_val).lower()
                    if val_str not in draft_response.lower() and attr_val != "unknown":
                        issues.append(f"Attribute {attr_key}={attr_val} may be misrepresented")
                        score -= 0.15

        if len(draft_response) < 20:
            issues.append("Response too short")
            score -= 0.1
        elif len(draft_response) > 500:
            issues.append("Response may be too long")
            score -= 0.05

        contradict_patterns = [r"\bbut\b.*\bhowever\b", r"\bnot\b.*\bbut\b.*\bis\b"]
        for pat in contradict_patterns:
            if re.search(pat, draft_response.lower()):
                issues.append("Potential contradiction detected")
                score -= 0.1

        result = {"entity": entity, "score": max(0.0, score), "issues": issues,
                  "passed": score >= 0.5, "timestamp": time.time()}
        self.test_log.append(result)
        return result

    def get_test_log(self, last_n=10):
        return self.test_log[-last_n:]


class ResponseReflector:
    """Think -> plan -> test -> finalize in one sequence."""

    def __init__(self, knowledge_engine=None, emotional_tracker=None,
                 preference_indexer=None, sentence_override=None,
                 challenge_sync=None, user_predictor=None, preview_tester=None):
        self.kb = knowledge_engine
        self.emotional = emotional_tracker
        self.preferences = preference_indexer
        self.sentence_override = sentence_override
        self.challenge_sync = challenge_sync
        self.user_predictor = user_predictor
        self.preview_tester = preview_tester
        self.reflections = []
        self.entity_reflections = defaultdict(list)

    def reflect(self, entity, draft_response, query="", context=None):
        context = context or {}
        thought = self._think(entity, draft_response, query, context)
        plan = self._plan(entity, thought, context)
        tested_response, test_result = self._test(entity, draft_response, plan, query)
        final = self._finalize(entity, tested_response, test_result, query)

        reflection = {
            "entity": entity, "query": query,
            "thought": thought, "plan": plan,
            "test_score": test_result.get("score", 0) if test_result else 0,
            "final_response": final, "timestamp": time.time()
        }
        self.reflections.append(reflection)
        self.entity_reflections[entity].append(reflection)
        if len(self.reflections) > 100:
            self.reflections = self.reflections[-100:]
        return final, reflection

    def _think(self, entity, draft, query, context):
        thought = {"strengths": [], "weaknesses": [], "emotional_context": "neutral"}

        if len(draft) > 50:
            thought["strengths"].append("substantial_content")
        if entity.lower() in draft.lower():
            thought["strengths"].append("entity_mentioned")
        if "?" in query and any(w in draft.lower() for w in ["because", "due to", "since"]):
            thought["strengths"].append("explanatory")

        if len(draft) < 30:
            thought["weaknesses"].append("too_brief")
        if "I don't know" in draft:
            thought["weaknesses"].append("uncertain")
        if self.sentence_override and self.sentence_override.was_overridden(draft[:50]):
            thought["weaknesses"].append("previously_overridden")

        if self.emotional:
            thought["emotional_context"] = self.emotional.get_dominant_emotion(entity)

        if self.user_predictor:
            user_pred = self.user_predictor.predict_user_reaction(entity, draft, query)
            thought["predicted_user_reaction"] = user_pred["prediction"]

        return thought

    def _plan(self, entity, thought, context):
        plan = {"adjustments": [], "priority": "normal"}

        if "too_brief" in thought.get("weaknesses", []):
            plan["adjustments"].append("expand_response")
            plan["priority"] = "high"
        if "uncertain" in thought.get("weaknesses", []):
            plan["adjustments"].append("seek_kb_facts")
            plan["priority"] = "high"
        if "previously_overridden" in thought.get("weaknesses", []):
            plan["adjustments"].append("avoid_old_pattern")
        if thought.get("emotional_context") in ("happy", "excited"):
            plan["adjustments"].append("match_enthusiasm")
        if thought.get("predicted_user_reaction", "").startswith("The user may want"):
            plan["adjustments"].append("add_detail")

        return plan

    def _test(self, entity, draft, plan, query):
        adjusted = draft

        if "expand_response" in plan.get("adjustments", []):
            if self.kb and entity.lower() in self.kb.entities:
                ent = self.kb.entities[entity.lower()]
                attrs = ent.get("attributes", {})
                for k, v in list(attrs.items())[:2]:
                    if k.lower() not in adjusted.lower():
                        adjusted = adjusted.rstrip(".") + f". The {k} of {entity} is {v}."
        if "seek_kb_facts" in plan.get("adjustments", []):
            if self.kb and entity.lower() in self.kb.entities:
                desc = self.kb.entities[entity.lower()].get("description", "")
                if desc and desc not in adjusted:
                    adjusted = adjusted.rstrip(".") + f". {desc}"

        test_result = None
        if self.preview_tester:
            test_result = self.preview_tester.test_response(entity, adjusted, query)

        return adjusted, test_result

    def _finalize(self, entity, tested_response, test_result, query):
        final = tested_response

        if test_result and not test_result.get("passed", True):
            issues = test_result.get("issues", [])
            for issue in issues:
                if "too short" in issue.lower() and self.kb:
                    if entity.lower() in self.kb.entities:
                        ent = self.kb.entities[entity.lower()]
                        attrs = ent.get("attributes", {})
                        if attrs:
                            k, v = list(attrs.items())[0]
                            final = final.rstrip(".") + f". The {k} of {entity} is {v}."

        if self.sentence_override and final != tested_response:
            self.sentence_override.record_override(tested_response, final, "reflect_finalize", "reflector")

        return final

    def get_reflections(self, entity=None, last_n=5):
        if entity:
            return self.entity_reflections.get(entity, [])[-last_n:]
        return self.reflections[-last_n:]


class GoalReviewSystem:
    """Reviews goals, indexes good/bad/true/false, plans updates."""

    def __init__(self, knowledge_engine=None):
        self.kb = knowledge_engine
        self.goals = {}
        self.review_log = []

    def create_goal(self, name, description="", steps=None):
        self.goals[name] = {
            "description": description, "status": "active",
            "progress": 0.0, "steps": steps or [],
            "assessments": [], "updates": [],
            "created": time.time()
        }

    def assess_goal(self, name, assessment_type, detail=""):
        if name not in self.goals:
            self.create_goal(name)
        assessment = {"type": assessment_type, "detail": detail, "timestamp": time.time()}
        self.goals[name]["assessments"].append(assessment)
        self.review_log.append({"goal": name, **assessment})

    def update_goal(self, name, progress=None, status=None, note=""):
        if name not in self.goals:
            self.create_goal(name)
        if progress is not None:
            self.goals[name]["progress"] = min(1.0, max(0.0, progress))
        if status:
            self.goals[name]["status"] = status
        self.goals[name]["updates"].append({"note": note, "timestamp": time.time()})

    def get_goal_summary(self, name):
        goal = self.goals.get(name, {})
        assessments = goal.get("assessments", [])
        good = sum(1 for a in assessments if a["type"] == "good")
        bad = sum(1 for a in assessments if a["type"] == "bad")
        true_count = sum(1 for a in assessments if a["type"] == "true")
        false_count = sum(1 for a in assessments if a["type"] == "false")
        return {
            "name": name, "status": goal.get("status", "unknown"),
            "progress": goal.get("progress", 0),
            "good": good, "bad": bad, "true": true_count, "false": false_count,
            "total_assessments": len(assessments)
        }

    def get_all_goals(self):
        return {name: self.get_goal_summary(name) for name in self.goals}

    def review(self):
        results = []
        for name in self.goals:
            summary = self.get_goal_summary(name)
            if summary["total_assessments"] > 0:
                net = summary["good"] - summary["bad"]
                accuracy = summary["true"] / max(1, summary["true"] + summary["false"])
                results.append({**summary, "net_assessment": net, "accuracy": accuracy})
        return results


class SubconsciousMemory:
    """Auto-indexes background patterns without explicit queries."""

    def __init__(self, knowledge_engine=None):
        self.kb = knowledge_engine
        self.patterns = defaultdict(lambda: {"count": 0, "last_seen": 0, "entities": set(), "confidence": 0.5})
        self.observations = defaultdict(list)
        self.max_observations = 200

    def observe(self, entity, text, source="conversation"):
        words = [w.lower() for w in text.split() if len(w) > 3]
        for word in words:
            pattern = self.patterns[word]
            pattern["count"] += 1
            pattern["last_seen"] = time.time()
            pattern["entities"].add(entity)
            pattern["confidence"] = min(1.0, pattern["confidence"] + 0.05)

        obs = {"entity": entity, "text": text[:100], "source": source, "timestamp": time.time()}
        self.observations[entity].append(obs)
        if len(self.observations[entity]) > 50:
            self.observations[entity] = self.observations[entity][-50:]
        total = sum(len(v) for v in self.observations.values())
        if total > self.max_observations:
            for e in list(self.observations.keys())[:5]:
                if len(self.observations[e]) > 10:
                    self.observations[e] = self.observations[e][-10:]

    def get_frequent_patterns(self, entity=None, top_n=10):
        if entity:
            entity_set = {entity}
            relevant = {k: v for k, v in self.patterns.items() if entity_set & v["entities"]}
        else:
            relevant = dict(self.patterns)
        sorted_p = sorted(relevant.items(), key=lambda x: x[1]["count"], reverse=True)
        return sorted_p[:top_n]

    def get_observations(self, entity, last_n=5):
        return self.observations.get(entity, [])[-last_n:]

    def get_confidence(self, word):
        return self.patterns.get(word, {}).get("confidence", 0.0)


class FavoritesIndex:
    """Manual favorites storage with supporting sentence index."""

    def __init__(self, knowledge_engine=None):
        self.kb = knowledge_engine
        self.favorites = defaultdict(dict)
        self.sentence_index = defaultdict(list)
        self.max_favorites = 100

    def add_favorite(self, entity, feature, reason=""):
        if entity not in self.favorites or len(self.favorites[entity]) < self.max_favorites:
            self.favorites[entity][feature] = {
                "reason": reason, "added": time.time(),
                "support_sentences": []
            }
            if self.kb and entity.lower() in self.kb.entities:
                ent = self.kb.entities[entity.lower()]
                attrs = ent.get("attributes", {})
                props = ent.get("properties", {})
                desc = ent.get("description", "")
                if desc:
                    self.sentence_index[desc].append({"entity": entity, "feature": feature})
                    self.favorites[entity][feature]["support_sentences"].append(desc)
                for k, v in {**attrs, **props}.items():
                    sent = f"The {k} of {entity} is {v}"
                    self.sentence_index[sent].append({"entity": entity, "feature": feature})
                    self.favorites[entity][feature]["support_sentences"].append(sent)

    def remove_favorite(self, entity, feature):
        if entity in self.favorites and feature in self.favorites[entity]:
            del self.favorites[entity][feature]

    def get_favorites(self, entity):
        return dict(self.favorites.get(entity, {}))

    def get_all_favorites(self):
        return {e: dict(f) for e, f in self.favorites.items() if f}

    def get_supporting_sentences(self, entity, feature):
        fav = self.favorites.get(entity, {}).get(feature, {})
        return fav.get("support_sentences", [])

    def search_sentences(self, query):
        query_lower = query.lower()
        results = []
        for sent, refs in self.sentence_index.items():
            if query_lower in sent.lower():
                results.append({"sentence": sent, "refs": refs})
        return results


class AutoCompareContrast:
    """Uses tree/hierarchy linked comparison across all current data."""

    def __init__(self, knowledge_engine=None, noun_hierarchy=None):
        self.kb = knowledge_engine
        self.hierarchy = noun_hierarchy
        self.comparison_cache = []

    def compare_entities(self, entity_a, entity_b):
        if not self.kb:
            return {"similarities": [], "differences": [], "score": 0}

        ent_a = self.kb.entities.get(entity_a.lower(), {})
        ent_b = self.kb.entities.get(entity_b.lower(), {})
        attrs_a = ent_a.get("attributes", {})
        attrs_b = ent_b.get("attributes", {})
        props_a = ent_a.get("properties", {})
        props_b = ent_b.get("properties", {})
        all_a = {**attrs_a, **props_a}
        all_b = {**attrs_b, **props_b}

        similarities = []
        differences = []
        all_keys = set(list(all_a.keys()) + list(all_b.keys()))
        for key in all_keys:
            val_a = all_a.get(key)
            val_b = all_b.get(key)
            if val_a is not None and val_b is not None:
                if str(val_a).lower() == str(val_b).lower():
                    similarities.append({"feature": key, "value": val_a})
                else:
                    differences.append({"feature": key, "value_a": val_a, "value_b": val_b})
            elif val_a is not None:
                differences.append({"feature": key, "value_a": val_a, "value_b": None})
            elif val_b is not None:
                differences.append({"feature": key, "value_a": None, "value_b": val_b})

        hier_sim = []
        if self.hierarchy and hasattr(self.hierarchy, 'find_common_ancestor'):
            ancestor = self.hierarchy.find_common_ancestor(entity_a, entity_b)
            if ancestor:
                hier_sim.append({"common_ancestor": ancestor})

        score = len(similarities) / max(1, len(similarities) + len(differences))
        result = {
            "entity_a": entity_a, "entity_b": entity_b,
            "similarities": similarities, "differences": differences,
            "hierarchy": hier_sim, "similarity_score": score,
            "timestamp": time.time()
        }
        self.comparison_cache.append(result)
        if len(self.comparison_cache) > 50:
            self.comparison_cache = self.comparison_cache[-50:]
        return result

    def auto_compare_all(self, top_n=5):
        if not self.kb:
            return []
        entities = list(self.kb.entities.keys())[:20]
        results = []
        for i in range(len(entities)):
            for j in range(i + 1, min(i + 5, len(entities))):
                comp = self.compare_entities(entities[i], entities[j])
                if comp["similarities"] or comp["differences"]:
                    results.append(comp)
        results.sort(key=lambda x: len(x["similarities"]) + len(x["differences"]), reverse=True)
        return results[:top_n]

    def contrast_report(self, entity_a, entity_b):
        comp = self.compare_entities(entity_a, entity_b)
        parts = [f"Comparing {entity_a} vs {entity_b}:"]
        if comp["similarities"]:
            sim_feats = ", ".join(s["feature"] for s in comp["similarities"][:3])
            parts.append(f"Similar: {sim_feats}")
        if comp["differences"]:
            diff_feats = ", ".join(d["feature"] for d in comp["differences"][:3])
            parts.append(f"Different: {diff_feats}")
        if comp["hierarchy"]:
            parts.append(f"Common ancestor: {comp['hierarchy'][0].get('common_ancestor', 'none')}")
        parts.append(f"Similarity score: {comp['similarity_score']:.2f}")
        return " ".join(parts)


# =========================
# RESPONSE REFINER
# =========================
# Refines responses through formatting loops, examples, suggestions,
# and response pools.

class ResponseRefiner:
    """Refines responses through iterative improvement loops."""
    def __init__(self):
        self.response_pools = defaultdict(list)  # topic -> [responses]
        self.refinement_log = []
        self.examples_db = defaultdict(list)  # topic -> [examples]
        self.format_rules = []

    def add_to_pool(self, topic, response, score=0.5):
        """Add a response to the pool for a topic."""
        pool = self.response_pools[topic]
        pool.append({"response": response, "score": score, "ts": time.time()})
        # Keep pool sorted by score, keep top 10
        pool.sort(key=lambda x: x["score"], reverse=True)
        if len(pool) > 10:
            pool.pop()

    def get_from_pool(self, topic, min_score=0.3):
        """Get the best response from pool."""
        pool = self.response_pools.get(topic, [])
        for entry in pool:
            if entry["score"] >= min_score:
                return entry["response"]
        return None

    def add_example(self, topic, example):
        """Add an example for a topic."""
        self.examples_db[topic].append(example)

    def get_examples(self, topic, max_count=3):
        """Get examples for a topic."""
        return self.examples_db.get(topic, [])[:max_count]

    def refine(self, response, context=None):
        """Refine a response through formatting, examples, suggestions."""
        context = context or {}
        refined = response
        changes = []

        # Apply format rules
        for rule in self.format_rules:
            if rule["condition"](refined):
                refined = rule["transform"](refined)
                changes.append(f"Applied rule: {rule['name']}")

        # Add suggestions if missing
        if context.get("add_suggestions") and "for example" not in refined.lower():
            examples = self.get_examples(context.get("topic", ""), 2)
            if examples:
                example_text = " For example, " + examples[0] + "."
                if example_text not in refined:
                    refined += example_text
                    changes.append("Added example")

        # Add related info if requested
        if context.get("add_related"):
            related = context.get("related_info", [])
            for info in related[:2]:
                if info.lower() not in refined.lower():
                    refined += f" Additionally, {info}."
                    changes.append(f"Added related: {info[:30]}")

        # Format consistency
        if not refined[0].isupper():
            refined = refined[0].upper() + refined[1:]
        if not refined.endswith((".", "!", "?")):
            refined += "."

        if changes:
            self.refinement_log.append({
                "original": response,
                "refined": refined,
                "changes": changes,
                "ts": time.time(),
            })

        return refined

    def build_response_from_pool(self, topic, query, min_responses=2):
        """Build a response by combining pool entries."""
        pool = self.response_pools.get(topic, [])
        if len(pool) < min_responses:
            return None

        # Combine top responses
        parts = []
        seen = []
        for entry in pool[:3]:
            words = set(entry["response"].lower().split())
            if not any(len(words & s) / max(len(words | s), 1) > 0.7 for s in seen):
                parts.append(entry["response"])
                seen.append(words)

        return " ".join(parts) if parts else None

    def add_format_rule(self, name, condition_fn, transform_fn):
        self.format_rules.append({"name": name, "condition": condition_fn, "transform": transform_fn})


# =========================
# PROPOSAL ENGINE
# =========================
# First presentation: test -> update -> cross-reference -> log -> present final answer.
# Continues improving and retrying until quality threshold met.

class ProposalEngine:
    """Manages the proposal, testing, and final presentation of operations."""
    def __init__(self):
        self.proposals = []
        self.test_results = []
        self.final_answers = []
        self.quality_threshold = 0.7
        self.max_refinement_rounds = 3

    def create_proposal(self, query, initial_response, context=None):
        """Create a proposal for a query response."""
        proposal = {
            "id": len(self.proposals) + 1,
            "query": query,
            "initial_response": initial_response,
            "context": context or {},
            "rounds": [],
            "status": "pending",
            "created_at": time.time(),
        }
        self.proposals.append(proposal)
        return proposal

    def test_proposal(self, proposal, test_func=None):
        """Test a proposal and return results."""
        response = proposal["initial_response"]
        round_num = len(proposal["rounds"]) + 1

        test_result = {"round": round_num, "response": response}

        if test_func:
            try:
                quality = test_func(response)
                test_result["quality"] = quality
                test_result["passed"] = quality >= self.quality_threshold
            except Exception as e:
                test_result["quality"] = 0.0
                test_result["passed"] = False
                test_result["error"] = str(e)
        else:
            # Default quality checks
            quality = 0.5
            if len(response) > 20:
                quality += 0.1
            if response[0].isupper():
                quality += 0.1
            if response.endswith((".", "!", "?")):
                quality += 0.1
            if any(kw in response.lower() for kw in ["cat", "dog", "help"]):
                quality += 0.1
            test_result["quality"] = min(1.0, quality)
            test_result["passed"] = quality >= self.quality_threshold

        proposal["rounds"].append(test_result)
        self.test_results.append(test_result)

        return test_result

    def refine_proposal(self, proposal, refiner, context=None):
        """Refine a proposal based on test results."""
        if not proposal["rounds"]:
            return proposal["initial_response"]

        last_round = proposal["rounds"][-1]
        if last_round.get("passed"):
            return last_round["response"]

        # Refine
        refined = refiner.refine(last_round["response"], context or {})
        proposal["initial_response"] = refined
        return refined

    def cross_reference(self, proposal, knowledge_engine):
        """Cross-reference proposal against knowledge base."""
        query = proposal["query"]
        response = proposal["initial_response"]
        refs = []

        # Check against entity data
        query_words = set(re.findall(r'\w+', query.lower()))
        for entity_name, data in knowledge_engine.entities.items():
            descs = data.get("descriptions", [])
            for desc in descs:
                desc_words = set(re.findall(r'\w+', desc.lower()))
                if len(set(query_words) & desc_words) >= 2:
                    refs.append({"entity": entity_name, "fact": desc, "source": "kb"})

        # Check against dataset
        for q, a in knowledge_engine.dataset_qa:
            q_words = set(re.findall(r'\w+', q.lower()))
            if len(query_words & q_words) >= 2:
                refs.append({"fact": a, "source": "dataset", "score": 0.8})

        # Verify response against references
        verified = False
        for ref in refs[:3]:
            ref_words = set(re.findall(r'\w+', ref["fact"].lower()))
            resp_words = set(re.findall(r'\w+', response.lower()))
            if len(ref_words & resp_words) >= 2:
                verified = True
                break

        return {"references": refs[:5], "verified": verified, "ref_count": len(refs)}

    def finalize(self, proposal, cross_ref=None):
        """Finalize a proposal into a final answer."""
        response = proposal["initial_response"]
        round_count = len(proposal["rounds"])

        # Build final answer
        final_parts = [response]

        # Add cross-reference info if available (skip if too similar to existing parts)
        if cross_ref and cross_ref.get("references"):
            for ref in cross_ref["references"][:2]:
                fact = ref.get("fact", "")
                if not fact or fact.lower() in response.lower():
                    continue
                # Skip if too similar to existing parts
                fact_words = set(fact.lower().split())
                if any(len(fact_words & set(p.lower().split())) / max(len(fact_words | set(p.lower().split())), 1) > 0.5 for p in final_parts):
                    continue
                final_parts.append(f"Note: {fact}.")

        final_answer = {
            "query": proposal["query"],
            "answer": " ".join(final_parts),
            "rounds": round_count,
            "quality": proposal["rounds"][-1].get("quality", 0) if proposal["rounds"] else 0,
            "cross_referenced": bool(cross_ref and cross_ref.get("verified")),
            "timestamp": time.time(),
        }

        proposal["status"] = "finalized"
        self.final_answers.append(final_answer)

        return final_answer

    def process_full_cycle(self, query, response, knowledge_engine, refiner):
        """Run full proposal cycle: create -> test -> refine -> cross-ref -> finalize."""
        # Create
        proposal = self.create_proposal(query, response)

        # Test and refine loop
        for round_num in range(self.max_refinement_rounds):
            test_result = self.test_proposal(proposal)
            if test_result.get("passed"):
                break
            self.refine_proposal(proposal, refiner)

        # Cross-reference
        cross_ref = self.cross_reference(proposal, knowledge_engine)

        # Finalize
        final = self.finalize(proposal, cross_ref)

        return final


# =========================
# INFERENCE ENGINE
# =========================
# Chains facts into logical conclusions. When user says "I want cats to help
# me on 2 computers", the engine reasons: cat → can't use python → may watch →
# can keep you company → suggests practical steps.

# Built-in common-sense rules: (entity, relation, target, confidence)
COMMON_SENSE_RULES = [
    # Ability rules
    ("cat", "can_do", "watch", 0.95),
    ("cat", "can_do", "sleep", 0.95),
    ("cat", "can_do", "play", 0.9),
    ("cat", "can_do", "keep_company", 0.85),
    ("cat", "can_do", "catch_mice", 0.9),
    ("cat", "can_do", "purr", 0.95),
    ("cat", "cannot_do", "type", 0.95),
    ("cat", "cannot_do", "use_python", 0.98),
    ("cat", "cannot_do", "steal_money", 0.95),
    ("cat", "cannot_do", "drive", 0.98),
    ("cat", "cannot_do", "cook", 0.98),
    ("cat", "may_do", "knock_things_off_desk", 0.8),
    ("cat", "may_do", "sit_on_keyboard", 0.7),
    ("cat", "may_do", "walk_across_keyboard", 0.6),
    ("dog", "can_do", "keep_company", 0.9),
    ("dog", "can_do", "fetch", 0.9),
    ("dog", "can_do", "guard", 0.85),
    ("dog", "cannot_do", "use_python", 0.98),
    ("dog", "cannot_do", "type", 0.95),
    ("diamond", "can_do", "be_purchased", 0.8),
    ("diamond", "cannot_do", "be_eaten", 0.99),
    # Relationship rules
    ("cat", "similar_to", "dog", 0.8),
    ("cat", "is_a", "mammal", 0.95),
    ("cat", "is_a", "pet", 0.9),
    ("cat", "lives_with", "human", 0.85),
    ("cat", "enjoys", "human_company", 0.8),
    ("cat", "needs", "food", 0.95),
    ("cat", "needs", "water", 0.95),
    ("cat", "needs", "shelter", 0.9),
    # Obtainability rules
    ("cat", "can_be_obtained", "pet_store", 0.7),
    ("cat", "can_be_obtained", "adoption", 0.8),
    ("cat", "can_be_obtained", "breeder", 0.6),
    ("cat", "can_be_obtained", "rescue", 0.75),
    ("diamond", "can_be_obtained", "jewelry_store", 0.9),
    ("diamond", "can_be_obtained", "mine", 0.3),
    # Focus/productivity
    ("cat", "may_help_with", "focus", 0.6),
    ("cat", "may_help_with", "stress_relief", 0.7),
    ("cat", "may_hinder", "focus", 0.4),
]

# Intent detection patterns — order matters, first match wins for same type
INTENT_PATTERNS = {
    "buy_obtain": [
        r"buy (.+?)(?:\.|$)",
        r"get a? ?(.+?)(?:\.|$)",
        r"obtain (.+?)(?:\.|$)",
        r"where (?:can|do) (?:i|we) (?:get|buy|find) (.+?)(?:\?|$)",
        r"how (?:do|can) (?:i|we) (?:get|buy|obtain) (.+?)(?:\?|$)",
    ],
    "want_to": [
        r"i want (?:to )?(.+?)(?:\.|$)",
        r"i need (?:to )?(.+?)(?:\.|$)",
        r"^(?:can|could) (?:you |we )(.+?)(?:\?|$)",
    ],
    "help_with": [
        r"help (?:me|us) (?:with )?(.+?)(?:\.|$)",
        r"assist (?:me|us) (?:with )?(.+?)(?:\.|$)",
        r"(.+?) to help (?:me|us)(?:\.|$)",
    ],
    "what_can_entity_do": [
        r"what (?:can|do) (\w+) (?:do for|help)(?:\s*(?:me|us))?(?:\?|$)",
    ],
    "how_many": [
        r"how (?:many|much) (.+?)(?:\?|$)",
        r"(\d+) (.+?)(?:\.|$)",
    ],
}


class InferenceEngine:
    def __init__(self, knowledge_engine, conversation_memory):
        self.knowledge_engine = knowledge_engine
        self.memory = conversation_memory
        self.rules = list(COMMON_SENSE_RULES)
        self.inference_log = []  # Track what was inferred
        self.derived_facts = {}  # (entity, relation) -> {fact, confidence, source}

    def detect_intent(self, user_input):
        """Detect user intent from natural language. Returns most specific match."""
        lower = user_input.lower().strip()
        intents = []
        for intent_type, patterns in INTENT_PATTERNS.items():
            for pat in patterns:
                m = re.search(pat, lower)
                if m:
                    # Only take first (most specific) pattern per intent type
                    if not any(i["type"] == intent_type for i in intents):
                        intents.append({
                            "type": intent_type,
                            "groups": m.groups(),
                            "raw": m.group(0)
                        })
                    break  # Stop after first match for this intent type
        return intents

    def get_entity_facts(self, entity):
        """Get all known facts about an entity (KB + rules + inferences)."""
        entity_lower = entity.lower()
        facts = {
            "abilities": [],
            "cannot_do": [],
            "needs": [],
            "similar_to": [],
            "obtainable_from": [],
            "enjoys": [],
            "may_do": [],
            "properties": {},
            "dataset_facts": [],
        }

        # From KB
        data = self.knowledge_engine.entities.get(entity_lower, {})
        # Physical/property attributes that aren't "abilities"
        property_attrs = {"tail", "fur", "feathers", "shell", "venom", "domestication",
                          "predator status", "nocturnal behavior", "habitat", "reproduction",
                          "is_aquatic", "is_nocturnal", "is_predator", "is_venomous",
                          "is_domesticated", "has_fur", "has_feathers", "has_shell", "has_tail"}
        for attr, val in data.get("attributes", {}).items():
            readable = self.knowledge_engine._format_attr_name(attr)
            if isinstance(val, bool):
                if val:
                    # Only add as "ability" if it's an action, not a property
                    if attr.lower() not in property_attrs and readable.lower() not in property_attrs:
                        facts["abilities"].append(readable)
                else:
                    facts["cannot_do"].append(f"not {readable}")
            else:
                facts["properties"][readable] = val
        for desc in data.get("descriptions", []):
            facts["dataset_facts"].append(desc)

        # From rules
        for rule_entity, relation, target, conf in self.rules:
            if rule_entity == entity_lower:
                if relation == "can_do":
                    facts["abilities"].append(target)
                elif relation == "cannot_do":
                    facts["cannot_do"].append(target)
                elif relation == "may_do":
                    facts["may_do"].append(target)
                elif relation == "similar_to":
                    facts["similar_to"].append(target)
                elif relation == "needs":
                    facts["needs"].append(target)
                elif relation == "enjoys":
                    facts["enjoys"].append(target)
                elif relation == "can_be_obtained":
                    facts["obtainable_from"].append(target)

        # From derived facts
        for (ent, rel), info in self.derived_facts.items():
            if ent == entity_lower:
                facts["dataset_facts"].append(info["fact"])

        return facts

    def chain_facts(self, entity, context=None):
        """Chain facts into logical conclusions for answer building."""
        entity_lower = entity.lower()
        facts = self.get_entity_facts(entity_lower)
        chains = []

        # Build capability chains
        can_do = facts["abilities"]
        cannot_do = facts["cannot_do"]
        may_do = facts["may_do"]

        # Generate practical suggestions from abilities
        if "keep_company" in can_do:
            chains.append({
                "type": "suggestion",
                "text": f"A {entity_lower} can keep you company",
                "confidence": 0.85
            })
        if "watch" in can_do:
            chains.append({
                "type": "suggestion",
                "text": f"A {entity_lower} can watch you work",
                "confidence": 0.8
            })
        if "focus" in [f.split()[-1] if " " in f else f for f in can_do]:
            chains.append({
                "type": "suggestion",
                "text": f"A {entity_lower} may help you stay focused",
                "confidence": 0.6
            })

        # Generate limitation chains
        for ability in cannot_do:
            if "use" in ability or "type" in ability or "python" in ability:
                chains.append({
                    "type": "limitation",
                    "text": f"A {entity_lower} cannot {ability.replace('not ', '')}",
                    "confidence": 0.95
                })

        # Generate obtainability chains
        if facts["obtainable_from"]:
            sources = ", ".join(facts["obtainable_from"][:3])
            chains.append({
                "type": "obtainability",
                "text": f"You can get a {entity_lower} from a {sources}",
                "confidence": 0.7
            })

        # Generate need chains
        if facts["needs"]:
            needs_str = ", ".join(facts["needs"][:3])
            chains.append({
                "type": "needs",
                "text": f"A {entity_lower} needs {needs_str}",
                "confidence": 0.9
            })

        # Generate enjoyment chains
        if facts["enjoys"]:
            enjoys_str = ", ".join(facts["enjoys"][:3])
            chains.append({
                "type": "enjoyment",
                "text": f"A {entity_lower} enjoys {enjoys_str}",
                "confidence": 0.75
            })

        # Context-specific chains
        if context:
            context_lower = context.lower()
            if any(w in context_lower for w in ["computer", "work", "focus", "productivity"]):
                if "keep_company" in can_do:
                    chains.append({
                        "type": "context_suggestion",
                        "text": f"While a {entity_lower} can't use a computer, it can certainly keep you company while you work",
                        "confidence": 0.85
                    })
                if "may_hinder" in [r[2] for r in self.rules if r[0] == entity_lower]:
                    chains.append({
                        "type": "context_warning",
                        "text": f"Be aware that a {entity_lower} might also distract you from work",
                        "confidence": 0.5
                    })

        return chains

    def infer_from_context(self, entity, user_intent, conversation_history):
        """Generate inferences based on entity + user intent + conversation."""
        entity_lower = entity.lower()
        inferences = []
        facts = self.get_entity_facts(entity_lower)

        # If user wants entity to do something
        if user_intent["type"] == "want_to":
            raw_action = user_intent["groups"][0] if user_intent["groups"] else ""
            # Clean up action: remove entity name, prepositions, etc.
            action = self._clean_action(raw_action, entity_lower)

            # Extract second entity if present (e.g., "walk into my dog")
            second_entity = self._extract_second_entity(raw_action, 
                list(self.knowledge_engine.entities.keys()))

            # Special handling for "help me" type actions
            is_help_action = any(kw in raw_action.lower() for kw in ["help", "assist", "support", "accompany"])

            # Multi-entity scenario: use KB facts for both entities
            if second_entity and second_entity.lower() != entity_lower:
                consequences = self._generate_consequences(entity, raw_action, second_entity, facts)
                if consequences:
                    # Build detailed response using KB facts
                    entity_cap = entity.title()
                    second_cap = second_entity.title()
                    
                    # Get properties for both entities
                    data1 = self.knowledge_engine.entities.get(entity_lower, {})
                    data2 = self.knowledge_engine.entities.get(second_entity.lower(), {})
                    props1 = data1.get("properties", {})
                    props2 = data2.get("properties", {})
                    
                    # Build response with consequences
                    intro = f"While {entity_cap} and {second_cap} are both animals, "
                    consequence_text = ". ".join(consequences[:4])
                    outro = f". This could affect their behavior and health."
                    
                    # Add property-based context
                    prop_context = []
                    for pkey, pval in props1.items():
                        if "speed" in pkey.lower():
                            prop_context.append(f"{entity_cap}'s speed is {pval}")
                        elif "weight" in pkey.lower():
                            prop_context.append(f"{entity_cap} weighs {pval}")
                    for pkey, pval in props2.items():
                        if "speed" in pkey.lower():
                            prop_context.append(f"{second_cap}'s speed is {pval}")
                        elif "weight" in pkey.lower():
                            prop_context.append(f"{second_cap} weighs {pval}")
                    
                    full_response = intro + consequence_text + outro
                    if prop_context:
                        full_response += " For context, " + "; ".join(prop_context[:2]) + "."
                    
                    inferences.append({
                        "type": "scenario_analysis",
                        "text": full_response,
                        "confidence": 0.85
                    })
                    
                    # Add possibility assessment
                    inferences.append({
                        "type": "possibility",
                        "text": f"It's theoretically possible, but would likely result in {entity_cap} being startled or {second_cap} reacting defensively.",
                        "confidence": 0.75
                    })
                    
                    # Add state prediction
                    inferences.append({
                        "type": "state_prediction",
                        "text": f"This could affect {entity_cap}'s mood and health, potentially reducing its willingness to interact with {second_cap}.",
                        "confidence": 0.7
                    })

            # Single entity scenario
            elif is_help_action and "keep_company" in facts["abilities"]:
                inferences.append({
                    "type": "suggestion",
                    "text": f"A {entity_lower} can keep you company. "
                           f"It cannot do the work itself, but its presence can make the experience more enjoyable.",
                    "confidence": 0.85
                })
                if facts.get("similar_to"):
                    similar = facts["similar_to"][0]
                    inferences.append({
                        "type": "expansion",
                        "text": f"A {similar} could also provide companionship.",
                        "confidence": 0.6
                    })
            elif cannot_do_it if 'cannot_do_it' in dir() else False:
                ability_text = self._format_list(facts["abilities"][:2]) if facts["abilities"] else "help in other ways"
                inferences.append({
                    "type": "correction",
                    "text": f"A {entity_lower} cannot {action}, but it can {ability_text}.",
                    "confidence": 0.9
                })
            elif not can_do_it if 'can_do_it' in dir() else True and facts["may_do"]:
                may_text = self._format_list(facts["may_do"][:2])
                inferences.append({
                    "type": "alternative",
                    "text": f"A {entity_lower} may not be able to {action} directly, but it may {may_text}.",
                    "confidence": 0.7
                })
            else:
                inferences.append({
                    "type": "suggestion",
                    "text": f"A {entity_lower} can help you {action}.",
                    "confidence": 0.75
                })

        # If user wants to buy/obtain
        if user_intent["type"] == "buy_obtain":
            item = user_intent["groups"][0] if user_intent["groups"] else ""
            # Check if entity relates to the item
            if entity_lower in item or item in entity_lower or self._words_overlap(entity_lower, item) > 0.3:
                if facts["obtainable_from"]:
                    sources = " or ".join(facts["obtainable_from"][:2])
                    inferences.append({
                        "type": "suggestion",
                        "text": f"You can get a {entity_lower} from a {sources}.",
                        "confidence": 0.8
                    })
                # Check if it's obtainable at all
                for rule_entity, relation, target, conf in self.rules:
                    if rule_entity == entity_lower and relation == "cannot_be_obtained":
                        inferences.append({
                            "type": "correction",
                            "text": f"A {entity_lower} cannot be obtained from a {target}.",
                            "confidence": conf
                        })

        # If user wants help with something
        if user_intent["type"] == "help_with":
            task = user_intent["groups"][0] if user_intent["groups"] else ""
            if any(kw in task for kw in ["computer", "work", "focus", "study", "productivity"]):
                if "keep_company" in facts["abilities"]:
                    inferences.append({
                        "type": "suggestion",
                        "text": f"A {entity_lower} can help by keeping you company while you {task}. "
                               f"It cannot {task} itself, but its presence can make the experience more enjoyable.",
                        "confidence": 0.8
                    })
                if facts.get("similar_to"):
                    similar = facts["similar_to"][0]
                    inferences.append({
                        "type": "expansion",
                        "text": f"Similarly, a {similar} could also help you {task} by providing companionship.",
                        "confidence": 0.6
                    })
            else:
                # Generic help
                if facts["abilities"]:
                    ability_text = self._format_list(facts["abilities"][:2])
                    inferences.append({
                        "type": "suggestion",
                        "text": f"A {entity_lower} can help with {task} by {ability_text}.",
                        "confidence": 0.65
                    })

        # If user asks what can entity do
        if user_intent["type"] == "what_can_entity_do":
            # Generate a capability summary
            if facts["abilities"]:
                ability_text = self._format_list(facts["abilities"][:4])
                inferences.append({
                    "type": "capability",
                    "text": f"A {entity_lower} can {ability_text}.",
                    "confidence": 0.85
                })
            if facts["may_do"]:
                may_text = self._format_list(facts["may_do"][:2])
                inferences.append({
                    "type": "possibility",
                    "text": f"It may also {may_text}.",
                    "confidence": 0.7
                })

        return inferences

    def _clean_action(self, raw_action, entity_name):
        """Clean up raw action text to be more readable."""
        action = raw_action.lower().strip()
        # Remove entity name and plurals
        action = action.replace(entity_name, "").strip()
        action = action.replace(entity_name + "s", "").strip()
        # Remove common noise phrases
        noise_phrases = ["help me", "help us", "help you", "for me", "for us",
                         "on 2 computers", "on a computer", "on the computer",
                         "on computers", "with computers", "with me"]
        for phrase in noise_phrases:
            action = action.replace(phrase, "").strip()
        # Remove common noise words at start
        for prefix in ["to ", "on ", "with ", "for ", "and ", "the ", "a ", "an ", "s ", "es "]:
            while action.startswith(prefix):
                action = action[len(prefix):].strip()
        # Remove trailing noise
        for suffix in [" on", " with", " for", " to", " and"]:
            while action.endswith(suffix):
                action = action[:-len(suffix)].strip()
        # If too short or garbage, use a generic phrase
        if len(action) < 3:
            action = "do that"
        return action

    def _extract_second_entity(self, raw_action, known_entities):
        """Extract a second entity from the action text."""
        action_lower = raw_action.lower()
        for ent in known_entities:
            ent_lower = ent.lower()
            if ent_lower in action_lower:
                return ent
        # Try common animal/object names
        common_entities = ["cat", "dog", "bird", "fish", "turtle", "diamond", "ruby", "python"]
        for ent in common_entities:
            if ent in action_lower and ent not in raw_action.lower().split()[0:2]:
                return ent
        return None

    def _generate_consequences(self, entity, action, second_entity, kb_facts):
        """Generate consequences for multi-entity actions using KB facts."""
        consequences = []
        entity_lower = entity.lower() if entity else ""
        second_lower = second_entity.lower() if second_entity else ""

        # Get KB data for both entities
        data1 = self.knowledge_engine.entities.get(entity_lower, {})
        data2 = self.knowledge_engine.entities.get(second_lower, {}) if second_lower else {}

        props1 = data1.get("properties", {})
        props2 = data2.get("properties", {})
        attrs1 = data1.get("attributes", {})
        attrs2 = data2.get("attributes", {})

        # Check speed facts
        speed1 = None
        speed2 = None
        for pkey, pval in props1.items():
            if "speed" in pkey.lower():
                try:
                    speed1 = float(str(pval).replace("km/h", "").replace("kph", "").strip())
                except:
                    pass
        for pkey, pval in props2.items():
            if "speed" in pkey.lower():
                try:
                    speed2 = float(str(pval).replace("km/h", "").replace("kph", "").strip())
                except:
                    pass

        # Extract speed from action if mentioned
        import re
        speed_match = re.search(r'(\d+)\s*(?:kph|km/h|mph)', action.lower())
        mentioned_speed = None
        if speed_match:
            mentioned_speed = float(speed_match.group(1))

        # Generate consequences based on KB facts
        if speed1 and mentioned_speed:
            if mentioned_speed > speed1:
                consequences.append(f"At {mentioned_speed}kph, this would exceed {entity}'s natural speed of {speed1}kph, which is physically demanding")
            else:
                consequences.append(f"At {mentioned_speed}kph, this is within {entity}'s speed range of {speed1}kph")

        # Health impact
        if attrs1.get("is_predator") or attrs2.get("is_predator"):
            consequences.append("Since both are predators, there's a risk of injury")
        if attrs1.get("has_fur") and attrs2.get("has_fur"):
            consequences.append("Both have fur, so contact might cause minor discomfort")

        # Behavioral consequences
        if attrs1.get("is_domestic") and attrs2.get("is_domestic"):
            consequences.append("As domestic animals, they might play or interact")
        if attrs1.get("is_nocturnal") or attrs2.get("is_nocturnal"):
            consequences.append("One of them might be more active at night")

        # State changes
        consequences.append(f"This could affect {entity}'s health and mood")
        if second_entity:
            consequences.append(f"{second_entity} might react defensively")

        return consequences

    def _format_list(self, items):
        """Format a list of items into readable text."""
        clean = [i.replace("_", " ") for i in items]
        if len(clean) == 1:
            return clean[0]
        elif len(clean) == 2:
            return f"{clean[0]} and {clean[1]}"
        else:
            return ", ".join(clean[:-1]) + f", and {clean[-1]}"

    def derive_new_fact(self, entity, relation, target, confidence=0.7, source="inference"):
        """Store a new derived fact."""
        key = (entity.lower(), relation)
        if key not in self.derived_facts or self.derived_facts[key]["confidence"] < confidence:
            self.derived_facts[key] = {
                "fact": f"{entity} {relation.replace('_', ' ')} {target}",
                "confidence": confidence,
                "source": source
            }
            self.inference_log.append({
                "entity": entity,
                "relation": relation,
                "target": target,
                "confidence": confidence,
                "source": source
            })

    def _words_overlap(self, text1, text2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0
        return len(words1 & words2) / max(len(words1), len(words2))

    def get_practical_suggestions(self, entity, context=None):
        """Get practical, actionable suggestions about an entity."""
        entity_lower = entity.lower()
        facts = self.get_entity_facts(entity_lower)
        suggestions = []
        context_lower = context.lower() if context else ""

        # Only include context-relevant suggestions
        work_context = any(kw in context_lower for kw in ["computer", "work", "focus", "study", "help"])

        # What can it do for the user
        if work_context:
            for ability in facts["abilities"]:
                if ability in ("keep_company", "watch", "play", "purr"):
                    suggestions.append(f"A {entity_lower} can keep you company while you work")
                    break  # Just one companionship suggestion
        else:
            for ability in facts["abilities"][:3]:
                if ability in ("keep_company", "watch", "play", "purr", "catch_mice"):
                    suggestions.append(f"A {entity_lower} can {ability.replace('_', ' ')}")

        # What it might do (only if relevant)
        if work_context:
            suggestions.append(f"A {entity_lower} may also sit on your keyboard or knock things off your desk")

        return suggestions

        return suggestions


# =========================
# PERSPECTIVE TRACKER
# =========================

class PerspectiveTracker:
    """Tracks evolving understanding of entities as new information arrives."""
    def __init__(self):
        self.perspectives = defaultdict(lambda: {
            "facts": [],
            "confidence": 0.5,
            "last_updated": 0,
            "source_history": [],
            "inferences": []
        })

    def update_perspective(self, entity, fact, confidence=0.7, source="conversation"):
        entity_lower = entity.lower()
        p = self.perspectives[entity_lower]
        p["facts"].append({"fact": fact, "confidence": confidence, "source": source})
        p["source_history"].append(source)
        p["last_updated"] = len(p["facts"])
        # Recalculate confidence as weighted average
        if p["facts"]:
            total_weight = sum(f["confidence"] for f in p["facts"])
            p["confidence"] = min(1.0, total_weight / len(p["facts"]))

    def get_perspective(self, entity):
        return self.perspectives.get(entity.lower(), {})

    def get_view(self, entity):
        """Get current understanding of entity as a summary."""
        p = self.get_perspective(entity)
        if not p or not p["facts"]:
            return None
        recent_facts = [f["fact"] for f in p["facts"][-5:]]
        return {
            "summary": ". ".join(recent_facts),
            "confidence": p["confidence"],
            "fact_count": len(p["facts"]),
            "sources": list(set(p["source_history"]))
        }


# =========================
# TOKENIZER + VOCAB
# =========================

class Tokenizer:
    def __init__(self):
        self.word2id = {"<pad>": 0, "<unk>": 1}
        self.id2word = {0: "<pad>", 1: "<unk>"}

    def build_vocab(self, text):
        chunk_size = 100000
        counts = Counter()
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].lower().split()
            counts.update(chunk)
        for w in counts:
            if w not in self.word2id:
                idx = len(self.word2id)
                self.word2id[w] = idx
                self.id2word[idx] = w

    def encode(self, text):
        return [self.word2id.get(w, 1) for w in text.lower().split()]

    def decode(self, ids):
        return " ".join(self.id2word.get(i, "<unk>") for i in ids)


# =========================
# COGNITIVE STATE
# =========================

class EmotionalState:
    def __init__(self):
        self.nervousness = 0.0
        self.heat = 0.0
        self.confidence = 0.5
        self.familiarity = 0.0

    def update(self, query, responses, history_length):
        if not responses:
            self.nervousness = min(1.0, self.nervousness + 0.2)
        else:
            self.nervousness = max(0.0, self.nervousness - 0.1)
        self.heat = min(1.0, len(query.split()) / 20.0)
        if responses:
            avg_score = sum(r.get("score", r.get("tested_score", 0)) for r in responses) / len(responses)
            self.confidence = min(1.0, avg_score * 2)
        self.familiarity = min(1.0, history_length / 10.0)

    def get_state_description(self):
        states = []
        if self.nervousness > 0.5: states.append("nervous")
        if self.heat > 0.5: states.append("hot")
        if self.confidence < 0.3: states.append("uncertain")
        if self.familiarity > 0.7: states.append("familiar")
        return states if states else ["calm"]


class CognitiveGrid:
    def __init__(self, size=10):
        self.size = size
        self.grid = [[0.0 for _ in range(size)] for _ in range(size)]
        self.current_position = (size // 2, size // 2)
        self.best_spot = (size // 2, size // 2)
        self.position_history = []

    def update_position(self, query, quality):
        query_hash = hash(query) % (self.size * self.size)
        target_x = query_hash % self.size
        target_y = (query_hash // self.size) % self.size
        x, y = self.current_position
        x = int(x + (target_x - x) * 0.4)
        y = int(y + (target_y - y) * 0.4)
        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))
        self.current_position = (x, y)
        self.grid[x][y] = quality
        self.position_history.append((x, y, quality))
        if len(self.position_history) > 20:
            self.position_history.pop(0)
        if quality > self.grid[self.best_spot[0]][self.best_spot[1]]:
            self.best_spot = (x, y)

    def get_pressure(self):
        x, y = self.current_position
        return self.grid[x][y]


# =========================
# TRANSFORMER MODEL
# =========================

if TORCH_AVAILABLE:
    class SelfAttention(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.heads = heads
            self.head_dim = dim // heads
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
            self.out = nn.Linear(dim, dim)

        def forward(self, x):
            B, T, C = x.shape
            q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
            k = self.k(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
            v = self.v(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = torch.tril(att)
            att = F.softmax(att, dim=-1)
            out = att @ v
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.out(out)

    class Block(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = SelfAttention(dim, HEADS)
            self.ff = nn.Sequential(nn.Linear(dim, FF_DIM), nn.ReLU(), nn.Linear(FF_DIM, dim))
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x

    class GPTLike(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM)
            self.pos_emb = nn.Embedding(MAX_LEN, EMBED_DIM)
            self.blocks = nn.Sequential(*[Block(EMBED_DIM) for _ in range(LAYERS)])
            self.ln = nn.LayerNorm(EMBED_DIM)
            self.head = nn.Linear(EMBED_DIM, vocab_size)

        def forward(self, x):
            B, T = x.shape
            pos = torch.arange(0, T, device=x.device).unsqueeze(0)
            x = self.tok_emb(x) + self.pos_emb(pos)
            x = self.blocks(x)
            x = self.ln(x)
            return self.head(x)


def train_model(model, data, tokenizer):
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    tokens = tokenizer.encode(data)
    tokens = torch.tensor(tokens)
    for step in range(1000):
        ix = torch.randint(0, len(tokens) - MAX_LEN - 1, (BATCH_SIZE,))
        x = torch.stack([tokens[i:i+MAX_LEN] for i in ix])
        y = torch.stack([tokens[i+1:i+MAX_LEN+1] for i in ix])
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f}")

# =========================
# RETRIEVAL MEMORY (backward compat)
# =========================

class MemoryStore:
    def __init__(self):
        self.qa_pairs = []
        self.question_index = defaultdict(list)
        self.answer_index = defaultdict(list)
        self.word_freq = Counter()
        self.conversation_history = []
        self.max_history = 11

    def add_qa_pairs(self, text):
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.endswith('?'):
                question = line
                answer = ""
                j = i + 1
                while j < len(lines) and lines[j].strip():
                    answer += lines[j].strip() + " "
                    j += 1
                if answer:
                    idx = len(self.qa_pairs)
                    self.qa_pairs.append((question, answer.strip()))
                    for w in question.lower().split():
                        cw = w.rstrip('?!.,;:')
                        if cw not in STOP_WORDS:
                            self.question_index[cw].append(idx)
                            self.word_freq[cw] += 1
                    for w in answer.lower().split():
                        cw = w.rstrip('?!.,;:')
                        if cw not in STOP_WORDS:
                            self.answer_index[cw].append(idx)
                            self.word_freq[cw] += 1
                i = j
            else:
                i += 1

    def retrieve(self, query, top_k=3):
        words = [w.rstrip('?!.,;:') for w in query.lower().split() if w not in STOP_WORDS]
        scores = defaultdict(float)
        for w in words:
            idf = 1.0 / (self.word_freq.get(w, 1) + 1)
            for idx in self.question_index.get(w, []):
                scores[idx] += idf * 2.0
            for idx in self.answer_index.get(w, []):
                scores[idx] += idf * 0.8
        sorted_idx = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.qa_pairs[i][0], self.qa_pairs[i][1], s) for i, s in sorted_idx[:top_k] if i < len(self.qa_pairs)]

    def add_to_history(self, user_input, ai_response):
        self.conversation_history.append({"user": user_input, "ai": ai_response})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

# =========================
# FILE LEARNER
# =========================
# Handles loading data from files, testing, updating, and appending responses.
# Supports JSON, TXT, and CSV formats.

class FileLearner:
    """Learns from files, tests knowledge, and updates responses."""
    def __init__(self, knowledge_engine):
        self.ke = knowledge_engine
        self.loaded_files = {}
        self.knowledge_store = {}  # topic -> {facts: [], source_file: str, last_updated: float}
        self.response_pools = defaultdict(list)  # topic -> [responses]
        self.test_results = []

    def load_file(self, filepath, topic=None):
        """Load data from a file and index it."""
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filepath}"}

        ext = os.path.splitext(filepath)[1].lower()
        facts = []

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if ext == ".json":
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            q = item.get("question", item.get("q", ""))
                            a = item.get("answer", item.get("a", ""))
                            if q and a:
                                facts.append({"question": q, "answer": a})
                        elif isinstance(item, str):
                            facts.append({"fact": item})
                elif isinstance(data, dict):
                    for k, v in data.items():
                        facts.append({"topic": k, "fact": str(v)})
            elif ext == ".csv":
                lines = content.strip().split("\n")
                for line in lines[1:]:  # skip header
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        facts.append({"question": parts[0].strip(), "answer": parts[1].strip()})
            else:  # txt or other
                # Parse as QA pairs or fact lines
                lines = content.strip().split("\n")
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.endswith("?") or line.endswith(":"):
                        question = line
                        answer_lines = []
                        i += 1
                        while i < len(lines) and lines[i].strip():
                            answer_lines.append(lines[i].strip())
                            i += 1
                        if answer_lines:
                            facts.append({"question": question, "answer": " ".join(answer_lines)})
                    elif line and not line.startswith("#"):
                        facts.append({"fact": line})
                    i += 1

        except Exception as e:
            return {"error": str(e)}

        if not facts:
            return {"error": "No data found in file"}

        # Index the facts
        topic_name = topic or os.path.splitext(os.path.basename(filepath))[0]
        self.knowledge_store[topic_name] = {
            "facts": facts,
            "source_file": filepath,
            "last_updated": time.time(),
        }
        self.loaded_files[filepath] = topic_name

        # Add to knowledge engine for QA matching
        for fact in facts:
            q = fact.get("question", fact.get("fact", ""))
            a = fact.get("answer", fact.get("fact", ""))
            if q and a:
                self.ke.dataset_qa.append((q, a))

        return {"status": "loaded", "topic": topic_name, "facts_count": len(facts)}

    def test_knowledge(self, topic, query):
        """Test if we can answer a query about a topic."""
        if topic not in self.knowledge_store:
            return {"error": f"Topic '{topic}' not loaded"}

        facts = self.knowledge_store[topic]["facts"]
        query_lower = query.lower()

        # Search for matching facts
        matches = []
        for fact in facts:
            q = fact.get("question", fact.get("fact", "")).lower()
            a = fact.get("answer", fact.get("fact", "")).lower()
            # Score by word overlap
            q_words = set(re.findall(r'\w+', q))
            a_words = set(re.findall(r'\w+', a))
            all_words = q_words | a_words
            query_words = set(re.findall(r'\w+', query_lower))

            overlap = len(query_words & all_words) / max(len(query_words), 1)
            if overlap > 0.2:
                matches.append({
                    "fact": fact,
                    "score": overlap,
                    "answer": fact.get("answer", fact.get("fact", "")),
                })

        matches.sort(key=lambda x: x["score"], reverse=True)
        return {"matches": matches[:5], "total_facts": len(facts)}

    def append_to_file(self, filepath, new_data):
        """Append new data to an existing file."""
        try:
            ext = os.path.splitext(filepath)[1].lower()
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                existing = f.read()

            with open(filepath, "a", encoding="utf-8") as f:
                if ext == ".json":
                    # Parse existing, append, rewrite
                    try:
                        data = json.loads(existing)
                    except:
                        data = []
                    if isinstance(data, list):
                        data.append(new_data)
                    else:
                        data = [data, new_data]
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=2)
                else:
                    if existing and not existing.endswith("\n"):
                        f.write("\n")
                    if isinstance(new_data, dict):
                        f.write(f"{new_data.get('question', '')}: {new_data.get('answer', '')}\n")
                    else:
                        f.write(f"{new_data}\n")

            # Reload into knowledge store
            if filepath in self.loaded_files:
                self.load_file(filepath, self.loaded_files[filepath])

            return {"status": "appended", "file": filepath}
        except Exception as e:
            return {"error": str(e)}

    def update_fact(self, topic, old_answer, new_answer):
        """Update a fact in the knowledge store."""
        if topic not in self.knowledge_store:
            return {"error": f"Topic '{topic}' not found"}

        facts = self.knowledge_store[topic]["facts"]
        updated = False
        for fact in facts:
            if fact.get("answer", "") == old_answer:
                fact["answer"] = new_answer
                updated = True
                break

        if updated:
            # Update in dataset_qa too
            for i, (q, a) in enumerate(self.ke.dataset_qa):
                if a == old_answer:
                    self.ke.dataset_qa[i] = (q, new_answer)
                    break
            return {"status": "updated", "topic": topic}
        return {"error": "Fact not found"}

    def get_summary(self):
        """Get summary of all loaded knowledge."""
        summary = {}
        for topic, data in self.knowledge_store.items():
            summary[topic] = {
                "facts_count": len(data["facts"]),
                "source_file": data["source_file"],
                "last_updated": datetime.fromtimestamp(data["last_updated"]).isoformat(),
            }
        return summary


# =========================
# CHAT ENGINE
# =========================

class ChatEngine:
    def __init__(self, model, tokenizer, memory):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory

        # Core systems
        self.knowledge_engine = KnowledgeEngine()
        self.knowledge_engine.build_default_kb()

        self.conversation_memory = ConversationMemory(self.knowledge_engine)
        self.optimizer = ResponseOptimizer(self.knowledge_engine, self.conversation_memory)

        # State tracking
        self.entity_states = {}  # entity_name -> EntityState
        self.keyword_linker = KeywordLinker()
        self.perspective_mapper = PerspectiveMapper()

        # Behavioral systems
        self.behavior_tracker = BehaviorTracker()
        self.decision_engine = DecisionEngine(self.knowledge_engine, self.behavior_tracker)
        self.performance_logger = PerformanceLogger()

        # Enhancement systems
        self.dataset_extractor = DatasetExtractor(self.knowledge_engine)
        self.retry_manager = RetryManager()
        self.intent_interpreter = IntentInterpreter()
        self.thought_tracker = ThoughtTracker()
        self.response_refiner = ResponseRefiner()
        self.proposal_engine = ProposalEngine()
        self.file_learner = FileLearner(self.knowledge_engine)

        # Inference + Perspective
        self.inference_engine = InferenceEngine(self.knowledge_engine, self.conversation_memory)
        self.perspective_tracker = PerspectiveTracker()

        # Conclusion engine draws inferences from conversation history/behavior
        self.conclusion_engine = ConclusionEngine(
            self.knowledge_engine, self.conversation_memory, self.perspective_mapper
        )

        # Response Generator for multiple variations (now goal-seeking + conclusion-aware)
        self.response_generator = ResponseGenerator(
            self.knowledge_engine, self.conversation_memory, self.conclusion_engine
        )

        # Re-init thinking with all systems
        self.thinking = ThinkingPipeline(
            self.knowledge_engine, self.conversation_memory, self.optimizer,
            self.inference_engine, self.perspective_tracker,
            self.entity_states, self.keyword_linker, self.perspective_mapper,
            self.behavior_tracker, self.decision_engine, self.performance_logger,
            self.response_generator
        )

        # Pipeline integrator — unified loop
        self.pipeline = PipelineIntegrator(
            self.knowledge_engine, self.conversation_memory, self.thinking,
            self.entity_states, self.behavior_tracker, self.decision_engine,
            self.performance_logger, self.inference_engine, self.keyword_linker,
            self.perspective_mapper, self.perspective_tracker,
            self.dataset_extractor, self.retry_manager, self.intent_interpreter,
            self.thought_tracker, self.response_refiner, self.proposal_engine
        )

        # Cognitive
        self.emotional_state = EmotionalState()
        self.cognitive_grid = CognitiveGrid()

        self.style = "neutral"
        self.turn_count = 0

    def get_or_create_entity_state(self, entity_name, entity_type=None):
        """Get or create EntityState for an entity."""
        key = entity_name.lower()
        if key not in self.entity_states:
            self.entity_states[key] = EntityState(key, entity_type)
        return self.entity_states[key]

    def generate_top_responses(self, query, num_responses=3):
        # Use full pipeline (unified loop with all systems)
        results = self.pipeline.process(query, num_responses)
        if results:
            # Track conversation turn
            entities = self.thinking._extract_entities(query)
            categories = []
            for e in entities:
                cat = self.knowledge_engine.get_category(e)
                if cat:
                    categories.append(cat)
            primary_text = results[0]["text"] if results else ""
            self.conversation_memory.add_turn(query, primary_text, entities, categories)
            self.memory.add_to_history(query, primary_text)
            return results

        # Fallback: old retrieval
        qa_results = self.memory.retrieve(query, top_k=num_responses)
        if qa_results:
            return [{"text": a, "source": "retrieval", "score": s} for q, a, s in qa_results]

        return [{"text": "I don't have relevant information to answer that question.", "source": "fallback", "score": 0.0}]

    def chat(self):
        import sys
        print("\nChat ready. Type 'exit' to quit.", flush=True)
        print(f"[SYSTEM] Knowledge engine loaded with {len(self.knowledge_engine.entities)} entities", flush=True)
        print(f"[SYSTEM] Dataset QA pairs: {len(self.knowledge_engine.dataset_qa)}", flush=True)
        print(f"[SYSTEM] Pipeline integrator: active", flush=True)
        print(f"[SYSTEM] Entity states tracked: {len(self.entity_states)}", flush=True)
        print(flush=True)

        while True:
            try:
                msg = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!", flush=True)
                break
            if not msg:
                continue
            if msg.lower() in ("exit", "quit", "q"):
                print("Goodbye!", flush=True)
                break

            # File commands
            if msg.lower().startswith("/load "):
                filepath = msg[6:].strip()
                result = self.file_learner.load_file(filepath)
                print(f"\n[FILE] {result}", flush=True)
                continue
            elif msg.lower().startswith("/test "):
                parts = msg[6:].strip().split(" ", 1)
                if len(parts) == 2:
                    topic, query = parts
                    result = self.file_learner.test_knowledge(topic, query)
                    if "error" in result:
                        print(f"\n[FILE] {result['error']}", flush=True)
                    else:
                        print(f"\n[FILE] Found {len(result['matches'])} matches for '{query}' in '{topic}':", flush=True)
                        for m in result["matches"][:3]:
                            print(f"  - {m['answer'][:100]} (score: {m['score']:.2f})", flush=True)
                else:
                    print("\n[FILE] Usage: /test <topic> <query>", flush=True)
                continue
            elif msg.lower().startswith("/append "):
                parts = msg[8:].strip().split(" ", 2)
                if len(parts) >= 3:
                    filepath, question, answer = parts[0], parts[1], parts[2]
                    result = self.file_learner.append_to_file(filepath, {"question": question, "answer": answer})
                    print(f"\n[FILE] {result}", flush=True)
                else:
                    print("\n[FILE] Usage: /append <filepath> <question> <answer>", flush=True)
                continue
            elif msg.lower() == "/summary":
                summary = self.file_learner.get_summary()
                if summary:
                    print(f"\n[FILE] Loaded topics:", flush=True)
                    for topic, info in summary.items():
                        print(f"  - {topic}: {info['facts_count']} facts from {info['source_file']}", flush=True)
                else:
                    print("\n[FILE] No files loaded yet.", flush=True)
                continue
            elif msg.lower() == "/save":
                result = self.save_session_state()
                print(f"\n[SESSION] {result}", flush=True)
                continue
            elif msg.lower() == "/load_session":
                result = self.load_session_state()
                print(f"\n[SESSION] {result}", flush=True)
                continue
            # NEW: Self-awareness and usage commands
            elif msg.lower() == "/status":
                if hasattr(self, 'thinking') and self.thinking and hasattr(self.thinking, 'self_tracker') and self.thinking.self_tracker:
                    stats = self.thinking.self_tracker.get_stats()
                    print(f"\n[STATUS] Uptime: {stats['uptime_formatted']}", flush=True)
                    print(f"  Queries: {stats['total_queries']}, Responses: {stats['total_responses']}", flush=True)
                    print(f"  Actions logged: {stats['actions_logged']}, Decisions: {stats['decisions_made']}", flush=True)
                    print(f"  Entities tracked: {stats['entities_tracked']}, Files edited: {stats['files_edited']}", flush=True)
                    if stats['current_focus']:
                        print(f"  Current focus: {stats['current_focus']}", flush=True)
                else:
                    print("\n[STATUS] Not available", flush=True)
                continue
            elif msg.lower() == "/history":
                if hasattr(self, 'thinking') and self.thinking and hasattr(self.thinking, 'self_tracker') and self.thinking.self_tracker:
                    actions = self.thinking.self_tracker.get_recent_actions(10)
                    if actions:
                        print(f"\n[HISTORY] Last {len(actions)} actions:", flush=True)
                        for a in actions:
                            elapsed = a.get('duration', 0)
                            ts = time.strftime('%H:%M:%S', time.localtime(a['timestamp']))
                            print(f"  [{ts}] {a['action']} {a.get('target', '')} ({elapsed:.0f}s)", flush=True)
                    else:
                        print("\n[HISTORY] No actions logged yet", flush=True)
                else:
                    print("\n[HISTORY] Not available", flush=True)
                continue
            elif msg.lower() == "/myusage":
                if hasattr(self, 'thinking') and self.thinking and hasattr(self.thinking, 'user_tracker') and self.thinking.user_tracker:
                    usage = self.thinking.user_tracker.get_usage_summary()
                    print(f"\n[USAGE]\n{usage}", flush=True)
                else:
                    print("\n[USAGE] Not available", flush=True)
                continue
            elif msg.lower() == "/testme":
                if hasattr(self, 'thinking') and self.thinking and hasattr(self.thinking, 'self_test') and self.thinking.self_test:
                    # Run consistency tests on common queries
                    test_queries = ["what is a cat", "is a cat a predator", "can a cat fly"]
                    print(f"\n[TEST] Running {len(test_queries)} consistency tests...", flush=True)
                    for tq in test_queries:
                        results = self.thinking.process(tq)
                        if results:
                            check = self.thinking.self_test.check_consistency(tq, results[0]["text"])
                            status = "PASS" if check["consistent"] else "FAIL"
                            print(f"  {status}: '{tq[:40]}' (similarity={check['similarity']:.2f})", flush=True)
                    summary = self.thinking.self_test.get_test_summary()
                    print(f"  Summary: {summary['total_completed']} completed, {summary['total_passed']} passed", flush=True)
                else:
                    print("\n[TESTME] Not available", flush=True)
                continue
            # NEW: Background analysis commands
            elif msg.lower() == "/analyze":
                if hasattr(self, 'thinking') and self.thinking and hasattr(self.thinking, 'background_analyzer') and self.thinking.background_analyzer:
                    summary = self.thinking.background_analyzer.run_full_analysis()
                    print(f"\n[ANALYSIS] {json.dumps(summary, indent=2)}", flush=True)
                else:
                    print("\n[ANALYSIS] Not available", flush=True)
                continue
            elif msg.lower().startswith("/profile "):
                entity = msg[9:].strip()
                if hasattr(self, 'thinking') and self.thinking and hasattr(self.thinking, 'background_analyzer') and self.thinking.background_analyzer:
                    profile_text = self.thinking.insight_crud.import_to_conversation(entity)
                    print(f"\n{profile_text}", flush=True)
                else:
                    print("\n[PROFILE] Not available", flush=True)
                continue
            elif msg.lower() == "/gaps":
                if hasattr(self, 'thinking') and self.thinking and hasattr(self.thinking, 'background_analyzer') and self.thinking.background_analyzer:
                    gaps = self.thinking.background_analyzer.gaps
                    if gaps:
                        print(f"\n[GAPS] {len(gaps)} gaps detected:", flush=True)
                        for g in gaps[:10]:
                            print(f"  - {g['entity']}: {g['gap']} ({g['priority']})", flush=True)
                    else:
                        print("\n[GAPS] No gaps detected", flush=True)
                else:
                    print("\n[GAPS] Not available", flush=True)
                continue
            elif msg.lower() == "/insights":
                if hasattr(self, 'thinking') and self.thinking and hasattr(self.thinking, 'insight_crud') and self.thinking.insight_crud:
                    insight_text = self.thinking.insight_crud.import_to_conversation()
                    print(f"\n{insight_text}", flush=True)
                else:
                    print("\n[INSIGHTS] Not available", flush=True)
                continue
            elif msg.lower().startswith("/simulate "):
                scenario = msg[10:].strip()
                if hasattr(self, 'thinking') and self.thinking and hasattr(self.thinking, 'simulation_engine') and self.thinking.simulation_engine:
                    entities = self.thinking._extract_entities(scenario)
                    entity = entities[0] if entities else "cat"
                    sim_log = self.thinking.simulation_engine.simulate(entity, scenario, turns=5)
                    summary = self.thinking.simulation_engine.get_simulation_summary(sim_log)
                    print(f"\n[SIM] {json.dumps(summary, indent=2)}", flush=True)
                    for entry in sim_log:
                        print(f"  Turn {entry['turn']}: {entry['event']} (health={entry['state'].get('health', '?')}%)", flush=True)
                else:
                    print("\n[SIM] Not available", flush=True)
                continue
            elif msg.lower() == "/contradictions":
                if hasattr(self, 'thinking') and self.thinking and hasattr(self.thinking, 'background_analyzer') and self.thinking.background_analyzer:
                    contras = self.thinking.background_analyzer.contradictions
                    if contras:
                        print(f"\n[CONTRADICTIONS] {len(contras)} found:", flush=True)
                        for c in contras[:5]:
                            print(f"  - {c['type']}: {c.get('qa1', {}).get('question', '')[:50]}", flush=True)
                    else:
                        print("\n[CONTRADICTIONS] None found", flush=True)
                else:
                    print("\n[CONTRADICTIONS] Not available", flush=True)
                continue
            elif msg.lower().startswith("/export "):
                filepath = msg[8:].strip() or "insights_export.json"
                if hasattr(self, 'thinking') and self.thinking and hasattr(self.thinking, 'insight_crud') and self.thinking.insight_crud:
                    result = self.thinking.insight_crud.export_to_file(filepath)
                    print(f"\n[EXPORT] {result}", flush=True)
                else:
                    print("\n[EXPORT] Not available", flush=True)
                continue

            start_time = time.time()
            self.turn_count += 1

            try:
                # Generate responses
                responses = self.generate_top_responses(msg, num_responses=3)

                # Run simulation and testing on top candidates
                responses = self.thinking.simulate_and_test(msg, responses)

                # Update cognitive state
                history_len = len(self.conversation_memory.turns)
                self.emotional_state.update(msg, responses, history_len)
                self.cognitive_grid.update_position(msg, responses[0]["tested_score"] if responses else 0)

                # Extract entities for memory tracking
                entities = self.thinking._extract_entities(msg)
                categories = []
                for e in entities:
                    cat = self.knowledge_engine.get_category(e)
                    if cat:
                        categories.append(cat)

                elapsed = time.time() - start_time

                # Display
                print(f"\nAI [{', '.join(self.emotional_state.get_state_description())}] "
                      f"(thinking: {elapsed:.2f}s):", flush=True)
                for i, resp in enumerate(responses[:3], 1):
                    score = resp.get("tested_score", resp.get("score", 0))
                    source = resp.get("source", "unknown")
                    consistency = resp.get("consistency", "?")
                    extra = f" | goal={resp['goal_progress']}%" if "goal_progress" in resp else ""
                    print(f"  {i}. {resp['text']}", flush=True)
                    print(f"     [score={score:.3f} | source={source} | consistency={consistency}{extra}]", flush=True)
                print(flush=True)

                # Store in memory
                primary_text = responses[0]["text"] if responses else ""
                self.conversation_memory.add_turn(msg, primary_text, entities, categories)
                self.memory.add_to_history(msg, primary_text)

                # Log target answer for training/analysis
                self.log_target_answer(msg, primary_text)

            except Exception as e:
                import traceback
                print(f"\n[ERROR] {e}", flush=True)
                traceback.print_exc()
                print(flush=True)

    def save_session_state(self, filepath="session_state.json"):
        state = {
            "goal_progress": self.response_generator.goal_composer.to_dict() if hasattr(self.response_generator, 'goal_composer') else {},
            "conclusions_given": self.conclusion_engine.to_dict() if hasattr(self, 'conclusion_engine') else {},
            "turn_count": self.turn_count,
        }
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            return {"status": "saved", "file": filepath}
        except Exception as e:
            return {"error": str(e)}

    def load_session_state(self, filepath="session_state.json"):
        if not os.path.exists(filepath):
            return {"error": "no saved session found"}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
            if hasattr(self.response_generator, 'goal_composer'):
                self.response_generator.goal_composer.from_dict(state.get("goal_progress", {}))
            if hasattr(self, 'conclusion_engine'):
                self.conclusion_engine.from_dict(state.get("conclusions_given", {}))
            self.turn_count = state.get("turn_count", 0)
            return {"status": "loaded", "file": filepath}
        except Exception as e:
            return {"error": str(e)}

    def log_target_answer(self, user_input, ai_response, suggested_response=None, filepath="target_answers.jsonl"):
        """Log user input, AI response, and optional suggested response to a JSONL file."""
        entry = {
            "timestamp": time.time(),
            "turn": self.turn_count,
            "user_input": user_input,
            "ai_response": ai_response,
            "suggested_response": suggested_response,
        }
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass


# =========================
# MAIN
# =========================

def main():
    print(f"[BOOT] Device: {DEVICE}")
    print(f"[BOOT] PyTorch: {TORCH_AVAILABLE}")

    use_custom = input("Select custom dataset folder? (y/n): ").lower() == 'y'
    if use_custom:
        folder = select_dataset_folder()
        if folder:
            text = load_from_folder(folder)
        else:
            text = load_largest_dataset(BASE_DIR)
    else:
        text = load_largest_dataset(BASE_DIR)

    tokenizer = Tokenizer()
    tokenizer.build_vocab(text)

    memory = MemoryStore()
    memory.add_qa_pairs(text)
    print(f"[MEMORY] Loaded {len(memory.qa_pairs)} QA pairs")

    # Optional GloVe embeddings
    glove_embeddings = None
    try:
        use_glove = input("Load GloVe embeddings? (y/n): ").lower() == 'y'
    except EOFError:
        use_glove = False
    if use_glove:
        glove_embeddings = auto_load_glove(BASE_DIR)
        if glove_embeddings:
            print(f"[GLOVE] Loaded {len(glove_embeddings)} word vectors")

    # Optional SQuAD dataset via Kaggle
    try:
        use_squad = input("Load SQuAD dataset from Kaggle? (y/n): ").lower() == 'y'
    except EOFError:
        use_squad = False
    if use_squad:
        squad_raw = download_squad_dataset(BASE_DIR)
        if squad_raw:
            squad_pairs = load_squad_dataset(squad_raw, max_pairs=1000)
            for q, a in squad_pairs:
                text += f"{q}\n{a}\n\n"
            print(f"[SQuAD] Merged {len(squad_pairs)} QA pairs into dataset text")

    model = None
    if TORCH_AVAILABLE:
        model = GPTLike(len(tokenizer.word2id)).to(DEVICE)
        if not SKIP_TRAINING:
            print("\nTraining model...")
            train_model(model, text, tokenizer)
        else:
            print("[SKIP] Training skipped")

    chat = ChatEngine(model, tokenizer, memory)

    # Load dataset QA into knowledge engine too
    chat.knowledge_engine.load_dataset_qa(text)
    print(f"[KB] Knowledge engine has {len(chat.knowledge_engine.dataset_qa)} dataset QA pairs")

    # Auto-load session state if available
    if os.path.exists("session_state.json"):
        load_result = chat.load_session_state()
        print(f"[SESSION] {load_result}")

    chat.chat()


if __name__ == "__main__":
    main()
