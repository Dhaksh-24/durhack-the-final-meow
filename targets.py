"""targets.py -- the things we are trying to find, and how we ask for them.

aliases  label names that count as this target when reading a dataset.
         Datasets disagree: "Phone", "mobile phone" and "cell phone" are
         all one thing to us. Matched after lower-casing and turning - and _
         into spaces, and matched exactly -- "keyboard" must not match "key".
prompts  phrasings to try on an open-vocabulary model. bench.py scores each
         one, so add freely; the losers cost nothing once we have picked.
"""

TARGETS = {
    "keys": {
        "aliases": ["key", "keys", "car key", "car keys", "house key",
                    "house keys", "keychain", "key chain", "keyring"],
        "prompts": ["keys", "key", "bunch of keys", "keychain", "car keys",
                    "house keys", "metal keys", "keys on a keyring"],
    },
    "wallet": {
        "aliases": ["wallet", "wallets", "billfold"],
        "prompts": ["wallet", "leather wallet", "purse", "billfold",
                    "card holder", "folded wallet"],
    },
    "phone": {
        "aliases": ["phone", "phones", "cell phone", "cellphone", "mobile",
                    "mobile phone", "mobile phones", "smartphone", "iphone"],
        "prompts": ["phone", "cell phone", "smartphone", "mobile phone",
                    "iphone", "phone screen"],
    },
    "remote": {
        "aliases": ["remote", "remote control", "remote controller", "tv remote"],
        "prompts": ["remote", "remote control", "tv remote",
                    "television remote control"],
    },
    "glasses": {
        "aliases": ["glasses", "eyeglasses", "eye glasses", "spectacles",
                    "sunglasses"],
        "prompts": ["glasses", "eyeglasses", "spectacles", "pair of glasses",
                    "sunglasses"],
    },
    "watch": {
        "aliases": ["watch", "watches", "wristwatch", "wrist watch", "smartwatch"],
        "prompts": ["watch", "wristwatch", "wrist watch", "smartwatch"],
    },
    "headphones": {
        "aliases": ["headphones", "headphone", "headset", "earphones",
                    "earbuds", "airpods"],
        "prompts": ["headphones", "headset", "earphones", "earbuds",
                    "wireless earbuds case"],
    },
    "bag": {
        "aliases": ["bag", "bags", "backpack", "handbag", "rucksack"],
        "prompts": ["bag", "backpack", "handbag", "rucksack", "tote bag"],
    },
}

# Round 1 result: best prompt per target, each scored alone with
# yolov8s-worldv2 at imgsz=640 on the LVIS sets from pull_lvis.py.
# conf is the threshold where F1 peaked. a+b means load both as one set.
BEST = {
    "phone":      {"prompt": "cell phone",       "conf": 0.14, "ap50": 0.618},
    "remote":     {"prompt": "remote",           "conf": 0.07, "ap50": 0.558},
    "watch":      {"prompt": "wrist watch",      "conf": 0.24, "ap50": 0.557},
    "glasses":    {"prompt": "pair of glasses",  "conf": 0.26, "ap50": 0.481},
    "bag":        {"prompt": "handbag+backpack", "conf": 0.14, "ap50": 0.431},
    "wallet":     {"prompt": "wallet",           "conf": 0.06, "ap50": 0.274},  # only 29 boxes
    "headphones": {"prompt": "headset",          "conf": 0.03, "ap50": 0.234},
    "keys":       {"prompt": "car keys",         "conf": 0.05, "ap50": 0.111},  # 0.145 at imgsz=1280
}
