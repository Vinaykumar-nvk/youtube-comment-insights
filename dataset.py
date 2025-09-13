# -*- coding: utf-8 -*-
import json
import random
from collections import Counter
import re

# Categories with expanded, unique rules
categories = ["Feedback", "Questions", "Praise", "Suggestions", "Criticism", "Complaints", "Off-Topic/Spam"]
rules = {
    "Feedback": [
        # English: Opinions, reflections, neutral tones
        "thought", "opinion", "feel", "vibe", "review", "take", "seems", "idk", "kinda", "sorta", "guess", "like", "meh", "ok", "well", "tbh", "ngl", "imho", "fyi", "btw",
        "vibes", "mood", "chill", "random", "weird", "odd", "hm", "perspective", "insight", "reckon", "ponder", "sense", "dig", "nah fam", "just sayin", "my 2 cents",
        "feelin it", "lowkey", "highkey", "on the fence", "mixed bag", "ehh", "not sure", "kinda vibin", "deep take",
        # Telugu: Neutral reflections
        "nenu", "cheppu", "emi", "yela", "telusu", "nuvvu cheppu", "edo okati", "చెప్పు", "ఏమి", "ఎలా", "తెలుసు", "ఏదో ఒకటి", "emo", "teliyadu", "నాకు తెలియదు",
        "nenu chusi", "nee feel", "okati cheppu", "nenu anukunnanu", "అనుకున్నాను", "nuvvu emanna", "yedo", "nee gola", "చూసి", "ఏదో", "నీ గొల",
        # Slang
        "bruh", "yo", "fam", "vibin", "sus", "cap", "no cap", "real talk", "spill", "tea", "bars", "drip", "sheesh", "bet",
        # Abusive (mild)
        "dude chill", "man up", "quit yappin",
        # Emojis
        "🤔", "😐", "😶", "🤷‍♂️", "🧐", "👀"
    ],
    "Questions": [
        # English: Inquisitive, curious, probing
        "how", "what", "where", "why", "when", "who", "huh", "wassup", "how come", "really", "whats this", "hows that", "whens it", "who dis", "what now", "y tho",
        "u sure", "rly", "wtf is this", "whatchu mean", "where u at", "why not", "who tf", "whats good", "how u holdin", "y u gotta", "whaaa", "say what", "u good",
        "r u fr", "whats poppin", "yall know", "how tf", "why tf", "where tf", "u kidding", "whozzat", "whens next", "whats ur deal", "how u even", "y u quiet",
        # Telugu: Inquisitive
        "eppudu", "enduku", "ela", "evadu", "enti", "emi", "ఎప్పుడు", "ఎందుకు", "ఎలా", "ఎవడు", "ఏంటి", "ఏమి", "rey ela", "eppudu ra", "yenti ra", "endku ra",
        "evadu vachadu", "ela chesav", "nuvvu eppudu", "enti nee plan", "yela ra", "eppudu vastav", "ఎవడు వచ్చాడు", "ఎలా చేశావ్", "నీవు ఎప్పుడు", "ఏంటి నీ ప్లాన్",
        # Slang
        "bro whats", "dude why", "rey enti idi", "fam how", "yo when", "nuvvu yela ra",
        # Abusive (mild)
        "rey evadu ra", "enti ra nee thopu", "nuvvu enduku ra",
        # Emojis
        "❓", "⁉️", "🤷", "😕", "🤨", "❔"
    ],
    "Praise": [
        # English: Positive, enthusiastic, celebratory
        "great", "awesome", "love", "amazing", "good", "best", "thanks", "dope", "lit", "fire", "cool", "nice", "sweet", "sick", "rad", "perfect", "stellar", "top",
        "slay", "killed it", "on fleek", "goat", "legend", "king", "queen", "banger", "smash", "hit", "vibes", "chef kiss", "gold", "ace", "pro", "mad props", "salute",
        "big up", "holy", "blessed", "lit af", "fire af", "turnt", "hype", "pog", "based", "chad", "sigma", "peak", "prime", "shine", "glow", "spark", "bling", "swag",
        # Telugu: Positive
        "bagundi", "superu", "keka", "mass", "thoppu", "abbah", "kummi", "బాగుంది", "సూపరు", "కేక", "మాస్", "తోపు", "అబ్బా", "కుమ్మి", "chala bagundi", "vere level",
        "gundello", "pichi thoppu", "massu massu", "superu bro", "keka kummi", "గుండెల్లో", "చాలా బాగుంది", "వేరే లెవెల్",
        # Slang
        "lit af", "fire bro", "keka ra", "massu", "drip king", "slaps", "bussin", "yeet", "to the moon", "yass", "extra", "too good",
        # Emojis
        "👍", "🔥", "❤️", "😍", "🙌", "💯", "🌟", "🎉", "👑", "💪"
    ],
    "Suggestions": [
        # English: Advisory, constructive, helpful
        "should", "try", "suggest", "next", "please", "add", "do", "more", "maybe", "how about", "go for", "put", "give", "make", "pls", "u gotta", "drop", "run",
        "throw", "cook", "fix", "get", "grab", "post", "share", "show", "tell", "keep", "stop", "start", "chill", "hype", "boost", "mix", "switch", "swap", "change",
        "update", "redo", "test", "check", "look", "peek", "pop", "bang", "slap", "vibe", "jam", "push", "pull", "lift",
        # Telugu: Advisory
        "chesi", "pettu", "ivvu", "konchem", "inka", "chey ra", "చేసి", "పెట్టు", "ఇవ్వు", "కొంచెం", "ఇంకా", "చేయ్ రా", "chesi chudu", "pettu ra idi", "ivvu naku",
        "next chey", "konchem chey", "inka ivvu", "chey bro", "pettu superu", "చేసి చూడు", "పెట్టు రా ఇది", "ఇవ్వు నాకు", "నెక్స్ట్ చేయ్",
        # Slang
        "try this fam", "add idi bro", "pettu ra idi", "go ham", "pop off", "mix it up", "spice it", "flip it", "crank it", "hustle",
        # Emojis
        "👇", "➡️", "✨", "📌", "🔧", "🚀"
    ],
    "Criticism": [
        # English: Negative judgment, disapproval
        "bad", "poor", "hate", "worst", "dislike", "lame", "cringe", "stupid", "dumb", "trash", "garbage", "sucks", "pathetic", "fail", "boring", "weak", "meh",
        "nope", "gross", "yuck", "bleh", "dull", "overrated", "mid", "clown", "joke", "ridiculous", "absurd", "braindead", "pure trash", "total crap", "lousy",
        # Telugu: Negative judgment
        "nache", "bekaar", "wasteu", "pichi", "chetha", "నచ్చలేదు", "వేస్టు", "పిచ్చి", "చెత్త", "nee thikka", "gadida", "pichhuk", "donga", "thikkodi", "baffoon",
        "మూర్ఖుడు", "గాడిద", "పిచ్చుక", "డొంగ", "తిక్కోడి", "nee munda", "thikka thikka", "అర్థం లేదు", "ముండ",
        # Slang/Abusive
        "dumbass", "idiot", "trash af", "pichi ra", "gadida ra", "moron", "clownin", "hot garbage", "braindead af", "fucking waste",
        # Emojis
        "👎", "🤮", "💩", "😒", "🙅", "🗑️"
    ],
    "Complaints": [
        # English: Issues, frustration, problems
        "broken", "lag", "fail", "error", "fix", "sucks", "mess", "disaster", "wreck", "ruined", "wtf", "damn", "shit", "crap", "piss", "pissed", "mad", "angry",
        "dogshit", "trainwreck", "cooked", "burnt", "toasted", "grilled", "shat", "fuck this", "what a mess", "total fail", "big oof", "yikes", "ugh", "ew", "why tf",
        "how tf", "this blows", "piece of shit", "absolute crap", "damn mess",
        # Telugu: Issues
        "padindi", "slowga", "problem", "gandu", "పాడైంది", "స్లోగా", "ప్రాబ్లెమ్", "గండు", "thikkodi", "slowga undi", "idi padindi ra", "problem ra", "gandu ra",
        "fix chey ra", "padindi idi", "slowga chesav", "ఫిక్స్ చేయ్ రా", "స్లోగా చేశావ్", "ఇది పాడైంది రా",
        # Slang/Abusive
        "fuck this bro", "shit ra", "pichi problem", "loose ra", "thikkodi ra", "damn gandu", "wtf ra idi",
        # Emojis
        "😡", "🤬", "🚫", "💥", "😤", "🔥👎"
    ],
    "Off-Topic/Spam": [
        # English: Irrelevant, promotional, off-context
        "http", "subscribe", "check", "promo", "link", "follow", "click", "sub", "pls", "my vid", "join", "free", "win", "signup", "hit sub", "smash sub", "like",
        "share", "comment", "sub 4 sub", "follow 4 follow", "check my", "new vid", "out now", "drop a sub", "give like", "pls follow", "yo sub", "sub plz", "click here",
        "promo code", "spam this", "random af", "off topic", "buy now", "deal alert",
        # Telugu: Promotional
        "chudu", "naa channel", "sub chey", "link lo", "వీడియో చూడు", "సబ్ చేయ్", "chudu ra", "naa video", "sub chey ra", "link chudu", "rey link idi", "chudu bro",
        "link lo chudu", "sub chey pichi", "చూడు బ్రో", "లింక్ లో చూడు", "సబ్ చేయ్ పిచ్చి",
        # Slang
        "sub me bro", "click here fam", "yo check this", "spam ra", "random shit",
        # Emojis
        "📺", "🔗", "👀", "📢", "💸", "🎥"
    ]
}

# Expanded emoji mappings
emoji_map = {
    "🤔": "thought", "😐": "meh", "😶": "idk", "🤷‍♂️": "huh", "🧐": "guess", "👀": "vibe",
    "❓": "what", "⁉️": "why", "🤷": "how", "😕": "whens", "🤨": "who", "❔": "where",
    "👍": "good", "🔥": "fire", "❤️": "love", "😍": "awesome", "🙌": "great", "💯": "perfect", "🌟": "stellar", "🎉": "lit", "👑": "king", "💪": "strong",
    "👇": "try", "➡️": "next", "✨": "do", "📌": "add", "🔧": "fix", "🚀": "boost",
    "👎": "bad", "🤮": "hate", "💩": "trash", "😒": "lame", "🙅": "nope", "🗑️": "garbage",
    "😡": "sucks", "🤬": "damn", "🚫": "fail", "💥": "mess", "😤": "angry",
    "📺": "check", "🔗": "link", "👀": "sub", "📢": "promo", "💸": "free", "🎥": "vid"
}

# Expanded multi-meaning words with category context
multi_meaning = {
    "cool": ["Praise", "Feedback"],  # "cool beat" (Praise) vs "cool vibe" (Feedback)
    "shit": ["Complaints", "Criticism"],  # "shit lags" (Complaints) vs "shit quality" (Criticism)
    "dope": ["Praise", "Feedback"],  # "dope play" (Praise) vs "dope take" (Feedback)
    "pichi": ["Criticism", "Complaints"],  # "pichi ra" (Criticism) vs "pichi error" (Complaints)
    "fix": ["Suggestions", "Complaints"],  # "fix this" (Suggestions) vs "fix ur crap" (Complaints)
    "check": ["Suggestions", "Off-Topic/Spam"],  # "check this out" (Suggestions) vs "check my vid" (Spam)
    "vibe": ["Feedback", "Praise"],  # "weird vibe" (Feedback) vs "fire vibe" (Praise)
    "super": ["Praise", "Feedback"],  # "super clip" (Praise) vs "super random" (Feedback)
    "damn": ["Complaints", "Criticism"],  # "damn lag" (Complaints) vs "damn boring" (Criticism)
    "yo": ["Feedback", "Questions"],  # "yo seems off" (Feedback) vs "yo whats this" (Questions)
}

# Expanded content types
content_types = {
    "Vlogs": ["travel", "daily", "prank", "adventure", "family", "roadtrip", "vlogmas", "move", "shop", "date", "chill", "explore", "pet", "party", "drama"],
    "Cooking": ["baking", "spicy", "quick", "vegan", "grill", "dessert", "curry", "pizza", "soup", "snack", "breakfast", "dinner", "cake", "drink", "street"],
    "Tech": ["unboxing", "review", "hack", "phone", "laptop", "gadget", "app", "DIY", "mod", "setup", "AI", "VR", "drone", "repair", "code"],
    "Gaming": ["playthrough", "fail", "tips", "esports", "RPG", "FPS", "retro", "mobile", "strat", "sim", "horror", "multi", "speedrun", "glitch", "clan"],
    "Music": ["cover", "live", "beats", "rock", "pop", "jazz", "rap", "EDM", "folk", "metal", "lyric", "dance", "chill", "vocal", "drop"],
    "Comedy": ["skit", "meme", "roast", "prank", "standup", "fail", "parody", "troll", "silly", "dark", "satire", "improv", "joke", "laugh", "goof"],
    "Fitness": ["workout", "yoga", "diet", "gym", "run", "bike", "lift", "crossfit", "abs", "bulk", "cut", "stretch", "challenge", "sweat", "fit"],
    "Fashion": ["haul", "makeup", "outfit", "hair", "nails", "trend", "vintage", "street", "jewelry", "shoes", "cosplay", "glam", "DIY", "look", "style"],
    "Education": ["math", "history", "science", "code", "lang", "physics", "bio", "geo", "econ", "art", "space", "stats", "logic", "exam", "tip"],
    "Movies": ["action", "drama", "comedy", "horror", "sci-fi", "romance", "thriller", "anime", "doc", "superhero", "mystery", "fantasy", "noir", "short", "plot"],
    "News": ["breaking", "politics", "tech", "sports", "world", "local", "crime", "weather", "celeb", "finance", "health", "sci", "trend", "live", "alert"],
    "Art": ["draw", "paint", "digital", "craft", "sculpt", "photo", "edit", "graffiti", "tattoo", "comic", "3D", "abstract", "portrait", "color", "inspo"]
}

# Expanded structures
structures = [
    "{subject} {keyword} {extra}",              # "travel prank seems chill"
    "{keyword} {subject} {extra}",              # "how gaming fail huh"
    "{subject} is {keyword} {extra}",           # "daily clip is trash bro"
    "Rey {keyword} {subject} {extra}",          # "Rey bagundi music beats ra"
    "{keyword} this {subject} {extra}",         # "love this skit meme lit"
    "{subject} {keyword} ra {extra}",           # "unboxing review ra nice"
    "Yo {subject} is {keyword} {extra}",        # "Yo gaming tips is dope fam"
    "{keyword} {subject} tho {extra}",          # "fix tech hack tho yo"
    "Nuvvu {keyword} {subject} {extra}",        # "Nuvvu chesi cooking quick bro"
    "{subject} got {keyword} {extra}",          # "vlog adventure got vibe sheesh"
    "This {subject} {keyword} {extra}",         # "This art draw sucks damn"
    "Bro {keyword} {subject} {extra}",          # "Bro what comedy roast wtf"
    "{keyword} ra {subject} {extra}",           # "superu ra music live fire"
    "Hey {subject} {keyword} {extra}",          # "Hey fitness workout dope ra"
    "{subject} {keyword} af {extra}"            # "movies thriller lame af fam"
]

# Expanded actions
actions = [
    "show", "do", "share", "review", "play", "chesi", "pettu", "chudu", "make", "drop", "hit", "run", "cook", "fix", "grab", "post", "tell", "keep", "stop", "start",
    "boost", "mix", "swap", "update", "test", "peek", "bang", "slap", "jam", "push", "lift", "throw", "yeet", "skrt", "vibe", "hype", "chill", "grind", "snap",
    "roll", "kick", "punch", "spin", "twist", "crush", "smash", "build", "cut", "paint"
]

# Expanded extras
extras = [
    # Neutral
    "so cool", "chill", "random", "weird", "ok", "bagundi", "telusu", "బాగుంది", "fine", "alright", "ehh", "whatever", "kinda dope", "not bad", "pretty wild",
    "straight up", "for sure", "legit", "vibez", "just ok", "nuvvu telusu", "ఎలాగో",
    # Positive
    "lit", "fire", "nice", "sweet", "superu", "keka", "mass", "సూపరు", "కేక", "awesome", "rad", "perfect", "stellar", "top", "banger", "smash", "gold", "ace",
    "mad props", "big W", "hype", "pog", "based", "shine", "glow", "bling", "swag", "thoppu ra", "kummi bro", "vere level", "అబ్బా చా",
    # Negative
    "trash", "crap", "sucks", "wasteu", "pichi", "chetha", "వేస్టు", "చెత్త", "lame", "gross", "yuck", "bleh", "dull", "weak", "fail", "mid", "clown", "joke",
    "dogshit", "wreck", "burnt", "shat", "piss", "gandu ra", "thikkodi ra", "nee munda",
    # Slang
    "bro", "fam", "ra", "dude", "yo", "sis", "homie", "mate", "pal", "sheesh", "whew", "bet", "word", "tru", "fax", "spill", "tea", "bars", "drip", "extra", "turnt",
    "pop off", "yass", "big mood", "vibes", "energy", "skrt", "to the moon", "outta here",
    # Abusive
    "dumbass", "idiot", "moron", "gadida", "pichhuk", "donga", "thikkodi", "baffoon", "loose", "gundu", "pilla", "chetha ra", "nee thopu", "fucking hell",
    # Emojis
    "👍", "🔥", "👎", "🤔", "😡", "😂", "💩", "❤️", "😍", "🙌", "💯", "🌟", "🎉", "👑", "💪", "👇", "➡️", "✨", "📌", "🔧", "🚀", "🤮", "😒", "🙅", "🗑️", "💥", "😤", "📺", "🔗", "👀", "📢", "💸", "🎥"
]

# Expanded typo and variant generator
def add_typo_or_variant(text):
    if random.random() < 0.35:  # 35% chance for typos or variants
        words = text.split()
        if words:
            idx = random.randint(0, len(words) - 1)
            word = words[idx].lower()
            variants = {
                "eppudu": ["epudu", "yeppudu", "eppadu", "epuduu", "eppdu"], "when": ["wen", "whn", "wenn", "whenn"],
                "bagundi": ["bagndi", "bgundi", "bagundii", "baggundi", "baguni"], "good": ["god", "gud", "goood", "godd"],
                "chala": ["chaala", "chla", "chalaa", "challa", "chlaa"], "fix": ["fx", "fixx", "fxi", "fiks", "ffix"],
                "superu": ["supru", "superuu", "spueru", "superru"], "great": ["gr8", "gret", "greaaat", "grt"],
                "pichi": ["pichhi", "picchi", "pchhi", "pihci"], "crazy": ["craxy", "crazzy", "crzy", "craz"],
                "thoppu": ["thopu", "toppu", "thopuu", "thppu"], "awesome": ["aweosme", "awsm", "awsome", "awesm"],
                "keka": ["kekka", "keeka", "kkaa", "kekaa"], "cool": ["cooll", "kol", "coool", "cll"],
                "wasteu": ["wastu", "wsteu", "wasteuu"], "trash": ["trsh", "tras", "trassh"],
                "sub": ["sbu", "sbb", "subb"], "check": ["chek", "chck", "chekk"]
            }
            if word in variants:
                words[idx] = random.choice(variants[word])
            elif len(word) > 3:
                typo = random.choice(["swap", "drop", "add", "double"])
                if typo == "swap":
                    i = random.randint(0, len(word) - 2)
                    words[idx] = word[:i] + word[i + 1] + word[i] + word[i + 2:]
                elif typo == "drop":
                    i = random.randint(0, len(word) - 1)
                    words[idx] = word[:i] + word[i + 1:]
                elif typo == "add":
                    words[idx] = word + random.choice("aeiou")
                else:  # double
                    i = random.randint(0, len(word) - 1)
                    words[idx] = word[:i] + word[i] + word[i:]
        return " ".join(words)
    return text

# Emoji preprocessor
def preprocess_emojis(text):
    for emoji, meaning in emoji_map.items():
        text = text.replace(emoji, f" {meaning} ")
    return text

comments = set()
dataset = []

target_size = 20000
while len(dataset) < target_size:
    content_type = random.choice(list(content_types.keys()))
    subtype = random.choice(content_types[content_type])
    category = random.choice(categories)
    lang_choice = random.random()  # 50% English, 30% Telugu, 20% Mixed

    structure = random.choice(structures)
    keyword = random.choice(rules[category])
    if keyword in multi_meaning and category not in multi_meaning[keyword]:
        continue  # Skip if keyword doesn’t fit category context
    action = random.choice(actions)
    extra = random.choice(extras)
    subject = f"{subtype} {random.choice(['vibe', 'moment', 'clip', 'bit', 'scene', 'twist', 'jam', 'drop', 'flow', 'shot'])}"

    if lang_choice < 0.5:  # English
        comment = structure.format(subject=subject, keyword=keyword, extra=extra)
    elif lang_choice < 0.8:  # Telugu
        comment = structure.format(subject=subject, keyword=keyword, extra=extra) + " ra"
    else:  # Mixed
        comment = f"Rey {subject} {keyword} {extra}"

    comment = preprocess_emojis(comment)
    comment = add_typo_or_variant(comment)
    comment = re.sub(r'\s+', ' ', comment.strip())

    if comment not in comments and 3 <= len(comment.split()) <= 20:
        comments.add(comment)
        multi_label = random.random() < 0.2  # 20% multi-label
        dataset.append({
            "text": comment,
            "categories": [category] if not multi_label else [category, random.choice([c for c in categories if c != category])]
        })

    if len(dataset) % 1000 == 0:  # Balance categories
        category_counts = Counter([cat for entry in dataset for cat in entry["categories"]])
        least_category = min(categories, key=lambda c: category_counts.get(c, 0))
        if category_counts.get(least_category, 0) < target_size // 10:
            category = least_category

with open('dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Generated {len(dataset)} unique comments in dataset.json")