import re
import difflib
import math
from collections import defaultdict, Counter
import sacrebleu

ALLOWED_TAGS = {
    "define_problem", "drug_decision", "evaluate_result",
    "contact_related", "therapeutic_procedure", "advice_and_precaution",
    "gather_info", "treatment_goal", "defer_decision",
}

_ANY_TAG_LOOSE = re.compile(r"</?(?:" + "|".join(ALLOWED_TAGS) + r")>")


def _get_text(completion) -> str:
    """Extract text from TRL completion format and STRIP <think> tags."""
    text = ""
    if isinstance(completion, list):
        if len(completion) > 0 and isinstance(completion[-1], dict):
            text = completion[-1].get("content", "")
    else:
        text = str(completion)
        
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif text.strip().startswith("<think>"):
        return ""
        
    return text


def _strip_all_tags(text: str) -> str:
    cleaned = _ANY_TAG_LOOSE.sub("", text)
    return " ".join(cleaned.split())


# ── Reward : Fidelity [0, 1] ──
def reward_fidelity(completions, raw_text, **kwargs) -> list[float]:
    scores = []
    for comp, orig in zip(completions, raw_text):
        text = _get_text(comp)
        cleaned_comp = _strip_all_tags(text)
        cleaned_orig = _strip_all_tags(orig)

        if not cleaned_orig:
            scores.append(1.0)
            continue

        ratio = difflib.SequenceMatcher(None, cleaned_comp, cleaned_orig).ratio()

        if ratio >= 0.99:
            scores.append(1.0)
        elif ratio < 0.90:
            scores.append(0.0)
        else:
            scores.append((ratio - 0.90) / 0.09)
    return scores


TAG_RE = re.compile(r'</?[a-zA-Z_]+>')
LEADING_RE = re.compile(r'^[^a-zA-Z0-9*]+')
TRAILING_RE = re.compile(r'[^a-zA-Z0-9*]+$')

def _clean_extracted_text(text: str) -> str:
    
    # 1. Remove nested tags using TAG_RE
    text = TAG_RE.sub('', text).strip()
    if not text:
        return ""

    # 2. Handle prefix (Prefix Protection)
    prefix = ""
    if text.startswith("[**"):
        prefix = "[**"
        text = text[3:]
    
    # Clear leading garbage characters using LEADING_RE
    text = LEADING_RE.sub('', text)
    text = prefix + text

    # 3. Handle suffix (Suffix Protection)
    suffix = ""
    if text.endswith("**]"):
        suffix = "**]"
        text = text[:-3]
        
    # Clear trailing garbage characters using TRAILING_RE
    text = TRAILING_RE.sub('', text)
    text = text + suffix

    return text.lower()


def _extract_clean_strings_with_tags(tagged_text: str) -> list[tuple[str, str]]:
    """Extract (Tag, Clean_String) using a stack system, 100% supporting Nested/Overlap"""
    spans = []
    stack = defaultdict(list)
    
    i = 0
    text_len = len(tagged_text)
    
    while i < text_len:
        if tagged_text[i] == '<':
            tag_end = tagged_text.find('>', i)
            if tag_end == -1:
                i += 1
                continue
            
            tag_content = tagged_text[i+1:tag_end]
            is_close = tag_content.startswith('/')
            tag_name = tag_content[1:] if is_close else tag_content
            
            if tag_name in ALLOWED_TAGS:
                if not is_close:
                    # Store coordinate "after the > sign" as the start of the text
                    stack[tag_name].append(tag_end + 1)
                else:
                    if stack[tag_name]:
                        start_idx = stack[tag_name].pop()
                        # Extract the raw inner text
                        raw_inner = tagged_text[start_idx:i]
                        # Pass through the cleaner (remove nested tags + punctuation)
                        clean_str = _clean_extracted_text(raw_inner)
                        
                        if clean_str:
                            spans.append((tag_name, clean_str))
                i = tag_end + 1
            else:
                i += 1
        else:
            i += 1
            
    return spans


def reward_fast_token_f1(completions, gold, **kwargs) -> list[float]:
    scores = []
    for comp, gold_text in zip(completions, gold):
        text = _get_text(comp)
        
        pred_spans = _extract_clean_strings_with_tags(text)
        gold_spans = _extract_clean_strings_with_tags(gold_text)

        if not pred_spans and not gold_spans:
            scores.append(1.0)
            continue
        if not pred_spans or not gold_spans:
            scores.append(0.0)
            continue

        # Group "words (Tokens)" by Tag
        pred_by_cat = defaultdict(list)
        gold_by_cat = defaultdict(list)

        # Split text into words using .split()
        for tag, string in pred_spans:
            pred_by_cat[tag].extend(string.split())
        for tag, string in gold_spans:
            gold_by_cat[tag].extend(string.split())

        all_cats = set(pred_by_cat) | set(gold_by_cat)
        cat_f1s = []

        for cat in all_cats:
            p_tokens = pred_by_cat.get(cat, [])
            g_tokens = gold_by_cat.get(cat, [])

            if not p_tokens and not g_tokens:
                cat_f1s.append(1.0)
                continue
            if not p_tokens or not g_tokens:
                cat_f1s.append(0.0)
                continue

            # Count matching words (Intersection) using Counter
            p_counter = Counter(p_tokens)
            g_counter = Counter(g_tokens)
            
            tp = sum((p_counter & g_counter).values())
            
            p = tp / len(p_tokens)
            r = tp / len(g_tokens)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            cat_f1s.append(f1)

        scores.append(sum(cat_f1s) / len(cat_f1s))
        
    return scores


def reward_exact_match_f1(completions, gold, **kwargs) -> list[float]:
    """Must exactly match the entire text block (after cleaning punctuation)."""
    scores = []
    for comp, gold_text in zip(completions, gold):
        text = _get_text(comp)
        
        pred_spans = set(_extract_clean_strings_with_tags(text))
        gold_spans = set(_extract_clean_strings_with_tags(gold_text))

        if not pred_spans and not gold_spans:
            scores.append(1.0)
            continue
        if not pred_spans or not gold_spans:
            scores.append(0.0)
            continue

        pred_by_cat = defaultdict(set)
        gold_by_cat = defaultdict(set)

        for tag, string in pred_spans:
            pred_by_cat[tag].add(string)
        for tag, string in gold_spans:
            gold_by_cat[tag].add(string)

        all_cats = set(pred_by_cat) | set(gold_by_cat)
        cat_f1s = []

        for cat in all_cats:
            p_set = pred_by_cat.get(cat, set())
            g_set = gold_by_cat.get(cat, set())

            if not p_set and not g_set:
                cat_f1s.append(1.0)
                continue
            if not p_set or not g_set:
                cat_f1s.append(0.0)
                continue

            # Exact Match: Count only text blocks that are exactly identical in every way
            tp = len(p_set & g_set)
            p = tp / len(p_set) if len(p_set) > 0 else 0.0
            r = tp / len(g_set) if len(g_set) > 0 else 0.0
            
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            cat_f1s.append(f1)

        scores.append(sum(cat_f1s) / len(cat_f1s))
        
    return scores


def reward_fast_token_precision(completions, gold, **kwargs) -> list[float]:
    """Score as Token Precision: Count only correctly predicted words divided by total predicted words."""
    scores = []
    for comp, gold_text in zip(completions, gold):
        text = _get_text(comp)
        
        pred_spans = _extract_clean_strings_with_tags(text)
        gold_spans = _extract_clean_strings_with_tags(gold_text)

        if not pred_spans and not gold_spans:
            scores.append(1.0)
            continue
        if not pred_spans or not gold_spans:
            scores.append(0.0)
            continue

        # Group "words (Tokens)" by Tag
        pred_by_cat = defaultdict(list)
        gold_by_cat = defaultdict(list)

        # Split text into words using .split()
        for tag, string in pred_spans:
            pred_by_cat[tag].extend(string.split())
        for tag, string in gold_spans:
            gold_by_cat[tag].extend(string.split())

        all_cats = set(pred_by_cat) | set(gold_by_cat)
        cat_scores = []  # Changed variable name from cat_f1s to cat_scores

        for cat in all_cats:
            p_tokens = pred_by_cat.get(cat, [])
            g_tokens = gold_by_cat.get(cat, [])

            if not p_tokens and not g_tokens:
                cat_scores.append(1.0)
                continue
            if not p_tokens or not g_tokens:
                cat_scores.append(0.0)
                continue

            # Count matching words (Intersection) using Counter
            p_counter = Counter(p_tokens)
            g_counter = Counter(g_tokens)
            
            # Number of True Positives (correctly predicted words present in the gold standard)
            tp = sum((p_counter & g_counter).values())
            
            # 🌟 Use pure Precision: (correct predictions) / (total predictions made)
            precision = tp / len(p_tokens) if len(p_tokens) > 0 else 0.0
            
            cat_scores.append(precision)

        scores.append(sum(cat_scores) / len(cat_scores))
        
    return scores


def reward_exact_match_precision(completions, gold, **kwargs) -> list[float]:
    """Must exactly match the entire text block (after cleaning punctuation) (using pure Precision)."""
    scores = []
    for comp, gold_text in zip(completions, gold):
        text = _get_text(comp)
        
        pred_spans = set(_extract_clean_strings_with_tags(text))
        gold_spans = set(_extract_clean_strings_with_tags(gold_text))

        if not pred_spans and not gold_spans:
            scores.append(1.0)
            continue
        if not pred_spans or not gold_spans:
            scores.append(0.0)
            continue

        pred_by_cat = defaultdict(set)
        gold_by_cat = defaultdict(set)

        for tag, string in pred_spans:
            pred_by_cat[tag].add(string)
        for tag, string in gold_spans:
            gold_by_cat[tag].add(string)

        all_cats = set(pred_by_cat) | set(gold_by_cat)
        cat_scores = []

        for cat in all_cats:
            p_set = pred_by_cat.get(cat, set())
            g_set = gold_by_cat.get(cat, set())

            if not p_set and not g_set:
                cat_scores.append(1.0)
                continue
            if not p_set or not g_set:
                cat_scores.append(0.0)
                continue

            tp = len(p_set & g_set)
            
            precision = tp / len(p_set) if len(p_set) > 0 else 0.0
            
            cat_scores.append(precision)

        scores.append(sum(cat_scores) / len(cat_scores))
        
    return scores


def reward_fast_token_f05(completions, gold, **kwargs) -> list[float]:
    scores = []
    for comp, gold_text in zip(completions, gold):
        text = _get_text(comp)
        
        pred_spans = _extract_clean_strings_with_tags(text)
        gold_spans = _extract_clean_strings_with_tags(gold_text)

        if not pred_spans and not gold_spans:
            scores.append(1.0)
            continue
        if not pred_spans or not gold_spans:
            scores.append(0.0)
            continue

        pred_by_cat = defaultdict(list)
        gold_by_cat = defaultdict(list)

        for tag, string in pred_spans:
            pred_by_cat[tag].extend(string.split())
        for tag, string in gold_spans:
            gold_by_cat[tag].extend(string.split())

        all_cats = set(pred_by_cat) | set(gold_by_cat)
        cat_f1s = []

        for cat in all_cats:
            p_tokens = pred_by_cat.get(cat, [])
            g_tokens = gold_by_cat.get(cat, [])

            if not p_tokens and not g_tokens:
                cat_f1s.append(1.0)
                continue
            if not p_tokens or not g_tokens:
                cat_f1s.append(0.0)
                continue

            # Count matching words (Intersection) using Counter
            p_counter = Counter(p_tokens)
            g_counter = Counter(g_tokens)
            
            tp = sum((p_counter & g_counter).values())
            
            p = tp / len(p_tokens)
            r = tp / len(g_tokens)
            
            beta_sq = 0.25
            f05 = ((1 + beta_sq) * p * r) / ((beta_sq * p) + r) if (p + r) > 0 else 0.0
            cat_f1s.append(f05)

        scores.append(sum(cat_f1s) / len(cat_f1s))
        
    return scores


def reward_exact_match_f05(completions, gold, **kwargs) -> list[float]:
    """Must exactly match the entire text block (after cleaning punctuation)."""
    scores = []
    for comp, gold_text in zip(completions, gold):
        text = _get_text(comp)
        
        pred_spans = set(_extract_clean_strings_with_tags(text))
        gold_spans = set(_extract_clean_strings_with_tags(gold_text))

        if not pred_spans and not gold_spans:
            scores.append(1.0)
            continue
        if not pred_spans or not gold_spans:
            scores.append(0.0)
            continue

        pred_by_cat = defaultdict(set)
        gold_by_cat = defaultdict(set)

        for tag, string in pred_spans:
            pred_by_cat[tag].add(string)
        for tag, string in gold_spans:
            gold_by_cat[tag].add(string)

        all_cats = set(pred_by_cat) | set(gold_by_cat)
        cat_f1s = []

        for cat in all_cats:
            p_set = pred_by_cat.get(cat, set())
            g_set = gold_by_cat.get(cat, set())

            if not p_set and not g_set:
                cat_f1s.append(1.0)
                continue
            if not p_set or not g_set:
                cat_f1s.append(0.0)
                continue

            # Exact Match: Count only text blocks that are exactly identical in every way
            tp = len(p_set & g_set)
            p = tp / len(p_set) if len(p_set) > 0 else 0.0
            r = tp / len(g_set) if len(g_set) > 0 else 0.0
            
            beta_sq = 0.25
            f05 = ((1 + beta_sq) * p * r) / ((beta_sq * p) + r) if (p + r) > 0 else 0.0
            cat_f1s.append(f05)

        scores.append(sum(cat_f1s) / len(cat_f1s))
        
    return scores


def _space_out_tags(text: str) -> str:
    """Add spaces around tags so they can be separated as distinct Tokens"""
    spaced_text = re.sub(r'(</?[a-zA-Z_]+>)', r' \1 ', text)
    return " ".join(spaced_text.split())


def reward_bleu_exact_tags(completions, gold, **kwargs) -> list[float]:
    """
    Reward Function: Calculate BLEU-4 score including Tags as part of the sentence.
    Uses sacrebleu instead of a custom BLEU implementation.
    """
    scores = []
    for comp, gold_text in zip(completions, gold):
        pred_text = _get_text(comp)
        
        if not pred_text and not gold_text:
            scores.append(1.0)
            continue
        if not pred_text or not gold_text:
            scores.append(0.0)
            continue
            
        # Add spaces around tags so they can be separated as distinct Tokens
        spaced_pred = _space_out_tags(pred_text)
        spaced_gold = _space_out_tags(gold_text)
        
        bleu_score = sacrebleu.sentence_bleu(spaced_pred, [spaced_gold], tokenize='whitespace').score
        scores.append(bleu_score / 100.0)
        
    return scores