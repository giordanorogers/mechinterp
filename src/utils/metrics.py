import logging
from typing import Sequence
from src import functional
from src.utils.typing import StrSequence, ArrayLike
import re
from typing import Sequence, Union, List
from collections.abc import Sequence as ABCSequence
import numpy as np

logger = logging.getLogger(__name__)

def _validate_same_length(**kwargs: Sequence | ArrayLike) -> None:
    """Validate all batch sizes are the same."""
    lengths = {key: len(seq) for key, seq in kwargs.items()}
    if len(set(lengths.values())) > 1:
        message = "inconsistent batch sizes:" + "\n\t"
        message += "\n\t".join(f"{key}={length}" for key, length in lengths.items())
        raise ValueError(message)

def _recall(predictions: Sequence[StrSequence], targets: StrSequence) -> list[float]:
    """Compute the recall@k for predicted tokens.

    A prediction is considered correct if it is a prefix of the target.
    Insensitive to case and whitespace.

    Args:
        predictions: List of top-k predicted tokens.
        targets: Target tokens. Must be the same length as `predictions`.

    Returns:
        List of [recall@1, recall@2, ..., recall@k].

    """
    _validate_same_length(predictions=predictions, targets=targets)
    if len(predictions) == 0:
        return None  # type: ignore

    k = max(map(len, predictions))
    recalls = [0.0] * k
    for topk, target in zip(predictions, targets):
        for i in range(k):
            if functional.any_is_nontrivial_prefix(topk[: i + 1], target) or any(
                functional.is_nontrivial_prefix(target, pred) for pred in topk[: i + 1]
            ):
                recalls[i] += 1
            else:
                logger.info(f"No Correct Prediction: {topk} | Target: {target}")

def normalize_country_name(name: str) -> str:
    """Normalize country names to handle variations."""
    name = name.lower().strip()
    
    # Remove leading/trailing punctuation and spaces
    name = re.sub(r'^[\s\'"]+|[\s\'"]+$', '', name)
    
    # Common country name mappings (country -> normalized form)
    country_mappings = {
        # Full names to short forms
        'united states': 'us',
        'united states of america': 'us',
        'usa': 'us',
        'america': 'us',
        'american': 'us',
        
        'united kingdom': 'uk',
        'great britain': 'uk',
        'britain': 'uk',
        'british': 'uk',
        'english': 'uk',  # Often refers to UK
        
        'south korea': 'korea',
        'republic of korea': 'korea',
        
        # Demonyms to countries
        'pakistani': 'pakistan',
        'brazilian': 'brazil',
        'mexican': 'mexico',
        'french': 'france',
        'italian': 'italy',
        'argentine': 'argentina',
        'argentinian': 'argentina',
        'turkish': 'turkey',
        'polish': 'poland',
        'canadian': 'canada',
        'moroccan': 'morocco',
        'australian': 'australia',
        'dutch': 'netherlands',
        'greek': 'greece',
        'swedish': 'sweden',
        'bangladeshi': 'bangladesh',
        'bang': 'bangladesh',  # From your logs
        'beng': 'bangladesh',  # From your logs
        'filipino': 'philippines',
        'portuguese': 'portugal',
        'norwegian': 'norway',
        'spanish': 'spain',
        'chilean': 'chile',
        'iranian': 'iran',
        'persian': 'iran',  # Persian often refers to Iran
        'israeli': 'israel',
        'german': 'germany',
        'austrian': 'austria',
        'swiss': 'switzerland',
        'belgian': 'belgium',
        'danish': 'denmark',
        'finnish': 'finland',
        'icelandic': 'iceland',
        'irish': 'ireland',
        'scottish': 'scotland',
        'welsh': 'wales',
        'egyptian': 'egypt',
        'algerian': 'algeria',
        'tunisian': 'tunisia',
        'syrian': 'syria',
        'palestinian': 'palestine',
        'korean': 'korea',
        'chinese': 'china',
        'japanese': 'japan',
        'indian': 'india',
        'russian': 'russia',
        'ukrainian': 'ukraine',
        'colombian': 'colombia',
        'cuban': 'cuba',
        'puerto rican': 'puerto rico',
        'venezuelan': 'venezuela',
        'peruvian': 'peru',
        'ecuadorian': 'ecuador',
        'bolivian': 'bolivia',
        'paraguayan': 'paraguay',
        'uruguayan': 'uruguay',
        'serbian': 'serbia',
        'croatian': 'croatia',
        'bulgarian': 'bulgaria',
        'romanian': 'romania',
        'hungarian': 'hungary',
        'czech': 'czech republic',
        'slovak': 'slovakia',
        'slovenian': 'slovenia',
        'albanian': 'albania',
        'macedonian': 'macedonia',
        'bosnian': 'bosnia',
        'lithuanian': 'lithuania',
        'latvian': 'latvia',
        'estonian': 'estonia',
        'armenian': 'armenia',
        'georgian': 'georgia',
        'azerbaijani': 'azerbaijan',
        'kazakh': 'kazakhstan',
        'uzbek': 'uzbekistan',
        'afghan': 'afghanistan',
        'thai': 'thailand',
        'vietnamese': 'vietnam',
        'malaysian': 'malaysia',
        'indonesian': 'indonesia',
        'singaporean': 'singapore',
        'new zealander': 'new zealand',
        'kiwi': 'new zealand',
        'south african': 'south africa',
        'nigerian': 'nigeria',
        'kenyan': 'kenya',
        'ethiopian': 'ethiopia',
        'ghanaian': 'ghana',
        'senegalese': 'senegal',
        'lebanese': 'lebanon',
        'jordanian': 'jordan',
        'saudi': 'saudi arabia',
        'emirati': 'uae',
        'kuwaiti': 'kuwait',
        'qatari': 'qatar',
        'omani': 'oman',
        'yemeni': 'yemen',
        'libyan': 'libya',
        'sudanese': 'sudan',
        'somali': 'somalia',
        'zimbabwean': 'zimbabwe',
        'zambian': 'zambia',
        'tanzanian': 'tanzania',
        'ugandan': 'uganda',
        'rwandan': 'rwanda',
        'cameroonian': 'cameroon',
        'ivorian': 'ivory coast',
        'malian': 'mali',
        'burkinabe': 'burkina faso',
        'nigerien': 'niger',
        'chadian': 'chad',
        'congolese': 'congo',
        'angolan': 'angola',
        'mozambican': 'mozambique',
        'malagasy': 'madagascar',
        'namibian': 'namibia',
        'botswanan': 'botswana',
    }
    
    # Apply mapping if exists
    if name in country_mappings:
        return country_mappings[name]
    
    # Also check without 's' at the end (for plurals)
    if name.endswith('s') and name[:-1] in country_mappings:
        return country_mappings[name[:-1]]
    
    return name

def countries_match(prediction: str, target: str) -> bool:
    """Check if prediction matches target country, handling variations."""
    # Normalize both
    norm_pred = normalize_country_name(prediction)
    norm_target = normalize_country_name(target)
    
    # Direct match
    if norm_pred == norm_target:
        return True
    
    # Check if one is a substring of the other (for partial matches)
    if len(norm_pred) >= 3 and len(norm_target) >= 3:  # Avoid too short matches
        if norm_pred in norm_target or norm_target in norm_pred:
            return True
    
    # Check if they share significant overlap (for compound names)
    pred_parts = set(norm_pred.split())
    target_parts = set(norm_target.split())
    if pred_parts and target_parts and pred_parts.intersection(target_parts):
        return True
    
    return False

def recall(predictions: Sequence[StrSequence], targets: StrSequence, logger=None) -> List[float]:
    """Compute the recall@k for predicted tokens.

    A prediction is considered correct if it matches the target country,
    handling variations in naming (e.g., demonyms, adjectives).

    Args:
        predictions: List of top-k predicted tokens.
        targets: Target tokens. Must be the same length as `predictions`.
        logger: Optional logger for debugging.

    Returns:
        List of [recall@1, recall@2, ..., recall@k].
    """
    _validate_same_length(predictions=predictions, targets=targets)
    if len(predictions) == 0:
        return []
    
    k = max(map(len, predictions))
    recalls = [0.0] * k
    
    for topk, target in zip(predictions, targets):
        found = False
        for i in range(min(k, len(topk))):
            # Check if any of the top i+1 predictions match the target
            for j in range(i + 1):
                if j < len(topk) and countries_match(topk[j], target):
                    found = True
                    break
            
            if found:
                # Once found, mark all subsequent recall@k as correct
                for j in range(i, k):
                    recalls[j] += 1
                break
        
        if not found and logger:
            logger.info(f"No Correct Prediction: {topk[:min(15, len(topk))]} | Target: {target}")
    
    return [r / len(targets) for r in recalls]

    return [r / len(targets) for r in recalls]