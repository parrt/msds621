import numpy as np
from sklearn.naive_bayes import GaussianNB

from bayes import *

# These are hardcoded but you can change to your directory.
# I'll change this during testing.
neg_dir = 'review_polarity/txt_sentoken/neg'
pos_dir = 'review_polarity/txt_sentoken/pos'

NEG3 = ['disconnect', 'phone', 'line', 'don', 'accept', 'charges', 'avoid', 'wretched',
        'melodramatic', 'sisterhood', 'dramedy', 'hanging', 'figured', 'needed', 'touch',
        'feminine', 'hanging', 'like', 'ideal', 'opportunity', 'film', 'features',
        'incredible', 'palate', 'female', 'talent', 'capability', 'camera', 'brought',
        'mind', 'sparkling', 'gems', 'sleepless', 'seattle', 'unsung', 'heroes', 'meg',
        'ryan', 'diane', 'keaton', 'lisa', 'kudrow', 'play', 'trio', 'sisters',
        'separated', 'career', 'judgments', 'family', 'ties', 'reunite', 'father',
        'walter', 'matthau', 'admitted', 'hospital', 'alzheimer', 'disease', 'read',
        'like', 'optimum', 'opportunity', 'rekindle', 'relationship', 'reflect',
        'poignancy', 'past', 'script', 'sisters', 'delia', 'nora', 'ephron',
        'exasperating', 'shapeless', 'dreck', 'teeming', 'emotional', 'fakery', 'hanging',
        'overall', 'effect', 'tele', 'marketer', 'pestering', 'hours', 'don', 'option',
        'doing', 'title', 'suggests', 'half', 'hour', 'ephron', 'sisters', 'use',
        'telephone', 'conversations', 'basis', 'character', 'development', 'annoying',
        'ineffective', 'device', 'cell', 'phones', 'ring', 'minutes', 'hurriedly',
        'rushes', 'leaving', 'marginal', 'time', 'frustrated', 'viewer', 'relate',
        'sisters', 'issues', 'problems', 'hanging', 'apple', 'pie', 'felt', 'getting',
        'mere', 'crust', 'story', 'granted', 'genuine', 'moments', 'film', 'help',
        'establish', 'remainder', 'strained', 'emotions', 'inferior', 'dramatic', 'muck',
        'outrageous', 'strategy', 'hanging', 'series', 'largely', 'unrealized',
        'attempts', 'character', 'development', 'expected', 'exhibit', 'compassion',
        'courtesy', 'sisters', 'join', 'melodramatic', 'finale', 'able', 'identify',
        'eve', 'ryan', 'open', 'caring', 'daughter', 'stayed', 'father', 'moved',
        'forward', 'pursue', 'impending', 'career', 'georgia', 'keaton', 'eldest',
        'daughter', 'celebrating', 'fifth', 'year', 'anniversary', 'magazine', 'called',
        'georgia', 'maddy', 'kudrow', 'soap', 'opera', 'actress', 'spends', 'time',
        'contemplating', 'possible', 'path', 'stardom', 'nursing', 'dog', 'ryan',
        'convincing', 'performance', 'diverting', 'cuteness', 'agreeable', 'aspects',
        'hanging', 'kudrow', 'delightfully', 'eccentric', 'kilter', 'airhead', 'phoebe',
        'friends', 'totally', 'wasted', 'ditto', 'keaton', 'serving', 'double', 'shift',
        'star', 'director', 'time', 'slot', 'difficult', 'priority', 'juggle', 'frenzy',
        'apparent', 'chick', 'flick', 'distressing', 'lack', 'chuckles', 'reliable',
        'matthau', 'reduced', 'chaotic', 'shtick', 'given', 'character', 'situation',
        'depressing', 'amusing', 'peak', 'form', 'humor', 'hanging', 'represented',
        'matthau', 'nasty', 'quips', 'ryan', 'eternal', 'battle', 'aforementioned',
        'pooch', 'swallow', 'pill', 'accounts', 'chuckles', 'expel', 'film', 'curiosity',
        'suddenly', 'tweaked', 'discover', 'promising', 'star', 'studded', 'approach',
        'turn', 'viciously', 'sour', 'really', 'mystery', 'predictable', 'melodramatic',
        'filth', 'hanging', 'certainly', 'fault', 'actresses', 'pin', 'screenplay',
        'attempts', 'clear', 'vital', 'issues', 'minutes', 'spending', 'rest', 'running',
        'time', 'annoying', 'flurry', 'phone', 'conversations', 'certainly', 'far',
        'label', 'rewarding', 'experience', 'hanging', 'enjoyable', 'wrong', 'number',
        'beginning']
POS11 = ['rented', 'brokedown', 'palace', 'night', 'blind', 'having', 'heard', 'enjoyed',
         'immensely', 'despite', 'flaws', 'wishing', 'experience', 'suggest', 'reserving',
         'judgement', 'movie', 'viewing', 'entirety', 'easy', 'task', 'superficially',
         'bears', 'unfortunate', 'necessarily', 'unintended', 'resemblance', 'movies',
         'notably', 'return', 'paradise', 'midnight', 'express', 'result', 'nearly',
         'review', 'brokedown', 'palace', 'subsequently', 'read', 'hopelessly',
         'entangled', 'making', 'obvious', 'comparisons', 'consequence', 'nearly',
         'universal', 'condemnation', 'shame', 'fine', 'film', 'view', 'movie', 'let',
         'say', 'attempt', 'portray', 'nightmarish', 'reality', 'world', 'criminal',
         'justice', 'midnight', 'express', 'completely', 'moral', 'dilemma',
         'examination', 'meaning', 'friendship', 'humanity', 'heart', 'return',
         'paradise', 'view', 'film', 'compared', 'source', 'joseph', 'conrad',
         'acclaimed', 'novel', 'lord', 'jim', 'problematically', 'basic', 'storyline',
         'familiar', 'american', 'teenage', 'girls', 'vacation', 'sentenced', 'spend',
         'lives', 'thai', 'prison', 'drug', 'smuggling', 'obvious', 'set', 'involving',
         'suave', 'man', 'shadowy', 'criminal', 'conspiracy', 'corrupt', 'world',
         'justice', 'girls', 'alice', 'claire', 'danes', 'darlene', 'kate', 'beckinsale',
         'life', 'long', 'buddies', 'planned', 'high', 'school', 'graduation', 'trip',
         'hawaii', 'secretly', 'changed', 'destination', 'exotic', 'thailand', 'telling',
         'parents', 'hotel', 'sight', 'seeing', 'includes', 'sneaking', 'luxury', 'hotel',
         'sip', 'expensive', 'drinks', 'poolside', 'caught', 'trying', 'charge', 'wrong',
         'room', 'minor', 'transgression', 'later', 'come', 'haunt', 'saved', 'hotel',
         'security', 'charming', 'friendly', 'australian', 'nick', 'daniel', 'lapaine',
         'takes', 'care', 'polished', 'execution', 'girl', 'scam', 'proceeds', 'separate',
         'girls', 'make', 'smooth', 'moves', 'alice', 'darlene', 'alarm', 'bells',
         'going', 'viewers', 'present', 'nick', 'slick', 'stories', 'don', 'add', 'girls',
         'course', 'naive', 'notice', 'long', 'happens', 'anticipating', 'inevitable',
         'disappearance', 'fast', 'talking', 'smuggler', 'arrest', 'teenaged', 'sitting',
         'ducks', 'airport', 'route', 'hong', 'kong', 'caught', 'holding', 'bag',
         'literally', 'containing', 'heroin', 'just', 'predictably', 'thai', 'police',
         'courts', 'meting', 'injustice', 'trusting', 'tourists', 'prison', 'bound',
         'long', 'stretch', 'left', 'unanswered', 'red', 'herring', 'issue', 'girls',
         'willing', 'accomplice', 'need', 'ready', 'answer', 'suggest', 'closer',
         'scrutiny', 'bell', 'hop', 'girl', 'fleabag', 'hotel', 'door', 'comfortable',
         'life', 'closed', 'girls', 'families', 'turn', 'desperation', 'noiresque',
         'expatriate', 'lawyer', 'fixer', 'yankee', 'hank', 'pullam', 'thai', 'born',
         'partner', 'wife', 'recurring', 'element', 'movie', 'tension', 'appearance',
         'reality', 'expressed', 'film', 'tag', 'lines', 'trust', 'hank', 'exception',
         'seasoned', 'movie', 'goers', 'familiar', 'pullman', 'oeuvre', 'surprises',
         'remainder', 'movie', 'smorgasbord', 'intriguing', 'themes', 'incompletely',
         'explored', 'short', 'hand', 'fashion', 'lou', 'diamond', 'phillips', 'instance',
         'plays', 'delightfully', 'sinister', 'callous', 'dea', 'agent', 'appearing',
         'casually', 'accommodating', 'hank', 'withholds', 'vital', 'information',
         'crucial', 'moments', 'wider', 'conspiracy', 'inherently', 'powerful',
         'somewhat', 'tired', 'premise', 'film', 'offers', 'parts', 'riveting',
         'courtroom', 'drama', 'prison', 'story', 'potential', 'character', 'study',
         'american', 'teens', 'relationship', 'constitutes', 'friendship', 'result',
         'reasonably', 'engaging', 'suspenseful', 'girls', 'interaction', 'hank',
         'investigation', 'various', 'trials', 'hearings', 'offering', 'hope', 'release',
         'delivering', 'tension', 'does', 'foredoomed', 'possibility', 'escape',
         'brokedown', 'palace', 'major', 'flaw', 'creators', 'tendency', 'like', 'time',
         'constrained', 'tourists', 'frequent', 'trips', 'fascinating', 'alleys',
         'reverse', 'direction', 'half', 'way', 'return', 'story', 'main', 'avenue',
         'brokedown', 'palace', 'wouldn', 'good', 'movie', 'believe', 'writers',
         'director', 'bigger', 'game', 'succeeded', 'main', 'theme', 'movie', 'like',
         'proffered', 'location', 'freedom', 'permutations', 'ultimately', 'sub',
         'themes', 'considered', 'window', 'dressing', 'young', 'pretty', 'alice',
         'danes', 'old', 'soul', 'wild', 'streetwise', 'teenager', 'thirst', 'freedom',
         'adventure', 'presented', 'perfect', 'blend', 'yin', 'yang', 'dark', 'light',
         'cautious', 'best', 'friend', 'darlene', 'beckinsale', 'clear', 'eyed',
         'straightforward', 'alice', 'complex', 'friend', 'comes', 'poorer', 'background',
         'reputation', 'getting', 'trouble', 'lost', 'trust', 'including', 'father',
         'darlene', 'life', 'track', 'aimed', 'college', 'marriage', 'kids', 'career',
         'suburban', 'home', 'middle', 'age', 'fulfillment', 'alice', 'uncertain',
         'unfocused', 'yearning', 'poignant', 'scene', 'film', 'shows', 'darlene',
         'shouting', 'open', 'moat', 'visitors', 'friends', 'relatives', 'home', 'lives',
         'continue', 'limbo', 'tellingly', 'alice', 'present', 'included', 'just',
         'revealing', 'different', 'personalities', 'alice', 'dar', 'come', 'thailand',
         'openness', 'delight', 'alice', 'face', 'doesn', 'read', 'simple', 'naivet',
         'way', 'stands', 'stretches', 'friend', 'ride', 'small', 'boat', 'reaching',
         'sun', 'really', 'drinking', 'believes', 'freedom', 'dar', 'remains', 'seated',
         'shade', 'brokedown', 'palace', 'begins', 'admission', 'alice', 'guilt', 'tape',
         'recording', 'sent', 'hank', 'unintentional', 'alice', 'fault', 'responsible',
         'persuading', 'friend', 'lie', 'parents', 'sneak', 'away', 'safety', 'hawaii',
         'perils', 'thailand', 'try', 'petty', 'scam', 'places', 'clutches', 'evil',
         'nick', 'case', 'misses', 'point', 'inevitable', 'confusion', 'film',
         'beginning', 'darlene', 'obligingly', 'reminds', 'alice', 'culpability',
         'prison', 'dar', 'course', 'willing', 'dupe', 'view', 'confers', 'innocence',
         'mind', 'coercion', 'reluctant', 'alice', 'accompany', 'hong', 'kong', 'placed',
         'police', 'custody', 'place', 'naive', 'confession', 'sealed', 'fate', 'dar',
         'innocent', 'doesn', 'matter', 'alice', 'subject', 'movie', 'journey',
         'personal', 'freedom', 'way', 'treated', 'unsympathetic', 'portrait', 'shallow',
         'american', 'culture', 'created', 'girls', 'half', 'baked', 'sensibilities',
         'materialistic', 'goals', 'end', 'culture', 'like', 'representatives', 'yankee',
         'hank', 'dar', 'father', 'man', 'knows', 'grease', 'wheels', 'proves',
         'impotent', 'government', 'face', 'girl', 'tragedy', 'thailand', 'culture',
         'contrary', 'opinion', 'comes', 'better', 'comparison', 'amazes', 'reviewers',
         'argued', 'point', 'extremes', 'believe', 'filmmaker', 'view', 'thai', 'culture',
         'vastly', 'different', 'american', 'necessarily', 'inferior', 'thai', 'sole',
         'exceptions', 'corrupt', 'official', 'spiteful', 'prison', 'spy', 'uniformly',
         'consistent', 'behavior', 'true', 'principles', 'girls', 'shown', 'treated',
         'better', 'certainly', 'worse', 'native', 'born', 'prison', 'stark', 'contrast',
         'probable', 'reality', 'hellhole', 'relatively', 'clean', 'sunlit', 'prison',
         'authorities', 'demanded', 'good', 'hygiene', 'provided', 'medical', 'care',
         'needed', 'hard', 'manual', 'labor', 'consisted', 'picking', 'grass', 'thai',
         'guards', 'authoritarian', 'certainly', 'routinely', 'sadistic', 'thai',
         'justice', 'reasoning', 'thai', 'judges', 'appeal', 'hearing', 'film',
         'penultimate', 'scene', 'devastating', 'logic', 'morality', 'freedom', 'faces',
         'brokedown', 'palace', 'explores', 'extreme', 'freedom', 'body', 'freedom',
         'spirit', 'settle', 'remain', 'imprisoned', 'entire', 'nation', 'roam',
         'freedom', 'seldom', 'comes', 'price', 'movie', 'tag', 'lines', 'dream', 'far',
         'believe', 'make', 'good', 'case', 'interpretation', 'person', 'attains',
         'freedom', 'incarcerated', 'film', 'end', 'alice', 'finds', 'redemption',
         'salvation', 'acceptance', 'personal', 'responsibility', 'think', 'light',
         'bathing', 'figure', 'assembled', 'prisoners', 'final', 'scene', 'visually',
         'signals', 'fact', 'kate', 'beckinsale', 'character', 'properly', 'likened',
         'released', 'temple', 'bird', 'referred', 'twice', 'film', 'trained', 'fly',
         'cage', 'thai', 'magistrate', 'observed', 'film', 'climactic', 'scene', 'issue',
         'character', 'jamaican', 'prisoner', 'clear', 'freedom', 'achieved', 'oneself',
         'thinks', 'movie', 'character', 'transformed', 'experiences', 'increasingly',
         'cinema', 'landscape', 'littered', 'endless', 'permutations', 'kung', 'woman',
         'female', 'characters', 'virtually', 'indistinguishable', 'male', 'action',
         'figures', 'story', 'modern', 'heroine', 'reading', 'user', 'comments', 'struck',
         'unusual', 'phenomenon', 'person', 'liked', 'movie', 'praised', 'actors',
         'think', 'time', 'came', 'away', 'motion', 'picture', 'possibly', 'hating',
         'raving', 'performances', 'favor', 'rent', 'brokedown', 'palace', 'watch',
         'open', 'mind', 'meets', 'eye']

VOCAB_SUBSET100 = ['acceptable', 'accompanies', 'alek', 'allows', 'amistad', 'amnesia',
                   'anti', 'armored', 'arty', 'atrophied', 'authentically', 'barbecue',
                   'bastille', 'battles', 'beatrice', 'bedtimes', 'bolt', 'bombarded',
                   'braun', 'breathed', 'cavern', 'characers', 'charms', 'cimino',
                   'comely', 'compensating', 'contentious', 'delayed', 'deliveree',
                   'denise', 'dependant', 'deuce', 'disintegrated', 'doom', 'embarassed',
                   'enterprises', 'entrepreneur', 'eurocentrism', 'examinations',
                   'existing', 'exposure', 'fahdlan', 'fer', 'flirts', 'franken', 'gait',
                   'gloat', 'goal', 'groaning', 'groundbreaking', 'homeworld',
                   'hovertank', 'independance', 'inputs', 'instinctively',
                   'invincibility', 'kermit', 'lanai', 'lava', 'lavender', 'libidinous',
                   'locating', 'meshes', 'metamorphoses', 'moff', 'moribund', 'mortal',
                   'neptune', 'observatory', 'onstage', 'orbiting', 'overemotional',
                   'overly', 'paradise', 'paramedic', 'parent', 'paz', 'portion', 'prays',
                   'pseudonym', 'psycholically', 'quinland', 'redcoats', 'robo', 'sacred',
                   'shorten', 'silence', 'sincerely', 'solution', 'straits',
                   'supernaturally', 'taste', 'tryst', 'uneasiness', 'uninterrupted',
                   'walkway', 'wasting', 'won', 'xer', 'yield']

neg = pos = None

def load():
    global neg, pos
    if neg is None or pos is None:
        neg = load_docs(neg_dir)
        pos = load_docs(pos_dir)
    return neg, pos


def test_load():
    neg, pos = load()
    # Pick sample docs to compare
    assert len(neg) == 1000
    assert len(pos) == 1000
    # print(neg[3])
    # print(pos[11])
    assert neg[3]==NEG3
    assert pos[11]==POS11


def test_vocab():
    neg, pos = load()
    V = vocab([neg,pos])
    assert len(V)==38372+1 # Add one for unknown at index 0

    rs = np.random.RandomState(42) # get same list back each time
    idx = rs.randint(0,len(V),size=100)
    allwords = np.array([*V.keys()])
    subset = allwords[idx]
    # print(sorted(subset))
    assert sorted(subset) == VOCAB_SUBSET100


def test_vectorize():
    neg, pos = load()
    V = vocab([neg,pos])
    vneg = vectorize_docs(neg, V)
    vpos = vectorize_docs(pos, V)
    allwords = np.array([*V.keys()])
    # print(list(allwords[np.where(vneg[3]>1)]))
    assert list(allwords[np.where(vneg[3] > 1)]) == \
           ['annoying', 'attempts', 'career', 'certainly', 'character', 'chuckles',
            'conversations', 'daughter', 'development', 'don', 'ephron', 'father', 'film',
            'georgia', 'hanging', 'issues', 'keaton', 'kudrow', 'like', 'matthau',
            'melodramatic', 'minutes', 'opportunity', 'phone', 'ryan', 'sisters', 'star', 'time']
    # print(list(allwords[np.where(vpos[5]>1)]))
    assert list(allwords[np.where(vpos[5] > 1)]) == \
           ['action', 'actor', 'apart', 'ben', 'bruce', 'certainly', 'character',
            'characters', 'day', 'don', 'fall', 'family', 'film', 'films', 'flashbacks',
            'good', 'grown', 'heartbreaking', 'jordan', 'katie', 'life', 'like', 'lives',
            'main', 'make', 'marriage', 'michelle', 'movies', 'nelson', 'nicely',
            'performance', 'pfeiffer', 'picture', 'real', 'realistic', 'really', 'say',
            'script', 'sense', 'shows', 'sophisticated', 'stepmom', 'story', 'strong',
            'therapist', 'things', 'times', 'told', 'touching', 'viewer', 'willis',
            'written', 'year', 'years']


def test_training_error():
    neg, pos = load()
    V = vocab([neg,pos])
    vneg = vectorize_docs(neg, V)
    vpos = vectorize_docs(pos, V)

    X = np.vstack([vneg, vpos])
    y = np.vstack(
        [np.zeros(shape=(len(vneg), 1)), np.ones(shape=(len(vpos), 1))]).reshape(-1)
    model = NaiveBayes621()
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = np.sum(y==y_pred) / len(y)
    assert accuracy > 0.97, f"Correct = {np.sum(y==y_pred)} / {len(y)} = {100*accuracy:.1f}%"


def test_kfold_621():
    neg, pos = load()
    V = vocab([neg,pos])
    vneg = vectorize_docs(neg, V)
    vpos = vectorize_docs(pos, V)

    X = np.vstack([vneg, vpos])
    y = np.vstack(
        [np.zeros(shape=(len(vneg), 1)), np.ones(shape=(len(vpos), 1))]).reshape(-1)
    model = NaiveBayes621()

    accuracies = kfold_CV(model, X, y, k=4)
    true_accuracies = np.array([.836, .80, .806, .786])
    areclose = np.abs(true_accuracies - accuracies) < np.array([.5, .5, .5, .5])
    assert areclose.all(), f"true accuracies {true_accuracies} and yours {accuracies} differ"


def test_kfold_sklearn_vs_621():
    neg, pos = load()
    V = vocab([neg,pos])
    vneg = vectorize_docs(neg, V)
    vpos = vectorize_docs(pos, V)

    X = np.vstack([vneg, vpos])
    y = np.vstack([np.zeros(shape=(len(vneg), 1)), np.ones(shape=(len(vpos), 1))]).reshape(-1)

    accuracies = kfold_CV(NaiveBayes621(), X, y, k=4)

    sklearn_accuracies = kfold_CV(GaussianNB(), X, y, k=4)
    sklearn_true_accuracies = np.array([0.666, 0.678, 0.636, 0.662])

    werebetter = (accuracies - sklearn_true_accuracies) > np.array([.1, .1, .1, .1])
    assert werebetter.all(), f"Your accuracies {accuracies} should be better than sklearn's {sklearn_accuracies}"