import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, confusion_matrix, precision_score, recall_score

from bayes import *

np.random.seed(999)

# These are hardcoded but you can change to your directory.
# I'll change this during testing.
neg_dir = 'review_polarity/txt_sentoken/neg'
pos_dir = 'review_polarity/txt_sentoken/pos'

NEG3 = ['quest', 'camelot', 'warner', 'bros', 'feature', 'length', 'fully', 'animated',
        'attempt', 'steal', 'clout', 'disney', 'cartoon', 'empire', 'mouse', 'reason',
        'worried', 'recent', 'challenger', 'throne', 'fall', 'promising', 'flawed',
        'century', 'fox', 'production', 'anastasia', 'disney', 'hercules', 'lively',
        'cast', 'colorful', 'palate', 'beat', 'hands', 'came', 'time', 'crown', 'best',
        'piece', 'animation', 'year', 'contest', 'quest', 'camelot', 'pretty', 'dead',
        'arrival', 'magic', 'kingdom', 'mediocre', 'pocahontas', 'keeping', 'score',
        'isn', 'nearly', 'dull', 'story', 'revolves', 'adventures', 'free', 'spirited',
        'kayley', 'voiced', 'jessalyn', 'gilsig', 'early', 'teen', 'daughter', 'belated',
        'knight', 'king', 'arthur', 'round', 'table', 'kayley', 'dream', 'follow',
        'father', 'footsteps', 'gets', 'chance', 'evil', 'warlord', 'ruber', 'gary',
        'oldman', 'round', 'table', 'member', 'gone', 'bad', 'steals', 'arthur',
        'magical', 'sword', 'excalibur', 'accidentally', 'loses', 'dangerous', 'booby',
        'trapped', 'forest', 'help', 'hunky', 'blind', 'timberland', 'dweller', 'garrett',
        'carey', 'elwes', 'headed', 'dragon', 'eric', 'idle', 'don', 'rickles', 'arguing',
        'kayley', 'just', 'able', 'break', 'medieval', 'sexist', 'mold', 'prove', 'worth',
        'fighter', 'arthur', 'quest', 'camelot', 'missing', 'pure', 'showmanship',
        'essential', 'element', 'expected', 'climb', 'high', 'ranks', 'disney',
        'differentiates', 'quest', 'given', 'saturday', 'morning', 'cartoon', 'subpar',
        'animation', 'instantly', 'forgettable', 'songs', 'poorly', 'integrated',
        'computerized', 'footage', 'compare', 'kayley', 'garrett', 'run', 'angry', 'ogre',
        'herc', 'battle', 'hydra', 'rest', 'case', 'characters', 'stink', 'remotely',
        'interesting', 'film', 'race', 'bland', 'end', 'tie', 'win', 'dragon', 'comedy',
        'shtick', 'awfully', 'cloying', 'shows', 'signs', 'pulse', 'fans', 'early',
        'tgif', 'television', 'line', 'thrilled', 'jaleel', 'urkel', 'white', 'bronson',
        'balki', 'pinchot', 'sharing', 'footage', 'scenes', 'nicely', 'realized', 'loss',
        'recall', 'specific', 'actors', 'providing', 'voice', 'talent', 'enthusiastic',
        'paired', 'singers', 'don', 'sound', 'thing', 'like', 'big', 'musical', 'moments',
        'jane', 'seymour', 'celine', 'dion', 'strain', 'mess', 'good', 'aside', 'fact',
        'children', 'probably', 'bored', 'watching', 'adults', 'quest', 'camelot',
        'grievous', 'error', 'complete', 'lack', 'personality', 'personality', 'learn',
        'mess', 'goes', 'long', 'way']
POS11 = ['noticed', 'lately', 'thought', 'pseudo', 'substance', 'hollywood', 'faking',
         'deep', 'meanings', 'films', 'seen', 'movie', 'really', 'enjoyed', 'look',
         'realize', 'missing', 'filmmakers', 'putting', 'rehearsed', 'melodramatic',
         'films', 'evoke', 'strong', 'connotations', 'great', 'film', 'step', 'aside',
         'reflect', 'movie', 'going', 'experience', 'just', 'discover', 'elegantly',
         'presented', 'fluff', 'trying', 'say', 'city', 'angels', 'bad', 'lot', 'going',
         'way', 'faltered', 'underneath', 'seemingly', 'poetic', 'beauty', 'gigantic',
         'hole', 'somebody', 'covered', 'iridescent', 'performances', 'glossy',
         'cinematography', 'predictable', 'ending', 'shattered', 'hopes', 'saw', 'coming',
         'added', 'disappointment', 'hour', 'city', 'angels', 'worth', 'time', 'nicolas',
         'cage', 'seth', 'guardian', 'angel', 'like', 'hundreds', 'likely', 'thousands',
         'millions', 'angels', 'spends', 'eternity', 'watching', 'citizens', 'mortality',
         'humans', 'aware', 'celestial', 'intervention', 'occurs', 'life', 'meg', 'ryan',
         'subdued', 'performance', 'plays', 'maggie', 'doctor', 'begins', 'ponder',
         'exactly', 'fighting', 'fight', 'alive', 'losing', 'patient', 'surgery', 'table',
         'questions', 'envelope', 'maggie', 'seth', 'angel', 'oversee', 'patient',
         'transition', 'afterlife', 'immediately', 'captivated', 'doctor', 'begins',
         'following', 'observing', 'maggie', 'falling', 'love', 'everyday', 'angels',
         'quickly', 'learn', 'humans', 'experience', 'human', 'sensations', 'taste',
         'touch', 'ability', 'make', 'seen', 'desire', 'seth', 'adoration', 'resist',
         'eventually', 'does', 'appear', 'maggie', 'quite', 'regularly', 'thing', 'taboo',
         'angelic', 'community', 'angels', 'interestingly', 'presented', 'dressed',
         'black', 'reminiscent', 'hitmen', 'traditional', 'glowing', 'white', 'entities',
         'nice', 'touch', 'like', 'mere', 'attempt', 'uniqueness', 'cage', 'wonderfully',
         'versatile', 'actor', 'think', 'face', 'raising', 'arizona', 'happen', 'combo',
         'slips', 'role', 'heavenly', 'agent', 'quite', 'nicely', 'threatens',
         'sappiness', 'nice', 'ryan', 'pick', 'roles', 'like', 'courage', 'aren',
         'comparable', 'deviate', 'usual', 'intelligently', 'ditzy', 'romantic', 'comedy',
         'roles', 'impressive', 'ryan', 'movie', 'goers', 'rarely', 'chance', 'enjoy',
         'leads', 'impressive', 'job', 'dennis', 'franz', 'grabs', 'interpretation',
         'hospital', 'patient', 'knows', 'meets', 'eye', 'shame', 'going', 'city',
         'angels', 'falters', 'final', 'stages', 'leaving', 'realization', 'emotionally',
         'incredible', 'movie', 'just', 'didn', 'know', 'quite', 'struggle', 'impacting',
         'conclusion', 'wind', 'painful', 'thud', 'exhilarating', 'high', 'filmmakers',
         'know', 'final', 'impression', 'linger', 'remember', 'convey', 'word', 'mouth',
         'telling', 'minutes', 'film', 'glorious', 'masterpiece', 'sure', 'leave',
         'disheartening', 'taste', 'mediocrity', 'mouths', 'based', 'german', 'film',
         'wings', 'desire', 'english', 'title', 'course', 'city', 'angels', 'ninety',
         'percent', 'success', 'make', 'people', 'forgive', 'shortcomings',
         'devastatingly', 'disappointing', 'ending', 'movie', 'goers', 'non', 'cynics',
         'wrapped', 'surreal', 'atmosphere', 'criticism', 'needs', 'criticized',
         'nonetheless', 'city', 'angels', 'beautifully', 'captivating', 'probably',
         'satisfy', 'poetic', 'viewers', 'appreciate', 'delve', 'rich', 'emotional',
         'territories']

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


def training_data():
    neg, pos = load()
    V = vocab(neg, pos)
    vneg = vectorize_docs(neg, V)
    vpos = vectorize_docs(pos, V)
    X = np.vstack([vneg, vpos])
    y = np.vstack([np.zeros(shape=(len(vneg), 1)),
                   np.ones(shape=(len(vpos), 1))]).reshape(-1)
    return V, X, y


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
    V = vocab(neg,pos)
    assert len(V)==38373 # includes unknown

    rs = np.random.RandomState(42) # get same list back each time
    idx = rs.randint(0,len(V),size=100)
    allwords = np.array([*V.keys()])
    subset = allwords[idx]
    # print(sorted(subset))
    assert sorted(subset) == VOCAB_SUBSET100


def test_vectorize_docs():
    neg, pos = load()
    V = vocab(neg,pos)
    vneg = vectorize_docs(neg, V)
    vpos = vectorize_docs(pos, V)
    allwords = np.array([*V.keys()])
    # print(list(allwords[np.where(vneg[3]>1)]))
    assert list(allwords[np.where(vneg[3] > 1)]) == \
           ['animation', 'arthur', 'camelot', 'cartoon', 'disney', 'don', 'dragon',
            'early', 'footage', 'garrett', 'kayley', 'mess', 'personality', 'quest',
            'round', 'table']
    # print(list(allwords[np.where(vpos[5]>1)]))
    assert list(allwords[np.where(vpos[5] > 1)]) == \
           ['african', 'belgian', 'black', 'colonial', 'congo', 'country', 'death',
            'docudrama', 'does', 'ebouaney', 'eriq', 'fall', 'family', 'feel', 'good',
            'government', 'history', 'honest', 'independence', 'known', 'leader',
            'leadership', 'life', 'little', 'lumumba', 'man', 'masters', 'men',
            'minister', 'mnc', 'months', 'nation', 'national', 'nations', 'new', 'pascal',
            'patrice', 'peck', 'period', 'place', 'political', 'position', 'powerful',
            'production', 'right', 'small', 'story', 'strive', 'struggling', 'supporting',
            'time', 'told', 'troops', 'true', 'world', 'young']

def test_vectorize():
    V, X, y = training_data()
    d1 = vectorize(V, words("mostly very funny , the story is quite appealing."))
    d2 = vectorize(V, words("there is already a candidate for the worst of 1997."))
    p = len(V)
    assert len(d1)==p, f"d1 should be 1x{p} but is 1x{len(d1)}"
    assert len(d2)==p, f"d2 should be 1x{p} but is 1x{len(d2)}"
    d1_idx = np.nonzero(d1)
    d2_idx = np.nonzero(d2)
    true_d1_idx = np.array([ 1367, 13337, 26872, 32570 ])
    true_d2_idx = np.array([ 4676, 37932 ])
    assert (d1_idx==true_d1_idx).all(), f"{d1_idx} should be {true_d1_idx}"
    assert (d2_idx==true_d2_idx).all(), f"{d2_idx} should be {true_d2_idx}"


def test_simple_docs_error():
    V, X, y = training_data()
    d1 = vectorize(V, words("mostly very funny, the story is quite appealing."))
    d2 = vectorize(V, words("there is already a candidate for the worst of 1997."))
    p = len(V)
    assert len(d1)==p, f"d1 should be 1x{p} but is 1x{len(d1)}"
    assert len(d2)==p, f"d2 should be 1x{p} but is 1x{len(d2)}"
    y_test = np.array([1,0])

    X_test = np.vstack([d1,d2])
    model = NaiveBayes621()
    model.fit(X, y)
    y_pred = model.predict(X_test)
    accuracy = np.sum(y_test==y_pred) / 2
    # print(f"train accuracy {accuracy}")
    assert accuracy == 1.0, f"Correct = {np.sum(y==y_pred)} / {len(y)} = {100*accuracy:.1f}%"


def test_unknown_words_vectorize():
    V, X, y = training_data()
    d1_words = words("blort blort loved movie but xyrzf abcdefgh not so much")
    d2_words = words("brexit vote postponed")
    d1 = vectorize(V, d1_words)
    d2 = vectorize(V, d2_words)
    p = len(V)
    assert len(d1)==p, f"d1 should be 1x{p} but is 1x{len(d1)}"
    assert len(d2)==p, f"d2 should be 1x{p} but is 1x{len(d2)}"
    assert d1[0]==4, f"d1 should have 4 unknown words"
    assert d2[0]==1, f"d2 should have 1 unknown word"

    d1_idx = np.nonzero(d1)
    d2_idx = np.nonzero(d2)
    true_d1_idx = np.array([ 0, 19965, 22121 ])
    true_d2_idx = np.array([ 0, 25740, 36959 ])
    assert (d1_idx==true_d1_idx).all(), f"{d1_idx} should be {true_d1_idx}"
    assert (d2_idx==true_d2_idx).all(), f"{d2_idx} should be {true_d2_idx}"


def test_unknown_words_training_error():
    V, X, y = training_data()
    # xyzdef and brexit are not in V
    d1 = vectorize(V, words("very good, the story is xyzdef appealing. i also try to recommend excellent films like this"))
    d2 = vectorize(V, words("brexit vote postponed hated movie; a van damme movie has become a painful chore"))
    y_test = np.array([1, 0])

    X_test = np.vstack([d1, d2])
    model = NaiveBayes621()
    model.fit(X, y)
    y_pred = model.predict(X_test)
    accuracy = np.sum(y_test == y_pred) / 2
    # print(f"train accuracy {accuracy}, {y_pred}")
    assert accuracy == 1.0, f"Correct = {np.sum(y_test == y_pred)} / {len(y_test)} = {100 * accuracy:.1f}%"


def test_training_error():
    V, X, y = training_data()
    model = NaiveBayes621()
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = np.sum(y==y_pred) / len(y)
    # print(f"training accuracy {accuracy}")
    assert accuracy > 0.965, f"Correct = {np.sum(y==y_pred)} / {len(y)} = {100*accuracy:.1f}%"


def test_kfold_621():
    # Test just kfold stuff so use sklearn model
    V, X, y = training_data()

    sklearn_accuracies = kfold_CV(GaussianNB(), X, y, k=4)
    true_sklearn_accuracies = np.array([0.638, 0.644, 0.67, 0.63])
    sklearn_avg = np.mean(sklearn_accuracies)
    true_avg = np.mean(true_sklearn_accuracies)
    # print(f"kfold {sklearn_accuracies} vs true {true_sklearn_accuracies}")
    # print(np.abs(sklearn_avg-true_avg))
    assert np.abs(sklearn_avg-true_avg) < 0.003, f"true accuracies {true_sklearn_accuracies} and your kfold {sklearn_accuracies} differ"


def test_kfold_sklearn_vs_621():
    V, X, y = training_data()

    accuracies = kfold_CV(NaiveBayes621(), X, y, k=4)
    sklearn_accuracies = kfold_CV(GaussianNB(), X, y, k=4)
    # print(f"sklearn kfold {sklearn_accuracies} vs yours {accuracies}")

    our_avg = np.mean(accuracies)
    sklearn_avg = np.mean(sklearn_accuracies)

    assert our_avg-sklearn_avg > 0.10, f"sklearn accuracies {sklearn_avg} and your kfold {our_avg} differ"

