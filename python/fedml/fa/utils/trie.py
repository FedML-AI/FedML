import sys

"""
reference: dict_trie
"""


if sys.version_info.major < 3:
    from itertools import imap as map


def _add(root, word, count):
    """Add a word to a trie.

    :arg dict root: Root of the trie.
    :arg str word: A word.
    :arg int count: Multiplicity of `word`.
    """
    node = root

    for char in word:
        if char not in node:
            node[char] = {}
        node = node[char]

    if '' not in node:
        node[''] = 0
    node[''] += count


def _find(root, word):
    """Find the node after following the path in a trie given by {word}.

    :arg dict root: Root of the trie.
    :arg str word: A word.

    :returns dict: The node if found, {} otherwise.
    """
    node = root

    for char in word:
        if char not in node:
            return {}
        node = node[char]

    return node


def _remove(node, word, count):
    """Remove a word from a trie.

    :arg dict node: Current node.
    :arg str word: Word to be removed.
    :arg int count: Multiplicity of `word`, force remove if this is -1.

    :returns bool: True if the last occurrence of `word` is removed.
    """
    if not word:
        if '' in node:
            node[''] -= count
            if node[''] < 1 or count == -1:
                node.pop('')
                return True
        return False

    car, cdr = word[0], word[1:]
    if car not in node:
        return False

    result = _remove(node[car], cdr, count)
    if result:
        if not node[car]:
            node.pop(car)

    return result


def _iterate(path, node, unique):
    """Convert a trie into a list.

    :arg str path: Path taken so far to reach the current node.
    :arg dict node: Current node.
    :arg bool unique: Do not list multiplicities.

    :returns iter: All words in a trie.
    """
    if '' in node:
        if not unique:
            for _ in range(1, node['']):
                yield path
        yield path

    for char in node:
        if char:
            for result in _iterate(path + char, node[char], unique):
                yield result


def _fill(node, alphabet, length):
    """Make a full trie using the characters in {alphabet}.

    :arg dict node: Current node.
    :arg tuple alphabet: Used alphabet.
    :arg int length: Length of the words to be generated.

    :returns iter: Trie containing all words of length {length} over alphabet
        {alphabet}.
    """
    if not length:
        node[''] = 1
        return

    for char in alphabet:
        node[char] = {}
        _fill(node[char], alphabet, length - 1)


def _hamming(path, node, word, distance, cigar):
    """Find all paths in a trie that are within a certain hamming distance of
    {word}.

    :arg str path: Path taken so far to reach the current node.
    :arg dict node: Current node.
    :arg str word: Query word.
    :arg int distance: Amount of allowed errors.

    :returns iter: All words in a trie that have Hamming distance of at most
        {distance} to {word}.
    """
    if distance < 0:
        return
    if not word:
        if '' in node:
            yield (path, distance, cigar)
        return

    car, cdr = word[0], word[1:]
    for char in node:
        if char:
            if char == car:
                penalty = 0
                operation = '='
            else:
                penalty = 1
                operation = 'X'
            for result in _hamming(
                    path + char, node[char], cdr, distance - penalty,
                    cigar + operation):
                yield result


def _levenshtein(path, node, word, distance, cigar):
    """Find all paths in a trie that are within a certain Levenshtein
    distance of {word}.

    :arg str path: Path taken so far to reach the current node.
    :arg dict node: Current node.
    :arg str word: Query word.
    :arg int distance: Amount of allowed errors.

    :returns iter: All words in a trie that have Hamming distance of at most
        {distance} to {word}.
    """
    if distance < 0:
        return
    if not word:
        if '' in node:
            yield (path, distance, cigar)
        car, cdr = '', ''
    else:
        car, cdr = word[0], word[1:]

    # Deletion.
    for result in _levenshtein(path, node, cdr, distance - 1, cigar + 'D'):
        yield result

    for char in node:
        if char:
            # Substitution.
            if car:
                if char == car:
                    penalty = 0
                    operation = '='
                else:
                    penalty = 1
                    operation = 'X'
                for result in _levenshtein(
                        path + char, node[char], cdr, distance - penalty,
                        cigar + operation):
                    yield result
            # Insertion.
            for result in _levenshtein(
                    path + char, node[char], word, distance - 1, cigar + 'I'):
                yield result


class Trie(object):
    """
    A Trie data structure for efficiently storing and searching words.

    Args:
        words (list): List of words to initialize the Trie.

    Attributes:
        root (dict): The root of the Trie.

    Methods:
        __contains__(word):
            Check if a word is present in the Trie.

        __iter__():
            Get an iterator for the words in the Trie.

        list(unique=True):
            Get a list of words in the Trie.

        add(word, count=1):
            Add a word to the Trie.

        get(word):
            Get the count of a word in the Trie.

        remove(word, count=1):
            Remove a word from the Trie.

        has_prefix(word):
            Check if any word in the Trie has a given prefix.

        fill(alphabet, length):
            Fill the Trie with words of a given length using characters from the alphabet.

        all_hamming_(word, distance):
            Find all words in the Trie within a given Hamming distance.

        all_hamming(word, distance):
            Find all words in the Trie within a given Hamming distance (returns words only).

        hamming(word, distance):
            Find the first word in the Trie within a given Hamming distance.

        best_hamming(word, distance):
            Find the best match for a word in the Trie within a given Hamming distance.

        all_levenshtein_(word, distance):
            Find all words in the Trie within a given Levenshtein distance.

        all_levenshtein(word, distance):
            Find all words in the Trie within a given Levenshtein distance (returns words only).

        levenshtein(word, distance):
            Find the first word in the Trie within a given Levenshtein distance.

        best_levenshtein(word, distance):
            Find the best match for a word in the Trie within a given Levenshtein distance.
    """
    def __init__(self, words=None):
        """
        Initialize the Trie class.

        Args:
            words (list): List of words to initialize the Trie.
        """
        self.root = {}

        if words:
            for word in words:
                self.add(word)

    def __contains__(self, word):
        """
        Check if a word is present in the Trie.

        Args:
            word (str): The word to check.

        Returns:
            bool: True if the word is in the Trie, False otherwise.
        """
        return '' in _find(self.root, word)

    def __iter__(self):
        """
        Get an iterator for the words in the Trie.

        Returns:
            Iterator: An iterator object for iterating through words in the Trie.
        """
        return _iterate('', self.root, True)

    def list(self, unique=True):
        """
        Get a list of words in the Trie.

        Args:
            unique (bool): Whether to return unique words only (default is True).

        Returns:
            list: A list of words in the Trie.
        """
        return _iterate('', self.root, unique)

    def add(self, word, count=1):
        """
        Add a word to the Trie.

        Args:
            word (str): The word to add.
            count (int): The count to associate with the word (default is 1).
        """
        _add(self.root, word, count)

    def get(self, word):
        """
        Get the count of a word in the Trie.

        Args:
            word (str): The word to get the count for.

        Returns:
            int: The count of the word in the Trie or None if not found.
        """
        node = _find(self.root, word)
        if '' in node:
            return node['']
        return None

    def remove(self, word, count=1):
        """
        Remove a word from the Trie.

        Args:
            word (str): The word to remove.
            count (int): The count to decrement (default is 1).

        Returns:
            int: The remaining count of the word in the Trie or None if not found.
        """
        return _remove(self.root, word, count)

    def has_prefix(self, word):
        """
        Check if any word in the Trie has a given prefix.

        Args:
            word (str): The prefix to check.

        Returns:
            bool: True if any word has the given prefix, False otherwise.
        """
        return _find(self.root, word) != {}

    def fill(self, alphabet, length):
        """
        Fill the Trie with words of a given length using characters from the alphabet.

        Args:
            alphabet (str): The characters to use for filling.
            length (int): The length of words to generate and add to the Trie.
        """
        _fill(self.root, alphabet, length)

    def all_hamming_(self, word, distance):
        """
        Find all words in the Trie within a given Hamming distance and return detailed results.

        Args:
            word (str): Query word.
            distance (int): Maximum allowed Hamming distance.

        Returns:
            map: A map containing tuples with (word, remaining distance, count).
        """
        return map(
            lambda x: (x[0], distance - x[1], x[2]),
            _hamming('', self.root, word, distance, ''))

    def all_hamming(self, word, distance):
        """
        Find all words in the Trie within a given Hamming distance and return words only.

        Args:
            word (str): Query word.
            distance (int): Maximum allowed Hamming distance.

        Returns:
            map: A map containing words within the specified Hamming distance.
        """
        return map(
            lambda x: x[0], _hamming('', self.root, word, distance, ''))

    def hamming(self, word, distance):
        """
        Find the first word in the Trie within a given Hamming distance.

        Args:
            word (str): Query word.
            distance (int): Maximum allowed Hamming distance.

        Returns:
            str: The first word within the specified Hamming distance or None if not found.
        """
        try:
            return next(self.all_hamming(word, distance))
        except StopIteration:
            return None

    def best_hamming(self, word, distance):
        """
        Find the best match with {word} in a trie using Hamming distance.

        Args:
            word (str): Query word.
            distance (int): Maximum allowed Hamming distance.

        Returns:
            str: Best match with {word}.
        """
        if self.get(word):
            return word

        for i in range(1, distance + 1):
            result = self.hamming(word, i)
            if result is not None:
                return result

        return None

    def all_levenshtein_(self, word, distance):
        """
        Find all words in the Trie within a given Levenshtein distance and return detailed results.

        Args:
            word (str): Query word.
            distance (int): Maximum allowed Levenshtein distance.

        Returns:
            map: A map containing tuples with (word, remaining distance, count).
        """
        return map(
            lambda x: (x[0], distance - x[1], x[2]),
            _levenshtein('', self.root, word, distance, ''))

    def all_levenshtein(self, word, distance):
        """
        Find all words in the Trie within a given Levenshtein distance and return words only.

        Args:
            word (str): Query word.
            distance (int): Maximum allowed Levenshtein distance.

        Returns:
            map: A map containing words within the specified Levenshtein distance.
        """
        return map(
            lambda x: x[0], _levenshtein('', self.root, word, distance, ''))

    def levenshtein(self, word, distance):
        """
        Find the first word in the Trie within a given Levenshtein distance.

        Args:
            word (str): Query word.
            distance (int): Maximum allowed Levenshtein distance.

        Returns:
            str: The first word within the specified Levenshtein distance or None if not found.
        """
        try:
            return next(self.all_levenshtein(word, distance))
        except StopIteration:
            return None

    def best_levenshtein(self, word, distance):
        """
        Find the best match with {word} in a trie using Levenshtein distance.

        Args:
            word (str): Query word.
            distance (int): Maximum allowed Levenshtein distance.

        Returns:
            str: Best match with {word}.
        """
        if self.get(word):
            return word

        for i in range(1, distance + 1):
            result = self.levenshtein(word, i)
            if result is not None:
                return result

        return None
