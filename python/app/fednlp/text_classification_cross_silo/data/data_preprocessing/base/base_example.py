class InputExample(object):
    def __init__(self, guid):
        self.guid = guid


class TextClassificationInputExample(InputExample):
    """A single training/test example for simple sequence classification."""

    def __init__(
        self, guid, text_a, text_b=None, label=None, x0=None, y0=None, x1=None, y1=None
    ):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        super().__init__(guid)
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        if x0 is None:
            self.bboxes = None
        else:
            self.bboxes = [[a, b, c, d] for a, b, c, d in zip(x0, y0, x1, y1)]


class SeqTaggingInputExample(InputExample):
    """A single training/test example for simple sequence tagging."""

    def __init__(self, guid, words, labels, x0=None, y0=None, x1=None, y1=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The tokens of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
            x0: (Optional) list. The list of x0 coordinates for each word.
            y0: (Optional) list. The list of y0 coordinates for each word.
            x1: (Optional) list. The list of x1 coordinates for each word.
            y1: (Optional) list. The list of y1 coordinates for each word.
        """
        super().__init__(guid)
        self.words = words
        self.labels = labels
        if x0 is None:
            self.bboxes = None
        else:
            self.bboxes = [[a, b, c, d] for a, b, c, d in zip(x0, y0, x1, y1)]


class SpanExtractionInputExample(InputExample):
    """A single training/test example for simple span extraction."""

    def __init__(
        self,
        guid,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        """
        Constructs a InputExample.
        """
        super().__init__(guid)
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        def _is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(
                    start_position_character + len(answer_text) - 1,
                    len(char_to_word_offset) - 1,
                )
            ]


class Seq2SeqInputExample(InputExample):
    """A single training/test example for simple sequence2sequence."""

    def __init__(self, guid, input_text, target_text):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            input_text: string. The untokenized text of the input sequence.
            target_text: string. The untokenized text of the target sequence.
        """
        super().__init__(guid)
        self.input_text = input_text
        self.target_text = target_text
