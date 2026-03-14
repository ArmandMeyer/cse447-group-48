#!/usr/bin/env python
import os
import json
import string
import random
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    Character-level n-gram model with backoff for next-character prediction.
    Trains on NLTK corpora (Gutenberg + Brown) and uses n-grams from order 5
    down to 1, backing off to shorter contexts when higher-order counts are
    unavailable.
    """

    MAX_N = 5  # highest n-gram order to build

    def __init__(self):
        # ngrams[n][context] = {char: count, ...}
        # context is a string of length n-1
        self.ngrams = {}
        # unigram fallback: {char: count}
        self.unigrams = defaultdict(int)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @classmethod
    def load_training_data(cls):
        """Return a single long string of training text from NLTK corpora."""
        try:
            import ssl
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
            except AttributeError:
                pass

            import nltk
            corpora = ('gutenberg', 'brown', 'udhr', 'cess_esp', 'floresta', 'mac_morpho', 'indian', 'machado', 'sinica_treebank', 'jeita')
            for corpus_id in corpora:
                nltk.download(corpus_id, quiet=True)

            text = ""
            for corpus_id in corpora:
                try:
                    corpus = getattr(nltk.corpus, corpus_id)
                    if hasattr(corpus, 'raw'):
                        if corpus_id == 'udhr':
                            for fileid in corpus.fileids():
                                text += corpus.raw(fileid)
                        else:
                            try:
                                text += corpus.raw()
                            except:
                                try:
                                    for fileid in corpus.fileids():
                                        text += corpus.raw(fileid)
                                except:
                                    pass
                except Exception as e:
                    pass

            print(f'Loaded {len(text):,} characters from NLTK corpora')
            return [text]
        except Exception as e:
            print(f'NLTK load failed ({e}), falling back to ascii printable chars as dummy corpus')
            return []

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # strip trailing newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run_train(self, data, work_dir):
        """
        Build n-gram frequency tables from the training corpus.
        data is a list of strings (usually just one large string).
        """
        # Initialise tables for orders 2..MAX_N  (order 1 = unigram handled separately)
        for n in range(2, self.MAX_N + 1):
            self.ngrams[n] = defaultdict(lambda: defaultdict(int))

        total_chars = 0
        for text in data:
            for i, ch in enumerate(text):
                self.unigrams[ch] += 1
                total_chars += 1
                # For each n-gram order, record context → next char
                for n in range(2, self.MAX_N + 1):
                    if i >= n - 1:
                        context = text[i - (n - 1): i]
                        self.ngrams[n][context][ch] += 1

        print(f'Trained on {total_chars:,} characters')
        print(f'Unigram vocab size: {len(self.unigrams)}')
        for n in range(2, self.MAX_N + 1):
            print(f'  {n}-gram contexts: {len(self.ngrams[n]):,}')

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_next(self, context):
        """
        Predict the 3 most likely next characters after `context`.
        Uses backoff from MAX_N down to unigram.
        """
        guesses = []
        
        # Try from longest context down to bigram
        for n in range(self.MAX_N, 1, -1):
            if len(context) >= n - 1:
                ctx = context[-(n - 1):]
                if ctx in self.ngrams.get(n, {}):
                    candidates = sorted(self.ngrams[n][ctx].items(), key=lambda x: -x[1])
                    for ch, _ in candidates:
                        if ch not in guesses:
                            guesses.append(ch)
                        if len(guesses) >= 3:
                            return guesses

        # Unigram fallback
        if self.unigrams:
            candidates = sorted(self.unigrams.items(), key=lambda x: -x[1])
            for ch, _ in candidates:
                if ch not in guesses:
                    guesses.append(ch)
                if len(guesses) >= 3:
                    return guesses

        # Last resort: random printable ASCII
        for ch in string.ascii_letters:
            if ch not in guesses:
                guesses.append(ch)
            if len(guesses) >= 3:
                return guesses
                
        return guesses

    def run_pred(self, data):
        preds = []
        for inp in data:
            top_guesses = self.predict_next(inp)
            # Pad to exactly 3 guesses if needed
            while len(top_guesses) < 3:
                candidate = random.choice(string.ascii_letters)
                if candidate not in top_guesses:
                    top_guesses.append(candidate)
            preds.append(''.join(top_guesses[:3]))
        return preds

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, work_dir):
        checkpoint = {
            'unigrams': dict(self.unigrams),
            'ngrams': {
                str(n): {ctx: dict(chars) for ctx, chars in table.items()}
                for n, table in self.ngrams.items()
            }
        }
        path = os.path.join(work_dir, 'model.checkpoint')
        with open(path, 'wt', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False)
        print(f'Model saved to {path}')

    @classmethod
    def load(cls, work_dir):
        path = os.path.join(work_dir, 'model.checkpoint')
        with open(path, encoding='utf-8') as f:
            checkpoint = json.load(f)

        model = cls()
        model.unigrams = defaultdict(int, checkpoint['unigrams'])
        model.ngrams = {}
        for n_str, table in checkpoint['ngrams'].items():
            n = int(n_str)
            model.ngrams[n] = defaultdict(lambda: defaultdict(int))
            for ctx, chars in table.items():
                model.ngrams[n][ctx] = defaultdict(int, chars)

        print(f'Model loaded from {path}')
        return model


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)

    elif args.mode == 'test':
        
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)

    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))