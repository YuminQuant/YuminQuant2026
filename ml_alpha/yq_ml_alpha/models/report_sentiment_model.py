"""DFZQ report 86 RPTF: title log term-frequency -> earnings revision."""
from __future__ import annotations

from yq_ml_alpha.output import artifact_io

from collections import Counter
from importlib.metadata import version
import json
import re

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel


_POS = frozenset(("d", "vn", "t", "vd", "an", "ad", "v", "a", "nz", "nr", "n"))
_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            import jieba
            import jieba.posseg
        except ImportError as exc:
            raise ImportError("RPTF requires jieba==0.42.1; install the ml_alpha 'text' extra") from exc
        if version("jieba") != "0.42.1":
            raise RuntimeError("RPTF tokenizer is pinned to jieba==0.42.1")
        tokenizer = jieba.Tokenizer()
        # Jieba.initialize normally writes a pickle under the OS temp directory.
        # Build the prefix dictionary in memory instead; no cache anywhere.
        with tokenizer.get_dict_file() as dictionary:
            tokenizer.FREQ, tokenizer.total = tokenizer.gen_pfdict(dictionary)
        tokenizer.initialized = True
        _TOKENIZER = jieba.posseg.POSTokenizer(tokenizer)
    return _TOKENIZER


def _tokens(frame):
    tokenizer = _tokenizer()
    result = []
    for title, name, code in frame[["report_title", "name", "ts_code"]].itertuples(index=False, name=None):
        text = str(title)
        if name:
            text = text.replace(str(name), "")
        text = text.replace(str(code), "").replace(str(code).split(".")[0], "")
        text = re.sub(r"[\x00-\x1f]+", " ", text).strip(" ：:，,（）()")
        result.append([token.word for token in tokenizer.cut(text)
                       if token.flag in _POS and token.word.strip()])
    return result


class ReportTermFrequencyAlphaModel(AlphaModel):
    @classmethod
    def check_dependencies(cls):
        # Fail before reading years of source data, without initializing a
        # dictionary or creating an OS-temp cache.
        from importlib.util import find_spec

        missing = [name for name in ("jieba", "scipy", "xgboost") if find_spec(name) is None]
        if missing:
            raise ImportError(f"RPTF requires ml_alpha text dependencies; missing: {', '.join(missing)}")
        if version("jieba") != "0.42.1":
            raise RuntimeError("RPTF tokenizer is pinned to jieba==0.42.1")

    def fit(self, train_data, valid_data, context):
        import xgboost as xgb
        from scipy.stats import norm, rankdata

        if context.model_search:
            raise ValueError("RPTF uses fixed parameters, not a search configuration")
        tokens = _tokens(train_data)
        keep = np.asarray([bool(words) for words in tokens])
        target = train_data[context.label_column].to_numpy(dtype=np.float64)
        keep &= np.isfinite(target)
        tokens = [words for words, usable in zip(tokens, keep) if usable]
        target = target[keep]
        if len(target) < 2:
            raise ValueError("RPTF requires at least two labeled, nonempty report titles")
        self.target_values = np.unique(target)
        self.target_quantiles = (rankdata(target, method="average") - 0.5) / len(target)
        order = np.argsort(target, kind="mergesort")
        _, first = np.unique(target[order], return_index=True)
        self.target_quantiles = self.target_quantiles[order][first]
        counts = Counter(word for words in tokens for word in words)
        params = dict(context.model_params)
        size = int(params.pop("max_title_features", 1000))
        self.prediction_batch_size = int(params.pop("prediction_batch_size", 4096))
        if self.prediction_batch_size <= 0:
            raise ValueError("prediction_batch_size must be positive")
        if size <= 0:
            raise ValueError("max_title_features must be positive")
        self.vocabulary = {word: index for index, (word, _) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:size])}
        if not self.vocabulary:
            raise ValueError("empty training vocabulary")
        self.tokenizer_version = version("jieba")
        self.estimator = xgb.XGBRegressor(**params)
        x = self._matrix(tokens)
        y = norm.ppf(np.interp(target, self.target_values, self.target_quantiles))
        print(f"ml-alpha rptf fit reports={len(y)} vocabulary={x.shape[1]} "
              f"matrix_mib={x.nbytes / 2**20:.1f}", flush=True)
        self.estimator.fit(x, y, verbose=False)
        self.info = {"train_reports": len(y), "vocabulary_size": len(self.vocabulary),
                     "tokenizer": f"jieba=={self.tokenizer_version}",
                     "xgboost_version": xgb.__version__, "params": params,
                     "train_mse": float(np.mean((self.estimator.predict(x) - y) ** 2))}
        del x
        if not valid_data.empty:
            valid_tokens = _tokens(valid_data)
            valid_keep = np.asarray([bool(words) for words in valid_tokens])
            raw = valid_data[context.label_column].to_numpy(dtype=np.float64)
            valid_keep &= np.isfinite(raw)
            if valid_keep.any():
                vy = norm.ppf(np.interp(raw[valid_keep], self.target_values, self.target_quantiles))
                self.info["valid_reports"] = int(valid_keep.sum())
                prediction = self._predict_tokens(
                    [words for words, usable in zip(valid_tokens, valid_keep) if usable])
                self.info["valid_mse"] = float(np.mean((prediction - vy) ** 2))

    def _matrix(self, tokens):
        # Numeric zero is a word's absence, not an XGBoost missing value.
        # Fill directly to avoid an additional CSR and corpus-size Python lists.
        matrix = np.zeros((len(tokens), len(self.vocabulary)), dtype=np.float32)
        for row, words in enumerate(tokens):
            counts = Counter(self.vocabulary[word] for word in words if word in self.vocabulary)
            if counts:
                matrix[row, list(counts)] = np.log1p(list(counts.values()))
        return matrix

    def _predict_tokens(self, tokens):
        parts = []
        for start in range(0, len(tokens), self.prediction_batch_size):
            parts.append(self.estimator.predict(
                self._matrix(tokens[start:start + self.prediction_batch_size])))
        return np.concatenate(parts)

    def predict(self, data, context):
        if version("jieba") != self.tokenizer_version:
            raise ValueError("jieba version changed since training; retrain or restore the tokenizer")
        result = np.full(len(data), np.nan, dtype=np.float32)
        if not data.empty:
            tokens = _tokens(data)
            keep = np.asarray([bool(words) for words in tokens])
            if keep.any():
                result[keep] = self._predict_tokens(
                    [words for words, usable in zip(tokens, keep) if usable])
        return pd.Series(result, index=data.index)

    def write_diagnostics(self, context):
        artifact_io.mkdir(context.artifact_dir, parents=True, exist_ok=True)
        info = context.artifact_dir / "model_info.json"
        vocab = context.artifact_dir / "vocabulary.json"
        artifact_io.write_text(info, json.dumps(self.info, ensure_ascii=False, indent=2), encoding="utf-8")
        artifact_io.write_text(vocab, json.dumps(self.vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
        return [info, vocab]
