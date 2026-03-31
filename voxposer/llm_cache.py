"""Disk-based cache for LLM API responses.

Ported from VoxPoser/src/LLM_cache.py. Stores (key, value) pairs as
pickle files keyed by SHA-1 hash of the JSON-serialized request params.
"""

import hashlib
import json
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class DiskCache:
    """Persistent disk cache for LLM responses.

    Keys are dicts (model, prompt, temperature, etc.) serialized to JSON.
    Values are response strings stored as pickle files on disk.
    """

    def __init__(self, cache_dir='cache', load_cache=True):
        self.cache_dir = cache_dir
        self.data = {}

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        elif load_cache:
            self._load_cache()

    def _generate_filename(self, key):
        key_str = json.dumps(key, sort_keys=True)
        key_hash = hashlib.sha1(key_str.encode('utf-8')).hexdigest()
        return f"{key_hash}.pkl"

    def _load_cache(self):
        count = 0
        for filename in os.listdir(self.cache_dir):
            if not filename.endswith('.pkl'):
                continue
            try:
                with open(os.path.join(self.cache_dir, filename), 'rb') as f:
                    key, value = pickle.load(f)
                    self.data[json.dumps(key, sort_keys=True)] = value
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to load cache file {filename}: {e}")
        if count > 0:
            logger.info(f"Loaded {count} cached LLM responses from {self.cache_dir}")

    def _save_to_disk(self, key, value):
        filename = self._generate_filename(key)
        with open(os.path.join(self.cache_dir, filename), 'wb') as f:
            pickle.dump((key, value), f)

    def __setitem__(self, key, value):
        str_key = json.dumps(key, sort_keys=True)
        self.data[str_key] = value
        self._save_to_disk(key, value)

    def __getitem__(self, key):
        str_key = json.dumps(key, sort_keys=True)
        return self.data[str_key]

    def __contains__(self, key):
        str_key = json.dumps(key, sort_keys=True)
        return str_key in self.data

    def __repr__(self):
        return f"DiskCache(entries={len(self.data)}, dir='{self.cache_dir}')"
