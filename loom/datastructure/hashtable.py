import mmh3
from numpy import max as np_max
from numpy import nonzero


class BaseHashmap:
    def _get_header_fields(self):
        return {}

    def _remove_database_reference(self):
        if hasattr(self, "_db"):
            del self._db

    def _add_database_reference(self, db):
        self._db = db

    def _hash(self, key, seed=0):
        if not isinstance(key, str):
            key = str(key)
        return mmh3.hash(key, seed=seed, signed=False)

    def get(self, key, res=None):
        try:
            return self.lookup(key)
        except KeyError:
            return res

    def __contains__(self, key):
        return self.contains(key)

    def __getitem__(self, key):
        return self.lookup(key)

    def __setitem__(self, key, data):
        data[self.key] = key
        self.insert(data)

    def __delitem__(self, key):
        self.delete(key)


class Hashmap(BaseHashmap):
    def __init__(
        self, dataset, key,
        growth_factor=2, p_init=10, probe_factor=.5,
        n_bloom_filters=10, bloom_seed=0, cache_len=100000
    ):
        self.key = key

        # hashtable parametrization
        self.p_init = p_init
        self.probe_factor = probe_factor
        self.growth_factor = growth_factor
        self.n_bloom_filters = n_bloom_filters
        self.bloom_seed = bloom_seed

        # cache management
        self.cache_len = cache_len

        self.dataset = dataset
        self.dstruct_name = f"{dataset.name}_LHT"
        self.tables_id_key = f"{self.dstruct_name}_tables_id"
        self._block_id_key = f"{self.dstruct_name}_block_id"
        self._bloom_id_key = f"{self.dstruct_name}_bloom_id"
        self._bloom_filter_key = f"{self.dstruct_name}_bloom_filter"

    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------

    def _get_header_fields(self):
        return {
            f"{self._block_id_key}": "uint64",
            f"{self._bloom_id_key}": "uint64",
        }

    def _initialize(self):
        # create header and arrays
        self._block_id = self._db.header[self._block_id_key]
        self._positions = self._db.create_array(self.tables_id_key, "uint64")
        self._bloom = self._db.create_array(self._bloom_filter_key, "uint8")

        if self.cache_len > 0:
            from lru import LRU
            self.cache = LRU(self.cache_len)

        # if not yet initialized
        if self._block_id == 0:
            # create array of hashtables positions
            self._block_id = self._positions.new_block(32)
            self._db.header[self._block_id_key] = self._block_id
            # allocate first table
            capacity = self._get_capacity(self.p_init)
            table_id = self.dataset.new_block(capacity)
            self._positions.set_value(self._block_id, 0, table_id)
            self.p_last = self.p_init
            self._load_tables_id()

            if self.n_bloom_filters > 0:
                # create array of bloom filters positions
                self._bloom_id = self._positions.new_block(32)
                self._db.header[self._bloom_id_key] = self._bloom_id
                filter_id = self._bloom.new_block(
                    capacity * self.n_bloom_filters)
                self._positions.set_value(self._bloom_id, 0, filter_id)
                self._load_bloom_filters()
                self.find_lookup_position = self._find_lookup_position_filtered
        else:
            self._load_tables_id()
            self.p_last = np_max(nonzero(self.tables_id)) + self.p_init

            if self.n_bloom_filters > 0:
                self._bloom_id = self._db.header[self._bloom_id_key]
                self._load_bloom_filters()
                self.find_lookup_position = self._find_lookup_position_filtered

        self.get = self.dataset.get
        self.exists = self.dataset.exists
        self.status = self.dataset.status
        self.get_value = self.dataset.get_value

    # -------------------------------------------------------------------------
    # refresh functions
    # -------------------------------------------------------------------------

    def _save_tables_id(self, index, table_id):
        self._positions.set_value(self._block_id, index, table_id)

    def _load_tables_id(self):
        self.tables_id = list(
            self._positions.get_values(self._block_id, 0, 32))

    def _load_bloom_filters(self):
        self.bloom_filters = list(
            self._positions.get_values(self._bloom_id, 0, 32))

    # -------------------------------------------------------------------------
    # utilities
    # -------------------------------------------------------------------------

    def _get_range(self, p,):
        return range(int(round(p * self.probe_factor * self.growth_factor)))

    def _get_capacity(self, p):
        return self.growth_factor**p

    # -------------------------------------------------------------------------
    # positioning functions
    # -------------------------------------------------------------------------

    def _find_insert_or_lookup_position(self, key, key_hash):
        try:
            p, position = self.find_lookup_position(key, key_hash)
            return p, position, False
        except KeyError:
            p, position = self._find_insert_position(key, key_hash)
            return p, position, True

    def _find_insert_or_lookup_index(self, key, key_hash):
        try:
            p, position = self.find_lookup_position(key, key_hash)
            table_id = self.tables_id[p - self.p_init]
            return table_id, position, False
        except KeyError:
            p, position = self._find_insert_position(key, key_hash)
            table_id = self.tables_id[p - self.p_init]
            return table_id, position, True

    def _find_insert_position(self, key, key_hash):
        for p in range(self.p_last, self.p_init - 1, -1):
            try:
                p, position = self._find_insert_position_in_table(
                    key, key_hash, p)
                return p, position
            except KeyError:
                continue

        self._create_new_hashtable()
        p, position = self._find_insert_position_in_table(
            key, key_hash, self.p_last)
        return p, position

    def _find_insert_position_in_table(self, key, key_hash, p):
        key_name = self.key

        capacity = self._get_capacity(p)
        bucket = key_hash % capacity
        table_id = self.tables_id[p - self.p_init]
        for i in self._get_range(p):
            position = (bucket + i) % capacity
            status = self.status(table_id, position)
            if status == 1:
                key_current = self.get_value(
                    table_id, position, key_name)
                if key_current != key:
                    continue
                return p, position
            else:
                return p, position
        raise KeyError

    def _find_lookup_position_filtered(self, key, key_hash):
        if self.cache_len > 0:
            p, position = self.cache.get(key, (None, None))
            if p is not None:
                return p, position

        bloom_hash = self._hash(key, self.bloom_seed)
        for p in range(self.p_last, self.p_init - 1, -1):
            bloom_p = self.bloom_filters[p - self.p_init]
            bloom_capacity = self._get_capacity(p) * self.n_bloom_filters
            bucket = bloom_hash % bloom_capacity
            if self._bloom.get_value(bloom_p, bucket) == 0:
                continue
            try:
                p, position = self._find_lookup_position_in_table(
                    key, key_hash, p)
                return p, position
            except KeyError:
                pass
        raise KeyError

    def find_lookup_position(self, key, key_hash):
        if self.cache_len > 0:
            p, position = self.cache.get(key, (None, None))
            if p is not None:
                return p, position

        for p in range(self.p_last, self.p_init - 1, -1):
            try:
                p, position = self._find_lookup_position_in_table(
                    key, key_hash, p)
                return p, position
            except KeyError:
                pass
        raise KeyError

    def _find_lookup_position_in_table(self, key, key_hash, p):
        key_name = self.key

        capacity = self._get_capacity(p)
        bucket = key_hash % capacity
        table_id = self.tables_id[p - self.p_init]
        for i in self._get_range(p):
            position = (bucket + i) % capacity
            status = self.status(table_id, position)
            if status == 1:
                key_current = self.get_value(table_id, position, key_name)
                if key_current != key:
                    continue
                return p, position
            elif status == -1:
                continue
            else:
                raise KeyError
        raise KeyError

    # -------------------------------------------------------------------------
    # create, read, update, delete
    # -------------------------------------------------------------------------

    def _create_new_hashtable(self):
        self.p_last += 1
        capacity = self._get_capacity(self.p_last)
        table_id = self.dataset.new_block(capacity)
        index = self.p_last - self.p_init
        self.tables_id[index] = table_id
        self._save_tables_id(index, table_id)

        # create a new bloom filters
        if self.n_bloom_filters > 0:
            filter_id = self._bloom.new_block(
                capacity * self.n_bloom_filters)
            self._positions.set_value(self._bloom_id, index, filter_id)
            self.bloom_filters[index] = filter_id

    def _insert_in_bloom(self, p, key):
        key_hash = self._hash(key, self.bloom_seed)
        bloom_p = self.bloom_filters[p - self.p_init]
        bloom_capacity = self._get_capacity(p) * self.n_bloom_filters
        bucket = key_hash % bloom_capacity

        bloom_value = self._bloom.get_value(bloom_p, bucket)
        self._bloom.set_value(bloom_p, bucket, bloom_value + 1)

    def insert(self, data):
        key = data[self.key]
        key_hash = self._hash(key)

        p, position, _ = self._find_insert_or_lookup_position(
            key, key_hash)
        table_id = self.tables_id[p - self.p_init]

        self.dataset.set(table_id, position, data)
        if self.n_bloom_filters > 0:
            self._insert_in_bloom(p, key)
        if self.cache_len > 0:
            self.cache[key] = p, position

    def lookup(self, key):
        key_hash = self._hash(key)
        p, position = self.find_lookup_position(key, key_hash)
        table_id = self.tables_id[p - self.p_init]
        return self.get(table_id, position)

    def delete(self, key):
        key_hash = self._hash(key)
        p, position = self.find_lookup_position(key, key_hash)
        table_id = self.tables_id[p - self.p_init]
        self.dataset.delete(table_id, position)
        # remove from cache
        if self.cache_len > 0:
            if key in self.cache:
                del self.cache[key]

    # -------------------------------------------------------------------------
    # overload
    # -------------------------------------------------------------------------

    def __iter__(self):
        for p in range(self.p_last - self.p_init + 1):
            table_id = self.tables_id[p]
            capacity = self._get_capacity(p + self.p_init)
            for i in range(capacity):
                if self.exists(table_id, i):
                    yield self.get(table_id, i)
