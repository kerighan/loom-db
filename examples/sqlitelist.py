import sqlite3
import pickle

class SqliteList:
    def __init__(self, db_file, table_name="unnamed", default_value=None):
        self.db_file = db_file
        self.table_name = table_name
        self.default_value = default_value

        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        q = f'CREATE TABLE IF NOT EXISTS "{self.table_name}" (ind INTEGER PRIMARY KEY, value BLOB)'
        print(q)
        cursor.execute(q)
        conn.commit()
        conn.close()

    def __len__(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def __getitem__(self, index):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(f"SELECT value FROM {self.table_name} WHERE ind=?", (index,))
        row = cursor.fetchone()
        conn.close()

        if row is not None:
            return pickle.loads(row[0])
        elif self.default_value is not None:
            return self.default_value
        else:
            raise IndexError("list index out of range")

    def __setitem__(self, index, value):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(f'INSERT OR REPLACE INTO "{self.table_name}" (ind, value) VALUES (?, ?)', (index, pickle.dumps(value)))
        conn.commit()
        conn.close()

    def __delitem__(self, index):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {self.table_name} WHERE ind=?", (index,))
        conn.commit()
        conn.close()

    def append(self, value):
        index = len(self)
        self[index] = value

    def pop(self, index=-1):
        value = self[index]
        del self[index]
        return value
