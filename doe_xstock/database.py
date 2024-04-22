import math
from pathlib import Path
import re
import sqlite3
from typing import List, Mapping, Union
import numpy as np
import pandas as pd

class SQLiteDatabase:
    def __init__(self, filepath: Union[Path, str], timeout: int = None):
        self.filepath = filepath
        self.__register_adapter()
        self.timeout = timeout
    
    @property
    def filepath(self) -> Union[Path, str]:
        return self.__filepath
    
    @property
    def timeout(self) -> int:
        return self.__timeout
    
    @filepath.setter
    def filepath(self, value: Union[Path, str]):
        self.__filepath = value

    def __get_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.filepath, timeout=self.timeout)
        connection = self.create_functions(connection)

        return connection
    
    @timeout.setter
    def timeout(self, value: int):
        self.__timeout = 60 if value is None else value # seconds

    def __validate_query(self, query: str):
        query = query.replace(',)' ,')')
        
        return query

    def get_table(self, table_name: str) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name}"
        
        return self.query_table(self.__validate_query(query))

    def query(self, query: str) -> list:
        responses = []
        connection = self.__get_connection()
        cursor = connection.cursor()
        
        try:
            queries = query.split(';')
            queries = [q for q in queries if q.strip() != '']
            
            for q in queries:
                cursor.execute(self.__validate_query(q))
                
                if cursor.description is not None:
                    responses.append(cursor.fetchall())
                
                else:
                    pass

                connection.commit()
               
        finally:
            cursor.close()
            connection.close()

        return responses
    
    def query_table_from_file(self, filepath: str, replace: Mapping[str, str] = None) -> pd.DataFrame:
        with open(filepath, 'r') as f:
            query = f.read()
        
        query = self.replace_query(query, replace)

        return self.query_table(query)

    def query_table(self,query) -> pd.DataFrame:
        try:
            connection = self.__get_connection()
            df = pd.read_sql(self.__validate_query(query), connection)
            connection.commit()
        
        finally:
            connection.close()

        return df

    def get_schema(self) -> str:
        try:
            connection = self.__get_connection()
            query = "SELECT * FROM sqlite_master WHERE type IN ('table', 'view')"
            schema = pd.read_sql(self.__validate_query(query), connection)['sql'].tolist()
        finally:
            connection.close()
        
        schema = '\n\n'.join(schema)

        return schema

    def vacuum(self):
        try:
            connection = self.__get_connection()
            connection.execute('VACUUM')
            connection.commit()
        
        finally:
            connection.close()

    def drop(self,name,is_view=False):    
        try:
            connection = self.__get_connection()
            query = f"DROP {'VIEW' if is_view else 'TABLE'} IF EXISTS {name}"
            connection.execute(self.__validate_query(query))
            connection.commit()
        
        finally:
            connection.close()

    def execute_sql_from_file(self, filepath, replace=None):
        with open(filepath,'r') as f:
            queries = f.read()

        queries = self.replace_query(queries, replace)
        
        try:
            connection = self.__get_connection()

            for query in queries.split(';'):
                connection.execute(self.__validate_query(query))
                connection.commit()
        
        finally:
            connection.close()

    def replace_query(self, query: str, replace: Mapping[str, str]) -> str:
        if replace is not None:
            for k, v in replace.items():
                query = query.replace(k, v)
        
        else:
            pass

        return query

    def insert_file(self, filepath: str, table_name: str, **kwargs):
        df = self.read_file(filepath)
        kwargs['values'] = df.to_records(index=False)
        kwargs['fields'] = kwargs.get('fields', list(df.columns))
        kwargs['table_name'] = table_name
        kwargs['on_conflict_fields'] = kwargs.get('on_conflict_fields', None)
        kwargs['ignore_on_conflict'] = kwargs.get('ignore_on_conflict', False)
        self.insert(**kwargs)

    def read_file(self, filepath: str) -> pd.DataFrame:
        reader = {
            'csv': pd.read_csv,
            'pkl': pd.read_pickle,
            'parquet': pd.read_parquet,
        }
        extension = filepath.split('.')[-1]
        method = reader.get(extension,None)

        if method is not None:
            df = method(filepath)
        else:
            raise TypeError(f'Unsupported file extension: .{extension}. Supported file extensions are {list(reader.keys())}')
        
        return df

    def insert(self, table_name: str, fields: List[str], values: List[List[Union[float, int, str]]], on_conflict_fields: List[str] = None, ignore_on_conflict: bool = None):
        ignore_on_conflict = False if ignore_on_conflict is None else ignore_on_conflict
        values = [
            [
                None if isinstance(values[i][j], (int, float)) and math.isnan(values[i][j])\
                    else values[i][j] for j in range(len(values[i]))
            ] for i in range(len(values))
        ]
        fields_placeholder = ', '.join([f'\"{field}\"' for field in fields])
        values_placeholder = ', '.join(['?' for _ in fields])
        query = f"INSERT INTO {table_name} ({fields_placeholder}) VALUES ({values_placeholder})"

        if on_conflict_fields:
            on_conflict_update_fields = [f'\"{field}\"' for field in fields if field not in on_conflict_fields]
            on_conflict_fields_placeholder = ', '.join([f'\"{field}\"' for field in on_conflict_fields])
            on_conflict_placeholder = f'({", ".join(on_conflict_update_fields)}) = '\
                f'({", ".join(["EXCLUDED." + field for field in on_conflict_update_fields])})'

            if ignore_on_conflict or len(set(fields + on_conflict_fields)) == len(on_conflict_fields):
                query = query.replace('INSERT', 'INSERT OR IGNORE')
            
            else:
                query += f"\nON CONFLICT ({on_conflict_fields_placeholder}) DO UPDATE SET {on_conflict_placeholder}"
        
        else:
            pass
        
        try:
            connection = self.__get_connection()
            query = self.__validate_query(query)
            connection.executemany(query, values)
            connection.commit()
        
        finally:
            connection.close()

    def insert_batch(self, query_list: List[str], values_list: List[List[Union[List[Union[float, int, str]], Mapping[str, Union[float, int, str]]]]]):
        connection = self.__get_connection()

        try:
            for query, values in zip(query_list, values_list):
                query = self.__validate_query(query)
                _ = connection.executemany(query, values)
            
            connection.commit()
            
        finally:
            connection.close()

    def __register_adapter(self):
        sqlite3.register_adapter(np.int64,lambda x: int(x))
        sqlite3.register_adapter(np.int32,lambda x: int(x))
        sqlite3.register_adapter(np.datetime64,lambda x: np.datetime_as_string(x,unit='s').replace('T', ' '))

    def create_functions(self, connection: sqlite3.Connection):
        connection.create_function('REGEXP', 2, self.__regexp)

        return connection
    
    def __regexp(self, expr: str, item: str) -> bool:
        reg = re.compile(expr)
        
        return reg.search(item) is not None