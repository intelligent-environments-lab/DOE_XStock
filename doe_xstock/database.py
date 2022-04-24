import math
import sqlite3
import numpy as np
import pandas as pd

class SQLiteDatabase:
    def __init__(self,filepath):
        self.filepath = filepath
        self.__register_adapter()
    
    @property
    def filepath(self):
        return self.__filepath
    
    @filepath.setter
    def filepath(self,filepath):
        self.__filepath = filepath

    def __get_connection(self):
        return sqlite3.connect(self.filepath)

    def __validate_query(self,query):
        query = query.replace(',)',')')
        return query

    def get_table(self,table_name):
        query = f"SELECT * FROM {table_name}"
        return self.query_table(self.__validate_query(query))

    def query(self,query):
        responses = []
        conn = self.__get_connection()
        cur = conn.cursor()
        
        try:
            queries = query.split(';')
            queries = [q for q in queries if q.strip() != '']
            
            for q in queries:
                cur.execute(self.__validate_query(q))
                
                if cur.description is not None:
                    responses.append(cur.fetchall())
                else:
                    pass

                conn.commit()
               
        finally:
            cur.close()
            conn.close()

        return responses

    def query_table(self,query):
        try:
            connection = self.__get_connection()
            df = pd.read_sql(self.__validate_query(query),connection)
            connection.commit()
        finally:
            connection.close()

        return df

    def get_schema(self):
        try:
            connection = self.__get_connection()
            query = "SELECT * FROM sqlite_master WHERE type IN ('table', 'view')"
            schema = pd.read_sql(self.__validate_query(query),connection)['sql'].tolist()
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

    def execute_sql_from_file(self,filepath):
        with open(filepath,'r') as f:
            queries = f.read()
        
        try:
            connection = self.__get_connection()

            for query in queries.split(';'):
                connection.execute(self.__validate_query(query))
                connection.commit()
        
        finally:
            connection.close()

    def insert_file(self,filepath,table_name,**kwargs):
        df = self.read_file(filepath)
        kwargs['values'] = df.to_records(index=False)
        kwargs['fields'] = kwargs.get('fields',list(df.columns))
        kwargs['table_name'] = table_name
        kwargs['on_conflict_fields'] = kwargs.get('on_conflict_fields',None)
        kwargs['ignore_on_conflict'] = kwargs.get('ignore_on_conflict',False)
        self.insert(**kwargs)

    def read_file(self,filepath):
        reader = {
            'csv':pd.read_csv,
            'pkl':pd.read_pickle,
            'parquet':pd.read_parquet,
        }
        extension = filepath.split('.')[-1]
        method = reader.get(extension,None)

        if method is not None:
            df = method(filepath)
        else:
            raise TypeError(f'Unsupported file extension: .{extension}. Supported file extensions are {list(reader.keys())}')
        
        return df

    def insert(self,table_name,fields,values,on_conflict_fields=None,ignore_on_conflict=False):
        values = [
            [
                None if isinstance(values[i][j],(int,float)) and math.isnan(values[i][j])\
                    else values[i][j] for j in range(len(values[i]))
            ] for i in range(len(values))
        ]
        fields_placeholder = ', '.join([f'\"{field}\"' for field in fields])
        values_placeholder = ', '.join(['?' for _ in fields])
        query = f"""
        INSERT INTO {table_name} ({fields_placeholder}) VALUES ({values_placeholder})
        """

        if on_conflict_fields:
            on_conflict_update_fields = [f'\"{field}\"' for field in fields if field not in on_conflict_fields]
            on_conflict_fields_placeholder = ', '.join([f'\"{field}\"' for field in on_conflict_fields])
            on_conflict_placeholder = f'({", ".join(on_conflict_update_fields)}) = '\
                f'({", ".join(["EXCLUDED." + field for field in on_conflict_update_fields])})'

            if ignore_on_conflict or len(set(fields+on_conflict_fields)) == len(on_conflict_fields):
                query = query.replace('INSERT','INSERT OR IGNORE')
            else:
                query += f"ON CONFLICT ({on_conflict_fields_placeholder}) DO UPDATE SET {on_conflict_placeholder}"
        
        else:
            pass
        
        try:
            connection = self.__get_connection()
            query = self.__validate_query(query)
            connection.executemany(query,values)
            connection.commit()
        finally:
            connection.close()

    def __register_adapter(self):
        sqlite3.register_adapter(np.int64,lambda x: int(x))
        sqlite3.register_adapter(np.int32,lambda x: int(x))
        sqlite3.register_adapter(np.datetime64,lambda x: np.datetime_as_string(x,unit='s').replace('T',' '))