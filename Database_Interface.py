import mysql.connector



class DatabaseInterface:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None

    def connect(self):
        self.connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        self.cursor = self.connection.cursor()

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def create_tables(self):
        self.connect()
        # SQL statements to create the tables
        sql_queries = [
            """
            CREATE TABLE IF NOT EXISTS known_person (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255),
                embedding BLOB
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS unknown_person (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255),
                embedding BLOB
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS known_person_occurrence (
                id VARCHAR(255),
                camid VARCHAR(255),
                time DATETIME,
                FOREIGN KEY (id) REFERENCES known_person(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS unknown_person_occurrence (
                id VARCHAR(255),
                camid VARCHAR(255),
                time DATETIME,
                FOREIGN KEY (id) REFERENCES unknown_person(id)
            )
            """
            ]    
            
        for query in sql_queries:
            self.cursor.execute(query)
        self.connection.commit()
        self.disconnect()

    def insert(self, table, id, name=None, embedding=None, camid=None, time=None):
        self.connect()
        if table == 'known_person':
            sql_query = "INSERT INTO known_person (id, name, embedding) VALUES (%s, %s, %s)"
            values = (id, name, embedding)
        elif table == 'unknown_person':
            sql_query = "INSERT INTO unknown_person (id, name, embedding) VALUES (%s, %s, %s)"
            values = (id, name, embedding)
        elif table == 'known_person_occurrence':
            sql_query = "INSERT INTO known_person_occurrence (id, camid, time) VALUES (%s, %s, %s)"
            values = (id, camid, time)
        elif table == 'unknown_person_occurrence':
            sql_query = "INSERT INTO unknown_person_occurrence (id, camid, time) VALUES (%s, %s, %s)"
            values = (id, camid, time)
        else:
            raise ValueError("Invalid table name")
        
        self.cursor.execute(sql_query, values)
        self.connection.commit()
        self.disconnect()

    def get_known_db(self):
        self.connect()
        sql_query = "SELECT * FROM known_person"
        self.cursor.execute(sql_query)
        results = self.cursor.fetchall()
        self.disconnect()
        return results

    def get_unknown_db(self):
        self.connect()
        sql_query = "SELECT * FROM unknown_person"
        self.cursor.execute(sql_query)
        results = self.cursor.fetchall()
        self.disconnect()
        return results

    def get_known_occurrence_id(self, id):
        self.connect()
        sql_query = "SELECT * FROM occurrence_known WHERE id = %s"
        self.cursor.execute(sql_query, (id,))
        results = self.cursor.fetchall()
        self.disconnect()
        return results

    def get_unknown_occurrence_id(self, id):
        self.connect()
        sql_query = "SELECT * FROM occurrence_unknown WHERE id = %s"
        self.cursor.execute(sql_query, (id,))
        results = self.cursor.fetchall()
        self.disconnect()
        return results