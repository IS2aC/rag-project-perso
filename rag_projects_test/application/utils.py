from datetime import datetime
import os, pandas as pd
from enum import Enum

BACKLOG_FILE = "backlog.txt"

class BacklogAction(str, Enum):

    """Enumerateur des valeurs des types actions sur les logs"""

    ADD = "ADD"
    DELETE = "DELETE"
    UPDATE = "UPDATE"

class DocEduc:

    """ Objet qui definit une données stocker dans notre stockage objet au besoin de notre RAG system"""

    def __init__(self, course, description, path):
        self.course =  course
        self.description =  description
        self.path = path

class LoggingLogicFunctions: 

    """Ensemble des fonctions  d'acting et checkpoints sur les fichiers backlog.txt & checkpoints.csv"""

    @staticmethod
    def acting_backlog(document:DocEduc, action: BacklogAction):
        file_exists = os.path.isfile(BACKLOG_FILE)
        headers = "log,date,course,description,path_bucket\n"

        with open(BACKLOG_FILE, "a") as f:
            if not file_exists:
                f.write(headers)


            date_now =  datetime.now()
            line = f"{action},{date_now},{document.course},{document.description},{document.path}\n"
            f.write(line)

    @staticmethod
    def acting_checkpoints():
        # already file will exist
        # read backlog.txt
        df_backlog =  pd.read_csv(BACKLOG_FILE, sep = ',', engine="python")
        df_backlog_sort_date =  df_backlog.sort_values('date')

        # most recent information about path_bucket
        columns_without_date =  ['log', 'course', 'description', 'path_bucket']
        df_backlog_recent = df_backlog_sort_date.groupby('path_bucket').last().reset_index()

        # select columns
        df_backlog_recent =  df_backlog_recent[columns_without_date]

        # # drop duplicates
        df_backlog_drop_duplicates =  df_backlog_recent.drop_duplicates()

        # drop delete
        df_backlog_drop_duplicates =  df_backlog_drop_duplicates[df_backlog_drop_duplicates['log'] == 'ADD']

        # # write as checkpoints.csv
        df_backlog_drop_duplicates.to_csv('checkpoints.csv', index =  False)

        