# -*- coding: utf-8 -*-

"""
Note: When deploying the application to a online webserver, this method may need to be changed.
Boning Dong, CY Xu (cxu@ucsb.edu)
"""

import os
from datetime import datetime
import shutil
from typing import Tuple
import time

FILE_PERSIST_TIME_MINS = 0.1


def file_recycle_thread(file_manager):
    print("recycling uploaded files ...")
    keys_to_delete = []
    for key, record in file_manager.upload_file_records.items():
        dt = record.create_time - datetime.now()
        if dt.seconds >= FILE_PERSIST_TIME_MINS * 60:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        # delete the file
        record = file_manager.upload_file_records[key]
        os.remove(record.file_path)
        # delete the record
        del file_manager.upload_file_records[key]


class UploadFileRecord:
    def __init__(self, file_path):
        self.file_path = file_path
        self.create_time = datetime.now()


class UploadFileManager:
    def __init__(self, storage_path):
        self.upload_file_records = {}
        self.storage_path = storage_path
        self._create_storage_folder()

    def _create_storage_folder(self):
        if os.path.exists(self.storage_path):
            shutil.rmtree(self.storage_path)
        os.makedirs(self.storage_path)

    def store_uploaded_file(self, key, file):
        file_name = key
        file_path = os.path.join(self.storage_path, file_name)
        # save the record
        file_record = UploadFileRecord(file_path)
        self.upload_file_records[key] = file_record

        # save the file to the storage, remove the old one first
        if os.path.exists(file_path):
            os.remove(file_path)

        file.save(file_path)
        # print ("Saved to path: ", file_path)

    def fetch_uploaded_file_path(self, key) -> Tuple[bool, str]:
        """
        Returns a tuple. [status: bool, file_path: str]
        """
        try:
            record = self.upload_file_records[key]
        except:
            return False, "Record doesn't exits"

        try:
            file_path = record.file_path
        except:
            return False, "Bad record"

        return True, file_path
