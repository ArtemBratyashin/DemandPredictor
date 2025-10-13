import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import random
from src.excel_file import excel_file

class TestExcelFile(unittest.TestCase):

    def test_object_cannot_fail_to_convert_time_column_to_datetime(self):
        temp_dir = tempfile.TemporaryDirectory()
        excel_path = os.path.join(temp_dir.name, f"{random.randint(1,10000)}.xlsx")
        df = pd.DataFrame({
            'Время': [f"20{random.randint(20,25)}-{random.randint(1,12):02d}-01" for _ in range(4)],
            'Сделки': [random.randint(0,200) for _ in range(4)]
        })
        df.to_excel(excel_path, sheet_name="Данные", index=False)
        ef = excel_file(excel_path, "Данные")
        ef.process_time()
        temp_dir.cleanup()
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(ef._excel_file__data['Время']),
            "Object cannot convert 'Время' column to pandas datetime type"
        )

    def test_object_cannot_fail_to_generate_valid_sin_seasonality_feature(self):
        temp_dir = tempfile.TemporaryDirectory()
        excel_path = os.path.join(temp_dir.name, f"{random.randint(1,10000)}.xlsx")
        df = pd.DataFrame({
            'Время': [f"20{random.randint(20,23)}-{random.randint(1,12):02d}-01" for _ in range(4)],
            'Сделки': [random.randint(-100, 200) for _ in range(4)]
        })
        df.to_excel(excel_path, sheet_name="Данные", index=False)
        ef = excel_file(excel_path, "Данные")
        ef.add_sin_seasonality(period=12)
        temp_dir.cleanup()
        self.assertTrue(
            'sin_season' in ef._excel_file__data.columns and
            (ef._excel_file__data['sin_season'] <= 1).all() and (ef._excel_file__data['sin_season'] >= -1).all(),
            "Object cannot generate valid 'sin_season' feature within [-1, 1] range"
        )

    def test_object_cannot_fail_to_generate_valid_cos_seasonality_feature(self):
        temp_dir = tempfile.TemporaryDirectory()
        excel_path = os.path.join(temp_dir.name, f"{random.randint(1,10000)}.xlsx")
        df = pd.DataFrame({
            'Время': [f"20{random.randint(22,27)}-{random.randint(1,12):02d}-01" for _ in range(4)],
            'Сделки': [random.randint(-150, 200) for _ in range(4)]
        })
        df.to_excel(excel_path, sheet_name="Данные", index=False)
        ef = excel_file(excel_path, "Данные")
        ef.add_cos_seasonality(period=12)
        temp_dir.cleanup()
        self.assertTrue(
            'cos_season' in ef._excel_file__data.columns and
            (ef._excel_file__data['cos_season'] <= 1).all() and (ef._excel_file__data['cos_season'] >= -1).all(),
            "Object cannot generate valid 'cos_season' feature within [-1, 1] range"
        )

    def test_object_cannot_fail_to_create_lag_for_numeric_columns(self):
        temp_dir = tempfile.TemporaryDirectory()
        excel_path = os.path.join(temp_dir.name, f"{random.randint(1,10000)}.xlsx")
        df = pd.DataFrame({
            'Время': [f"2024-{random.randint(1,12):02d}-01" for _ in range(4)],
            'Сделки': [random.randint(1, 500) for _ in range(4)],
            '☆Другое': [random.randint(1, 999) for _ in range(4)]
        })
        df.to_excel(excel_path, sheet_name="Данные", index=False)
        ef = excel_file(excel_path, "Данные")
        ef.add_lags()
        temp_dir.cleanup()
        self.assertTrue(
            'Сделки_lag1' in ef._excel_file__data.columns and ef._excel_file__data['Сделки_lag1'].isnull().sum() == 1,
            "Object cannot create lagged feature 'Сделки_lag1' with correct missing values"
        )

    def test_object_cannot_fail_to_remove_nan_and_reset_index(self):
        temp_dir = tempfile.TemporaryDirectory()
        excel_path = os.path.join(temp_dir.name, f"{random.randint(1,10000)}.xlsx")
        df = pd.DataFrame({
            'Время': [f"2022-{random.randint(1,12):02d}-01" for _ in range(4)],
            'Сделки': [random.randint(1, 500) for _ in range(4)]
        })
        df.to_excel(excel_path, sheet_name="Данные", index=False)
        ef = excel_file(excel_path, "Данные")
        ef.add_lags()
        df_clean = ef.prepared_data()
        temp_dir.cleanup()
        self.assertTrue(
            not df_clean.isnull().any().any() and df_clean.index.min() == 0,
            "Object cannot remove NaN values or reset DataFrame index"
        )

if __name__ == "__main__":
    unittest.main()