import unittest
import pandas as pd
import tempfile
import os
import random
from src.features import csv_file

class test_csv_file(unittest.TestCase):

    def test_object_cannot_fail_to_save_dataframe_to_csv_format(self):
        temp_dir = tempfile.TemporaryDirectory()
        csv_path = os.path.join(temp_dir.name, f"{random.randint(1000, 9999)}.csv")
        df = pd.DataFrame({
            'Время': [f'2024-{random.randint(1, 12):02d}-01' for _ in range(3)],
            'Сделки': [random.randint(1, 500) for _ in range(3)],
            'Кол🤑во': [random.randint(1, 100) for _ in range(3)]
        })
        cf = csv_file(csv_path)
        cf.save(df)
        # Проверка ПЕРЕД cleanup
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        temp_dir.cleanup()
        self.assertTrue(
            file_exists,
            "Object cannot save dataframe to CSV file with non-zero size"
        )

    def test_object_cannot_fail_to_load_dataframe_from_existing_csv(self):
        temp_dir = tempfile.TemporaryDirectory()
        csv_path = os.path.join(temp_dir.name, f"{random.randint(1000,9999)}.csv")
        original_df = pd.DataFrame({
            'Test_Колонка': [random.randint(-100,200) for _ in range(3)],
            '🎯Значения': [f'text_{random.randint(1,100)}' for _ in range(3)]
        })
        original_df.to_csv(csv_path, index=False, encoding='utf-8')
        cf = csv_file(csv_path)
        loaded_df = cf.load_df()
        temp_dir.cleanup()
        self.assertTrue(
            len(loaded_df) == 3 and 'Test_Колонка' in loaded_df.columns,
            "Object cannot load dataframe with correct shape and column names"
        )

    def test_object_cannot_fail_to_return_self_after_save_operation(self):
        temp_dir = tempfile.TemporaryDirectory()
        csv_path = os.path.join(temp_dir.name, f"{random.randint(1000,9999)}.csv")
        df = pd.DataFrame({
            'Col1': [random.randint(1,50) for _ in range(2)],
            'Col2': [random.randint(51,100) for _ in range(2)]
        })
        cf = csv_file(csv_path)
        result = cf.save(df)
        temp_dir.cleanup()
        self.assertTrue(
            result is cf,
            "Object cannot return self instance after save operation"
        )

    def test_object_cannot_fail_to_preserve_utf8_encoding_during_save_and_load(self):
        temp_dir = tempfile.TemporaryDirectory()
        csv_path = os.path.join(temp_dir.name, f"{random.randint(1000,9999)}.csv")
        df = pd.DataFrame({
            'Русский': ['привет', 'мир', 'тест'],
            'Эмодзи': ['🤖', '📊', '🎯']
        })
        cf = csv_file(csv_path)
        cf.save(df)
        loaded_df = cf.load_df()
        temp_dir.cleanup()
        self.assertTrue(
            loaded_df['Русский'].iloc[0] == 'привет' and loaded_df['Эмодзи'].iloc[0] == '🤖',
            "Object cannot preserve UTF-8 characters during save and load cycle"
        )

if __name__ == "__main__":
    unittest.main()