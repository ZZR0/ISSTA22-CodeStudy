import unittest
import os
from pathlib import Path

# Import your actual classes
from index_methods import FileMethodIndex, MethodRecord
from java_treesitter_parser import JavaTreeSitterParser

class TestIndexMethodsRealFile(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        Ensures the ./data/ directory and A.java exist.
        """
        self.data_dir = Path("./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "A.java"

        # If A.java doesn't exist, we create a temporary one for testing.
        self.created_temp_file = False
        if not self.file_path.exists():
            self.created_temp_file = True
            java_code = """
            package com.example;

            public class A {
                public int calculateSum(int a, int b) {
                    return a + b;
                }
                
                private void printMessage(String msg) {
                    System.out.println(msg);
                }
            }
            """
            self.file_path.write_text(java_code, encoding="utf-8")
        
        # Initialize the indexer
        self.indexer = FileMethodIndex()

    def tearDown(self):
        """
        Clean up the test environment.
        Removes A.java only if it was created specifically for this test.
        """
        # if self.created_temp_file and self.file_path.exists():
            # self.file_path.unlink()
            # Remove the ./data/ directory if it's now empty
            # if not any(self.data_dir.iterdir()):
                # self.data_dir.rmdir()

    def test_file_indexing(self):
        """
        Verifies that FileMethodIndex correctly reads A.java, parses it, 
        and extracts the methods into MethodRecord objects.
        """
        
        
        # 1. Run the indexer on the real file
        parser, methods = self.indexer.get(str(self.file_path))

        # 2. Verify the parser was created and loaded the file
        self.assertIsInstance(parser, JavaTreeSitterParser)
        self.assertTrue(len(parser.src_text) > 0)

        # 3. Verify the methods dictionary is populated
        # Note: The exact keys depend on your util_ast logic. 
        # Assuming it formats as "ClassName.methodName":
        expected_keys = ["A.calculateSum", "A.printMessage"]
        
        for key in expected_keys:
            self.assertIn(key, methods, f"Expected method {key} was not found in the index.")
            
            # Verify the structure of the MethodRecord
            record = methods[key]
            self.assertIsInstance(record, MethodRecord)
            self.assertEqual(record.class_name, "A")
            self.assertIsNotNone(record.node, "AST Node should be stored in the record")
            
            # Verify we can extract the original source code from the node
            method_source = parser.text_of(record.node)
            self.assertIn(record.method_info["name"], method_source)

    # def test_caching_mechanism(self):
    #     """
    #     Verifies that requesting A.java multiple times does not re-read the file,
    #     but returns the exact same cached instances.
    #     """
    #     path_str = str(self.file_path)
        
    #     # First request (reads file, parses AST)
    #     parser1, methods1 = self.indexer.get(path_str)
        
    #     # Second request (should hit the cache)
    #     parser2, methods2 = self.indexer.get(path_str)
        
    #     # Use assertIs to check that they are the exact same objects in memory
    #     self.assertIs(parser1, parser2, "Cache failed: A new parser was created.")
    #     self.assertIs(methods1, methods2, "Cache failed: A new methods dictionary was created.")

if __name__ == "__main__":
    unittest.main()