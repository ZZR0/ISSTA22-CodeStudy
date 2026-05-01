import unittest
from unittest.mock import MagicMock, patch
from AST_Clone_Extractability.main import analyze_nicad

class TestCloneExtractability(unittest.TestCase):

    def setUp(self):
        # Full ground truth covering all 10 instances provided 
        self.ground_truth_instances = [
            {"pcid": "66", "classid": 1, "range": "88-101", "qname": "org.example.junitlauncher.Tracker.executionStarted(...)", "expected": {"Extractable": True, "CFHazard": False, "In": ["testIdentifier"], "Out": []}},
            {"pcid": "86", "classid": 2, "range": "28-40", "qname": "org.example.junitlauncher.jupiter.SharedDataAccessorTest2.testData()", "expected": {"Extractable": True, "CFHazard": False, "In": [], "Out": []}},
            {"pcid": "107", "classid": 2, "range": "28-40", "qname": "org.example.junitlauncher.jupiter.SharedDataAccessorTest1.testData()", "expected": {"Extractable": True, "CFHazard": False, "In": [], "Out": []}},
            {"pcid": "1021", "classid": 18, "range": "459-516", "qname": "org.apache.tools.ant.taskdefs.optional.junit.JUnitTaskTest.testJunitOnCpArguments()", "expected": {"Extractable": False, "CFHazard": True, "HazardType": "break_statement"}},
            {"pcid": "2066", "classid": 42, "range": "297-355", "qname": "org.apache.tools.ant.taskdefs.JarTest.testIndexJarsPlusJarMarker()", "expected": {"Extractable": False, "CFHazard": True, "HazardType": "break_statement"}},
            {"pcid": "5582", "classid": 104, "range": "729-777", "qname": "org.apache.tools.ant.taskdefs.optional.net.FTP.checkRemoteSensitivity(...)", "expected": {"Extractable": False, "CFHazard": True, "HazardType": "return_statement"}},
            {"pcid": "1219", "classid": 25, "range": "137-146", "qname": "org.apache.tools.ant.taskdefs.optional.image.ImageIOTest.testFailOnError()", "expected": {"Extractable": True, "CFHazard": False, "In": [], "Out": []}},
            {"pcid": "1577", "classid": 33, "range": "177-216", "qname": "org.apache.tools.ant.taskdefs.compilers.JavacExternalTest.classpathWithWildcardsIsMovedToBeginning()", "expected": {"Extractable": True, "CFHazard": False, "In": [], "Out": []}},
            {"pcid": "1180", "classid": 21, "range": "78-92", "qname": "org.apache.tools.ant.taskdefs.optional.junitlauncher.JUnitLauncherTaskTest.testFailureStopsBuild()", "expected": {"Extractable": False, "CFHazard": True, "HazardType": "throw_statement"}},
            {"pcid": "1302", "classid": 26, "range": "64-111", "qname": "org.apache.tools.ant.taskdefs.optional.TraXLiaisonTest.testXalan2RedirectViaJDKFactory()", "expected": {"Extractable": False, "CFHazard": True, "HazardType": "throw_statement"}}
        ]

    @patch("AST_Clone_Extractability.main.FileMethodIndex")
    def test_analyze_nicad_against_ground_truth(self, MockIndex):
        mock_idx_instance = MockIndex.return_value
        fake_parser = MagicMock()
        mock_idx_instance.get.return_value = (fake_parser, {})

        print(f"\n{'PCID':<10} | {'Extractable':<12} | {'CFHazard':<10} | {'Result'}")
        print("-" * 50)

        with patch("AST_Clone_Extractability.main.extract_rw_by_region") as mock_rw, \
             patch("AST_Clone_Extractability.main.detect_cf_hazard_detail") as mock_hazard, \
             patch("AST_Clone_Extractability.main.compute_in_out") as mock_io:

            rw_obj = MagicMock()
            rw_obj.locals_in_method = set()
            rw_obj.params_in_method = set()
            rw_obj.vr = {"CloneRegion_pre": set(), "CloneRegion_within": set(), "CloneRegion_post": set()}
            rw_obj.vw = {"CloneRegion_pre": set(), "CloneRegion_within": set(), "CloneRegion_post": set()}
            mock_rw.return_value = rw_obj

            for gt in self.ground_truth_instances:
                with self.subTest(pcid=gt["pcid"]):
                    exp = gt["expected"]
                    mock_io.return_value = (exp.get("In", []), exp.get("Out", []))
                    mock_hazard.return_value = (
                        exp["CFHazard"],
                        {"type": exp.get("HazardType"), "line": 0} if exp["CFHazard"] else None
                    )

                    input_data = [{"classid": gt["classid"], "sources": [{"file": "dummy.java", "range": gt["range"], "qualified_name": gt["qname"], "pcid": gt["pcid"]}]}]
                    results = analyze_nicad(input_data, P=7, R=1)
                    inst = results[0]["sources"][0]

                    # Verification assertions 
                    self.assertEqual(inst["Extractable"], exp["Extractable"], f"PCID {gt['pcid']} extractability mismatch")
                    self.assertEqual(inst["CFHazard"], exp["CFHazard"], f"PCID {gt['pcid']} hazard detection mismatch")
                    if exp["CFHazard"]:
                        self.assertEqual(inst["CFHazard_detail"]["type"], exp["HazardType"])
                    
                    print(f"{gt['pcid']:<10} | {str(inst['Extractable']):<12} | {str(inst['CFHazard']):<10} | PASS ✅")

    def test_decide_extractable_logic(self):
        from AST_Clone_Extractability.feasibility import decide_extractable
        # Case 1: Within thresholds 
        self.assertTrue(decide_extractable(In=["a", "b"], Out=["c"], cf_hazard=False, P=7, R=1))
        # Case 2: Exceeds Parameter threshold (P) 
        self.assertFalse(decide_extractable(In=["a", "b", "c"], Out=[], cf_hazard=False, P=2, R=1))
        # Case 3: Exceeds Return threshold (R) 
        self.assertFalse(decide_extractable(In=["a"], Out=["b", "c"], cf_hazard=False, P=7, R=1))
        # Case 4: Control flow hazard present 
        self.assertFalse(decide_extractable(In=[], Out=[], cf_hazard=True, P=7, R=1))

if __name__ == "__main__":
    unittest.main()