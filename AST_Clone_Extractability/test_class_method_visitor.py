# test_class_method_visitor.py
import unittest
from collections import defaultdict

from java_treesitter_parser import JavaTreeSitterParser
from java_class_method_visitor import JavaClassMethodVisitor
from util_ast import iter_descendants


INPUT_SAMPLE_SRC = """
class InputSample {
    void foo(String str) {
        System.out.println(str);
        String var1 = str;
        baz(var1);
    }
    class NestedClass {
        void bar(String parm) {
            System.out.println("bar - " + parm);
            baz(parm);
        }
    }

    void baz(String str) {
        System.out.println(str + "baz");
    }

    void outer() {
        Runnable r = new Runnable() {
            public void run() {
                innerCall();
            }
        };
        after();
    }
}
""".lstrip()


class RecordingVisitor(JavaClassMethodVisitor):
    """
    Minimal test visitor that records classes, methods, and per-method call counts.
    It relies on JavaClassMethodVisitor.run() to orchestrate traversal and supply
    method_info to visit_method(...).
    """
    def __init__(self, parser):
        super().__init__(parser)
        self.classes = []                                 # order of classes encountered
        self.methods_by_class = defaultdict(list)         # class_name -> [method names]
        self.qualified_methods = []                       # ["InputSample.foo", ...]
        self.calls_by_qualified = {}                      # "InputSample.foo" -> int
        self._class_count = 0
        self._method_count = 0

    def enter_class(self, class_name, node):
        self._class_count += 1
        self.classes.append(class_name)

    def visit_method(self, class_name, method_info, node):
        self._method_count += 1

        qname = method_info["qualified"]
        mname = method_info["name"]

        # record ownership and order
        self.qualified_methods.append(qname)
        self.methods_by_class[class_name].append(mname)

        # count method_invocation nodes inside this method
        calls = sum(1 for ch in iter_descendants(node) if ch.type == "method_invocation")
        self.calls_by_qualified[qname] = calls

    def exit_class(self, class_name, node):
        pass


class TestClassMethodVisitor(unittest.TestCase):
    def setUp(self):
        self.parser = JavaTreeSitterParser(INPUT_SAMPLE_SRC)
        self.visitor = RecordingVisitor(self.parser)
        self.visitor.run()

    def test_class_counts(self):
        # Expect exactly two classes: InputSample and NestedClass (in DFS order)
        self.assertEqual(self.visitor._class_count, 2)
        self.assertEqual(self.visitor.classes, ["InputSample", "NestedClass"])

    def test_method_counts(self):
        # Expect five methods total (foo, baz, outer, run (anonymous inner), and NestedClass.bar)
        self.assertEqual(self.visitor._method_count, 5)

    def test_methods_by_class(self):
        # Methods owned by InputSample (order by appearance)
        self.assertEqual(
            self.visitor.methods_by_class["InputSample"],
            ["foo", "baz", "outer", "run"],
        )
        # Methods owned by NestedClass
        self.assertEqual(
            self.visitor.methods_by_class["NestedClass"],
            ["bar"],
        )

    def test_qualified_names(self):
        expected = {
            "InputSample.foo",
            "InputSample.baz",
            "InputSample.outer",
            "InputSample.run",       # run() from the anonymous Runnable is attributed to the outer class
            "NestedClass.bar",
        }
        self.assertSetEqual(set(self.visitor.qualified_methods), expected)

    def test_call_counts(self):
        # Expected call counts derived from the sample:
        # foo: println + baz = 2
        # baz: println = 1
        # outer: innerCall (inside anonymous class) + after = 2
        # run (anonymous Runnable): innerCall = 1
        # NestedClass.bar: println + baz = 2
        expected_calls = {
            "InputSample.foo": 2,
            "InputSample.baz": 1,
            "InputSample.outer": 2,
            "InputSample.run": 1,
            "NestedClass.bar": 2,
        }
        self.assertEqual(self.visitor.calls_by_qualified, expected_calls)


if __name__ == "__main__":
    unittest.main()
