import ast
import unittest

from reproreduce.hunt import Operation, Program, TensorSpec


class HuntProgramTests(unittest.TestCase):
    def test_serialization_and_identity_are_deterministic(self):
        program = Program(
            inputs=(TensorSpec("x", (2, 3)), TensorSpec("y", (2, 3))),
            operations=(Operation.create("add", ("x", "y")),),
            output="v0",
        )
        clone = Program(
            inputs=(TensorSpec("x", (2, 3)), TensorSpec("y", (2, 3))),
            operations=(Operation.create("add", ("x", "y")),),
            output="v0",
        )
        self.assertEqual(program.serialize(), clone.serialize())
        self.assertEqual(program.identity, clone.identity)

    def test_source_is_valid_and_reconstructs_program(self):
        program = Program(
            inputs=(TensorSpec("x", (2, 3)), TensorSpec("y", (2, 3))),
            operations=(
                Operation.create("transpose", ("x",), dim0=0, dim1=1),
                Operation.create("sin", ("v0",)),
            ),
            output="v1",
        )
        source = program.to_source()
        ast.parse(source)
        self.assertIn("x.transpose(0, 1)", source)
        self.assertIn("torch.sin(v0)", source)

    def test_operation_kwargs_are_canonical(self):
        operation = Operation.create("reshape", ("x",), shape=(2, 3))
        self.assertEqual(operation.kwargs_dict(), {"shape": (2, 3)})
        self.assertEqual(operation.to_dict()["name"], "reshape")


if __name__ == "__main__":
    unittest.main()
