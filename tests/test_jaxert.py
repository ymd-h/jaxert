import unittest

import jax
import jax.numpy as jnp
import jaxlib

from jaxert import jax_assert, assert_allclose


class TestAssert(unittest.TestCase):
    def test_nojit(self):
        jax_assert(True)

    def test_nojit_raise(self):
        with self.assertRaises(jaxlib.xla_extension.XlaRuntimeError):
            jax_assert(False)

    def test_jit(self):
        @jax.jit
        def f():
            jax_assert(True)

        f()

    def test_jit_raise(self):
        @jax.jit
        def f():
            jax_assert(False)

        with self.assertRaises(jaxlib.xla_extension.XlaRuntimeError):
            f()

    def test_transform(self):
        @jax.vmap
        def f(x):
            jax_assert(x)

        f(jnp.asarray([True, True, True]))

    def test_transform_raise(self):
        @jax.vmap
        def f(x):
            jax_assert(x)

        with self.assertRaises(AssertionError):
            f(jnp.asarray([False, False, False]))

    def test_disable(self):
        jax_assert(False, debug=False)


class TestAssertAllClose(unittest.TestCase):
    def test_same(self):
        assert_allclose(
            jnp.asarray([1.0, 0.8, 1.7]),
            jnp.asarray([1.0, 0.8, 1.7]),
        )

    def test_different(self):
        with self.assertRaises(jaxlib.xla_extension.XlaRuntimeError):
            assert_allclose(
                jnp.asarray([1.0, 0.8, 1.7]),
                jnp.asarray([1.0, 0.8, 0.8]),
            )

    def test_shape_mismatch(self):
        with self.assertRaises(jaxlib.xla_extension.XlaRuntimeError):
            assert_allclose(
                jnp.asarray([1.0, 0.8, 1.7]),
                jnp.asarray([1.0, 0.8]),
            )
            

if __name__ == "__main__":
    unittest.main()
