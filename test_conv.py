from convention import conv
import pytest

def test_conv():
    nb_erreur = conv()
    assert nb_erreur == 0
    