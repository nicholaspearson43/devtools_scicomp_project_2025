from pyclassify.utils import distance, majority_vote
from pyclassify.classifier import kNN
import pytest

def test_distance():
    point1 = [1, 2, 3]
    point2 = [4, 5, 6]
    point3 = [7, 8, 9]
    # Sym
    assert distance(point1=point1, point2=point2) == distance(point1=point2, point2=point1)
    assert distance(point1=point1, point2=point2) + distance(point2, point3) >= distance(point1, point3)
    assert distance(point1, point1) == 0
    assert distance(point1, point2)>=0



def test_majority_vote():
    y_test = [1, 0, 0, 0]
    assert majority_vote(y_test) == 0

def test_constr():
    k = 2
    cf = kNN(k)

    assert cf.k == k

