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

# def test_constr():
#     k = 2
#     cf = kNN(k)

#     assert cf.k == k

@pytest.mark.parametrize(
        "k, backhand, e_k, e_backhand",
        [
            (5, "plain", 5, "plain"),
            (5, "numpy", 5, "numpy")
        ]
)

def test_knn_constructor(k, backhand,  e_k, e_backhand):
    cf = kNN(k, backhand)
    assert cf.k == e_k
    assert cf.backhand == e_backhand

@pytest.mark.parametrize(
        "k, backhand",
        [
            ("5", "plain"),
            (5.25, "numpy")
        ]
)

def test_knn_contructor_te(k, backhand):
    with pytest.raises(TypeError):
        cf = kNN(k, backhand)

@pytest.mark.parametrize(
        "k, backhand",
        [
            (5, "pippo")
        ]
)

def test_knn_contructor_te(k, backhand):
    with pytest.raises(ValueError):
        cf = kNN(k, backhand)