import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fruit_catcher_students.dt import train_decision_tree

X = [
    ['apple', 'red', 'circle'],
    ['banana', 'yellow', 'curved'],
    ['pear', 'green', 'oval'],
    ['apple', 'green', 'circle']
]
y = [1, 1, -1, -1]

tree = train_decision_tree(X, y)

def test_predict():
    assert tree.predict(['apple', 'red', 'circle']) == 1
    assert tree.predict(['banana', 'yellow', 'curved']) == 1
    assert tree.predict(['pear', 'green', 'oval']) == -1
    assert tree.predict(['apple', 'green', 'circle']) == -1
    assert tree.predict(['banana', 'green', 'oval']) in [1, -1]  # Valor desconhecido

    print("✅ Todos os testes passaram!")

test_predict()
