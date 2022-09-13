class Node:
    attribute = -1
    attributeClass = ""
    subChildren = []  # Node list
    label = ""

    def __init__(self, A, label,className):
        self.attribute = A
        self.attributeClass = className
        self.label = label
        self.subChildren = []

    def addNode(self,newNode):
        self.subChildren.append(newNode)