class Node:
    attribute = -1
    attributeClass = ""
    subChildren = []  # Node list
    label = ""
    depth = 0

    def __init__(self, A, label,className,depth=100):
        self.attribute = A
        self.attributeClass = className
        self.label = label
        self.subChildren = []
        self.depth = depth

    def addNode(self,newNode):
        self.subChildren.append(newNode)