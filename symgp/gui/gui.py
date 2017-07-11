from PyQt5 import QtCore, QtGui, QtWidgets

from sympy import *

from symgp import *

class SuperMatDataModel(QtCore.QAbstractListModel):
    def __init__(self, data=None, parent=None):
        super(SuperMatDataModel, self).__init__(parent)
        
        self.smData = []
        
        if data:
            self.smData = data
            
    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.smData)
    
    def data(self, index, role=QtCore.Qt.DisplayRole):
        
        if not index.isValid():
            return QtCore.QVariant()
        
        if index.row() >= len(self.smData):
            return QtCore.QVariant()
        
        if role == QtCore.Qt.DisplayRole:
            return self.smData[index.row()].name
        else:
            return QtCore.QVariant()
            
    
    def flags(self, index):
        
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled

        return QtCore.QAbstractItemModel.flags(self, index) | QtCore.Qt.ItemIsEditable
    
    def addObj(self, item):
        self.smData.append(item)
        #self.insertRows(self.rowCount(), 1)
        newIndex = self.createIndex(self.rowCount()-1,0)
        self.dataChanged.emit(newIndex, newIndex)
        
    
    """def setData(self, index, value, role=QtCore.Qt.EditRole):
        
        if (index.isValid() and role == QtCore.Qt.EditRole) {

            
            self.dataChanged(index, index).emit()
            return True;
        }
        return False;"""
    
    """def insertRows(self, position, rows, index=QtCore.QModelIndex()):
        
        self.beginInsertRows(QtCore.QModelIndex(), position, position+rows-1)
        
        for row in range(rows):
            self.distrs.append"""
            
    #def removeRows(self, position, rows, index=QtCore.QModelIndex()):
        
            
class NewDistrDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(NewDistrDialog, self).__init__(parent)
        
        prefixLabel = QtWidgets.QLabel(self.tr("Prefix:"))
        self.prefixLine = QtWidgets.QLineEdit('p')#self.tr("e.g. 'p', 'q'"))
        
        shapeLabel = QtWidgets.QLabel("Shape:")
        self.shapeLine = QtWidgets.QLineEdit('n')#"e.g. 'm', '(n,1)', '(p,q)' ")
        
        depVarsLabel = QtWidgets.QLabel("Dependent variables:")
        self.depVarsLine = QtWidgets.QLineEdit()#"Enter dependent variables separated by commas")
        
        condVarsLabel = QtWidgets.QLabel("Observed variables:")
        self.condVarsLine = QtWidgets.QLineEdit()#"Enter conditional variables separated by commas")
        
        variablesLabel = QtWidgets.QLabel("Variables:")
        self.variablesLine = QtWidgets.QLineEdit('x')#"e.g. 'x', 'y', 'A'")
        
        meanLabel = QtWidgets.QLabel("Mean:")
        self.meanText = QtWidgets.QTextEdit('ZeroMatrix(n,1)')#"Enter Python code for mean")
        
        covLabel = QtWidgets.QLabel("Covariance:")
        self.covText = QtWidgets.QTextEdit('Identity(n)')#"Enter Python code for covariance")
        
        OKButton = QtWidgets.QPushButton("OK")   
        OKButton.clicked.connect(self.accept)
        
        layout = QtWidgets.QFormLayout()   # Add a QValidator to check validity of form entries
        layout.addRow(prefixLabel, self.prefixLine)
        layout.addRow(shapeLabel, self.shapeLine)
        layout.addRow(depVarsLabel, self.depVarsLine)
        layout.addRow(condVarsLabel, self.condVarsLine)
        layout.addRow(variablesLabel, self.variablesLine)
        layout.addRow(meanLabel, self.meanText)
        layout.addRow(covLabel, self.covText)
        layout.addRow(OKButton)
        
        #self.accepted.connect(parent.)
        
        self.setLayout(layout)
        self.setWindowTitle("Add Distribution")
    
    def getData(self):
        prefix = self.prefixLine.text()
        shape = self.shapeLine.text()
        depVars = self.depVarsLine.text()
        condVars = self.condVarsLine.text()
        variables = self.variablesLine.text()
        mean = self.meanText.toPlainText()
        cov = self.covText.toPlainText()
        
        return prefix, shape, depVars, condVars, variables, mean, cov
        
class NewModelDialog(QtWidgets.QDialog):
    def __init__(self, currentVars, parent=None):
        super(NewModelDialog, self).__init__(parent)
        
        self.currentVars = currentVars
        
        topLabel = QtWidgets.QLabel("Add probability distribution: ")
        
        self.distrsModel = SuperMatDataModel()    # Return MVG objects
        self.distributionsListBox = QtWidgets.QListView(self)
        self.distributionsListBox.setModel(self.distrsModel)
        
        
        addDistrButton = QtWidgets.QPushButton("Add..")    # Create a new distribution
        addDistrButton.clicked.connect(self.addDistr)
        
        OKButton = QtWidgets.QPushButton("OK")   
        OKButton.clicked.connect(self.accept)
        
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(topLabel)
        layout.addWidget(self.distributionsListBox)
        layout.addWidget(addDistrButton)
        layout.addWidget(OKButton)
        
        self.setLayout(layout)
        self.setWindowTitle("Create Model")
    
    def addDistr(self, checked=False, debug=False):
        
        newDistrDialog = NewDistrDialog(self)
        
        if newDistrDialog.exec() == QtWidgets.QDialog.Accepted:
            prefix, shape, depVars, condVars, variables, mean, cov = newDistrDialog.getData()
            if debug:
                print("prefix: ",prefix)
                print("shape: ",shape)
                print("depVars: ",depVars)
                print("condVars: ",condVars)
                print("variables: ",variables)
                print("mean: ",mean)
                print("cov: ",cov)
                
        else:
            return
        
        # Convert shape string to tuple
        shape = sympify(shape)
        if debug:
            print("shape: ",shape)
        
        if not isinstance(shape, tuple):
            shape = (shape,1)
        
        if debug:
            print("shape: ",shape)
        
        # Create/verify all the variables in depVars
        if len(depVars) > 0 and not depVars.isspace():
            depVars = depVars.split(',')
        
            for i, v in enumerate(depVars):
                if not SuperMatSymbol.used(v):
                    self.currentVars[depVars[i]] = Constant(depVars[i], shape[0], shape[1])
            
            if debug:
                print("depVars: ",depVars) 
        
        # Create/verify all observed variables/constants (condVars)
        if len(condVars) > 0 and not condVars.isspace():
            condVars = condVars.split(',')
        
            for i, v in enumerate(condVars):
                if not SuperMatSymbol.used(v):
                    self.currentVars[condVars[i]] = Variable(condVars[i], shape[0], shape[1])
            
                condVars[i] = self.currentVars[condVars[i]]
        
        else:
            condVars = [] 
        
        if debug:
            print("condVars: ",condVars)
        
        variables = variables.split(',')
        
        if debug:
            print("variables: ",variables)
        
        if len(variables) > 1:
            pass
        else:
            
            if not self.currentVars.get(variables[0]):
                v = Variable(variables[0], shape[0], shape[1])
                self.currentVars[variables[0]] = v
            else:
                v = self.currentVars[variables[0]]
                
            # We need to create a dictionary of symbols to allow 'sympify' to easily interpret string expression
            ns = {**self.currentVars, **{'SuperDiagMat': SuperDiagMat, 'SuperBlockDiagMat': SuperBlockDiagMat}}
            
            try:
                mean_expr = sympify(mean, locals=ns) 
                # mean_expr = self.parse(mean)
                if debug:
                    print("mean_expr: ", mean_expr)
                
                # Go through expression tree and substitute for symbols with corresponding SuperMatSymbols
            except Exception:
                print("Error in entry for mean")
                raise
            
            try:
                cov_expr = sympify(cov, locals=ns)
                # cov_expr = self.parse(cov)
                if debug:
                    print("cov_expr: ",cov_expr)
                    
            except Exception:
                print("Error in entry for covariance")
                raise
            
            self.distrsModel.addObj(MVG([v],mean=mean_expr,cov=cov_expr,cond_vars=condVars,prefix=prefix))
                
    def getData(self, debug=False):
        distrs = self.distributionsListBox.model().smData
        variables = self.currentVars
        
        return distrs, variables       
    
    def parse(self, expr):
        
        """
            specials = re.compile(r"\,|\||\^|\\|\/|{0}|{1}|{2}".format(operators.pattern, lparen.pattern, rparen.pattern))
            lparen = re.compile(r"\{|\(|\[")
            rparen = re.compile(r"\}|\)|\]")
            
            mat_name = re.compile(r"{0}(?:{0}|{1}|{2}|{3})*".format(upper_char.pattern, lower_char.pattern, digit.pattern, specials.pattern))
            matrix = re.compile(r"(%s|%s)"%(mat_identifier.pattern, mat_name.pattern))
            vec_name = re.compile(r"{1}(?:{0}|{1}|{2}|{3})*".format(upper_char.pattern, lower_char.pattern, digit.pattern, specials.pattern))
            vector = re.compile(r"(%s|%s)"%(vec_identifier.pattern, vec_name.pattern))
            
            
            
        
            expr = re.compile(r"^(?:(%s)\[)?\(?(%s)\)?((?:(%s)\(?(%s)\)?){0,5})\]?" % (diag_op.pattern, symbols.pattern, operators.pattern, symbols.pattern))
        """
        
        #tokens = self.get_tokens(expr)
    
    def get_tokens(self, expr):
        
        """# Regex expressions
        digit = re.compile(r"[0-9_]")
        lower_char = re.compile(r"[a-z]")
        upper_char = re.compile(r"[A-Z]")
        operators = re.compile(r"\+|\-|\*")
        diag_op = re.compile(r"diag|blkdiag|blockdiag")
        
        mat_identifier = re.compile(r"{1}(?:{0}|{1}|{2})*".format(\
                                lower_char.pattern, upper_char.pattern, digit.pattern))
        vec_identifier = re.compile(r"{0}(?:{0}|{1}|{2})*".format(\
                                lower_char.pattern, upper_char.pattern, digit.pattern))
        
        kernel = re.compile(r"(?:{0}|{1})\((?:{2}|{3}),(?:{2}|{3})\)".format(\
                                lower_char.pattern, upper_char.pattern, vec_identifier.pattern, mat_identifier.pattern))
                                
        inv_op = re.compile(r"(?:%s|%s)(?:\.I|\^\-1|\^\{\-1\})"%(\
                                mat_identifier.pattern, kernel.pattern))
        inv_op_grouped = re.compile(r"(%s|%s)(?:\.I|\^\-1|\^\{\-1\})"%(\
                                mat_identifier.pattern, kernel.pattern))
        trans_op = re.compile(r"(?:%s|%s|%s)(?:\.T|\'|\^t|\^T|\^\{t\}|\^\{T\})"%(\
                                mat_identifier.pattern, vec_identifier.pattern, kernel.pattern))
        trans_op_grouped = re.compile(r"(%s|%s|%s)(?:\.T|\'|\^t|\^T|\^\{t\}|\^\{T\})"%(\
                                mat_identifier.pattern, vec_identifier.pattern, kernel.pattern))
        
        symbols = re.compile(r"{0}|{1}|{2}|{3}|{4}".format(\
                                mat_identifier.pattern, vec_identifier.pattern, kernel.pattern,\
                                trans_op.pattern, inv_op.pattern))
        
        expr_re = re.compile(r"^(?:({0})\[)?(\()?({1})(\))?((?:(?:{2})\(?(?:{1})\)?)*)\]?".format(\
                                diag_op.pattern, symbols.pattern, operators.pattern))
        
        def match2Symbol(s):
            #Determines whether expr matches to mat_identifier, vec_identifier, kernel
            
            if mat_identifier.fullmatch(s):
                return mat_identifier_token(s)
            else if vec_identifier.fullmatch(s):
                return vec_identifier_token(s)
            else if kernel.fullmatch(s):
                
                # Break up 's' into the kernel name and the two arguments
                match = s.split("(")
                name = match[0]
                
                arg1, arg2 = match[1].strip(")").split(",")  
           
                return kernel_token(name, arg1, arg2)
            else:
                return None
            
        
        tokens = []
        expr_match = expr_re.fullmatch(expr)
        if expr_match:
            groups = expr_match.groups()
            if groups[0]:
                tokens.append(diag_token(groups[0]))
            
            if groups[1]:
                tokens.append(paren_token(groups[1]))
            
            if groups[2]:
                
                token = match2Symbol(groups[2])
                
                # token must be inv_op or trans_op
                if not token:
                    if trans_op.fullmatch(groups[2]):
                        token = match2Symbol(trans_op_grouped.fullmatch(groups[2]).groups()[0])
                        token = trans_token(token)
                    else: # inv_op.fullmatch(groups[2]):
                        token = match2Symbol(inv_op_grouped.fullmatch(groups[2]).groups()[0])
                        token = inv_token(token)
                
                tokens.append(token)
            
            if groups[3]:
                tokens.append(paren_token(groups[3]))
            
            right = groups[4]
            right_regex = re.compile(r"^({0})(\()?({1})(\))?((?:(?:{0})\(?(?:{1})\)?)*)\]?".format(\
                                        operators.pattern, symbols.pattern))
            while len(right) > 0:
                groups = right_regex.fullmatch(right).groups()
                
                if groups[0]:
                    tokens.append(operator_token(groups[0]))
                
                if groups[1]:
                    tokens.append(paren_token(groups[1]))
                
                if groups[2]:
                    token = match2Symbol(groups[2])
                
                    # token must be inv_op or trans_op
                    if not token:
                        if trans_op.fullmatch(groups[2]):
                            token = match2Symbol(trans_op_grouped.fullmatch(groups[2]).groups()[0])
                            token = trans_token(token)
                        else: # inv_op.fullmatch(groups[2]):
                            token = match2Symbol(inv_op_grouped.fullmatch(groups[2]).groups()[0])
                            token = inv_token(token)
                
                    tokens.append(token)
                
                if groups[3]:
                    tokens.append(paren_token(groups[3]))
                
                right = groups[4]
            
                            
            
        else:
            raise Exception("Invalid input")"""
        
        
class MVGDisplayScene(QtWidgets.QGraphicsScene):
    
    def __init__(self, parent=None):
        super(MVGDisplayScene, self).__init__(parent)
        
               
        
        
                       
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        self.outputFrame = QtWidgets.QFrame(self)
        self.outputFrame.setMinimumSize(500, 500)
        self.outputFrame.setFrameStyle(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Sunken)
        
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        scrollArea.setWidget(self.outputFrame)
        
        self.setCentralWidget(self.outputFrame)
        
        self.variablesModel = SuperMatDataModel()
        self.distributionsModel = SuperMatDataModel()
        
        self.currentVars = {}
        self.currentDistrs = []
        self.currentDisplays = []
        
        self.createActions()
        #self.createMenus()
        self.createToolBars()
        self.createStatusBar()
        self.createDockWindows()
        
        self.setWindowTitle("SymGP GUI")
    
    
    def newModel(self):
        
        oldVars = self.currentVars.copy()
        print("oldVars: ", oldVars)
        
        newModelDialog = NewModelDialog(self.currentVars, parent=self)
        
        if newModelDialog.exec() == QtWidgets.QDialog.Accepted:
            distrs, variables = newModelDialog.getData()
            print("distrs: ",[type(d) for d in distrs]," variables: ",variables)
            #print("newModelDialog.distributionsListBox.model().smData: ",\
            #       newModelDialog.distributionsListBox.model().smData)
            #print("self.distributionsModel.smData: ", self.distributionsModel.smData)
            #print("newModelDialog.distributionsListBox.model() == self.distributionsModel?", \
            #       newModelDialog.distributionsListBox.model() == self.distributionsModel)
            #print("newModelDialog.distributionsListBox.model() == self.variablesModel?", \
            #       newModelDialog.distributionsListBox.model() == self.variablesModel)
            #print("self.variablesModel == self.distributionsModel?", \
            #       self.variablesModel == self.distributionsModel)
            # Update currentVars
            
            self.currentVars = variables 
            
            for v in variables:
                print("v: ",v)
                print("oldVars.get(v): ",oldVars.get(v))
                print("oldVars: ",oldVars)
                if not variables[v] in self.variablesModel.smData:
                    print("variables[v]: ",variables[v])
                    self.variablesModel.addObj(variables[v])
            
            print("self.variablesModel.smData:",self.variablesModel.smData)
            for d in distrs:
                #print("distrs: ",distrs)
                #print("newModelDialog.distributionsListBox.model().smData: ",\
                #       newModelDialog.distributionsListBox.model().smData)
                #print("d: ",d)
                #print("d: %s, d.name: %s" % (d,d.name))
                self.distributionsModel.addObj(d)
        else:
            print("Rejected")
    
    def createActions(self):
        self.newModelAct = QtWidgets.QAction("&New Model", self, 
                statusTip="Create a new Gaussian model", triggered=self.newModel)
                
        #self.genLatexAct = QAction("&Generate Latex", self,
        #        statusTip="Generates the Latex code for distributions in window", triggered=self.genLatex)
    
    def createToolBars(self):
        self.fileToolBar = self.addToolBar("File")
        self.fileToolBar.addAction(self.newModelAct)
        #self.fileToolBar.addAction(self.saveAct)
        #self.fileToolBar.addAction(self.printAct)

        #self.editToolBar = self.addToolBar("Edit")
        #self.editToolBar.addAction(self.undoAct)
        
    def createStatusBar(self):
        self.statusBar().showMessage("Ready")
                    
    def createDockWindows(self):
        dock = QtWidgets.QDockWidget("Variables/Constants", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        
        self.variablesList = QtWidgets.QListView(dock)
        self.variablesList.setModel(self.variablesModel)
        dock.setWidget(self.variablesList)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        
        dock = QtWidgets.QDockWidget("Distributions", self)        
        self.distrsList = QtWidgets.QListView(dock)
        self.distrsList.setModel(self.distributionsModel)
        self.distrsList.activated.connect(self.addLatexWindow)
        dock.setWidget(self.distrsList)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    
    def addLatexWindow(self, index, debug=False):
        
        name = self.distrsList.model().data(index)
        
        newdock = QtWidgets.QDockWidget(name, self)
        
        newScene = MVGDisplayScene()
        newView = QtWidgets.QGraphicsView
        
        
        
        
if __name__ == '__main__':

    import sys

    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
    
#class MVGWindow():

#class VarWindow():

#class ButtonsBar(QBoxLayout):