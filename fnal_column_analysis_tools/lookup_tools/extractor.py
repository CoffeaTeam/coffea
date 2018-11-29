import uproot
from evaluator import evaluator

TH1D = "<class 'uproot.rootio.TH1D'>"
TH2D = "<class 'uproot.rootio.TH2D'>"
TH1F = "<class 'uproot.rootio.TH1F'>"
TH2F = "<class 'uproot.rootio.TH2F'>"
RootDir = "<class 'uproot.rootio.ROOTDirectory'>"

def convert_root_file(file):
    converted_file = {}
    dumFile = uproot.open(file.strip())
    for i, key in enumerate(dumFile.keys()):
        histType = str(type(dumFile[key[:-2]]))
        if histType == TH1D or histType == TH2D or histType==TH1F or histType==TH2F:
            if  not ("bound method" in str(dumFile[key[:-2]].numpy)):
                converted_file[key[:-2]] = dumFile[key[:-2]].numpy
            else:
                converted_file[key[:-2]] = dumFile[key[:-2]].numpy()
        elif histType == RootDir: #means there are subdirectories wihin main directory
            for j, key2 in enumerate(dumFile[key[:-2]].keys()):
                histType2 = str(type(dumFile[key[:-2]][key2[:-2]]))
                if histType2 == TH1D or histType2 == TH2D or histType2==TH1F or histTyp2==TH2F:
                    if not ("bound method" in str(dumFile[key[:-2]][key2[:-2]].numpy)):
                        converted_file[key[:-2]+'/'+key2[:-2]] = dumFile[key[:-2]][key2[:-2]].numpy
                    else:
                        converted_file[key[:-2]+'/'+key2[:-2]] = dumFile[key[:-2]][key2[:-2]].numpy()           
        else:
            tempArrX= dumFile[key[:-2]]._fEX
            tempArrY= dumFile[key[:-2]]._fEY
            converted_file[key[:-2]] = [tempArrX, tempArrY]
    return converted_file

def csvToVec(csvFilePath):
    p = csvFilePath
    f = open(p).readlines()
    
    retVec = []
    for line in f:
        line = re.sub(',', '', line)
        vec = line.split()
        retVec.append(vec)
    retVec.pop(0)
    for i, e in enumerate(retVec):
        retVec[i][0]= int(e[0])
        retVec[i][3]=int(e[3])
        retVec[i][4]=float(e[4])
        retVec[i][5]=float(e[5])
        retVec[i][6]=float(e[6])
        retVec[i][7]=float(e[7])
        retVec[i][8]=float(e[8])
        retVec[i][9]=float(e[9])
        retVec[i][10]=eval(e[10])
        
    return retVec

def makeSfSpecificVec(csvVec,OP, mType,sysType, flavor):
    retVec = []
    for i,e in enumerate(csvVec):
        if e[0] == OP and e[1] ==mType and e[2]==sysType and e[3] == flavor:
            e.pop(0)
            e.pop(0)
            e.pop(0)
            e.pop(0)
            retVec.append(e)
    
    return retVec

file_converters = {'root':convert_root_file,
                   'csv':csvToVec}

class extractor(object):
    def __init__(self):
        self.__weights = []
        self.__names = {}
        self.__filecache = {}
        self.__finalized = False
    
    def add_weight_set(self,local_name,weights):
        if self.__finalized: 
            raise Exception('extractor is finalized cannot add new weights!')
        if local_name in self.__names.keys():
            raise Exception('weights name "{}" already defined'.format(local_name))
        self.__names[local_name] = len(self.__weights)
        self.__weights.append(weights)
    
    def add_weight_sets(self,weightsdescs):
        # expect file to be formatted <local name> <name> <weights file>
        # allow * * <file> and <prefix> * <file> to do easy imports of whole file
        for weightdesc in weightsdescs:
            temp = weightdesc.split(' ')
            if len(temp) != 3: 
                raise Exception('"{}" not formatted as "<local name> <name> <weights file>"'.format(weightdesc))
            (local_name,name,file) = tuple(temp)
            if name == '*':
                self.import_file(file)
                weights = self.__filecache[file]
                for key, value in weights.iteritems():
                    if local_name == '*':
                        self.add_weight_set(key,value)
                    else:
                        self.add_weight_set(local_name+key,value)
            else:
                weights = self.extract_from_file(file,name)
                self.add_weight_set(local_name,weights)
    
    def import_file(self,file):
        if file not in self.__filecache.keys():
            self.__filecache[file] = file_converters[file.split('.')[-1].strip()](file)
    
    def extract_from_file(self, file, name):        
        self.import_file(file)       
        weights = self.__filecache[file]        
        if name not in weights.keys(): 
            raise Exception('Weights named "{}" not in {}!'.format(name,file))        
        return weights[name]
          
    def finalize(self):
        if self.__finalized: 
            raise Exception('extractor is already finalized!')
        del self.__filecache
        self.__finalized = True
    
    def make_evaluator(self):
        return evaluator(self.__names,self.__weights)