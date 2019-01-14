import operator

class Apriori:
    def __init__(self, transactions):

        self.transactions = transactions
        self.totalTransactions = len(self.transactions)

        # It defines the minimum number of items in a transaction so that the transaction will be processed
        self.minimumItemsInTransaction = 2        
        self.minimumSupport = 0.003
        self.minimumConfidence = 0.2


        # reverse map contains: items as keys and values  = list of indexes of transactions where that key is present
        self.reverseMap = self.buildReverseMap()

        self.supportMap = self.buildSupportMap()

        self.confidenceMap = self.buildConfidenceMap()

        self.liftMap = self.buildLiftMap()

        self.sortedLifts = sorted(self.liftMap.items(), key=operator.itemgetter(1), reverse=True)


        # print (self.reverseMap)
        # print (self.supportMap)
        # print (self.confidenceMap)
        # print (self.liftMap)

    def writeToCsv(self,filename):

        with open(filename + '.csv','w') as file:
            # file.write( 'Item1, Item2, Support1, Support2, Confidence, Lift \n' )
            for pair in self.sortedLifts:
                item1 = pair[0][0]
                item2 = pair[0][1]
                file.write(item1 + ',' + item2 + ',' + str(self.supportMap[item1])+',' + str(self.supportMap[item2]) +',' + str(self.confidenceMap[item1][item2]) +',' +str(pair[1]))
                file.write('\n')


    def getLiftMap(self):
        return self.sortedLifts

    def buildConfidenceMap(self):
        dic = {}

        for item in self.supportMap:
            for item2 in self.supportMap:
                if item == item2:
                    continue
                list1 = self.reverseMap[item]
                list2 = self.reverseMap[item2]
                lenOfCommonTransactions = len((set(list1) & set(list2)))                
                confidence = float(lenOfCommonTransactions) / len(self.reverseMap[item])

                if confidence > self.minimumConfidence :
                    if item in dic:
                        dic[item][item2] = confidence
                    else :
                        dic[item] = {item2 :confidence}

        return dic

    def buildLiftMap(self):
        dic = {}
        for item in self.confidenceMap:
            for item2 in self.confidenceMap[item]:
                lift = self.confidenceMap[item][item2]/ self.supportMap[item2]
                dic[(item,item2)]= lift

        return dic             

    def buildSupportMap(self):
        dic = {}
        for item in self.reverseMap:
            supportOfItem = len(self.reverseMap[item]) / float(self.totalTransactions)
            if supportOfItem >= self.minimumSupport:
                dic[item] = supportOfItem
        return dic

    def buildReverseMap(self):
        dic = {}
        for i in range(self.totalTransactions):
            transaction = self.transactions[i]
            if(len(transaction) < self.minimumItemsInTransaction):
                continue            
            for item in transaction:
                if item in dic:
                    dic[item].append(i)
                else :
                    dic[item] = [i]
        return dic


# t= [['a','b','c'], ['a','b'], ['a','b','c','d']]


# apriori = Apriori(t)
# print (apriori.getLiftMap())
# apriori.writeToCsv('temp')
