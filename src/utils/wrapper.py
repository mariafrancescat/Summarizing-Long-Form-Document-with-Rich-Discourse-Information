from src.models.heroes import *

class Wrapper:

    @staticmethod
    def wrapperFromTo(fromClass, toClass):
        mapping = {
            (ContentRanking,'TODO: BART'): Wrapper.fromContentRankingToBart
        }
        return mapping[(fromClass,toClass)]

    @staticmethod
    def fromContentRankingToBart(docs):
        pass
        #TODO: from document digest to json format of bart