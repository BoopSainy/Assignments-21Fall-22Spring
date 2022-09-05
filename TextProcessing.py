import math

class Retrieve:

    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents() # size of collection: total num of docs
                                                           # in collection
        # construct a dictionary containing idf of all terms
        self.idf = self.compute_idf() # {term_1: idf_1, term_2: idf_2 ...}
        
        # construct a dictionary containing document vector size for all docs
        self.doc_vec_size = self.compute_document_vector_size() # {doc_1: vec_size_1, doc_2: vec_size_2 ...}
    

    # design a method to calculate the number of docs in collection
    def compute_number_of_documents(self):
        self.doc_ids = set() # a set containing all id of docs
        for term in self.index:
            self.doc_ids.update(self.index[term])
            
        return len(self.doc_ids)
    

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query): # query is a list

        # convert query from list to a dictionary (map terms' name to occurrences' count)
        query_dict = {} # {term_1: count_1, term_2: count_2, ..., term_n: count_n}
        for term in query:
            if (term not in query_dict):
                query_dict[term] = 1
            else:
                query_dict[term] += 1

        # if term_weighting is "binary"
        # call "self.binary" method to get the ranked doc list
        if (self.term_weighting == "binary"):
            ranked_list = self.binary(query_dict)

        # if term_weighting is "tf"
        # call "self.tf" method to get the ranked doc list
        if (self.term_weighting == "tf"):
            ranked_list = self.tf(query_dict)

        # if term_weighting is "tfidf"
        # call "self.tfidf" method to get the ranked doc list
        if (self.term_weighting == "tfidf"):
            ranked_list = self.tfidf(query_dict)


        return ranked_list

    def binary(self, query_dict): # rank the relevance of all docs under "binary" scheme
        similarity = {} # containing the docids and corresponding similarity scores
        
        term_list = self.index.keys()
        
        for doc in self.doc_ids:
            similarity[doc] = 0
            
            for check in self.doc_content[doc]:# determine if the doc contain at least one term of query
                                               # if not, then current doc could be ignored
                if (check in query_dict):
                    for term in query_dict: # compute the similarity score's numerator for doc
                        if (term in term_list): # determine if the term in the collection
                            if (doc in self.index[term]):
                                similarity[doc] += 1
                    break

            similarity[doc] = similarity[doc]/self.doc_vec_size[doc] # compute the similarity score
                                                                     # for doc
        # rank the relevance by comparing similarity score
        ranked_list = sorted(similarity, key = lambda x:similarity[x], reverse = True)

        return ranked_list[:10]

    def tf(self, query_dict): # rank the relevance of all docs under "tf" scheme
        similarity = {} # containing the docids and corresponding similarity scores
        
        term_list = self.index.keys()
        
        for doc in self.doc_ids:
            similarity[doc] = 0
            
            for check in self.doc_content[doc]: # determine if the doc contain at least one term of query
                                                # if not, then current doc could be ignored
                if (check in query_dict):
                    for term in query_dict: # computer the similarity score's numerator for doc
                        if (term in term_list): # determine if the term in the collection
                            if (doc in self.index[term]):
                                similarity[doc] += query_dict[term] * self.index[term][doc]
                    break
                    

            similarity[doc] = similarity[doc]/self.doc_vec_size[doc]# compute the similarity score
                                                                    # for doc

        # rank the relevance by comparing similarity scores
        ranked_list = sorted(similarity, key = lambda x:similarity[x], reverse = True)


        return ranked_list[:10]


    def tfidf(self, query_dict): # rank the relevance of all docs under "tfidf" scheme
        similarity = {} # containing the docids and corresponding tfidf similarity scores
        
        term_list = self.index.keys()
        
        for doc in self.doc_ids:
            similarity[doc] = 0
            
            for check in self.doc_content[doc]:# determine if the doc contain at least one term of query
                                               # if not, then current doc could be ignored
                if (check in query_dict):
                    for term in query_dict: # compute the similarity score's numerator for doc
                        if (term in term_list): # determine if the term in the collection
                            if (doc in self.index[term]):
                                similarity[doc] += query_dict[term] * self.idf[term] **2 *\
                                    self.index[term][doc]
                    
                    break
                    
                             
            similarity[doc] = similarity[doc]/self.doc_vec_size[doc] # compute the similarity score
                                                                     # for doc
                                                                     
        # rank the relevance by comparing similarity score under "tfidf" scheme
        ranked_list = sorted(similarity, key = lambda x:similarity[x], reverse = True)
        

        return ranked_list[:10]


    def compute_idf(self):
        # initialize a dictionary to store the inverse document frequency for each term
        idf = {}
        for term in self.index:
            df = len(self.index[term]) # number of documents containing "term"

            idf[term] = math.log10(self.num_docs/df) # idf of "term"
        

        return idf


    def compute_document_vector_size(self):
        # initialize a dictionary to store the terms of each document
        self.doc_content = {}
        # initialize a dictionary to store the documents vector size
        doc_vec_size = {}
        for doc in self.doc_ids:
            self.doc_content[doc] = []
            doc_vec_size[doc] = 0
            
        
        # compute the documents vector size when the scheme is binary
        if (self.term_weighting == "binary"):
            for term, doc_appeared in self.index.items():
                for doc in doc_appeared:
                    self.doc_content[doc].append(term)
                    doc_vec_size[doc] += 1
            
            for doc in doc_vec_size:
                doc_vec_size[doc] = math.sqrt(doc_vec_size[doc])
        
            return doc_vec_size
        # compute the documents vector size when the scheme is "TF"
        elif (self.term_weighting == "tf"):
            for term, doc_appeared in self.index.items():
                for doc, termfre in doc_appeared.items():
                    self.doc_content[doc].append(term)
                    doc_vec_size[doc] += termfre ** 2
            
            for doc in doc_vec_size:
                doc_vec_size[doc] = math.sqrt(doc_vec_size[doc])

            return doc_vec_size
        # compute the documents vector size when the scheme is "TFIDF"
        elif (self.term_weighting == "tfidf"):
            for term, doc_appeared in zip(self.index.keys(), self.index.values()):
                for doc, termfre in doc_appeared.items():
                    self.doc_content[doc].append(term)
                    doc_vec_size[doc] += (termfre * self.idf[term]) ** 2
            
            for doc in doc_vec_size:
                doc_vec_size[doc] = math.sqrt(doc_vec_size[doc])
        
            return doc_vec_size


