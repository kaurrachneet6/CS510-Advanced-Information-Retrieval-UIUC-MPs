import metapy    
class TwoStageSmoothing(metapy.index.LanguageModelRanker):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """
    def __init__(self, jm_param=0.7, dp_param=800):
        # You *must* call the base class constructor here!
        super(TwoStageSmoothing, self).__init__()
        self.jm_param = jm_param
        self.dp_param = dp_param

    def smoothed_prob(self, sd):
        """
        Returns the smoothed probability of a term that is seen in a
        document. See the definition of p_s(w) in section 2 and 3 of
        http://www.stat.uchicago.edu/~lafferty/pdf/smooth-tois.pdf

        See help(metapy.index.ScoreData) for a list of variables that are
        available in the parameter sd.

        See also
        https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        pw_C =float(sd.corpus_term_count)/sd.total_terms #Prob(word|Corpus)
        p_seen = (1.-self.jm_param)*((sd.doc_term_count+self.dp_param*pw_C)/(sd.doc_size+self.dp_param))+self.jm_param*pw_C
        return p_seen # REPLACE ME

    def doc_constant(self, sd):
        """
        Returns the document-specific constant alpha_d. See the definition
        of p(w | d) in section 3 of
        kttp://www.stat.uchicago.edu/~lafferty/pdf/smooth-tois.pdf

        The easiest way to derive this is to consider what the coefficient
        would be in front of p(w | C) in the formula for p(w | d) when w
        does not occur in d.

        You can see Table I in the above TOIS paper for the values for
        alpha_d for traditional smoothing methods like Jelinek-Mercer and
        Dirichlet prior to check your understanding of the meaning of this
        constant.
        """
        alpha_d=((1.-self.jm_param)*self.dp_param/(sd.doc_size+self.dp_param))+self.jm_param
        return alpha_d # REPLACE ME

