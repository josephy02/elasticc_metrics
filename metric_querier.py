import sys
import requests
import json
import numpy
import pandas
import logging

class ELAsTiCCMetricsQuerier:
    """Class to send some queries for ELAsTiCC Metrics.

    Instantiate one of these.  Once you've done so, you can:

    * call run_query to send a general query to the elasticc database
      (see
      https://github.com/LSSTDESC/tom_desc/blob/main/sql_query_tom_db.py)
      for a hopefully-up-to-date set of schema

    * Look at the classifier_info property to get a dictionary of dictionaries:
       { classifierId: { 'brokerName': <string>, 'brokerVersion': <string>,
                         'classifierName': <string>, 'classifierParams': <string> } }
      where classifierId is an integer

   * Look at the classname property to get a dictionary:
       { classId: <string> }
     where classId is an integer

   * Call the get_probhist method to get the contents of the
     elasticc_view_classifications_probmetrics materialized view as a
     ~16MB pandas dataframe.  (70MB is the size of the database table in
     postgres; I'm not sure why the discrepancy.)  Note that this method
     gives you a copy of an internally cached table, so any changes you
     make will not be reflected by the return value from subsequent
     calls to this method.  (So, if you want to make changes, store the
     return value in your own variable.)

     This view is based on the elasticc_view_dedupedclassifications
     materialized view, which tried to de-duplicate entries in the
     broker classifications table by looking for repeats of
     classifierId/diaSourceId, and taking the average of the
     probabilities assigned by the duplicates.  The columns in the
     elasticc_view_classifications_probmetrics are:
        classifierId
        classId
        trueClassId
        tbin
        probbin
        count
     count has the number of classifications of sources that were in time
     bin tbin relative to peak (from the truth table), which were from
     objects of type trueClassId, that were given a probability within
     probbin of being classId by the classifier with id classifierId.

     tbin is defined by the postgres function
          width_bucket(dt, -30, 100, 26)
     where dt is midPointTai (from the source alert) minus peakmjd (from
     the truth table).  This ignores the difference between MJD and TAI
     (which should be a few seconds at most, if I understand the
     midPointTAI frame, and indeed it's possible for elasticc that we
     never converted).  In pratice, this means that tbin can be interpreted:
          tbin=0 : dt < -30 days
          tbin=1 : dt in [-30, -25) days
          tbin=2 : dt in [-25, -20) days
          ...
          tbin=26 : dt in [95, 100) days
          tbin=27 : dt >= 100 days

     pbin is defined by the postgres function
           width_bucket(probability, 0, 1, 20) 
     where probability is from the brokerclassifications table.  In
     practice, this means that pbin can be interpreted:
           pbin=0 :  probability < 0
           pbin=1 :  probability in [0.00, 0.05)
           pbin=2 :  probability in [0.05, 0.10)
           ...
           pbin=20 : probability in [0.95, 1.00)
           pbin=21 : prbability >= 1.00
     note that if the broker sent probability 1.0, it will be included in
     pbin 21, not in pbin 20, because the postgram width_bucket histogram
     function has an open upper end on its bins.

     The first time you call this method, it will take several seconds
     as it pulls the full table from the database.  Thereafter, you will
     get a copy of the pandas dataframe cached in memory by the object
     you instantiated.  (This does mean that if the database materialized
     view is regenerated, you won't get the updates, but that should
     happen rarely or never.)

   * Call the methods tbin_val(tbin) and probbin_val(pbin) to get the
     values at the middle of the bins in the table returned by
     get_probhist.  You can pass either a scalar or a numpy array for
     tbin or pbin, and will get the same thing back.

     tbin will return -32.t for tbin=0, even though the bin is [-∞,
     -30), and will return 102.5 even though the bin is [100, ∞).

     probbin_val will return -1 for pbin=0 (the bin is [-∞, 0), but
     well-behaved classifiers will never have returned a negative
     probability, so this should never happend) and 1 for pbin=20 (the
     bin is [1.0, ∞), but well-behaved classifiers will only return 1.0
     in this bin, so this estimate should be correct.)

   * Call the method right_probdiffs_for_object( diaObjectID ) to get a
     dataframe that has
       classifierId
       trueClassId   [ will be identical in every row ]
       earlytimebin
       earlytimet0
       earlytimet1
       latetimebin
       latetimet0
       latetimet1
       probdiff      [ max probability in late time bin minus max probability in early time bin ]
      
     Note that not all bins will be present -- only bins with data will
     be present.  probdiff difference is the probability assigned to the
     true class by the classifier between the late and early times.

     The lowest time bin is [-99999, -20) and the highest time bin is
     [100, 99999).  That nines are for internal database usage;
     interpret these as "less than -20 days" and "100 days or greater".

  * Call the method right_profdiffs_hist() to get a dataframe that has:
      Indexes:
        classifierId
        trueClassId
        earlytimebin
        latetimebin
        probdiffbin
      Columns:
        earlytimet0
        earlytimet1
        latetimet0
        latetimet1
        binmeanprobdiff
        count
        frac

    For a given classifier and trueClassId, this looks at all the
    probabilities for objects given to that classId by that classifier.

    It looks at the differences in the maximum probabilities in the two
    time ranges.

    It histograms those differences, so that count is the number of
    objects that end up in the early time bin, late time bin, and
    probability difference bin.

    Things named "*bin" are a bin number.  (early|late)time(0|1) give
    the [t0, t1) range of that bin; see previous section for caveat
    about lowest and highest time bin.  binmeanprobdiff gives the middle
    of the probability bin (subject to floating point roundoff, so round
    it to two decimal places when printing).  Note that the lowest and
    highest probability bins are [-1.025, 0.975 ) and [ 0.975, 1.025 ).
    The outside half of these bins are not actually part of the
    distribution, so you'll need to rescale the counts to get a
    propability density.  (If just plotting counts, probably plot the
    bar half as wide at the low and high bins.)

    frac is count divided by the sum of count over probability bins.

    That's all complicated and stuff.

    """

    def __init__( self, tomusername=None, tompasswd=None, logger=None, url="https://desc-tom.lbl.gov" ):
        if ( tomusername is None ) or ( tompasswd is None ):
            raise RuntimError( "Must pass tomusername and tompasswd" )

        if logger is None:
            self.logger = logging.getLogger( "ELAsTiCCMetricsQuerier" )
            if not self.logger.hasHandlers():
                logout = logging.StreamHandler( sys.stderr )
                self.logger.addHandler( logout )
                formatter = logging.formatter( f'[%(asctime)s - %(levelname)s] - %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S' )
                logout.setFormatter( formatter )
            self.logger.setLevel( logging.DEBUG )
        else:
            self.logger = logger

        self.url = url
        self.rqs = requests.session()
        self.rqs.get( f'{self.url}/accounts/login/' )
        res = self.rqs.post( f'{self.url}/accounts/login/',
                             data={ "username": tomusername,
                                    "password": tompasswd,
                                    "csrfmiddlewaretoken": self.rqs.cookies['csrftoken'] } )
        if res.status_code != 200:
            raise RuntimeError( f"Failed to log in; http status: {res.status_code}" )
        if 'Please enter a correct' in res.text:
            raise RuntimeError( "Failed to log in.  I think.  Put in a debug break and look at res.text" )
        self.rqs.headers.update( { 'X-CSRFToken': self.rqs.cookies['csrftoken'] } )

        self._classname = None
        self._classifier_info = None
        self._probhist = None

        self._tbin_min = -30.
        self._tbin_max = 100.
        self._tbin_num = 26
        self._delta_tbin = ( self._tbin_max - self._tbin_min ) / self._tbin_num
        self._probbin_min = 0.
        self._probbin_max = 1.
        self._probbin_num = 20
        self._delta_probbin = ( self._probbin_max - self._probbin_min ) / self._probbin_num
        
    def run_query( self, query, subdict=None ):
        if subdict == None:
            subdict = {}
        result = self.rqs.post( f'{self.url}/db/runsqlquery/',
                                json={ 'query': query, 'subdict': subdict } )
        if result.status_code != 200:
            sys.stderr.write( f"ERROR: got status code {result.status_code} ({result.reason})\n" )
        else:
            data = json.loads( result.text )
            if ( 'status' not in data ) or ( data['status'] != 'ok' ):
                sys.stderr.write( "Got unexpected response\n" )
                print(data['error'])
            else:
                return data['rows']

    @property
    def classname( self ):
        if self._classname is not None:
            return self._classname

        rows = self.run_query( 'SELECT DISTINCT ON ("classId") "classId",description '
                               'FROM elasticc_gentypeofclassid '
                               'ORDER BY "classId"' )
        self._classname = { row["classId"]: row["description"] for row in rows }

        return self._classname

    @property
    def classifier_info( self ):
        if self._classifier_info is not None:
            return self._classifier_info
        
        rows = self.run_query( 'SELECT "classifierId","brokerName","brokerVersion",'
                               '"classifierName","classifierParams" '
                               'FROM elasticc_brokerclassifier' )
        self._classifier_info = { row["classifierId"]: row for row in rows }

        return self._classifier_info

    @property
    def tbin_min( self ):
        """Middle value of Δt for tbin=1"""
        return self._tbin_min

    @property
    def tbin_max( self ):
        """Middle of Δt for tbin=tbin_num"""
        return self._tbin_max
    
    @property
    def tbin_num( self ):
        """(Sort of) the number of time bins.

        See documentation on the class.  This is currently 26, but tbin values will be in the range [0,27].
        """
        return self._tbin_num

    @property
    def probbin_min( self ):
        """Middle value of probability for probbin=1"""
        return self._probbin_min

    @property
    def probbin_max( self ):
        """Middle value of probability for probbin=probbin_num"""
        return self._probbin_max
    
    @property
    def probbin_num( self ):
        """(Sort of) the number of probability bins.

        See documentation on the class.  This is currently 20, but probbin values will be in the range [0,21].
        """
        return self._probbin_num
    
    def probhist( self ):
        if self._probhist is not None:
            return self._probhist.copy( deep=True )

        self.logger.debug( "Sending query to get probabilistic metrics histogram table" )
        rows = self.run_query( "SELECT * FROM elasticc_view_classifications_probmetrics" )
        self.logger.debug( "Got response, pandifying" )
        self._probhist = pandas.DataFrame( rows )
        self._probhist.sort_values( ['classifierId', 'trueClassId', 'classId', 'tbin', 'probbin'], inplace=True )
        self._probhist.set_index( ['classifierId', 'trueClassId', 'classId', 'tbin', 'probbin'], inplace=True )
        self.logger.debug( "Done" )

        return self._probhist.copy( deep=True )
                
    def tbin_val( self, intbin ):
        tbin = numpy.atleast_1d( intbin ).copy()
        tbin[ tbin < 0 ] = 0
        tbin[ tbin > self._tbin_num+1 ] = self._tbin_num + 1
        rval = numpy.array( tbin.shape )
        rval = ( tbin  - 1 ) * self._delta_tbin + self._tbin_min + self._delta_tbin  / 2.
        # I think these next two are redundant
        rval[ tbin==0 ] = self._tbin_min - self._delta_tbin / 2.
        rval[ tbin==self._tbin_num+1 ] = self._tbin_max + self._delta_tbin / 2.
        return rval[0] if numpy.isscalar( intbin ) else rval

    def probbin_val( self, inprobbin ):
        probbin = numpy.atleast_1d( inprobbin ).copy()
        probbin[ probbin < 0 ] = 0
        probbin[ probbin > self._probbin_num+1 ] = self._probbin_num + 1
        rval = numpy.array( probbin.shape )        
        rval = ( probbin - 1 ) * self._delta_probbin + self._delta_probbin / 2.
        rval[ probbin==0 ] = -1
        rval[ probbin==self._probbin_num+1 ] = 1
        return rval[0] if numpy.isscalar( inprobbin ) else rval


    def right_probdiffs_for_object( self, diaObjectId ):
        self.logger.debug( f"Sending query to get probability differences for object {diaObjectId}" )
        rows = self.run_query( 'SELECT v."classifierId", v."trueClassId", '
                               '  v.earlytimebin, tbe.dtmin AS earlytimet0, tbe.dtmax AS earlytimet1,'
                               '  v.latetimebin, tbl.dtmin AS latetimet0, tbl.dtmax AS latetimet1,'
                               '  probdiff ',
                               'FROM elasticc_view_maxprobdiff v '
                               'INNER JOIN elasticc_maxprob_timebins tbe ON v.earlytimebin=tbe.timebin '
                               'INNER JOIN elasticc_maxprob_timebins tbl ON v.latetimebin=tbl.timebin '
                               'WHERE v."diaObjectId"=%(objid)s '
                               'ORDER BY "classifierId", earlytimebin, latetimebin',
                               { 'objid': diaObjectId } )
        self.logger.debug( f"Query done, pandafying" )
        return pandas.DataFrame( rows )
        

    def right_probdiffs_hist( self ):
        self.logger.debug( "Sending query to get the probability differences histogram thingy" )
        rows = self.run_query( 'SELECT v."classifierId", v."trueClassId", '
                               '  v.earlytimebin, tbe.dtmin AS earlytimet0, tbe.dtmax AS earlytimet1,'
                               '  v.latetimebin, tbl.dtmin AS latetimet0, tbl.dtmax AS latetimet1,'
                               '  probdiffbin, binmeanprobdiff, count '
                               'FROM elasticc_view_maxprobdiff_hist v '
                               'INNER JOIN elasticc_maxprob_timebins tbe ON v.earlytimebin=tbe.timebin '
                               'INNER JOIN elasticc_maxprob_timebins tbl ON v.latetimebin=tbl.timebin '
                               'ORDER BY "trueClassId", "classifierId", earlytimebin, latetimebin, probdiffbin ' )
        self.logger.debug( "Query done, pandafying" )
        df = pandas.DataFrame( rows )
        df['binmeanprobdiff'] = numpy.array( df['binmeanprobdiff'], dtype=numpy.float64 )
        df.set_index( [ 'classifierId', 'trueClassId', 'earlytimebin', 'latetimebin', 'probdiffbin' ], inplace=True )
        df['frac'] = ( df['count']
                       / df.groupby( ['classifierId','trueClassId',
                                      'earlytimebin','latetimebin'] )['count'].sum() )
        return df

    def right_probdiffs_hist_probbin_mean( self, probbin ):
        return -1.025 + 0.05/2 + (probbin-1) * 0.05
