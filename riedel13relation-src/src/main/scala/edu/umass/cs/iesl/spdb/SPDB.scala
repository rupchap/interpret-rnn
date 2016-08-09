package edu.umass.cs.iesl.spdb

import scala.Array._
import collection.mutable.{HashSet, HashMap, ArrayBuffer}
import util.Random
import math._
import java.util
import java.io.PrintStream
import cc.factorie.optimize.LimitedMemoryBFGS
import org.riedelcastro.nurupo.ProgressMonitor

/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 10/30/12
 * Time: 11:21 AM
 * Map arg to a low dimensional vector, so does relation(dep path)
 */

class SPDB(initialNumComponents: Int = 2) extends HasLogger {

  class Cell(val relation: String,
             val tuple: Seq[Any],
             var gold: Double,
             var hide: Boolean,
             var predicted: Double = 0.5,
             var feature: Boolean = false,
             val toPredict: Boolean = true) {

    def labeledTrainingData = !hide && toPredict
    def testData = hide & (gold != 0.5)

    var target = gold
    var constrained = false

    def turnOnFeature() {
      if (!feature) {
        feature = true
        featureCells.getOrElseUpdate(tuple, new ArrayBuffer[Cell]) += this
      }
    }

    def update(tensor : Boolean = true, pair : Boolean = true, bigram : Boolean = true) = {
      predicted = calculateScore( tuple, relation,  tensor, pair, bigram)
      predicted
    }

  }

  val cells = new HashMap[(String, Seq[Any]), Cell]
  val unaryCells = new HashMap[(String, Seq[Any]), Cell]
  val hidden = new ArrayBuffer[Cell]
  var _numComponents = initialNumComponents

  var _weights: Array[Double] = null

  var parameterCount = 0

  val relationComponentIndices = new HashMap[String, Array[Int]]
  val srcRelComponentIndices = new HashMap[String, Array[Int]]
  val dstRelComponentIndices = new HashMap[String, Array[Int]]
  val entComponentIndices = new HashMap[Any, Array[Int]]    //each argument has a low dimensional mapping
  val srcEntComponentIndices = new HashMap[Any, Array[Int]]   //source ent embedding
  val dstEntComponentIndices = new HashMap[Any, Array[Int]]   //dest ent embedding
  val tupleComponentIndices = new HashMap[Seq[Any], Array[Int]]
  val relationCells = new HashMap[String, ArrayBuffer[Cell]]

  //todo: not sure how to handle this, useful for bias
  val featureCells = new HashMap[Seq[Any], ArrayBuffer[Cell]]
  val featureIndices = new HashMap[Cell, Array[Int]]
  val relPairIndices = new HashMap[(String, String), Int]

  val entity2Cells = new HashMap[Any, ArrayBuffer[Cell]]
  var entities : Set[Any] = null
  val allEnts = new ArrayBuffer[Any]
  val srcEntities = new HashSet[Any]
  val dstEntities = new HashSet[Any]
  val tupleCells = new HashMap[Seq[Any], ArrayBuffer[Cell]]

  var indicesInitialized = false
  var weightsInitialized = false

  val labeledTraining = new HashSet[Cell]

  var maxAPIterations = 1
  var maxIterations = Int.MaxValue

  val random = new Random(Conf.conf.getLong("pca.seed"))
  logger.info("seed:" + Conf.conf.getLong("pca.seed"))

  var lambdaRel = 1.0
  var lambdaTuple = 1.0
  var lambdaEntity = 1.0

  var globalBias = 0.0
  var useGlobalBias = false
  var alphaIndices : Array[Int] = null  // alpha for tensor term
  val relationBiases = new HashMap[String,Int]
  val srcEntBiases = new HashMap[Any,Int]
  val dstEntBiases = new HashMap[Any,Int]
  val entBiases = new HashMap[Any,Int]
  val tupleBiases = new HashMap[Seq[Any],Int]
  var lambdaBias = 0.1
  var bias = true
  var alphaNorm = true


  var lambdaRelPair = 0.0

  var useCoordinateDescent = false
  var maxCores = 1
  var tolerance = 1e-9
  var gradientTolerance = 1e-9


  def relations = relationCells.keys
  def relationSet = relationCells.keySet
  
  val relationArray = new ArrayBuffer[String]
  val unaryRelArray = new ArrayBuffer[String]
  val allRelArray = new ArrayBuffer[String]
  val relStat = new HashMap[String,Int]
  

  def tuples = tupleCells.keys

  def getCells(relation: String) = relationCells.getOrElse(relation, Seq.empty)
  def getCells(tuple: Seq[Any]) = if(tuple.size > 1) tupleCells.getOrElse(tuple, Seq.empty)  else  entity2Cells.getOrElse(tuple(0),Seq.empty)
  def getCell(relation : String, tuple : Seq[Any]) : Option[Cell] = cells.get(relation->tuple)

  def numComponents = _numComponents

  def isToBePredicted(relation: String) = getCells(relation).exists(_.toPredict)

  final def sq(num: Double) = num * num

  def numComponents_=(c: Int) {
    _numComponents = c
  }

  def addCell(relation: String, tuple: Seq[Any], value: Double = 1.0,
              hide: Boolean = false, feature: Boolean = false, predict: Boolean = true) {
    for (entity <- tuple) allEnts += entity

    if (tuple.size == 2) {
      srcEntities += tuple(0)
      dstEntities += tuple(1)
    }
    val cell = new Cell(relation, tuple, value, hide, feature = feature, toPredict = predict)

    cells(relation -> tuple) = cell
    //if(tuple.size > 1) cells(relation -> tuple) = cell   else unaryCells(relation->tuple) = cell

    if (tuple.size == 1){
      for (ent <- tuple) entity2Cells.getOrElseUpdate(ent, new ArrayBuffer[Cell]) += cell
    }

    relationCells.getOrElseUpdate(relation, new ArrayBuffer[Cell]) += cell
    if (tuple.size > 1)
      tupleCells.getOrElseUpdate(tuple,new ArrayBuffer[Cell] )  += cell

    if (cell.feature)
      featureCells.getOrElseUpdate(tuple, new ArrayBuffer[Cell]) += cell
    if (hide && predict)
      hidden += cell
    if (cell.labeledTrainingData) labeledTraining += cell
    
    if(!hide && predict){
      //allRelArray += relation
      if (tuple.size == 2)
        relationArray += relation
      else unaryRelArray += relation
      val oldCount = relStat.getOrElseUpdate(relation,0)
      relStat.update(relation,oldCount+1)
    }
  }


  def calculateScore(cell : Cell, weights : Array[Double]) = {
    val theta = calculateScoreRaw(cell.tuple,cell.relation, weights)
    calculateProb(theta)
  }

  def calculateScore(tuple: Seq[Any], relation: String,  tensor : Boolean = true, pair : Boolean = true, bigram : Boolean = true) = {
    val theta = calculateScoreRaw(tuple,relation, _weights, tensor, pair, bigram)
    calculateProb(theta)
  }

  final def calculateProb(theta: Double): Double = {
    1.0 / (1.0 + exp(-theta))
  }

  final def calculateScoreRaw(tuple: Seq[Any], relation: String, weights : Array[Double], tensor : Boolean = true, pair : Boolean = true, bigram : Boolean = true) : Double = {
    var theta = 0.0
    if (useGlobalBias)   theta += globalBias

    if(tuple.size == 1)  {
      theta += calculateUnaryScore(tuple,relation,weights)
      if (bias){
        for (relBias <- relationBiases.get(relation); entBias <- entBiases.get(tuple(0)))
          theta += weights(relBias)  + weights(entBias)
      }

      return theta
    }

    val tensorScore = if (tuple.size > 1 && tensor) calculateTensorScore(tuple, relation, weights) else 0.0
    val tupleScore = if (pair) calculateTupleScore(tuple,relation,weights)  else 0.0
    val bigramScore = calculateBigramScore(tuple,relation,weights)
    //val featScore = calculateFeatScore(tuple, relation)
    if(tensor)
      theta += tensorScore
    if(pair)
      theta += tupleScore
    if (bigram)
      theta += bigramScore


    
    if(bias && tensor && pair) {

      for (relBias <- relationBiases.get(relation); srcBias <- srcEntBiases.get(tuple(0)); dstBias <- dstEntBiases.get(tuple(1)); tupleBias <- tupleBiases.get(tuple))
        theta += weights(relBias) + weights(srcBias) + weights(dstBias)  + weights(tupleBias)
    }
    else if (bias && tensor) {
      //logger.info("tensor bias")
      for (relBias <- relationBiases.get(relation); srcBias <- srcEntBiases.get(tuple(0)); dstBias <- dstEntBiases.get(tuple(1)))
        theta += weights(relBias) + weights(srcBias)  + weights(dstBias)
    }
    else if (bias && pair) {

      for (relBias <- relationBiases.get(relation); tupleBias <- tupleBiases.get(tuple))
        theta += weights(relBias)  + weights(tupleBias)
    }

    //logger.info( "Tensor score:"+tensorScore+",pair score:"+tupleScore)
    
    theta
  }

  final def calculateTensorScore(tuple: Seq[Any], relation: String, weights : Array[Double]) = {

    var sum = 0.0
    for (relIndices <- relationComponentIndices.get(relation);
         arg1Indices <- srcEntComponentIndices.get(tuple(0)); arg2Indices <- dstEntComponentIndices.get(tuple(1))) {
      var c = 0

      c = 0
      while (c < _numComponents) {
        val relIndex = relIndices(c)
        val relWeight = weights(relIndex)

        val arg1Index = arg1Indices(c)
        val arg2Index = arg2Indices(c)

        if(weights(arg1Index) < 0)   weights(arg1Index) = - weights(arg1Index)
        if(weights(arg2Index) < 0)   weights(arg2Index) = - weights(arg2Index)

        val arg1Weight = weights(arg1Index)
        val arg2Weight = weights(arg2Index)

        val alpha = weights(alphaIndices(c))
        
        if (alphaNorm)
          sum += arg1Weight * arg2Weight * relWeight * alpha
        else
          sum += arg1Weight*arg2Weight * relWeight
        //if (sum > 100) logger.info("weights:" + arg1Weight + " " + arg2Weight + " " + relWeight)
        c += 1
      }

    }

    sum
  }

  final def calculateTupleScore(tuple: Seq[Any], relation: String, weights : Array[Double]) = {

    var sum = 0.0

    for (relIndices <- relationComponentIndices.get(relation); tupleIndices <- tupleComponentIndices.get(tuple)  ) {
      var c = 0

      c = 0
      while (c < _numComponents) {
        val relIndex = relIndices(c)
        val relWeight = weights(relIndex)

        val tupleIndex = tupleIndices(c)
        val tupleWeight = weights(tupleIndex)

        sum += tupleWeight * relWeight

        c += 1
      }

    }

    sum
  }

  final def calculateBigramScore(tuple: Seq[Any], relation: String, weights : Array[Double]) : Double = {

    var sum = 0.0

    if(tuple.size == 1) return sum
    
    for (relIndices <- relationComponentIndices.get(relation); argIndices <- entComponentIndices.get(tuple(0)) ) {
      var c = 0

      c = 0
      while (c < _numComponents) {
        val relIndex = relIndices(c)
        val relWeight = weights(relIndex)

        val argIndex = argIndices(c)
        val argWeight = weights(argIndex)

        sum += argWeight * relWeight

        c += 1
      }
    }

    for (relIndices <- relationComponentIndices.get(relation); argIndices <- entComponentIndices.get(tuple(1)) ) {
      var c = 0

      c = 0
      while (c < _numComponents) {
        val relIndex = relIndices(c)
        val relWeight = weights(relIndex)

        val argIndex = argIndices(c)
        val argWeight = weights(argIndex)

        sum += argWeight * relWeight

        c += 1
      }
    }

    sum
  }

  final def calculateUnaryScore(tuple: Seq[Any], relation: String, weights : Array[Double]) : Double = {

    var sum = 0.0

    if(tuple.size != 1) return sum

    for (relIndices <- relationComponentIndices.get(relation); argIndices <- entComponentIndices.get(tuple(0)) ) {
      var c = 0

      c = 0
      while (c < _numComponents) {
        val relIndex = relIndices(c)
        val relWeight = weights(relIndex)

        val argIndex = argIndices(c)
        val argWeight = weights(argIndex)

        sum += argWeight * relWeight

        c += 1
      }
    }
    sum
  }

  def initializeIndices() {
    logger.info("Initializing Indices")
    def allocateIndices(howMany: Int) = {
      parameterCount += howMany
      Range(parameterCount - howMany, parameterCount).toArray
    }
    
    alphaIndices = allocateIndices(_numComponents)

    val relsToPredict = relations.filter(isToBePredicted).toSeq
    entities = allEnts.toSet
    
    logger.info("Initializing biases Indices")
    for (rel <- relsToPredict) {
      relationBiases.getOrElseUpdate(rel, allocateIndices(1)(0))
    }
    for (entity <- srcEntities) {
      srcEntBiases.getOrElseUpdate(entity, allocateIndices(1)(0))
    }
    for (entity <- dstEntities) {
      dstEntBiases.getOrElseUpdate(entity, allocateIndices(1)(0))
    }
    for (entity <- entities) {
      entBiases.getOrElseUpdate(entity, allocateIndices(1)(0))
    }
    for (tuple <- tuples) {
      tupleBiases.getOrElseUpdate(tuple, allocateIndices(1)(0))
    }

    logger.info("Initializing Relation Indices")       //todo: arg component indices for (src,rel) (rel,dst) matrix (rel, src-dst) matrix, maybe they should share relation vectors
    for (rel <- relsToPredict) {
      relationComponentIndices.getOrElseUpdate(rel, allocateIndices(_numComponents))
    }

//    for (rel <- relsToPredict) {
//      srcRelComponentIndices.getOrElseUpdate(rel, allocateIndices(_numComponents))
//    }
//
//    for (rel <- relsToPredict) {
//      dstRelComponentIndices.getOrElseUpdate(rel, allocateIndices(_numComponents))
//    }

    logger.info("Initializing Tuple Indices")
    for (tuple <- tuples) {
      tupleComponentIndices.getOrElseUpdate(tuple, allocateIndices(_numComponents))
    }

    logger.info("Initializing Entity Indices")

    for (entity <- entities) {
      entComponentIndices.getOrElseUpdate(entity, allocateIndices(_numComponents))     //todo: unary predicates separate from relationComponentIndices
    }

    for (entity <- srcEntities) {
      srcEntComponentIndices.getOrElseUpdate(entity, allocateIndices(_numComponents))
    }

    for (entity <- dstEntities) {
      dstEntComponentIndices.getOrElseUpdate(entity, allocateIndices(_numComponents))
    }

   /* logger.info("Initializing Feature Indices")   //todo:not sure about this for tensor, right now it is from tuple,bias
    for (cell <- cells.values; if (cell.toPredict)) {
      val relTarget = cell.relation
      val features = featureCells.getOrElse(cell.tuple, Seq.empty).filter(_ != cell)
      featureIndices(cell) = features.view.map(feat => {
        val relFeature = feat.relation
        assert(relFeature != relTarget)
        relPairIndices.getOrElseUpdate(relFeature -> relTarget, {
          parameterCount += 1
          parameterCount - 1
        })
      }).toArray
    }
    //todo: do this before the previous step
    ///create all pairs
    val allFeatRelations = featureCells.flatMap(_._2).map(_.relation).toSet
    for (feat <- allFeatRelations; target <- relsToPredict; if (target != feat))
      relPairIndices.getOrElseUpdate(feat -> target, {
        parameterCount += 1
        parameterCount - 1
      }) */


    logger.info("Num# Cells:               " + cells.size)
    logger.info("Num# Components:          " + _numComponents)
    logger.info("Num# Entities:            " + entities.size)
    logger.info("Num# Tuples:            " + tupleComponentIndices.size)
    logger.info("Num# Relations:           " + relationComponentIndices.size)
    logger.info("Num# Rel Pairs:           " + relPairIndices.size)
    logger.info("Num# Parameters:          " + parameterCount)
    logger.info("Num# training cells:      " + labeledTraining.size)

    indicesInitialized = true
  }

  def initializeWeights() {
    //initialize weights randomly
    _weights = Array.ofDim[Double](parameterCount)
    for (i <- _weights.indices) _weights(i) = random.nextDouble() //, divided by 100 is learned from Ben Marlin, commented by Limin
    //re-initialize tuple indices,  lambdaEnt = lambdaRel = 0.1, labmdaTuple = 10.0, tupleWeight = 0.01, argWeight = relWeight = 0.1
    //for (i <- entComponentIndices.values.flatMap(x=>x)) _weights(i) = random.nextDouble()/ 10.0
    weightsInitialized = true
  }

  //add bias for each tuple, each tuple has a bias component, specifically, for each tuple, we add a cell ('bias', tuple, hidden = false, feature = true, toPredict=false)
  //each relation except bias has a new weight, as explained by Sebastian weight(r',bias)
/*  def addBias() {
    for (tuple <- tuples) {
      addCell("bias", tuple, 1.0, hide = false, feature = true, predict = false)
    }
  } */


  def runLBFGS(updateRel: Boolean = true, updateEnt: Boolean = true,
               updateFeat: Boolean = true,
               normalizer: Option[Double] = None) {
    if (!indicesInitialized) initializeIndices()
    if (!weightsInitialized) initializeWeights()

    //create optimization problems
    val mProjectionObjective = new LLObjective(updateRel = updateRel, updateEnt = updateEnt, updateFeat = updateFeat,
      normalizer = normalizer)

    //create optimizer
    val mProjector = new LimitedMemoryBFGS(mProjectionObjective) //ConjugateGradient(mProjectionObjective) //new
    mProjector.maxIterations = maxIterations
    mProjector.tolerance = tolerance
    mProjector.gradientTolerance = gradientTolerance

    //initialize optimizer
    mProjectionObjective.setOptimizableParameters(_weights)

    //moment projection, for now, we are only doing mProjection, we should do infinite iterations
    def mProjection() {
      mProjector.optimize(maxIterations)
      _weights = mProjectionObjective.parameters
    }

    //run AP
    for (iteration <- 0 until maxAPIterations) {


      logger.info("M-Projection %d".format(iteration))
      mProjection()

    }
  }

  final def calculateCellGradientAndObjective(cell: Cell,
                                              gradient: Array[Double],
                                              weights: Array[Double],
                                              lagrangeWeights: Array[Double] = null,
                                              calculateGradient: Boolean = true,
                                              updateRelGradient: Boolean = true,
                                              updateEntGradient: Boolean = true,
                                              updateTupleGradient : Boolean = true,
                                              updateFeatGradient: Boolean = false): Double = {
    import cell._

    var theta: Double = calculateScoreRaw(cell.tuple, cell.relation, weights)
    val mu = calculateProb(theta)

    if((target*theta - log1p(exp(theta))).isNegInfinity ){
      logger.info("Theta(NegInfinity):" + theta)
      logger.info("Prob:" + mu + ",target:" + target)
    } 

    for (relIndices <- relationComponentIndices.get(relation); tupleIndices <- tupleComponentIndices.get(tuple);
         arg1Indices <- entComponentIndices.get(tuple(0)); arg2Indices <- entComponentIndices.get(tuple(1))) {

      var c = 0
      val feats = featureIndices.get(cell)
      val scale = 1 //tupleNormalizer(cell.tuple) * relationNormalizer(cell.relation)

      if (calculateGradient) {
        if (lagrangeWeights == null) {
          c = 0
          while (c < _numComponents) {
            val relIndex = relIndices(c)
            val relWeight = weights(relIndex)
            
            val tupleIndex = tupleIndices(c)
            val tupleWeight = weights(tupleIndex)
            
            val arg1Index = arg1Indices(c)
            val arg2Index = arg2Indices(c)
            val arg1Weight = weights(arg1Index)
            val arg2Weight = weights(arg2Index)

            if (updateRelGradient) {
              gradient(relIndex) += scale * arg1Weight * arg2Weight * (target - mu)  + scale * tupleWeight * (target - mu)
            }
            if (updateTupleGradient){
              gradient(tupleIndex) += scale * relWeight * (target - mu)
            }
            if (updateEntGradient) {
              gradient(arg1Index) += scale * arg2Weight * relWeight * (target - mu)
              gradient(arg2Index) += scale * arg1Weight * relWeight * (target - mu)
            }
            c += 1
          }

          if (updateFeatGradient) for (featIndices <- feats) {
            for (i <- featIndices) {
              gradient(i) += (target - mu)
            }
          }
        }
        else {
          //calculate the gradient and objective for the aux distribution
//          for (indices <- lagrangeIndices.get(cell); coefficients <- lagrangeCoefficients.get(cell)) {
//            var i = 0
//            while (i < indices.size) {
//              val index = indices(i)
//              gradient(index) -= mu * coefficients(i)
//              i += 1
//            }
//          }

        }
      }
    }

    if (lagrangeWeights == null) {
      
      target * theta - log1p(exp(theta))
    }
    // log1p(x) = log (1+x), this is natural parameter*suffificent stat - partition function
    else
      -log1p(exp(theta))
  }

  // update weights with scale, and return regularization term , lr: learning rate
  def updateOnCell(relation: String, tuple: Seq[Any], scale: Double, lr: Double, weights: Array[Double],   
                    lambdaRel: Double = lambdaRel,
                  lambdaEnt1: Double = lambdaEntity, lambdaEnt2 : Double = lambdaEntity,
                  lambdaTuple : Double = lambdaTuple,
                   lambdaFeat: Double = lambdaRelPair,
                    tensor : Boolean = true, pair : Boolean = true, bigram : Boolean = true,
                    updateEntGradient1 : Boolean = true, updateEntGradient2 : Boolean = true,
                    updateTupleGradient : Boolean = true) = {

    val updateRelGradient = true  

    val updateEntGradient = true

    var regObj = 0.0
  
    if (tuple.size > 1 ) { //binary case
      if (tensor && pair && !bigram) {

        for (relIndices <- relationComponentIndices.get(relation); tupleIndices <- tupleComponentIndices.get(tuple);
             arg1Indices <- srcEntComponentIndices.get(tuple(0)); arg2Indices <- dstEntComponentIndices.get(tuple(1))) {
          var c = 0
          while (c < _numComponents) {
            val relIndex = relIndices(c)
            val relWeight = weights(relIndex)

            val tupleIndex = tupleIndices(c)
            val tupleWeight = weights(tupleIndex)

            val arg1Index = arg1Indices(c)
            val arg2Index = arg2Indices(c)

            val arg1Weight = weights(arg1Index)
            val arg2Weight = weights(arg2Index)

            val alpha = weights(alphaIndices(c))

            if (updateRelGradient) {
              if (alphaNorm)
                weights(relIndex) += lr * (scale * arg1Weight * arg2Weight * alpha + scale * tupleWeight - lambdaRel * relWeight)
              else
                weights(relIndex) += lr * (scale * arg1Weight * arg2Weight + scale * tupleWeight - lambdaRel * relWeight)
              regObj -= (lambdaRel / 2.0 * sq(relWeight))
            }

            if (updateTupleGradient){
              weights(tupleIndex) +=  lr * (scale * relWeight - lambdaTuple * tupleWeight)
              //logger.info("tuple grad:" + scale * relWeight + "\t" + lambdaTuple * tupleWeight)
              regObj -= lambdaTuple / 2.0 * sq (tupleWeight)
            }


            if (updateEntGradient1){
              if (alphaNorm)
              weights(arg1Index) +=  lr * (scale * arg2Weight * relWeight * alpha - lambdaEnt1  * arg1Weight)
              else  weights(arg1Index) +=  lr * (scale * arg2Weight * relWeight  - lambdaEnt1  * arg1Weight)
              regObj -= lambdaEnt1 / 2.0 * sq(arg1Weight)
            }
            if (updateEntGradient2){
              if (alphaNorm)   weights(arg2Index) += lr * (scale * arg1Weight * relWeight * alpha - lambdaEnt2 * arg2Weight)
              else weights(arg2Index) += lr * (scale * arg1Weight * relWeight  - lambdaEnt2 * arg2Weight)
              regObj -= lambdaEnt2 / 2.0 * sq(arg2Weight)
            }
            
            
            //update alpha
            if(alphaNorm)
              weights(alphaIndices(c)) += lr * scale * arg1Weight * arg2Weight * relWeight
            c += 1

          }

        }

        if (bias){   //todo: for tuple
          for (relBias <- relationBiases.get(relation); srcBias <- srcEntBiases.get(tuple(0)); dstBias <- dstEntBiases.get(tuple(1)); tupleBias <- tupleBiases.get(tuple))  {
            regObj -= 0.5 * lambdaBias * sq(weights(relBias))
            regObj -= 0.5 * lambdaBias * sq(weights(srcBias))
            regObj -= 0.5 * lambdaBias * sq(weights(dstBias))
            regObj -= 0.5 * lambdaBias * sq(weights(tupleBias))

            weights(relBias) += lr * scale - weights(relBias) * lambdaBias
            weights(srcBias) += lr * scale - weights(srcBias) *  lambdaBias
            weights(dstBias) += lr * scale - weights(dstBias) *  lambdaBias
            weights(tupleBias) += lr * scale - weights(tupleBias) * lambdaBias
          }
        }
      }
      if (tensor && !pair && !bigram){
        for (relIndices <- relationComponentIndices.get(relation);
             arg1Indices <- srcEntComponentIndices.get(tuple(0)); arg2Indices <- dstEntComponentIndices.get(tuple(1))) {
          var c = 0
          while (c < _numComponents) {
            val relIndex = relIndices(c)
            val relWeight = weights(relIndex)

            val arg1Index = arg1Indices(c)
            val arg2Index = arg2Indices(c)

            if(weights(arg1Index) < 0)   weights(arg1Index) = - weights(arg1Index)
            if(weights(arg2Index) < 0)   weights(arg2Index) = - weights(arg2Index)

            val arg1Weight = weights(arg1Index)
            val arg2Weight = weights(arg2Index)

            val alpha = weights(alphaIndices(c))

            if (updateRelGradient) {
              if(alphaNorm)
                weights(relIndex) +=  lr * (scale * arg1Weight * arg2Weight * alpha - lambdaRel * relWeight)
              else  weights(relIndex) +=  lr * (scale * arg1Weight*arg2Weight  - lambdaRel * relWeight)
              regObj -= (lambdaRel / 2.0 * sq(relWeight))
            }

            if (updateEntGradient1){

              if (alphaNorm)
                weights(arg1Index) +=  lr * (scale * arg2Weight * relWeight * alpha - lambdaEnt1  * arg1Weight)
              else  weights(arg1Index) +=  lr * (scale * arg2Weight  * relWeight  - lambdaEnt1  * arg1Weight)
              regObj -= lambdaEnt1 / 2.0 * sq(arg1Weight)
            }
            if (updateEntGradient2){

              if (alphaNorm)   weights(arg2Index) += lr * (scale * arg1Weight * relWeight * alpha - lambdaEnt2 * arg2Weight)
              else weights(arg2Index) += lr * (scale * arg1Weight * relWeight  - lambdaEnt2 * arg2Weight)
              regObj -= lambdaEnt2 / 2.0 * sq(arg2Weight)
            }

            //update alpha
            if(alphaNorm)
              weights(alphaIndices(c)) += lr * scale * arg1Weight * arg2Weight * relWeight

            c += 1
          }
        }

        if (bias){   //todo: for tuple
          for (relBias <- relationBiases.get(relation); srcBias <- srcEntBiases.get(tuple(0)); dstBias <- dstEntBiases.get(tuple(1)))  {
            regObj -= 0.5 * lambdaBias * sq(weights(relBias))
            regObj -= 0.5 * lambdaBias * sq(weights(srcBias))
            regObj -= 0.5 * lambdaBias * sq(weights(dstBias))

            weights(relBias) += lr * scale - weights(relBias) * lambdaBias
            weights(srcBias) += lr * scale - weights(srcBias) *  lambdaBias
            weights(dstBias) += lr * scale - weights(dstBias) *  lambdaBias
          }
        }
      }
      if(pair && !tensor && !bigram){

        for (relIndices <- relationComponentIndices.get(relation);
             tupleIndices <- tupleComponentIndices.get(tuple)) {
          var c = 0
          while (c < _numComponents) {
            val relIndex = relIndices(c)
            val relWeight = weights(relIndex)
            val tupleIndex = tupleIndices(c)
            val tupleWeight = weights(tupleIndex)

            if (updateRelGradient) weights(relIndex) += lr * (scale * tupleWeight - lambdaRel * relWeight)
            if (updateTupleGradient) weights(tupleIndex) += lr * (scale * relWeight -  lambdaTuple * tupleWeight)
            if (updateRelGradient) regObj -= (lambdaRel / 2.0 * sq(relWeight))
            if (updateTupleGradient) regObj -= ( lambdaTuple / 2.0 * sq(tupleWeight))
            c += 1
          }
        }

        if(bias)
          for (relBias <- relationBiases.get(relation); tupleBias <- tupleBiases.get(tuple)){
            regObj -= 0.5 * lambdaBias * sq(weights(relBias))
            regObj -= 0.5 * lambdaBias * sq(weights(tupleBias))

            weights(relBias) += lr * scale - weights(relBias) * lambdaBias
            weights(tupleBias) += lr * scale - weights(tupleBias) * lambdaBias
          }
      }
      if(pair && bigram && !tensor){   // naacl f+e model
        //logger.info("bigram and pair update!")
        for (relIndices <- relationComponentIndices.get(relation);
             tupleIndices <- tupleComponentIndices.get(tuple)) {
          var c = 0
          while (c < _numComponents) {
            val relIndex = relIndices(c)
            val relWeight = weights(relIndex)
            val tupleIndex = tupleIndices(c)
            val tupleWeight = weights(tupleIndex)

            if (updateRelGradient) weights(relIndex) += lr * (scale * tupleWeight - lambdaRel * relWeight)
            if (updateTupleGradient) weights(tupleIndex) += lr * (scale * relWeight -  lambdaTuple * tupleWeight)
            if (updateRelGradient) regObj -= (lambdaRel / 2.0 * sq(relWeight))
            if (updateTupleGradient) regObj -= ( lambdaTuple / 2.0 * sq(tupleWeight))
            c += 1
          }
        }

        for (relIndices <- relationComponentIndices.get(relation);
             argIndices <- entComponentIndices.get(tuple(0))) {   //this is arg1 indices
          var c = 0
          while (c < _numComponents) {
            val relIndex = relIndices(c)
            val relWeight = weights(relIndex)
            val argIndex = argIndices(c)
            val argWeight = weights(argIndex)

            if (updateRelGradient) weights(relIndex) += lr * ( scale * argWeight - lambdaRel * relWeight)
            if (updateEntGradient) weights(argIndex) += lr * ( scale * relWeight -  lambdaEnt1 * argWeight)
            if (updateRelGradient) regObj -= (lambdaRel / 2.0 * sq(relWeight))
            if (updateEntGradient) regObj -= ( lambdaEnt1 / 2.0 * sq(argWeight))
            c += 1
          }
        }

        for (relIndices <- relationComponentIndices.get(relation);
             argIndices <- entComponentIndices.get(tuple(1))) {   //this is arg2 indices
          var c = 0
          while (c < _numComponents) {
            val relIndex = relIndices(c)
            val relWeight = weights(relIndex)
            val argIndex = argIndices(c)
            val argWeight = weights(argIndex)

            if (updateRelGradient) weights(relIndex) += lr * ( scale * argWeight - lambdaRel * relWeight)
            if (updateEntGradient) weights(argIndex) += lr * ( scale * relWeight -  lambdaEnt2 * argWeight)
            if (updateRelGradient) regObj -= (lambdaRel / 2.0 * sq(relWeight))
            if (updateEntGradient) regObj -= ( lambdaEnt2 / 2.0 * sq(argWeight))
            c += 1
          }
        }

      }
      if(bigram && !pair && !tensor){
        //logger.info("bigram only update!")
        for (relIndices <- relationComponentIndices.get(relation);
             argIndices <- entComponentIndices.get(tuple(0))) {   //this is arg1 indices
          var c = 0
          while (c < _numComponents) {
            val relIndex = relIndices(c)
            val relWeight = weights(relIndex)
            val argIndex = argIndices(c)
            val argWeight = weights(argIndex)

            if (updateRelGradient) weights(relIndex) += lr * ( scale * argWeight - lambdaRel * relWeight)
            if (updateEntGradient) weights(argIndex) += lr * ( scale * relWeight -  lambdaEnt1 * argWeight)
            if (updateRelGradient) regObj -= (lambdaRel / 2.0 * sq(relWeight))
            if (updateEntGradient) regObj -= ( lambdaEnt1 / 2.0 * sq(argWeight))
            c += 1
          }
        }

        for (relIndices <- relationComponentIndices.get(relation);
             argIndices <- entComponentIndices.get(tuple(1))) {   //this is arg2 indices
          var c = 0
          while (c < _numComponents) {
            val relIndex = relIndices(c)
            val relWeight = weights(relIndex)
            val argIndex = argIndices(c)
            val argWeight = weights(argIndex)

            if (updateRelGradient) weights(relIndex) += lr * ( scale * argWeight - lambdaRel * relWeight)
            if (updateEntGradient) weights(argIndex) += lr * ( scale * relWeight -  lambdaEnt2 * argWeight)
            if (updateRelGradient) regObj -= (lambdaRel / 2.0 * sq(relWeight))
            if (updateEntGradient) regObj -= ( lambdaEnt2 / 2.0 * sq(argWeight))
            c += 1
          }
        }
      }
    }else{   //unary case
      //logger.info("unary update")
      for (relIndices <- relationComponentIndices.get(relation); entIndices <- entComponentIndices.get(tuple(0))){
        var c = 0
        while (c < _numComponents) {
          val relIndex = relIndices(c)     // unary relation, attributes, fine-grained entity types
          val relWeight = weights(relIndex)

          val entIndex = entIndices(c)
          val entWeight = weights(entIndex)

          if (updateRelGradient) {
            weights(relIndex) += lr * (scale * entWeight - lambdaRel * relWeight)
            regObj -= (lambdaRel / 2.0 * sq(relWeight))
          }

          if (updateEntGradient){
            weights(entIndex) += lr * (scale * relWeight - lambdaEnt1 * entWeight)
            regObj -= lambdaEnt1 / 2.0 * sq (entWeight)
          }
          c += 1
        }
      }

      if(bias)
        for (relBias <- relationBiases.get(relation); entBias <- entBiases.get(tuple(0))){
          regObj -= 0.5 * lambdaBias * sq(weights(relBias))
          regObj -= 0.5 * lambdaBias * sq(weights(entBias))

          weights(relBias) += lr * scale - weights(relBias) * lambdaBias
          weights(entBias) += lr * scale - weights(entBias) * lambdaBias
        }
    }

    if (useGlobalBias){
      globalBias += lr * scale
      //logger.info("global bias:" + globalBias)
    }

    regObj
  }

  class LLObjective(updateRel: Boolean = true,
                    updateEnt: Boolean = true,
                    updateTuple : Boolean = true,
                    updateFeat: Boolean = true,
                    normalize: Boolean = true,
                    normalizer: Option[Double] = None) extends Objective {
    val gradient = Array.ofDim[Double](parameterCount)
    val cores = math.min(maxCores, Runtime.getRuntime.availableProcessors())
    lazy val gradientPerCore = for (_ <- 0 until cores) yield
      Array.ofDim[Double](parameterCount)

    def calculateGradientAndObjective(parameters: Array[Double]) = {
      var obj = 0.0
      util.Arrays.fill(gradient, 0.0)
      def addCellGradientsAndObjective(gradient: Array[Double]) = {
        var obj = 0.0
        for ( cell <- labeledTraining) {
          obj += calculateCellGradientAndObjective(cell, gradient, parameters,
            updateRelGradient = updateRel, updateEntGradient = updateEnt, updateTupleGradient = updateTuple,
            updateFeatGradient = updateFeat
          )
        }
        obj
      }
      if (cores == 1) {
        obj += addCellGradientsAndObjective(gradient)
      }
      else {
        logger.info("Calculating gradients on %d cores".format(cores))
        logger.info("Not implemented yet!")

      }

      logger.info("Likelihood:                   " + obj) //added by Limin Yao, so that we log it in log.txt

      //add relation regularizer
      for ((_, relComponents) <- relationComponentIndices) {   // rel -> Array[Int]
        var c = 0
        while (c < numComponents) {
          val index = relComponents(c)
          val weight = parameters(index)
          obj -= lambdaRel / 2.0 * sq(weight)
          if (updateRel) gradient(index) -= lambdaRel * weight
          c += 1
        }
      }

      for ((_, tupleComponents) <- tupleComponentIndices) {   // rel -> Array[Int]
        var c = 0
        while (c < numComponents) {
          val index = tupleComponents(c)
          val weight = parameters(index)
          obj -= lambdaTuple / 2.0 * sq(weight)
          if (updateTuple) gradient(index) -= lambdaTuple * weight
          c += 1
        }
      }

      //add ent regularizer
      for ((_, entComponents) <- entComponentIndices) {
        var c = 0
        while (c < numComponents) {
          val index = entComponents(c)
          val weight = parameters(index)

          obj -= lambdaEntity / 2.0 * sq(weight)
          if (updateEnt) gradient(index) -= lambdaEntity * weight

          c += 1
        }
      }

      //add rel pair regularizer , todo
//      for ((pair, index) <- relPairIndices) {
//        val weight = parameters(index)
//        obj -= lambdaRelPair / 2.0 * sq(weight)
//        if (updateFeat) gradient(index) -= lambdaRelPair * weight
//      }

      logger.info("Likelihood after regularizer: " + obj) //added by Limin Yao, so that we log it in log.txt
      //divide the obj and gradient by the number of entities, by Limin, learned from Ben Marlin
      val entSize = normalizer.getOrElse(entities.size.toDouble)
      if (normalize) obj = obj / entSize
      var norm = 0.0
      for (i <- gradient.indices) {
        if (normalize) gradient(i) = gradient(i) / entSize
        norm += gradient(i) * gradient(i)

        if(abs(_weights(i))>1.0)
          logger.info("weight:"+ i + ":" + _weights(i))
      }

      logger.info("Gradient Norm:                " + sqrt(norm))
      logger.info("Likelihood after divided by tuplesize: " + obj) //added by Limin Yao, so that we log it in log.txt
      //result
      GradientAndObjective(gradient, obj)
    }

    def domainSize = parameterCount
  }

  def runSGD(){
    if (!indicesInitialized) initializeIndices()
    if (!weightsInitialized) initializeWeights()

    val allowed = relations.toSet
    Logger.info("Preparing training data")
//    val cellsInOrder = tuples.toSeq.flatMap(getCells(_).view.filter(c => c.labeledTrainingData && allowed(c.relation)))
//    val trainingCells = Random.shuffle(cellsInOrder).toArray
    
    val trainingCells = cells.values.filter(c=>c.labeledTrainingData).toArray
    //val relationArray = relations.filter(isToBePredicted).toArray
    val tupleArray = tuples.toArray

    var learningRate = 0.05
    val randomSamples = trainingCells.size * 1

    ProgressMonitor.start("SGD", "SGD", trainingCells.size , maxIterations)
    //gpca style
    for (epoch <- 0 until maxIterations) {
      //do updates
      var count = 0
      var total = 0.0

      def pcaUpdate(tuple: Seq[Any], relation: String, target: Double, weight: Double = 1.0, updateEnt1 : Boolean = true, updateEnt2 : Boolean = true) {
        val score = calculateScoreRaw(tuple, relation, _weights, tensor = false)
        val prob = calculateProb(score)
        val scale = (target - prob) * weight
        val oldReg = updateOnCell(relation, tuple, scale, learningRate, _weights,updateEntGradient1 = updateEnt1, updateEntGradient2 = updateEnt2, tensor = false)
        val oldLL = target * score - log1p(exp(score))
        val oldObj = oldLL + oldReg
        total += oldObj
        count += 1
        ProgressMonitor.progress("SGD", 1, "%-14.6f".format(total / count))
      }

      //observed cells
      for (i <- 0 until trainingCells.size) {
        val cell = trainingCells(random.nextInt(trainingCells.size))
        //get objective and gradient from cell

        pcaUpdate(cell.tuple, cell.relation, cell.target)
      }
//      if (epoch > 100)
//        learningRate = 5*1.0/epoch
    }
  }

  //sample negative cells as in runBPR
  def runDynamicSGD(tensor : Boolean = false, pair : Boolean = true, bigram : Boolean = true){
    if (!indicesInitialized) initializeIndices()
    if (!weightsInitialized) initializeWeights()

    val allowed = relations.toSet
    Logger.info("Preparing training data")
    //    val cellsInOrder = tuples.toSeq.flatMap(getCells(_).view.filter(c => c.labeledTrainingData && allowed(c.relation)))
    //    val trainingCells = Random.shuffle(cellsInOrder).toArray

    val trainingCells = cells.values.filter(c=>c.labeledTrainingData).toArray //toSeq ++ unaryCells.values.toSeq
    val tupleArray = tuples.toArray
    val entArray = entities.toArray
    
    val unaryCandidates = relStat.toArray.sortBy(_._2).map(_._1).filterNot(rel =>rel.startsWith("path#")).take(relationSet.size/5) ++ unaryRelArray    //todo: get a threshold for cutting off
    

    var learningRate = 0.05

    ProgressMonitor.start("SGD-DYN", "SGD-DYN", trainingCells.size*(1+Conf.conf.getInt("pca.neg-features")) , maxIterations)
    //gpca style
    for (epoch <- 0 until maxIterations) {
      //do updates
      var count = 0
      var total = 0.0

      def pcaUpdate(tuple: Seq[Any], relation: String, target: Double, weight: Double = 1.0, tensor : Boolean = true, pair : Boolean = true, bigram : Boolean = true,
                    updateEnt1 : Boolean = true, updateEnt2 : Boolean = true) {
//        val tuplescore = calculateTupleScore(tuple,relation,_weights)
//        val tensorscore = calculateTensorScore(tuple,relation,_weights)
        val score = calculateScoreRaw(tuple, relation, _weights,  tensor, pair, bigram)
 //       logger.info("before update:" + tuple.mkString("|") + "\t" + relation + "\t" + score )
//        if (math.abs(score) > 100) {
//          logger.info("relation comp:" + relationComponentIndices.get(relation).get.map(indice=>_weights(indice)).mkString(" "))
//          logger.info("arg comp:" + entComponentIndices.get(tuple(0)).get.map(indice=>_weights(indice)).mkString(" "))
//        }
        val prob = calculateProb(score)
        val scale = (target - prob) * weight
        val oldReg = updateOnCell(relation, tuple, scale, learningRate, _weights, tensor = tensor, pair = pair, bigram = bigram,  updateEntGradient1 = updateEnt1, updateEntGradient2 = updateEnt2 )
        val oldLL = target * score - log1p(exp(score))
        val oldObj = oldLL + oldReg

        if (oldReg.isNegInfinity || oldReg.isNaN || oldLL.isNegInfinity || oldLL.isNaN){
          logger.info(tuple.mkString("|") + ", " + relation + ", prob:" + prob + " "+score)
          logger.info("relation comp:" + relationComponentIndices.get(relation).get.map(indice=>_weights(indice)).mkString(" "))
        }

        total += oldObj
        count += 1
       // logger.info("after update:" + score + "\t" + target)
        ProgressMonitor.progress("SGD-DYN", 1, "%-14.6f %1.1f %2.4f %2.4f %4.3f".format(total / count, target, oldLL, oldReg,  scale))
      }

      //observed cells
      for (i <- 0 until  trainingCells.size) {
        val cell = trainingCells(random.nextInt(trainingCells.size))
        
        var candidates : Seq[String] = if(cell.tuple.size > 1) relationArray else unaryRelArray
        //val candidates =  if(cell.tuple.size > 1) binaryCandidates  else unaryCandidates

        //positive cell update
        if (epoch < maxIterations/3)
          pcaUpdate(cell.tuple, cell.relation, cell.target, tensor = tensor, pair = pair, bigram = false, updateEnt1 = false)
        else if (epoch < maxIterations*2/3)  pcaUpdate(cell.tuple, cell.relation, cell.target, tensor = tensor, pair = pair, bigram = false, updateEnt2 = false)
//        if (epoch < maxIterations/2)
//          pcaUpdate(cell.tuple, cell.relation, cell.target, tensor = false, pair = pair)
        else
          pcaUpdate(cell.tuple, cell.relation, cell.target, tensor = tensor, pair = pair, bigram = bigram)
        
//        //weigh the unary cells
//        if (cell.tuple.size > 1)  pcaUpdate(cell.tuple, cell.relation, cell.target, tensor = tensor, pair = pair, bigram = bigram)
//        else  pcaUpdate(cell.tuple, cell.relation, cell.target, Conf.conf.getDouble("pca.unary-weight"), tensor = tensor, pair = pair, bigram = bigram)
        var relation2 = cell.relation
        /*  while (getCell(relation2, cell.tuple).isDefined) {
relation2 = relationArray(random.nextInt(relationArray.size))
}
pcaUpdate(cell.tuple, relation2, 0.0, tensor = tensor, pair = pair)  */

        //sample negative cells and update
        var miss = 0
        var negSamples = if (epoch < maxIterations/3) Conf.conf.getInt("pca.neg-features") + 1 else Conf.conf.getInt("pca.neg-features") - 1
        for (negSample <- 0 until negSamples) {
          var sampledIndex = random.nextInt(candidates.size)
          relation2 = candidates(sampledIndex)
          miss = 0
          while (getCell(relation2,cell.tuple).isDefined  && miss < 3) {
            sampledIndex += 1
            sampledIndex = sampledIndex % candidates.size
            relation2 = candidates(sampledIndex)
            miss += 1
          }
          if (getCell(relation2, cell.tuple ).isEmpty ) {
            if (epoch < maxIterations/3)
              pcaUpdate(cell.tuple, relation2, 0.0, tensor = tensor, pair = pair, bigram = bigram,  updateEnt1 = false)
            else if (epoch < maxIterations*2/3)  pcaUpdate(cell.tuple, relation2, 0.0, tensor = tensor, pair = pair, bigram = bigram, updateEnt2 = false)
//             if (epoch < maxIterations/2)
//              pcaUpdate(cell.tuple, relation2, 0.0, tensor = false, pair = pair)
            else
              pcaUpdate(cell.tuple, relation2, 0.0, tensor = tensor, pair = pair, bigram = bigram)

//            if(cell.tuple.size > 1)  pcaUpdate(cell.tuple, relation2, 0.0, tensor = tensor, pair = pair, bigram = bigram)
//            else pcaUpdate(cell.tuple, relation2, 0.0, Conf.conf.getDouble("pca.unary-weight"), tensor = tensor, pair = pair, bigram = bigram)

          }
          else  ProgressMonitor.progress("SGD-DYN", 1, "No cand")
        }


      /*  var tuple2 = cell.tuple
        if (cell.tuple.size == 1){
          val candidateTuples = entArray
          while (getCell(cell.relation, tuple2).isDefined) {
            tuple2 = Seq(candidateTuples(random.nextInt(candidateTuples.size)))
          }
        } else{
          val candidateTuples =  tupleArray
          while (getCell(cell.relation, tuple2).isDefined) {
            tuple2 =  candidateTuples(random.nextInt(candidateTuples.size))
          }
        }
        if (epoch < maxIterations/3)
          pcaUpdate(tuple2, cell.relation, 0.0, tensor = tensor, pair = pair, updateEnt2 = false)
        else if (epoch < maxIterations*2/3)  pcaUpdate(tuple2, cell.relation, 0.0, tensor = tensor, pair = pair, updateEnt1 = false)
//        if (epoch < maxIterations/2)
//          pcaUpdate(tuple2, cell.relation, 0.0, tensor = false, pair = pair)
        else
          pcaUpdate(tuple2, cell.relation, 0.0, tensor = tensor, pair = pair) */

      }

      if (epoch > 50 ) learningRate = 5.0/epoch  //     if (epoch > 20 ) learningRate = 1.0/epoch
    }

  }

  //tuple r > tuple r'     s, d, r > s, d, r'    s, d, r > s' d r    s, d, r > s, d', r   todo:distinguish unary and binary relations
  def runBPR(tensor : Boolean = true, pair : Boolean = true) {
    if (!indicesInitialized) initializeIndices()
    if (!weightsInitialized) initializeWeights()

    val allowed = relations.toSet
    Logger.info("Preparing training data")
    val trainingCells = cells.values.filter(c=>c.labeledTrainingData && c.target > 0.5).toArray
    //val relationArray = relations.filter(isToBePredicted).toArray
    val tupleArray = tuples.toArray
    val entArray = entities.toArray
    val arg1ToTuples = tuples.groupBy(_.apply(0))
    val arg2ToTuples = tuples.groupBy(_.apply(1))
    

    //averaging parameters
    var sum_weights = Array.ofDim[Double](parameterCount)
    util.Arrays.fill(sum_weights, 0.0)
    var learningRate = 0.02

    ProgressMonitor.start("BPR", "BPR", trainingCells.size, maxIterations)
    for (epoch <- 0 until maxIterations) {
      var total = 0.0
      var count = 0

      def bprUpdate(tuple: Seq[Any], relation1: String, relation2: String, target: Double, weight: Double = 1.0, tensor : Boolean = true, pair : Boolean = true,
                     updateEnt1 : Boolean = true, updateEnt2 : Boolean = true) {
        var score1 = calculateScoreRaw(tuple, relation1, _weights, tensor, pair)
        var score2 = calculateScoreRaw(tuple, relation2, _weights, tensor, pair)
        val score = score1 - score2
        val prob = calculateProb(score)
        val scale = (target - prob) * weight
        
        //update pos rel1 neg rel2
        val oldReg1 = updateOnCell(relation1, tuple, scale, learningRate, _weights, tensor = tensor,  pair = pair,
          updateEntGradient1 = updateEnt1, updateEntGradient2 = updateEnt2)
        //do not regularize tuple and ent since we already did it in oldReg1
        val oldReg2 = updateOnCell(relation2, tuple, -scale, learningRate, _weights, lambdaTuple = 0.0, lambdaEnt1 = 0.0, lambdaEnt2 = 0.0, tensor = tensor,  pair = pair,
          updateEntGradient1 = updateEnt1, updateEntGradient2 = updateEnt2)
        val oldReg = oldReg1 + oldReg2
        val oldLL = target * score - log1p(exp(score))
        //logger.info("LL obj:" + oldLL + ", Regularizer:" + oldReg)
        val oldObj = oldLL + oldReg
        total += oldObj
        count += 1
        ProgressMonitor.progress("BPR", 1, "%-14.6f %1.1f %3.4f".format(total / count, target, score))
      }

      def bprUpdate1(tuple1: Seq[Any], tuple2: Seq[Any], relation: String, target: Double, weight: Double = 1.0, tensor : Boolean = true, pair : Boolean = true) {
        val score1 = calculateScoreRaw(tuple1, relation, _weights, tensor, pair)
        val score2 = calculateScoreRaw(tuple2, relation, _weights, tensor, pair)
        val score = score1 - score2
        val prob = calculateProb(score)
        val scale = (target - prob) * weight
        //both updates change the relation components. This affects the oldReg term computation. We avoid this by
        //skipping relation reg. in the second step.
        val oldReg1 = updateOnCell(relation, tuple1, scale, learningRate, _weights, tensor = tensor, pair = pair)
        val oldReg2 = updateOnCell(relation, tuple2, -scale, learningRate, _weights, lambdaRel = 0.0, tensor = tensor, pair = pair)
        val oldReg = oldReg1 + oldReg2
        val oldLL = target * score - log1p(exp(score))
        val oldObj = oldLL + oldReg
        total += oldObj
        count += 1
        ProgressMonitor.progress("BPR", 1, "%-14.6f %1.1f %3.4f".format(total / count, target, score))
      }


      for (i <- 0 until trainingCells.size) {
        val cell = trainingCells(random.nextInt(trainingCells.size))

     /*   val candidates = if(cell.tuple.size > 1) relationArray else unaryRelArray

        var relation2 = cell.relation
        while (getCell(relation2, cell.tuple).isDefined) {
          relation2 = candidates(random.nextInt(candidates.size))
        }
        if (cell.target > 0.5) {
          bprUpdate(cell.tuple, cell.relation, relation2, 1.0, tensor = tensor, pair = pair)
        }   */


        var tuple2 = cell.tuple

        if (cell.tuple.size == 1){
          val candidateTuples = entArray
          while (getCell(cell.relation, tuple2).isDefined) {
            tuple2 = Seq(candidateTuples(random.nextInt(candidateTuples.size)))
          }
        } else{
          val candidateTuples =  tupleArray
          while (getCell(cell.relation, tuple2).isDefined) {
            tuple2 =  candidateTuples(random.nextInt(candidateTuples.size))
          }
        }


        if (cell.target > 0.5){
          if (epoch < maxIterations/2)
            bprUpdate1(cell.tuple, tuple2, cell.relation, 1.0, tensor = false, pair = pair)
          else
            bprUpdate1(cell.tuple, tuple2, cell.relation, 1.0, tensor = tensor, pair = pair)
        }
//        else
//          bprUpdate1(tuple2, cell.tuple, cell.relation, 1.0, tensor = tensor, pair = pair)



      }

      if (epoch > 50) learningRate = 2*1.0/epoch
    }
    
    //for debugging
//    for (cell <- trainingCells.take(100)){
//      val tensorScore = calculateTensorScore(cell.tuple,cell.relation,_weights)
//      val tupleScore = calculateTupleScore(cell.tuple,cell.relation,_weights)
//      logger.info(cell.tuple.mkString("|") + "\t" + cell.relation + "\t" + cell.target + "\t" + tensorScore + "\t" + tupleScore)
//    }

  }



  def debugTuples(tuples: Iterable[Seq[Any]], out: PrintStream, entityPrinter: Any => String = id => id.toString, tensor : Boolean = true, pair : Boolean = true, bigram : Boolean = true) {
    if(tuples == null || tuples.size == 0) return
    logger.info("Writing out tuples information")
    for (tuple <- tuples) {

      out.println("----")
      out.println(tuple.map(entityPrinter(_)).mkString(" | "))
      val components = tupleComponentIndices.get(tuple).map(_.map(_weights(_)).mkString(" "))
      out.println(components.getOrElse("No Components"))
      val allCells = getCells(tuple)
      val nonPredict = allCells.filter(!_.toPredict)
      out.println("Context:")
      for (cell <- nonPredict.filterNot(_.feature)) {
        out.println("  %s".format(cell.relation))
      }
      out.println("Feats:")
      for (cell <- nonPredict.filter(_.feature)) {
        out.println("  %s".format(cell.relation))
      }
      val cells = allCells.filter(_.toPredict)
      cells.foreach(_.update(tensor,pair,bigram))
      val sortedObs = cells.filter(!_.hide).sortBy(-_.predicted)
      out.println("Observed:")
      for (cell <- sortedObs) {
        out.println("  %s %6.4f %6.4f %s ".format(if (cell.hide) "H" else "O", cell.target, cell.predicted, cell.relation))
      }
      out.println("Hidden:")
      val sortedHidden = cells.filter(_.hide)/*.filter(_.predicted > 0.5)*/.sortBy(-_.predicted)

      for (cell <- sortedHidden) {
        out.println("  %s %6.4f %6.4f %s ".format(if (cell.hide) "H" else "O", cell.target, cell.predicted, cell.relation))
      }

    }
    out.close
  }

  //print out top predictions, either binary relations for a pair, or unary relations for an entity
  def debugMAP(tuples: Iterable[Seq[Any]], out: PrintStream,  entityPrinter: Any => String = id => id.toString, tensor : Boolean = true, pair : Boolean = true, bigram : Boolean = true) {
    if(tuples == null || tuples.size == 0) return
    logger.info("Writing out map tuple predictions")
    for (tuple <- tuples) {

     
      out.println("----")
      out.println(tuple.map(entityPrinter(_)).mkString(" | "))

      val tupleComponents =  tupleComponentIndices.get(tuple).map(_.map(_weights(_)).sortWith((x,y)=> x > y).take(10).mkString(" ")).getOrElse("None" )
//      val arg1Components = if(entComponentIndices.get(tuple(0)).isDefined) entComponentIndices.get(tuple(0)).map(_.map(_weights(_)).sortWith((x,y)=> x > y).take(10).mkString(" "))    else "None"
//      val arg2Components = if(entComponentIndices.get(tuple(1)).isDefined)  entComponentIndices.get(tuple(1)).map(_.map(_weights(_)).sortWith((x,y)=> x > y).take(10).mkString(" "))    else "None"
      
      out.println("tuple comp:" + tupleComponents)
//      out.println("ent comp:" + arg1Components )
//      out.println("ent comp:" + arg2Components )

      val obsCells = getCells(tuple).filter(!_.hide)
      out.println(obsCells.map(cell=>"  Obs:" + cell.target + " " +  cell.relation).mkString("\n"))

      val preds = new ArrayBuffer[(String,Double)]
   
      val candRels = if (tuple.size > 1) relations.filter(rel => rel.startsWith("path#") || rel.startsWith("REL$")) else  relations.filterNot(rel => rel.startsWith("path#") && rel.startsWith("REL$"))
      for (relation <- candRels){
        val score = calculateScoreRaw(tuple,relation, _weights, tensor, pair, bigram)
        preds += relation -> score
      }
      
      val sorted = preds.sortWith((x,y) => x._2 > y._2).take(10)
      for ((rel,score) <- sorted ){
        val prob = calculateProb(score)
        if(!getCell(rel,tuple).isEmpty) {
          for (cell <- getCell(rel,tuple) )  {
            //out.println("  %s %6.4f %6.4f %s ".format(if (cell.hide) "H" else "O", cell.target, prob, cell.relation))
            if (!rel.startsWith("REL") && !cell.hide)
              out.println("1\t" + cell.relation + "\t" + tuple.mkString("\t")  + "\t" + prob )
            else out.println("\t" + cell.relation + "\t" + tuple.mkString("\t")  + "\t" + prob )
          }
        } else {
          //out.println("  %s %6.4f %6.4f %s ".format( "H" , 0.0, prob , rel))
          out.println("\t" + rel + "\t" + tuple.mkString("\t")  + "\t" + prob )
        }
      }
      

    }
    out.close
  }
  
  def heldoutRank(tuples : Iterable[Seq[Any]], out : PrintStream, filter : Boolean = false, entityPrinter: Any => String = id => id.toString, tensor : Boolean = true, pair : Boolean = true, bigram : Boolean = true) {
    if(tuples == null || tuples.size == 0) return
    logger.info("Writing out top predictions for tuple")
    var threshold = 0.0
    val ranks = new ArrayBuffer[Int]

    for (tuple <- tuples) {

      out.println("----")
      out.println(tuple.map(entityPrinter(_)).mkString(" | "))

      val preds = new ArrayBuffer[(String,Double)]
      threshold = 1.0e+9
      val hiddenCells = if(!filter) getCells(tuple).filter(_.target>0.5).filter(_.hide) else getCells(tuple).filter(_.target>0.5).filter(_.hide).filter(_.relation.startsWith("REL$"))

      val obsCells = getCells(tuple).filter(!_.hide)
      out.println(obsCells.map(cell=>"  Obs:" + cell.target + " " +  cell.relation).mkString("\n"))
      for (cell <- hiddenCells){
        val score = calculateScoreRaw(cell.tuple,cell.relation,_weights, tensor, pair, bigram)

        if (score < threshold) {
          threshold = score
        }
      }
      //only rank binary relations for an entity pair
      val candRels = if(!filter) relations.filterNot(_.startsWith("REL$")) else relations.filter(_.startsWith("REL$")) // this is for ranking Freebase relations
      for (relation <- candRels){
        val score = calculateScoreRaw(tuple,relation, _weights, tensor, pair, bigram)
        if (score >= threshold) {
          preds += relation -> score
        }

      }

      val sorted = preds.sortWith((x,y) => x._2 > y._2)
      var rank = 1
      for ((rel,score) <- sorted ){
        val prob = calculateProb(score)
        if(!getCell(rel,tuple).isEmpty) {
          for (cell <- getCell(rel,tuple) )  {
            if (rank <= 10)
              out.println("  %s %6.4f %6.4f %s ".format(if (cell.hide) "H" else "O", cell.target, prob, cell.relation))
            if (cell.hide && cell.target > 0.5) {
              ranks += rank
              out.println("Rank\t" + tuple.mkString("|") + "\t" +  cell.relation + "\t" + prob +  "\t" + rank)
            }
          }
        } else {
          if (rank <= 10)
            out.println("  %s %6.4f %6.4f %s ".format( "H" , 0.0, prob , rel))
        }
        rank += 1
      }
    }

    val sum = ranks.foldLeft(0.0)((x,y) => x + y)
    out.println("Size:" + ranks.size)
    out.println("Mean rank:" + sum*1.0/ranks.size)
    val sortedRanks = ranks.sortWith((x,y)=>x<y)
    out.println("Median rank:" + sortedRanks(ranks.size/2))


    out.close
  }


  def heldoutDS(tuples : Iterable[Seq[Any]], out : PrintStream, filter : Boolean = false, entityPrinter: Any => String = id => id.toString, tensor : Boolean = true, pair : Boolean = true, bigram : Boolean = true) {
    if(tuples == null || tuples.size == 0) return
    logger.info("Writing out top predictions for tuple")
   
    val predsOverall = new ArrayBuffer[(Double, String, String, String)]

    for (tuple <- tuples) {

      val hiddenCells = if(!filter) getCells(tuple).filter(_.target>0.5).filter(_.hide) else getCells(tuple).filter(_.target>0.5).filter(_.hide).filter(_.relation.startsWith("REL$"))

      if (hiddenCells.size > 0) {
        val goldlabel = hiddenCells(0).relation
        //only rank binary relations for an entity pair
        val candRels = if(!filter) relations.filterNot(_.startsWith("REL$")) else relations.filter(_.startsWith("REL$")) // this is for ranking Freebase relations
        for (relation <- candRels){
          val score = calculateScoreRaw(tuple,relation, _weights, tensor, pair, bigram)
          val prob =  calculateProb (score)
          predsOverall += ((prob, tuple.mkString("|"), goldlabel, relation))
        }
      }
    }
  

    val sorted = predsOverall.sortBy(-_._1)

    for ((score, tupleString, goldLabel, bestLabel) <- sorted) {

      out.println(score + "\t" + tupleString + "\t" + goldLabel + "\t" + bestLabel)
    }

    out.close
  }

  
  def debugModel(out: PrintStream, binary : Boolean = true, unary : Boolean = false) {

    out.println("Global Bias: " + globalBias)
    
    //out.println("Tensor coefficients:" + alphaIndices.map( x=> _weights(x)).mkString(" ") )
    //for each tuple component show top k tuples

    out.println("Entity/Relation components")
    for (c <- 0 until numComponents) {
      out.println(" Component " + c)
      if (binary){
      val sortedSrcEnts = srcEntComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      val sortedDstEnts = dstEntComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      val sortedTuples = tupleComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      val sortedEnts = entComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      val sortedRelations = relationComponentIndices.filter(rel => rel._1.startsWith("path#") && rel._1.startsWith("REL$")).toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      //val sortedRelations = relationComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      for ((tuple , relation) <- sortedTuples.take(10) zip  sortedRelations.take(10)) {
        out.println("   %6.4f %-50s %7.4f %s".format( tuple._2, tuple._1.mkString(" | "), relation._2, relation._1))
      }
      for( (ent, relation) <- sortedEnts.take(10) zip  sortedRelations.take(10)) {
        out.println("   %6.4f %-50s %7.4f %s".format(ent._2, ent._1, relation._2, relation._1))
      }


      }

      if (unary) {
      val sortedEnts = entComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      val sortedUnaryRelations = relationComponentIndices.filterNot( rel => rel._1.startsWith("path#") && rel._1.startsWith("REL$")).toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      for ((ent, relation) <- sortedEnts.take(10) zip sortedUnaryRelations.take(10)) {
        out.println("   %6.4f %-50s %7.4f %s".format(ent._2, ent._1, relation._2, relation._1))
      }
      }

    }
    
    out.println("Rel vectors")
    for (relation <- relations){
      out.println(relation + ":" + relationComponentIndices.get(relation).map(_.map(_weights(_)).zipWithIndex.sortWith((x,y)=> x._1 > y._1).map(x=>x._2 + ":"+"%4.3f".format(x._1)).mkString(" ")).getOrElse("None") )
    }
    
    out.println("Rel biases")
    val biasSeq = new ArrayBuffer[(String, Double)]
    for (relation <- relations){
      biasSeq += relation -> _weights(relationBiases.get(relation).get)
      //out.println(relation +":" + relationBiases.get(relation).map (idx=>"%4.3f".format(_weights(idx))).getOrElse ("None") )
    }
    val biasSorted = biasSeq.sortBy(-_._2)
    out.println(biasSorted.map(x=>x._1+":"+x._2).mkString("\n"))
//    out.println("Ent vectors")
//    for (ent <- srcEntities){
//      out.println(ent + ":" + srcEntComponentIndices.get(ent).map(_.map(_weights(_)).zipWithIndex.sortWith((x,y)=> x._1 > y._1).map(x=>x._2 + ":"+"%4.3f".format(x._1)).mkString(" ")).getOrElse("None") )
//    }
//
//    for (ent <- dstEntities){
//      out.println(ent + ":" + dstEntComponentIndices.get(ent).map(_.map(_weights(_)).zipWithIndex.sortWith((x,y)=> x._1 > y._1).map(x=>x._2 + ":"+"%4.3f".format(x._1)).mkString(" ")).getOrElse("None") )
//    }

//    out.println("Tuple vectors")
//    for (tuple <- tuples){
//      out.println(tuple.mkString("|")  + ":" + tupleComponentIndices.get(tuple).map(_.map(_weights(_)).zipWithIndex.sortWith((x,y)=> x._1 > y._1).map(x=>x._2 + ":"+"%4.3f".format(x._1)).mkString(" ")).getOrElse("None") )
//    }

    out.close
  }

}