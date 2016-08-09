package edu.umass.cs.iesl.spdb

import org.riedelcastro.nurupo._
import math._
import util.Random
import cc.factorie.optimize._
import io.Source
import scala.Array
import collection.mutable.{HashSet, ArrayBuffer, HashMap}
import java.io.{PrintStream, InputStream, File, FileInputStream}
import java.util
import collection.mutable

/**
 * A SelfPredictingDatabase is a collection of relations and a probabilistic model that learns how to predict the cells
 * of the database based on other cells.
 * @author sriedel
 */
class SelfPredictingDatabase(initialNumComponents: Int = 2, initialNumArgComponents: Int = 0) extends HasLogger {

  /*
      Different types of cell:
        (1) labeled training
        (2) unlabeled training
        (3) feature
        (4) test (may also be unlabeled training)

      toPredict: whether we maximize the loglikelihood of this cell, true means we maximize over this cell, used as a training cell if the cell is observed
      feature:  whether the cell is used as a feature, we don't try to maximize the loglikelihood of this cell
      default settings for gPCA: feature = false, toPredict = true
      fixed feature/classifier: feature = true, toPredict = false
      gPCA + classifier : feature = true, toPredict = true
   */

  class Cell(val relation: String,
             val tuple: Seq[Any],
             var gold: Double,
             var hide: Boolean,
             var predicted: Double = 0.5,
             var feature: Boolean = false,
             val toPredict: Boolean = true) {

    def labeledTrainingData = !hide && toPredict

    def unlabeledTrainingData = hide && constrained

    def testData = hide & (gold != 0.5)

    var target = gold
    var constrained = false

    def turnOnFeature() {
      if (!feature) {
        feature = true
        featureCells.getOrElseUpdate(tuple, new ArrayBuffer[Cell]) += this
      }
    }

    def update() = {
      predicted = predictCell(relation, tuple)
      predicted
    }

    lazy val indices =
      (relationComponentIndices(relation) ++ tupleComponentIndices(tuple) ++
        featureCells(tuple).filter(_ != this).map(c => relPairIndices(c.relation, relation)) ++
        (if (useBias) Array(biasIndex) else Array.empty[Int]))

  }

  private val cells = new HashMap[(String, Seq[Any]), Cell]
  private val hidden = new ArrayBuffer[Cell]
  private var _numComponents = initialNumComponents
  private var _numArgComponents = initialNumArgComponents
  private var _weights: Array[Double] = null
  private var _langrangeWeights: Array[Double] = null


  private var parameterCount = 0
  private var lagrangeParameterCount = 0
  private val relationComponentIndices = new HashMap[String, Array[Int]]
  private val tupleComponentIndices = new HashMap[Seq[Any], Array[Int]]
  private val tupleCells = new HashMap[Seq[Any], ArrayBuffer[Cell]]
  private val relationCells = new HashMap[String, ArrayBuffer[Cell]]
  private val featureCells = new HashMap[Seq[Any], ArrayBuffer[Cell]]
  private val featureIndices = new HashMap[Cell, Array[Int]]
  private val relPairIndices = new HashMap[(String, String), Int]
  private val counts = new mutable.HashMap[(String,String),Int]()

  private val relPairIndicesInv = new HashMap[Int, (String, String)]
  private val entity2Cells = new HashMap[Any, ArrayBuffer[Cell]]
  private val entities = new HashSet[Any]
  private var indicesInitialized = false
  private var weightsInitialized = false

  private val learningRate = Conf.conf.getDouble("pca.learning-rate")


  def featureIndicesFor(tuple: Seq[Any], relation: String) = {
    //    //for debugging
    //    val cell = getCell(relation,tuple)
    //    if(cell.size > 0) logger.info(cell.get.relation + "\t" + cell.get.tuple.mkString ("|") )
    //    else logger.info("No cell for " + relation + "\t" + tuple.mkString ("|"))

    val cached = getCell(relation, tuple).map(featureIndices(_))
    val result = cached.getOrElse(featureCells.getOrElse(tuple, Seq.empty).filter(_.relation != relation).flatMap(c => relPairIndices.get(c.relation -> relation)).toArray)
    result
  }

  //for each relation and argument we have components that will get dot-multipled with a
  //representation of the entity at that argument
  private val argComponentIndices = new HashMap[(String, Int), Array[Int]]()
  private val entityComponentIndices = new HashMap[Any, Array[Int]]()
  private var biasIndex = -1

  private val labeledTraining = new HashSet[Cell]
  private val unlabeledTraining = new HashSet[Cell]

  private val constraints = new ArrayBuffer[Constraint]
  private val lagrangeIndices = HashMap[Cell, Array[Int]]()
  private val lagrangeCoefficients = HashMap[Cell, Array[Double]]()

  var maxAPIterations = 1
  var maxIterations = Int.MaxValue

  val random = new Random(Conf.conf.getLong("pca.seed"))

  var lambdaRel = 1.0
  var lambdaTuple = 1.0
  var lambdaArg = 1.0
  var lambdaEntity = 1.0

  var lambdaRelPair = 0.0
  var lambdaBias = 0.1
  var useBias = false
  var useCoordinateDescent = false
  var maxCores = 1
  var tolerance = 1e-9
  var gradientTolerance = 1e-9
  var tupleNormalizer: (Seq[Any] => Double) = t => 1.0
  var relationNormalizer: (String => Double) = t => 1.0

  val tupleLambdas = new HashMap[Seq[Any], Double]
  val tupleParents = new HashMap[Seq[Any], Seq[Any]]
  val entityParents = new HashMap[Any, Any]
  val entityLambdas = new HashMap[Any, Double]
  private val parameterParents = new HashMap[Int, Int]
  private val parameterLambdas = new HashMap[Int, Double]

  private val touched = new mutable.HashMap[Int,Int]


  def numComponents_=(c: Int) {
    _numComponents = c
  }

  def numArgComponents_=(c: Int) {
    _numArgComponents = c
  }

  def relations = relationCells.keys

  def relationSet = relationCells.keySet

  def tuples = tupleCells.keys

  def numArgComponents = _numArgComponents

  def numComponents = _numComponents

  val geTerms = new ArrayBuffer[GETerm]

  /**
   * A GE term we can add to the LL objective.
   */
  trait GETerm {
    def calculateGradientAndObj(weights: Array[Double], gradient: Array[Double]): Double
  }

  class NegKLofTargetAndCellAverage(val cells: Seq[Cell], val target: Array[Double],
                                    val scale: Double = 1.0, val name: String = "No Name") extends GETerm {
    def calculateGradientAndObj(weights: Array[Double], gradient: Array[Double]) = {
      val current = Array.ofDim[Double](2)
      val size = cells.size
      //first calculate objective
      for (cell <- cells) {
        val theta = calculateCellScore(cell, weights)
        val mu = calculateProb(theta)
        current(0) += (1.0 - mu) / size
        current(1) += mu / size
      }
      val quot = Array(target(0) / current(0), target(1) / current(1))
      val obj = -scale * ((target(0) * log(quot(0))) + (target(1) * log(quot(1))))
      logger.info("GE Term:              " + name)
      logger.info("Target Expectations:  " + target.map(t => "%7.3f".format(t)).mkString(" "))
      logger.info("Current Expectations: " + current.map(t => "%7.3f".format(t)).mkString(" "))
      logger.info("Current KL Obj:       " + obj)
      //now calculate gradient
      var gradientNorm = 0.0
      for (cell <- cells) {
        val theta = calculateCellScore(cell, weights)
        val mu = calculateProb(theta)
        for (relIndices <- relationComponentIndices.get(cell.relation);
             tupleIndices <- tupleComponentIndices.get(cell.tuple)) {
          var c = 0
          val feats = featureIndices.get(cell)
          if (useBias) {
            val biasGradient = scale * (quot(0) * (-mu) + quot(1) * (1 - mu)) / size
            gradient(biasIndex) += biasGradient
            gradientNorm += sq(biasGradient)
          }
          while (c < _numComponents) {
            val relIndex = relIndices(c)
            val relWeight = weights(relIndex)
            val tupleIndex = tupleIndices(c)
            val tupleWeight = weights(tupleIndex)
            val relGradient = scale * (quot(0) * (tupleWeight * (-mu)) + quot(1) * (tupleWeight * (1 - mu))) / size
            val tupleGradient = scale * (quot(0) * (relWeight * (-mu)) + quot(1) * (relWeight * (1 - mu))) / size
            gradient(relIndex) += relGradient
            gradient(tupleIndex) += tupleGradient
            gradientNorm += sq(relGradient)
            gradientNorm += sq(tupleGradient)
            c += 1
          }
          if (_numArgComponents > 0) {
            for (arg <- 0 until arity(cell.relation)) {
              for (argIndices <- argComponentIndices.get(cell.relation -> arg);
                   entIndices <- entityComponentIndices.get(cell.tuple(arg))) {
                var ac = 0
                while (ac < _numArgComponents) {
                  val argIndex = argIndices(ac)
                  val entIndex = entIndices(ac)
                  val argWeight = weights(argIndex)
                  val entWeight = weights(entIndex)
                  val argGradient = scale * (quot(0) * (entWeight * (-mu)) + quot(1) * (entWeight * (1 - mu))) / size
                  val entGradient = scale * (quot(0) * (argWeight * (-mu)) + quot(1) * (argWeight * (1 - mu))) / size
                  gradient(argIndex) += argGradient
                  gradient(entIndex) += entGradient
                  gradientNorm += sq(argGradient)
                  gradientNorm += sq(entGradient)
                  ac += 1
                }
              }
            }
          }
          for (featIndices <- feats) {
            for (i <- featIndices) {
              val featGradient = scale * (quot(0) * (-mu) + quot(1) * (1 - mu)) / size
              gradient(i) += featGradient
              gradientNorm += sq(featGradient)
            }
          }
        }
      }
      logger.info("GEGradient Norm:      " + sqrt(gradientNorm))

      obj
    }
  }


  trait Constraint {
    def cells: Seq[Cell]

    //index of constraint wrt to each cell, will be set externally
    var localIndices: Array[Int] = null
    var lagrangeIndex: Int = -1

    def lagrangeBounded: Boolean

    def regularizer: Double = 0.0

    def target: Double

    def l1: Boolean = false
  }

  class AtMostNConstraint(val cells: Seq[Cell], n: Double = 1.0) extends Constraint {
    def lagrangeBounded = true

    def target = n
  }

  def isToBePredicted(relation: String) = getCells(relation).exists(_.toPredict)

  def getCellsForEntity(entity:Any) = entity2Cells.getOrElse(entity,Seq.empty)

  def addCell(relation: String, tuple: Seq[Any], value: Double = 1.0,
              hide: Boolean = false, feature: Boolean = false, predict: Boolean = true) {
    for (entity <- tuple) entities += entity
    val cell = new Cell(relation, tuple, value, hide, feature = feature, toPredict = predict)
    cells(relation -> tuple) = cell
    relationCells.getOrElseUpdate(relation, new ArrayBuffer[Cell]) += cell
    tupleCells.getOrElseUpdate(tuple, new ArrayBuffer[Cell]) += cell
    if (cell.feature)
      featureCells.getOrElseUpdate(tuple, new ArrayBuffer[Cell]) += cell
    if (hide && predict)
      hidden += cell
    if (cell.labeledTrainingData) labeledTraining += cell
    for (ent <- tuple) entity2Cells.getOrElseUpdate(ent, new ArrayBuffer[Cell]) += cell
  }

  def removeCell(cell: Cell) {
    require(!cell.constrained)
    import cell._
    cells.remove(relation -> tuple)
    relationCells(relation).remove(relationCells(relation).indexOf(cell))
    tupleCells(tuple).remove(tupleCells(tuple).indexOf(cell))
    if (relationCells(relation).isEmpty) relationCells.remove(relation)
    if (tupleCells(tuple).isEmpty) tupleCells.remove(tuple)
    if (feature)
      featureCells(tuple).remove(featureCells(tuple).indexOf(cell))
    if (hide && toPredict)
      hidden -= cell
    if (labeledTrainingData) labeledTraining -= cell
    for (ent <- tuple) entity2Cells(ent) -= cell


  }

  def addConstraint(constraint: Constraint) {
    for (cell <- constraint.cells) {
      cell.constrained = true
      if (cell.unlabeledTrainingData) unlabeledTraining += cell
    }
    constraints += constraint
  }

  def getCell(relation: String, tuple: Seq[Any]): Option[Cell] = cells.get(relation -> tuple)

  def getCells(tuple: Seq[Any]) = tupleCells.getOrElse(tuple, Seq.empty)

  def getCells(relation: String) = relationCells.getOrElse(relation, Seq.empty)

  final def calculateArgEntScore(tuple: Seq[Any], relation: String) = {
    if (_numArgComponents == 0) 0.0
    else {
      var theta = 0.0
      for (arg <- 0 until arity(relation)) {
        for (argIndices <- argComponentIndices.get(relation -> arg);
             entIndices <- entityComponentIndices.get(tuple(arg))) {
          var ac = 0
          while (ac < _numArgComponents) {
            val argWeight = _weights(argIndices(ac))
            val entWeight = _weights(entIndices(ac))
            theta += argWeight * entWeight
            ac += 1
          }
        }
      }
      theta
    }

  }

  final def calculateFeatScore(tuple: Seq[Any], relation: String) = {
    var theta = 0.0
    if (indicesInitialized) for (i <- featureIndicesFor(tuple, relation)) theta += _weights(i)
    theta
  }


  final def calculateRelTupleScore(tuple: Seq[Any], relation: String) = {
    var theta = 0.0
    val scale = tupleNormalizer(tuple) * relationNormalizer(relation)
    for (relIndices <- relationComponentIndices.get(relation);
         tupleIndices <- tupleComponentIndices.get(tuple)) {
      for (c <- 0 until _numComponents) theta += scale * _weights(relIndices(c)) * _weights(tupleIndices(c))
    }
    theta
  }

  final def calculateScoreRaw(tuple: Seq[Any], relation: String, feat : Boolean = true) = {
    var theta = if (useBias && _weights != null) _weights(biasIndex) else 0.0
    val tupleScore = calculateRelTupleScore(tuple, relation)
    val featScore = if (feat) calculateFeatScore(tuple, relation) else 0.0
    val entScore = calculateArgEntScore(tuple, relation)
    theta += tupleScore
    theta += featScore
    theta += entScore
    theta
  }

  final def calculateProbRaw(tuple:Seq[Any], relation:String) = {
    calculateProb(calculateScoreRaw(tuple,relation))
  }

  final def calculateScore(cell: Cell, aux: Boolean = false): Double = {
    var theta = calculateScoreRaw(cell.tuple, cell.relation)
    if (aux) {
      for (indices <- lagrangeIndices.get(cell); coefficients <- lagrangeCoefficients.get(cell)) {
        var i = 0
        while (i < indices.size) {
          theta += _langrangeWeights(indices(i)) * coefficients(i)
          i += 1
        }
      }
    }
    theta
  }

  final def calculateScore(relation: String, tuple: scala.Seq[Any]): Double = {
    val cell = cells(relation -> tuple)
    calculateScore(cell)
  }

  private final def sq(num: Double) = num * num

  final def calculateCellGradientAndObjective(cell: Cell,
                                              gradient: Array[Double],
                                              weights: Array[Double],
                                              lagrangeWeights: Array[Double] = null,
                                              calculateGradient: Boolean = true,
                                              updateRelGradient: Boolean = true,
                                              updateTupleGradient: Boolean = true,
                                              updateArgGradient: Boolean = true,
                                              updateEntGradient: Boolean = true,
                                              updateFeatGradient: Boolean = true,
                                              updateBiasGradient: Boolean = true): Double = {
    import cell._

    var theta: Double = calculateCellScore(cell, weights)

    if (lagrangeWeights != null) {
      for (indices <- lagrangeIndices.get(cell); coefficients <- lagrangeCoefficients.get(cell)) {
        var i = 0
        while (i < indices.size) {
          theta += lagrangeWeights(indices(i)) * coefficients(i)
          i += 1
        }
      }
    }

    val mu = calculateProb(theta)
    for (relIndices <- relationComponentIndices.get(relation);
         tupleIndices <- tupleComponentIndices.get(tuple)) {
      var c = 0
      val feats = featureIndices.get(cell)
      val scale = tupleNormalizer(cell.tuple) * relationNormalizer(cell.relation)

      if (calculateGradient) {
        if (lagrangeWeights == null) {
          c = 0
          if (useBias && updateBiasGradient) gradient(biasIndex) += (target - mu)
          while (c < _numComponents) {
            val relIndex = relIndices(c)
            val relWeight = weights(relIndex)
            val tupleIndex = tupleIndices(c)
            val tupleWeight = weights(tupleIndex)
            if (updateRelGradient) gradient(relIndex) += scale * tupleWeight * (target - mu)
            if (updateTupleGradient) gradient(tupleIndex) += scale * relWeight * (target - mu)
            c += 1
          }
          if (_numArgComponents > 0) {
            for (arg <- 0 until arity(cell.relation)) {
              for (argIndices <- argComponentIndices.get(cell.relation -> arg);
                   entIndices <- entityComponentIndices.get(cell.tuple(arg))) {
                var ac = 0
                while (ac < _numArgComponents) {
                  val argIndex = argIndices(ac)
                  val entIndex = entIndices(ac)
                  val argWeight = weights(argIndex)
                  val entWeight = weights(entIndex)
                  if (updateArgGradient) gradient(argIndex) += entWeight * (target - mu)
                  if (updateEntGradient) gradient(entIndex) += argWeight * (target - mu)
                  ac += 1
                }
              }
            }
          }
          if (updateFeatGradient) for (featIndices <- feats) {
            for (i <- featIndices) {
              gradient(i) += (target - mu)
            }
          }
        }
        else {
          //calculate the gradient and objective for the aux distribution
          for (indices <- lagrangeIndices.get(cell); coefficients <- lagrangeCoefficients.get(cell)) {
            var i = 0
            while (i < indices.size) {
              val index = indices(i)
              gradient(index) -= mu * coefficients(i)
              i += 1
            }
          }

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


  final def calculateProb(theta: Double): Double = {
    1.0 / (1.0 + exp(-theta))
  }

  def calculateCellScore(cell: SelfPredictingDatabase.this.type#Cell, weights: Array[Double]): Double = {
    var theta = if (useBias) weights(biasIndex) else 0.0
    for (relIndices <- relationComponentIndices.get(cell.relation);
         tupleIndices <- tupleComponentIndices.get(cell.tuple)) {
      val feats = featureIndices.get(cell)
      val scale = tupleNormalizer(cell.tuple) * relationNormalizer(cell.relation)
      var c = 0
      while (c < _numComponents) {
        theta += scale * weights(relIndices(c)) * weights(tupleIndices(c))
        c += 1
      }

      for (featIndices <- feats) {
        for (i <- featIndices) {
          theta += weights(i)
        }
      }
      if (_numArgComponents > 0) {
        //todo: could optimize this code
        for (arg <- 0 until arity(cell.relation)) {
          for (argIndices <- argComponentIndices.get(cell.relation -> arg);
               entIndices <- entityComponentIndices.get(cell.tuple(arg))) {
            var ac = 0
            while (ac < _numArgComponents) {
              theta += weights(argIndices(ac)) * weights(entIndices(ac))
              ac += 1
            }
          }
        }
      }
    }
    theta
  }

  def arity(relation: String) = relation.substring(relation.length - 2) match {
    case "/1" => 1
    case _ => 2
  }

  def initializeWeights() {
    //initialize weights randomly
    _weights = Array.ofDim[Double](parameterCount)
    for (i <- _weights.indices) _weights(i) = random.nextDouble() / 100.0 //random.nextGaussian()/100.0 /// 10.0 //random.nextDouble()/100.0 , divided by 100 is learned from Ben Marlin, commented by Limin
    _langrangeWeights = Array.fill(lagrangeParameterCount)(0.0)
    weightsInitialized = true
  }

  def runLBFGS(tuples: Iterable[Seq[Any]] = tuples,
               updateRel: Boolean = true, updateTuple: Boolean = true,
               updateFeat: Boolean = true, updateBias: Boolean = true,
               normalizer: Option[Double] = None) {
    if (!indicesInitialized) initializeIndices()
    if (!weightsInitialized) initializeWeights()

    //create optimization problems
    val iProjectionObjective = new AuxObjective
    val mProjectionObjective = new LLObjective(tuples = tuples,
      updateRel = updateRel, updateTuple = updateTuple, updateFeat = updateFeat, updateBias = updateBias,
      normalizer = normalizer)

    //create optimizer
    val iProjector = new MyLimitedMemoryBFGS(iProjectionObjective) //ConjugateGradient(iProjectionObjective)//
    val mProjector = new MyLimitedMemoryBFGS(mProjectionObjective) //ConjugateGradient(mProjectionObjective) //new
    mProjector.maxIterations = maxIterations
    mProjector.tolerance = tolerance
    mProjector.gradientTolerance = gradientTolerance

    //initialize optimizer
    iProjectionObjective.setOptimizableParameters(_langrangeWeights)
    mProjectionObjective.setOptimizableParameters(_weights)

    //moment projection, for now, we are only doing mProjection, we should do infinite iterations
    def mProjection() {
      mProjector.optimize(maxIterations)
      _weights = mProjectionObjective.parameters
    }

    //information projection
    def iProjection() {
      if (lagrangeParameterCount > 0) {
        iProjector.optimize(20)
        _langrangeWeights = iProjectionObjective.parameters

        logger.info("Setting targets on unlabeled data.")
        for (cell <- unlabeledTraining) {
          cell.target = predictCell(cell, true)
        }
      }
    }

    //run AP
    for (iteration <- 0 until maxAPIterations) {
      logger.info("I-Projection %d".format(iteration))
      iProjection()

      logger.info("M-Projection %d".format(iteration))
      mProjection()

    }
  }


  def initializeIndices() {
    logger.info("Initializing Indices")
    def allocateIndices(howMany: Int) = {
      parameterCount += howMany
      Range(parameterCount - howMany, parameterCount).toArray
    }
    if (useBias) biasIndex = allocateIndices(1)(0)

    logger.info("Initializing Relation Indices")
    val relsToPredict = relations.filter(isToBePredicted).toSeq
    for (rel <- relsToPredict) {
      relationComponentIndices.getOrElseUpdate(rel, allocateIndices(numComponents))
      if (_numArgComponents > 0) for (arg <- 0 until arity(rel)) {
        argComponentIndices.getOrElseUpdate(rel -> arg, allocateIndices(_numArgComponents))
      }
    }
    logger.info("Initializing Tuple Indices")
    for (tuple <- tuples) {
      tupleComponentIndices.getOrElseUpdate(tuple, allocateIndices(numComponents))
    }
    if (_numArgComponents > 0) for (entity <- entities) {
      entityComponentIndices.getOrElseUpdate(entity, allocateIndices(_numArgComponents))
    }

    logger.info("Counting correlations")
    val cutoff = Conf.conf.getDouble("pca.corr-cutoff")
    for ((tuple,cells) <- tupleCells) {
      for (i <- 0 until cells.size; j <- i + 1 until cells.size) {
        val rel1 = cells(i).relation
        val rel2 = cells(j).relation
        val pair = rel1 -> rel2
        counts(pair) = counts.getOrElse(pair,0) + 1
      }
    }

    logger.info("Initializing Feature Indices")
    for (cell <- cells.values; if (cell.toPredict)) {
      val relTarget = cell.relation
      val features = featureCells.getOrElse(cell.tuple, Seq.empty).filter(_ != cell)
      featureIndices(cell) = features.view.filter(feat => counts.getOrElse(feat.relation -> relTarget, 0) >= cutoff).map(feat => {
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
      })

    //relPairIndicesInv ++= relPairIndices.map(_.swap)

    def allocateLagrangeIndex = {
      lagrangeParameterCount += 1
      lagrangeParameterCount - 1
    }

    //prepare lagrange variable indices for constraints
    for (constraint <- constraints) {
      val index = allocateLagrangeIndex
      constraint.lagrangeIndex = index
      constraint.localIndices = (for (cell <- constraint.cells) yield {
        val cellIndices = lagrangeIndices.getOrElse(cell, Array.empty[Int]) :+ index
        lagrangeIndices(cell) = cellIndices
        lagrangeCoefficients(cell) = Array.fill(cellIndices.size)(1.0)
        cellIndices.size - 1
      }).toArray
    }

    //map tuple/relation/ent/arg parents to weight index parents
    for ((child, parent) <- tupleParents) {
      for ((c, p) <- tupleComponentIndices(child) zip tupleComponentIndices(parent))
        parameterParents(c) = p
      //for (c <- 0 until numComponents)
    }
    for ((child, parent) <- entityParents) {
      for ((c, p) <- entityComponentIndices.getOrElse(child, Array.empty) zip entityComponentIndices.getOrElse(parent, Array.empty))
        parameterParents(c) = p
      //for (c <- 0 until numComponents)
    }


    //map tuple/relation/ent/arg lambdas to parameter lambdas
    for ((tuple, lambda) <- tupleLambdas) {
      for (c <- tupleComponentIndices(tuple)) parameterLambdas(c) = lambda
    }
    for ((ent, lambda) <- entityLambdas) {
      for (c <- entityComponentIndices.getOrElse(ent, Array.empty)) parameterLambdas(c) = lambda
    }

    logger.info("Num# Cells:               " + cells.size)
    logger.info("Num# Components:          " + numComponents)
    logger.info("Num# Arg Components:      " + _numArgComponents)
    logger.info("Num# Entities:            " + entities.size)
    logger.info("Num# Relations:           " + relationComponentIndices.size)
    logger.info("Num# Tuples:              " + tupleComponentIndices.size)
    logger.info("Num# Rel Pairs:           " + relPairIndices.size)
    logger.info("Num# Parameters:          " + parameterCount)
    logger.info("Num# Lagrange Parameters: " + lagrangeParameterCount)
    logger.info("Num# Parameter Parents:   " + parameterParents.size)
    logger.info("Num# Special Lambdas:     " + parameterLambdas.size)
    logger.info("Num# training cells:      " + labeledTraining.size)

    indicesInitialized = true
  }

  //add bias for each tuple, each tuple has a bias component, specifically, for each tuple, we add a cell ('bias', tuple, hidden = false, feature = true, toPredict=false)
  //each relation except bias has a new weight, as explained by Sebastian weight(r',bias)
  def addBias() {
    for (tuple <- tuples) {
      addCell("bias", tuple, 1.0, hide = false, feature = true, predict = false)
    }
  }

  def projectLangrangeWeights(weights: Array[Double]) {
    for (c <- constraints; if (c.lagrangeBounded))
      weights(c.lagrangeIndex) = max(0.0, weights(c.lagrangeIndex))

  }

  class AuxObjective extends Objective {
    def domainSize = lagrangeParameterCount

    val gradient = Array.ofDim[Double](lagrangeParameterCount)

    //do we need to initialize the gradients to 0.0 for each iteration? todo: asked by Limin
    def calculateGradientAndObjective(parameters: Array[Double]) = {
      var obj = 0.0
      //project parameters here?
      projectLangrangeWeights(parameters)
      //iterate over unlabeled cells
      for (cell <- unlabeledTraining) {
        obj += calculateCellGradientAndObjective(cell, gradient, _weights, _langrangeWeights)
      }
      //add constraint-specific regularizers and targets
      for (c <- constraints) {
        val weight = parameters(c.lagrangeIndex)
        if (c.regularizer > 0.0) {
          if (c.l1) {
            obj -= c.regularizer * 0.5 * abs(weight)
          } else {
            obj -= c.regularizer * 0.5 * (weight * weight)
          }
        }
        obj += weight * c.target
        var localGradient = c.target
        if (c.regularizer > 0.0) {
          if (c.l1) localGradient -= c.regularizer * (if (weight > 0) 1.0 else -1.0)
          else localGradient -= c.regularizer * weight
        }
        gradient(c.lagrangeIndex) += localGradient
      }
      GradientAndObjective(gradient, obj)
    }
  }

  class LLObjective(updateRel: Boolean = true,
                    updateTuple: Boolean = true,
                    updateArg: Boolean = true,
                    updateEnt: Boolean = true,
                    updateFeat: Boolean = true,
                    updateBias: Boolean = true,
                    normalize: Boolean = true,
                    tuples: Iterable[Seq[Any]] = tuples,
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
        for (tuple <- tuples; cell <- getCells(tuple).filter(_.labeledTrainingData)) {
          obj += calculateCellGradientAndObjective(cell, gradient, parameters,
            updateRelGradient = updateRel, updateTupleGradient = updateTuple,
            updateArgGradient = updateArg, updateEntGradient = updateEnt,
            updateFeatGradient = updateFeat, updateBiasGradient = updateBias
          )
        }
        for (tuple <- tuples; cell <- getCells(tuple).filter(_.unlabeledTrainingData)) {
          obj += calculateCellGradientAndObjective(cell, gradient, parameters,
            updateRelGradient = updateRel, updateTupleGradient = updateTuple,
            updateArgGradient = updateArg, updateEntGradient = updateEnt,
            updateFeatGradient = updateFeat, updateBiasGradient = updateBias)
        }
        obj
      }
      if (cores == 1) {
        obj += addCellGradientsAndObjective(gradient)
      }
      else {
        logger.info("Calculating gradients on %d cores".format(cores))
        for (gradient <- gradientPerCore) util.Arrays.fill(gradient, 0.0)
        val grouped = tuples.view.grouped(tuples.size / (cores - 1)).toSeq.par
        val objectives = for ((chunk, index) <- grouped.zipWithIndex) yield {
          addCellGradientsAndObjective(gradientPerCore(index))
        }
        obj += objectives.sum
        logger.info("Summing up gradients".format(cores))
        for (core <- 0 until cores) {
          val coreGradient = gradientPerCore(core)
          for (index <- 0 until parameterCount)
            gradient(index) += coreGradient(index)
        }
        //parallel
      }

      logger.info("Likelihood:                   " + obj) //added by Limin Yao, so that we log it in log.txt

      //add GE terms
      for (ge <- geTerms) {
        obj += ge.calculateGradientAndObj(parameters, gradient)
      }

      //add relation regularizer
      for ((_, relComponents) <- relationComponentIndices) {
        var c = 0
        while (c < numComponents) {
          val index = relComponents(c)
          val lambda = parameterLambdas.getOrElse(index, lambdaRel)
          val mean = parameterParents.get(index).map(parameters).getOrElse(0.0)
          val weight = parameters(index)
          obj -= lambda / 2.0 * sq(weight - mean)
          if (updateRel) gradient(index) -= lambda * (weight - mean)
          for (parentIndex <- parameterParents.get(index))
            gradient(parentIndex) += lambda * (weight - mean)
          c += 1
        }
      }
      //add tuple regularizer
      for ((_, tupleComponents) <- tupleComponentIndices) {
        var c = 0
        while (c < numComponents) {
          val index = tupleComponents(c)
          val lambda = parameterLambdas.getOrElse(index, lambdaTuple)
          val mean = parameterParents.get(index).map(parameters).getOrElse(0.0)
          val weight = parameters(index)
          obj -= lambda / 2.0 * sq(weight - mean)
          if (updateTuple) gradient(index) -= lambda * (weight - mean)
          for (parentIndex <- parameterParents.get(index))
            gradient(parentIndex) += lambda * (weight - mean)
          c += 1
        }
      }
      //add arg regularizer
      for ((_, argComponents) <- argComponentIndices) {
        var c = 0
        while (c < _numArgComponents) {
          val index = argComponents(c)
          val weight = parameters(index)
          val lambda = parameterLambdas.getOrElse(index, lambdaArg)
          val mean = parameterParents.get(index).map(parameters).getOrElse(0.0)
          obj -= lambda / 2.0 * sq(weight - mean)
          if (updateArg) gradient(index) -= lambda * (weight - mean)
          for (parentIndex <- parameterParents.get(index))
            gradient(parentIndex) += lambda * (weight - mean)
          c += 1
        }
      }

      //add ent regularizer
      for ((_, entComponents) <- entityComponentIndices) {
        var c = 0
        while (c < _numArgComponents) {
          val index = entComponents(c)
          val weight = parameters(index)
          val lambda = parameterLambdas.getOrElse(index, lambdaEntity)
          val mean = parameterParents.get(index).map(parameters).getOrElse(0.0)
          obj -= lambda / 2.0 * sq(weight - mean)
          if (updateEnt) gradient(index) -= lambda * (weight - mean)
          for (parentIndex <- parameterParents.get(index))
            gradient(parentIndex) += lambda * (weight - mean)
          c += 1
        }
      }

      //add rel pair regularizer
      for ((pair, index) <- relPairIndices) {
        val weight = parameters(index)
        obj -= lambdaRelPair / 2.0 * sq(weight)
        if (updateFeat) gradient(index) -= lambdaRelPair * weight
      }

      if (useBias) {
        val biasWeight = parameters(biasIndex)
        obj -= lambdaBias / 2.0 * sq(biasWeight)
        if (updateBias) gradient(biasIndex) -= lambdaBias * biasWeight
      }
      logger.info("Likelihood after regularizer: " + obj) //added by Limin Yao, so that we log it in log.txt
      //divide the obj and gradient by the number of tuples, except bias, by Limin, learned from Ben Marlin
      val tupleSize = normalizer.getOrElse(tuples.size.toDouble)
      if (normalize) obj = obj / tupleSize
      var norm = 0.0
      for (i <- gradient.indices) {
        if (normalize) gradient(i) = gradient(i) / tupleSize
        norm += gradient(i) * gradient(i)
      }

      logger.info("Gradient Norm:                " + sqrt(norm))
      logger.info("Likelihood after divided by tuplesize: " + obj) //added by Limin Yao, so that we log it in log.txt
      //result
      GradientAndObjective(gradient, obj)
    }

    def domainSize = parameterCount
  }

  // update weights with scale, and return regularization term
  def updateOnCell(relation: String, tuple: Seq[Any], scale: Double, lr: Double, weights: Array[Double],
                   lambdaBias: Double = lambdaBias,
                   lambdaRel: Double = lambdaRel, lambdaTuple: Double = lambdaTuple,
                   lambdaArg: Double = lambdaArg, lambdaEnt: Double = lambdaEntity,
                   lambdaFeat: Double = lambdaRelPair) = {

    val updateBiasGradient = true
    val updateRelGradient = true
    val updateTupleGradient = true
    val updateArgGradient = true
    val updateEntGradient = true
    val updateFeatGradient = true

    var regObj = 0.0
    if (useBias && updateBiasGradient) {
      val biasWeight = weights(biasIndex)
      weights(biasIndex) += lr * (scale - lambdaBias * biasWeight)
      regObj -= (lambdaBias / 2.0 * sq(biasWeight))
    }
    if (_numComponents > 0) {
      val count = 1.0 //tupleCells(tuple).size
      for (relIndices <- relationComponentIndices.get(relation);
           tupleIndices <- tupleComponentIndices.get(tuple)) {
        var c = 0
        while (c < _numComponents) {
          val relIndex = relIndices(c)
          val relWeight = weights(relIndex)
          val tupleIndex = tupleIndices(c)
          val tupleWeight = weights(tupleIndex)
//          touched(relIndex) = touched.getOrElse(relIndex, 0) + 1
//          touched(tupleIndex) = touched.getOrElse(tupleIndex, 0) + 1
//          val touchedRel= touched(relIndex)
//          val touchedTuple = touched(tupleIndex)
          if (updateRelGradient) weights(relIndex) += lr * (scale * tupleWeight - lambdaRel * relWeight)
          if (updateTupleGradient) weights(tupleIndex) += lr * (scale * relWeight - count * lambdaTuple * tupleWeight)
          if (updateRelGradient) regObj -= (lambdaRel / 2.0 * sq(relWeight))
          if (updateTupleGradient) regObj -= (count * lambdaTuple / 2.0 * sq(tupleWeight))
          c += 1
        }
      }
    }
    if (_numArgComponents > 0) {
      for (arg <- 0 until 1) {
        //
        //        for (arg <- 0 until arity(relation)) {
        val entity = tuple(arg)
        val count = 1.0 //entity2Cells(entity).size
        for (argIndices <- argComponentIndices.get(relation -> arg);
             entIndices <- entityComponentIndices.get(entity)) {
          var ac = 0
          while (ac < _numArgComponents) {
            val argIndex = argIndices(ac)
            val entIndex = entIndices(ac)
            val argWeight = weights(argIndex)
            val entWeight = weights(entIndex)
//            touched(argIndex) = touched.getOrElse(argIndex, 0) + 1
//            touched(entIndex) = touched.getOrElse(entIndex, 0) + 1
//            val touchedArg = touched(argIndex)
//            val touchedEnt = touched(entIndex)
//            val entityCount = entity2Cells.get(entity).map(_.size).getOrElse(0)


            if (updateArgGradient) weights(argIndex) += lr * (entWeight * scale - lambdaArg * argWeight)
            if (updateEntGradient) weights(entIndex) += lr * (argWeight * scale - count * lambdaEnt * entWeight)
            if (updateArgGradient) regObj -= (lambdaArg / 2.0 * sq(argWeight))
            if (updateEntGradient) regObj -= (count * lambdaEnt / 2.0 * sq(entWeight))
            ac += 1
          }
        }
      }
    }
    if (updateFeatGradient) {
      for (i <- featureIndicesFor(tuple, relation)) {
//        val pair = relPairIndicesInv(i)
        val featWeight = weights(i)
        weights(i) += lr * (scale - lambdaFeat * featWeight)
        val after = weights(i)
        regObj -= (lambdaFeat / 2.0 * sq(featWeight))
      }
    }
    regObj

  }


  /**
   * Runs SGD on the gPCA.
   * @param tuples tuples to consider
   * @param relations relations to consider
   */
  def runSGD(tuples: Iterable[Seq[Any]] = tuples, relations: Iterable[String] = relations) {
    if (!indicesInitialized) initializeIndices()
    if (!weightsInitialized) initializeWeights()

    val allowed = relations.toSet
    Logger.info("Preparing training data")
    val cellsInOrder = tuples.toSeq.flatMap(getCells(_).view.filter(c => c.labeledTrainingData && allowed(c.relation)))
    val trainingCells = Random.shuffle(cellsInOrder).toArray
    val relationArray = relations.filter(isToBePredicted).toArray
    val tupleArray = tuples.toArray

    var learningRate = 0.05
    val randomSamples = 0 //trainingCells.size * 1
    val negWeight = 0.2
    ProgressMonitor.start("SGD", "SGD", trainingCells.size + randomSamples, maxIterations)
    //gpca style
    for (epoch <- 0 until maxIterations) {
      //do updates
      var count = 0
      var total = 0.0

      def pcaUpdate(tuple: Seq[Any], relation: String, target: Double, weight: Double = 1.0) {
        val score = calculateScoreRaw(tuple, relation)
        val prob = calculateProb(score)
        val scale = (target - prob) * weight
        val oldReg = updateOnCell(relation, tuple, scale, learningRate, _weights)
        val oldLL = target * score - log1p(exp(score))
        val oldObj = oldLL + oldReg
        total += oldObj
        count += 1
        ProgressMonitor.progress("SGD", 1, "%-14.6f".format(total / count))
      }

      //observed cells
      for (cell <- trainingCells) {
        //get objective and gradient from cell
        pcaUpdate(cell.tuple, cell.relation, cell.target)
      }

      //sample random cells
      for (i <- 0 until randomSamples) {
        val relation = relationArray(random.nextInt(relationArray.size))
        val tuple = tupleArray(random.nextInt(tupleArray.size))
        val target = getCell(relation, tuple).map(_.target).getOrElse(0.0)
        pcaUpdate(tuple, relation, target, negWeight)
      }
    }
  }

  /**
   * Runs Bayesian Personalized Ranking (pairwise ranking over tuples with same relation) using SGD.
   * @param tuples tuples to consider
   * @param relations relations to consider
   */
  def runBPR(tuples: Iterable[Seq[Any]] = tuples, relations: Iterable[String] = relations) {
    if (!indicesInitialized) initializeIndices()
    if (!weightsInitialized) initializeWeights()

    val allowed = relations.toSet
    Logger.info("Preparing training data")
    val cellsInOrder = tuples.toSeq.flatMap(getCells(_).view.filter(c => c.labeledTrainingData && allowed(c.relation)))
    val allTrainingCells = cellsInOrder.toArray
    //    val trainingCells = cellsInOrder.filter(_.target > 0.5).toArray
    val relationArray = relations.filter(isToBePredicted).toArray
    val tupleArray = tuples.toArray
    val lambdaNegRel = lambdaRel
    val lambdaNegTuple = lambdaTuple
    val rel2Positive = allTrainingCells.filter(_.target > 0.5).groupBy(_.relation)
    val restrictMostLikelyTuples = false
    val atMostOneTuplePerArg = Conf.conf.getBoolean("pca.at-most-one-entity")

    ProgressMonitor.start("BPR", "BPR", allTrainingCells.size, maxIterations)
    val tupleMap = new mutable.HashMap[String, Array[Seq[Any]]]()


    def createEntityFilteredCells() = {
      val groupedByArg1 = allTrainingCells.groupBy(cell => cell.tuple(0))
      val filteredArg1 = groupedByArg1.map(pair => pair._2(random.nextInt(pair._2.size))).toArray
      val groupedByArg2 = filteredArg1.groupBy(cell => cell.tuple(1))
      val result = groupedByArg2.map(pair => pair._2(random.nextInt(pair._2.size)))
//      entity2Cells.clear()
//      for (cell <- result; ent <- cell.tuple)
//        entity2Cells.getOrElseUpdate(ent, new ArrayBuffer[Cell]) += cell
      result.toSet.toArray
    }

    def createEntityFilteredTuples() = {
      val filteredTuplesArg1 = tupleArray.groupBy(_.apply(0)).map(pair => pair._2(random.nextInt(pair._2.size))).toArray
      val filteredTuplesArg2 = filteredTuplesArg1.groupBy(_.apply(1)).map(pair => pair._2(random.nextInt(pair._2.size))).toArray
      filteredTuplesArg2
    }

    for (i <- 0 until maxIterations) {
      var total = 0.0
      var count = 0
      val trainingCells = if (!atMostOneTuplePerArg) allTrainingCells else createEntityFilteredCells()
      val trainingTuples = if (!atMostOneTuplePerArg) tupleArray else createEntityFilteredTuples()
      ProgressMonitor.setMaxSteps("BPR", trainingCells.size)
      if (restrictMostLikelyTuples && i % 10 == 0) {
        ProgressMonitor.start("Sort", "Sort", relationArray.size, 1)
        for (relation <- relationArray) {
          if (relation.startsWith("REL$")) {
            val unobserved = tuples.filter(t => getCell(relation, t).filter(_.target > 0.5).isEmpty)
            val scoredTuples = unobserved.map(t => t -> calculateProb(calculateScoreRaw(t, relation))).toIndexedSeq
            val sorted = scoredTuples.sortBy(-_._2).take(100).map(_._1).toArray
            tupleMap(relation) = sorted
          } else
            tupleMap(relation) = tupleArray
          ProgressMonitor.progress("Sort", 1)
        }
      }
      def rel2Tuples(relation: String) = restrictMostLikelyTuples match {
        case true => tupleMap(relation)
        case false => trainingTuples // tupleArray
      }

      def bprUpdate(tuple1: Seq[Any], tuple2: Seq[Any], relation: String, target: Double, weight: Double = 1.0) {
        val score1 = calculateScoreRaw(tuple1, relation)
        val score2 = calculateScoreRaw(tuple2, relation)
        val score = score1 - score2
        val prob = calculateProb(score)
        val scale = (target - prob) * weight
        //both updates change the relation components. This affects the oldReg term computation. We avoid this by
        //skipping relation reg. in the second step.
        val oldReg1 = updateOnCell(relation, tuple1, scale, learningRate, _weights)
        val oldReg2 = updateOnCell(relation, tuple2, -scale, learningRate, _weights, lambdaRel = 0.0, lambdaArg = 0.0)
        val oldReg = oldReg1 + oldReg2
        val oldLL = target * score - log1p(exp(score))
        val oldObj = oldLL + oldReg
        total += oldObj
        count += 1
        ProgressMonitor.progress("BPR", 1, "%-14.6f %1.1f %3.4f".format(total / count, target, score))
      }

      for (i <- 0 until trainingCells.size) {

        //sample observed cell (tuple,relation)
        val cell = trainingCells(random.nextInt(trainingCells.size))

        //sample unobserved cell with same relation (tuple',relation)
        var tuple2 = cell.tuple
        val candidateTuples = rel2Tuples(cell.relation)
        while (getCell(cell.relation, tuple2).isDefined) {
          tuple2 = candidateTuples(random.nextInt(candidateTuples.size))
        }
        if (cell.target > 0.5)
          bprUpdate(cell.tuple, tuple2, cell.relation, 1.0)
        else
          bprUpdate(tuple2, cell.tuple, cell.relation, 1.0)

      }


    }


  }

  def runBPRAll(tuples: Iterable[Seq[Any]] = tuples, relations: Iterable[String] = relations, entityCentric: Boolean = false) {
    if (!indicesInitialized) initializeIndices()
    if (!weightsInitialized) initializeWeights()

    val allowed = relations.toSet
    Logger.info("Preparing training data")
    val cellsInOrder = tuples.toSeq.flatMap(getCells(_).view.filter(c => c.labeledTrainingData && allowed(c.relation)))
    val trainingCells = cellsInOrder.toArray
    val relationArray = relations.filter(isToBePredicted).toArray
    val tupleArray = tuples.toArray
    val lambdaNegRel = lambdaRel
    val lambdaNegTuple = lambdaTuple
    val ent2CellArg0 = tuples.groupBy(_.apply(0)).mapValues(_.toArray)
    val ent2CellArg1 = tuples.groupBy(_.apply(1)).mapValues(_.toArray)


    ProgressMonitor.start("BPR", "BPR", trainingCells.size, maxIterations)

    for (i <- 0 until maxIterations) {
      var total = 0.0
      var count = 0

      def bprUpdate(tuple1: Seq[Any], tuple2: Seq[Any], relation1: String, relation2: String, target: Double, weight: Double = 1.0) {
        val score1 = calculateScoreRaw(tuple1, relation1)
        val score2 = calculateScoreRaw(tuple2, relation2)
        val score = score1 - score2
        val prob = calculateProb(score)
        val scale = (target - prob) * weight
        //both updates change the relation components. This affects the oldReg term computation. We avoid this by
        //skipping relation reg. in the second step.
        val oldReg1 = updateOnCell(relation1, tuple1, scale, learningRate, _weights)
        val oldReg2 = updateOnCell(relation2, tuple2, -scale, learningRate, _weights)
        val oldReg = oldReg1 + oldReg2
        val oldLL = target * score - log1p(exp(score))
        val oldObj = oldLL + oldReg
        total += oldObj
        count += 1
        ProgressMonitor.progress("BPR", 1, "%-14.6f".format(total / count))
      }

      for (i <- 0 until trainingCells.size) {

        //sample observed cell (tuple,relation)
        val cell = trainingCells(random.nextInt(trainingCells.size))

        //sample unobserved cell with same relation (tuple',relation)
        var tuple2 = cell.tuple
        var relation2 = cell.relation
        val candidateTuples = if (entityCentric) {
          if (random.nextBoolean()) ent2CellArg0(cell.tuple(0)) else ent2CellArg1(cell.tuple(1))
        } else tupleArray
        while (getCell(relation2, tuple2).isDefined) {
          tuple2 = candidateTuples(random.nextInt(candidateTuples.size))
          relation2 = relationArray(random.nextInt(relationArray.size))
        }

        //do the update
        //if the training cell is positive it should be ranked higher than the unobserved cell
        //if it is negative, it should be ranked lower
        if (cell.target > 0.5)
          bprUpdate(cell.tuple, tuple2, cell.relation, relation2, 1.0)
        else
          bprUpdate(tuple2, cell.tuple, relation2, cell.relation, 1.0)

      }


    }


  }

  def printRankedTuples(rel: String, out: PrintStream) {
    val scoredTuples = tuples.map(t => t -> calculateProb(calculateScoreRaw(t, rel))).toIndexedSeq
    val sorted = scoredTuples.sortBy(_._2)
    val withLastRelevant = sorted.dropWhile(pair => getCell(rel, pair._1).filter(_.target > 0.5).isEmpty)
    var map = 0.0
    var total = 0
    var relevant = 0
    var totalForMap = 0

    for ((tuple, prob) <- sorted.reverse) {
      val isRelevant = getCell(rel, tuple).filter(c => !c.hide && c.target > 0.5).isDefined
      if (isRelevant) relevant += 1
      total += 1
      val prec = relevant.toDouble / total
      if (total < withLastRelevant.size) {
        totalForMap += 1
        map += prec
      }
      val observed = getCells(tuple).filter(_.labeledTrainingData).map(_.relation)
      out.println("%-6d %7.3f %7.3f %4.1f %s: %s".format(total, prob, prec, if (isRelevant) 1.0 else 0.0, tuple.mkString(" | "), observed.mkString(" , ")))
    }
    map = map / totalForMap
    out.println("MAP: " + map)
    //    for (cell <- sorted) {
    //      out.println("%7.3f\t%s".format(cell.predicted, cell.tuple.mkString("\t")))
    //    }

  }

  /**
   * Runs Bayesian Personalized Ranking (pairwise ranking over relations with same tuple) using SGD.
   * @param tuples tuples to consider
   * @param relations relations to consider
   */
  def runBPRInv(tuples: Iterable[Seq[Any]] = tuples, relations: Iterable[String] = relations) {
    if (!indicesInitialized) initializeIndices()
    if (!weightsInitialized) initializeWeights()

    val allowed = relations.toSet
    Logger.info("Preparing training data")
    val cellsInOrder = tuples.toSeq.flatMap(getCells(_).view.filter(c => c.labeledTrainingData && allowed(c.relation)))
    val trainingCells = cellsInOrder.toArray
    val relationArray = relations.filter(isToBePredicted).toArray
    val tupleArray = tuples.toArray
    val lambdaNegRel = lambdaRel
    val lambdaNegTuple = lambdaTuple
    val learningRate = 0.05

    ProgressMonitor.start("BPR", "BPR-Inv", trainingCells.size, maxIterations)

    for (i <- 0 until maxIterations) {
      var total = 0.0
      var count = 0

      def bprUpdate(tuple: Seq[Any], relation1: String, relation2: String, target: Double, weight: Double = 1.0) {
        val score1 = calculateScoreRaw(tuple, relation1)
        val score2 = calculateScoreRaw(tuple, relation2)
        val score = score1 - score2
        val prob = calculateProb(score)
        val scale = (target - prob) * weight
        //both updates change the relation components. This affects the oldReg term computation. We avoid this by
        //skipping relation reg. in the second step.
        val oldReg1 = updateOnCell(relation1, tuple, scale, learningRate, _weights)
        val oldReg2 = updateOnCell(relation2, tuple, -scale, learningRate, _weights, lambdaTuple = 0.0, lambdaEnt = 0.0, lambdaFeat = 0.0)
        val oldReg = oldReg1 + oldReg2
        val oldLL = target * score - log1p(exp(score))
        val oldObj = oldLL + oldReg
        total += oldObj
        count += 1
        ProgressMonitor.progress("BPR", 1, "%-14.6f".format(total / count))
      }

      for (i <- 0 until trainingCells.size) {

        //sample observed cell (tuple,relation)
        val cell = trainingCells(random.nextInt(trainingCells.size))

        //sample unobserved cell with same tuple (tuple',relation)
        var relation2 = cell.relation
        while (getCell(relation2, cell.tuple).isDefined) {
          relation2 = relationArray(random.nextInt(relationArray.size))
        }

        //do the update
        //if the training cell is positive it should be ranked higher than the unobserved cell
        //if it is negative, it should be ranked lower
        if (cell.target > 0.5)
          bprUpdate(cell.tuple, cell.relation, relation2, 1.0)
        else
          bprUpdate(cell.tuple, relation2, cell.relation, 1.0)

      }


    }


  }

  //run bpr, for each tuple, rank two relations , also consider negative observations
  def runBPRByTuple(tuples: Iterable[Seq[Any]] = tuples, relations: Iterable[String] = relations) {
    if (!indicesInitialized) initializeIndices()
    if (!weightsInitialized) initializeWeights()

    val allowed = relations.toSet
    Logger.info("Preparing training data")
    val cellsInOrder = tuples.toSeq.flatMap(getCells(_).view.filter(c => c.labeledTrainingData && allowed(c.relation)))
    val trainingCells = cellsInOrder.toArray
    val relationArray = relations.filter(isToBePredicted).toArray
    val tupleArray = tuples.toArray
    val lambdaNegRel = lambdaRel
    val lambdaNegTuple = lambdaTuple
    val learningRate = 0.05

    ProgressMonitor.start("BPR", "BPR", trainingCells.size, maxIterations)
    for (i <- 0 until maxIterations) {
      var total = 0.0
      var count = 0

      def pcaUpdate(tuple: Seq[Any], relation: String, target: Double, weight: Double = 1.0) {
        val score = calculateScoreRaw(tuple, relation)
        val prob = calculateProb(score)
        val scale = (target - prob) * weight
        val oldReg = updateOnCell(relation, tuple, scale, learningRate, _weights)
        val oldLL = target * score - log1p(exp(score))
        val oldObj = oldLL + oldReg
        total += oldObj
        count += 1
        ProgressMonitor.progress("BPR", 1, "%-14.6f".format(total / count))
      }

      def bprUpdate(tuple: Seq[Any], relation1: String, relation2: String, target: Double, weight: Double = 1.0) {
        var score1 = calculateScoreRaw(tuple, relation1)
        var score2 = calculateScoreRaw(tuple, relation2)
        val score = score1 - score2
        val prob = calculateProb(score)
        val scale = (target - prob) * weight
        
        //logger.info("Score1: " + score1 + ", score2: " + score2 + ", prob for ranking:" + prob)
        
        val oldReg1 = updateOnCell(relation1, tuple, scale, learningRate, _weights)
        val oldReg2 = updateOnCell(relation2, tuple, -scale, learningRate, _weights, lambdaTuple = 0.0, lambdaEnt = 0.0, lambdaFeat = 0.0)
        val oldReg = oldReg1 + oldReg2
        val oldLL = target * score - log1p(exp(score))
        val oldObj = oldLL + oldReg
        total += oldObj
        count += 1
        ProgressMonitor.progress("BPR", 1, "%-14.6f".format(total / count))
      }
      

      for (i <- 0 until trainingCells.size) {

        //sample observed cell (tuple,relation)
        val cell = trainingCells(random.nextInt(trainingCells.size))

        if(random.nextBoolean()) pcaUpdate(cell.tuple,cell.relation,1.0)  //blend of maximum likelihood and ranking
        else{
        //sample unobserved cell with same tuple (tuple',relation)
        var relation2 = cell.relation
        if(cell.target > 0.5){  //positive cell
          while (getCell(relation2, cell.tuple).isDefined) {
            relation2 = relationArray(random.nextInt(relationArray.size))
          }
          bprUpdate(cell.tuple, cell.relation, relation2, 1.0)
        }          
        else { //negative cell, find out all positive observations, rank them higher than this negative cell
          val poscells4tuple = getCells(cell.tuple).filter(c => c.labeledTrainingData && allowed(c.relation) & c.target>0.5) //these are all positive observations for the tuple of the current cell
          if (poscells4tuple.size == 0) logger.info("No positive obs for tuple " + cell.tuple.mkString("|"))
          relation2 = poscells4tuple(random.nextInt(poscells4tuple.size)).relation
          bprUpdate(cell.tuple, relation2, cell.relation, 1.0)
//          for (poscell <- poscells4tuple)
//            bprUpdate(cell.tuple, poscell.relation, cell.relation, 1.0)
        }
        }
      }
    }

  }

  //this is still mapping a tuple to low dimension, and a relation to a low dimension, not entity to low dimension and rel to low dimension
  //run bpr, for each tuple, rank two relations , for arg1-rel, rank arg1-rel-arg2 above unobserved arg1-rel-arg2', for rel-arg2, rank arg1-rel-arg2 above unobserved arg1'-rel-arg2
  def runBPRTensor(tuples: Iterable[Seq[Any]] = tuples, relations: Iterable[String] = relations) {
    if (!indicesInitialized) initializeIndices()
    if (!weightsInitialized) initializeWeights()

    val allowed = relations.toSet
    Logger.info("Preparing training data")
    val cellsInOrder = tuples.toSeq.flatMap(getCells(_).view.filter(c => c.labeledTrainingData && allowed(c.relation)))
    val trainingCells = cellsInOrder.toArray
    val relationArray = relations.filter(isToBePredicted).toArray
    val entitiesArray = entities.toArray
    val tupleArray = tuples.toArray
    val lambdaNegRel = lambdaRel
    val lambdaNegTuple = lambdaTuple
    val learningRate = 0.05

    logger.info("Tensor: Starting ranking based training!")
    for (iter <- 0 until maxIterations) {
      var total = 0.0
      var count = 0

      def bprUpdate(tuple1: Seq[Any], relation1: String, tuple2: Seq[Any], relation2: String, target: Double, weight: Double = 1.0) {
        val score1 = calculateScoreRaw(tuple1, relation1)
        val score2 = calculateScoreRaw(tuple2, relation2)
        val score = score1 - score2
        val prob = calculateProb(score)
        val scale = (target - prob) * weight
        //both updates change the relation components. This affects the oldReg term computation. We avoid this by
        //skipping relation reg. in the second step.
        val oldReg1 = updateOnCell(relation1, tuple1, scale, learningRate, _weights)
        val oldReg2 = updateOnCell(relation2, tuple2, -scale, learningRate, _weights)
        val oldReg = oldReg1 + oldReg2
        val oldLL = target * score - log1p(exp(score))
        val oldObj = oldLL + oldReg
        total += oldObj
        count += 1
        logger.info("BPR Tensor" + "%-14.6f".format(total / count))
      }

      for (i <- 0 until trainingCells.size) {

        //sample observed cell (tuple,relation)
        val cell = trainingCells(random.nextInt(trainingCells.size))
        
        if (cell.target > 0.5)  {     //negative should be dealt with separately, negative should be lower than positive, not than unobserved cell

          //sample unobserved cell with same tuple (tuple,relation2)
          var relation2 = cell.relation
          while (getCell(relation2, cell.tuple).isDefined) {
            relation2 = relationArray(random.nextInt(relationArray.size))
          }
        

          bprUpdate(cell.tuple, cell.relation, cell.tuple, relation2, 1.0)


          //for arg1-rel, rank arg1-rel-arg2 above unobserved arg1-rel-arg2'
          var arg1 = cell.tuple(0)
          var arg2 = cell.tuple(1)
          var tuple2 = cell.tuple
          while(getCell(cell.relation, tuple2).isDefined) {
            arg2 = entitiesArray(random.nextInt(entitiesArray.size))
            tuple2 = Seq(arg1,arg2)
          }


          bprUpdate(cell.tuple, cell.relation, tuple2, cell.relation, 1.0)
        
          arg2 = cell.tuple(1)  //re-initialize arg2
          tuple2 = cell.tuple
          // rel-arg2, rank arg1-rel-arg2 above unobserved arg1'-rel-arg2
          while(getCell(cell.relation, tuple2).isDefined) {
            arg1 = entitiesArray(random.nextInt(entitiesArray.size))
            tuple2 = Seq(arg1,arg2)
          }

          bprUpdate(cell.tuple, cell.relation, tuple2, cell.relation, 1.0)

        }
        
        logger.info("Processed " + i*1.0/trainingCells.size + " cells in iteration " + iter)
      }
      
      logger.info("Finished Iteration:" + iter)
    }

  }

  def loadCells(source: Source,
                toPredict: (Seq[Any], String) => Boolean,
                isFeature: (Seq[Any], String) => Boolean,
                toHide: (Seq[Any], String) => Boolean) {
    def target(s: String): Option[Double] = try {
      val targetStartIndex = s.indexWhere(_ == '@')
      Some(s.substring(targetStartIndex).toDouble)
    } catch {
      case _: Exception => None
    }

    for (line <- source.getLines()) {
      val fields = line.split("\t")
      val tuple = fields(0).split("\\|")
      val relations = fields.drop(1)
      for (relation <- relations) {
        addCell(relation, tuple,
          value = target(relation).getOrElse(1.0),
          hide = toHide(tuple, relation),
          predict = toPredict(tuple, relation),
          feature = isFeature(tuple, relation))
      }
    }

  }

  def printOutRankedList(tuples: Iterable[Seq[Any]], relFilter: String => Boolean, out: PrintStream,
                         printAnalysis: Boolean = true) {
    //get all hidden cells matching tuples and relation filter.
    val cells = tuples.flatMap(getCells(_).filter(c => c.testData && relFilter(c.relation))).toSeq
    cells.foreach(_.update())
    //rank down tuples with no observed cells
    for (cell <- cells; if (!getCells(cell.tuple).exists(_.labeledTrainingData))) cell.predicted = 0.0

    val sorted = cells.sortBy(-_.predicted)
    logger.info("Preparing sorted tuples and relations")

    val componentAnalysis = false

    def onlyIfAnalysis[T](t: =>T):T = if (componentAnalysis) t else null.asInstanceOf[T]

    val sortedTupleComponents = onlyIfAnalysis(for (c <- 0 until numComponents) yield {
      tupleComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
    })
    val sortedEntComponents = onlyIfAnalysis(for (c <- 0 until numArgComponents) yield {
      entityComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
    })
    val sortedRelationComponents = onlyIfAnalysis(for (c <- 0 until numComponents) yield {
      relationComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
    })
    val sortedArgComponents = onlyIfAnalysis(for (c <- 0 until numArgComponents) yield {
      argComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
    })
    val relMaps = onlyIfAnalysis(sortedRelationComponents.map(_.toMap))
    val tupleMaps = onlyIfAnalysis(sortedTupleComponents.map(_.toMap))
//    val entMaps = sortedEntComponents.map(_.toMap)
//    val argMaps = sortedArgComponents.map(_.toMap)

    logger.info("Preparing sorted columns")
    val sortedColumns = onlyIfAnalysis((for (relation <- relations; if (isToBePredicted(relation))) yield {
      val cellsInColumn = getCells(relation)
      val obsTuples = cellsInColumn.filter(_.labeledTrainingData).filter(_.target > 0.0).map(_.tuple)
      val sortedObsTuples = for (c <- 0 until numComponents) yield {
        obsTuples.map(t => t -> tupleMaps(c)(t)).sortBy(-_._2)
      }
      relation -> sortedObsTuples
    }).toMap)

    logger.info("Finding highest-scoring ent/arg components")
    val entComponent2Best = new mutable.HashMap[(Any, Int), (Cell, Double)]()
    val argComponent2Best = new mutable.HashMap[((String, Int), Int), (Cell, Double)]()
    if (numArgComponents > 0) for (cell <- this.cells.values.filter(_.labeledTrainingData)) {
      for (arg <- 0 until arity(cell.relation)) {
        val ent = cell.tuple(arg)
        val relArg = cell.relation -> arg
        val entComponents = entityComponentIndices(ent)
        val argComponents = argComponentIndices(relArg)
        for (c <- 0 until numArgComponents) {
          val entW = _weights(entComponents(c))
          val argW = _weights(argComponents(c))
          val score = entW * argW
          entComponent2Best.get(ent -> c) match {
            case Some((_, old)) if (score > old) => entComponent2Best(ent -> c) = cell -> score
            case None => entComponent2Best(ent -> c) = cell -> score
            case _ =>
          }
          argComponent2Best.get(relArg -> c) match {
            case Some((_, old)) if (score > old) => argComponent2Best(relArg -> c) = cell -> score
            case None => argComponent2Best(relArg -> c) = cell -> score
            case _ =>
          }

        }
      }
    }


    var processed = 0
    var correct = 0
    def currentPrecision = correct.toDouble / processed
    var AUC = 0.0

    logger.info("Printing out ranked cells")
    for (cell <- sorted) {
      processed += 1
      if (cell.target > 0.5) correct += 1
      AUC += currentPrecision
      if (printAnalysis) out.println("-------")
      out.println("> %6.3f\t%6.3f\t%s #\t%s\t%s".format(cell.predicted, cell.target, cell.relation, cell.tuple.mkString("|"),
        /*getCells(cell.tuple).view.filterNot(_.hide).map(_.relation).mkString(" :: ")*/"Observations")
      )
      if (printAnalysis) {
        for (ent <- cell.tuple) {
          out.println("%-30s %d".format(ent, entity2Cells.get(ent).map(_.size).getOrElse(0)))
        }
        val relTupleScore = calculateRelTupleScore(cell.tuple, cell.relation)
        val argEntScore = calculateArgEntScore(cell.tuple, cell.relation)
        val featScore = calculateFeatScore(cell.tuple, cell.relation)
        out.println("Rel/Tuple score: " + relTupleScore)
        out.println("Arg/Ent score:   " + argEntScore)
        out.println("Feat score:      " + featScore)
        val cellsInRow = getCells(cell.tuple).view

        if (componentAnalysis) {
          val obsRelations = cellsInRow.filter(_.labeledTrainingData).filter(_.target > 0.5).map(_.relation)
          val sortedObsRelations = (for (c <- 0 until numComponents) yield {
            obsRelations.map(r => r -> relMaps(c)(r)).sortBy(-_._2)
          })
          val obsRelationsNeg = cellsInRow.filter(_.labeledTrainingData).filter(_.target <= 0.5).map(_.relation)
          val sortedObsRelationsNeg = (for (c <- 0 until numComponents) yield {
            obsRelationsNeg.map(r => r -> relMaps(c)(r)).sortBy(-_._2)
          })


          val sortedObsTuples = sortedColumns(cell.relation)

          val relComponents = relationComponentIndices(cell.relation).map(_weights(_))
          val tupleComponents = tupleComponentIndices(cell.tuple).map(_weights(_))

          def relProto(c: Int) = if (relComponents(c) < 0.0) sortedRelationComponents(c).last._1 else sortedRelationComponents(c).head._1
          def tupleProto(c: Int) = if (tupleComponents(c) < 0.0) sortedTupleComponents(c).last._1 else sortedTupleComponents(c).head._1
          def relProtoObserved(c: Int) = if (relComponents(c) < 0.0) sortedObsRelations(c).lastOption else sortedObsRelations(c).headOption
          def relProtoObservedNeg(c: Int) = if (relComponents(c) < 0.0) sortedObsRelationsNeg(c).lastOption else sortedObsRelationsNeg(c).headOption
          def tupleProtoObserved(c: Int) = if (tupleComponents(c) < 0.0) sortedObsTuples(c).lastOption else sortedObsTuples(c).headOption
          val sortedComponents = (relComponents zip tupleComponents zipWithIndex).map(t => t._2 -> t._1._1 * t._1._2).sortBy(-_._2)
          val sortedProtos = sortedComponents.map(p => (
            tupleComponents(p._1), relComponents(p._1),
            tupleProto(p._1), relProto(p._1), tupleProtoObserved(p._1),
            relProtoObserved(p._1), relProtoObservedNeg(p._1)))


          out.println("Rel/Tuple Components: ")
          for (((component, score), (wTuple, wRel, tuple, relation, tupleObs, relObs, relObsNeg)) <- sortedComponents zip sortedProtos) {
            out.println("  %7.3f %7.3f %7.3f %3d %7.3f %-50s %7.3f %-60s %7.3f %s".format(score, wTuple, wRel, component,
              tupleObs.map(_._2).getOrElse(0.0), tupleObs.map(_._1.mkString(" | ")).getOrElse("NA"),
              relObs.map(_._2).getOrElse(0.0), relObs.map(_._1).getOrElse("NA"),
              relObsNeg.map(_._2).getOrElse(0.0), relObsNeg.map(_._1).getOrElse("NA")
            ))
          }
          out.println("Arg/Ent Components: ")
          //print components, score, index
          if (numArgComponents > 0) for (arg <- 0 until arity(cell.relation)) {
            out.println(" Arg: " + arg)
            val ent = cell.tuple(arg)
            val relArg = cell.relation -> arg
            val entComponents = entityComponentIndices(ent)
            val argComponents = argComponentIndices(relArg)
            def printCell(cell: Cell) = "%5.2f %s %-40s".format(cell.target, cell.tuple.map(e => "%-20s".format(e)).mkString("|"), cell.relation)
            for (c <- 0 until numArgComponents) {
              val entW = _weights(entComponents(c))
              val argW = _weights(argComponents(c))
              val bestEnt = entComponent2Best.get(ent -> c)
              val bestRelArg = argComponent2Best.get(relArg -> c)
              out.println("  Component %3d %7.3f %7.3f %7.3f".format(c, entW * argW, entW, argW))
              out.println("    %7.3f %s".format(bestEnt.map(_._2).getOrElse(0.0), bestEnt.map(p => printCell(p._1)).getOrElse("NA")))
              out.println("    %7.3f %s".format(bestRelArg.map(_._2).getOrElse(0.0), bestRelArg.map(p => printCell(p._1)).getOrElse("NA")))

            }
          }
          cellsInRow.filter(_.toPredict).foreach(_.update())
          val maxObserved = cellsInRow.filter(_.labeledTrainingData).sortBy(-_.predicted)
          val maxHidden = cellsInRow.filter(_.hide).sortBy(-_.predicted)
          out.println("  Max Observed: ")
          out.println(maxObserved.map(c => "%6.3f %6.3f %s".format(c.predicted, c.target, c.relation)).mkString("    ", "\n    ", ""))
          out.println("  Max Hidden: ")
          out.println(maxHidden.map(c => "%6.3f %6.3f %s".format(c.predicted, c.target, c.relation)).mkString("    ", "\n    ", ""))
          out.println("  Context: ")
          out.println(cellsInRow.filter(c => !c.feature && !c.toPredict).map(_.relation).mkString("    ", "\n    ", ""))
          out.println("  Feats: ")
          out.println(cellsInRow.filter(c => c.feature && c.relation != cell.relation).map(c => {
            val index = relPairIndices(c.relation, cell.relation)
            "%6.3f %s".format(_weights(index), c.relation)
          }).mkString("    ", "\n    ", ""))
        }

      }
    }
    out.println("Total: " + processed)
    out.println("AUC:   " + AUC / processed)
  }

  def debugTuples(tuples: Iterable[Seq[Any]], out: PrintStream, entityPrinter: Any => String = id => id.toString) {
    logger.info("Writing out tuples information")
    for (tuple <- tuples) {
      val components = tupleComponentIndices.get(tuple).map(_.map(_weights(_)).mkString(" "))
  //    if (components.isDefined) {

        out.println("----")
        out.println(tuple.map(entityPrinter(_)).mkString(" | "))

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
        cells.foreach(_.update())
        val sortedObs = cells.filter(!_.hide).sortBy(-_.predicted)
        out.println("Observed:")
        for (cell <- sortedObs) {
          out.println("  %s %6.4f %6.4f %s ".format(if (cell.hide) "H" else "O", cell.target, cell.predicted, cell.relation))
        }
        out.println("Hidden:")
        val sortedHidden = cells.filter(_.hide).sortBy(-_.predicted)
        for (cell <- sortedHidden) {
          out.println("  %s %6.4f %6.4f %s ".format(if (cell.hide) "H" else "O", cell.target, cell.predicted, cell.relation))
        }

  //    }
    }

  }

  def heldoutRank(tuples : Iterable[Seq[Any]], out : PrintStream, entityPrinter: Any => String = id => id.toString) {
    if(tuples == null || tuples.size == 0) return
    logger.info("Writing out top predictions for tuple")
    var threshold = 0.0
    val ranks = new ArrayBuffer[Int]

    for (tuple <- tuples) {

      out.println("----")
      out.println(tuple.map(entityPrinter(_)).mkString(" | "))

      val preds = new ArrayBuffer[(String,Double)]
      threshold = 1.0e+9
      val hiddenCells = getCells(tuple).filter(_.hide)

      val obsCells = getCells(tuple).filter(!_.hide)
      out.println(obsCells.map(cell=>"  Obs:" + cell.target + " " +  cell.relation).mkString("\n"))
      for (cell <- hiddenCells){
        val score = calculateScoreRaw(cell.tuple,cell.relation)

        if (score < threshold) {
          threshold = score
        }
      }

      val candRels = relations
      for (relation <- candRels){
        val score = calculateScoreRaw(tuple,relation)
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
            if (cell.hide) {
              ranks += rank
              out.println("Rank\t" + tuple.mkString("|") + "\t" +  cell.relation +  "\t" + rank)
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


  def debugCooccurence(out: PrintStream) {
    //go over all tuples, create all pairs, then group
    logger.info("Writing out cooccurence information")
    val pairs = tuples.toSeq.flatMap(tuple => {
      val rels = getCells(tuple).filter(_.target > 0.5).map(_.relation).filter(isToBePredicted)
      for (rel1 <- rels; rel2 <- rels; if (rel1 != rel2)) yield rel1 -> rel2
    })
    val grouped = pairs.groupBy(_._1)
    for ((rel, rels) <- grouped) {
      out.println(rel)
      for ((_, list) <- rels.groupBy(_._2).toSeq.sortBy(-_._2.size)) {
        out.println("  %d %s".format(list.size, list.head._2))
      }
    }
  }

  def debugModel(out: PrintStream) {
    //for each tuple component show top k tuples
    if (useBias) out.println("Global Bias: " + _weights(biasIndex))
    out.println("Tuple/Relation components")
    for (c <- 0 until numComponents) {
      out.println(" Component " + c)
      val sortedTuples = tupleComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      val sortedRelations = relationComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      for ((tuple, relation) <- sortedTuples.take(10) zip sortedRelations.take(10)) {
        out.println("   %6.4f %-50s %7.4f %s".format(tuple._2, tuple._1.mkString(" | "), relation._2, relation._1))
      }
      out.println("    ...")
      for ((tuple, relation) <- sortedTuples.takeRight(10) zip sortedRelations.takeRight(10)) {
        out.println("   %6.4f %-50s %7.4f %s".format(tuple._2, tuple._1.mkString(" | "), relation._2, relation._1))
      }
    }
    out.println("Entity/Arg components")
    for (c <- 0 until _numArgComponents) {
      out.println(" Component " + c)
      val sortedEntities = entityComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      val sortedArgs = argComponentIndices.toSeq.map(pair => pair._1 -> _weights(pair._2(c))).sortBy(-_._2)
      for ((ent, arg) <- sortedEntities.take(10) zip sortedArgs.take(10)) {
        out.println("   %6.4f %-50s %7.4f %s".format(ent._2, ent._1, arg._2, arg._1))
      }
      out.println("    ...")
      for ((ent, arg) <- sortedEntities.takeRight(10) zip sortedArgs.takeRight(10)) {
        out.println("   %6.4f %-50s %7.4f %s".format(ent._2, ent._1, arg._2, arg._1))
      }
    }

    out.println("Per Relation")
    for (rel <- relations.toSeq.sorted; if (isToBePredicted(rel))) {
      out.println("   %-50s %s".format(rel, relationComponentIndices.get(rel).map(_.map(i => "%6.3f".format(_weights(i))).mkString(" ")).getOrElse("NA")))
    }

    out.println("Feature weights")
    val sortedRelPairs = relPairIndices.view.toSeq.map(pair => pair._1 -> _weights(pair._2)).sortBy(-_._2)
    for (((rel1, rel2), weight) <- sortedRelPairs) {
      out.println("  %30f %-30s %-30s".format(weight, rel1, rel2))
    }

  }

  //added by Limin Yao, now only handle tuple, relation, bias parameters for relation
  def saveModel(out: PrintStream) {
    //for each predicate, output:  pred \t weight weight weight
    out.println("#Relation parameters!")
    for (rel <- relations; if (rel != "bias")) {
      out.print(rel)
      val relIndices = relationComponentIndices(rel)
      for (c <- 0 until numComponents) {
        out.print("\t" + _weights(relIndices(c)))
      }
      out.println()
    }

    out.println("#Bias parameters for each relation")
    val relPairs = relPairIndices.view.toSeq.map(pair => pair._1 -> _weights(pair._2))
    for (((rel1, rel2), weight) <- relPairs) {
      out.println(rel1 + "|" + rel2 + "\t" + weight)
    }


    out.println("#Tuple parameters!")
    for (tuple <- tuples) {
      out.print(tuple.mkString("|"))
      val tupleIndices = tupleComponentIndices(tuple)
      for (c <- 0 until numComponents) {
        out.print("\t" + _weights(tupleIndices(c)))
      }
      out.println()
    }
  }


  class Eval {
    var fp = 0
    var fn = 0
    var tp = 0
    var tn = 0

    def pos = fp + tp

    def neg = fn + tn

    def all = neg + pos

    def goldPos = tp + fn

    def goldNeg = tn + fp

    def p = tp.toDouble / (tp + fp)

    def r = tp.toDouble / (tp + fn)

    def debug(out: PrintStream) {
      out.println("==============")
      out.println("Total: %d".format(all))
      out.println("True Positive:   %d".format(tp))
      out.println("Guess (POS):   %d".format(pos))
      out.println("Guess (NEG):   %d".format(neg))
      out.println("Guess Ratio:   %f".format(pos.toDouble / all))
      out.println("Gold  (POS):   %d".format(goldPos))
      out.println("Gold  (NEG):   %d".format(goldNeg))
      out.println("Gold  Ratio:   %f".format(goldPos.toDouble / all))
      out.println("Prec:          %f".format(p))
      out.println("Rec:           %f".format(r))

    }

  }


  def evaluate(threshold: Double = 0.5): Eval = {
    val result = new Eval
    for (cell <- hidden) {
      val guess = predictCell(cell.relation, cell.tuple)
      val gold = cell.target
      (guess >= threshold) -> (gold >= threshold) match {
        case (true, true) => result.tp += 1
        case (true, false) => result.fp += 1
        case (false, true) => result.fn += 1
        case (false, false) => result.tn += 1
      }
    }
    result
  }

  final def predictCell(relation: String, tuple: Seq[Any]): Double = {
    //use components
    predictCell(cells(relation -> tuple))
  }

  final def predictCell(cell: Cell, aux: Boolean = false): Double = {
    //use components
    val theta = calculateScore(cell, aux)
    calculateProb(theta)
  }


  def loadTupleFile(in: InputStream, limit: Int = 1000, hideHowMany: Int = 1,
                    isFeature: String => Boolean = (s: String) => false) {
    val tupleSource = Source.fromInputStream(in, "latin1")
    val oldCount = cells.size
    for (line <- tupleSource.getLines().take(limit)) {
      val split = line.split("\t")
      val pair = split(1).split("\\|").toSeq
      val relations = random.shuffle(split.drop(2).map(_.split("#").head).toSeq)
      val (hidden, observed) = relations.splitAt(hideHowMany)
      for (rel <- observed) addCell(rel, pair, 1.0, false, isFeature(rel), !isFeature(rel))
      for (rel <- hidden) addCell(rel, pair, 1.0, true, isFeature(rel), !isFeature(rel))
    }
    tupleSource.close()
    logger.info("Loaded %d cells.".format(cells.size - oldCount))
  }

}


object SelfPredictingDatabase {
  def main(args: Array[String]) {
    val conf = new Config(new FileInputStream("conf/toy.properties"))

    val spdb = new SelfPredictingDatabase(conf.get("num-components", 10), conf.get("num-arg-components", 5))
    spdb.lambdaRel = conf.get("lambda-rel", 0.0)
    spdb.lambdaTuple = conf.get("lambda-tuple", 0.0)


    val tupleFile = conf.get[File]("tuple-file")

    //load positive cells
    def isFeature(rel: String) = rel.startsWith("lc:") || rel.startsWith("rc:")
    spdb.loadTupleFile(new FileInputStream(tupleFile), 10000, 1, isFeature(_))

    //generate negative cells
    val relations = spdb.relations.toSeq.filter(!isFeature(_))
    for (tuple <- spdb.tuples) {
      val shuffled = spdb.random.shuffle(relations)
      val observed = shuffled.take(3)
      val hidden = shuffled.takeRight(3)
      for (rel <- observed; if (spdb.getCell(rel, tuple).isEmpty)) spdb.addCell(rel, tuple, 0.0, false)
      for (rel <- hidden; if (spdb.getCell(rel, tuple).isEmpty)) spdb.addCell(rel, tuple, 0.0, true)
    }

    //train model
    spdb.runLBFGS()

    val eval = spdb.evaluate(0.5)

    println(eval.p)
    println(eval.r)

    spdb.debugTuples(spdb.tuples.take(10000), new PrintStream("predictions.txt"))
    spdb.debugModel(new PrintStream("spdb.debug"))


  }

}
