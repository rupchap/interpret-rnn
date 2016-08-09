package edu.umass.cs.iesl.spdb

import java.io.{File, PrintStream}
import collection.mutable.{HashSet, ArrayBuffer}
import io.Source
import util.Random
import collection.mutable

/**
 * entry for loading row style training and test data, each row has a pair and a predicate
 * hidden means observed or not, when cell.target is 0.0, two cases, if hidden, means ? else means 0.0
 *
 * @author lmyao
 */
object PcaDSRun extends HasLogger {
  val allRelationsToPredict = new HashSet[String]
  // for generating negative predicates, we do not enlarge predicate space when loading test data
  val labels = new HashSet[String]
  val testTuples = new HashSet[Seq[Any]]
  var trainTuples: HashSet[Seq[Any]] = null

  def shouldBeFeature(relation: String) = {
    feats.exists(relation.startsWith(_))
  }

  def shouldBePredicted(relation: String) = {
    predict.exists(relation.startsWith(_))
  }

  def shouldBeSkipped(relation: String) = {
    unused.exists(relation.startsWith(_))
  }

  import collection.JavaConversions._

  lazy val feats = Conf.conf.getStringList("pca.feats").toSet
  // Set("lc", "rc", "pos")
  lazy val predict = Conf.conf.getStringList("pca.predict").toSet
  //Set("path", "lex", "REL$")
  lazy val unused = Conf.conf.getStringList("pca.unused").toSet //Set("lc","rc","trigger","ner")


  //add entity-attr unary relational data
  def addUnaryData(spdb: SelfPredictingDatabase, file: String) {
    logger.info("Loading unary data from file " + file)
    val source = Source.fromFile(file, "ISO-8859-1")
    var count = 0
    for (line <- source.getLines()) {
      count += 1
      val fields = line.split("\t")
      val entity = fields(0)
      for (rel <- fields.drop(1)){
        if (spdb.getCell(rel, Seq(entity, "dummy")).isEmpty) {
          spdb.addCell(rel,Seq(entity, "dummy"))
        }
      }

      if (count % 10000 == 0) logger.info("Loaded entities:" + count)
    }

    logger.info("#entities after adding unary data:" + spdb.tuples.size)
    logger.info("#unary relations:" + spdb.relations.size)
  }

  //posfile,   arg arg pattern, labelPrefix indicates the distant supervision relation labels, could be train, could be test
  def addTrainData(spdb: SelfPredictingDatabase, file: String, labelPrefix: String = "REL$", minCellsPerRelation: Int = 2, minCellsPerTuple: Int = 2, percentage: Double = 0.1) {
    logger.info("Loading training data from file " + file)
    val cells = new ArrayBuffer[(Seq[Any], String)]
    val taggedTuples = new HashSet[Seq[Any]]
    val random = new Random(0)
    var skip = false

    val source = Source.fromFile(file, "ISO-8859-1")
    for (line <- source.getLines()) {
      var fields: Array[String] = null

      skip = false
      if (line.contains("REL$NA") && random.nextDouble() > percentage) {
        //0.1 could be a parameter, skipping instances could be unlabeled data,todo
        skip = true
      } // skip 90% of the negative examples

      if (line.startsWith("POSITIVE") || line.startsWith("NEGATIVE") || line.startsWith("UNLABELED")) {
        fields = line.split("\t").drop(1)
      } else
        fields = line.split("\t")
      val tuple = fields.take(2)
      if (!skip) {
        //positive, negative and unlabeled data for training
        for (relation <- fields.drop(2))
          cells += ((tuple, relation))
        if (!line.startsWith("UNLABELED") ) taggedTuples += tuple
      }
      else {
        //change negative to unlabeled , skipping REL$NA
        for (relation <- fields.drop(2); if (!relation.startsWith("REL")))
          cells += ((tuple, relation))
      }
    }
    source.close()

    logger.info("Loaded files with %d cells in total, now adding cells".format(cells.size))

    val relation2cells = cells.groupBy(_._2)
    logger.info("Removing infrequent relations")
    val frequentRelCells = cells.filter(c =>
    //c._2.startsWith("sen#") ||
    //c._2.startsWith("REL$") ||
      relation2cells.getOrElse(c._2, Seq.empty).size >= minCellsPerRelation)

    val tuple2PredictedCells = frequentRelCells.filter(pair => pair._2 != "REL$NA" && shouldBePredicted(pair._2)).groupBy(_._1)



    //this does row filtering
    val frequentCells = frequentRelCells.filter(c => tuple2PredictedCells.getOrElse(c._1, Seq.empty).size >= minCellsPerTuple && tuple2PredictedCells.getOrElse(c._1, Seq.empty).size < 50)

    var numCells = 0

    for ((tuple, relation) <- frequentCells; if (!shouldBeSkipped(relation))) {
      val feature = shouldBeFeature(relation)
      val toPredict = shouldBePredicted(relation)
      if (spdb.getCell(relation, tuple).isEmpty) {
        spdb.addCell(relation, tuple, 1.0, hide = false, feature = feature, predict = toPredict)
        if (!relation.contains("|") && relation.startsWith(labelPrefix))
          labels += relation
        else if (toPredict) allRelationsToPredict += relation
        numCells += 1
        if (numCells % 100000 == 0) logger.info("Added %d cells".format(numCells))

      }
      Unit

    }

    trainTuples = taggedTuples.filter(tuple => spdb.getCells(tuple).size > 0)

    logger.info("Added %d cells".format(numCells))
    logger.info("Number of tuples " + spdb.tuples.size)
    logger.info("Number of predicates: " + spdb.relations.size)
    logger.info("Number of labels:" + labels.size)
    logger.info("Loaded train tuples:" + trainTuples.size)


    var os = new PrintStream(new File(Conf.outDir, "labels.txt"))
    os.println(labels.mkString("\n"))
    os.close()

    os = new PrintStream(new File(Conf.outDir, "patterns.txt")) //all predicates/patterns,exluding labels
    os.println(spdb.relations.filter(x => !x.startsWith("REL")).mkString("\n")) //if we have features and predicates, we should include both

    os.close()

    os = new PrintStream(new File(Conf.outDir, "trainTuples.txt")) //train pos and neg data
    os.println(trainTuples.map(_.mkString("|")).mkString("\n"))
    os.close()
  }

  //for test, we do not need unlabeled data
  def addTestData(spdb: SelfPredictingDatabase, file: String) {
    logger.info("Loading test data from file " + file)
    val source = Source.fromFile(file, "ISO-8859-1")
    for (line <- source.getLines(); if (!line.startsWith("UNLABELED"))) {
      var fields: Array[String] = null
      if (line.startsWith("POSITIVE") || line.startsWith("NEGATIVE")) {
        fields = line.split("\t").drop(1)
      } else
        fields = line.split("\t")
      val tuple = fields.take(2)

      val relations = fields.drop(2)
      if (relations.exists(r => (shouldBePredicted(r) && spdb.relationSet(r)) || labels(r))) for (relation <- relations; if (!shouldBeSkipped(relation))) {
        val feature = shouldBeFeature(relation)
        val toPredict = shouldBePredicted(relation)
        if (spdb.getCell(relation, tuple).isEmpty) {

          if (labels.contains(relation)) {
            spdb.addCell(relation, tuple, 1.0, hide = true, feature = feature, predict = toPredict)
          }
          else if (spdb.relationSet.contains(relation)) {
            //keep the feature space fixed, only add to existing predicates
            spdb.addCell(relation, tuple, 1.0, hide = false, feature = feature, predict = toPredict) //this is feature, should be observed, not hidden
          }
        }

      }

      //add other ds labels only for test data, not for unlabeled data
      //number of cells in one row should be larger than 1 to make sure it has features, todo
      //todo: we should also add test cells for which we don't have features
      var count = 0
   //   if (spdb.getCells(tuple).filterNot(_.hide).size > 0) {
        testTuples += tuple
        count = testTuples.size
        //adding other labels for test, require labels is filled in with train data
        for (label <- labels)
          if (spdb.getCell(label, tuple).isEmpty) spdb.addCell(label, tuple, 0.0, hide = true)
        if (count % 10000 == 0) logger.info("Loaded test tuples:" + count)
    //  }

    }
    source.close()
    logger.info("Number of predicates after adding test data " + spdb.relations.size)
    logger.info("#test tuples:" + testTuples.size)
    val os = new PrintStream(new File(Conf.outDir, "testTuples.txt"))
    os.println(testTuples.map(_.mkString("|")).mkString("\n"))
    os.close()
  }

  def loadData(spdb:SelfPredictingDatabase, file : String, hidden : Boolean = false){
    val source = Source.fromFile(file)
    var count = 0
    for (line <- source.getLines()) {
      val fields = line.split("\t")
      val tuple = fields.take(2)
      val relation = fields(2)
      if (spdb.getCell(relation,tuple).isEmpty) {
        spdb.addCell(relation,tuple, 1.0, hide = hidden)
        if (hidden) testTuples += tuple
        count += 1
      }
    }
    logger.info("Added %d cells".format(count))
    logger.info("Number of predicates " + spdb.relations.size)

    if (testTuples.size > 0) {
      val os = new PrintStream(new File(Conf.outDir, "testTuples.txt"))
      os.println(testTuples.map(_.mkString("|")).mkString("\n"))
      os.close()
    }

  }

  def isDuplicate(tuple: Seq[Any]) = {
    tuple.head.toString.startsWith("dup#")
  }

  def addCellWithRowDuplication(spdb: SelfPredictingDatabase,
                                relation: String, tuple: Seq[Any],
                                target: Double, hide: Boolean, feature: Boolean, toPredict: Boolean) {
    val oldRow = spdb.getCells(tuple)
    val newTuple = tuple.map(_.toString + " [" + relation + "]")
    for (c <- oldRow)
      if (spdb.getCell(c.relation, newTuple).isEmpty)
        spdb.addCell(c.relation, newTuple, c.target, c.hide, c.feature, c.toPredict)
    if (spdb.getCell(relation, newTuple).isEmpty)
      spdb.addCell(relation, newTuple, target, hide = hide, feature = feature, predict = toPredict)
  }

  //for one instance, one label is true, all others are false, the true label could be 'NA', pay attention to 'NA',
  def addNegativeDSLabelsForTraining(spdb: SelfPredictingDatabase) {
    val hidden = false
    val allowAllNegTuples = Conf.conf.getBoolean("pca.use-all-neg-tuples")
    logger.info("Adding other distant supervision labels for train tuples")
    var count = 0

    val duplicateNegDSRows = Conf.conf.getBoolean("pca.dup-neg-ds")
    def addNegCell(label: String, tuple: Seq[Any]) {
      if (duplicateNegDSRows)
        addCellWithRowDuplication(spdb, label, tuple, 0.0, hidden, feature = false, toPredict = true)
      else
        spdb.addCell(label, tuple, 0.0, hidden)
    }

    //due to row filtering, some tuples may be filtered out, make sure that we only add to tuples which have features
    //todo: this is risky here: it means that we never see instances with all freebase relations = false, which
    //todo: means we have a bias towards active freebase relations
    val tuples = if (trainTuples != null) trainTuples else spdb.tuples.filterNot(isDuplicate)
    for (tuple <- tuples; if (allowAllNegTuples || spdb.getCells(tuple).filter(_.toPredict).size > 0)) {
      for (label <- labels) {
        if (spdb.getCell(label, tuple).isEmpty) addNegCell(label, tuple)
      }
      count += 1
      if (count % 10000 == 0) logger.info("Processed tuples:" + count)
    }

  }

  //
  //one question is whether we still need to include negative predicates for a pair, maybe we need, for both train and test data, this is not involving any DS labels
  def addNegativeRelations(spdb: SelfPredictingDatabase) {

    val tuples = spdb.tuples.filterNot(isDuplicate)
    val negFeatureRatio = Conf.conf.getInt("pca.neg-features")
    val negFeatureDynamic = Conf.conf.getBoolean("pca.neg-dynamic")

    val duplicateNegPatternRows = Conf.conf.getBoolean("pca.dup-neg-pattern")
    def addNegCell(label: String, tuple: Seq[Any]) {
      if (duplicateNegPatternRows)
        addCellWithRowDuplication(spdb, label, tuple, 0.0, hide = false, feature = false, toPredict = true)
      else
        spdb.addCell(label, tuple, 0.0, hide = false)
    }


    logger.info("Adding negative surface patterns!")
    val startTime = System.currentTimeMillis()
    val allRelationsToPredictSeq = allRelationsToPredict.toIndexedSeq
    //for each tuple, sample 10 times negative surface relations
    var count = 0
    for (tuple <- tuples) {
      //  logger.info("Add negative surface patterns for " + tuple.mkString("\t") )
      val tupleRelations = new HashSet[String]
      tupleRelations ++= spdb.getCells(tuple).map(_.relation).filterNot(_.startsWith("REL$"))

      //when the original patterns co-occurring with the tuple are too many, we add few negative features
      val total = if (negFeatureDynamic && tupleRelations.size < 10) tupleRelations.size * negFeatureRatio else negFeatureRatio
      for (iter <- 0 until total) {
        var sampledIndex = (math.random * allRelationsToPredictSeq.size).toInt
        var sampledRelation = allRelationsToPredictSeq(sampledIndex)
        while (tupleRelations.contains(sampledRelation)) {
          sampledIndex += 1
          sampledIndex = sampledIndex % allRelationsToPredictSeq.size
          sampledRelation = allRelationsToPredictSeq(sampledIndex)
        }
        if (spdb.getCell(sampledRelation, tuple).isEmpty)
          addNegCell(sampledRelation, tuple)
        //          spdb.addCell(sampledRelation, tuple, 0.0, false)
      }
      count += 1
      if (count % 10000 == 0) logger.info("Processed tuples:" + count)
    }
    logger.info("Finish adding negative features within " + (System.currentTimeMillis() - startTime) / 1000 + " seconds!")
  }

  def main(args: Array[String]) {
    val configFile = if (args.length > 0) args(0) else "relation-matrix.conf"
    Conf.add(configFile)

    var watson = false
    if (configFile.startsWith("watson")) watson = true


    runPCA(watson)
  }


  /**
   * Writing out some debug output into the run-specific output directory.      todo:save model and load model, inference, total new rows unobserved during training
   * @param spdb the matrix to debug.
   */
  def debugOutput(spdb: SelfPredictingDatabase, prefix: String) {

    spdb.debugModel(new PrintStream(new File(Conf.outDir, prefix + ".model.txt")))
    spdb.debugTuples(testTuples, new PrintStream(new File(Conf.outDir, prefix + ".tuples.test.pred.txt")))
    spdb.debugTuples(trainTuples, new PrintStream(new File(Conf.outDir, prefix + ".tuples.train.pred.txt")))
    spdb.debugTuples(spdb.tuples.filterNot(t => trainTuples(t) || testTuples(t)),
      new PrintStream(new File(Conf.outDir, prefix + ".tuples.remainder.pred.txt")))
  }

  /**
   * Writes out results in readable format, src dest pat score.    also write out the original data as ground truth, evaluation can also be here
   * @param spdb the matrix.
   * @param threshold when to predict a label is true.
   */
  def writeOutResults(spdb: SelfPredictingDatabase, prefix: String, threshold: Double = 0.5,
                      multilabel: Boolean = true, extraRelations: Seq[String] = Seq.empty) {
    logger.info("Writing out distant supervision predictions.")
    val result = new PrintStream(new File(Conf.outDir, prefix + ".curve.txt")) //the best label, second best if best is NA
    val os = new PrintStream(new File(Conf.outDir, prefix + ".rank.txt")) //all labels with score above 0.5
    var cor = 0
    var gold = 0
    var total = 0
    val preds = new ArrayBuffer[(Double, String, String, String)]
    //get predicted query cells, for distant supervision, we only pick the one with the highest score, if it is NA, try the second, score to be decreased by 1
    for (tuple <- testTuples) {
      //find one one label with the highest score
      var maxScore = 0.0
      var bestLabel = "REL$NA"
      var naScore = 0.0
      var goldLabel = "REL$NA"
      var goldLabels = new HashSet[String]
      val predictedCells = new ArrayBuffer[SelfPredictingDatabase#Cell]
      val cells = spdb.getCells(tuple)
      val tupleString = tuple.mkString("\t")
      for (cell <- cells; if (labels.contains(cell.relation))) {
        if (cell.target > threshold) {
          goldLabel = cell.relation
          goldLabels += cell.relation
        }
        cell.update()
        if (cell.predicted > threshold) {
          predictedCells += cell
          if (cell.relation == "REL$NA") naScore = cell.predicted
          else //this handles non-NA labels
          if (cell.predicted > maxScore) {
            maxScore = cell.predicted
            bestLabel = cell.relation
          }
          if (multilabel) preds += ((cell.predicted, tupleString, goldLabel, cell.relation))
        }
      }
      if (goldLabels.size > 0) gold += 1
      //extra relations
      for (rel <- extraRelations) {
        spdb.getCell(rel, tuple) match {
          case None =>
            val prob = spdb.calculateProbRaw(tuple, rel)
            val gold = "REL$NA"
            preds += ((prob, tupleString, gold, rel))
          case Some(cell) if (cell.testData) =>
            val prob = cell.update()
            val gold = if (cell.target > 0.5) cell.relation else "REL$NA"
            preds += ((prob, tupleString, gold, rel))
          case Some(cell) =>
        }
      }

      if (maxScore < naScore) maxScore -= 1.0
      if(goldLabels.contains(bestLabel)) goldLabel = bestLabel
      if (!multilabel) preds += ((maxScore, tupleString, goldLabel, bestLabel))
    }

    val sorted = preds.sortBy(-_._1)

    var id = 0
    for ((score, tupleString, goldLabel, bestLabel) <- sorted) {
      if (goldLabel == bestLabel) {
        cor += 1
      }
      total += 1
      id += 1
      result.println(id + "\t" + cor * 1.0 / gold + "\t" + cor * 1.0 / total)
      os.println(score + "\t" + tupleString + "\t" + goldLabel + "\t" + bestLabel)
    }
    os.close()
    result.close()
  }

  /**
   * Removes tuples for with  one or less active training cells, and no test cells.
   * @param spdb the database to remove from.
   */
  def removeUninformativeTrainingCells(spdb: SelfPredictingDatabase, onlyFreebase: Boolean = false) {
    logger.info("Searching for uniformative training cells")
    val toRemove = new ArrayBuffer[spdb.Cell]
    for (tuple <- spdb.tuples) {
      val cells = spdb.getCells(tuple)
      val testing = cells.exists(_.hide)
      if (!testing) {
        val uninformative = cells.view.filter(c => !c.hide && c.toPredict && c.target > 0.0).size <= 1
        val freebase = !onlyFreebase || cells.view.exists(c => c.target > 0.5 && c.relation.startsWith("REL$"))
        if (uninformative || !freebase)
          toRemove ++= cells
      }
    }
    logger.info("Removing %d uninformative cells".format(toRemove.size))
    for (cell <- toRemove) spdb.removeCell(cell)
  }

  /**
   * Prepares the matrix, runs the actual PCA algorithm, does some debug and evaluation output, and then writes out
   * the result in TAC format.
   */
  def runPCA(watson: Boolean) {
    logger.info("Output dir: " + Conf.outDir.getAbsolutePath)
    logger.info("Preparing PCA")
    val spdb = new SelfPredictingDatabase()
    val normalizer = new mutable.HashMap[Any, Double]()
    spdb.numComponents = Conf.conf.getInt("pca.rel-components")
    spdb.numArgComponents = Conf.conf.getInt("pca.arg-components")
    spdb.lambdaRel = Conf.conf.getDouble("pca.lambda-rel")
    spdb.lambdaTuple = Conf.conf.getDouble("pca.lambda-tuple")
    spdb.lambdaRelPair = Conf.conf.getDouble("pca.lambda-feat")
    spdb.lambdaBias = Conf.conf.getDouble("pca.lambda-bias")
    spdb.lambdaEntity = Conf.conf.getDouble("pca.lambda-ent")
    spdb.lambdaArg = Conf.conf.getDouble("pca.lambda-arg")
    spdb.maxIterations = Conf.conf.getInt("pca.max-msteps")
    spdb.useBias = Conf.conf.getBoolean("pca.use-global-bias")
    spdb.tolerance = Conf.conf.getDouble("pca.tolerance")
    spdb.gradientTolerance = Conf.conf.getDouble("pca.gradient-tolerance")
    spdb.maxCores = Conf.conf.getInt("pca.max-cores")
    spdb.relationNormalizer = Conf.conf.getString("pca.rel-normalizer") match {
      case "count" => r => normalizer.getOrElseUpdate(r, 1.0 / spdb.getCells(r).filter(_.labeledTrainingData).size)
      case _ => r => 1.0
    }
    spdb.tupleNormalizer = Conf.conf.getString("pca.tuple-normalizer") match {
      case "count" => t => normalizer.getOrElseUpdate(t, 1.0 / spdb.getCells(t).filter(_.labeledTrainingData).size)
      case _ => t => 1.0
    }

    if (!watson)
      addTrainData(spdb, Conf.conf.getString("source-data.train"), "REL$",
        Conf.conf.getInt("pca.rel-cutoff"), Conf.conf.getInt("pca.tuple-cutoff"),
        Conf.conf.getDouble("source-data.percentage") //skipping (1-percentage) negative examples
      )
    else
      addTrainData(spdb, Conf.conf.getString("source-data.train"), "TWREX$", Conf.conf.getInt("pca.rel-cutoff"))

    if (!Conf.conf.getBoolean("pca.no-neg-freebase"))
      addNegativeDSLabelsForTraining(spdb)

    if (Conf.conf.getBoolean("pca.print-cooccur"))
      spdb.debugCooccurence(new PrintStream(new File(Conf.outDir, "cooccurence.txt")))

    addTestData(spdb, Conf.conf.getString("source-data.test"))

    //addNegativeRelations(spdb)

    //analyseSecondOrderRules(spdb, labels, new PrintStream(new File(Conf.outDir, "second-order.txt")))

    if (Conf.conf.getBoolean("pca.remove-uninformative"))
      removeUninformativeTrainingCells(spdb, Conf.conf.getBoolean("pca.freebase-only"))

    if (Conf.conf.getBoolean("pca.relation-bias"))
      spdb.addBias()

    //writing out tuples before inference
    //spdb.debugTuples(spdb.tuples, new PrintStream(new File(Conf.outDir, "tuples.init.txt")))
    //    spdb.debugTuples(testTuples, new PrintStream(new File(Conf.outDir, "tuples.test.init.txt")))
    //    spdb.debugTuples(trainTuples, new PrintStream(new File(Conf.outDir, "tuples.train.init.txt")))
    //create per relation tuples for debugging later
    val perRelationTuples = if (Conf.conf.getBoolean("pca.create-per-rel-tuples"))
      createPerRelationTuples(spdb)
    else Seq.empty

    //create proto tuple cells for GE constraint
    if (Conf.conf.getDouble("pca.proto-scale") > 0.0)
      setupPrototypes(spdb)

    //turn all training cells to be predicted (which are not freebase labels) into features
    if (Conf.conf.getBoolean("pca.pseudo")) {
      //get pair counts
      for (tuple <- spdb.tuples) {
        val cells = spdb.getCells(tuple).filter(_.labeledTrainingData).filterNot(_.relation.startsWith("REL$"))
        for (cell <- cells) cell.turnOnFeature()
      }
    }
    
    //addUnaryData(spdb, Conf.conf.getString("source-data.unary"))

//    loadData(spdb,Conf.conf.getString("heldout.train"))
//    loadData(spdb,Conf.conf.getString("heldout.test"),true)

    //run PCA
    Conf.conf.getString("pca.mode") match {
      case "sgd" => spdb.runSGD()
      case "bpr" => spdb.runBPR()
      case "bpr-inv" => spdb.runBPRInv()
      case "bpr-ent" => spdb.runBPRAll(entityCentric = true)
      case "bpr-all" => spdb.runBPRAll()
      case "bpr-row" => spdb.runBPRByTuple()
      case _ => spdb.runLBFGS()
    }

    //run PCA only on the perRelationTuples
    //    logger.info("Running on per relation tuples")
    //    spdb.updateModel(perRelationTuples.take(1), updateRel = false, updateBias = false, updateFeat = false)

    //print out debug statistics
//    debugOutput(spdb, "nyt_pair")
    writeOutResults(spdb, "nyt_pair_old", 0.0, false)

    //write out results
    val extraRelations = Conf.conf.getStringList("eval.extra-relations").toSeq
    writeOutResults(spdb, "nyt_pair", 0.0, extraRelations = extraRelations)
    //spdb.heldoutRank(testTuples, new PrintStream(new File(Conf.outDir,  "nyt_pair.tuples.test.top.txt")) )
    //spdb.debugModel(new PrintStream(new File(Conf.outDir,  "nyt_pair.model.txt")))
    spdb.debugTuples(testTuples, new PrintStream(new File(Conf.outDir,  "nyt_pair.tuples.test.pred.txt")))

//    val goldFile = new File(Conf.conf.getString("eval.gold"))
//    val relPatterns = Conf.conf.getStringList("eval.targets").toSeq.map(_.r)
//
//    EvaluationTool.evaluate(Seq(new File(Conf.outDir, "nyt_pair.rank.txt")), goldFile, new PrintStream(new File(Conf.outDir,"eval.txt")), relPatterns, Seq("(FE)NAACL"))

    FilterRankFile.filter(new File(Conf.outDir, "nyt_pair.sub.rank.txt").getAbsolutePath,
      Conf.conf.getString("eval.subsample"), new File(Conf.outDir, "nyt_pair.rank.txt").getAbsolutePath)

    //print per-relation cells
  //  spdb.debugTuples(perRelationTuples, new PrintStream(new File(Conf.outDir, "tuples.perRel.txt")))

    //print ranked list of tuple predictions
//    if (Conf.conf.getBoolean("pca.print-ranks"))
//      spdb.printOutRankedList(spdb.tuples.filter(_.apply(1) != "X"), rel => rel.startsWith("REL$") && !rel.startsWith("REL$NA"),
//        new PrintStream(new File(Conf.outDir, "freebase.ranked.txt")),
//        printAnalysis = Conf.conf.getBoolean("pca.print-rank-analysis"))


  }


  def setupPrototypes(spdb: SelfPredictingDatabase) {
    val target = Conf.conf.getDouble("pca.proto-target")
    val proto = Seq("Proto1", "Proto2")

    spdb.tupleLambdas(proto) = 0.01
    spdb.entityLambdas(proto(0)) = 0.01
    spdb.entityLambdas(proto(1)) = 0.01


    //all tuples inherit from the proto tuple.
    if (Conf.conf.getBoolean("pca.proto-parents")) {
      for (tuple <- spdb.tuples) {
        spdb.tupleParents(tuple) = proto
        spdb.entityParents(tuple(0)) = proto(0)
        spdb.entityParents(tuple(1)) = proto(1)
      }
    }

    for (relation <- allRelationsToPredict /* ++ labels */ ) {
      spdb.addCell(relation, proto, 0.5, hide = true, feature = false, predict = true)
    }
    val ge = new spdb.NegKLofTargetAndCellAverage(spdb.getCells(proto), Array(1.0 - target, target), Conf.conf.getDouble("pca.proto-scale"))
    spdb.geTerms += ge
  }

  def createPerRelationTuples(spdb: SelfPredictingDatabase): Seq[Seq[String]] = {
    for (relation <- allRelationsToPredict.toSeq ++ labels.toSeq) yield {
      val tuple = Seq(relation, relation)
      spdb.addCell("bias", tuple, 1.0, hide = false, feature = true, predict = false)
      spdb.addCell(relation, tuple, 1.0, hide = false, feature = false, predict = true)
      for (other <- allRelationsToPredict ++ labels; if (other != relation)) {
        spdb.addCell(other, tuple, 1.0, hide = true, feature = false, predict = true)
      }
      if (Conf.conf.getBoolean("pca.create-per-rel-ge")) {
        val target = 0.05
        val ge = new spdb.NegKLofTargetAndCellAverage(spdb.getCells(tuple), Array(1.0 - target, target), 1.0, relation)
        spdb.geTerms += ge
      }
      tuple
    }
  }

  def analyseSecondOrderRules(spdb: SelfPredictingDatabase, relations:collection.Set[String], out:PrintStream) {
    Logger.info("Analysing Second Order rules...")
    //go over all tuples
    for (tuple <- spdb.tuples) {
      val cells = spdb.getCells(tuple).filter(_.target > 0.5)
      if (cells.view.map(_.relation).exists(relations)) {
        //for each tuple (A,B), find all
        //   C such that there is an observation for (A,C) and (C,B)
        val (a,b) = tuple(0) -> tuple(1)
        val cellsA = spdb.getCellsForEntity(a).filter(_.tuple(0) == a)
        val cellsB = spdb.getCellsForEntity(b).filter(_.tuple(1) == b)
        val arg2ToCellsA = cellsA.groupBy(_.tuple(1))
        val arg1ToCellsB = cellsB.groupBy(_.tuple(0))
        out.println("====")
        out.println("tuple (A,B): " + tuple.mkString(" | "))
        out.println("Relations:   " + cells.view.map(_.relation).mkString(","))
        for ((c,cellsAC) <- arg2ToCellsA) {
          val cellsCB = arg1ToCellsB.getOrElse(c,Seq.empty)
          if (!cellsCB.isEmpty){
            out.println("Shared Entity C: " + c)
            out.println("Relations (A,C): \n" + cellsAC.view.filter(_.target > 0.5).map(_.relation).mkString("  ","\n  ", ""))
            out.println("Relations (C,B): \n" + cellsCB.view.filter(_.target > 0.5).map(_.relation).mkString("  ","\n  ", ""))
          }
        }


      }
    }

  }

}
