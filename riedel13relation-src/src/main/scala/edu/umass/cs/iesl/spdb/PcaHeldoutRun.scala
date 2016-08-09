package edu.umass.cs.iesl.spdb

import java.io.{File, PrintStream}
import io.Source
import util.Random
import collection.mutable
import mutable.{HashMap, HashSet, ArrayBuffer}

/**
 * entry for loading row style training and test data, each row has a pair and a predicate
 * hidden means observed or not, when cell.target is 0.0, two cases, if hidden, means ? else means 0.0
 *
 * @author lmyao
 */
object PcaHeldoutRun extends HasLogger {
  val allRelationsToPredict = new HashSet[String]
  // for generating negative predicates, we do not enlarge predicate space when loading test data
  val labels = new HashSet[String]
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
  lazy val unused = Conf.conf.getStringList("pca.unused").toSet //Set("lc","rc","trigger","ner", "REL$")

  //posfile,   arg arg pattern, labelPrefix indicates the distant supervision relation labels, could be train, could be test
  def addTrainData(spdb: SelfPredictingDatabase, file: String, labelPrefix: String = "REL$", minCellsPerRelation: Int = 2, minCellsPerTuple: Int = 3, percentage: Double = 0.1) {
    logger.info("Loading training data from, hold out half features " + file)
    val cells = new ArrayBuffer[(Seq[Any], String)]

    val random = new Random(0)

    val source = Source.fromFile(file, "ISO-8859-1")
    for (line <- source.getLines()) {
      var fields: Array[String] = null

      if (line.startsWith("POSITIVE") || line.startsWith("NEGATIVE") || line.startsWith("UNLABELED")) {
        fields = line.split("\t").drop(1)
      } else
        fields = line.split("\t")
      val tuple = fields.take(2)

      for (relation <- fields.drop(2); if (!shouldBeSkipped(relation)))
        cells += ((tuple, relation))

    }
    source.close()

    logger.info("Loaded files with %d cells in total, now adding cells".format(cells.size))

    val relation2cells = cells.groupBy(_._2)
    val frequentRelCells = cells.filter(c =>
      //c._2.startsWith("sen#") ||
      //c._2.startsWith("REL$") ||  
      relation2cells.getOrElse(c._2, Seq.empty).size >= minCellsPerRelation)
    val tuple2cells = frequentRelCells.groupBy(_._1)

    //this does row filtering
    val frequentCells = frequentRelCells.filter(c => tuple2cells.getOrElse(c._1, Seq.empty).size >= minCellsPerTuple && tuple2cells.getOrElse(c._1, Seq.empty).size < 50)

    var numCells = 0

    for ((tuple, relation) <- frequentCells; if (!shouldBeSkipped(relation))) {
      val feature = shouldBeFeature(relation)
      val toPredict = shouldBePredicted(relation)

      var hide = false//random.nextBoolean()
      if (spdb.getCell(relation, tuple).isEmpty) {
        if (relation.startsWith("REL$"))  {
          spdb.addCell(relation, tuple, 1.0, false, feature = feature, predict = toPredict)
          labels += relation
        }
        else spdb.addCell(relation, tuple, 1.0, hide, feature = feature, predict = toPredict)
        if (!hide && toPredict) allRelationsToPredict += relation
        numCells += 1
        if (numCells % 100000 == 0) logger.info("Added %d cells".format(numCells))

      }
      Unit
    }

    logger.info("Added %d cells".format(numCells))
    logger.info("Number of pairs " + spdb.tuples.size)
    logger.info("Number of predicates " + spdb.relations.size)
    logger.info("Number of labels:" + labels.size)

    var os = new PrintStream(new File(Conf.outDir, "labels.txt"))
    os.println(labels.mkString("\n"))
    os.close()

    os = new PrintStream(new File(Conf.outDir, "patterns.txt")) //all predicates/patterns,exluding labels
    os.println(spdb.relations.filter(x => !x.startsWith("REL")).mkString("\n")) //if we have features and predicates, we should include both

    os.close()
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
    val tuples = spdb.tuples.filterNot(isDuplicate)
    for (tuple <- tuples; if (allowAllNegTuples || spdb.getCells(tuple).filter(_.toPredict).size > 0)) {
      for (label <- labels) {
        if (spdb.getCell(label, tuple).isEmpty) addNegCell(label, tuple)
      }
      count += 1
      if (count % 10000 == 0) logger.info("Processed tuples:" + count)
    }

  }

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
    val configFile = if (args.length > 0) args(0) else "relation_pat_heldout.conf"
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
    spdb.debugTuples(spdb.tuples, new PrintStream(new File(Conf.outDir, prefix + ".tuples.pred.txt")))
  }

  /**
   * Writes out results in readable format, src dest pat score.    also write out the original data as ground truth, evaluation can also be here
   * @param spdb the matrix.
   * @param threshold when to predict a label is true.
   */
  def writeOutResults(spdb: SelfPredictingDatabase, prefix: String, threshold: Double = 0.5, entityPrinter: Any => String = id => id.toString) {
    logger.info("Writing out predicting information")
    val out = new PrintStream(new File(Conf.outDir, prefix + ".tuples.pat.pred.txt"))
    for (tuple <- spdb.tuples) {
      out.println("****")
      out.println("TUPLE:"+ tuple.map(entityPrinter(_)).mkString("|"))

      //_1 tuple _2 relation, _3 target _4 hid   value:score
      var cells = new HashMap[  (Seq[Any], String, Double, Boolean),Double ]   //tuple, relation, target, hidden or not, predict

   //   add all relations to be predicted as candidate cells, including data cells
      for (relation <- allRelationsToPredict){
        val cell = spdb.getCell(relation,tuple)
        if (cell.isEmpty) cells += ( tuple, relation, 0.0, true) -> 0.5  //new cells, hidden = true
        else if (cell.get.toPredict) cells += (tuple, relation, cell.get.target, cell.get.hide) -> 0.5  //data cells
      }
      
//      for (relation <- allRelationsToPredict) {
//        if (spdb.getCell(relation,tuple).isEmpty) spdb.addCell(relation,tuple,0.0, true,false,true)
//      }

//      val allCells = spdb.getCells(tuple)
//      val cells = allCells.filter(_.toPredict)
//      //cells.foreach(_.update())

      for (cell <- cells.keySet) {
        cells(cell) =  spdb.calculateProb ( spdb.calculateScoreRaw(cell._1,cell._2)  )
      }

      //H/O predScore target relation tuple
      val sortedObs = cells.keySet.toArray.filter(!_._4).sortBy(c => - cells(c))
      out.println("Observed:")
      for (cell <- sortedObs) {
        out.println( " %s %6.4f %6.4f %s ".format(if (cell._4) "H" else "O", cells(cell), cell._3, cell._2) + cell._1.mkString("|") )
      }
      out.println("Hidden:")

      val sortedHidden = cells.keySet.toArray.filter(_._4).filter(c => cells(c) > threshold).sortBy(c => -cells(c))
      for (cell <- sortedHidden) {
        out.println( " %s %6.4f %6.4f %s ".format(if (cell._4) "H" else "O", cells(cell), cell._3 , cell._2) + cell._1.mkString("|"))
      }

    }
    out.close
  }

  /**
   * Prepares the matrix, runs the actual PCA algorithm, does some debug and evaluation output, and then writes out
   * the result in TAC format.
   */
  def runPCA(watson: Boolean) {
    logger.info("Preparing PCA")
    val spdb = new SelfPredictingDatabase()
    val normalizer = new mutable.HashMap[Any,Double]()
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

    addNegativeDSLabelsForTraining(spdb)

    addNegativeRelations(spdb)

    if (Conf.conf.getBoolean("pca.print-cooccur")) spdb.debugCooccurence(new PrintStream(new File(Conf.outDir, "cooccurence.txt")))

    if (Conf.conf.getBoolean("pca.relation-bias")) spdb.addBias()

    //writing out tuples before inference
    spdb.debugTuples(spdb.tuples, new PrintStream(new File(Conf.outDir,  "tuples.init.txt")))

    //run PCA
    Conf.conf.getString("pca.mode") match {
      case "sgd" => spdb.runSGD()
      case "bpr" => spdb.runBPR()
      case "bpr-inv" => spdb.runBPRInv()
      case "bpr-ent" => spdb.runBPRAll(entityCentric = true)
      case "bpr-all" => spdb.runBPRAll()
      case _ => spdb.runLBFGS()
    }

    //print out debug statistics
    debugOutput(spdb, "nyt_pair")

    //write out results
    writeOutResults(spdb, "nyt_pair", 0.5)
  }

}
