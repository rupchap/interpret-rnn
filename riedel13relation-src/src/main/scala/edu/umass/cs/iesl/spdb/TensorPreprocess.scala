
package edu.umass.cs.iesl.spdb

import io.Source
import collection.mutable.{ArrayBuffer, HashSet, HashMap}
import util.Random
import java.io.{File, PrintStream}
import collection.mutable

/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 10/10/12
 * Time: 10:47 AM
 * To change this template use File | Settings | File Templates.
 */

object TensorPreprocess extends HasLogger{
  def main(args: Array[String]) {
    val dir = if(args.length > 0) args(0) else "/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/tensor"
    val freebaseFile = dir + "/../nyt-freebase-triples.match.txt"
    val output = dir + "/nyt-freebase-2hop.txt"
    val allfreebaseFile = "/iesl/canvas/lmyao/ie_ner/relation/freebase/nyt-freebase-triples.txt"
    extractTensorTriples(freebaseFile,output,allfreebaseFile)
  }

  //find transitions from matched triples, check new triple from all freebase triples
  def extractTensorTriples(triplefile : String,  output : String, file:String)  {
    val triples = Util.loadTripleHash(file)
    val res = new HashMap[String, String]    // arg1 -> rel arg1 arg2
    var source = Source.fromFile(triplefile)
    for (line <- source.getLines) {
      val fields = line.split("\t")
      res += fields(1) -> line
    }
    source.close()
    source = Source.fromFile(triplefile)
    val os = new PrintStream(output)
    for (line <- source.getLines()){
      val fields = line.split("\t")
      if (res.contains(fields(2))){
        val triple2 = res(fields(2)).split("\t")
        val arg1new = fields(1)
        val arg2new = triple2(2)
        if(triples.contains(arg1new+"\t"+arg2new)) os.print("POS\t" + triples(arg1new+"\t"+arg2new)+"\t"+arg1new+"\t"+arg2new + "\t")
        else if (triples.contains(arg2new+"\t"+arg1new))    os.print("POS\t" + triples(arg2new+"\t"+arg1new)+"\t"+arg2new+"\t"+arg1new + "\t")
        os.println(line + "\t" + res(fields(2)))
      }
    }
    os.close()
    source.close()
  }
}

//using SPDB, canonical decomposition for (e,e,r)
object TensorDSRun extends HasLogger {
  val allRelationsToPredict = new HashSet[String]
  // for generating negative predicates, we do not enlarge predicate space when loading test data
  val labels = new HashSet[String]
  val testTuples = new HashSet[Seq[Any]]
  var trainTuples : HashSet[Seq[Any]] = null

  //unary relations
  val unaryRelations = new HashSet[String]

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

  //posfile,   arg arg pattern, labelPrefix indicates the distant supervision relation labels, could be train, could be test
  def addTrainData(spdb: SPDB, file: String, labelPrefix: String = "REL$", minCellsPerRelation: Int = 2, minCellsPerTuple: Int = 2, percentage: Double = 0.1) {
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
        if (!line.startsWith("UNLABELED")) taggedTuples += tuple
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
    val frequentRelCells = cells.filter(c =>
    //c._2.startsWith("sen#") ||
    //c._2.startsWith("REL$") ||
      relation2cells.getOrElse(c._2, Seq.empty).size >= minCellsPerRelation)

    //this only consider cells to be predicted, excluding 'NA' cell, since we don't use 'NA'
    val tuple2PredictedCells = frequentRelCells.filter(pair => pair._2 != "REL$NA" && shouldBePredicted(pair._2)).groupBy(_._1)

    //this does row filtering , too many cells for one tuple mean this tuple is not related, such as Bush,Iraq
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
    logger.info("Number of ents " + spdb.allEnts.size)
    logger.info("Number of predicates " + spdb.relations.size)
    logger.info("Number of labels:" + labels.size)
    logger.info("Loaded train tuples:" + trainTuples.size)


    var os = new PrintStream(new File(Conf.outDir, "labels.txt"))
    os.println(labels.mkString("\n"))
    os.close()

    os = new PrintStream(new File(Conf.outDir, "patterns.txt")) //all predicates/patterns,exluding labels
    os.println(spdb.relations.filter(x => x.startsWith("path#")).mkString("\n")) //if we have features and predicates, we should include both

    os.close()

    os = new PrintStream(new File(Conf.outDir, "trainTuples.txt")) //train pos and neg data
    os.println(trainTuples.map(_.mkString("|")).mkString("\n"))
    os.close()
  }

  //for test, we do not need unlabeled data
  def addTestData(spdb: SPDB, file: String) {
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

      //add other ds labels only for test data
      //number of observed cells in one row should be larger than 0 to make sure it has features, todo
      //todo: we should also add test cells for which we don't have features
      var count = 0
      if (spdb.getCells(tuple).filterNot(_.hide).size > 0) {   //modified by limin, only keep instances that have more than 1 observed features
        testTuples += tuple
        count = testTuples.size
        //adding other labels for test, require labels is filled in with train data
        for (label <- labels)
          if (spdb.getCell(label, tuple).isEmpty) spdb.addCell(label, tuple, 0.0, hide = true)
        if (count % 10000 == 0) logger.info("Loaded test tuples:" + count)
      }

    }
    source.close()
    logger.info("Number of predicates after adding test data " + spdb.relations.size)
    logger.info("#test tuples:" + testTuples.size)
    logger.info("#entities:" + spdb.allEnts.size)

    val os = new PrintStream(new File(Conf.outDir, "testTuples.txt"))
    os.println(testTuples.map(_.mkString("|")).mkString("\n"))
    os.close()
  }

  def loadData(spdb:SPDB, file : String, hidden : Boolean = false, unary : Boolean = false){
    val source = Source.fromFile(file)
    for (line <- source.getLines()) {
      val fields = line.split("\t")
      val tuple = if(unary) fields.take(1)  else fields.take(2)
      val relations = if (unary) fields.drop(1)  else fields.drop(2)
      for (relation <- relations){
        if (spdb.getCell(relation,tuple).isEmpty) {
          spdb.addCell(relation,tuple, 1.0, hide = hidden)
        }
      }
      if (hidden) testTuples += tuple
      else {
        if (trainTuples == null) trainTuples = new HashSet[Seq[Any]]
        trainTuples += tuple
      }
    }
    logger.info("Added %d cells".format(spdb.cells.size))
    logger.info("Number of predicates " + spdb.relations.size)

    if (testTuples.size > 0) {
      val os = new PrintStream(new File(Conf.outDir, "testTuples.txt"))
      os.println(testTuples.map(_.mkString("|")).mkString("\n"))
      os.close()
    }
      
  }

  def loadDataWithFreebaseHidden(spdb:SPDB, file : String, hidden : Boolean = false, unary : Boolean = false) {
    val overlapping = new HashSet[Seq[Any]]
    val source = Source.fromFile(file)
    for (line <- source.getLines()) {
      val fields = line.split("\t")
      val tuple = if(unary) fields.take(1)  else fields.take(2)
      val relations = if (unary) fields.drop(1)  else fields.drop(2)
      var hiddenflag = false
      for (relation <- relations){
        if (spdb.getCell(relation,tuple).isEmpty) {
          if (hidden && relation.startsWith ("REL$"))  {
            spdb.addCell(relation,tuple, 1.0, hide = hidden)
            hiddenflag = true
          }
          else  spdb.addCell(relation,tuple, 1.0)
        }else if (relation.startsWith ("REL$")) overlapping += tuple
      }
      if (hiddenflag) testTuples += tuple
    }
    logger.info("Added %d cells".format(spdb.cells.size))
    logger.info("Number of predicates " + spdb.relations.size)

    if (testTuples.size > 0) {
      val os = new PrintStream(new File(Conf.outDir, "testTuples.txt"))
      os.println(testTuples.map(_.mkString("|")).mkString("\n"))
      os.close()
    }

    if (overlapping.size > 0) {
      val os = new PrintStream(new File(Conf.outDir, "overlapTuples.txt"))
      os.println(overlapping.map(_.mkString("|")).mkString("\n"))
      os.close()
    }
  }
  
  //add entity-attr unary relational data
  def addUnaryData(spdb: SPDB, file: String, overlap : Boolean = false) {
    logger.info("Loading unary data from file " + file)
    val source = Source.fromFile(file, "ISO-8859-1")
    var count = 0
    val ents = spdb.allEnts.toSet
    val os = new PrintStream(new File(Conf.outDir, "entities.overlap.txt"))
    for (line <- source.getLines()) {
      count += 1
      val fields = line.split("\t")
      val entity = fields(0)
      for (rel <- fields.drop(1)){
        if (overlap){
          if (ents.contains(entity)){
            if (spdb.getCell(rel, Seq(entity)).isEmpty) {
              spdb.addCell(rel,Seq(entity))
              unaryRelations += rel
              os.println(entity)
            }
          }
        } else
        if (spdb.getCell(rel, Seq(entity)).isEmpty) {
          spdb.addCell(rel,Seq(entity))
          unaryRelations += rel
        }
      }

      if (count % 10000 == 0) logger.info("Loaded entities:" + count)
    }
    os.close()
    logger.info("#entities after adding unary data:" + spdb.allEnts.toSet.size)
    logger.info("#unary relations:" + unaryRelations.size)
  }

  def addNegativeUnaryRelations(spdb: SPDB) {
    logger.info("Get all entities!")
    val tuples = spdb.entities
    val negFeatureRatio = Conf.conf.getInt("pca.neg-features")
    val negFeatureDynamic = Conf.conf.getBoolean("pca.neg-dynamic")

    val duplicateNegPatternRows = Conf.conf.getBoolean("pca.dup-neg-pattern")
    def addNegCell(label: String, tuple: Seq[Any]) {
      spdb.addCell(label, tuple, 0.0, hide = false)
    }


    logger.info("Adding negative unary relations!")
    val startTime = System.currentTimeMillis()
    val allUnaryRelationsSeq = unaryRelations.toIndexedSeq
    //for each tuple, sample 10 times negative surface relations
    var count = 0
    for (tuple <- tuples) {
      //  logger.info("Add negative surface patterns for " + tuple.mkString("\t") )
      val tupleRelations = new HashSet[String]
      tupleRelations ++= spdb.getCells(Seq(tuple)).map(_.relation)

      //when the original patterns co-occurring with the tuple are too many, we add few negative features
      val total = if (negFeatureDynamic && tupleRelations.size < 10) tupleRelations.size * negFeatureRatio else negFeatureRatio
      for (iter <- 0 until total) {
        var sampledIndex = (math.random * allUnaryRelationsSeq.size).toInt
        var sampledRelation = allUnaryRelationsSeq(sampledIndex)
        while (tupleRelations.contains(sampledRelation)) {
          sampledIndex += 1
          sampledIndex = sampledIndex % allUnaryRelationsSeq.size
          sampledRelation = allUnaryRelationsSeq(sampledIndex)
        }
        if (spdb.getCell(sampledRelation, Seq(tuple)).isEmpty)
          addNegCell(sampledRelation, Seq(tuple))
      }
      count += 1
      if (count % 10000 == 0) logger.info("Processed tuples:" + count)
    }
    logger.info("Finish adding negative unary relations within " + (System.currentTimeMillis() - startTime) / 1000 + " seconds!")
  }


  def isDuplicate(tuple: Seq[Any]) = {
    tuple.head.toString.startsWith("dup#")
  }

  def addCellWithRowDuplication(spdb: SPDB,
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
  def addNegativeDSLabelsForTraining(spdb: SPDB) {
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
  def addNegativeRelations(spdb: SPDB) {
    logger.info("Get all tuples!")
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
    val configFile = if (args.length > 0) args(0) else "relation.conf"
    Conf.add(configFile)

    runPCA
  }


  /**
   * Writing out some debug output into the run-specific output directory.      todo:save model and load model, inference, total new rows unobserved during training
   * @param spdb the matrix to debug.
   */
  def debugOutput(spdb: SPDB, prefix: String, unary : Boolean = false, tensor : Boolean = true, pair : Boolean = true, bigram : Boolean = true) {

    spdb.debugModel(new PrintStream(new File(Conf.outDir, prefix + ".model.txt")), true,  true)
    spdb.debugTuples(testTuples, new PrintStream(new File(Conf.outDir, prefix + ".tuples.test.pred.txt")), tensor = tensor, pair = pair, bigram = bigram)
    spdb.debugTuples(trainTuples, new PrintStream(new File(Conf.outDir, prefix + ".tuples.train.pred.txt")), tensor = tensor, pair = pair, bigram = bigram)
    spdb.debugTuples(spdb.tuples.filterNot(t => trainTuples(t) || testTuples(t)),
      new PrintStream(new File(Conf.outDir, prefix + ".tuples.remainder.pred.txt")))
    
    //unary tuples
    if (unary)
      spdb.debugTuples(spdb.entities.map(ent=>Seq(ent)), new PrintStream(new File(Conf.outDir, prefix + ".entities.pred.txt")) )


//    //debug map predictions
//    spdb.debugMAP(testTuples, new PrintStream(new File(Conf.outDir, prefix + ".tuples.test.top.txt")))
//    spdb.debugMAP(trainTuples, new PrintStream(new File(Conf.outDir, prefix + ".tuples.train.top.txt")))
//    if (unary)
//      spdb.debugMAP(spdb.entities.map(ent=>Seq(ent)), new PrintStream(new File(Conf.outDir, prefix + ".entities.top.txt")) )
  }

  /**
   * Writes out results in readable format, src dest pat score.    also write out the original data as ground truth, evaluation can also be here
   * @param spdb the matrix.
   * @param threshold when to predict a label is true.         todo:handle multiple labels, copy from PcaDSRun
   */
  def writeOutResults(spdb: SPDB, prefix: String, threshold: Double = 0.5, tensor : Boolean = true, pair : Boolean = true, bigram : Boolean = true,  multilabel : Boolean = true, extraRelations: Seq[String] = Seq.empty) {
    logger.info("Writing out distant supervision predictions.")
    //val result = new PrintStream(new File(Conf.outDir, prefix + ".curve.txt")) //the best label, second best if best is NA
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
      val predictedCells = new ArrayBuffer[SPDB#Cell]
      val tupleString = tuple.mkString("\t")
     // for (cell <- spdb.getCells(tuple); if (labels.contains(cell.relation))) {
      for (cell <- spdb.getCells(tuple); if (cell.relation.startsWith("REL$"))) {
          if (cell.target > threshold) {
          goldLabels += cell.relation
          goldLabel = cell.relation
        }
        cell.update(tensor,pair, bigram) //todo:remember to correct this back
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
      if (goldLabels.size > 0) {
        gold += 1
      }

      //extra relations
      for (rel <- extraRelations) {
        spdb.getCell(rel, tuple) match {
          case None =>
            val prob = spdb.calculateScore(tuple, rel, tensor, pair, bigram)
            val gold = "REL$NA"
            preds += ((prob, tupleString, gold, rel))
          case Some(cell) if (cell.testData) =>
            val prob = cell.update(tensor, pair)
            val gold = if (cell.target > 0.5) cell.relation else "REL$NA"
            preds += ((prob, tupleString, gold, rel))
          case Some(cell) =>
        }
      }

      if (maxScore < naScore) maxScore -= 1.0
      
      //select one gold label from goldLabels
      if(goldLabels.contains(bestLabel)) goldLabel = bestLabel
      if (!multilabel) preds += ((maxScore, tupleString, goldLabel, bestLabel))
    }
    val sorted = preds.sortBy(-_._1)

    var id = 0
    for ((score, tupleString, goldLabel, bestLabel) <- sorted) {

      if (goldLabel.compareTo(bestLabel) == 0) {
        cor += 1
      }
      total += 1
      id += 1
    //  result.println(id + "\t" + cor * 1.0 / gold + "\t" + cor * 1.0 / total)
      os.println(score + "\t" + tupleString + "\t" + goldLabel + "\t" + bestLabel)
    }

    os.close()
    //result.close()
  }


  /**
   * Prepares the matrix, runs the actual PCA algorithm, does some debug and evaluation output, and then writes out
   * the result in TAC format.
   */
  def runPCA {
    logger.info("Output dir: " + Conf.outDir.getAbsolutePath)
    logger.info("Preparing PCA")
    val spdb = new SPDB

    spdb.numComponents = Conf.conf.getInt("pca.rel-components")
    spdb.lambdaRel = Conf.conf.getDouble("pca.lambda-rel")
    spdb.lambdaTuple = Conf.conf.getDouble("pca.lambda-tuple") 
    spdb.lambdaEntity = Conf.conf.getDouble("pca.lambda-ent")
    spdb.lambdaRelPair = Conf.conf.getDouble("pca.lambda-feat")
    spdb.lambdaBias = Conf.conf.getDouble("pca.lambda-bias")
    spdb.bias = Conf.conf.getBoolean("pca.bias")
    spdb.useGlobalBias = Conf.conf.getBoolean("pca.use-global-bias")
    spdb.alphaNorm = Conf.conf.getBoolean("pca.alpha")
    
    spdb.maxIterations = Conf.conf.getInt("pca.max-msteps")

    spdb.tolerance = Conf.conf.getDouble("pca.tolerance")
    spdb.gradientTolerance = Conf.conf.getDouble("pca.gradient-tolerance")
    spdb.maxCores = Conf.conf.getInt("pca.max-cores")


    val binaryRelations = Conf.conf.getBoolean("source-data.binary")

    if (binaryRelations) {
    addTrainData(spdb, Conf.conf.getString("source-data.train"), "REL$",
      Conf.conf.getInt("pca.rel-cutoff"),  Conf.conf.getInt("pca.tuple-cutoff"),
      Conf.conf.getDouble("source-data.percentage")
    )


    //if (!Conf.conf.getBoolean("pca.no-neg-freebase"))   addNegativeDSLabelsForTraining(spdb)

    addTestData(spdb, Conf.conf.getString("source-data.test"))

    //addNegativeRelations(spdb)
    }

    //unary relations todo:only add entities that participate in relations
    val unary = Conf.conf.getBoolean("source-data.unary")
    if(unary) {
      addUnaryData(spdb, Conf.conf.getString("source-data.unary-data"),true) //check overlapping entities
    }

    if (Conf.conf.getBoolean("heldout.eval")){
      loadData(spdb,Conf.conf.getString("heldout.train"),false,true)   //unary=true
      //loadData(spdb,Conf.conf.getString("heldout.test"),true,true)    //hidden = unary = true , test data has no Freebase labels
      loadDataWithFreebaseHidden(spdb,Conf.conf.getString("heldout.test"),true,true)  //test data has Freebase labels
    }

    
    //run PCA
    val tensor = Conf.conf.getBoolean("pca.tensor")     //tensor=false for now
    val pair = Conf.conf.getBoolean("pca.pair")
    val bigram = Conf.conf.getBoolean("pca.bigram")
    Conf.conf.getString("pca.mode") match {
      case "sgd" => spdb.runSGD()
      case "bpr" => spdb.runBPR(tensor,pair)
      case "sgd-dyn" => spdb.runDynamicSGD(tensor,pair, bigram)
      case _ => spdb.runSGD()
    }


    if (Conf.conf.getBoolean("heldout.eval")){
      spdb.heldoutDS(testTuples, new PrintStream(new File(Conf.outDir,  "cand.unary.test.rank.txt")), tensor = tensor, pair = pair, bigram = bigram, filter = true )

      //spdb.heldoutRank(testTuples, new PrintStream(new File(Conf.outDir,  "svo.tuples.test.top.txt")), tensor = tensor, pair = pair, bigram = bigram )
      //spdb.heldoutRank(testTuples, new PrintStream(new File(Conf.outDir,  "svo.tuples.test.top.txt")), tensor = tensor, pair = pair, bigram = bigram, filter = true )

      spdb.debugModel(new PrintStream(new File(Conf.outDir,  "svo.model.txt")), false, true) //heldout for unary
      spdb.debugTuples(trainTuples, new PrintStream(new File(Conf.outDir,  "svo.tuples.train.pred.txt")), tensor = tensor, pair = pair, bigram = bigram)
      spdb.debugMAP(testTuples, new PrintStream(new File(Conf.outDir,  "cand.unary.test.tuples.top.txt")), tensor = tensor, pair = pair, bigram = bigram)
    }
    else{
    //write out results
    val extraRelations = Conf.conf.getStringList("eval.extra-relations").toSeq
    writeOutResults(spdb, "nyt_pair", 0.0, tensor, pair, bigram, extraRelations = extraRelations)


    //this is for producing curve w.r.t freebase annotations
//    writeOutResults(spdb, "nyt_pair_old", 0.0, tensor, pair, multilabel = false)
//    spdb.heldoutRank(testTuples, new PrintStream(new File(Conf.outDir,  "nyt_pair.tuples.test.top.txt")), tensor = tensor, pair = pair, filter = true)

    //print out debug statistics
    debugOutput(spdb, "nyt_pair", true, tensor = tensor, pair = pair, bigram = bigram)
      
      //print top ranked predictions for some particular tuples
      val tuples = new HashSet[Seq[Any]]()
      for (line <- Source.fromFile("/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/riedel_annotation/nyt-freebase.dev.sample.pairs.txt").getLines()) {
        val fields = line.split("\\t")
        tuples += fields.toSeq
      }
      spdb.debugMAP(tuples,new PrintStream(new File(Conf.outDir,  "nyt_pair.tuples.sample.top.txt")), tensor = tensor, pair = pair, bigram = bigram)

    }
  }



}
