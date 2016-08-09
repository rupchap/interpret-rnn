package edu.umass.cs.iesl.spdb

import cc.factorie.db.mongo.MongoCubbieImplicits._
import java.io.{File, PrintStream}
import collection.mutable.{HashSet, HashMap, ArrayBuffer}
import io.Source
import collection.mutable

/**
 * entry for loading row style training and test data, each row has a pair and a predicate
 *
 * @author lmyao
 */
object PCARun extends HasLogger {

  //posfile,   CUI CUI pattern, todo: CUI CUI pattern hidden, for heldout training, with TWREX relations
  def addPositiveRelations(spdb : SelfPredictingDatabase,  posfile : String ) : Seq[String] =  {
    logger.info("Loading data from file " + posfile)
    val relations = new HashSet[String]
    val source = Source.fromFile(posfile)
    for (line <- source.getLines) {
      val fields = line.split("\t")
      val tuple = fields.take(2)
      for(relation <- fields.drop(2) ){
        spdb.addCell(relation, tuple, 1.0, false)    //todo:confirm predict=true means pseudolikelihood
        relations += relation
      }
    }
    source.close
    logger.info("Number of pairs " + spdb.tuples.size)
    logger.info("Number of predicates " + spdb.relations.size)
    relations.toSeq
  }

  //add negative surface pattern relations, implemented by Limin Yao
  //collect statistics of all surface pattern relations,
  def addNegativeRelations(spdb : SelfPredictingDatabase, relations : Seq[String] ) {

    val tuples = spdb.tuples
    
    logger.info("Adding negative surface patterns!")
    val startTime = System.currentTimeMillis()
    //for each tuple, sample 10 times negative surface relations
    var count = 0
    for(tuple <- tuples){
    //  logger.info("Add negative surface patterns for " + tuple.mkString("\t") )
      val tupleRelations = new HashSet[String]
      tupleRelations ++= spdb.getCells(tuple).map(_.relation)

      val total = tupleRelations.size * Conf.conf.getInt("pca.neg-features")
      for(iter <- 0 until  total){
        var sampledIndex = (math.random * relations.size ).toInt
        var sampledRelation = relations(sampledIndex)
        while(tupleRelations.contains(sampledRelation)){
          sampledIndex += 1
          sampledIndex = sampledIndex % relations.size
          sampledRelation = relations(sampledIndex)
        }
        if (spdb.getCell(sampledRelation, tuple).isEmpty)
          spdb.addCell(sampledRelation, tuple, 0.0, false)   // the last two boolean values are default, feature = false, toPredict=true
      }
      count += 1
      if(count % 10000 == 0) logger.info("Processed tuples:" + count)
    }
    logger.info("Finish adding negative features within " + (System.currentTimeMillis() - startTime)/1000 + " seconds!")
  }

  def main(args: Array[String]) {
    val configFile =  "watsonmd.conf"
    Conf.add(configFile)
    runPCA()
  }


  /**
   * Writing out some debug output into the run-specific output directory.      todo:save model and load model, inference, total new rows unobserved during training
   * @param spdb the matrix to debug.
   */
  def debugOutput(spdb: SelfPredictingDatabase, prefix : String) {

    spdb.debugModel(new PrintStream(new File(Conf.outDir, prefix +".model.debug.txt")))
    spdb.debugTuples(spdb.tuples.take(1000), new PrintStream(new File(Conf.outDir, prefix + "tuples.pred.txt")) )
  }

  /**
   * Writes out results in readable format, src dest pat score.    also write out the original data as ground truth, evaluation can also be here
   * @param spdb the matrix.
   * @param threshold when to predict a slot as filled.
   */
  def writeOutResults(spdb: SelfPredictingDatabase, prefix : String, rowStart : Int, rowEnd : Int, threshold: Double = 0.5) {
    logger.info("Writing out pair predictions.")
    val result = new PrintStream(new File(Conf.outDir, prefix + ".reconstruction." + rowStart + "_" + rowEnd + ".txt"))
    val os = new PrintStream(new File(Conf.outDir, prefix + ".source." + rowStart + "_" + rowEnd + ".txt"))    //ground trueth, original input for some rows, not all the rows
    val eos = new PrintStream(new File(Conf.outDir, prefix + ".eval." + rowStart + "_" + rowEnd + ".txt"))
    var cor = 0
    var gold = 0
    var pred = 0
    val tuples = spdb.tuples.toSeq

    //get predicted query cells
    for (rowId <- rowStart until rowEnd) { //todo:fromIndex toIndex
      val tuple = tuples(rowId)
      val predictedCells = new ArrayBuffer[SelfPredictingDatabase#Cell]
      result.println("Pair\t" + tuple.mkString("\t"))
      os.println("Pair\t" + tuple.mkString("\t"))

      for (cell <- spdb.getCells(tuple); if(!cell.feature)) {
        os.println(tuple.mkString("\t") + cell.relation + "\t" + {if(cell.target > threshold)  "true" else "false"})    //positive and negative examples
        if(cell.target > threshold) gold += 1 // target is 1.0, positive example
        cell.update()
        if (cell.predicted > threshold) {
          predictedCells += cell
          pred += 1

          if(cell.target >= threshold) cor += 1 //both prediction and gold are true
        }
      }
      //sort the cells, and output them one by one
      result.println(predictedCells.toList.sortWith((x,y) => x.predicted > y.predicted).take(100).map({case x=> tuple.mkString("\t") + "\t" + x.relation + "\t" + x.predicted}).mkString("\n") )
    }

    eos.println("Cor\tPred\tGold\tPrec\tRec")
    eos.println(cor + "\t" + pred + "\t" + gold + "\t" + cor*1.0/pred + "\t" + cor*1.0/gold)

    result.close()
    os.close
    eos.close
  }

  /**
   * Prepares the matrix, runs the actual PCA algorithm, does some debug and evaluation output, and then writes out
   * the result in TAC format.
   */
  def runPCA() {
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

    val relations = addPositiveRelations(spdb, Conf.conf.getString("source-data.posdata"))
    addNegativeRelations(spdb, relations)

    //run PCA
    spdb.addBias()
    spdb.runLBFGS()

    //print out debug statistics
    debugOutput(spdb, "cui_pair")

    val os = new PrintStream(new File(Conf.outDir,"cui_pair.model.txt"))
    spdb.saveModel(os)
    os.close
    //write out results
    writeOutResults(spdb, "cui_pair", Conf.conf.getInt("source-data.rowStart"), Conf.conf.getInt("source-data.rowEnd"), 0.0)

  }
}
